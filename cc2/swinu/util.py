import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from typing import Dict
from torch.fft import rfftfreq, fftfreq


def apply_hann_window(field: torch.Tensor, H: int, W: int):
    """
    Apply 2D Hann window normalised to unit RMS.
    Works for field shape [B, T, C, H, W].
    """
    device = field.device
    wh = torch.hann_window(H, device=device).unsqueeze(1)  # (H, 1)
    ww = torch.hann_window(W, device=device).unsqueeze(0)  # (1, W)
    win = (wh @ ww).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,H,W]
    win_rms = (win**2).mean().sqrt()
    return field * win / win_rms


def radial_bins_rfft(Hf: int, Wf: int, device: str, n_bins: int | None):
    # This function is used to create the radial bins for DSE calculation
    fy = fftfreq(Hf, d=1.0, device=device)
    fx = rfftfreq(2 * (Wf - 1), d=1.0, device=device)
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(FY**2 + FX**2)
    is_dc = r == 0.0
    r = r / r.max().clamp(min=1e-8)
    if n_bins is None:
        n_bins = max(8, min(Hf, 2 * (Wf - 1)) // 2)
    edges = torch.linspace(1e-8, 1.0000001, n_bins + 1, device=device)
    bin_index = torch.bucketize(r.reshape(-1), edges) - 1
    bin_index[is_dc.flatten()] = -1
    bin_index = bin_index.reshape(Hf, Wf)
    mask = bin_index >= 0
    counts = torch.bincount(bin_index[mask].flatten(), minlength=n_bins).clamp(min=1)
    return bin_index, mask, counts, n_bins


def scheduled_sampling_inputs(
    previous_state: torch.Tensor,
    current_state: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    t: int,
    step: int,
    max_step: int,
    steepness: float = 10.0,
    ss_pred_min: float = 0.0,
    ss_pred_max: float = 1.0,
) -> torch.Tensor:
    """
    Mixes ground-truth and model predictions for two-lag inputs at rollout step t.

    Bengio et al: Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks

    Args:
        previous_state: Tensor of shape [B,1,C,H,W], model's last prediction or ground-truth mix.
        current_state:  Tensor of shape [B,1,C,H,W], model's current prediction or ground-truth mix.
        x:             Original input sequence Tensor [B,T,C,H,W].
        y:             Ground-truth future targets Tensor [B,n_step,C,H,W].
        t:             Current rollout step (0-based).
        step:          Current step index.
        max_step:      Total number of steps for scheduling.
        steepness:     Controls sigmoid slope (higher = sharper transition).
        ss_pred_min:   Minimum value for ss_pred
        ss_pred_max:   Maximum values for ss_pred

    Returns:
        input_state:  Mixed input Tensor to feed into model, shape [B,2,C,H,W].
    """
    B = previous_state.size(0)
    device = previous_state.device

    # Compute probability of using model prediction (p_pred from 0 -> 1)
    raw_pred = torch.sigmoid(
        torch.tensor((step / max_step) * steepness - steepness / 2, device=device)
    )

    p_pred = ss_pred_min + (ss_pred_max - ss_pred_min) * raw_pred

    # Sample independent Bernoulli for each lag
    mask_prev = torch.bernoulli(p_pred * torch.ones([B, 1, 1, 1, 1], device=device))
    mask_curr = torch.bernoulli(p_pred * torch.ones([B, 1, 1, 1, 1], device=device))

    # Determine ground-truth frames for this step
    if t == 0:
        gt_prev = x[:, -2:-1, ...]  # second-to-last input
        gt_curr = x[:, -1:, ...]  # last input
    elif t == 1:
        gt_prev = x[:, -1:, ...]
        gt_curr = y[:, 0:1, ...]
    else:
        gt_prev = y[:, t - 2 : t - 1, ...]
        gt_curr = y[:, t - 1 : t, ...]

    # Mix: mask=1 -> use prediction; mask=0 -> use ground truth
    input_prev = mask_prev * previous_state + (1 - mask_prev) * gt_prev
    input_curr = mask_curr * current_state + (1 - mask_curr) * gt_curr

    return (
        torch.cat([input_prev, input_curr], dim=1),
        p_pred.item(),
        mask_prev.float().mean(),
        mask_curr.float().mean(),
    )


def rollout_weights(step: int, max_step: int, R: int, device: str, q: float = 0.3):
    s = min(1.0, step / max_step)

    a = torch.zeros(R)
    a[0] = 1.0 - q
    a[1] = q
    b = [1.0 / R] * R

    w = [(1.0 - s) * a_t + s * b_t for a_t, b_t in zip(a, b)]

    return torch.tensor(w).to(device)


def ste_clamp(x, use_ste=False):
    if use_ste:
        x_clamped = x.clamp(0.0, 1.0)
        return x + (x_clamped - x).detach()
    else:
        return x.clamp(0.0, 1.0)


def _build_input_state(
    previous_state,
    current_state,
    x,
    y,
    t,
    use_scheduled_sampling: bool,
    step: int | None,
    max_step: int | None,
    ss_pred_min: float,
    ss_pred_max: float,
):
    metrics = {}
    if use_scheduled_sampling:
        assert step is not None and max_step is not None, "SS requires step/max_step"
        input_state, p_pred, mask_prev, mask_curr = scheduled_sampling_inputs(
            previous_state,
            current_state,
            x,
            y,
            t,
            step,
            max_step,
            ss_pred_min=ss_pred_min,
            ss_pred_max=ss_pred_max,
        )
        metrics["ss_p_pred"] = p_pred
        metrics["ss_mask_prev"] = mask_prev
        metrics["ss_mask_curr"] = mask_curr
        return input_state, p_pred, metrics
    else:
        input_state = torch.cat([previous_state, current_state], dim=1)
        return input_state, None, metrics


def _maybe_calibrate_input(model, input_state, month_idx, do_calib: bool):
    if do_calib:
        assert month_idx is not None, "month_idx required when calibration is enabled"
        return model.logit_calibrator(input_state, month_idx)
    return input_state


def _forward_and_unpack(model, input_state, step_forcing, t):
    out = model(input_state, step_forcing, t)
    metrics = {}
    if isinstance(out, dict):
        tendency_core = out["core"]
        tendency_obs = out["obs"]
        metrics.update(out.get("diag", {}))
    else:
        tendency_core = tendency_obs = out
    return tendency_core, tendency_obs, metrics


def _compute_step_loss(loss_fn, x, y, t, next_pred_obs, tendency_obs, step):
    if loss_fn is None:
        return None

    y_true_full = y[:, t : t + 1, ...]
    if t == 0:
        y_true_delta = y_true_full - x[:, -1, ...].unsqueeze(1)
    else:
        y_true_delta = y_true_full - y[:, t - 1 : t, ...]

    return loss_fn(
        y_true_full=y_true_full,
        y_pred_full=next_pred_obs,
        y_true_delta=y_true_delta,
        y_pred_delta=tendency_obs,
        global_step=step,
    )


def _next_core_state(
    model,
    current_state,
    next_pred_core,
    y,
    t,
    training: bool,
    use_scheduled_sampling: bool,
    p_pred: float | None,
    do_calib: bool,
    month_idx: torch.Tensor | None,
):
    metrics = {}

    if not training:
        return ste_clamp(next_pred_core, False), metrics

    if use_scheduled_sampling:
        assert p_pred is not None
        mask_next = torch.bernoulli(
            p_pred * torch.ones_like(current_state[:, :1, :, :, :])
        )
        pred_core_clamped = ste_clamp(next_pred_core, True)

        next_gt = y[:, t : t + 1, ...]
        if do_calib:
            assert month_idx is not None
            next_gt = model.logit_calibrator(next_gt, month_idx)

        next_state_core = mask_next * pred_core_clamped + (1 - mask_next) * next_gt
        metrics["ss_mask_next"] = mask_next.float().mean()
        return next_state_core, metrics

    # training without SS feedback
    return ste_clamp(next_pred_core, True), metrics


def _materialize_obs(next_pred_obs, training: bool):
    return (
        ste_clamp(next_pred_obs, True) if training else ste_clamp(next_pred_obs, False)
    )


def _aggregate_losses(
    all_losses: list,
    use_rollout_weighting: bool,
    step: int,
    n_step: int,
    max_step: int,
    metrics: dict,
):
    if len(all_losses) == 0:
        return None, metrics

    # aggregate step losses
    loss = {"loss": []}
    for l in all_losses:
        for k, v in l.items():
            if k == "loss":
                loss["loss"].append(v)
            else:
                try:
                    loss[k].append(v)
                except KeyError:
                    loss[k] = [v]

    if use_rollout_weighting and n_step > 1 and step is not None:
        w = rollout_weights(step, max_step, n_step, loss["loss"][0].device)

        loss["loss"] = torch.mean(w * torch.stack(loss["loss"]))
        for i, _w in enumerate(w):
            metrics[f"rollout_weight_{i+1}"] = _w.item()

    else:
        loss["loss"] = torch.mean(torch.stack(loss["loss"]))

    for k, v in loss.items():
        if k == "loss":
            continue

        loss[k] = torch.stack(loss[k])

    return loss, metrics


def roll_forecast(
    model: nn.Module,
    data: list[torch.Tensor],
    forcing: torch.Tensor,
    n_step: int,
    loss_fn: nn.Module | None,
    use_scheduled_sampling: bool,
    step: int | None = None,
    max_step: int | None = None,
    ss_pred_min: float = 0.0,
    ss_pred_max: float = 1.0,
    use_rollout_weighting: bool = False,
    month_idx: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    # torch.Size([32, 2, 1, 128, 128]) torch.Size([32, 1, 1, 128, 128])
    x, y = data
    B, T, C_data, H, W = x.shape

    if loss_fn is not None:
        _, T_y, _, _, _ = y.shape
        assert T_y == n_step, "y does not match n_steps: {} vs {}".format(T_y, n_step)

    assert (
        forcing.shape[1] == n_step + T
    ), "Forcing length {} insufficient for rollout R={}".format(
        forcing.shape[1], n_step
    )

    # Initial state is the last state from input sequence
    current_state = x[:, -1, ...].unsqueeze(1)  # Shape: [B, 1, C, H, W]
    previous_state = x[:, -2, ...].unsqueeze(1)

    # Initialize empty lists for multi-step evaluation
    all_losses = []
    all_predictions_obs = []
    all_tendencies_obs = []
    all_predictions_core = []
    all_tendencies_core = []

    metrics = {}

    n_calib_steps = 2

    # Loop through each rollout step
    for t in range(n_step):
        do_calib = t < n_calib_steps
        # Model always sees forcings two history times and one prediction time
        step_forcing = forcing[:, t : t + T + 1, ...]

        # 1. Assemble input state

        input_state, p_pred, ss_metrics = _build_input_state(
            previous_state,
            current_state,
            x,
            y,
            t,
            use_scheduled_sampling,
            step,
            max_step,
            ss_pred_min,
            ss_pred_max,
        )

        metrics.update(ss_metrics)

        # 2. Forward and unpack

        input_state = _maybe_calibrate_input(model, input_state, month_idx, do_calib)

        tendency_core, tendency_obs, diag = _forward_and_unpack(
            model, input_state, step_forcing, t
        )
        metrics.update(diag)

        next_pred_core = current_state + tendency_core
        next_pred_obs = current_state + tendency_obs

        metrics[f"abs_delta_pred_obst_t{t+1}"] = torch.abs(next_pred_core - next_pred_obs).mean()
        den = torch.abs(next_pred_core - current_state).mean().clamp_min(1e-6)
        metrics[f"rel_abs_delta_obs_t{t+1}"] = torch.abs(next_pred_core - next_pred_obs).mean() / den

        # 3. Calculate loss

        step_loss = _compute_step_loss(
            loss_fn, x, y, t, next_pred_obs, tendency_obs, step
        )
        if step_loss is not None:
            all_losses.append(step_loss)

        # 4. Get next core state

        next_state_core, fb_metrics = _next_core_state(
            model,
            current_state,
            next_pred_core,
            y,
            t,
            training=model.training,
            use_scheduled_sampling=use_scheduled_sampling,
            p_pred=p_pred,
            do_calib=do_calib,
            month_idx=month_idx,
        )
        metrics.update(fb_metrics)

        next_state_obs = _materialize_obs(next_pred_obs, training=model.training)

        # Store the prediction
        all_predictions_core.append(next_state_core.detach())
        all_tendencies_core.append(tendency_core.detach())
        all_predictions_obs.append(next_state_obs.detach())
        all_tendencies_obs.append(tendency_obs.detach())

        # Update current state for next iteration
        previous_state = current_state
        current_state = next_state_core

    loss, metrics = _aggregate_losses(
        all_losses, use_rollout_weighting, step, n_step, max_step, metrics
    )

    assert all_tendencies_core[0].ndim == 5

    if loss is not None:
        loss.update(metrics)

    out = {
        "tendencies_core": torch.cat(all_tendencies_core, dim=1),
        "predictions_core": torch.cat(all_predictions_core, dim=1),
        "tendencies_obs": torch.cat(all_tendencies_obs, dim=1),
        "predictions_obs": torch.cat(all_predictions_obs, dim=1),
    }

    return loss, out
