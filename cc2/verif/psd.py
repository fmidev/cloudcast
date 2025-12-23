import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _hann2d(nx: int, ny: int, device, dtype, periodic=True):
    wx = torch.hann_window(nx, periodic=periodic, device=device, dtype=dtype)
    wy = torch.hann_window(ny, periodic=periodic, device=device, dtype=dtype)
    return wx[:, None] * wy[None, :]


def infer_spacing_from_original(
    nx_down,
    ny_down,
    nx_orig=949,
    ny_orig=1069,
    dx_orig_km=2.5,
    dy_orig_km=2.5,
    periodic=False,
):
    """
    Returns dx, dy (km) for the *downscaled* data, assuming it spans the same
    physical extent as the original grid.

    periodic=False -> L = (N_orig-1)*dx_orig
    periodic=True  -> L =  N_orig   *dx_orig
    """
    if periodic:
        Lx = nx_orig * dx_orig_km
        Ly = ny_orig * dy_orig_km
    else:
        Lx = (nx_orig - 1) * dx_orig_km
        Ly = (ny_orig - 1) * dy_orig_km
    dx = Lx / nx_down
    dy = Ly / ny_down
    return dx, dy


def _radial_bin(kx, ky, psd2, k_max=None):
    device, dtype = psd2.device, psd2.dtype
    nx, ny = kx.numel(), ky.numel()

    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    kr = torch.sqrt(KX**2 + KY**2)

    nr = min(nx, ny) // 2
    # cap radial bins at axis Nyquist (avoid diagonal > Nyquist artifacts)
    if k_max is None:
        k_max = min(kx.abs().max(), ky.abs().max())
    edges = torch.linspace(0.0, k_max, nr + 1, device=device, dtype=dtype)

    bin_idx = torch.bucketize(kr.reshape(-1), edges, right=False) - 1
    bin_idx = bin_idx.clamp(min=0, max=nr - 1).reshape(nx, ny)

    Pk = []
    counts = []
    for b in range(nr):
        mask = bin_idx == b
        c = mask.sum()
        if c == 0:
            Pk.append(torch.zeros(psd2.shape[:-2], device=device, dtype=dtype))
            counts.append(torch.tensor(1, device=device))
        else:
            val = (psd2 * mask).sum(dim=(-2, -1))
            Pk.append(val)
            counts.append(c)

    Pk = torch.stack(Pk, dim=-1)
    counts = torch.stack(counts).to(Pk)
    Pk = Pk / counts
    k_centers = 0.5 * (edges[:-1] + edges[1:])
    return k_centers, Pk


def interp1d_torch(x, xp, fp):
    """
    Torch version of numpy.interp. Works on 1D tensors.
    x  : points to evaluate
    xp : ascending reference x
    fp : values at xp
    """
    # ensure 1D
    original_shape = fp.shape[:-1]

    # Flatten fp to (batch, m) where batch = product of all dims except last
    fp_2d = fp.reshape(-1, fp.shape[-1])  # (batch, m)

    # Find indices for interpolation
    inds = torch.searchsorted(xp, x)
    inds = torch.clamp(inds, 1, len(xp) - 1)

    # Get neighboring points
    x0, x1 = xp[inds - 1], xp[inds]
    f0 = fp_2d[:, inds - 1]  # (batch, n)
    f1 = fp_2d[:, inds]  # (batch, n)

    # Interpolate
    slope = (f1 - f0) / (x1 - x0)
    result = f0 + slope * (x - x0)

    # Reshape back to original shape with new last dim
    return result.reshape(*original_shape, len(x))


def calculate_psd(data: torch.Tensor):
    data = data.squeeze()
    B, T, nx, ny = data.shape
    device, dtype = data.device, data.dtype
    window = _hann2d(nx, ny, device, dtype, periodic=True)
    win_power = (window**2).sum()

    # d = data.reshape(-1, nx, ny) * window
    d = data * window

    dx, dy = infer_spacing_from_original(
        nx_down=nx,
        ny_down=ny,
        nx_orig=949,
        ny_orig=1069,
        dx_orig_km=2.5,
        dy_orig_km=2.5,
        periodic=False,
    )

    F = torch.fft.fft2(d, dim=(-2, -1), norm="ortho")
    F = torch.fft.fftshift(F, dim=(-2, -1))
    P2 = F.real**2 + F.imag**2

    kx = torch.fft.fftfreq(nx, d=dx, device=device, dtype=dtype)
    ky = torch.fft.fftfreq(ny, d=dy, device=device, dtype=dtype)
    kx = torch.fft.fftshift(kx)
    ky = torch.fft.fftshift(ky)

    # axis Nyquist in km^-1 (smallest resolvable scale = 2*min(dx,dy))
    k_max_axis = min(0.5 / dx, 0.5 / dy)
    k, Pk = _radial_bin(kx, ky, P2, k_max=k_max_axis)

    nz = (k > 0) & (k <= k_max_axis)  # enforce scale >= 2*min(dx,dy)
    k = k[nz]
    Pk = Pk[:, :, nz]

    wavelength = 1.0 / k
    Pk = Pk / (win_power / (nx * ny))
    Pk_mean = Pk.mean(dim=0)

    return wavelength, wavelength, Pk_mean


def psd(all_truth: torch.tensor, all_predictions: torch.tensor, save_path: str):
    _ensure_dir(os.path.join(save_path, "results"))

    truth = all_truth[0]
    truth = truth.to(device)
    # Remove initialization time as it skewes the results
    truth = truth[:, 1:, ...]

    sx, sy, psd_q = calculate_psd(truth)
    observed_psd = {"sx": sx, "sy": sy, "psd": psd_q}

    predicted_psds = []
    for i in range(len(all_predictions)):
        prediction = all_predictions[i]  # F, T, C, H, W

        # Remove initialization time as it skewes the results
        prediction = prediction[:, 1:, ...]

        sx_p, sy_p, psd_p = calculate_psd(prediction.to(device))
        predicted_psds.append({"sx": sx_p, "sy": sy_p, "psd": psd_p})

    torch.save(observed_psd, f"{save_path}/results/observed_psd.pt")
    torch.save(predicted_psds, f"{save_path}/results/predicted_psd.pt")

    return observed_psd, predicted_psds


def plot_psd(
    run_name: list[str],
    obs_psd: dict,
    pred_psds: list[dict],
    #    pred_psds_r1: list[dict],
    save_path: str,
):
    _ensure_dir(os.path.join(save_path, "figures"))

    def set_x_axis(ax, sx_o):
        left_endpoint = 2400  # float(sx_o.max())
        right_endpoint = 10
        ax.set_xlim(left_endpoint, right_endpoint)

        log_min = np.floor(np.log10(right_endpoint))  # e.g., log10(11) -> 1
        log_max = np.ceil(np.log10(left_endpoint))  # e.g., log10(5329) -> 4
        major_log_ticks = [10**i for i in range(int(log_min), int(log_max))]
        major_log_ticks = [
            t for t in major_log_ticks if right_endpoint <= t <= left_endpoint
        ]

        final_ticks = sorted(
            list(set(major_log_ticks + [left_endpoint, right_endpoint]))
        )

        final_labels = []
        for tick in final_ticks:
            if tick in major_log_ticks:
                # Format as a power of 10, e.g., $10^3$
                exponent = int(np.log10(tick))
                final_labels.append(f"$10^{{{exponent}}}$")
            else:
                # Format endpoints as rounded integers
                final_labels.append(f"{int(round(tick))}")

        ax.set_xticks(final_ticks)
        ax.set_xticklabels(final_labels)
        ax.minorticks_on()
        ax.xaxis.set_minor_formatter(NullFormatter())

    def _to_np(t):
        return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t

    def init_plot(opsd):
        plt.figure(figsize=(8, 5))
        plt.xlabel("Horizontal Scale (km)", fontsize=12)
        plt.ylabel("PSD", fontsize=12)
        sx = obs_psd["sx"]
        # psd = obs_psd["psd"]
        plt.loglog(
            _to_np(sx), _to_np(opsd), label="Observed", linewidth=2, color="black"
        )

    # Copy to cpu

    sx_o = obs_psd["sx"].to("cpu")
    psd_o = obs_psd["psd"].to("cpu")

    for i in range(len(run_name)):
        pred_psds[i]["sx"] = pred_psds[i]["sx"].cpu()
        pred_psds[i]["psd"] = pred_psds[i]["psd"].cpu()

    def absolute_psd():

        # ---------------- Absolute PSD ----------------
        init_plot(psd_o.mean(dim=0))
        plt.title("Power Spectral Density", fontsize=14)
        plt.grid(True, alpha=0.7)

        for i in range(len(run_name)):
            sx = pred_psds[i]["sx"]
            psd = pred_psds[i]["psd"].mean(dim=0)
            if sx[0] > sx[-1]:
                sx = torch.flip(sx, dims=[0])
                psd = torch.flip(psd, dims=[0])
            psd_interp = interp1d_torch(sx_o, sx, psd)
            plt.loglog(
                _to_np(sx_o), _to_np(psd_interp), label=run_name[i], linewidth=1.8
            )

        ax = plt.gca()
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=10))
        ax.invert_xaxis()
        ax.set_xlim(float(sx_o.max()), float(sx_o.min()))
        plt.legend(fontsize=10)
        set_x_axis(ax, sx_o)

        filename = f"{save_path}/figures/psd.png"
        plt.savefig(filename, dpi=200)
        print(f"Plot saved to {filename}")
        plt.close()

    def absolute_1h_psd():
        # ---------------- Rollout-1 Absolute ----------------
        init_plot(psd_o[0])
        plt.title("Power Spectral Density Rollout 1", fontsize=14)
        plt.grid(True, alpha=0.7)

        for i in range(len(run_name)):
            sx = pred_psds[i]["sx"].cpu()
            psd = pred_psds[i]["psd"].cpu()[0]
            if sx[0] > sx[-1]:
                sx = torch.flip(sx, dims=[0])
                psd = torch.flip(psd, dims=[0])
            psd_interp = interp1d_torch(sx_o, sx, psd)
            plt.loglog(
                _to_np(sx_o), _to_np(psd_interp), label=run_name[i], linewidth=1.8
            )

        ax = plt.gca()
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=10))
        ax.invert_xaxis()
        ax.set_xlim(float(sx_o.max()), float(sx_o.min()))
        plt.legend(fontsize=10)

        set_x_axis(ax, sx_o)

        filename = f"{save_path}/figures/psd_r1.png"
        plt.savefig(filename, dpi=200)
        print(f"Plot saved to {filename}")
        plt.close()
        plt.clf()

    def anomaly_psd():
        # ---------------- Anomaly vs Observed ----------------
        plt.figure(figsize=(8, 5))
        plt.title("PSD Anomaly", fontsize=14)
        plt.xlabel("Horizontal Scale (km)", fontsize=12)
        plt.ylabel("log10(pred/obs)", fontsize=12)
        plt.grid(True, alpha=0.7)
        plt.axhline(0.0, color="k", linestyle="-", linewidth=1, label="Observed")

        for i in range(len(run_name)):
            sx = pred_psds[i]["sx"].cpu()
            psd = pred_psds[i]["psd"].cpu().mean(dim=0)
            if sx[0] > sx[-1]:
                sx = torch.flip(sx, dims=[0])
                psd = torch.flip(psd, dims=[0])
            psd_interp = interp1d_torch(sx_o, sx, psd)
            anomaly = torch.log10(psd_interp) - torch.log10(psd_o.mean(dim=0))
            plt.semilogx(
                _to_np(sx_o), _to_np(anomaly), label=run_name[i], linewidth=1.8
            )

        ax = plt.gca()
        ax.invert_xaxis()
        ax.set_xlim(float(sx_o.max()), float(sx_o.min()))
        plt.legend(fontsize=10, ncol=2)

        set_x_axis(ax, sx_o)

        filename = f"{save_path}/figures/psd_anomaly.png"
        plt.savefig(filename, dpi=200)
        print(f"Plot saved to {filename}")
        plt.close()
        plt.clf()

    def anomaly_psd_leadtime():
        plt.figure(figsize=(8, 5))
        plt.title("PSD Anomaly scales <100km", fontsize=14)
        plt.xlabel("Leadtime (h)", fontsize=12)
        plt.ylabel("log10(pred/obs)", fontsize=12)
        plt.grid(True, alpha=0.7)
        plt.axhline(0.0, color="k", linestyle="-", linewidth=1, label="Observed")

        for i in range(len(run_name)):
            sx = pred_psds[i]["sx"]
            psd = pred_psds[i]["psd"]
            psd_interp = []
            if sx[0] > sx[-1]:
                sx = torch.flip(sx, dims=[0]).cpu()
                psd = torch.flip(psd, dims=[1]).cpu()

            for j in range(psd.shape[0]):
                psd_interp.append(interp1d_torch(sx_o, sx, psd[j]))

            psd_interp = torch.stack(psd_interp)
            mask = sx_o < 100

            psd_o_masked = psd_o[:, mask]
            psd_masked = psd_interp[:, mask]

            assert psd_o_masked.shape == psd_masked.shape

            anomaly = (torch.log10(psd_masked) - torch.log10(psd_o_masked)).mean(dim=1)

            plt.plot(
                np.arange(1, psd_masked.shape[0] + 1),
                _to_np(anomaly),
                label=run_name[i],
                linewidth=1.8,
            )

        plt.legend(fontsize=10, ncol=2)

        filename = f"{save_path}/figures/psd_anomaly_100km.png"
        plt.savefig(filename, dpi=200)
        print(f"Plot saved to {filename}")
        plt.close()
        plt.clf()

    def anomaly_1h_psd():
        # ---------------- Rollout-1 Anomaly ----------------
        plt.figure(figsize=(8, 5))
        plt.title("PSD Anomaly Rollout 1", fontsize=14)
        plt.xlabel("Horizontal Scale (km)", fontsize=12)
        plt.ylabel("log10(pred/obs)", fontsize=12)
        plt.grid(True, alpha=0.7)
        # Horizontal reference line at 0 (observed spectrum)
        plt.axhline(0.0, color="k", linestyle="-", linewidth=1, label="Observed")

        for i in range(len(run_name)):
            sx = pred_psds[i]["sx"].cpu()
            psd = pred_psds[i]["psd"].cpu()[0]
            if sx[0] > sx[-1]:
                sx = torch.flip(sx, dims=[0]).cpu()
                psd = torch.flip(psd, dims=[0]).cpu()
            psd_interp = interp1d_torch(sx_o, sx, psd)
            anomaly = torch.log10(psd_interp) - torch.log10(psd_o[0])
            plt.semilogx(
                _to_np(sx_o), _to_np(anomaly), label=run_name[i], linewidth=1.8
            )

        ax = plt.gca()
        ax.invert_xaxis()
        ax.set_xlim(float(sx_o.max()), float(sx_o.min()))
        plt.legend(fontsize=10, ncol=2)

        set_x_axis(ax, sx_o)

        filename = f"{save_path}/figures/psd_r1_anomaly.png"
        plt.savefig(filename, dpi=200)
        print(f"Plot saved to {filename}")
        plt.close()
        plt.clf()

    # absolute_psd()
    # absolute_1h_psd()
    anomaly_psd()
    anomaly_1h_psd()
    anomaly_psd_leadtime()
