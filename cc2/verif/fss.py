import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
categories = ["Clear", "Partly cloudy", "Mostly cloudy", "Overcast"]


def get_mask_sizes(n: int = 4) -> list[int]:
    mask_sizes = [2]
    for i in range(1, n):
        mask_sizes.append(mask_sizes[-1] + int(round(1.5**i)))

    return torch.tensor(mask_sizes)


def compute_fss_per_leadtime(
    truth: torch.Tensor,
    preds: torch.Tensor,
    thresholds: list[tuple[float, float]],
    mask_sizes: list[int],
) -> torch.Tensor:
    """
    Compute Fractions Skill Score (FSS) separately for each lead time.

    Args:
        truth (torch.Tensor): Ground truth of shape [N, T, 1, H, W].
        preds (torch.Tensor): Predictions of shape [N, T, 1, H, W].
        thresholds (list of (low, high)): Cloud-cover bins.
        mask_sizes (list of int): Square mask sizes in pixels.

    Returns:
        torch.Tensor: FSS scores of shape [T, len(thresholds), len(mask_sizes)].
    """
    N, T, C, H, W = truth.shape
    n_thresh = len(thresholds)
    n_sizes = len(mask_sizes)

    # Prepare output: per lead time
    fss_scores = torch.zeros((T, n_thresh, n_sizes), device="cpu")

    with tqdm(total=n_thresh * n_sizes, desc="Calculating FSS") as pbar:
        # Loop over lead times
        for t in range(T):
            truth_t = truth[:, t, :, :, :]  # [N,1,H,W]
            pred_t = preds[:, t, :, :, :]

            # For each category
            for i, (low, high) in enumerate(thresholds):
                truth_bin = ((truth_t >= low) & (truth_t < high)).float()
                pred_bin = ((pred_t >= low) & (pred_t < high)).float()

                # Loop over mask sizes
                for j, size in enumerate(mask_sizes):
                    pad = (size // 2).item()
                    # Compute local fractions

                    frac_truth = F.avg_pool2d(
                        truth_bin, kernel_size=size.item(), stride=1, padding=pad
                    )
                    frac_pred = F.avg_pool2d(
                        pred_bin, kernel_size=size.item(), stride=1, padding=pad
                    )

                    # Flatten to vector
                    f_t = frac_truth.view(-1)
                    f_o = frac_pred.view(-1)

                    # Compute FSS
                    mse = torch.mean((f_o - f_t) ** 2)
                    mse_ref = torch.mean(f_o**2 + f_t**2)
                    fss_scores[t, i, j] = (1 - mse / mse_ref).cpu().item()

                    pbar.update(1)

    # leadtimes, categories, masks to categories, masks, leadtimes
    fss_scores = torch.permute(fss_scores, (1, 2, 0))

    return fss_scores


def fss(
    run_name: list[str],
    all_truth: torch.Tensor,
    all_predictions: list[torch.Tensor],
    save_path: str,
    thresholds: list[tuple[float, float]] = None,
) -> torch.Tensor:
    """
    Compute Fractions Skill Score (FSS) for multiple model predictions across categories and mask sizes.

    Args:
        all_truth (torch.Tensor): Ground truth tensor of shape [N, T, 1, H, W].
        all_predictions (list of torch.Tensor): List of prediction tensors, each of shape [N, T, 1, H, W].
        thresholds (optional): List of (lower, upper) thresholds. Defaults to
            [(0, 0.0625), (0.0625, 0.5625), (0.5625, 0.9375), (0.9375, 1.01)].
        mask_sizes (optional): List of mask sizes (pixels). Defaults to first 10 odd sizes [1, 3, ..., 19].

    Returns:
        torch.Tensor: FSS scores of shape [M, len(thresholds), len(mask_sizes)], where M is # of models.
    """
    if thresholds is None:
        thresholds = [(0, 0.0625), (0.0625, 0.5625), (0.5625, 0.9375), (0.9375, 1.01)]

    mask_sizes = get_mask_sizes()

    n_models = len(all_predictions)

    n_thresh = len(thresholds)
    n_sizes = len(mask_sizes)

    results = []

    for idx, preds in enumerate(all_predictions):
        truth = all_truth[idx].to(device)
        preds = preds.to(device)
        r = compute_fss_per_leadtime(truth, preds, thresholds, mask_sizes)
        results.append(r)

    # obs
    truth_t = all_truth[0]
    observed_categories = []
    observed_sum = 0

    for i, (low, high) in enumerate(thresholds):
        truth_bin = ((truth_t >= low) & (truth_t < high)).float()
        observed_categories.append(torch.sum(truth_bin))
        observed_sum += observed_categories[-1]

    observed_categories_frac = [x / observed_sum for x in observed_categories]
    results.append(observed_categories_frac)

    torch.save(results, f"{save_path}/results/fss_leadtime.pt")

    m = 2  # mask index = 6px

    x = torch.arange(results[0].shape[2])

    df = []
    for i in range(n_models):
        for c in range(len(categories)):
            for t in range(results[0].shape[2]):
                fss_1d = results[i][c, :, t].mean()
                df.append(
                    {
                        "model": run_name[i],
                        "category": categories[c],
                        "timestep": t,
                        "fss": fss_1d.numpy().item(),
                    }
                )
                print(df[-1])

    df = pd.DataFrame(df)
    if not df.empty:
        df = df.sort_values(by=["category", "timestep", "model"])

    df.to_csv(f"{save_path}/results/fss.csv", index=False)
    return results, df


def plot_fss(
    run_name: list[str],
    results: torch.tensor,
    save_path: str,
):
    # plot_fss_2d(run_name, results, save_path)
    plot_fss_1d(run_name, results, save_path)


def plot_fss_1d(
    run_name: list[str],
    results: list[torch.tensor],
    save_path: str,
):
    results = results[:-1]
    plt.close("all")
    num_categories = results[0].shape[0]
    num_masks = results[0].shape[1]
    num_leadtimes = results[0].shape[2]

    mask_sizes = get_mask_sizes(num_masks)

    x = torch.arange(num_leadtimes)

    m = 2  # mask index = 6px

    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    ax = ax.flatten()

    for c in range(num_categories):
        for i, r in enumerate(results):
            ax[c].plot(x, r[c, m, :], label=run_name[i])
            ax[c].set_xlabel("Lead time (h)")
            ax[c].set_ylabel("Fraction Skill Score")
            ax[c].set_title(f"FSS for category {categories[c]}")
            ax[c].legend()
    plt.suptitle(f"FSS for four categories (mask size {mask_sizes[m]*5}km)")
    filename = f"{save_path}/figures/fss_1d.png"
    plt.savefig(filename)
    print(f"Plot saved to {filename}")


def plot_fss_2d(
    run_name: list[str],
    results: torch.tensor,
    save_path: str,
):
    plt.close("all")

    dx = 5  # kilometers
    observed = results.pop()

    # results: list of fss scores, each with shape (num_leadtimes, num_categories, num_masks)
    num_leadtimes = results[0].shape[2]
    num_categories = results[0].shape[0]
    num_masks = results[0].shape[1]

    categories = ["Clear", "Partly cloudy", "Mostly cloudy", "Overcast"]

    mask_sizes = get_mask_sizes(num_masks)

    x = torch.arange(num_leadtimes)
    y = mask_sizes * dx

    xx, yy = torch.meshgrid(x, y, indexing="ij")
    levels = torch.linspace(0.3, 1.0, 21)

    for i, r in enumerate(results):
        for j, c in enumerate(categories):
            plt.figure(figsize=(8, 8))

            fss_good = 0.5 + observed[j] * 0.5

            v = torch.permute(r[j], (1, 0))

            plt.contourf(xx, yy, v, levels=levels)
            plt.colorbar()
            plt.title(
                "FSS for category '{}' (FSS_good={:.2f}) '{}'".format(
                    c, fss_good, run_name[i]
                )
            )
            plt.xlabel("Leadtime (hours)")
            plt.ylabel("Mask size (km)")
            CS = plt.contour(xx, yy, v, [fss_good])
            plt.clabel(CS, inline=True, fontsize=10)

            model_name = run_name[i].replace("/", "_")
            cat_name = c.replace(" ", "_").lower()
            filename = f"{save_path}/figures/fss_2d_{model_name}_{cat_name}.png"
            plt.savefig(filename)
            plt.close()
            print(f"Plot saved to {filename}")
