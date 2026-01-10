import torch
import pandas as pd
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hist(
    run_name: list[str],
    all_truth: torch.tensor,
    all_predictions: torch.tensor,
    save_path: str,
):
    results = []

    for i in tqdm(range(len(all_predictions)), desc="Calculating hist"):
        predictions = all_predictions[i]
        truth = all_truth[i]
        B, T, C, H, W = predictions.shape

        _hist = torch.histogram(predictions)[0].numpy()

        # Append results in long format
        results.append(
              {
                  "model": run_name[i],
                  "hist": _hist,
              }
        )
    print(results)
    torch.save(results, f"{save_path}/results/hist.pt", weights_only=False)

    return results


def plot_hist(
    run_name: list[str],
    results: pd.DataFrame,
    save_path: str,
):
    # results shape: num_models, steps, height, width
    num_models = len(results)
    nrows = 1
    ncols = num_models

    fig_width = ncols * 2.5
    fig_height = nrows * 2.5
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_width, fig_height), sharex=True, sharey=True
    )

    fig.suptitle(f"Histograms", fontsize=16)

    axes = axes if axes.ndim == 2 else axes.reshape((nrows, ncols))

    for r in range(nrows):
        y_pred = results[r]
        for c in range(ncols):
            ax = axes[r, c]

            ax.set_title(run_name[c], fontsize=10)

            ax.set_xticks([])
            ax.set_yticks([])

            im = ax.hist(results[r], density=True)

    # Add colorbar at the far right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("MAE", fontsize=10)

    plt.subplots_adjust(wspace=0.1, hspace=0.1, right=0.9)

    filename = f"{save_path}/figures/mae2d.png"
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()


