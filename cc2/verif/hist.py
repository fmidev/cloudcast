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
    results: list,
    save_path: str,
):
    num_models = len(results)
    fig, axes = plt.subplots(1, num_models, figsize=(num_models * 3, 3), sharey=True)
    if num_models == 1:
        axes = [axes]

    fig.suptitle("TCC Histograms", fontsize=14)

    for c, (ax, entry) in enumerate(zip(axes, results)):
        counts = entry["hist"]
        ax.bar(range(len(counts)), counts / counts.sum(), width=1.0)
        ax.set_title(entry["model"], fontsize=9)
        ax.set_xlabel("TCC bin")
        if c == 0:
            ax.set_ylabel("Fraction")

    plt.tight_layout()
    filename = f"{save_path}/figures/hist.png"
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()


