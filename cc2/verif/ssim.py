import torch
import pandas as pd
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib.pyplot as plt
from torchmetrics.functional.image import (
    structural_similarity_index_measure,
)
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ssim(
    run_name: list[str],
    all_truth: torch.tensor,
    all_predictions: torch.tensor,
    save_path: str,
):
    results = []

    for i in range(len(all_predictions)):
        predictions = all_predictions[i]
        truth = all_truth[i]
        B, T, C, H, W = predictions.shape

        # Compute SSIM for each timestep
        ssim_per_step = []
        for t in tqdm(range(T), desc="Calculating SSIM"):
            ssim_t = structural_similarity_index_measure(
                predictions[:, t].to(device), truth[:, t].to(device), data_range=1.0
            ).item()
            ssim_per_step.append(ssim_t)

        # Append results in long format
        for timestep_index, ssim_score in enumerate(ssim_per_step):
            results.append(
                {
                    "model": run_name[i],
                    "timestep": timestep_index,
                    "ssim": ssim_score,
                }
            )

    # Add eulerian persistence

    base_truth = all_truth[0]  # just pick the first truth
    # persistence forecast = base_truth[0] at every step
    pers = base_truth[:, 0:1]  # shape (1, C, H, W)
    pers = pers.expand_as(base_truth)  # shape (num_steps, C, H, W)

    ssim_per_step = []
    for t in range(T):
        ssim_t = structural_similarity_index_measure(
            pers[:, t].to(device), all_truth[0][:, t].to(device), data_range=1.0
        ).item()
        ssim_per_step.append(ssim_t)

    for t, score in enumerate(ssim_per_step):
        results.append(
            {
                "model": "eulerian-persistence",
                "timestep": t,
                "ssim": score,
            }
        )

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(
            by=["timestep", "ssim"], ascending=[True, False]
        )

    results_df.to_csv(f"{save_path}/results/ssim.csv", index=False)

    return results_df


def plot_ssim(df: pd.DataFrame, save_path="runs/verification"):
    if df.empty:
        print("No results to plot.")
        return

    num_timesteps = df["timestep"].max() + 1
    num_models = df["model"].nunique()

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=df,
        x="timestep",
        y="ssim",
        hue="model",
        style="model",  # Optional: Different marker/line style for each model (useful for B&W)
        markers=True,  # Show markers on the points
        dashes=False,  # Use solid lines unless you have many models
    )

    plt.xlabel("Forecast Timestep Index")
    plt.ylabel("SSIM (Lower is Better)")
    plt.title("Model Comparison: SSIM per Forecast Timestep")
    plt.xticks(range(num_timesteps))  # Ensure ticks for each integer timestep
    plt.legend(title="Model ID", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend

    filename = f"{save_path}/figures/ssim_timeseries.png"
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()
