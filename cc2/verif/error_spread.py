import torch
import pandas as pd
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib.pyplot as plt


def rmse_std_ratio(predictions, truth):
    # pred shape: B, M, T, C, H, W:
    predictions = predictions.to("cuda")
    truth = truth.to("cuda")

    assert (
        predictions.shape[1] > 1
    ), "Cannot calculate variance if there is only one member"

    mean_pred = torch.mean(predictions, dim=1)
    std_pred = torch.std(predictions, dim=1)

    rmse = torch.mean((mean_pred - truth) ** 2, dim=(0, 2, 3, 4)).sqrt()
    std = torch.mean(std_pred, dim=(0, 2, 3, 4))

    return std.cpu(), rmse.cpu()


def error_spread(
    run_name: list[str],
    all_truth: torch.tensor,
    all_predictions: torch.tensor,
    save_path: str,
):
    results = []

    for i in range(len(all_predictions)):
        predictions = all_predictions[i]
        truth = all_truth[i]

        std, rmse = rmse_std_ratio(predictions, truth)

        # Append results in long format
        for j in range(std.shape[0]):
            results.append(
                {
                    "model": run_name[i],
                    "timestep": j,
                    "std": std[j].item(),
                    "rmse": rmse[j].item(),
                }
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{save_path}/results/error_spread.csv", index=False)

    return results_df


def plot_error_spread(df: pd.DataFrame, save_path="runs/verification"):
    if df.empty:
        print("No results to plot.")
        return

    num_timesteps = df["timestep"].max() + 1
    num_models = df["model"].nunique()

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=df,
        x="timestep",
        y="rmse",
        hue="model",
        style="model",  # Optional: Different marker/line style for each model (useful for B&W)
        markers=True,
        dashes=False,
    )
    sns.lineplot(
        data=df,
        x="timestep",
        y="std",
        hue="model",
        style="model",
        markers=True,
        dashes=True,
    )

    plt.xlabel("Forecast Timestep Index")
    #    plt.ylabel("Mean Absolute Error (Lower is Better)")
    plt.title("Error vs Spread")
    plt.xticks(range(num_timesteps))  # Ensure ticks for each integer timestep
    plt.legend(title="Model ID", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend

    filename = f"{save_path}/figures/error_spread.png"
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()


def plot_mae2d(
    run_name: list[str],
    results: torch.tensor,
    save_path: str,
):
    # results shape: num_models, steps, height, width
    num_models = len(results)
    num_timesteps = results[0].shape[0] - 1  # skip first step as mae = 0
    nrows = num_models
    ncols = num_timesteps

    fig_width = ncols * 2.5
    fig_height = nrows * 2.5
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_width, fig_height), sharex=True, sharey=True
    )

    cmap = mcolors.LinearSegmentedColormap.from_list("white_red", ["white", "red"])
    vmin, vmax = 0, 1

    fig.suptitle(f"Spatial MAE", fontsize=16)

    axes = axes if axes.ndim == 2 else axes.reshape((nrows, ncols))

    for r in range(nrows):
        y_pred = results[r]
        for c in range(ncols):
            ax = axes[r, c]

            if c == 0:
                ax.set_ylabel(run_name[r], fontsize=10, rotation=90, labelpad=10)

            if r == 0:
                ax.set_title(f"Timestep {c+1}h", fontsize=10)

            ax.set_xticks([])
            ax.set_yticks([])

            im = ax.imshow(y_pred[c + 1], cmap=cmap, vmin=vmin, vmax=vmax)

    # Add colorbar at the far right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("MAE", fontsize=10)

    plt.subplots_adjust(wspace=0.1, hspace=0.1, right=0.9)

    filename = f"{save_path}/figures/mae2d.png"
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()
