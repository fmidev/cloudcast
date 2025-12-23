import torch
import pandas as pd
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib.pyplot as plt


def mae(
    run_name: list[str],
    all_truth: torch.tensor,
    all_predictions: torch.tensor,
    save_path: str,
):
    results = []

    for i in range(len(all_predictions)):
        predictions = all_predictions[i]
        truth = all_truth[i]

        abs_diff = torch.abs(predictions - truth)
        dims_to_average = [i for i in range(predictions.ndim) if i != 1]
        mae_per_step = torch.mean(abs_diff, dim=dims_to_average)
        mae_per_step = mae_per_step.tolist()

        # Append results in long format
        for timestep_index, mae_score in enumerate(mae_per_step):
            results.append(
                {
                    "model": run_name[i],
                    "timestep": timestep_index,
                    "mae": mae_score,
                }
            )

    # Add eulerian persistence

    base_truth = all_truth[0]  # just pick the first truth
    # persistence forecast = base_truth[0] at every step
    pers = base_truth[:, 0:1]  # shape (1, C, H, W)
    pers = pers.expand_as(base_truth)  # shape (num_steps, C, H, W)

    abs_diff = torch.abs(pers - base_truth)

    mae_per_step = torch.mean(abs_diff, dim=(0, 2, 3, 4)).tolist()

    for t, score in enumerate(mae_per_step):
        results.append(
            {
                "model": "eulerian-persistence",
                "timestep": t,
                "mae": score,
            }
        )

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(by="mae", ascending=True)

    results_df.to_csv(f"{save_path}/results/mae.csv")

    return results_df


def mae2d(
    all_truth: torch.tensor,
    all_predictions: torch.tensor,
    save_path: str,
):
    results = []

    for i in range(len(all_predictions)):
        y_pred = all_predictions[i]
        y_true = all_truth[i]

        abs_diff = torch.abs(y_pred - y_true)

        assert len(abs_diff.shape) == 5, "Invalid shape: {}".format(
            abs_diff.shape
        )  # B, C, 1, H, W

        mae2d = torch.mean(abs_diff, dim=(0, 2))

        results.append(mae2d)

    torch.save(results, f"{save_path}/results/mae2d.pt")
    return results


def plot_mae_timeseries(
    df: pd.DataFrame, save_path="runs/verification"
):
    if df.empty:
        print("No results to plot.")
        return

    num_timesteps = df["timestep"].max() + 1
    num_models = df["model"].nunique()

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=df,
        x="timestep",
        y="mae",
        hue="model",
        style="model",  # Optional: Different marker/line style for each model (useful for B&W)
        markers=True,  # Show markers on the points
        dashes=False,  # Use solid lines unless you have many models
    )

    plt.xlabel("Forecast Timestep Index")
    plt.ylabel("Mean Absolute Error (Lower is Better)")
    plt.title("Model Comparison: MAE per Forecast Timestep")
    plt.xticks(range(num_timesteps))  # Ensure ticks for each integer timestep
    plt.legend(title="Model ID", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend

    filename = f"{save_path}/figures/mae_timeseries.png"
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
