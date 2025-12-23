# change_metrics.py
import torch
import pandas as pd
from verif.fft_utils import ensure_btchw
import seaborn as sns
import matplotlib.pyplot as plt


def change_metrics(
    run_name: list[str],
    all_truth: torch.Tensor,
    all_predictions: torch.Tensor,
    save_path: str,
    tau: float = 0.02,  # threshold for "significant change"
):
    """
    Computes per-timestep change detection metrics:
      - precision, recall, f1 (binary change detection vs truth)
      - tendency_corr (correlation of Δy_pred and Δy_true)
      - stationarity_ratio (#changed_pred / #changed_true)
    Saves CSV to {save_path}/results/change_metrics.csv
    """
    results = []

    for i in range(len(all_predictions)):
        y_pred = ensure_btchw(all_predictions[i]).to(torch.float32)
        y_true = ensure_btchw(all_truth[i]).to(torch.float32)
        assert y_pred.shape == y_true.shape

        B, T, C, H, W = y_pred.shape
        for t in range(T - 1):  # compare step t -> t+1
            yp0, yp1 = y_pred[:, t], y_pred[:, t + 1]
            yt0, yt1 = y_true[:, t], y_true[:, t + 1]

            dyp = yp1 - yp0
            dyt = yt1 - yt0

            # Binary masks of "change"
            changed_true = dyt.abs() > tau
            changed_pred = dyp.abs() > tau

            tp = (changed_true & changed_pred).sum().item()
            fp = (~changed_true & changed_pred).sum().item()
            fn = (changed_true & ~changed_pred).sum().item()

            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)

            # correlation of tendencies (flatten)
            if dyt.numel() > 1:
                corr_num = ((dyp - dyp.mean()) * (dyt - dyt.mean())).sum()
                corr_den = (
                    dyp.std(unbiased=False) * dyt.std(unbiased=False) * dyp.numel()
                )
                tendency_corr = (corr_num / (corr_den + 1e-8)).item()
            else:
                tendency_corr = float("nan")

            # stationarity ratio
            n_true_changes = changed_true.sum().item()
            n_pred_changes = changed_pred.sum().item()
            stationarity_ratio = n_pred_changes / (n_true_changes + 1e-8)

            results.append(
                {
                    "model": run_name[i],
                    "timestep": t + 1,  # we report the *lead time* (step t→t+1)
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "tendency_corr": tendency_corr,
                    "stationarity_ratio": stationarity_ratio,
                }
            )

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by=["timestep", "model"])
    df.to_csv(f"{save_path}/results/change_metrics.csv", index=False)
    return df


def plot_change_prf_timeseries(df: pd.DataFrame, save_path="runs/verification"):
    """
    Expects columns: model, timestep, precision, recall, f1
    Produces a single figure with three lines per model (precision/recall/f1).
    """
    required = {"model", "timestep", "precision", "recall", "f1"}
    if df.empty or not required.issubset(df.columns):
        print(f"No change PRF results to plot (need columns {required}).")
        return

    num_timesteps = int(df["timestep"].max()) + 1

    # Melt to long for seaborn
    plot_df = df.melt(
        id_vars=["model", "timestep"],
        value_vars=["precision", "recall", "f1"],
        var_name="metric",
        value_name="value",
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=plot_df,
        x="timestep",
        y="value",
        hue="model",
        style="metric",  # precision / recall / f1
        markers=True,
        dashes=True,
    )

    plt.ylim(0.0, 1.05)
    plt.xlabel("Forecast Timestep Index")
    plt.ylabel("Score")
    plt.title("Change Detection: Precision / Recall / F1 vs Lead Time")
    plt.xticks(range(num_timesteps))
    plt.legend(title="Model / Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    filename = f"{save_path}/figures/change_prf_timeseries.png"
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()


def plot_change_corr_stationarity_timeseries(
    df: pd.DataFrame,
    save_path="runs/verification",
):
    """
    Expects columns: model, timestep, tendency_corr, stationarity_ratio
    Draws two families of lines:
      - tendency_corr (target ~ 1.0 is best; 0 means uncorrelated)
      - stationarity_ratio (target ~ 1.0; <1 too conservative, >1 too jittery)
    """
    required = {"model", "timestep", "tendency_corr", "stationarity_ratio"}
    if df.empty or not required.issubset(df.columns):
        print(
            f"No change correlation/stationarity results to plot (need columns {required})."
        )
        return

    num_timesteps = int(df["timestep"].max()) + 1

    # Melt to long form
    plot_df = df.melt(
        id_vars=["model", "timestep"],
        value_vars=["tendency_corr", "stationarity_ratio"],
        var_name="metric",
        value_name="value",
    )
    # nicer labels
    plot_df["metric"] = plot_df["metric"].map(
        {
            "tendency_corr": "tendency_corr",
            "stationarity_ratio": "stationarity_ratio",
        }
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=plot_df,
        x="timestep",
        y="value",
        hue="model",
        style="metric",
        markers=True,
        dashes=True,
    )

    # Reference bands / lines
    # For correlation: [-1,1], we aim for closer to 1; for stationarity ratio: aim ~1
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.8)
    plt.fill_between(
        [0, num_timesteps - 1], 0.9, 1.1, color="gray", alpha=0.06, label=None
    )

    plt.xlabel("Forecast Timestep Index")
    plt.ylabel("Value")
    plt.title("Change Dynamics: Tendency Correlation & Stationarity Ratio vs Lead Time")
    plt.xticks(range(num_timesteps))
    plt.legend(title="Model / Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    filename = f"{save_path}/figures/change_corr_stationarity_timeseries.png"
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()
