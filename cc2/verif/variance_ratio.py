# variance_ratio.py
import torch
import pandas as pd
from verif.fft_utils import ensure_btchw
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def variance_ratio(
    run_name: list[str],
    all_truth: torch.Tensor,
    all_predictions: torch.Tensor,
    save_path: str,
):
    """
    var_ratio(t) = Var(pred[...,t,...]) / Var(true[...,t,...])
    Averages over [B,C,H,W] for that timestep.
    Saves CSV at {save_path}/results/variance_ratio.csv
    """
    results = []

    for i in range(len(all_predictions)):
        y_pred = ensure_btchw(all_predictions[i])
        y_true = ensure_btchw(all_truth[i])
        assert y_pred.shape == y_true.shape, "Mismatched shapes"

        # [B,T,C,H,W] -> loop timesteps
        T = y_pred.shape[1]
        for t in range(T):
            yp = y_pred[:, t].to(torch.float32)
            yt = y_true[:, t].to(torch.float32)
            var_pred = yp.var(
                unbiased=False, dim=(0, 1, 2, 3)
            ).mean()  # average channel variances
            var_true = yt.var(unbiased=False, dim=(0, 1, 2, 3)).mean()
            ratio = (var_pred / (var_true + 1e-8)).item()
            results.append(
                {"model": run_name[i], "timestep": t, "variance_ratio": ratio}
            )

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by=["timestep", "variance_ratio"], ascending=[True, False])
    df.to_csv(f"{save_path}/results/variance_ratio.csv", index=False)
    return df


def plot_variance_ratio(df: pd.DataFrame, save_path="runs/verification"):
    if df.empty:
        print("No variance_ratio results to plot.")
        return

    num_timesteps = df["timestep"].max() + 1

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="timestep",
        y="variance_ratio",
        hue="model",
        style="model",
        markers=True,
        dashes=False,
    )

    # reference lines
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.8)
    plt.fill_between([0, num_timesteps - 1], 0.9, 1.1, color="gray", alpha=0.08)

    plt.xlabel("Forecast Timestep Index")
    plt.ylabel("Variance Ratio (Pred / True)")
    plt.title("Variance Ratio vs Lead Time")
    plt.xticks(range(num_timesteps))
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    filename = f"{save_path}/figures/variance_ratio_timeseries.png"
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()
