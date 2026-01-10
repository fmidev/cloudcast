# highk_power_ratio.py
import torch
import pandas as pd
from torch.fft import rfft2
from verif.fft_utils import ensure_btchw, radial_bins_rfft, band_masks_from_wavelengths
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def highk_power_ratio(
    run_name: list[str],
    all_truth: torch.Tensor,
    all_predictions: torch.Tensor,
    save_path: str,
    dx_km: float = 5.0,
    high_km: float = 30.0,
    n_bins: int | None = None,
):
    """
    High-k power ratio over *physical* high-k band (<high_km, default <30 km):
      mean PSD_pred over bins / mean PSD_true over bins (same bins).
    """
    results = []

    for i in range(len(all_predictions)):
        y_pred = ensure_btchw(all_predictions[i]).to(torch.float32)
        y_true = ensure_btchw(all_truth[i]).to(torch.float32)
        assert y_pred.shape == y_true.shape

        B, T, C, H, W = y_pred.shape

        for t in range(T):
            yp = y_pred[:, t].to(device)  # [B,C,H,W]
            yt = y_true[:, t].to(device)

            X0 = rfft2(yp[0], norm="ortho")
            Hf, Wf = X0.shape[-2], X0.shape[-1]
            bin_index, counts, nb, edges, rmax = radial_bins_rfft(
                Hf, Wf, device=yp.device, n_bins=n_bins, return_meta=True
            )
            flat_idx = bin_index.flatten()

            # Use the SAME physical definition as above: high-k < 30 km
            _, high_mask, _ = band_masks_from_wavelengths(
                edges=edges,
                rmax=rmax,
                dx_km=dx_km,
                mid_km=(30.0, 60.0),
                high_km=high_km,
            )

            psd_pred_bins = []
            psd_true_bins = []

            for b in range(B):
                X = rfft2(yp[b], norm="ortho")  # [C,Hf,Wf]
                Y = rfft2(yt[b], norm="ortho")

                PX = (X.real**2 + X.imag**2).mean(dim=0)  # [Hf,Wf]
                PY = (Y.real**2 + Y.imag**2).mean(dim=0)

                def bin_mean(Z):
                    zb = Z.reshape(-1, Hf * Wf)
                    sums = torch.zeros(zb.shape[0], nb, device=Z.device, dtype=Z.dtype)
                    sums.index_add_(1, flat_idx, zb)
                    return (sums / counts).squeeze(0)

                psd_pred_bins.append(bin_mean(PX[None, ...]))
                psd_true_bins.append(bin_mean(PY[None, ...]))

            PSDp = torch.stack(psd_pred_bins, dim=0).mean(0)  # [nb]
            PSDt = torch.stack(psd_true_bins, dim=0).mean(0)  # [nb]

            if high_mask.any():
                hi_pred = PSDp[high_mask].mean()
                hi_true = PSDt[high_mask].mean()
                ratio = (hi_pred / (hi_true + 1e-8)).item()
            else:
                ratio = float("nan")

            results.append(
                {"model": run_name[i], "timestep": t, "highk_power_ratio": ratio}
            )

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(
            by=["timestep", "highk_power_ratio"], ascending=[True, False]
        )
    df.to_csv(f"{save_path}/results/highk_power_ratio.csv", index=False)
    return df


def plot_highk_power_ratio(df: pd.DataFrame, save_path="runs/verification"):
    if df.empty:
        print("No high_k_power_ratio results to plot.")
        return

    num_timesteps = df["timestep"].max() + 1

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="timestep",
        y="highk_power_ratio",
        hue="model",
        style="model",
        markers=True,
        dashes=False,
    )

    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.8)
    plt.fill_between([0, num_timesteps - 1], 0.85, 1.15, color="gray", alpha=0.08)

    plt.xlabel("Forecast Timestep Index")
    plt.ylabel("High-k Power Ratio (Pred / True)")
    plt.title("Fine-Scale Energy vs Lead Time")
    plt.xticks(range(num_timesteps))
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    filename = f"{save_path}/figures/highk_power_ratio_timeseries.png"
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()
