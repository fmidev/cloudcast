# highk_power_ratio.py
import torch
import pandas as pd
from torch.fft import rfft2
from verif.fft_utils import ensure_btchw, radial_bins_rfft
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def spectral_coherence_bands(
    run_name: list[str],
    all_truth: torch.Tensor,
    all_predictions: torch.Tensor,
    save_path: str,
    k_mid: float = 0.30,
    k_high: float = 0.60,
    n_bins: int | None = None,
):
    """
    For each timestep: compute coherence(k) = Re(X * conj(Y)) / sqrt(PSD_x * PSD_y),
    then report two points: Coh(k_mid), Coh(k_high).
    Saves {save_path}/results/spectral_coherence.csv
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

            # build bins once from first sample
            X0 = rfft2(yp[0], norm="ortho")
            Hf, Wf = X0.shape[-2], X0.shape[-1]
            bin_index, counts, nb = radial_bins_rfft(
                Hf, Wf, device=yp.device, n_bins=n_bins
            )
            flat_idx = bin_index.flatten()

            # accumulate binned PSD and cross-power over batch
            PSDx_list, PSDy_list, Rxy_list = [], [], []

            for b in range(yp.shape[0]):
                X = rfft2(yp[b], norm="ortho")  # [C,Hf,Wf]
                Y = rfft2(yt[b], norm="ortho")

                PX = (X.real**2 + X.imag**2).mean(dim=0)  # [Hf,Wf]
                PY = (Y.real**2 + Y.imag**2).mean(dim=0)
                Rxy = (X * torch.conj(Y)).mean(dim=0).real

                def bin_mean(Z):
                    zb = Z.reshape(-1, Hf * Wf)
                    sums = torch.zeros(zb.shape[0], nb, device=Z.device, dtype=Z.dtype)
                    sums.index_add_(1, flat_idx, zb)
                    return (sums / counts).squeeze(0)

                PSDx_list.append(bin_mean(PX[None, ...]))
                PSDy_list.append(bin_mean(PY[None, ...]))
                Rxy_list.append(bin_mean(Rxy[None, ...]))

            PSDx = torch.stack(PSDx_list, dim=0).mean(0).clamp_min(1e-8)  # [nb]
            PSDy = torch.stack(PSDy_list, dim=0).mean(0).clamp_min(1e-8)
            Rxy = torch.stack(Rxy_list, dim=0).mean(0)

            denom = (PSDx.sqrt() * PSDy.sqrt()).clamp_min(1e-8)
            Coh = (Rxy / denom).clamp(-1.0, 1.0)  # [nb]

            def pick_band(target: float) -> float:
                idx = int(target * (nb - 1))
                idx = max(0, min(nb - 1, idx))
                return Coh[idx].item()

            c_mid = pick_band(k_mid)
            c_high = pick_band(k_high)

            results.append(
                {
                    "model": run_name[i],
                    "timestep": t,
                    "coherence_mid": c_mid,
                    "coherence_high": c_high,
                }
            )

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by=["timestep", "model"])
    df.to_csv(f"{save_path}/results/spectral_coherence.csv", index=False)
    return df


def plot_spectral_coherence_bands(df: pd.DataFrame, save_path="runs/verification"):
    """
    Expects columns: model, timestep, coherence_mid, coherence_high
    Produces a single plot; each model has two lines (mid & high).
    """
    if df.empty:
        print("No spectral coherence results to plot.")
        return

    num_timesteps = df["timestep"].max() + 1

    # Melt to long for seaborn
    plot_df = df.melt(
        id_vars=["model", "timestep"],
        value_vars=["coherence_mid", "coherence_high"],
        var_name="band",
        value_name="coherence",
    )

    # nicer band labels
    plot_df["band"] = plot_df["band"].map(
        {
            "coherence_mid": "mid-k",
            "coherence_high": "high-k",
        }
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=plot_df,
        x="timestep",
        y="coherence",
        hue="model",
        style="band",
        markers=True,
        dashes=True,
    )

    plt.ylim(-0.05, 1.05)
    plt.xlabel("Forecast Timestep Index")
    plt.ylabel("Spectral Coherence")
    plt.title("Spectral Coherence (mid-k & high-k) vs Lead Time")
    plt.xticks(range(num_timesteps))
    plt.legend(title="Model / Band", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    filename = f"{save_path}/figures/spectral_coherence_timeseries.png"
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()
