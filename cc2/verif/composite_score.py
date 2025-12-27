import os
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

COMP_COLS = ["S_mae", "S_fss", "S_coh", "S_hk", "S_var", "S_change", "S_ssim"]
COMP_LABELS = {
    "S_mae": "MAE",
    "S_fss": "FSS",
    "S_coh": "Coherence",
    "S_hk": "High-k",
    "S_var": "Variance",
    "S_change": "Change",
    "S_ssim": "SSIM",
}

WEIGHTS = {
    "mae": 0.15,
    "ssim": 0.10,
    "fss": 0.20,
    "coh": 0.15,
    "hk": 0.10,
    "var": 0.15,
    "change": 0.15,
}


def _safe_mean(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else np.nan


def _ideal_one_score(value, tolerance):
    """Score for metrics where ideal value is 1.0"""
    if np.isnan(value):
        return np.nan
    return max(0.0, 1.0 - abs(value - 1.0) / tolerance)


def composite_score(
    run_name: list[str],
    values: dict,
    mae_ref: float | None = None,
    tol_var: float = 0.40,
    tol_hk: float = 0.40,
    tol_stat: float = 0.25,
    lead_weights: dict[int, float] | None = None,
    save_path: str = "runs/verification",
):
    df_mae = values["mae"]
    df_fss = values["fss"]
    df_var = values["variance_ratio"]
    df_hk = values["highk_power_ratio"]
    df_coh = values["spectral_coherence"]
    df_chg = values["change_metrics"]
    df_ssim = values["ssim"]

    # Helper to average over leads with optional weighting
    def avg_leads(group, col):
        if not lead_weights:
            return _safe_mean(group[col])
        g = group.copy()
        g["w"] = g["timestep"].map(lead_weights).fillna(1.0)
        vals = pd.to_numeric(g[col], errors="coerce")
        w = g["w"]
        ok = vals.notna() & w.notna()
        return float((vals[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan

    # Establish mae_ref if not provided
    if mae_ref is None and not df_mae.empty:
        persistence_mae = df_mae[
            df_mae["model"].str.contains("persistence", case=False, na=False)
        ]
        if not persistence_mae.empty:
            mae_ref = float(_safe_mean(persistence_mae["mae"]))
        else:
            # Fallback to median if no persistence
            tmp = df_mae.groupby("model").apply(lambda g: _safe_mean(g["mae"]))
            mae_ref = float(tmp.median()) if len(tmp) else 1.0
    if not mae_ref or mae_ref <= 0:
        mae_ref = 1.0

    # Get all unique models
    models = sorted(
        set(df_mae["model"])
        | set(df_fss.get("model", []))
        | set(df_var["model"])
        | set(df_hk["model"])
        | set(df_coh["model"])
        | set(df_chg["model"])
        | set(df_ssim["model"])
    )

    rows = []

    for m in models:
        # MAE (exponential decay from reference)
        mae_val = avg_leads(df_mae[df_mae["model"] == m], "mae")
        S_mae = (
            float(math.exp(-((mae_val / mae_ref) ** 2)))
            if not np.isnan(mae_val)
            else np.nan
        )

        # FSS (direct use, already 0-1)
        S_fss = (
            avg_leads(df_fss[df_fss["model"] == m], "fss")
            if not df_fss.empty
            else np.nan
        )

        # Variance & High-k (ideal = 1)
        S_var = _ideal_one_score(
            avg_leads(df_var[df_var["model"] == m], "variance_ratio"), tol_var
        )
        S_hk = _ideal_one_score(
            avg_leads(df_hk[df_hk["model"] == m], "highk_power_ratio"), tol_hk
        )

        # Coherence (average of mid & high)
        g = df_coh[df_coh["model"] == m]
        coh_mid = avg_leads(g, "coherence_mid") if not g.empty else np.nan
        coh_high = avg_leads(g, "coherence_high") if not g.empty else np.nan
        S_coh = (
            float(np.clip(0.5 * (coh_mid + coh_high), 0.0, 1.0))
            if not np.isnan(coh_mid) and not np.isnan(coh_high)
            else np.nan
        )

        # Change (weighted combination)
        g = df_chg[df_chg["model"] == m]
        f1 = avg_leads(g, "f1") if not g.empty else np.nan
        tcorr = avg_leads(g, "tendency_corr") if not g.empty else np.nan
        stat = avg_leads(g, "stationarity_ratio") if not g.empty else np.nan
        S_stat = _ideal_one_score(stat, tol_stat)

        parts = [p for p in [f1, tcorr, S_stat] if not np.isnan(p)]
        S_change = (
            0.5 * (f1 or 0) + 0.3 * (tcorr or 0) + 0.2 * (S_stat or 0)
            if parts
            else np.nan
        )

        ssim_val = (
            avg_leads(df_ssim[df_ssim["model"] == m], "ssim")
            if not df_ssim.empty
            else np.nan
        )
        # ssim_max = 0.1  # Focus on 0-0.1 range
        # S_ssim = (
        #    max(0.0, 1.0 - ssim_val / ssim_max) if not np.isnan(ssim_val) else np.nan
        # )
        S_ssim = ssim_val

        # Compute weighted score (ignoring NaNs)
        components = {
            "S_mae": S_mae,
            "S_fss": S_fss,
            "S_coh": S_coh,
            "S_hk": S_hk,
            "S_var": S_var,
            "S_change": S_change,
            "S_ssim": S_ssim,
        }
        key_map = {
            "S_mae": "mae",
            "S_fss": "fss",
            "S_coh": "coh",
            "S_hk": "hk",
            "S_var": "var",
            "S_change": "change",
            "S_ssim": "ssim",
        }

        num, den = 0.0, 0.0
        for k, v in components.items():
            if not np.isnan(v):
                wk = WEIGHTS[key_map[k]]
                num += wk * v
                den += wk

        score = num / den if den > 0 else np.nan
        rows.append({"model": m, "score": score, **components})

    out = pd.DataFrame(rows).sort_values("score", ascending=False)
    out.to_csv(f"{save_path}/results/composite_scores.csv", index=False)
    return out


def plot_composite_bars(
    df: pd.DataFrame, save_path: str = "runs/verification", top_n: int | None = None
):
    """Horizontal bar chart of composite scores"""
    os.makedirs(f"{save_path}/figures", exist_ok=True)

    # Exclude persistence
    df_plot = df[~df["model"].str.contains("persistence", case=False, na=False)].copy()

    df_sorted = df_plot.sort_values("score", ascending=False)
    if top_n:
        df_sorted = df_sorted.head(top_n)

    plt.figure(figsize=(10, max(4, 0.5 * len(df_sorted))))
    sns.barplot(
        data=df_sorted,
        y="model",
        x="score",
        palette="viridis",
        orient="h",
        edgecolor="black",
    )

    # Annotate bars
    for i, score in enumerate(df_sorted["score"]):
        plt.text(score + 0.01, i, f"{score:.3f}", va="center")

    plt.xlabel("Composite Score")
    plt.ylabel("Model")
    plt.title("Composite Score by Model")
    plt.xlim(0, 1)
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{save_path}/figures/composite_bars.png", dpi=200)
    plt.close()


def plot_component_contributions(
    df: pd.DataFrame, save_path: str = "runs/verification", top_n: int | None = None
):
    """Stacked bars showing component contributions with total score at end"""
    os.makedirs(f"{save_path}/figures", exist_ok=True)

    # Exclude persistence
    df_plot = df[~df["model"].str.contains("persistence", case=False, na=False)].copy()

    df_sorted = df_plot.sort_values("score", ascending=False)
    if top_n:
        df_sorted = df_sorted.head(top_n)

    # Compute weighted contributions
    contrib_data = []
    for _, row in df_sorted.iterrows():
        avail = {k: WEIGHTS[k.replace("S_", "")] for k in COMP_COLS if pd.notna(row[k])}
        wsum = sum(avail.values()) or 1.0

        parts = {
            k: (WEIGHTS[k.replace("S_", "")] / wsum) * row[k]
            for k in COMP_COLS
            if pd.notna(row[k])
        }
        # Rescale to match actual score
        total = sum(parts.values())
        scale = row["score"] / total if total > 0 else 0
        parts = {k: v * scale for k, v in parts.items()}
        contrib_data.append({"model": row["model"], "score": row["score"], **parts})

    contrib_df = pd.DataFrame(contrib_data)
    models = contrib_df["model"].tolist()

    # Plot stacked bars
    plt.figure(figsize=(10, max(4, 0.6 * len(models))))
    bottom = np.zeros(len(models))
    palette = sns.color_palette("Set2", n_colors=len(COMP_COLS))

    for idx, col in enumerate(COMP_COLS):
        vals = contrib_df[col].fillna(0).values
        plt.barh(
            models,
            vals,
            left=bottom,
            color=palette[idx],
            edgecolor="black",
            label=COMP_LABELS[col],
        )
        bottom += vals

    # Add total score at end of each bar
    for i, (model, score) in enumerate(zip(models, contrib_df["score"])):
        plt.text(score + 0.01, i, f"{score:.3f}", va="center", fontweight="bold")

    plt.xlabel("Contribution to Composite Score")
    plt.ylabel("Model")
    plt.title("Component Contributions to Composite Score")
    plt.xlim(0, max(1.0, contrib_df["score"].max() * 1.1))
    plt.legend(title="Component", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{save_path}/figures/composite_contributions.png", dpi=200)
    plt.close()
