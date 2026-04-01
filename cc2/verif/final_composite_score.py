#!/usr/bin/env python3
"""
Final model-selection composite for cloud cover forecasting.

This composite matches the actual evaluation priorities:

1. Pixel loss      -> MAE-based skill
2. Spatial score   -> FSS
3. Genesis/Lysis   -> pixel-count event CSI

The genesis/lysis block is intentionally the heaviest component because that is
the most important downstream scientific criterion.

Supported inputs
----------------

MAE CSV:
  columns: ["model", "timestep", "mae"]

FSS CSV:
  columns: ["model", "timestep", "fss"] and optionally category/scale columns.

Genesis/Lysis CSV (long format):
  columns: ["model", "genesis_csi", "lysis_csi"] and optionally
  ["time_window", "timestep", "correlation", "rmse", "n_genesis", "n_lysis"].

Genesis/Lysis CSV (existing pairwise format):
  columns like:
    cc1_genesis_csi, cc1_lysis_csi, cc2_genesis_csi, cc2_lysis_csi, ...
  and optional event count columns.
  Use --genesis-pair prefixes and model names to convert these into long format.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _safe_mean(x: pd.Series) -> float:
    v = _to_numeric(x).dropna()
    return float(v.mean()) if len(v) else float("nan")


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = _to_numeric(values)
    w = _to_numeric(weights)
    ok = v.notna() & w.notna()
    if not ok.any():
        return float("nan")
    num = float((v[ok] * w[ok]).sum())
    den = float(w[ok].sum())
    return num / den if den > 0 else float("nan")


def _clip01(x: float) -> float:
    if np.isnan(x):
        return float("nan")
    return float(np.clip(x, 0.0, 1.0))


def _avg_with_optional_weights(
    df: pd.DataFrame, value_col: str, weight_col: Optional[str] = None
) -> float:
    if df.empty or value_col not in df.columns:
        return float("nan")
    if weight_col is None or weight_col not in df.columns:
        return _safe_mean(df[value_col])
    return _weighted_mean(df[value_col], df[weight_col])


def _one_sided_log_gauss(r: float, sigma_log: float) -> float:
    if np.isnan(r) or r <= 0:
        return float("nan")
    if r <= 1.0:
        return 1.0
    z = math.log(r) / max(1e-8, sigma_log)
    return float(math.exp(-(z * z)))


@dataclass
class FinalCompositeConfig:
    pixel_weight: float = 0.20
    spatial_weight: float = 0.45
    genesis_weight: float = 0.35
    mae_sigma_ratio: float = 0.35
    genesis_event_weighted: bool = True
    fss_category: Optional[str] = None


def load_metric_csv(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def normalize_genesis_pairwise(
    df: pd.DataFrame,
    prefixes: Sequence[str],
    model_names: Sequence[str],
) -> pd.DataFrame:
    rows = []
    base_cols = [c for c in df.columns if not any(c.startswith(f"{p}_") for p in prefixes)]

    for prefix, model_name in zip(prefixes, model_names):
        row = pd.DataFrame()
        for c in base_cols:
            row[c] = df[c]

        rename_map = {}
        for c in df.columns:
            pre = f"{prefix}_"
            if c.startswith(pre):
                rename_map[c] = c[len(pre):]

        if not rename_map:
            raise ValueError(f"No columns found for prefix '{prefix}' in genesis CSV")

        sub = df[list(rename_map.keys())].rename(columns=rename_map)
        row = pd.concat([row, sub], axis=1)
        row["model"] = model_name
        rows.append(row)

    return pd.concat(rows, ignore_index=True)


def compute_final_composite(
    mae_df: pd.DataFrame,
    fss_df: pd.DataFrame,
    genesis_df: pd.DataFrame,
    cfg: Optional[FinalCompositeConfig] = None,
    lead_weights: Optional[Dict[int, float]] = None,
) -> pd.DataFrame:
    if cfg is None:
        cfg = FinalCompositeConfig()

    models_per_df = []
    for df in (mae_df, fss_df, genesis_df):
        if df.empty or "model" not in df.columns:
            models_per_df.append(set())
        else:
            models_per_df.append(set(df["model"].dropna().unique().tolist()))

    if any(not models for models in models_per_df):
        raise ValueError("MAE, FSS, and genesis/lysis tables must all contain model data")

    models = sorted(set.intersection(*models_per_df))

    if not models:
        raise ValueError("No models found across the provided metric tables")

    if (
        mae_df.empty
        or "model" not in mae_df.columns
        or "mae" not in mae_df.columns
    ):
        raise ValueError("MAE table must contain columns ['model', 'mae']")

    mae_ref = float(
        _safe_mean(
            mae_df[
                mae_df["model"].astype(str).str.contains(
                    "persistence", case=False, na=False
                )
            ]["mae"]
        )
    )
    if not np.isfinite(mae_ref):
        per_model_mae = mae_df.groupby("model", dropna=True)["mae"].apply(_safe_mean)
        mae_ref = float(per_model_mae.median()) if len(per_model_mae) else 1.0
    if not np.isfinite(mae_ref) or mae_ref <= 0:
        mae_ref = 1.0

    rows = []

    for model in models:
        m_mae = mae_df[mae_df["model"] == model].copy()
        if not m_mae.empty and lead_weights and "timestep" in m_mae.columns:
            m_mae["w"] = m_mae["timestep"].map(lead_weights).fillna(1.0)
            mae_value = _weighted_mean(m_mae["mae"], m_mae["w"])
        else:
            mae_value = _safe_mean(m_mae["mae"])

        # Lower MAE is better. Matching or beating the reference gets full score;
        # worse-than-reference decays smoothly on a log scale.
        mae_ratio = mae_value / mae_ref if np.isfinite(mae_value) and mae_ref > 0 else np.nan
        s_pixel = _one_sided_log_gauss(mae_ratio, cfg.mae_sigma_ratio)

        m_fss = fss_df[fss_df["model"] == model].copy()
        if cfg.fss_category and "category" in m_fss.columns:
            m_fss = m_fss[m_fss["category"] == cfg.fss_category]
        if not m_fss.empty and lead_weights and "timestep" in m_fss.columns:
            m_fss["w"] = m_fss["timestep"].map(lead_weights).fillna(1.0)
            fss_value = _weighted_mean(m_fss["fss"], m_fss["w"])
        else:
            fss_value = _safe_mean(m_fss["fss"]) if "fss" in m_fss.columns else float("nan")
        s_spatial = _clip01(fss_value)

        m_gen = genesis_df[genesis_df["model"] == model].copy()
        event_weight_col = None
        if cfg.genesis_event_weighted and "n_genesis" in m_gen.columns and "n_lysis" in m_gen.columns:
            m_gen["event_weight"] = _to_numeric(m_gen["n_genesis"]).fillna(0.0) + _to_numeric(m_gen["n_lysis"]).fillna(0.0)
            if float(m_gen["event_weight"].sum()) > 0:
                event_weight_col = "event_weight"

        genesis_csi = _avg_with_optional_weights(m_gen, "genesis_csi", event_weight_col)
        lysis_csi = _avg_with_optional_weights(m_gen, "lysis_csi", event_weight_col)
        s_genesis = _safe_mean(pd.Series([genesis_csi, lysis_csi]))

        component_values = {
            "pixel": s_pixel,
            "spatial": s_spatial,
            "genesis": s_genesis,
        }
        component_weights = {
            "pixel": cfg.pixel_weight,
            "spatial": cfg.spatial_weight,
            "genesis": cfg.genesis_weight,
        }

        available = {
            k: w
            for k, w in component_weights.items()
            if np.isfinite(component_values[k])
        }
        if not available:
            composite = float("nan")
        else:
            weight_sum = float(sum(available.values()))
            composite = sum(
                component_values[k] * component_weights[k] for k in available
            ) / weight_sum

        pixel_contrib = s_pixel * cfg.pixel_weight if np.isfinite(s_pixel) else float("nan")
        spatial_contrib = (
            s_spatial * cfg.spatial_weight if np.isfinite(s_spatial) else float("nan")
        )
        genesis_contrib = (
            s_genesis * cfg.genesis_weight if np.isfinite(s_genesis) else float("nan")
        )

        rows.append(
            {
                "model": model,
                "composite": composite,
                "S_pixel": s_pixel,
                "S_spatial": s_spatial,
                "S_genesis": s_genesis,
                "mae": mae_value,
                "mae_ref": mae_ref,
                "mae_ratio_to_ref": mae_ratio,
                "fss": fss_value,
                "genesis_csi": genesis_csi,
                "lysis_csi": lysis_csi,
                "C_pixel": pixel_contrib,
                "C_spatial": spatial_contrib,
                "C_genesis": genesis_contrib,
                "used_pixel_weight": cfg.pixel_weight if np.isfinite(s_pixel) else 0.0,
                "used_spatial_weight": cfg.spatial_weight if np.isfinite(s_spatial) else 0.0,
                "used_genesis_weight": cfg.genesis_weight if np.isfinite(s_genesis) else 0.0,
            }
        )

    out = pd.DataFrame(rows).sort_values("composite", ascending=False).reset_index(drop=True)
    return out


def plot_final_composite(df: pd.DataFrame, save_path: str) -> None:
    os.makedirs(save_path, exist_ok=True)
    plot_df = df.sort_values("composite", ascending=True)

    fig, ax = plt.subplots(figsize=(11, max(4, 0.55 * len(plot_df))))
    y = np.arange(len(plot_df))

    c_genesis = plot_df["C_genesis"].fillna(0.0)
    c_spatial = plot_df["C_spatial"].fillna(0.0)
    c_pixel = plot_df["C_pixel"].fillna(0.0)

    ax.barh(y, c_genesis, color="#0B6E4F", alpha=0.95, label="Genesis/Lysis")
    ax.barh(
        y,
        c_spatial,
        left=c_genesis,
        color="#D17A22",
        alpha=0.95,
        label="Spatial",
    )
    ax.barh(
        y,
        c_pixel,
        left=(c_genesis + c_spatial),
        color="#B33C2E",
        alpha=0.95,
        label="Pixel",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["model"])
    ax.set_xlabel("Weighted contribution")
    ax.set_title("Final Composite: Pixel + Spatial + Genesis/Lysis")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="lower right")

    x_pad = max(float(plot_df["composite"].max()) * 0.015, 0.005)
    for yi, score in zip(y, plot_df["composite"]):
        ax.text(score + x_pad, yi, f"{score:.3f}", va="center", ha="left", fontsize=10)

    ax.set_xlim(0, float(plot_df["composite"].max()) + 6 * x_pad)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, "final_composite_score.png"), dpi=200)
    plt.close(fig)


def parse_lead_weights(spec: Optional[str]) -> Optional[Dict[int, float]]:
    if not spec:
        return None
    out = {}
    for item in spec.split(","):
        if not item.strip():
            continue
        key, value = item.split(":")
        out[int(key)] = float(value)
    return out


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mae_csv", required=True)
    parser.add_argument("--fss_csv", required=True)
    parser.add_argument("--genesis_csv", required=True)
    parser.add_argument("--save_path", default="runs/verification")
    parser.add_argument("--out_csv", default="results/final_composite_scores.csv")
    parser.add_argument("--lead_weights", default=None, help="e.g. 1:1,2:1,3:0.8,4:0.6")
    parser.add_argument("--fss_category", default=None)
    parser.add_argument("--pixel_weight", type=float, default=0.20)
    parser.add_argument("--spatial_weight", type=float, default=0.45)
    parser.add_argument("--genesis_weight", type=float, default=0.35)
    parser.add_argument("--mae_sigma_ratio", type=float, default=0.35)
    parser.add_argument(
        "--genesis_pair_prefixes",
        nargs="+",
        default=None,
        help="Prefixes in pairwise genesis CSV, e.g. cc1 cc2",
    )
    parser.add_argument(
        "--genesis_pair_model_names",
        nargs="+",
        default=None,
        help="Model names matching pair prefixes, e.g. cloudcast-production gray-carrier",
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()
    lead_weights = parse_lead_weights(args.lead_weights)

    mae_df = load_metric_csv(args.mae_csv)
    fss_df = load_metric_csv(args.fss_csv)
    genesis_df = load_metric_csv(args.genesis_csv)

    if args.genesis_pair_prefixes or args.genesis_pair_model_names:
        if not (args.genesis_pair_prefixes and args.genesis_pair_model_names):
            raise ValueError(
                "Provide both --genesis_pair_prefixes and --genesis_pair_model_names"
            )
        if len(args.genesis_pair_prefixes) != len(args.genesis_pair_model_names):
            raise ValueError("Pair prefixes and model names must have equal length")
        genesis_df = normalize_genesis_pairwise(
            genesis_df, args.genesis_pair_prefixes, args.genesis_pair_model_names
        )

    cfg = FinalCompositeConfig(
        pixel_weight=args.pixel_weight,
        spatial_weight=args.spatial_weight,
        genesis_weight=args.genesis_weight,
        mae_sigma_ratio=args.mae_sigma_ratio,
        fss_category=args.fss_category,
    )

    out = compute_final_composite(mae_df, fss_df, genesis_df, cfg, lead_weights)

    out_dir = os.path.join(args.save_path, os.path.dirname(args.out_csv))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(args.save_path, args.out_csv)
    out.to_csv(out_path, index=False)
    plot_final_composite(out, os.path.join(args.save_path, "figures"))

    print(out.to_string(index=False))
    print(f"\nSaved CSV: {out_path}")
    print(f"Saved figure: {os.path.join(args.save_path, 'figures', 'final_composite_score.png')}")


if __name__ == "__main__":
    main()
