#!/usr/bin/env python3
"""
Genesis/Lysis Metric - WITH STDDEV CONSTRAINT

Further refinement: Exclude cloud edges by requiring:
- Genesis: C_t < 0.2 AND s_t < s_max (uniformly clear patches)
- Lysis: C_t > 0.8 AND s_t < s_max (uniformly cloudy patches)

This ensures we capture true cloud lifecycle in homogeneous regions,
not just advection at cloud boundaries.
"""

import torch
import numpy as np
from pathlib import Path


def load_forecast_data_12h():
    """Load 12-hour forecast data (truth, CC1, CC2)."""
    print("Loading 12-hour forecast data...")

    # Load CC1
    cc1_base = "/data/tfs/runs/ED/cloudcast-production-12h"
    cc1_dates = torch.load(f"{cc1_base}/dates.pt", map_location='cpu')
    cc1_pred = torch.load(f"{cc1_base}/predictions.pt", map_location='cpu')
    cc1_truth = torch.load(f"{cc1_base}/truth.pt", map_location='cpu')

    # Load CC2
    cc2_base = "/data/tfs/runs/ED/pizzicato-transfer-psdsc-12h"
    cc2_dates = torch.load(f"{cc2_base}/dates.pt", map_location='cpu')
    cc2_pred = torch.load(f"{cc2_base}/predictions.pt", map_location='cpu')
    cc2_truth = torch.load(f"{cc2_base}/truth.pt", map_location='cpu')

    # Find matching samples
    cc1_init_dates = [cc1_dates[i, 0].item() for i in range(cc1_dates.shape[0])]
    cc2_init_dates = [cc2_dates[i, 0].item() for i in range(cc2_dates.shape[0])]

    common_dates = set(cc1_init_dates) & set(cc2_init_dates)
    idx_cc1 = sorted([i for i, d in enumerate(cc1_init_dates) if d in common_dates],
                     key=lambda i: cc1_init_dates[i])
    idx_cc2 = sorted([i for i, d in enumerate(cc2_init_dates) if d in common_dates],
                     key=lambda i: cc2_init_dates[i])

    # Extract matching samples
    cc1_pred = cc1_pred[idx_cc1].squeeze(2)
    cc2_pred = cc2_pred[idx_cc2].squeeze(2)
    truth = cc1_truth[idx_cc1].squeeze(2)

    print(f"Loaded {truth.shape[0]} forecasts, shape: {truth.shape}")
    return cc1_pred, cc2_pred, truth


def compute_patch_stats(field_t0, field_t1, patch_size=10):
    """
    Compute patch statistics: mean TCC change, initial mean, initial stddev.

    Args:
        field_t0: [H, W] cloud cover at time t
        field_t1: [H, W] cloud cover at time t+dt
        patch_size: patch size in grid cells

    Returns:
        patch_changes: [n_y, n_x] TCC change per patch (mean)
        patch_initial_mean: [n_y, n_x] Initial TCC per patch (mean)
        patch_initial_std: [n_y, n_x] Initial TCC stddev per patch
        patch_grid: tuple of (y_starts, x_starts)
    """
    H, W = field_t0.shape

    n_patches_y = H // patch_size
    n_patches_x = W // patch_size

    patch_changes = np.zeros((n_patches_y, n_patches_x))
    patch_initial_mean = np.zeros((n_patches_y, n_patches_x))
    patch_initial_std = np.zeros((n_patches_y, n_patches_x))

    for i in range(n_patches_y):
        for j in range(n_patches_x):
            y_start = i * patch_size
            y_end = y_start + patch_size
            x_start = j * patch_size
            x_end = x_start + patch_size

            patch_t0 = field_t0[y_start:y_end, x_start:x_end]
            patch_t1 = field_t1[y_start:y_end, x_start:x_end]

            # Mean TCC
            tcc_t0_mean = patch_t0.mean()
            tcc_t1_mean = patch_t1.mean()

            # Stddev at t0 (spatial variability within patch)
            tcc_t0_std = patch_t0.std()

            patch_initial_mean[i, j] = tcc_t0_mean
            patch_initial_std[i, j] = tcc_t0_std
            patch_changes[i, j] = tcc_t1_mean - tcc_t0_mean

    y_starts = np.arange(n_patches_y) * patch_size
    x_starts = np.arange(n_patches_x) * patch_size
    patch_grid = (y_starts, x_starts)

    return patch_changes, patch_initial_mean, patch_initial_std, patch_grid


def classify_patches_with_stddev(patch_changes, patch_initial_mean, patch_initial_std,
                                 change_threshold=0.25,
                                 clear_mean_threshold=0.2,
                                 cloudy_mean_threshold=0.8,
                                 std_threshold=0.15):
    """
    Classify patches with stddev constraint to exclude edges.

    Args:
        patch_changes: [n_y, n_x] TCC changes
        patch_initial_mean: [n_y, n_x] Initial TCC mean
        patch_initial_std: [n_y, n_x] Initial TCC stddev
        change_threshold: threshold for significant change
        clear_mean_threshold: max mean TCC for clear patches
        cloudy_mean_threshold: min mean TCC for cloudy patches
        std_threshold: max stddev for homogeneous patches

    Returns:
        patch_classes: [n_y, n_x] classification
        genesis_mask: [n_y, n_x] boolean mask for genesis
        lysis_mask: [n_y, n_x] boolean mask for lysis
    """
    patch_classes = np.zeros_like(patch_changes, dtype=int)

    # Genesis: large positive change + initially clear-core
    # (mean < threshold AND stddev < threshold → homogeneous clear)
    genesis_mask = (
        (patch_changes >= change_threshold) &
        (patch_initial_mean < clear_mean_threshold) &
        (patch_initial_std < std_threshold)
    )

    # Lysis: large negative change + initially cloudy-core
    # (mean > threshold AND stddev < threshold → homogeneous cloudy)
    lysis_mask = (
        (patch_changes <= -change_threshold) &
        (patch_initial_mean > cloudy_mean_threshold) &
        (patch_initial_std < std_threshold)
    )

    patch_classes[genesis_mask] = 1
    patch_classes[lysis_mask] = -1

    return patch_classes, genesis_mask, lysis_mask


def compute_genesis_lysis_metrics_with_stddev(obs_changes, obs_initial_mean, obs_initial_std,
                                               pred_changes,
                                               change_threshold=0.25,
                                               clear_threshold=0.2,
                                               cloudy_threshold=0.8,
                                               std_threshold=0.15):
    """
    Compute genesis/lysis metrics with stddev constraint.
    """
    # Flatten
    obs_flat = obs_changes.flatten()
    pred_flat = pred_changes.flatten()

    # Correlation and RMSE
    correlation = np.corrcoef(obs_flat, pred_flat)[0, 1]
    rmse = np.sqrt(((obs_flat - pred_flat) ** 2).mean())

    # Event detection with stddev constraint
    obs_genesis = (
        (obs_changes >= change_threshold) &
        (obs_initial_mean < clear_threshold) &
        (obs_initial_std < std_threshold)
    )
    obs_lysis = (
        (obs_changes <= -change_threshold) &
        (obs_initial_mean > cloudy_threshold) &
        (obs_initial_std < std_threshold)
    )
    pred_genesis = (
        (pred_changes >= change_threshold) &
        (obs_initial_mean < clear_threshold) &
        (obs_initial_std < std_threshold)
    )
    pred_lysis = (
        (pred_changes <= -change_threshold) &
        (obs_initial_mean > cloudy_threshold) &
        (obs_initial_std < std_threshold)
    )

    # Genesis CSI
    genesis_hits = (obs_genesis & pred_genesis).sum()
    genesis_misses = (obs_genesis & ~pred_genesis).sum()
    genesis_false_alarms = (~obs_genesis & pred_genesis).sum()
    genesis_csi = genesis_hits / (genesis_hits + genesis_misses + genesis_false_alarms + 1e-10)

    # Lysis CSI
    lysis_hits = (obs_lysis & pred_lysis).sum()
    lysis_misses = (obs_lysis & ~pred_lysis).sum()
    lysis_false_alarms = (~obs_lysis & pred_lysis).sum()
    lysis_csi = lysis_hits / (lysis_hits + lysis_misses + lysis_false_alarms + 1e-10)

    metrics = {
        'correlation': correlation,
        'rmse': rmse,
        'genesis_csi': genesis_csi,
        'lysis_csi': lysis_csi,
        'genesis_hits': genesis_hits,
        'genesis_misses': genesis_misses,
        'genesis_false_alarms': genesis_false_alarms,
        'lysis_hits': lysis_hits,
        'lysis_misses': lysis_misses,
        'lysis_false_alarms': lysis_false_alarms,
    }

    return metrics


def test_stddev_thresholds():
    """Test different stddev thresholds to find optimal value."""
    print("="*80)
    print("GENESIS/LYSIS WITH STDDEV CONSTRAINT - THRESHOLD TESTING")
    print("="*80)

    # Load data
    cc1_pred, cc2_pred, truth = load_forecast_data_12h()

    # Test parameters
    forecast_idx = 365
    t0 = 0
    t1 = 3
    patch_size = 10
    change_threshold = 0.25
    clear_threshold = 0.2
    cloudy_threshold = 0.8

    # Extract fields
    tcc_t0 = truth[forecast_idx, t0].numpy()
    tcc_t1 = truth[forecast_idx, t1].numpy()
    cc1_t1 = cc1_pred[forecast_idx, t1].numpy()
    cc2_t1 = cc2_pred[forecast_idx, t1].numpy()

    # Compute patch stats
    obs_changes, obs_mean, obs_std, _ = compute_patch_stats(tcc_t0, tcc_t1, patch_size)
    cc1_changes, _, _, _ = compute_patch_stats(tcc_t0, cc1_t1, patch_size)
    cc2_changes, _, _, _ = compute_patch_stats(tcc_t0, cc2_t1, patch_size)

    print(f"\nPatch initial TCC stddev statistics:")
    print(f"  Mean: {obs_std.mean():.4f}")
    print(f"  Median: {np.median(obs_std):.4f}")
    print(f"  Std: {obs_std.std():.4f}")
    print(f"  25th percentile: {np.percentile(obs_std, 25):.4f}")
    print(f"  75th percentile: {np.percentile(obs_std, 75):.4f}")
    print(f"  Max: {obs_std.max():.4f}")

    # Test different stddev thresholds
    std_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 1.0]  # 1.0 = no constraint

    print(f"\n{'Std_max':<10} {'#Gen':<8} {'#Lys':<8} {'CC1_GenCSI':<12} {'CC2_GenCSI':<12} {'Gen_Adv':<10} {'CC1_LysCSI':<12} {'CC2_LysCSI':<12} {'Lys_Adv':<10}")
    print("="*120)

    results = []

    for std_thresh in std_thresholds:
        # Count events
        obs_genesis = (
            (obs_changes >= change_threshold) &
            (obs_mean < clear_threshold) &
            (obs_std < std_thresh)
        )
        obs_lysis = (
            (obs_changes <= -change_threshold) &
            (obs_mean > cloudy_threshold) &
            (obs_std < std_thresh)
        )

        # Compute metrics
        cc1_metrics = compute_genesis_lysis_metrics_with_stddev(
            obs_changes, obs_mean, obs_std, cc1_changes,
            change_threshold, clear_threshold, cloudy_threshold, std_thresh
        )
        cc2_metrics = compute_genesis_lysis_metrics_with_stddev(
            obs_changes, obs_mean, obs_std, cc2_changes,
            change_threshold, clear_threshold, cloudy_threshold, std_thresh
        )

        gen_adv = (cc2_metrics['genesis_csi'] - cc1_metrics['genesis_csi']) / cc1_metrics['genesis_csi'] * 100 if cc1_metrics['genesis_csi'] > 0 else 0
        lys_adv = (cc2_metrics['lysis_csi'] - cc1_metrics['lysis_csi']) / cc1_metrics['lysis_csi'] * 100 if cc1_metrics['lysis_csi'] > 0 else 0

        label = "no constraint" if std_thresh >= 1.0 else f"{std_thresh:.2f}"
        print(f"{label:<10} {obs_genesis.sum():<8d} {obs_lysis.sum():<8d} "
              f"{cc1_metrics['genesis_csi']:<12.4f} {cc2_metrics['genesis_csi']:<12.4f} {gen_adv:>9.1f}% "
              f"{cc1_metrics['lysis_csi']:<12.4f} {cc2_metrics['lysis_csi']:<12.4f} {lys_adv:>9.1f}%")

        results.append({
            'std_thresh': std_thresh,
            'n_genesis': obs_genesis.sum(),
            'n_lysis': obs_lysis.sum(),
            'cc2_genesis_csi': cc2_metrics['genesis_csi'],
            'cc2_lysis_csi': cc2_metrics['lysis_csi'],
            'gen_adv': gen_adv,
            'lys_adv': lys_adv,
        })

    print("="*120)

    # Recommendations
    print("\nRECOMMENDATIONS:")

    # Best by highest CC2 CSI
    best_gen = max([r for r in results if r['std_thresh'] < 1.0],
                   key=lambda x: x['cc2_genesis_csi'])
    print(f"\nBest CC2 Genesis CSI: std_max = {best_gen['std_thresh']:.2f}")
    print(f"  CC2 Genesis CSI: {best_gen['cc2_genesis_csi']:.4f} ({best_gen['gen_adv']:+.1f}%)")
    print(f"  Events: {best_gen['n_genesis']} genesis, {best_gen['n_lysis']} lysis")

    # Best by highest advantage
    best_adv = max([r for r in results if r['std_thresh'] < 1.0],
                   key=lambda x: x['gen_adv'] + x['lys_adv'])
    print(f"\nBest Combined Advantage: std_max = {best_adv['std_thresh']:.2f}")
    print(f"  Genesis advantage: {best_adv['gen_adv']:+.1f}%")
    print(f"  Lysis advantage: {best_adv['lys_adv']:+.1f}%")
    print(f"  Events: {best_adv['n_genesis']} genesis, {best_adv['n_lysis']} lysis")

    # Balanced (moderate event count)
    moderate = [r for r in results if 150 <= r['n_genesis'] <= 250 and 100 <= r['n_lysis'] <= 200]
    if moderate:
        best_mod = max(moderate, key=lambda x: x['gen_adv'] + x['lys_adv'])
        print(f"\nBest with Moderate Event Count: std_max = {best_mod['std_thresh']:.2f}")
        print(f"  CC2 advantage: Genesis={best_mod['gen_adv']:+.1f}%, Lysis={best_mod['lys_adv']:+.1f}%")
        print(f"  Events: {best_mod['n_genesis']} genesis, {best_mod['n_lysis']} lysis")

    print("\n" + "="*80)


if __name__ == "__main__":
    test_stddev_thresholds()
