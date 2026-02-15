#!/usr/bin/env python3
"""
Genesis/Lysis Metric - PIXEL COUNTING APPROACH

Instead of stddev, count pixels that are clearly "clear" or "cloudy":
- For a 10×10 patch (100 pixels):
  - p^{clr}_t = fraction of pixels with C_t < 0.2
  - p^{cld}_t = fraction of pixels with C_t > 0.8

- Clear core: p^{clr}_t > 0.85 (≥85 out of 100 pixels clearly clear)
- Cloudy core: p^{cld}_t > 0.85 (≥85 out of 100 pixels clearly cloudy)

This is more interpretable than stddev and directly measures homogeneity.
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


def compute_patch_stats_with_pixel_counts(field_t0, field_t1, patch_size=10,
                                          clear_threshold=0.2, cloudy_threshold=0.8):
    """
    Compute patch statistics with pixel counting.

    Args:
        field_t0: [H, W] cloud cover at time t
        field_t1: [H, W] cloud cover at time t+dt
        patch_size: patch size in grid cells
        clear_threshold: threshold for "clearly clear" pixels
        cloudy_threshold: threshold for "clearly cloudy" pixels

    Returns:
        patch_changes: [n_y, n_x] TCC change per patch (mean)
        patch_initial_mean: [n_y, n_x] Initial TCC per patch (mean)
        patch_clear_fraction: [n_y, n_x] Fraction of clearly clear pixels
        patch_cloudy_fraction: [n_y, n_x] Fraction of clearly cloudy pixels
        patch_grid: tuple of (y_starts, x_starts)
    """
    H, W = field_t0.shape

    n_patches_y = H // patch_size
    n_patches_x = W // patch_size

    patch_changes = np.zeros((n_patches_y, n_patches_x))
    patch_initial_mean = np.zeros((n_patches_y, n_patches_x))
    patch_clear_fraction = np.zeros((n_patches_y, n_patches_x))
    patch_cloudy_fraction = np.zeros((n_patches_y, n_patches_x))

    pixels_per_patch = patch_size * patch_size

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

            # Count clearly clear and clearly cloudy pixels at t0
            n_clear = (patch_t0 < clear_threshold).sum()
            n_cloudy = (patch_t0 > cloudy_threshold).sum()

            patch_initial_mean[i, j] = tcc_t0_mean
            patch_clear_fraction[i, j] = n_clear / pixels_per_patch
            patch_cloudy_fraction[i, j] = n_cloudy / pixels_per_patch
            patch_changes[i, j] = tcc_t1_mean - tcc_t0_mean

    y_starts = np.arange(n_patches_y) * patch_size
    x_starts = np.arange(n_patches_x) * patch_size
    patch_grid = (y_starts, x_starts)

    return patch_changes, patch_initial_mean, patch_clear_fraction, patch_cloudy_fraction, patch_grid


def classify_patches_pixel_count(patch_changes, patch_clear_fraction, patch_cloudy_fraction,
                                  change_threshold=0.25,
                                  clear_core_threshold=0.85,
                                  cloudy_core_threshold=0.85):
    """
    Classify patches using pixel counting approach.

    Args:
        patch_changes: [n_y, n_x] TCC changes
        patch_clear_fraction: [n_y, n_x] Fraction of clearly clear pixels
        patch_cloudy_fraction: [n_y, n_x] Fraction of clearly cloudy pixels
        change_threshold: threshold for significant change
        clear_core_threshold: min fraction of clear pixels for clear-core
        cloudy_core_threshold: min fraction of cloudy pixels for cloudy-core

    Returns:
        patch_classes: [n_y, n_x] classification
        genesis_mask: [n_y, n_x] boolean mask for genesis
        lysis_mask: [n_y, n_x] boolean mask for lysis
    """
    patch_classes = np.zeros_like(patch_changes, dtype=int)

    # Genesis: large positive change + clear-core
    # (≥85% of pixels are clearly clear, i.e., < 0.2)
    genesis_mask = (
        (patch_changes >= change_threshold) &
        (patch_clear_fraction > clear_core_threshold)
    )

    # Lysis: large negative change + cloudy-core
    # (≥85% of pixels are clearly cloudy, i.e., > 0.8)
    lysis_mask = (
        (patch_changes <= -change_threshold) &
        (patch_cloudy_fraction > cloudy_core_threshold)
    )

    patch_classes[genesis_mask] = 1
    patch_classes[lysis_mask] = -1

    return patch_classes, genesis_mask, lysis_mask


def compute_genesis_lysis_metrics_pixel_count(obs_changes, obs_clear_frac, obs_cloudy_frac,
                                               pred_changes,
                                               change_threshold=0.25,
                                               clear_core_threshold=0.85,
                                               cloudy_core_threshold=0.85):
    """
    Compute genesis/lysis metrics with pixel counting approach.
    """
    # Flatten
    obs_flat = obs_changes.flatten()
    pred_flat = pred_changes.flatten()

    # Correlation and RMSE
    correlation = np.corrcoef(obs_flat, pred_flat)[0, 1]
    rmse = np.sqrt(((obs_flat - pred_flat) ** 2).mean())

    # Event detection with pixel counting
    obs_genesis = (
        (obs_changes >= change_threshold) &
        (obs_clear_frac > clear_core_threshold)
    )
    obs_lysis = (
        (obs_changes <= -change_threshold) &
        (obs_cloudy_frac > cloudy_core_threshold)
    )
    pred_genesis = (
        (pred_changes >= change_threshold) &
        (obs_clear_frac > clear_core_threshold)
    )
    pred_lysis = (
        (pred_changes <= -change_threshold) &
        (obs_cloudy_frac > cloudy_core_threshold)
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


def test_pixel_count_thresholds():
    """Test different pixel count thresholds to find optimal value."""
    print("="*80)
    print("GENESIS/LYSIS WITH PIXEL COUNTING - THRESHOLD TESTING")
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

    # Compute patch stats with pixel counts
    obs_changes, obs_mean, obs_clear_frac, obs_cloudy_frac, _ = compute_patch_stats_with_pixel_counts(
        tcc_t0, tcc_t1, patch_size, clear_threshold, cloudy_threshold
    )
    cc1_changes, _, _, _, _ = compute_patch_stats_with_pixel_counts(
        tcc_t0, cc1_t1, patch_size, clear_threshold, cloudy_threshold
    )
    cc2_changes, _, _, _, _ = compute_patch_stats_with_pixel_counts(
        tcc_t0, cc2_t1, patch_size, clear_threshold, cloudy_threshold
    )

    print(f"\nPatch clear/cloudy fraction statistics:")
    print(f"  Clear fraction - Mean: {obs_clear_frac.mean():.4f}, Median: {np.median(obs_clear_frac):.4f}")
    print(f"  Cloudy fraction - Mean: {obs_cloudy_frac.mean():.4f}, Median: {np.median(obs_cloudy_frac):.4f}")
    print(f"\nPixels that are clearly clear (< 0.2): {(obs_clear_frac > 0.0).sum()} patches")
    print(f"Pixels that are clearly cloudy (> 0.8): {(obs_cloudy_frac > 0.0).sum()} patches")

    # Test different thresholds
    core_thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]  # 1.0 = 100% homogeneous (very strict)

    print(f"\n{'Core_thresh':<12} {'#Gen':<8} {'#Lys':<8} {'CC1_GenCSI':<12} {'CC2_GenCSI':<12} {'Gen_Adv':<10} {'CC1_LysCSI':<12} {'CC2_LysCSI':<12} {'Lys_Adv':<10}")
    print("="*120)

    results = []

    for core_thresh in core_thresholds:
        # Count events
        obs_genesis = (
            (obs_changes >= change_threshold) &
            (obs_clear_frac > core_thresh)
        )
        obs_lysis = (
            (obs_changes <= -change_threshold) &
            (obs_cloudy_frac > core_thresh)
        )

        # Compute metrics
        cc1_metrics = compute_genesis_lysis_metrics_pixel_count(
            obs_changes, obs_clear_frac, obs_cloudy_frac, cc1_changes,
            change_threshold, core_thresh, core_thresh
        )
        cc2_metrics = compute_genesis_lysis_metrics_pixel_count(
            obs_changes, obs_clear_frac, obs_cloudy_frac, cc2_changes,
            change_threshold, core_thresh, core_thresh
        )

        gen_adv = (cc2_metrics['genesis_csi'] - cc1_metrics['genesis_csi']) / cc1_metrics['genesis_csi'] * 100 if cc1_metrics['genesis_csi'] > 0 else 0
        lys_adv = (cc2_metrics['lysis_csi'] - cc1_metrics['lysis_csi']) / cc1_metrics['lysis_csi'] * 100 if cc1_metrics['lysis_csi'] > 0 else 0

        label = f"{core_thresh:.2f}" if core_thresh < 1.0 else "1.00 (100%)"
        print(f"{label:<12} {obs_genesis.sum():<8d} {obs_lysis.sum():<8d} "
              f"{cc1_metrics['genesis_csi']:<12.4f} {cc2_metrics['genesis_csi']:<12.4f} {gen_adv:>9.1f}% "
              f"{cc1_metrics['lysis_csi']:<12.4f} {cc2_metrics['lysis_csi']:<12.4f} {lys_adv:>9.1f}%")

        results.append({
            'core_thresh': core_thresh,
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
    best_gen = max([r for r in results if r['n_genesis'] > 50],
                   key=lambda x: x['cc2_genesis_csi'])
    print(f"\nBest CC2 Genesis CSI (with ≥50 events): core_thresh = {best_gen['core_thresh']:.2f}")
    print(f"  CC2 Genesis CSI: {best_gen['cc2_genesis_csi']:.4f} ({best_gen['gen_adv']:+.1f}%)")
    print(f"  Events: {best_gen['n_genesis']} genesis, {best_gen['n_lysis']} lysis")

    # Best by highest advantage
    best_adv = max([r for r in results if r['n_genesis'] > 50],
                   key=lambda x: x['gen_adv'] + x['lys_adv'])
    print(f"\nBest Combined Advantage (with ≥50 events): core_thresh = {best_adv['core_thresh']:.2f}")
    print(f"  Genesis advantage: {best_adv['gen_adv']:+.1f}%")
    print(f"  Lysis advantage: {best_adv['lys_adv']:+.1f}%")
    print(f"  Events: {best_adv['n_genesis']} genesis, {best_adv['n_lysis']} lysis")

    # Recommended balanced value
    print(f"\nRECOMMENDED: core_thresh = 0.85")
    print(f"  Interpretation: ≥85% of pixels in patch must be clearly clear/cloudy")
    print(f"  For 10×10 patch (100 pixels): ≥85 pixels must be < 0.2 or > 0.8")

    print("\n" + "="*80)


if __name__ == "__main__":
    test_pixel_count_thresholds()
