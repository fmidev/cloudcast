#!/usr/bin/env python3
"""
Spatial accuracy analysis for genesis/lysis predictions.

Assesses whether CC2 not only detects more events, but places them correctly.

Metrics:
1. Relaxed CSI with spatial tolerance (25km, 50km, 75km, 100km)
2. Distance to nearest event (median, 90th percentile)
3. Spatial hit/miss/false alarm decomposition
4. Object-based matching and displacement analysis
"""

import torch
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from genesis_lysis_metric_pixel_count import (
    load_forecast_data_12h,
    compute_patch_stats_with_pixel_counts,
    classify_patches_pixel_count
)


def compute_relaxed_csi(obs_events, pred_events, patch_coords, spatial_tolerance_km):
    """
    Compute CSI with spatial tolerance.

    An observed event is considered "hit" if there's a predicted event
    within spatial_tolerance_km.

    Args:
        obs_events: [n_y, n_x] boolean mask of observed events
        pred_events: [n_y, n_x] boolean mask of predicted events
        patch_coords: [n_y, n_x, 2] array of (y_km, x_km) coordinates
        spatial_tolerance_km: distance threshold in km

    Returns:
        relaxed_csi: CSI with spatial tolerance
        hits: number of observed events with prediction nearby
        misses: number of observed events without prediction nearby
        false_alarms: number of predicted events not near any observation
        mean_distance: mean distance for matched events
    """
    # Get coordinates of events
    obs_locs = patch_coords[obs_events]  # [n_obs, 2]
    pred_locs = patch_coords[pred_events]  # [n_pred, 2]

    if len(obs_locs) == 0 and len(pred_locs) == 0:
        return 1.0, 0, 0, 0, 0.0
    if len(obs_locs) == 0:
        return 0.0, 0, 0, len(pred_locs), 0.0
    if len(pred_locs) == 0:
        return 0.0, 0, len(obs_locs), 0, 0.0

    # Compute pairwise distances [n_obs, n_pred]
    distances = cdist(obs_locs, pred_locs, metric='euclidean')

    # For each observed event, find nearest prediction
    min_dist_to_pred = distances.min(axis=1)  # [n_obs]
    obs_matched = min_dist_to_pred <= spatial_tolerance_km

    # For each predicted event, find nearest observation
    min_dist_to_obs = distances.min(axis=0)  # [n_pred]
    pred_matched = min_dist_to_obs <= spatial_tolerance_km

    # Hits: observed events with nearby prediction
    hits = obs_matched.sum()

    # Misses: observed events without nearby prediction
    misses = (~obs_matched).sum()

    # False alarms: predicted events not near any observation
    false_alarms = (~pred_matched).sum()

    # Mean distance for matched events
    if hits > 0:
        mean_distance = min_dist_to_pred[obs_matched].mean()
    else:
        mean_distance = 0.0

    # Relaxed CSI
    relaxed_csi = hits / (hits + misses + false_alarms + 1e-10)

    return relaxed_csi, hits, misses, false_alarms, mean_distance


def compute_distance_metrics(obs_events, pred_events, patch_coords):
    """
    Compute distance-based metrics for spatial accuracy.

    For each observed event, finds the nearest predicted event and computes
    distance statistics.

    Returns:
        median_distance: median distance to nearest prediction
        p90_distance: 90th percentile distance
        mean_distance: mean distance
        distances: array of all distances
    """
    obs_locs = patch_coords[obs_events]
    pred_locs = patch_coords[pred_events]

    if len(obs_locs) == 0 or len(pred_locs) == 0:
        return np.nan, np.nan, np.nan, np.array([])

    # Distance from each observation to nearest prediction
    distances = cdist(obs_locs, pred_locs, metric='euclidean')
    min_distances = distances.min(axis=1)

    return (
        np.median(min_distances),
        np.percentile(min_distances, 90),
        np.mean(min_distances),
        min_distances
    )


def analyze_spatial_accuracy_single_case(forecast_idx, t0, t1, patch_size=10,
                                         change_threshold=0.25,
                                         clear_threshold=0.2,
                                         cloudy_threshold=0.8,
                                         core_threshold=0.85):
    """
    Analyze spatial accuracy for a single forecast case.

    Returns dictionary with metrics for both genesis and lysis.
    """
    # Load data
    cc1_pred, cc2_pred, truth = load_forecast_data_12h()

    # Extract fields
    tcc_t0 = truth[forecast_idx, t0].numpy()
    tcc_t1 = truth[forecast_idx, t1].numpy()
    cc1_t1 = cc1_pred[forecast_idx, t1].numpy()
    cc2_t1 = cc2_pred[forecast_idx, t1].numpy()

    # Compute patch stats
    obs_changes, obs_mean, obs_clear_frac, obs_cloudy_frac, patch_grid = \
        compute_patch_stats_with_pixel_counts(
            tcc_t0, tcc_t1, patch_size, clear_threshold, cloudy_threshold
        )
    cc1_changes, _, _, _, _ = compute_patch_stats_with_pixel_counts(
        tcc_t0, cc1_t1, patch_size, clear_threshold, cloudy_threshold
    )
    cc2_changes, _, _, _, _ = compute_patch_stats_with_pixel_counts(
        tcc_t0, cc2_t1, patch_size, clear_threshold, cloudy_threshold
    )

    # Classify events
    _, obs_genesis, obs_lysis = classify_patches_pixel_count(
        obs_changes, obs_clear_frac, obs_cloudy_frac,
        change_threshold, core_threshold, core_threshold
    )
    _, cc1_genesis, cc1_lysis = classify_patches_pixel_count(
        cc1_changes, obs_clear_frac, obs_cloudy_frac,
        change_threshold, core_threshold, core_threshold
    )
    _, cc2_genesis, cc2_lysis = classify_patches_pixel_count(
        cc2_changes, obs_clear_frac, obs_cloudy_frac,
        change_threshold, core_threshold, core_threshold
    )

    # Create coordinate grid (in km, assuming 5km grid spacing)
    y_starts, x_starts = patch_grid
    n_y, n_x = obs_genesis.shape
    patch_coords = np.zeros((n_y, n_x, 2))
    for i in range(n_y):
        for j in range(n_x):
            patch_coords[i, j] = [y_starts[i] * 5, x_starts[j] * 5]  # Convert to km

    # Test multiple spatial tolerances
    tolerances = [25, 50, 75, 100]  # km

    results = {
        'forecast_idx': forecast_idx,
        't0': t0,
        't1': t1,
        'n_obs_genesis': obs_genesis.sum(),
        'n_obs_lysis': obs_lysis.sum(),
        'n_cc1_genesis': cc1_genesis.sum(),
        'n_cc1_lysis': cc1_lysis.sum(),
        'n_cc2_genesis': cc2_genesis.sum(),
        'n_cc2_lysis': cc2_lysis.sum(),
    }

    # Standard CSI (no tolerance)
    standard_genesis_hits = (obs_genesis & cc1_genesis).sum()
    standard_genesis_misses = (obs_genesis & ~cc1_genesis).sum()
    standard_genesis_fa = (~obs_genesis & cc1_genesis).sum()
    results['cc1_genesis_csi_standard'] = standard_genesis_hits / (
        standard_genesis_hits + standard_genesis_misses + standard_genesis_fa + 1e-10
    )

    standard_genesis_hits = (obs_genesis & cc2_genesis).sum()
    standard_genesis_misses = (obs_genesis & ~cc2_genesis).sum()
    standard_genesis_fa = (~obs_genesis & cc2_genesis).sum()
    results['cc2_genesis_csi_standard'] = standard_genesis_hits / (
        standard_genesis_hits + standard_genesis_misses + standard_genesis_fa + 1e-10
    )

    # Relaxed CSI for each tolerance
    for tol in tolerances:
        # Genesis - CC1
        csi, hits, misses, fa, mean_dist = compute_relaxed_csi(
            obs_genesis, cc1_genesis, patch_coords, tol
        )
        results[f'cc1_genesis_csi_{tol}km'] = csi
        results[f'cc1_genesis_hits_{tol}km'] = hits
        results[f'cc1_genesis_mean_dist_{tol}km'] = mean_dist

        # Genesis - CC2
        csi, hits, misses, fa, mean_dist = compute_relaxed_csi(
            obs_genesis, cc2_genesis, patch_coords, tol
        )
        results[f'cc2_genesis_csi_{tol}km'] = csi
        results[f'cc2_genesis_hits_{tol}km'] = hits
        results[f'cc2_genesis_mean_dist_{tol}km'] = mean_dist

        # Lysis - CC1
        csi, hits, misses, fa, mean_dist = compute_relaxed_csi(
            obs_lysis, cc1_lysis, patch_coords, tol
        )
        results[f'cc1_lysis_csi_{tol}km'] = csi
        results[f'cc1_lysis_hits_{tol}km'] = hits
        results[f'cc1_lysis_mean_dist_{tol}km'] = mean_dist

        # Lysis - CC2
        csi, hits, misses, fa, mean_dist = compute_relaxed_csi(
            obs_lysis, cc2_lysis, patch_coords, tol
        )
        results[f'cc2_lysis_csi_{tol}km'] = csi
        results[f'cc2_lysis_hits_{tol}km'] = hits
        results[f'cc2_lysis_mean_dist_{tol}km'] = mean_dist

    # Distance metrics
    # Genesis
    med, p90, mean, dists = compute_distance_metrics(obs_genesis, cc1_genesis, patch_coords)
    results['cc1_genesis_median_dist'] = med
    results['cc1_genesis_p90_dist'] = p90

    med, p90, mean, dists = compute_distance_metrics(obs_genesis, cc2_genesis, patch_coords)
    results['cc2_genesis_median_dist'] = med
    results['cc2_genesis_p90_dist'] = p90

    # Lysis
    med, p90, mean, dists = compute_distance_metrics(obs_lysis, cc1_lysis, patch_coords)
    results['cc1_lysis_median_dist'] = med
    results['cc1_lysis_p90_dist'] = p90

    med, p90, mean, dists = compute_distance_metrics(obs_lysis, cc2_lysis, patch_coords)
    results['cc2_lysis_median_dist'] = med
    results['cc2_lysis_p90_dist'] = p90

    return results


def test_spatial_accuracy():
    """Test spatial accuracy on a single forecast."""
    print("="*80)
    print("SPATIAL ACCURACY ANALYSIS - SINGLE CASE TEST")
    print("="*80)

    forecast_idx = 365
    t0 = 0
    t1 = 3

    print(f"\nAnalyzing forecast {forecast_idx}, t={t0}h → t={t1}h...")

    results = analyze_spatial_accuracy_single_case(forecast_idx, t0, t1)

    print(f"\n{'Metric':<40} {'CC1':<15} {'CC2':<15} {'CC2 Adv':<10}")
    print("-" * 80)

    # Event counts
    print(f"{'Genesis events (observed)':<40} {results['n_obs_genesis']}")
    print(f"{'Genesis events (predicted)':<40} {results['n_cc1_genesis']:<15} {results['n_cc2_genesis']}")
    print(f"{'Lysis events (observed)':<40} {results['n_obs_lysis']}")
    print(f"{'Lysis events (predicted)':<40} {results['n_cc1_lysis']:<15} {results['n_cc2_lysis']}")
    print()

    # Standard CSI
    cc1_std = results['cc1_genesis_csi_standard']
    cc2_std = results['cc2_genesis_csi_standard']
    adv = (cc2_std - cc1_std) / cc1_std * 100 if cc1_std > 0 else 0
    print(f"{'Genesis CSI (standard, 0km tol)':<40} {cc1_std:<15.4f} {cc2_std:<15.4f} {adv:>9.1f}%")
    print()

    # Relaxed CSI at different tolerances
    print("Genesis CSI with spatial tolerance:")
    for tol in [25, 50, 75, 100]:
        cc1_val = results[f'cc1_genesis_csi_{tol}km']
        cc2_val = results[f'cc2_genesis_csi_{tol}km']
        cc1_hits = results[f'cc1_genesis_hits_{tol}km']
        cc2_hits = results[f'cc2_genesis_hits_{tol}km']
        adv = (cc2_val - cc1_val) / cc1_val * 100 if cc1_val > 0 else 0

        print(f"  {tol}km tolerance: {'':<20} {cc1_val:.4f} (H={cc1_hits:<3}) {cc2_val:.4f} (H={cc2_hits:<3}) {adv:>9.1f}%")

    print()
    print("Lysis CSI with spatial tolerance:")
    for tol in [25, 50, 75, 100]:
        cc1_val = results[f'cc1_lysis_csi_{tol}km']
        cc2_val = results[f'cc2_lysis_csi_{tol}km']
        cc1_hits = results[f'cc1_lysis_hits_{tol}km']
        cc2_hits = results[f'cc2_lysis_hits_{tol}km']
        adv = (cc2_val - cc1_val) / cc1_val * 100 if cc1_val > 0 else 0

        print(f"  {tol}km tolerance: {'':<20} {cc1_val:.4f} (H={cc1_hits:<3}) {cc2_val:.4f} (H={cc2_hits:<3}) {adv:>9.1f}%")

    # Distance metrics
    print()
    print(f"{'Distance to nearest prediction (Genesis):':<40}")
    cc1_med = results['cc1_genesis_median_dist']
    cc2_med = results['cc2_genesis_median_dist']
    print(f"  {'Median distance':<38} {cc1_med:<15.1f} {cc2_med:<15.1f}")

    cc1_p90 = results['cc1_genesis_p90_dist']
    cc2_p90 = results['cc2_genesis_p90_dist']
    print(f"  {'90th percentile':<38} {cc1_p90:<15.1f} {cc2_p90:<15.1f}")

    print()
    print(f"{'Distance to nearest prediction (Lysis):':<40}")
    cc1_med = results['cc1_lysis_median_dist']
    cc2_med = results['cc2_lysis_median_dist']
    print(f"  {'Median distance':<38} {cc1_med:<15.1f} {cc2_med:<15.1f}")

    cc1_p90 = results['cc1_lysis_p90_dist']
    cc2_p90 = results['cc2_lysis_p90_dist']
    print(f"  {'90th percentile':<38} {cc1_p90:<15.1f} {cc2_p90:<15.1f}")

    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    print("\n1. Standard CSI (0km tolerance) is very strict - requires exact match")
    print("2. Relaxed CSI increases with tolerance - shows 'near miss' predictions")
    print("3. If CC2 advantage INCREASES with tolerance → CC2 has better spatial accuracy")
    print("4. If CC2 advantage DECREASES with tolerance → CC2 just predicts more events")
    print("5. Lower median distance → better spatial placement")

    print("\n" + "="*80)


if __name__ == "__main__":
    test_spatial_accuracy()
