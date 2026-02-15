#!/usr/bin/env python3
"""
Full spatial accuracy analysis over all forecasts.

Computes relaxed CSI and distance metrics for 2,924 samples
(731 forecasts × 4 time windows).
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from scipy.spatial.distance import cdist
from scipy import stats
import matplotlib.pyplot as plt

from genesis_lysis_metric_pixel_count import (
    load_forecast_data_12h,
    compute_patch_stats_with_pixel_counts,
    classify_patches_pixel_count
)


def compute_relaxed_csi(obs_events, pred_events, patch_coords, spatial_tolerance_km):
    """
    Compute CSI with spatial tolerance.

    Returns:
        relaxed_csi, hits, misses, false_alarms, mean_distance
    """
    obs_locs = patch_coords[obs_events]
    pred_locs = patch_coords[pred_events]

    if len(obs_locs) == 0 and len(pred_locs) == 0:
        return 1.0, 0, 0, 0, 0.0
    if len(obs_locs) == 0:
        return 0.0, 0, 0, len(pred_locs), 0.0
    if len(pred_locs) == 0:
        return 0.0, 0, len(obs_locs), 0, 0.0

    distances = cdist(obs_locs, pred_locs, metric='euclidean')

    min_dist_to_pred = distances.min(axis=1)
    obs_matched = min_dist_to_pred <= spatial_tolerance_km

    min_dist_to_obs = distances.min(axis=0)
    pred_matched = min_dist_to_obs <= spatial_tolerance_km

    hits = obs_matched.sum()
    misses = (~obs_matched).sum()
    false_alarms = (~pred_matched).sum()

    if hits > 0:
        mean_distance = min_dist_to_pred[obs_matched].mean()
    else:
        mean_distance = 0.0

    relaxed_csi = hits / (hits + misses + false_alarms + 1e-10)

    return relaxed_csi, hits, misses, false_alarms, mean_distance


def compute_distance_metrics(obs_events, pred_events, patch_coords):
    """Compute distance from each observed event to nearest predicted event."""
    obs_locs = patch_coords[obs_events]
    pred_locs = patch_coords[pred_events]

    if len(obs_locs) == 0 or len(pred_locs) == 0:
        return np.nan, np.nan, np.nan

    distances = cdist(obs_locs, pred_locs, metric='euclidean')
    min_distances = distances.min(axis=1)

    return np.median(min_distances), np.percentile(min_distances, 90), np.mean(min_distances)


def full_spatial_accuracy_analysis():
    """Run spatial accuracy analysis over full dataset."""
    print("="*80)
    print("SPATIAL ACCURACY ANALYSIS - FULL DATASET")
    print("="*80)

    # Load data
    cc1_pred, cc2_pred, truth = load_forecast_data_12h()
    n_forecasts = truth.shape[0]

    # Parameters
    patch_size = 10
    change_threshold = 0.25
    clear_threshold = 0.2
    cloudy_threshold = 0.8
    core_threshold = 0.85
    tolerances = [25, 50, 75, 100]  # km

    # Time windows
    time_windows = [
        (0, 3, "t+0-3h"),
        (3, 6, "t+3-6h"),
        (6, 9, "t+6-9h"),
        (9, 12, "t+9-12h"),
    ]

    print(f"\nConfiguration:")
    print(f"  Forecasts: {n_forecasts}")
    print(f"  Windows: {len(time_windows)}")
    print(f"  Total samples: {n_forecasts * len(time_windows)}")
    print(f"  Spatial tolerances: {tolerances} km")

    results = []

    for forecast_idx in tqdm(range(n_forecasts), desc="Processing forecasts"):
        for t0, t1, window_label in time_windows:
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

            # Create coordinate grid
            y_starts, x_starts = patch_grid
            n_y, n_x = obs_genesis.shape
            patch_coords = np.zeros((n_y, n_x, 2))
            for i in range(n_y):
                for j in range(n_x):
                    patch_coords[i, j] = [y_starts[i] * 5, x_starts[j] * 5]

            # Store basic info
            sample_results = {
                'forecast_idx': forecast_idx,
                'time_window': window_label,
                't0': t0,
                't1': t1,
                'n_obs_genesis': obs_genesis.sum(),
                'n_obs_lysis': obs_lysis.sum(),
            }

            # Standard CSI (0km tolerance)
            hits = (obs_genesis & cc1_genesis).sum()
            misses = (obs_genesis & ~cc1_genesis).sum()
            fa = (~obs_genesis & cc1_genesis).sum()
            sample_results['cc1_genesis_csi_0km'] = hits / (hits + misses + fa + 1e-10)

            hits = (obs_genesis & cc2_genesis).sum()
            misses = (obs_genesis & ~cc2_genesis).sum()
            fa = (~obs_genesis & cc2_genesis).sum()
            sample_results['cc2_genesis_csi_0km'] = hits / (hits + misses + fa + 1e-10)

            hits = (obs_lysis & cc1_lysis).sum()
            misses = (obs_lysis & ~cc1_lysis).sum()
            fa = (~obs_lysis & cc1_lysis).sum()
            sample_results['cc1_lysis_csi_0km'] = hits / (hits + misses + fa + 1e-10)

            hits = (obs_lysis & cc2_lysis).sum()
            misses = (obs_lysis & ~cc2_lysis).sum()
            fa = (~obs_lysis & cc2_lysis).sum()
            sample_results['cc2_lysis_csi_0km'] = hits / (hits + misses + fa + 1e-10)

            # Relaxed CSI for each tolerance
            for tol in tolerances:
                # Genesis - CC1
                csi, hits, misses, fa, mean_dist = compute_relaxed_csi(
                    obs_genesis, cc1_genesis, patch_coords, tol
                )
                sample_results[f'cc1_genesis_csi_{tol}km'] = csi
                sample_results[f'cc1_genesis_hits_{tol}km'] = hits

                # Genesis - CC2
                csi, hits, misses, fa, mean_dist = compute_relaxed_csi(
                    obs_genesis, cc2_genesis, patch_coords, tol
                )
                sample_results[f'cc2_genesis_csi_{tol}km'] = csi
                sample_results[f'cc2_genesis_hits_{tol}km'] = hits

                # Lysis - CC1
                csi, hits, misses, fa, mean_dist = compute_relaxed_csi(
                    obs_lysis, cc1_lysis, patch_coords, tol
                )
                sample_results[f'cc1_lysis_csi_{tol}km'] = csi
                sample_results[f'cc1_lysis_hits_{tol}km'] = hits

                # Lysis - CC2
                csi, hits, misses, fa, mean_dist = compute_relaxed_csi(
                    obs_lysis, cc2_lysis, patch_coords, tol
                )
                sample_results[f'cc2_lysis_csi_{tol}km'] = csi
                sample_results[f'cc2_lysis_hits_{tol}km'] = hits

            # Distance metrics
            med, p90, mean = compute_distance_metrics(obs_genesis, cc1_genesis, patch_coords)
            sample_results['cc1_genesis_median_dist'] = med
            sample_results['cc1_genesis_p90_dist'] = p90

            med, p90, mean = compute_distance_metrics(obs_genesis, cc2_genesis, patch_coords)
            sample_results['cc2_genesis_median_dist'] = med
            sample_results['cc2_genesis_p90_dist'] = p90

            med, p90, mean = compute_distance_metrics(obs_lysis, cc1_lysis, patch_coords)
            sample_results['cc1_lysis_median_dist'] = med
            sample_results['cc1_lysis_p90_dist'] = p90

            med, p90, mean = compute_distance_metrics(obs_lysis, cc2_lysis, patch_coords)
            sample_results['cc2_lysis_median_dist'] = med
            sample_results['cc2_lysis_p90_dist'] = p90

            results.append(sample_results)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save results
    output_dir = Path("genesis_lysis_plots")
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "spatial_accuracy_full_results.csv", index=False)

    # Print summary statistics
    print_summary_statistics(df, tolerances, time_windows)

    # Create plots
    create_summary_plots(df, tolerances, time_windows, output_dir)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - spatial_accuracy_full_results.csv")
    print(f"  - spatial_accuracy_full_analysis.png")


def print_summary_statistics(df, tolerances, time_windows):
    """Print comprehensive summary statistics."""
    print("\n" + "="*80)
    print("SPATIAL ACCURACY - FULL ANALYSIS SUMMARY")
    print("="*80)

    print("\n" + "="*80)
    print("OVERALL STATISTICS (all time windows)")
    print("="*80)

    # Overall CSI at different tolerances
    print(f"\nGenesis CSI by Spatial Tolerance:")
    print(f"{'Tolerance':<12} {'CC1 CSI':<12} {'CC2 CSI':<12} {'CC2-CC1':<12} {'CC2 Adv':<12}")
    print("-" * 60)

    for tol_label in ['0km'] + [f'{tol}km' for tol in tolerances]:
        cc1_mean = df[f'cc1_genesis_csi_{tol_label}'].mean()
        cc2_mean = df[f'cc2_genesis_csi_{tol_label}'].mean()
        diff = cc2_mean - cc1_mean
        adv = (diff / cc1_mean * 100) if cc1_mean > 0 else 0

        print(f"{tol_label:<12} {cc1_mean:<12.4f} {cc2_mean:<12.4f} {diff:<12.4f} {adv:>10.1f}%")

    print(f"\nLysis CSI by Spatial Tolerance:")
    print(f"{'Tolerance':<12} {'CC1 CSI':<12} {'CC2 CSI':<12} {'CC2-CC1':<12} {'CC2 Adv':<12}")
    print("-" * 60)

    for tol_label in ['0km'] + [f'{tol}km' for tol in tolerances]:
        cc1_mean = df[f'cc1_lysis_csi_{tol_label}'].mean()
        cc2_mean = df[f'cc2_lysis_csi_{tol_label}'].mean()
        diff = cc2_mean - cc1_mean
        adv = (diff / cc1_mean * 100) if cc1_mean > 0 else 0

        print(f"{tol_label:<12} {cc1_mean:<12.4f} {cc2_mean:<12.4f} {diff:<12.4f} {adv:>10.1f}%")

    # Distance metrics
    print(f"\nDistance to Nearest Predicted Event (Genesis):")
    print(f"{'Metric':<20} {'CC1':<15} {'CC2':<15} {'Improvement':<15}")
    print("-" * 65)

    cc1_med = df['cc1_genesis_median_dist'].median()
    cc2_med = df['cc2_genesis_median_dist'].median()
    print(f"{'Median (km)':<20} {cc1_med:<15.1f} {cc2_med:<15.1f} {(cc1_med-cc2_med)/cc1_med*100:>13.1f}%")

    cc1_p90 = df['cc1_genesis_p90_dist'].median()
    cc2_p90 = df['cc2_genesis_p90_dist'].median()
    print(f"{'90th pct (km)':<20} {cc1_p90:<15.1f} {cc2_p90:<15.1f} {(cc1_p90-cc2_p90)/cc1_p90*100:>13.1f}%")

    print(f"\nDistance to Nearest Predicted Event (Lysis):")
    print(f"{'Metric':<20} {'CC1':<15} {'CC2':<15} {'Improvement':<15}")
    print("-" * 65)

    cc1_med = df['cc1_lysis_median_dist'].median()
    cc2_med = df['cc2_lysis_median_dist'].median()
    print(f"{'Median (km)':<20} {cc1_med:<15.1f} {cc2_med:<15.1f} {(cc1_med-cc2_med)/cc1_med*100:>13.1f}%")

    cc1_p90 = df['cc1_lysis_p90_dist'].median()
    cc2_p90 = df['cc2_lysis_p90_dist'].median()
    print(f"{'90th pct (km)':<20} {cc1_p90:<15.1f} {cc2_p90:<15.1f} {(cc1_p90-cc2_p90)/cc1_p90*100:>13.1f}%")

    # Statistical significance
    print(f"\nStatistical Significance (paired t-test):")
    for tol in [0, 50, 100]:
        tol_label = f'{tol}km' if tol > 0 else '0km'
        t_gen, p_gen = stats.ttest_rel(
            df[f'cc2_genesis_csi_{tol_label}'],
            df[f'cc1_genesis_csi_{tol_label}']
        )
        print(f"  Genesis CSI ({tol_label}): t={t_gen:.3f}, p={p_gen:.4e} {'***' if p_gen < 0.001 else ''}")

    # By time window
    print("\n" + "="*80)
    print("PERFORMANCE BY TIME WINDOW")
    print("="*80)

    for _, _, window_label in time_windows:
        window_data = df[df['time_window'] == window_label]

        print(f"\n{window_label}:")
        print(f"  Samples: {len(window_data)}")

        # CSI at 0km and 50km
        cc1_0km = window_data['cc1_genesis_csi_0km'].mean()
        cc2_0km = window_data['cc2_genesis_csi_0km'].mean()
        adv_0km = (cc2_0km - cc1_0km) / cc1_0km * 100

        cc1_50km = window_data['cc1_genesis_csi_50km'].mean()
        cc2_50km = window_data['cc2_genesis_csi_50km'].mean()
        adv_50km = (cc2_50km - cc1_50km) / cc1_50km * 100

        print(f"  Genesis CSI (0km):  CC1={cc1_0km:.4f}, CC2={cc2_0km:.4f} ({adv_0km:+.1f}%)")
        print(f"  Genesis CSI (50km): CC1={cc1_50km:.4f}, CC2={cc2_50km:.4f} ({adv_50km:+.1f}%)")

        # Distance
        cc1_dist = window_data['cc1_genesis_median_dist'].median()
        cc2_dist = window_data['cc2_genesis_median_dist'].median()
        print(f"  Median distance: CC1={cc1_dist:.1f}km, CC2={cc2_dist:.1f}km")

    print("\n" + "="*80)


def create_summary_plots(df, tolerances, time_windows, output_dir):
    """Create summary visualization."""
    print("\nCreating summary plots...")

    fig = plt.figure(figsize=(18, 12))

    windows = [label for _, _, label in time_windows]
    x_pos = np.arange(len(windows))

    # Panel 1: Genesis CSI at different tolerances
    ax1 = plt.subplot(2, 3, 1)
    for tol in [0, 50, 100]:
        tol_label = f'{tol}km' if tol > 0 else '0km'
        cc2_vals = [df[df['time_window']==w][f'cc2_genesis_csi_{tol_label}'].mean() for w in windows]
        ax1.plot(x_pos, cc2_vals, 'o-', label=f'CC2 ({tol_label})', linewidth=2, markersize=8)

    ax1.set_ylabel('Genesis CSI', fontsize=11)
    ax1.set_title('CC2 Genesis CSI vs Tolerance', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(windows)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: CC2 advantage vs tolerance
    ax2 = plt.subplot(2, 3, 2)
    for tol in [0, 50, 100]:
        tol_label = f'{tol}km' if tol > 0 else '0km'
        cc1_vals = [df[df['time_window']==w][f'cc1_genesis_csi_{tol_label}'].mean() for w in windows]
        cc2_vals = [df[df['time_window']==w][f'cc2_genesis_csi_{tol_label}'].mean() for w in windows]
        adv = [(c2-c1)/c1*100 for c1, c2 in zip(cc1_vals, cc2_vals)]
        ax2.plot(x_pos, adv, 'o-', label=f'{tol_label}', linewidth=2, markersize=8)

    ax2.set_ylabel('CC2 Advantage (%)', fontsize=11)
    ax2.set_title('CC2 Genesis Advantage vs Tolerance', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(windows)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # Panel 3: Median distance
    ax3 = plt.subplot(2, 3, 3)
    cc1_dist = [df[df['time_window']==w]['cc1_genesis_median_dist'].median() for w in windows]
    cc2_dist = [df[df['time_window']==w]['cc2_genesis_median_dist'].median() for w in windows]

    width = 0.35
    ax3.bar(x_pos - width/2, cc1_dist, width, label='CC1', color='orange', alpha=0.7)
    ax3.bar(x_pos + width/2, cc2_dist, width, label='CC2', color='green', alpha=0.7)
    ax3.set_ylabel('Median Distance (km)', fontsize=11)
    ax3.set_title('Distance to Nearest Prediction (Genesis)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(windows)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # Panel 4-6: Same for Lysis
    ax4 = plt.subplot(2, 3, 4)
    for tol in [0, 50, 100]:
        tol_label = f'{tol}km' if tol > 0 else '0km'
        cc2_vals = [df[df['time_window']==w][f'cc2_lysis_csi_{tol_label}'].mean() for w in windows]
        ax4.plot(x_pos, cc2_vals, 'o-', label=f'CC2 ({tol_label})', linewidth=2, markersize=8)

    ax4.set_ylabel('Lysis CSI', fontsize=11)
    ax4.set_title('CC2 Lysis CSI vs Tolerance', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(windows)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax5 = plt.subplot(2, 3, 5)
    for tol in [0, 50, 100]:
        tol_label = f'{tol}km' if tol > 0 else '0km'
        cc1_vals = [df[df['time_window']==w][f'cc1_lysis_csi_{tol_label}'].mean() for w in windows]
        cc2_vals = [df[df['time_window']==w][f'cc2_lysis_csi_{tol_label}'].mean() for w in windows]
        adv = [(c2-c1)/c1*100 for c1, c2 in zip(cc1_vals, cc2_vals)]
        ax5.plot(x_pos, adv, 'o-', label=f'{tol_label}', linewidth=2, markersize=8)

    ax5.set_ylabel('CC2 Advantage (%)', fontsize=11)
    ax5.set_title('CC2 Lysis Advantage vs Tolerance', fontsize=12, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(windows)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    ax6 = plt.subplot(2, 3, 6)
    cc1_dist = [df[df['time_window']==w]['cc1_lysis_median_dist'].median() for w in windows]
    cc2_dist = [df[df['time_window']==w]['cc2_lysis_median_dist'].median() for w in windows]

    ax6.bar(x_pos - width/2, cc1_dist, width, label='CC1', color='orange', alpha=0.7)
    ax6.bar(x_pos + width/2, cc2_dist, width, label='CC2', color='green', alpha=0.7)
    ax6.set_ylabel('Median Distance (km)', fontsize=11)
    ax6.set_title('Distance to Nearest Prediction (Lysis)', fontsize=12, fontweight='bold')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(windows)
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)

    plt.suptitle('Spatial Accuracy Analysis: Full Dataset (2,924 samples)\n' +
                 'Pixel Count ≥85% | Relaxed CSI with spatial tolerance',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    filename = output_dir / 'spatial_accuracy_full_analysis.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved: {filename}")
    plt.close()


if __name__ == "__main__":
    full_spatial_accuracy_analysis()
