#!/usr/bin/env python3
"""
Full analysis of genesis/lysis metric using pixel counting approach.

Analyzes 731 forecasts × 4 time windows = 2,924 samples
with pixel-based homogeneity constraint (≥85% pixels clearly clear/cloudy).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from scipy import stats

from genesis_lysis_metric_pixel_count import (
    load_forecast_data_12h,
    compute_patch_stats_with_pixel_counts,
    compute_genesis_lysis_metrics_pixel_count
)


def full_analysis_pixel_count():
    """Run full analysis with pixel counting constraint."""
    print("="*80)
    print("GENESIS/LYSIS WITH PIXEL COUNTING - FULL ANALYSIS")
    print("="*80)

    # Load data
    cc1_pred, cc2_pred, truth = load_forecast_data_12h()

    n_forecasts = truth.shape[0]

    # Parameters
    patch_size = 10
    change_threshold = 0.25
    clear_threshold = 0.2
    cloudy_threshold = 0.8
    core_threshold = 0.85  # ≥85% of pixels must be clearly clear/cloudy

    # Time windows
    time_windows = [
        (0, 3, "t+0-3h"),
        (3, 6, "t+3-6h"),
        (6, 9, "t+6-9h"),
        (9, 12, "t+9-12h"),
    ]

    print(f"\nComputing genesis/lysis metrics WITH pixel counting constraint...")
    print(f"  Forecasts: {n_forecasts}")
    print(f"  Windows: {len(time_windows)} non-overlapping 3h windows")
    print(f"  Total samples: {n_forecasts * len(time_windows)}")
    print(f"  Patch size: {patch_size} cells = {patch_size*5} km")
    print(f"  Change threshold: ±{change_threshold}")
    print(f"  Clear/cloudy thresholds: < {clear_threshold} / > {cloudy_threshold}")
    print(f"  Core threshold: ≥{core_threshold:.0%} pixels must be clearly clear/cloudy")

    # Store results
    results = []

    for forecast_idx in tqdm(range(n_forecasts), desc="Processing forecasts"):
        for t0, t1, window_label in time_windows:
            # Extract fields
            tcc_t0 = truth[forecast_idx, t0].numpy()
            tcc_t1 = truth[forecast_idx, t1].numpy()
            cc1_t1 = cc1_pred[forecast_idx, t1].numpy()
            cc2_t1 = cc2_pred[forecast_idx, t1].numpy()

            # Compute patch stats with pixel counts
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

            # Count events
            obs_genesis = (
                (obs_changes >= change_threshold) &
                (obs_clear_frac > core_threshold)
            )
            obs_lysis = (
                (obs_changes <= -change_threshold) &
                (obs_cloudy_frac > core_threshold)
            )

            n_patches = obs_changes.size
            n_genesis = obs_genesis.sum()
            n_lysis = obs_lysis.sum()
            n_neutral = n_patches - n_genesis - n_lysis

            # Compute metrics
            cc1_metrics = compute_genesis_lysis_metrics_pixel_count(
                obs_changes, obs_clear_frac, obs_cloudy_frac, cc1_changes,
                change_threshold, core_threshold, core_threshold
            )
            cc2_metrics = compute_genesis_lysis_metrics_pixel_count(
                obs_changes, obs_clear_frac, obs_cloudy_frac, cc2_changes,
                change_threshold, core_threshold, core_threshold
            )

            # Store results
            results.append({
                'forecast_idx': forecast_idx,
                'time_window': window_label,
                't0': t0,
                't1': t1,
                'n_patches': n_patches,
                'n_genesis': n_genesis,
                'n_lysis': n_lysis,
                'n_neutral': n_neutral,
                'genesis_fraction': n_genesis / n_patches,
                'lysis_fraction': n_lysis / n_patches,
                # CC1 metrics
                'cc1_correlation': cc1_metrics['correlation'],
                'cc1_rmse': cc1_metrics['rmse'],
                'cc1_genesis_csi': cc1_metrics['genesis_csi'],
                'cc1_lysis_csi': cc1_metrics['lysis_csi'],
                'cc1_genesis_hits': cc1_metrics['genesis_hits'],
                'cc1_genesis_misses': cc1_metrics['genesis_misses'],
                'cc1_genesis_false_alarms': cc1_metrics['genesis_false_alarms'],
                'cc1_lysis_hits': cc1_metrics['lysis_hits'],
                'cc1_lysis_misses': cc1_metrics['lysis_misses'],
                'cc1_lysis_false_alarms': cc1_metrics['lysis_false_alarms'],
                # CC2 metrics
                'cc2_correlation': cc2_metrics['correlation'],
                'cc2_rmse': cc2_metrics['rmse'],
                'cc2_genesis_csi': cc2_metrics['genesis_csi'],
                'cc2_lysis_csi': cc2_metrics['lysis_csi'],
                'cc2_genesis_hits': cc2_metrics['genesis_hits'],
                'cc2_genesis_misses': cc2_metrics['genesis_misses'],
                'cc2_genesis_false_alarms': cc2_metrics['genesis_false_alarms'],
                'cc2_lysis_hits': cc2_metrics['lysis_hits'],
                'cc2_lysis_misses': cc2_metrics['lysis_misses'],
                'cc2_lysis_false_alarms': cc2_metrics['lysis_false_alarms'],
            })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save results
    output_dir = Path("genesis_lysis_plots")
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "genesis_lysis_pixel_count_full_results.csv", index=False)

    # Compute summary statistics
    print_summary_statistics(df, time_windows)

    # Create plots
    create_summary_plots(df, time_windows, output_dir)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - genesis_lysis_pixel_count_full_results.csv")
    print(f"  - genesis_lysis_pixel_count_full_analysis.png")


def print_summary_statistics(df, time_windows):
    """Print comprehensive summary statistics."""
    print("\n" + "="*80)
    print("GENESIS/LYSIS WITH PIXEL COUNTING - FULL ANALYSIS SUMMARY")
    print("="*80)

    print("\n" + "="*80)
    print("OVERALL STATISTICS (all time windows)")
    print("="*80)

    print(f"\nTotal samples: {len(df)} ({df['forecast_idx'].nunique()} forecasts × {len(time_windows)} windows)")
    print(f"Average genesis events per sample: {df['n_genesis'].mean():.1f} ± {df['n_genesis'].std():.1f}")
    print(f"Average lysis events per sample: {df['n_lysis'].mean():.1f} ± {df['n_lysis'].std():.1f}")
    print(f"Average genesis fraction: {df['genesis_fraction'].mean()*100:.1f}% ± {df['genesis_fraction'].std()*100:.1f}%")
    print(f"Average lysis fraction: {df['lysis_fraction'].mean()*100:.1f}% ± {df['lysis_fraction'].std()*100:.1f}%")

    # CC1 performance
    print(f"\nCC1 Performance (mean ± std):")
    print(f"  Correlation: {df['cc1_correlation'].mean():.4f} ± {df['cc1_correlation'].std():.4f}")
    print(f"  RMSE: {df['cc1_rmse'].mean():.4f} ± {df['cc1_rmse'].std():.4f}")
    print(f"  Genesis CSI: {df['cc1_genesis_csi'].mean():.4f} ± {df['cc1_genesis_csi'].std():.4f}")
    print(f"  Lysis CSI: {df['cc1_lysis_csi'].mean():.4f} ± {df['cc1_lysis_csi'].std():.4f}")

    # CC2 performance
    print(f"\nCC2 Performance (mean ± std):")
    print(f"  Correlation: {df['cc2_correlation'].mean():.4f} ± {df['cc2_correlation'].std():.4f}")
    print(f"  RMSE: {df['cc2_rmse'].mean():.4f} ± {df['cc2_rmse'].std():.4f}")
    print(f"  Genesis CSI: {df['cc2_genesis_csi'].mean():.4f} ± {df['cc2_genesis_csi'].std():.4f}")
    print(f"  Lysis CSI: {df['cc2_lysis_csi'].mean():.4f} ± {df['cc2_lysis_csi'].std():.4f}")

    # CC2 vs CC1 comparison
    corr_diff = df['cc2_correlation'] - df['cc1_correlation']
    gen_csi_diff = df['cc2_genesis_csi'] - df['cc1_genesis_csi']
    lys_csi_diff = df['cc2_lysis_csi'] - df['cc1_lysis_csi']

    print(f"\nCC2 vs CC1 Comparison (overall):")
    print(f"  Correlation difference: {corr_diff.mean():.4f} ± {corr_diff.std():.4f}")
    print(f"    CC2 wins: {(corr_diff > 0).sum()} / {len(df)} ({(corr_diff > 0).sum()/len(df)*100:.1f}%)")

    gen_rel_improve = (df['cc2_genesis_csi'].mean() - df['cc1_genesis_csi'].mean()) / df['cc1_genesis_csi'].mean() * 100
    print(f"  Genesis CSI difference: {gen_csi_diff.mean():.4f} ± {gen_csi_diff.std():.4f}")
    print(f"    CC2 wins: {(gen_csi_diff > 0).sum()} / {len(df)} ({(gen_csi_diff > 0).sum()/len(df)*100:.1f}%)")
    print(f"    Relative improvement: {gen_rel_improve:+.1f}%")

    lys_rel_improve = (df['cc2_lysis_csi'].mean() - df['cc1_lysis_csi'].mean()) / df['cc1_lysis_csi'].mean() * 100
    print(f"  Lysis CSI difference: {lys_csi_diff.mean():.4f} ± {lys_csi_diff.std():.4f}")
    print(f"    CC2 wins: {(lys_csi_diff > 0).sum()} / {len(df)} ({(lys_csi_diff > 0).sum()/len(df)*100:.1f}%)")
    print(f"    Relative improvement: {lys_rel_improve:+.1f}%")

    # Statistical significance
    print(f"\nStatistical Significance (paired t-test, all samples):")
    t_corr, p_corr = stats.ttest_rel(df['cc2_correlation'], df['cc1_correlation'])
    t_gen, p_gen = stats.ttest_rel(df['cc2_genesis_csi'], df['cc1_genesis_csi'])
    t_lys, p_lys = stats.ttest_rel(df['cc2_lysis_csi'], df['cc1_lysis_csi'])

    print(f"  Correlation: t={t_corr:.3f}, p={p_corr:.4e} {'***' if p_corr < 0.001 else ''}")
    print(f"  Genesis CSI: t={t_gen:.3f}, p={p_gen:.4e} {'***' if p_gen < 0.001 else ''}")
    print(f"  Lysis CSI: t={t_lys:.3f}, p={p_lys:.4e} {'***' if p_lys < 0.001 else ''}")

    # By time window
    print("\n" + "="*80)
    print("PERFORMANCE BY TIME WINDOW")
    print("="*80)

    for _, _, window_label in time_windows:
        window_data = df[df['time_window'] == window_label]

        n_gen = window_data['n_genesis'].mean()
        n_lys = window_data['n_lysis'].mean()

        cc1_gen_csi = window_data['cc1_genesis_csi'].mean()
        cc2_gen_csi = window_data['cc2_genesis_csi'].mean()
        gen_improve = (cc2_gen_csi - cc1_gen_csi) / cc1_gen_csi * 100

        cc1_lys_csi = window_data['cc1_lysis_csi'].mean()
        cc2_lys_csi = window_data['cc2_lysis_csi'].mean()
        lys_improve = (cc2_lys_csi - cc1_lys_csi) / cc1_lys_csi * 100

        print(f"\n{window_label}:")
        print(f"  Samples: {len(window_data)}")
        print(f"  Avg events: Genesis={n_gen:.1f}, Lysis={n_lys:.1f}")
        print(f"  CC1: Gen_CSI={cc1_gen_csi:.4f}, Lys_CSI={cc1_lys_csi:.4f}")
        print(f"  CC2: Gen_CSI={cc2_gen_csi:.4f}, Lys_CSI={cc2_lys_csi:.4f}")
        print(f"  CC2 improvement: Genesis={gen_improve:+.1f}%, Lysis={lys_improve:+.1f}%")

    print("\n" + "="*80)


def create_summary_plots(df, time_windows, output_dir):
    """Create summary visualization plots."""
    print("\nCreating summary plots...")

    fig = plt.figure(figsize=(16, 10))

    # Extract data by time window
    windows = [label for _, _, label in time_windows]
    x_pos = np.arange(len(windows))

    # Panel 1: Event counts
    ax1 = plt.subplot(2, 3, 1)
    genesis_counts = [df[df['time_window']==w]['n_genesis'].mean() for w in windows]
    lysis_counts = [df[df['time_window']==w]['n_lysis'].mean() for w in windows]

    width = 0.35
    ax1.bar(x_pos - width/2, genesis_counts, width, label='Genesis', color='red', alpha=0.7)
    ax1.bar(x_pos + width/2, lysis_counts, width, label='Lysis', color='blue', alpha=0.7)
    ax1.set_ylabel('Events per sample', fontsize=11)
    ax1.set_title('Average Event Counts\n(pixel count ≥85%)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(windows)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Panel 2: Genesis CSI comparison
    ax2 = plt.subplot(2, 3, 2)
    cc1_gen = [df[df['time_window']==w]['cc1_genesis_csi'].mean() for w in windows]
    cc2_gen = [df[df['time_window']==w]['cc2_genesis_csi'].mean() for w in windows]

    ax2.plot(x_pos, cc1_gen, 'o-', label='CC1', linewidth=2, markersize=8, color='orange')
    ax2.plot(x_pos, cc2_gen, 's-', label='CC2', linewidth=2, markersize=8, color='green')
    ax2.set_ylabel('Genesis CSI', fontsize=11)
    ax2.set_title('Genesis Detection Performance', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(windows)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Lysis CSI comparison
    ax3 = plt.subplot(2, 3, 3)
    cc1_lys = [df[df['time_window']==w]['cc1_lysis_csi'].mean() for w in windows]
    cc2_lys = [df[df['time_window']==w]['cc2_lysis_csi'].mean() for w in windows]

    ax3.plot(x_pos, cc1_lys, 'o-', label='CC1', linewidth=2, markersize=8, color='orange')
    ax3.plot(x_pos, cc2_lys, 's-', label='CC2', linewidth=2, markersize=8, color='green')
    ax3.set_ylabel('Lysis CSI', fontsize=11)
    ax3.set_title('Lysis Detection Performance', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(windows)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: CC2 relative advantage
    ax4 = plt.subplot(2, 3, 4)
    gen_adv = [(cc2-cc1)/cc1*100 for cc1, cc2 in zip(cc1_gen, cc2_gen)]
    lys_adv = [(cc2-cc1)/cc1*100 for cc1, cc2 in zip(cc1_lys, cc2_lys)]

    width = 0.35
    ax4.bar(x_pos - width/2, gen_adv, width, label='Genesis', color='red', alpha=0.7)
    ax4.bar(x_pos + width/2, lys_adv, width, label='Lysis', color='blue', alpha=0.7)
    ax4.set_ylabel('CC2 Advantage (%)', fontsize=11)
    ax4.set_title('CC2 Relative Improvement over CC1', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(windows)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # Panel 5: Correlation
    ax5 = plt.subplot(2, 3, 5)
    cc1_corr = [df[df['time_window']==w]['cc1_correlation'].mean() for w in windows]
    cc2_corr = [df[df['time_window']==w]['cc2_correlation'].mean() for w in windows]

    ax5.plot(x_pos, cc1_corr, 'o-', label='CC1', linewidth=2, markersize=8, color='orange')
    ax5.plot(x_pos, cc2_corr, 's-', label='CC2', linewidth=2, markersize=8, color='green')
    ax5.set_ylabel('Correlation', fontsize=11)
    ax5.set_title('Patch Change Correlation', fontsize=12, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(windows)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Panel 6: Distribution of CSI differences
    ax6 = plt.subplot(2, 3, 6)
    gen_diffs = df['cc2_genesis_csi'] - df['cc1_genesis_csi']
    lys_diffs = df['cc2_lysis_csi'] - df['cc1_lysis_csi']

    ax6.hist(gen_diffs, bins=50, alpha=0.6, label='Genesis CSI diff', color='red', density=True)
    ax6.hist(lys_diffs, bins=50, alpha=0.6, label='Lysis CSI diff', color='blue', density=True)
    ax6.axvline(x=0, color='k', linestyle='--', linewidth=1)
    ax6.axvline(x=gen_diffs.mean(), color='red', linestyle='-', linewidth=2, alpha=0.7)
    ax6.axvline(x=lys_diffs.mean(), color='blue', linestyle='-', linewidth=2, alpha=0.7)
    ax6.set_xlabel('CC2 - CC1 CSI', fontsize=11)
    ax6.set_ylabel('Density', fontsize=11)
    ax6.set_title('Distribution of CSI Improvements', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Genesis/Lysis Analysis with Pixel Counting (≥85% pixels clearly clear/cloudy)\n' +
                 f'2,924 samples | Patch: 10×10 cells (50km) | Change threshold: ±0.25',
                 fontsize=13, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    filename = output_dir / 'genesis_lysis_pixel_count_full_analysis.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved: {filename.name}")
    plt.close()


if __name__ == "__main__":
    full_analysis_pixel_count()
