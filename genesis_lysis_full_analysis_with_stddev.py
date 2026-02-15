#!/usr/bin/env python3
"""
Full Genesis/Lysis Analysis - WITH STDDEV CONSTRAINT

Final refined method:
- Change threshold: 0.25
- Genesis: initial mean < 0.2 AND stddev < 0.15
- Lysis: initial mean > 0.8 AND stddev < 0.15
- Non-overlapping 3h windows: t=0→3, t=3→6, t=6→9, t=9→12
- Total: 731 forecasts × 4 windows = 2,924 samples
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from genesis_lysis_metric_with_stddev import (
    load_forecast_data_12h,
    compute_patch_stats,
    compute_genesis_lysis_metrics_with_stddev,
    classify_patches_with_stddev
)


def compute_all_forecasts_with_stddev(cc1_pred, cc2_pred, truth,
                                      patch_size=10,
                                      change_threshold=0.25,
                                      clear_threshold=0.2,
                                      cloudy_threshold=0.8,
                                      std_threshold=0.15):
    """
    Compute genesis/lysis metrics with stddev constraint for all forecasts.
    """
    n_forecasts = truth.shape[0]

    time_windows = [
        (0, 3, "t+0-3h"),
        (3, 6, "t+3-6h"),
        (6, 9, "t+6-9h"),
        (9, 12, "t+9-12h"),
    ]

    results = []

    print(f"\nComputing genesis/lysis metrics WITH stddev constraint...")
    print(f"  Forecasts: {n_forecasts}")
    print(f"  Windows: {len(time_windows)} non-overlapping 3h windows")
    print(f"  Total samples: {n_forecasts * len(time_windows)}")
    print(f"  Patch size: {patch_size} cells = {patch_size*5} km")
    print(f"  Change threshold: ±{change_threshold}")
    print(f"  Genesis: mean < {clear_threshold} AND stddev < {std_threshold}")
    print(f"  Lysis: mean > {cloudy_threshold} AND stddev < {std_threshold}\n")

    for forecast_idx in tqdm(range(n_forecasts), desc="Processing forecasts"):
        for t0, t1, window_label in time_windows:
            # Extract fields
            tcc_t0 = truth[forecast_idx, t0].numpy()
            tcc_t1 = truth[forecast_idx, t1].numpy()
            cc1_t1 = cc1_pred[forecast_idx, t1].numpy()
            cc2_t1 = cc2_pred[forecast_idx, t1].numpy()

            # Compute patch stats
            obs_changes, obs_mean, obs_std, patch_grid = compute_patch_stats(
                tcc_t0, tcc_t1, patch_size
            )
            cc1_changes, _, _, _ = compute_patch_stats(tcc_t0, cc1_t1, patch_size)
            cc2_changes, _, _, _ = compute_patch_stats(tcc_t0, cc2_t1, patch_size)

            # Classify with stddev constraint
            obs_classes, obs_genesis, obs_lysis = classify_patches_with_stddev(
                obs_changes, obs_mean, obs_std, change_threshold,
                clear_threshold, cloudy_threshold, std_threshold
            )

            n_genesis = obs_genesis.sum()
            n_lysis = obs_lysis.sum()
            n_neutral = (obs_classes == 0).sum()

            # Compute metrics
            cc1_metrics = compute_genesis_lysis_metrics_with_stddev(
                obs_changes, obs_mean, obs_std, cc1_changes,
                change_threshold, clear_threshold, cloudy_threshold, std_threshold
            )
            cc2_metrics = compute_genesis_lysis_metrics_with_stddev(
                obs_changes, obs_mean, obs_std, cc2_changes,
                change_threshold, clear_threshold, cloudy_threshold, std_threshold
            )

            # Store results
            result = {
                'forecast_idx': forecast_idx,
                'time_window': window_label,
                't0': t0,
                't1': t1,
                'n_patches': obs_changes.size,
                'n_genesis': n_genesis,
                'n_lysis': n_lysis,
                'n_neutral': n_neutral,
                'genesis_fraction': n_genesis / obs_changes.size,
                'lysis_fraction': n_lysis / obs_changes.size,
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
            }

            results.append(result)

    return results


def print_summary_statistics(df):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("GENESIS/LYSIS WITH STDDEV CONSTRAINT - FULL ANALYSIS SUMMARY")
    print("="*80)

    print("\n" + "="*80)
    print("OVERALL STATISTICS (all time windows)")
    print("="*80)

    print(f"\nTotal samples: {len(df)} (731 forecasts × 4 windows)")
    print(f"Average genesis events per sample: {df['n_genesis'].mean():.1f} ± {df['n_genesis'].std():.1f}")
    print(f"Average lysis events per sample: {df['n_lysis'].mean():.1f} ± {df['n_lysis'].std():.1f}")
    print(f"Average genesis fraction: {df['genesis_fraction'].mean()*100:.1f}% ± {df['genesis_fraction'].std()*100:.1f}%")
    print(f"Average lysis fraction: {df['lysis_fraction'].mean()*100:.1f}% ± {df['lysis_fraction'].std()*100:.1f}%")

    print("\nCC1 Performance (mean ± std):")
    print(f"  Correlation: {df['cc1_correlation'].mean():.4f} ± {df['cc1_correlation'].std():.4f}")
    print(f"  RMSE: {df['cc1_rmse'].mean():.4f} ± {df['cc1_rmse'].std():.4f}")
    print(f"  Genesis CSI: {df['cc1_genesis_csi'].mean():.4f} ± {df['cc1_genesis_csi'].std():.4f}")
    print(f"  Lysis CSI: {df['cc1_lysis_csi'].mean():.4f} ± {df['cc1_lysis_csi'].std():.4f}")

    print("\nCC2 Performance (mean ± std):")
    print(f"  Correlation: {df['cc2_correlation'].mean():.4f} ± {df['cc2_correlation'].std():.4f}")
    print(f"  RMSE: {df['cc2_rmse'].mean():.4f} ± {df['cc2_rmse'].std():.4f}")
    print(f"  Genesis CSI: {df['cc2_genesis_csi'].mean():.4f} ± {df['cc2_genesis_csi'].std():.4f}")
    print(f"  Lysis CSI: {df['cc2_lysis_csi'].mean():.4f} ± {df['cc2_lysis_csi'].std():.4f}")

    print("\nCC2 vs CC1 Comparison (overall):")
    corr_diff = df['cc2_correlation'] - df['cc1_correlation']
    genesis_diff = df['cc2_genesis_csi'] - df['cc1_genesis_csi']
    lysis_diff = df['cc2_lysis_csi'] - df['cc1_lysis_csi']

    print(f"  Correlation difference: {corr_diff.mean():.4f} ± {corr_diff.std():.4f}")
    print(f"    CC2 wins: {(corr_diff > 0).sum()} / {len(df)} ({(corr_diff > 0).sum()/len(df)*100:.1f}%)")

    print(f"  Genesis CSI difference: {genesis_diff.mean():.4f} ± {genesis_diff.std():.4f}")
    print(f"    CC2 wins: {(genesis_diff > 0).sum()} / {len(df)} ({(genesis_diff > 0).sum()/len(df)*100:.1f}%)")
    rel_gen_imp = (df['cc2_genesis_csi'].mean() / df['cc1_genesis_csi'].mean() - 1) * 100
    print(f"    Relative improvement: {rel_gen_imp:+.1f}%")

    print(f"  Lysis CSI difference: {lysis_diff.mean():.4f} ± {lysis_diff.std():.4f}")
    print(f"    CC2 wins: {(lysis_diff > 0).sum()} / {len(df)} ({(lysis_diff > 0).sum()/len(df)*100:.1f}%)")
    rel_lys_imp = (df['cc2_lysis_csi'].mean() / df['cc1_lysis_csi'].mean() - 1) * 100
    print(f"    Relative improvement: {rel_lys_imp:+.1f}%")

    # Statistical significance
    from scipy import stats

    print("\nStatistical Significance (paired t-test, all samples):")
    t_corr, p_corr = stats.ttest_rel(df['cc2_correlation'], df['cc1_correlation'])
    print(f"  Correlation: t={t_corr:.3f}, p={p_corr:.4e} {'***' if p_corr < 0.001 else '**' if p_corr < 0.01 else '*' if p_corr < 0.05 else 'ns'}")

    t_genesis, p_genesis = stats.ttest_rel(df['cc2_genesis_csi'], df['cc1_genesis_csi'])
    print(f"  Genesis CSI: t={t_genesis:.3f}, p={p_genesis:.4e} {'***' if p_genesis < 0.001 else '**' if p_genesis < 0.01 else '*' if p_genesis < 0.05 else 'ns'}")

    t_lysis, p_lysis = stats.ttest_rel(df['cc2_lysis_csi'], df['cc1_lysis_csi'])
    print(f"  Lysis CSI: t={t_lysis:.3f}, p={p_lysis:.4e} {'***' if p_lysis < 0.001 else '**' if p_lysis < 0.01 else '*' if p_lysis < 0.05 else 'ns'}")

    # By time window
    print("\n" + "="*80)
    print("PERFORMANCE BY TIME WINDOW")
    print("="*80)

    for window in df['time_window'].unique():
        df_window = df[df['time_window'] == window]

        print(f"\n{window}:")
        print(f"  Samples: {len(df_window)}")
        print(f"  Avg events: Genesis={df_window['n_genesis'].mean():.1f}, Lysis={df_window['n_lysis'].mean():.1f}")
        print(f"  CC1: Gen_CSI={df_window['cc1_genesis_csi'].mean():.4f}, Lys_CSI={df_window['cc1_lysis_csi'].mean():.4f}")
        print(f"  CC2: Gen_CSI={df_window['cc2_genesis_csi'].mean():.4f}, Lys_CSI={df_window['cc2_lysis_csi'].mean():.4f}")

        gen_imp = (df_window['cc2_genesis_csi'].mean() / df_window['cc1_genesis_csi'].mean() - 1) * 100
        lys_imp = (df_window['cc2_lysis_csi'].mean() / df_window['cc1_lysis_csi'].mean() - 1) * 100
        print(f"  CC2 improvement: Genesis={gen_imp:+.1f}%, Lysis={lys_imp:+.1f}%")

    print("\n" + "="*80)


def create_summary_plots(df, output_dir):
    """Create summary visualization plots."""

    fig = plt.figure(figsize=(18, 12))

    # Plot 1: Genesis CSI by time window
    ax1 = plt.subplot(2, 3, 1)
    windows = ['t+0-3h', 't+3-6h', 't+6-9h', 't+9-12h']
    x = np.arange(len(windows))
    cc1_gen = [df[df['time_window']==w]['cc1_genesis_csi'].mean() for w in windows]
    cc2_gen = [df[df['time_window']==w]['cc2_genesis_csi'].mean() for w in windows]

    width = 0.35
    ax1.bar(x - width/2, cc1_gen, width, label='CC1', color='lightgray', edgecolor='black')
    ax1.bar(x + width/2, cc2_gen, width, label='CC2', color='coral', edgecolor='black')
    ax1.set_xlabel('Time Window', fontsize=11)
    ax1.set_ylabel('Genesis CSI', fontsize=11)
    ax1.set_title('Genesis CSI by Time Window', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(windows)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Lysis CSI by time window
    ax2 = plt.subplot(2, 3, 2)
    cc1_lys = [df[df['time_window']==w]['cc1_lysis_csi'].mean() for w in windows]
    cc2_lys = [df[df['time_window']==w]['cc2_lysis_csi'].mean() for w in windows]

    ax2.bar(x - width/2, cc1_lys, width, label='CC1', color='lightgray', edgecolor='black')
    ax2.bar(x + width/2, cc2_lys, width, label='CC2', color='lightgreen', edgecolor='black')
    ax2.set_xlabel('Time Window', fontsize=11)
    ax2.set_ylabel('Lysis CSI', fontsize=11)
    ax2.set_title('Lysis CSI by Time Window', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(windows)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: CC2 advantage by time window
    ax3 = plt.subplot(2, 3, 3)
    gen_adv = [(cc2_gen[i] / cc1_gen[i] - 1) * 100 for i in range(len(windows))]
    lys_adv = [(cc2_lys[i] / cc1_lys[i] - 1) * 100 for i in range(len(windows))]

    ax3.bar(x - width/2, gen_adv, width, label='Genesis', color='coral', edgecolor='black')
    ax3.bar(x + width/2, lys_adv, width, label='Lysis', color='lightgreen', edgecolor='black')
    ax3.axhline(0, color='black', linestyle='--', linewidth=1)
    ax3.set_xlabel('Time Window', fontsize=11)
    ax3.set_ylabel('CC2 Improvement (%)', fontsize=11)
    ax3.set_title('CC2 Relative Improvement', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(windows)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Distribution of genesis CSI differences
    ax4 = plt.subplot(2, 3, 4)
    genesis_diff = df['cc2_genesis_csi'] - df['cc1_genesis_csi']
    ax4.hist(genesis_diff, bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
    ax4.axvline(genesis_diff.mean(), color='darkred', linestyle='-', linewidth=2,
                label=f'Mean = {genesis_diff.mean():.4f}')
    ax4.set_xlabel('CC2 - CC1 Genesis CSI', fontsize=11)
    ax4.set_ylabel('Number of Samples', fontsize=11)
    ax4.set_title('Distribution of Genesis CSI Differences', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Distribution of lysis CSI differences
    ax5 = plt.subplot(2, 3, 5)
    lysis_diff = df['cc2_lysis_csi'] - df['cc1_lysis_csi']
    ax5.hist(lysis_diff, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    ax5.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
    ax5.axvline(lysis_diff.mean(), color='darkgreen', linestyle='-', linewidth=2,
                label=f'Mean = {lysis_diff.mean():.4f}')
    ax5.set_xlabel('CC2 - CC1 Lysis CSI', fontsize=11)
    ax5.set_ylabel('Number of Samples', fontsize=11)
    ax5.set_title('Distribution of Lysis CSI Differences', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Scatter CC1 vs CC2 genesis CSI
    ax6 = plt.subplot(2, 3, 6)
    ax6.scatter(df['cc1_genesis_csi'], df['cc2_genesis_csi'], alpha=0.3, s=10, color='coral')
    lim_max = max(df['cc1_genesis_csi'].max(), df['cc2_genesis_csi'].max())
    lim_min = min(df['cc1_genesis_csi'].min(), df['cc2_genesis_csi'].min())
    ax6.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2, label='Equal performance')
    ax6.set_xlabel('CC1 Genesis CSI', fontsize=11)
    ax6.set_ylabel('CC2 Genesis CSI', fontsize=11)
    ax6.set_title('Genesis CSI: CC1 vs CC2\n(Above line = CC2 better)', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_aspect('equal')

    plt.suptitle('Genesis/Lysis Metric WITH Stddev Constraint: Full Analysis (2,924 samples)\n' +
                 'Threshold: 0.25, Genesis: mean<0.2 & std<0.15, Lysis: mean>0.8 & std<0.15',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_dir / 'genesis_lysis_with_stddev_full_analysis.png', dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved: genesis_lysis_with_stddev_full_analysis.png")
    plt.close()


def main():
    print("="*80)
    print("GENESIS/LYSIS WITH STDDEV CONSTRAINT - FULL ANALYSIS")
    print("="*80)

    output_dir = Path("genesis_lysis_plots")
    output_dir.mkdir(exist_ok=True)

    # Load data
    cc1_pred, cc2_pred, truth = load_forecast_data_12h()

    # Final parameters
    patch_size = 10
    change_threshold = 0.25
    clear_threshold = 0.2
    cloudy_threshold = 0.8
    std_threshold = 0.15

    # Compute metrics
    results = compute_all_forecasts_with_stddev(
        cc1_pred, cc2_pred, truth,
        patch_size=patch_size,
        change_threshold=change_threshold,
        clear_threshold=clear_threshold,
        cloudy_threshold=cloudy_threshold,
        std_threshold=std_threshold
    )

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save results
    csv_path = output_dir / 'genesis_lysis_with_stddev_full_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved: {csv_path.name}")

    # Print summary
    print_summary_statistics(df)

    # Create visualizations
    print("\nCreating summary plots...")
    create_summary_plots(df, output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - genesis_lysis_with_stddev_full_results.csv")
    print(f"  - genesis_lysis_with_stddev_full_analysis.png")


if __name__ == "__main__":
    main()
