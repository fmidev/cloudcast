#!/usr/bin/env python3
"""
Visualize CC1 vs CC2 predictions with genesis/lysis overlays.
Uses pixel counting method (≥85% pixels clearly clear/cloudy).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from genesis_lysis_metric_pixel_count import (
    load_forecast_data_12h,
    compute_patch_stats_with_pixel_counts,
    classify_patches_pixel_count
)


def create_prediction_comparison(forecast_indices, time_windows, patch_size,
                                 change_threshold, clear_threshold, cloudy_threshold,
                                 core_threshold, output_dir):
    """
    Create visualizations comparing CC1 vs CC2 predictions with G/L overlays.

    For each forecast and time window, shows:
    - Ground truth with observed G/L
    - CC1 prediction with detected G/L
    - CC2 prediction with detected G/L
    """
    # Load data
    print("Loading forecast data...")
    cc1_pred, cc2_pred, truth = load_forecast_data_12h()

    for forecast_idx in forecast_indices:
        print(f"\nProcessing forecast {forecast_idx}...")

        # Create figure: len(time_windows) rows × 3 columns
        n_windows = len(time_windows)
        fig = plt.figure(figsize=(18, 5 * n_windows))

        for row_idx, (t0, t1, window_label) in enumerate(time_windows):
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

            y_starts, x_starts = patch_grid

            # Classify patches using pixel counting
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

            # Helper function to overlay patches
            def overlay_patches(ax, base_field, changes, genesis_mask, lysis_mask,
                              y_starts, x_starts, patch_size):
                ax.imshow(base_field, cmap='gray', vmin=0, vmax=1, alpha=0.6)
                for i, y in enumerate(y_starts):
                    for j, x in enumerate(x_starts):
                        if genesis_mask[i, j]:
                            delta = changes[i, j]
                            alpha = min(delta / 0.5, 1.0) * 0.7
                            rect = mpatches.Rectangle((x, y), patch_size, patch_size,
                                                     linewidth=2, edgecolor='red',
                                                     facecolor='red', alpha=alpha)
                            ax.add_patch(rect)
                        elif lysis_mask[i, j]:
                            delta = abs(changes[i, j])
                            alpha = min(delta / 0.5, 1.0) * 0.7
                            rect = mpatches.Rectangle((x, y), patch_size, patch_size,
                                                     linewidth=2, edgecolor='blue',
                                                     facecolor='blue', alpha=alpha)
                            ax.add_patch(rect)
                ax.set_xlim([0, base_field.shape[1]])
                ax.set_ylim([base_field.shape[0], 0])

            # Column 1: Ground Truth with Observed G/L
            ax1 = plt.subplot(n_windows, 3, row_idx * 3 + 1)
            overlay_patches(ax1, tcc_t1, obs_changes, obs_genesis, obs_lysis,
                          y_starts, x_starts, patch_size)
            if row_idx == 0:
                ax1.set_title(f'Ground Truth\n{window_label}\nG={obs_genesis.sum()}, L={obs_lysis.sum()}',
                            fontsize=12, fontweight='bold')
            else:
                ax1.set_title(f'{window_label}\nG={obs_genesis.sum()}, L={obs_lysis.sum()}',
                            fontsize=11)
            ax1.axis('off')

            # Column 2: CC1 Prediction with Detected G/L
            ax2 = plt.subplot(n_windows, 3, row_idx * 3 + 2)
            overlay_patches(ax2, cc1_t1, cc1_changes, cc1_genesis, cc1_lysis,
                          y_starts, x_starts, patch_size)

            # Compute matches/misses
            genesis_match = (obs_genesis & cc1_genesis).sum()
            lysis_match = (obs_lysis & cc1_lysis).sum()

            if row_idx == 0:
                ax2.set_title(f'CC1 Prediction\n{window_label}\nG={cc1_genesis.sum()} (match={genesis_match}), L={cc1_lysis.sum()} (match={lysis_match})',
                            fontsize=12, fontweight='bold')
            else:
                ax2.set_title(f'G={cc1_genesis.sum()} (match={genesis_match}), L={cc1_lysis.sum()} (match={lysis_match})',
                            fontsize=11)
            ax2.axis('off')

            # Column 3: CC2 Prediction with Detected G/L
            ax3 = plt.subplot(n_windows, 3, row_idx * 3 + 3)
            overlay_patches(ax3, cc2_t1, cc2_changes, cc2_genesis, cc2_lysis,
                          y_starts, x_starts, patch_size)

            # Compute matches/misses
            genesis_match_cc2 = (obs_genesis & cc2_genesis).sum()
            lysis_match_cc2 = (obs_lysis & cc2_lysis).sum()

            if row_idx == 0:
                ax3.set_title(f'CC2 Prediction\n{window_label}\nG={cc2_genesis.sum()} (match={genesis_match_cc2}), L={cc2_lysis.sum()} (match={lysis_match_cc2})',
                            fontsize=12, fontweight='bold', color='green')
            else:
                ax3.set_title(f'G={cc2_genesis.sum()} (match={genesis_match_cc2}), L={cc2_lysis.sum()} (match={lysis_match_cc2})',
                            fontsize=11, color='green')
            ax3.axis('off')

        # Legend
        genesis_patch = mpatches.Patch(color='red', alpha=0.6, label='Genesis (cloud forming)')
        lysis_patch = mpatches.Patch(color='blue', alpha=0.6, label='Lysis (cloud dissipating)')
        fig.legend(handles=[genesis_patch, lysis_patch], loc='lower center',
                  ncol=2, fontsize=12, frameon=True)

        # Overall title
        plt.suptitle(f'Genesis/Lysis Detection: CC1 vs CC2 - Forecast {forecast_idx}\n' +
                     f'Pixel counting method (≥{core_threshold:.0%} pixels clearly clear/cloudy, patch={patch_size*5}km)',
                     fontsize=13, fontweight='bold', y=0.995)

        plt.tight_layout(rect=[0, 0.02, 1, 0.99])

        filename = output_dir / f'pixel_count_predictions_f{forecast_idx}.png'
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        print(f"  ✓ Saved: {filename.name}")
        plt.close()


def create_multi_forecast_overview(forecast_indices, t0, t1, patch_size,
                                   change_threshold, clear_threshold, cloudy_threshold,
                                   core_threshold, output_dir):
    """
    Create overview showing multiple forecasts in one figure.

    Rows: different forecasts
    Columns: Ground truth, CC1, CC2
    """
    # Load data
    print("\nCreating multi-forecast overview...")
    cc1_pred, cc2_pred, truth = load_forecast_data_12h()

    n_forecasts = len(forecast_indices)
    fig = plt.figure(figsize=(18, 5 * n_forecasts))

    for row_idx, forecast_idx in enumerate(forecast_indices):
        print(f"  Processing forecast {forecast_idx}...")

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

        y_starts, x_starts = patch_grid

        # Classify
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

        # Helper function
        def overlay_patches(ax, base_field, changes, genesis_mask, lysis_mask,
                          y_starts, x_starts, patch_size):
            ax.imshow(base_field, cmap='gray', vmin=0, vmax=1, alpha=0.6)
            for i, y in enumerate(y_starts):
                for j, x in enumerate(x_starts):
                    if genesis_mask[i, j]:
                        delta = changes[i, j]
                        alpha = min(delta / 0.5, 1.0) * 0.7
                        rect = mpatches.Rectangle((x, y), patch_size, patch_size,
                                                 linewidth=2, edgecolor='red',
                                                 facecolor='red', alpha=alpha)
                        ax.add_patch(rect)
                    elif lysis_mask[i, j]:
                        delta = abs(changes[i, j])
                        alpha = min(delta / 0.5, 1.0) * 0.7
                        rect = mpatches.Rectangle((x, y), patch_size, patch_size,
                                                 linewidth=2, edgecolor='blue',
                                                 facecolor='blue', alpha=alpha)
                        ax.add_patch(rect)
            ax.set_xlim([0, base_field.shape[1]])
            ax.set_ylim([base_field.shape[0], 0])

        # Ground Truth
        ax1 = plt.subplot(n_forecasts, 3, row_idx * 3 + 1)
        overlay_patches(ax1, tcc_t1, obs_changes, obs_genesis, obs_lysis,
                      y_starts, x_starts, patch_size)
        if row_idx == 0:
            ax1.set_title(f'Ground Truth\n(Forecast {forecast_idx})\nG={obs_genesis.sum()}, L={obs_lysis.sum()}',
                        fontsize=11, fontweight='bold')
        else:
            ax1.set_title(f'Forecast {forecast_idx}\nG={obs_genesis.sum()}, L={obs_lysis.sum()}',
                        fontsize=10)
        ax1.axis('off')

        # CC1
        ax2 = plt.subplot(n_forecasts, 3, row_idx * 3 + 2)
        overlay_patches(ax2, cc1_t1, cc1_changes, cc1_genesis, cc1_lysis,
                      y_starts, x_starts, patch_size)
        genesis_match = (obs_genesis & cc1_genesis).sum()
        lysis_match = (obs_lysis & cc1_lysis).sum()
        if row_idx == 0:
            ax2.set_title(f'CC1\nG={cc1_genesis.sum()} ({genesis_match}✓), L={cc1_lysis.sum()} ({lysis_match}✓)',
                        fontsize=11, fontweight='bold')
        else:
            ax2.set_title(f'G={cc1_genesis.sum()} ({genesis_match}✓), L={cc1_lysis.sum()} ({lysis_match}✓)',
                        fontsize=10)
        ax2.axis('off')

        # CC2
        ax3 = plt.subplot(n_forecasts, 3, row_idx * 3 + 3)
        overlay_patches(ax3, cc2_t1, cc2_changes, cc2_genesis, cc2_lysis,
                      y_starts, x_starts, patch_size)
        genesis_match_cc2 = (obs_genesis & cc2_genesis).sum()
        lysis_match_cc2 = (obs_lysis & cc2_lysis).sum()
        if row_idx == 0:
            ax3.set_title(f'CC2\nG={cc2_genesis.sum()} ({genesis_match_cc2}✓), L={cc2_lysis.sum()} ({lysis_match_cc2}✓)',
                        fontsize=11, fontweight='bold', color='green')
        else:
            ax3.set_title(f'G={cc2_genesis.sum()} ({genesis_match_cc2}✓), L={cc2_lysis.sum()} ({lysis_match_cc2}✓)',
                        fontsize=10, color='green')
        ax3.axis('off')

    # Legend
    genesis_patch = mpatches.Patch(color='red', alpha=0.6, label='Genesis (cloud forming)')
    lysis_patch = mpatches.Patch(color='blue', alpha=0.6, label='Lysis (cloud dissipating)')
    fig.legend(handles=[genesis_patch, lysis_patch], loc='lower center',
              ncol=2, fontsize=11, frameon=True)

    plt.suptitle(f'Genesis/Lysis: CC1 vs CC2 (t={t0}h → t={t1}h, Pixel Count ≥{core_threshold:.0%})\n' +
                 f'Numbers in parentheses show hits (correct detections)',
                 fontsize=13, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0.02, 1, 0.99])

    filename = output_dir / f'pixel_count_multi_forecast_overview.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved: {filename.name}")
    plt.close()


def main():
    print("="*80)
    print("VISUALIZE CC1 VS CC2 PREDICTIONS WITH PIXEL COUNTING")
    print("="*80)

    # Create output directory
    output_dir = Path("genesis_lysis_plots")
    output_dir.mkdir(exist_ok=True)

    # Parameters
    patch_size = 10
    change_threshold = 0.25
    clear_threshold = 0.2
    cloudy_threshold = 0.8
    core_threshold = 0.85  # ≥85% pixels must be clearly clear/cloudy

    print(f"\nConfiguration:")
    print(f"  Patch size: {patch_size} cells = {patch_size*5}km")
    print(f"  Change threshold: ±{change_threshold}")
    print(f"  Clear/cloudy thresholds: < {clear_threshold} / > {cloudy_threshold}")
    print(f"  Core threshold: ≥{core_threshold:.0%} pixels clearly clear/cloudy")

    # Create detailed comparisons for select forecasts with multiple time windows
    forecast_indices = [100, 250, 365]
    time_windows = [
        (0, 3, "t+0-3h"),
        (3, 6, "t+3-6h"),
        (6, 9, "t+6-9h"),
        (9, 12, "t+9-12h"),
    ]

    print(f"\nCreating detailed comparisons for forecasts: {forecast_indices}")
    create_prediction_comparison(
        forecast_indices, time_windows, patch_size,
        change_threshold, clear_threshold, cloudy_threshold,
        core_threshold, output_dir
    )

    # Create multi-forecast overview for one time window
    forecast_indices_overview = [50, 150, 250, 350, 450, 550, 650]
    print(f"\nCreating overview for forecasts: {forecast_indices_overview}")
    create_multi_forecast_overview(
        forecast_indices_overview, 0, 3, patch_size,
        change_threshold, clear_threshold, cloudy_threshold,
        core_threshold, output_dir
    )

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nGenerated files in {output_dir}/:")
    for idx in forecast_indices:
        print(f"  - pixel_count_predictions_f{idx}.png")
    print(f"  - pixel_count_multi_forecast_overview.png")


if __name__ == "__main__":
    main()
