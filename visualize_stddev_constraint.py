#!/usr/bin/env python3
"""
Visualize the effect of stddev constraint on genesis/lysis detection.

Shows how the constraint filters out cloud edges and retains only
homogeneous cores.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from genesis_lysis_metric_with_stddev import (
    load_forecast_data_12h,
    compute_patch_stats,
    classify_patches_with_stddev
)


def visualize_stddev_effect(forecast_idx, t0, t1, patch_size,
                            change_threshold, clear_threshold, cloudy_threshold,
                            std_threshold, output_dir):
    """
    Show how stddev constraint filters cloud edges.

    Creates 2-row figure:
    Row 1: Without stddev constraint
    Row 2: With stddev constraint
    """
    # Load data
    cc1_pred, cc2_pred, truth = load_forecast_data_12h()

    # Extract fields
    tcc_t0 = truth[forecast_idx, t0].numpy()
    tcc_t1 = truth[forecast_idx, t1].numpy()
    cc1_t1 = cc1_pred[forecast_idx, t1].numpy()
    cc2_t1 = cc2_pred[forecast_idx, t1].numpy()

    # Compute patch stats
    obs_changes, obs_mean, obs_std, patch_grid = compute_patch_stats(tcc_t0, tcc_t1, patch_size)
    cc1_changes, _, _, _ = compute_patch_stats(tcc_t0, cc1_t1, patch_size)
    cc2_changes, _, _, _ = compute_patch_stats(tcc_t0, cc2_t1, patch_size)

    y_starts, x_starts = patch_grid

    # Classify WITHOUT stddev constraint (std_threshold = 1.0)
    _, obs_genesis_no_std, obs_lysis_no_std = classify_patches_with_stddev(
        obs_changes, obs_mean, obs_std, change_threshold,
        clear_threshold, cloudy_threshold, std_threshold=1.0
    )
    _, cc1_genesis_no_std, cc1_lysis_no_std = classify_patches_with_stddev(
        cc1_changes, obs_mean, obs_std, change_threshold,
        clear_threshold, cloudy_threshold, std_threshold=1.0
    )
    _, cc2_genesis_no_std, cc2_lysis_no_std = classify_patches_with_stddev(
        cc2_changes, obs_mean, obs_std, change_threshold,
        clear_threshold, cloudy_threshold, std_threshold=1.0
    )

    # Classify WITH stddev constraint
    _, obs_genesis_with_std, obs_lysis_with_std = classify_patches_with_stddev(
        obs_changes, obs_mean, obs_std, change_threshold,
        clear_threshold, cloudy_threshold, std_threshold
    )
    _, cc1_genesis_with_std, cc1_lysis_with_std = classify_patches_with_stddev(
        cc1_changes, obs_mean, obs_std, change_threshold,
        clear_threshold, cloudy_threshold, std_threshold
    )
    _, cc2_genesis_with_std, cc2_lysis_with_std = classify_patches_with_stddev(
        cc2_changes, obs_mean, obs_std, change_threshold,
        clear_threshold, cloudy_threshold, std_threshold
    )

    # Create figure
    fig = plt.figure(figsize=(22, 10))

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

    # Row 1: WITHOUT stddev constraint
    ax1 = plt.subplot(2, 5, 1)
    ax1.imshow(tcc_t0, cmap='gray', vmin=0, vmax=1)
    ax1.set_title(f'Initial TCC\nt={t0}h', fontsize=11, fontweight='bold')
    ax1.axis('off')

    ax2 = plt.subplot(2, 5, 2)
    overlay_patches(ax2, tcc_t1, obs_changes, obs_genesis_no_std, obs_lysis_no_std,
                   y_starts, x_starts, patch_size)
    ax2.set_title(f'Observed G/L\nNO stddev constraint\nG={obs_genesis_no_std.sum()}, L={obs_lysis_no_std.sum()}',
                 fontsize=11, fontweight='bold')
    ax2.axis('off')

    ax3 = plt.subplot(2, 5, 3)
    overlay_patches(ax3, cc1_t1, cc1_changes, cc1_genesis_no_std, cc1_lysis_no_std,
                   y_starts, x_starts, patch_size)
    cc1_gen_match = (obs_genesis_no_std & cc1_genesis_no_std).sum()
    cc1_lys_match = (obs_lysis_no_std & cc1_lysis_no_std).sum()
    ax3.set_title(f'CC1\nG={cc1_genesis_no_std.sum()} ({cc1_gen_match}✓)\nL={cc1_lysis_no_std.sum()} ({cc1_lys_match}✓)',
                 fontsize=11, fontweight='bold')
    ax3.axis('off')

    ax4 = plt.subplot(2, 5, 4)
    overlay_patches(ax4, cc2_t1, cc2_changes, cc2_genesis_no_std, cc2_lysis_no_std,
                   y_starts, x_starts, patch_size)
    cc2_gen_match = (obs_genesis_no_std & cc2_genesis_no_std).sum()
    cc2_lys_match = (obs_lysis_no_std & cc2_lysis_no_std).sum()
    ax4.set_title(f'CC2\nG={cc2_genesis_no_std.sum()} ({cc2_gen_match}✓)\nL={cc2_lysis_no_std.sum()} ({cc2_lys_match}✓)',
                 fontsize=11, fontweight='bold', color='green')
    ax4.axis('off')

    ax5 = plt.subplot(2, 5, 5)
    im5 = ax5.imshow(obs_std, cmap='YlOrRd', vmin=0, vmax=0.3)
    ax5.contour(x_starts + patch_size/2, y_starts + patch_size/2,
               (obs_std >= std_threshold).astype(float), levels=[0.5],
               colors='black', linewidths=2)
    ax5.set_title(f'Initial TCC Stddev\n(black = filtered, std≥{std_threshold})',
                 fontsize=11, fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)

    # Row 2: WITH stddev constraint
    ax6 = plt.subplot(2, 5, 6)
    ax6.imshow(tcc_t1, cmap='gray', vmin=0, vmax=1)
    ax6.set_title(f'Final TCC\nt={t1}h', fontsize=11, fontweight='bold')
    ax6.axis('off')

    ax7 = plt.subplot(2, 5, 7)
    overlay_patches(ax7, tcc_t1, obs_changes, obs_genesis_with_std, obs_lysis_with_std,
                   y_starts, x_starts, patch_size)
    ax7.set_title(f'Observed G/L\nWITH stddev<{std_threshold}\nG={obs_genesis_with_std.sum()}, L={obs_lysis_with_std.sum()}',
                 fontsize=11, fontweight='bold', color='purple')
    ax7.axis('off')

    ax8 = plt.subplot(2, 5, 8)
    overlay_patches(ax8, cc1_t1, cc1_changes, cc1_genesis_with_std, cc1_lysis_with_std,
                   y_starts, x_starts, patch_size)
    cc1_gen_match_std = (obs_genesis_with_std & cc1_genesis_with_std).sum()
    cc1_lys_match_std = (obs_lysis_with_std & cc1_lysis_with_std).sum()
    ax8.set_title(f'CC1\nG={cc1_genesis_with_std.sum()} ({cc1_gen_match_std}✓)\nL={cc1_lysis_with_std.sum()} ({cc1_lys_match_std}✓)',
                 fontsize=11, fontweight='bold', color='purple')
    ax8.axis('off')

    ax9 = plt.subplot(2, 5, 9)
    overlay_patches(ax9, cc2_t1, cc2_changes, cc2_genesis_with_std, cc2_lysis_with_std,
                   y_starts, x_starts, patch_size)
    cc2_gen_match_std = (obs_genesis_with_std & cc2_genesis_with_std).sum()
    cc2_lys_match_std = (obs_lysis_with_std & cc2_lysis_with_std).sum()
    ax9.set_title(f'CC2\nG={cc2_genesis_with_std.sum()} ({cc2_gen_match_std}✓)\nL={cc2_lysis_with_std.sum()} ({cc2_lys_match_std}✓)',
                 fontsize=11, fontweight='bold', color='darkgreen')
    ax9.axis('off')

    ax10 = plt.subplot(2, 5, 10)
    # Show what was filtered out
    filtered_genesis = obs_genesis_no_std & ~obs_genesis_with_std
    filtered_lysis = obs_lysis_no_std & ~obs_lysis_with_std
    ax10.imshow(tcc_t1, cmap='gray', vmin=0, vmax=1, alpha=0.6)
    for i, y in enumerate(y_starts):
        for j, x in enumerate(x_starts):
            if filtered_genesis[i, j]:
                rect = mpatches.Rectangle((x, y), patch_size, patch_size,
                                         linewidth=2, edgecolor='orange',
                                         facecolor='orange', alpha=0.5)
                ax10.add_patch(rect)
            elif filtered_lysis[i, j]:
                rect = mpatches.Rectangle((x, y), patch_size, patch_size,
                                         linewidth=2, edgecolor='cyan',
                                         facecolor='cyan', alpha=0.5)
                ax10.add_patch(rect)
    ax10.set_title(f'Filtered Out\n(cloud edges)\nG={filtered_genesis.sum()}, L={filtered_lysis.sum()}',
                 fontsize=11, fontweight='bold', color='orange')
    ax10.axis('off')
    ax10.set_xlim([0, tcc_t1.shape[1]])
    ax10.set_ylim([tcc_t1.shape[0], 0])

    # Legend
    genesis_patch = mpatches.Patch(color='red', alpha=0.6, label='Genesis')
    lysis_patch = mpatches.Patch(color='blue', alpha=0.6, label='Lysis')
    filtered_patch = mpatches.Patch(color='orange', alpha=0.5, label='Filtered (edges)')
    fig.legend(handles=[genesis_patch, lysis_patch, filtered_patch],
              loc='lower center', ncol=3, fontsize=11, frameon=True)

    plt.suptitle(f'Effect of Stddev Constraint on Genesis/Lysis Detection - Forecast {forecast_idx}\n' +
                 f'Top: NO constraint (includes cloud edges) | Bottom: WITH stddev<{std_threshold} (homogeneous cores only)',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    filename = output_dir / f'stddev_constraint_effect_f{forecast_idx}.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {filename.name}")
    plt.close()


def create_multi_forecast_stddev_comparison(forecast_indices, t0, t1, patch_size,
                                           change_threshold, clear_threshold,
                                           cloudy_threshold, std_threshold, output_dir):
    """Show multiple forecasts with stddev constraint."""
    # Load data
    print("\nCreating multi-forecast comparison...")
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

        # Compute patch stats
        obs_changes, obs_mean, obs_std, patch_grid = compute_patch_stats(tcc_t0, tcc_t1, patch_size)
        cc1_changes, _, _, _ = compute_patch_stats(tcc_t0, cc1_t1, patch_size)
        cc2_changes, _, _, _ = compute_patch_stats(tcc_t0, cc2_t1, patch_size)

        y_starts, x_starts = patch_grid

        # Classify with stddev constraint
        _, obs_genesis, obs_lysis = classify_patches_with_stddev(
            obs_changes, obs_mean, obs_std, change_threshold,
            clear_threshold, cloudy_threshold, std_threshold
        )
        _, cc1_genesis, cc1_lysis = classify_patches_with_stddev(
            cc1_changes, obs_mean, obs_std, change_threshold,
            clear_threshold, cloudy_threshold, std_threshold
        )
        _, cc2_genesis, cc2_lysis = classify_patches_with_stddev(
            cc2_changes, obs_mean, obs_std, change_threshold,
            clear_threshold, cloudy_threshold, std_threshold
        )

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

        # Ground truth
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
        gen_match = (obs_genesis & cc1_genesis).sum()
        lys_match = (obs_lysis & cc1_lysis).sum()
        if row_idx == 0:
            ax2.set_title(f'CC1\nG={cc1_genesis.sum()} ({gen_match}✓), L={cc1_lysis.sum()} ({lys_match}✓)',
                        fontsize=11, fontweight='bold')
        else:
            ax2.set_title(f'G={cc1_genesis.sum()} ({gen_match}✓), L={cc1_lysis.sum()} ({lys_match}✓)',
                        fontsize=10)
        ax2.axis('off')

        # CC2
        ax3 = plt.subplot(n_forecasts, 3, row_idx * 3 + 3)
        overlay_patches(ax3, cc2_t1, cc2_changes, cc2_genesis, cc2_lysis,
                       y_starts, x_starts, patch_size)
        gen_match_cc2 = (obs_genesis & cc2_genesis).sum()
        lys_match_cc2 = (obs_lysis & cc2_lysis).sum()
        if row_idx == 0:
            ax3.set_title(f'CC2\nG={cc2_genesis.sum()} ({gen_match_cc2}✓), L={cc2_lysis.sum()} ({lys_match_cc2}✓)',
                        fontsize=11, fontweight='bold', color='green')
        else:
            ax3.set_title(f'G={cc2_genesis.sum()} ({gen_match_cc2}✓), L={cc2_lysis.sum()} ({lys_match_cc2}✓)',
                        fontsize=10, color='green')
        ax3.axis('off')

    # Legend
    genesis_patch = mpatches.Patch(color='red', alpha=0.6, label='Genesis')
    lysis_patch = mpatches.Patch(color='blue', alpha=0.6, label='Lysis')
    fig.legend(handles=[genesis_patch, lysis_patch], loc='lower center',
              ncol=2, fontsize=11, frameon=True)

    plt.suptitle(f'Genesis/Lysis with Stddev Constraint (std<{std_threshold})\n' +
                 f'Only homogeneous cores: clear-core (mean<{clear_threshold}, std<{std_threshold}) and cloudy-core (mean>{cloudy_threshold}, std<{std_threshold})',
                 fontsize=13, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0.02, 1, 0.99])

    filename = output_dir / f'stddev_constraint_multi_forecast_std{std_threshold:.2f}.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved: {filename.name}")
    plt.close()


def main():
    print("="*80)
    print("VISUALIZE STDDEV CONSTRAINT EFFECT")
    print("="*80)

    output_dir = Path("genesis_lysis_plots")
    output_dir.mkdir(exist_ok=True)

    # Parameters
    patch_size = 10
    change_threshold = 0.25
    clear_threshold = 0.2
    cloudy_threshold = 0.8
    std_threshold = 0.15  # Recommended

    print(f"\nConfiguration:")
    print(f"  Patch size: {patch_size} cells = {patch_size*5}km")
    print(f"  Change threshold: ±{change_threshold}")
    print(f"  Clear threshold: < {clear_threshold}")
    print(f"  Cloudy threshold: > {cloudy_threshold}")
    print(f"  Stddev threshold: < {std_threshold}")

    # Detailed view for one forecast
    print(f"\nCreating detailed comparison for forecast 365...")
    visualize_stddev_effect(
        365, 0, 3, patch_size, change_threshold,
        clear_threshold, cloudy_threshold, std_threshold, output_dir
    )

    # Multi-forecast overview
    forecast_indices = [50, 150, 250, 350, 450, 550, 650]
    create_multi_forecast_stddev_comparison(
        forecast_indices, 0, 3, patch_size, change_threshold,
        clear_threshold, cloudy_threshold, std_threshold, output_dir
    )

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
