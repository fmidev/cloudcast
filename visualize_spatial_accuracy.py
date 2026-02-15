#!/usr/bin/env python3
"""
Visualize spatial accuracy of genesis/lysis predictions.

Shows:
1. Spatial error maps (hits, misses, false alarms, near-misses)
2. Displacement vector fields
3. Distance histograms
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.spatial.distance import cdist

from genesis_lysis_metric_pixel_count import (
    load_forecast_data_12h,
    compute_patch_stats_with_pixel_counts,
    classify_patches_pixel_count
)


def create_spatial_error_map(forecast_idx, t0, t1, spatial_tolerance_km=50,
                             patch_size=10, change_threshold=0.25,
                             clear_threshold=0.2, cloudy_threshold=0.8,
                             core_threshold=0.85, output_dir=None):
    """
    Create spatial error visualization showing hits, misses, false alarms, near-misses.
    """
    # Load data
    print(f"Loading forecast {forecast_idx}...")
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

    # Create coordinate grid
    y_starts, x_starts = patch_grid
    n_y, n_x = obs_genesis.shape
    patch_coords = np.zeros((n_y, n_x, 2))
    for i in range(n_y):
        for j in range(n_x):
            patch_coords[i, j] = [y_starts[i] * 5, x_starts[j] * 5]

    # Classify spatial errors for genesis
    def classify_spatial_errors(obs_events, pred_events, patch_coords, tolerance):
        """
        Classify each patch as: hit, miss, false_alarm, or near_miss.

        Returns error_map with values:
        0 = no event
        1 = exact hit
        2 = near miss (within tolerance)
        3 = miss (observed but no prediction nearby)
        4 = false alarm (predicted but no observation nearby)
        """
        error_map = np.zeros_like(obs_events, dtype=int)

        # Get event locations
        obs_locs = patch_coords[obs_events]
        pred_locs = patch_coords[pred_events]

        if len(obs_locs) == 0 and len(pred_locs) == 0:
            return error_map

        # Exact hits
        exact_hits = obs_events & pred_events
        error_map[exact_hits] = 1

        if len(obs_locs) > 0 and len(pred_locs) > 0:
            # Compute distances
            distances = cdist(patch_coords.reshape(-1, 2),
                            pred_locs, metric='euclidean').reshape(n_y, n_x, -1)
            min_dist_to_pred = distances.min(axis=2)

            # Near misses: observed events without exact match but with prediction nearby
            near_miss = obs_events & ~exact_hits & (min_dist_to_pred <= tolerance)
            error_map[near_miss] = 2

            # Misses: observed events without exact match and no prediction nearby
            miss = obs_events & ~exact_hits & (min_dist_to_pred > tolerance)
            error_map[miss] = 3

            # False alarms: predicted events not near any observation
            distances_to_obs = cdist(patch_coords.reshape(-1, 2),
                                    obs_locs, metric='euclidean').reshape(n_y, n_x, -1)
            min_dist_to_obs = distances_to_obs.min(axis=2)
            false_alarm = pred_events & ~exact_hits & (min_dist_to_obs > tolerance)
            error_map[false_alarm] = 4
        else:
            # All observed are misses if no predictions
            if len(obs_locs) > 0:
                error_map[obs_events] = 3
            # All predicted are false alarms if no observations
            if len(pred_locs) > 0:
                error_map[pred_events] = 4

        return error_map

    # Create figure
    fig = plt.figure(figsize=(20, 10))

    # Genesis error maps
    cc1_genesis_errors = classify_spatial_errors(obs_genesis, cc1_genesis, patch_coords, spatial_tolerance_km)
    cc2_genesis_errors = classify_spatial_errors(obs_genesis, cc2_genesis, patch_coords, spatial_tolerance_km)

    # Lysis error maps
    cc1_lysis_errors = classify_spatial_errors(obs_lysis, cc1_lysis, patch_coords, spatial_tolerance_km)
    cc2_lysis_errors = classify_spatial_errors(obs_lysis, cc2_lysis, patch_coords, spatial_tolerance_km)

    # Color map: 0=white, 1=green (hit), 2=yellow (near miss), 3=red (miss), 4=orange (FA)
    from matplotlib.colors import ListedColormap
    colors = ['white', 'green', 'yellow', 'red', 'orange']
    cmap = ListedColormap(colors)

    # Panel 1: CC1 Genesis
    ax1 = plt.subplot(2, 2, 1)
    im1 = ax1.imshow(tcc_t1, cmap='gray', vmin=0, vmax=1, alpha=0.3)
    im1 = ax1.imshow(cc1_genesis_errors, cmap=cmap, vmin=0, vmax=4, alpha=0.7)
    ax1.set_title(f'CC1 Genesis Spatial Errors\n' +
                  f'Hit={np.sum(cc1_genesis_errors==1)}, NearMiss={np.sum(cc1_genesis_errors==2)}, ' +
                  f'Miss={np.sum(cc1_genesis_errors==3)}, FA={np.sum(cc1_genesis_errors==4)}',
                  fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Panel 2: CC2 Genesis
    ax2 = plt.subplot(2, 2, 2)
    im2 = ax2.imshow(tcc_t1, cmap='gray', vmin=0, vmax=1, alpha=0.3)
    im2 = ax2.imshow(cc2_genesis_errors, cmap=cmap, vmin=0, vmax=4, alpha=0.7)
    ax2.set_title(f'CC2 Genesis Spatial Errors\n' +
                  f'Hit={np.sum(cc2_genesis_errors==1)}, NearMiss={np.sum(cc2_genesis_errors==2)}, ' +
                  f'Miss={np.sum(cc2_genesis_errors==3)}, FA={np.sum(cc2_genesis_errors==4)}',
                  fontsize=12, fontweight='bold', color='green')
    ax2.axis('off')

    # Panel 3: CC1 Lysis
    ax3 = plt.subplot(2, 2, 3)
    im3 = ax3.imshow(tcc_t1, cmap='gray', vmin=0, vmax=1, alpha=0.3)
    im3 = ax3.imshow(cc1_lysis_errors, cmap=cmap, vmin=0, vmax=4, alpha=0.7)
    ax3.set_title(f'CC1 Lysis Spatial Errors\n' +
                  f'Hit={np.sum(cc1_lysis_errors==1)}, NearMiss={np.sum(cc1_lysis_errors==2)}, ' +
                  f'Miss={np.sum(cc1_lysis_errors==3)}, FA={np.sum(cc1_lysis_errors==4)}',
                  fontsize=12, fontweight='bold')
    ax3.axis('off')

    # Panel 4: CC2 Lysis
    ax4 = plt.subplot(2, 2, 4)
    im4 = ax4.imshow(tcc_t1, cmap='gray', vmin=0, vmax=1, alpha=0.3)
    im4 = ax4.imshow(cc2_lysis_errors, cmap=cmap, vmin=0, vmax=4, alpha=0.7)
    ax4.set_title(f'CC2 Lysis Spatial Errors\n' +
                  f'Hit={np.sum(cc2_lysis_errors==1)}, NearMiss={np.sum(cc2_lysis_errors==2)}, ' +
                  f'Miss={np.sum(cc2_lysis_errors==3)}, FA={np.sum(cc2_lysis_errors==4)}',
                  fontsize=12, fontweight='bold', color='green')
    ax4.axis('off')

    # Legend
    legend_elements = [
        mpatches.Patch(color='green', label=f'Exact Hit (0km)'),
        mpatches.Patch(color='yellow', label=f'Near Miss (<{spatial_tolerance_km}km)'),
        mpatches.Patch(color='red', label=f'Miss (>{spatial_tolerance_km}km)'),
        mpatches.Patch(color='orange', label='False Alarm'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11, frameon=True)

    plt.suptitle(f'Spatial Accuracy: Genesis/Lysis Events (Forecast {forecast_idx}, t={t0}→{t1}h)\n' +
                 f'Tolerance: {spatial_tolerance_km}km | Pixel Count ≥{core_threshold:.0%}',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    if output_dir:
        filename = output_dir / f'spatial_accuracy_map_f{forecast_idx}.png'
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        print(f"✓ Saved: {filename.name}")
    else:
        plt.show()

    plt.close()


def main():
    print("="*80)
    print("VISUALIZE SPATIAL ACCURACY")
    print("="*80)

    output_dir = Path("genesis_lysis_plots")
    output_dir.mkdir(exist_ok=True)

    # Create spatial error maps for select forecasts
    forecast_indices = [100, 250, 365]
    spatial_tolerance_km = 50

    for forecast_idx in forecast_indices:
        print(f"\nCreating spatial error map for forecast {forecast_idx}...")
        create_spatial_error_map(
            forecast_idx, 0, 3,
            spatial_tolerance_km=spatial_tolerance_km,
            output_dir=output_dir
        )

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nGenerated files in {output_dir}/:")
    for idx in forecast_indices:
        print(f"  - spatial_accuracy_map_f{idx}.png")


if __name__ == "__main__":
    main()
