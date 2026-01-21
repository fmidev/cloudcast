import torch
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from common.util import get_latest_run_dir
from tqdm import tqdm
from verif.mae import mae, mae2d, plot_mae_timeseries, plot_mae2d
from verif.psd import psd, plot_psd
from verif.fss import fss, plot_fss
from verif.error_spread import error_spread, plot_error_spread
from verif.variance_ratio import variance_ratio, plot_variance_ratio
from verif.highk_power_ratio import highk_power_ratio, plot_highk_power_ratio
from verif.spectral_coherence import (
    spectral_coherence_bands,
    plot_spectral_coherence_bands,
)
from verif.change_metrics import (
    change_metrics,
    # plot_change_prf_timeseries,
    plot_change_corr_stationarity_timeseries,
)
from verif.composite_score import (
    composite_score,
    plot_composite_bars,
    plot_component_contributions,
)
from verif.ssim import ssim, plot_ssim
from verif.hist import hist, plot_hist


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, required=False, nargs="+", default=[])
    parser.add_argument("--path_name", type=str, required=False, nargs="+", default=[])
    parser.add_argument(
        "--score",
        type=str,
        required=False,
        nargs="+",
        choices=[
            "stamps",
            "mae",
            "mae2d",
            "psd",
            "fss",
            "variance_ratio",
            "highk_power_ratio",
            "spectral_coherence",
            "change_metrics",
            "hist",
            "ssim",
        ],
        help="Score to produce",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="runs/verification",
        help="Path to save the verification results and plots",
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="If set, only plot the results without running the verification.",
    )
    parser.add_argument(
        "--crop",
        type=int,
        default=0,
        help="Crop pixels from each side",
    )

    args = parser.parse_args()

    os.makedirs(f"{args.save_path}/figures", exist_ok=True)
    os.makedirs(f"{args.save_path}/results", exist_ok=True)

    if args.score is None:
        args.score = ["mae"]

    if "stamps" in args.score and args.plot_only:
        raise ValueError("Cannot plot stamps without running verification.")

    assert args.run_name or args.path_name

    return args


def get_labels(args):
    labels = args.run_name.copy()
    for path_name in args.path_name:
        labels.append(os.path.basename(os.path.realpath(path_name)))
    return labels


def read_data_from_dir(file_path, truth_cache):
    truth_file = f"{file_path}/truth.pt"

    if truth_cache is not None:
        truth_file_real = os.path.realpath(truth_file)
        if truth_file_real in truth_cache:
            truth = truth_cache[truth_file_real]
        else:
            truth = torch.load(
                truth_file, map_location=torch.device("cpu"), weights_only=True
            )
            truth_cache[truth_file_real] = truth
    else:
        truth = torch.load(
            truth_file, map_location=torch.device("cpu"), weights_only=True
        )

    prediction_files = [
        f"{file_path}/predictions_noo.pt",
        f"{file_path}/predictions.pt",
    ]

    for file in prediction_files:
        if os.path.exists(file) is False:
            continue

        predictions = torch.load(
            file,
            map_location=torch.device("cpu"),
            weights_only=True,
        )

        if "noo.pt" in file:
            print(f"Read NOO file {file}")
        break
    dates = torch.load(
        f"{file_path}/dates.pt", map_location=torch.device("cpu"), weights_only=True
    )

    return truth, predictions, dates


def read_data(run_name, truth_cache=None):
    if "/" in run_name:
        run_dir = f"runs/{run_name}"
    else:
        run_dir = get_latest_run_dir(f"runs/{run_name}")

    assert run_dir, f"Run {run_name} not found in runs/"

    file_path = f"{run_dir}/test-output"

    truth, predictions, dates = read_data_from_dir(file_path, truth_cache)

    assert (
        predictions.ndim == 5
    ), "Predictions should be 5D tensor (batch, time, channel, height, width), got {}".format(
        predictions.shape
    )
    assert (
        truth.ndim == 5
    ), "Truth should be 5D tensor (batch, time, channel, height, width), got {}".format(
        truth.shape
    )
    return truth, predictions, dates


def check_that_all_files_exist(runs, paths):
    pbar = tqdm(total=3 * (len(runs) + len(paths)), desc="Checking files")

    def _check_files(path):
        for f in ("predictions.pt", "dates.pt", "truth.pt"):
            f = f"{path}/{f}"
            if not os.path.exists(f):
                raise ValueError(f"File {f} does not exist")
            pbar.update(1)

    for run_name in runs:
        if "/" in run_name:
            run_dir = f"runs/{run_name}"
        else:
            run_dir = get_latest_run_dir(f"runs/{run_name}")

        _check_files(f"{run_dir}/test-output")

    for path in paths:
        _check_files(path)

    pbar.close()


def intersection(all_truth, all_predictions, all_dates):
    # 1. Build intersection of all forecast sequences
    all_sets = [set(map(tuple, d.numpy())) for d in all_dates]
    intersection_keys = set.intersection(*all_sets)

    print(
        f"Total unique forecast sequences before intersection: {len(intersection_keys)}"
    )

    # 2. Sort the intersection chronologically (by first lead time)
    sorted_common_keys = sorted(intersection_keys, key=lambda row: row[0])

    # 3. For each run, filter and reorder according to sorted_common_keys
    new_truth, new_predictions, new_dates = [], [], []

    for i, (truth, pred, dates) in enumerate(
        zip(all_truth, all_predictions, all_dates)
    ):
        date_dict = {tuple(row): idx for idx, row in enumerate(dates.numpy())}

        filtered_truth = []
        filtered_pred = []
        filtered_dates = []

        for key in sorted_common_keys:
            if key in date_dict:
                idx = date_dict[key]
                filtered_truth.append(truth[idx])
                filtered_pred.append(pred[idx])
                filtered_dates.append(torch.tensor(key, dtype=torch.float64))

        if filtered_truth:
            new_truth.append(torch.stack(filtered_truth))
            new_predictions.append(torch.stack(filtered_pred))
            new_dates.append(torch.stack(filtered_dates))
        else:
            print(f"{run_name[i]}: No overlapping forecasts found!")

    print(f"Total forecast sequences after intersection: {new_predictions[0].shape[0]}")

    for i in range(1, len(new_predictions)):
        n_0 = new_predictions[i - 1].shape[0]
        n_1 = new_predictions[i].shape[0]
        assert (
            n_0 == n_1
        ), "number of forecasts do not match after intersect: {} vs {}".format(n_0, n_1)

    return new_truth, new_predictions, new_dates


def equalize_datasets(labels, all_truth, all_predictions, all_dates):
    need_intersection = False

    for i in tqdm(range(1, len(all_dates)), desc="Equalizing"):
        if all_dates[i - 1].shape != all_dates[i].shape:
            print(
                "Different shape of dates for {} ({}) and {} ({})".format(
                    labels[i - 1],
                    all_dates[i - 1].shape,
                    labels[i],
                    all_dates[i].shape,
                )
            )

            a = all_dates[i]
            b = all_dates[i - 1]

            a = set(map(tuple, a.numpy()))
            b = set(map(tuple, b.numpy()))

            A_not_in_B = a - b
            B_not_in_A = b - a

            # Print rows

            if len(A_not_in_B):
                print("Rows in {} but not in {}:".format(labels[i - 1], labels[i]))
                for row in A_not_in_B:
                    d_s = [np.datetime64(int(x), "s") for x in row]
                    print(d_s)

            if len(B_not_in_A):
                print("Rows in {} but not in {}:".format(labels[i], labels[i - 1]))
                for row in B_not_in_A:
                    d_s = [np.datetime64(int(x), "s") for x in row]
                    print(d_s)

            need_intersection = True

    if need_intersection:
        # Filter predictions, truth and dates so that an intersection of both datasets
        # is picked

        return intersection(all_truth, all_predictions, all_dates)

    return all_truth, all_predictions, all_dates


def prepare_data(args):
    all_truth, all_predictions, all_dates = [], [], []
    truth_cache = {}

    check_that_all_files_exist(args.run_name, args.path_name)

    if args.run_name:
        for run_name in tqdm(args.run_name, desc="Reading data"):

            truth, predictions, dates = read_data(run_name, truth_cache)

            all_truth.append(truth)
            all_predictions.append(predictions)
            all_dates.append(dates)

    if args.path_name:
        for path_name in tqdm(args.path_name, desc="Reading data"):

            truth, predictions, dates = read_data_from_dir(path_name, truth_cache)

            all_truth.append(truth)
            all_predictions.append(predictions)
            all_dates.append(dates)

        # return all_truth, all_predictions, all_dates

    if len(all_dates) > 1:
        all_truth, all_predictions, all_dates = equalize_datasets(
            get_labels(args), all_truth, all_predictions, all_dates
        )

    if args.crop:
        # B, T, C, H, W
        c = args.crop
        print(f"Cropping with c={c}")
        all_truth = [x[:, :, :, c:-c, c:-c] for x in all_truth]
        all_predictions = [x[:, :, :, c:-c, c:-c] for x in all_predictions]

    return all_truth, all_predictions, all_dates


def plot_one_stamp(truth, predictions, dates, filename):
    num_timesteps = truth.shape[0]
    num_models = len(predictions)
    nrows = 1 + num_models  # 1 row for truth + N rows for models
    ncols = num_timesteps

    # Dynamically adjust figsize - very rough estimate
    fig_width = ncols * 3
    fig_height = nrows * 3
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_width, fig_height), sharex=True, sharey=True
    )

    # If only one row/col, axes might not be a 2D array, handle this:
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])  # Make it 2D Nrows x 1 col

    fig.suptitle(
        f"Ground Truth vs. Model Predictions {np.datetime64(int(dates[0]), 's')}",
        fontsize=24,
    )
    cmap = "Blues"

    labels = get_labels(args)

    for r in range(nrows):
        if r == 0:
            img_data = truth
        else:
            img_data = predictions[r - 1]

        for c in range(ncols):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])

            if c == 0:  # Set row labels on the first column
                if r == 0:
                    ax.set_ylabel("Ground Truth", fontsize=12, rotation=90, labelpad=10)
                else:
                    ax.set_ylabel(
                        f"{labels[r-1]}", fontsize=12, rotation=90, labelpad=10
                    )

            im = ax.imshow(img_data[c, 0], cmap=cmap, vmin=0, vmax=1)

            if r == 0:  # Top row: Ground Truth
                ax.set_title(f"Leadtime {c}h", fontsize=16)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])

    # Add colorbar
    cbar_ax = fig.add_axes([0.86, 0.039, 0.015, 0.85])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Cloud cover", rotation=90, labelpad=15, fontsize=16)

    plt.savefig(filename)
    print(f"Timeseries plot saved to {filename}")
    plt.close(fig)


def plot_stamps(
    run_name: list[str],
    all_truth: list[torch.Tensor],
    all_predictions: list[torch.Tensor],
    all_dates: list[torch.Tensor],
    save_path: str,
):
    rands = torch.randint(low=1, high=all_truth[0].shape[0], size=(2,))
    rands = [x.item() for x in rands]

    forecasts_to_plot = [0] + rands

    for i, f_idx in enumerate(forecasts_to_plot):
        preds = []
        for j in range(len(all_predictions)):
            preds.append(all_predictions[j][f_idx])
        filename = f"{save_path}/figures/stamps{i}.png"

        plot_one_stamp(all_truth[0][f_idx], preds, all_dates[0][f_idx], filename)


if __name__ == "__main__":
    args = get_args()

    if args.plot_only is False:
        all_truth, all_predictions, all_dates = prepare_data(args)

    all_results = []

    labels = get_labels(args)

    for score in args.score:
        print(f"Score: {score}")
        if score == "mae":
            if args.plot_only:
                results = pd.read_csv(f"{args.save_path}/results/{score}.csv")
            else:
                results = mae(labels, all_truth, all_predictions, args.save_path)
            plot_mae_timeseries(results, args.save_path)
            print(results)

        elif score == "mae2d":
            if args.plot_only:
                results = torch.load(
                    f"{args.save_path}/results/{score}.pt", weights_only=False
                )
            else:
                results = mae2d(all_truth, all_predictions, args.save_path)
            plot_mae2d(labels, results, args.save_path)

        elif score == "psd":
            if args.plot_only:
                obs_psd = torch.load(
                    f"{args.save_path}/results/observed_psd.pt", weights_only=False
                )
                pred_psd = torch.load(
                    f"{args.save_path}/results/predicted_psd.pt", weights_only=False
                )
            else:
                obs_psd, pred_psd = psd(all_truth, all_predictions, args.save_path)
            plot_psd(labels, obs_psd, pred_psd, args.save_path)
            results = (obs_psd, pred_psd)

        elif score == "fss":
            if args.plot_only:
                results_t = torch.load(
                    f"{args.save_path}/results/{score}.pt", weights_only=False
                )
                results = pd.read_csv(f"{args.save_path}/results/{score}.csv")
            else:
                results_t, results = fss(
                    labels, all_truth, all_predictions, args.save_path
                )

            plot_fss(labels, results_t, args.save_path)

        elif score == "hist":
            if args.plot_only:
                results = pd.read_csv(f"{args.save_path}/results/{score}.csv")
            else:
                results = hist(labels, all_truth, all_predictions, args.save_path)

            plot_hist(labels, results, args.save_path)

        elif score == "variance_ratio":
            if args.plot_only:
                results = pd.read_csv(f"{args.save_path}/results/{score}.csv")
            else:
                results = variance_ratio(
                    labels, all_truth, all_predictions, args.save_path
                )

            print(results)
            plot_variance_ratio(results, args.save_path)

        elif score == "highk_power_ratio":
            if args.plot_only:
                results = pd.read_csv(f"{args.save_path}/results/{score}.csv")
            else:
                results = highk_power_ratio(
                    labels, all_truth, all_predictions, args.save_path
                )

            print(results)
            plot_highk_power_ratio(results, args.save_path)

        elif score == "spectral_coherence":
            if args.plot_only:
                results = pd.read_csv(f"{args.save_path}/results/{score}.csv")
            else:
                results = spectral_coherence_bands(
                    labels, all_truth, all_predictions, args.save_path
                )

            print(results)
            plot_spectral_coherence_bands(results, args.save_path)

        elif score == "change_metrics":
            if args.plot_only:
                results = pd.read_csv(f"{args.save_path}/results/{score}.csv")
            else:
                results = change_metrics(
                    labels, all_truth, all_predictions, args.save_path
                )

            print(results)
            # plot_change_prf_timeseries(results, args.save_path)
            plot_change_corr_stationarity_timeseries(results, args.save_path)

        elif score == "ssim":
            if args.plot_only:
                results = pd.read_csv(f"{args.save_path}/results/{score}.csv")
            else:
                results = ssim(labels, all_truth, all_predictions, args.save_path)
            plot_ssim(results, args.save_path)
            print(results)

        all_results.append(results)

    composite_score_metrics = [
        "mae",
        "fss",
        "variance_ratio",
        "highk_power_ratio",
        "spectral_coherence",
        "change_metrics",
        "ssim",
    ]

    composite_score_values = {}
    for s in composite_score_metrics:
        if s not in args.score:
            break

        i = args.score.index(s)
        composite_score_values[s] = all_results[i]

    if len(composite_score_values.keys()) == len(composite_score_metrics):
        composite_result = composite_score(labels, composite_score_values)
        # plot_composite_bars(composite_result, save_path=args.save_path)
        plot_component_contributions(composite_result, save_path=args.save_path)
        print("Produced composite scores")
        print(composite_result)
    else:
        print("Not producing composite score: some scores not calculated")

    if args.plot_only is False:
        plot_stamps(labels, all_truth, all_predictions, all_dates, args.save_path)
