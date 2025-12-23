import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import platform
import lightning as L
import sys
import os
import warnings
import shutil
from common.util import get_rank
from datetime import datetime
from matplotlib.ticker import ScalarFormatter
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities import rank_zero_only
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.use("Agg")

colors = [
    "blue",
    "orange",
    "green",
    "red",
    "black",
    "purple",
    "grey",
    "magenta",
]


# Create a custom colormap with white around zero
def create_colormap(white_range=0.05, vmin=-1, vmax=1):
    """
    Create a blue-white-red colormap with white around zero
    Blue = negative, Red = positive
    """
    # Get base colormaps
    blues = plt.colormaps["Blues"]
    reds = plt.colormaps["Reds"]
    # Calculate positions in 0-1 range
    zero_pos = -vmin / (vmax - vmin)  # Position of zero in normalized space
    white_width = white_range / (vmax - vmin)  # Width of white region

    # Define color positions
    colors = []
    positions = []

    # Blue region (negative values)
    positions.append(0)
    colors.append(blues(0.9))  # Dark blue

    positions.append(zero_pos - white_width)
    colors.append(blues(0.3))  # Light blue

    # White region around zero
    positions.append(zero_pos - white_width)
    colors.append("white")

    positions.append(zero_pos + white_width)
    colors.append("white")

    # Red region (positive values)
    positions.append(zero_pos + white_width)
    colors.append(reds(0.3))  # Light red

    positions.append(1)
    colors.append(reds(0.9))  # Dark red

    # Create the colormap
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "blue_white_red", list(zip(positions, colors))
    )

    return cmap


def run_info():
    run_name = os.environ["CC2_RUN_NAME"]
    run_number = os.environ["CC2_RUN_NUMBER"]
    run_dir = os.environ["CC2_RUN_DIR"]

    return run_name, run_number, run_dir


class PredictionPlotterCallback(L.Callback):
    def __init__(self, after_train: bool = True, after_val: bool = True):
        self.after_train = after_train
        self.after_val = after_val

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        if not self.after_train:
            return

        pl_module.eval()

        predictions = pl_module.latest_train_predictions
        x, y = pl_module.latest_train_data

        self.plot(
            x.cpu().detach().to(torch.float32),
            y.cpu().detach().to(torch.float32),
            predictions.cpu().detach().to(torch.float32),
            trainer.current_epoch,
            "train",
            trainer.sanity_checking,
        )

        # Restore model to training mode
        pl_module.train()

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.after_val:
            return
        pl_module.eval()

        predictions = pl_module.latest_val_predictions
        x, y = pl_module.latest_val_data

        self.plot(
            x.cpu().detach().to(torch.float32),
            y.cpu().detach().to(torch.float32),
            predictions.cpu().detach().to(torch.float32),
            trainer.current_epoch,
            "val",
            trainer.sanity_checking,
        )

        # Restore model to training mode
        pl_module.train()

    def plot(self, x, y, predictions, epoch, stage, sanity_check=False):
        # y shape: [B, T, 1, 128, 128]
        # prediction shape: [B, T, 1, 128, 128]
        # input_field = x[0].squeeze()
        run_name, run_number, run_dir = run_info()

        truth = y[0]

        x = x[0]  # T, C, H, W
        predictions = predictions[0]  # T, C, H, W (B removed)
        num_truth = truth.shape[0]
        num_hist = x.shape[0]

        rows = num_truth
        cols = num_hist + 1 + 1  # input fields, +1 for prediction, +1 for truth

        fig, ax = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows + 0.5))
        ax = np.atleast_2d(ax)

        for i in range(num_truth):  # times
            if i == 0:
                # First prediction: all ground truth
                input_field = x.squeeze()
            elif i <= num_hist:
                # gradually using own predictions as input
                num_ground_truth = max(0, num_hist - i)
                num_pred = num_hist - num_ground_truth
                start_pred = max(0, i - num_hist)

                input_field = torch.cat(
                    [
                        x[i:num_hist].squeeze(1),
                        predictions[start_pred:num_pred].squeeze(1),
                    ]
                )
            elif i > num_hist:
                start_pred = max(0, i - num_hist)
                input_field = predictions[start_pred : start_pred + num_hist].squeeze()

            for j in range(num_hist):
                num = i - num_hist + j + 1
                ax[i, j].imshow(input_field[j])
                ax[i, j].set_title(f"Time T{num:+}")
                ax[i, j].set_axis_off()
            ax[i, num_hist].imshow(predictions[i].squeeze())
            ax[i, num_hist].set_title(f"Time T{i+1:+} prediction")
            ax[i, num_hist].set_axis_off()
            ax[i, num_hist + 1].imshow(truth[i].squeeze())
            ax[i, num_hist + 1].set_title(f"Time T{i+1:+} truth")
            ax[i, num_hist + 1].set_axis_off()

        title = (
            "Training time prediction"
            if stage == "train"
            else "Validation time prediction"
        )

        title = "{} for {} num={} at epoch {} (host={}, time={})".format(
            title,
            run_name,
            run_number,
            epoch,
            platform.node(),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        filename = "{}/figures/{}_{}_{}_epoch_{:03d}_{}.png".format(
            run_dir,
            platform.node(),
            run_name,
            run_number,
            epoch,
            stage,
        )

        plt.suptitle(title)

        if sanity_check is False:
            plt.savefig(filename)
        plt.close(fig)


class DiagnosticCallback(L.Callback):
    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):

        if not trainer.is_global_zero or trainer.sanity_checking:
            return

        tendencies = pl_module.latest_val_tendencies
        predictions = pl_module.latest_val_predictions
        x, y = pl_module.latest_val_data

        self.plot_visual(
            x[0].cpu().detach().to(torch.float32),
            y[0].cpu().detach().to(torch.float32),
            predictions[0].cpu().detach().to(torch.float32),
            tendencies[0].cpu().detach().to(torch.float32),
            trainer.current_epoch,
            trainer.sanity_checking,
        )

    def plot_visual(self, input_field, truth, pred, tendencies, epoch, sanity_checking):
        run_name, run_number, run_dir = run_info()

        plt.suptitle(
            "{} num={} at epoch {} (host={}, time={})".format(
                run_name,
                run_number,
                epoch,
                platform.node(),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        )

        # input_field T, C, H, W
        # truth T, C, H, W
        # pred T, C, H, W
        # tend T, C, H, W

        input_field = input_field.squeeze(1)
        truth = truth.squeeze(1)  # remove channel dim
        pred = pred.squeeze(1)
        tendencies = tendencies.squeeze(1)

        T = pred.shape[0]

        fig, ax = plt.subplots(T, 5, figsize=(15, 3 * T + 0.5))
        ax = np.atleast_2d(ax)

        cmap = plt.cm.coolwarm
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

        for t in range(T):
            if t == 0:
                true_tendencies = truth[t].squeeze() - input_field[-1].squeeze()
            else:
                true_tendencies = truth[t].squeeze() - truth[t - 1].squeeze()

            im = ax[t, 0].imshow(true_tendencies, cmap=cmap, norm=norm)
            ax[t, 0].set_title(f"True tendencies step={t}")
            ax[t, 0].set_axis_off()

            divider = make_axes_locatable(ax[t, 0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)

            ax[t, 1].imshow(tendencies[t], cmap=cmap, norm=norm)
            ax[t, 1].set_title(f"Predicted tendencies step={t}")
            ax[t, 1].set_axis_off()

            data = true_tendencies - tendencies[t]
            ax[t, 2].set_title(f"Tendencies bias step={t}")
            ax[t, 2].set_axis_off()
            im = ax[t, 2].imshow(data.cpu(), cmap=create_colormap())

            divider_bias = make_axes_locatable(ax[t, 2])
            cax_bias = divider_bias.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax_bias)

            ax[t, 3].set_title(f"True histogram step={t}")
            ax[t, 3].hist(true_tendencies.flatten(), bins=50, density=True)

            ax[t, 4].set_title(f"Predicted histogram step={t}")
            ax[t, 4].hist(tendencies[t].flatten(), bins=50, density=True)

        plt.tight_layout()

        if sanity_checking:
            return

        plt.savefig(
            "{}/figures/{}_{}_{}_epoch_{:03d}_analysis.png".format(
                run_dir,
                platform.node(),
                run_name,
                run_number,
                epoch,
            )
        )

        plt.close(fig)


class CleanupFailedRunCallback(L.Callback):
    def on_exception(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        exception: BaseException,
    ) -> None:
        # Ensure cleanup only happens on the main process (rank 0) in distributed settings
        if trainer.is_global_zero:
            run_dir = os.environ.get("CC2_RUN_DIR")

            if run_dir and os.path.isdir(run_dir):
                # Check that run_dir contains more than just config.yaml -- figures or checkpoints
                try:
                    file_count = sum(len(files) for _, _, files in os.walk(run_dir))
                    if file_count <= 1:
                        shutil.rmtree(run_dir)
                        print(f"Removed empty run directory: {run_dir}")
                except OSError as e:
                    print(f"Error during cleanup check/removal of {log_dir}: {e}")
