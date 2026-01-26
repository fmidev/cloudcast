import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import lightning as L
from anemoi.datasets import open_dataset
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn


def get_default_normalization_methods(custom_methods):
    default_methods = {
        # prognostic parameters
        "tcc": "none",
        # dynamic forcings
        "r_1000": "none",
        "r_500": "none",
        "r_700": "none",
        "r_850": "none",
        "r_925": "none",
        "t_1000": "standard",
        "t_500": "standard",
        "t_700": "standard",
        "t_850": "standard",
        "t_925": "standard",
        "u_1000": "standard",
        "u_500": "standard",
        "u_700": "standard",
        "u_850": "standard",
        "u_925": "standard",
        "v_1000": "standard",
        "v_500": "standard",
        "v_700": "standard",
        "v_850": "standard",
        "v_925": "standard",
        "z_1000": "standard",
        "z_500": "standard",
        "z_700": "standard",
        "z_850": "standard",
        "z_925": "standard",
        # environment dynamic forcings
        "insolation": "none",
        "cos_julian_day": "none",
        "sin_julian_day": "none",
        "cos_local_time": "none",
        "sin_local_time": "none",
        # environment static forcings
        "lsm": "none",
        "z": "minmax",
        "cos_latitude": "none",
        "sin_latitude": "none",
        "cos_longitude": "none",
        "sin_longitude": "none",
    }

    if custom_methods is not None:
        default_methods.update(custom_methods)

    # rank_zero_info(f"normalization methods: {default_methods}")

    return default_methods


def gaussian_smooth(x, sigma=0.8, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1

    # Create 1D Gaussian kernel
    gauss = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
    gauss = torch.exp(-(gauss**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()

    # Create 2D kernel by outer product
    kernel = gauss[:, None] @ gauss[None, :]
    kernel = kernel.to(device=x.device, dtype=x.dtype)

    # Reshape kernel for PyTorch conv2d
    kernel = kernel[None, None, :, :]

    orig_shape = x.shape

    # Add batch dimension if needed
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
        B, C, H, W = x.shape
    elif len(x.shape) == 5:
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
    else:
        B, C, H, W = x.shape

    assert C < 50, "The order should be B, C, H, W, but got: {}".format(x.shape)

    assert x.ndim == 4, "data needs to have 4 dimensions, got: {}".format(x.shape)

    # Move kernel to same device as input
    kernel = kernel.to(x.device)

    # Apply smoothing channel by channel
    pad = kernel_size // 2
    smoothed = []
    for c in range(C):
        channel = x[:, c : c + 1, ...]
        channel = F.pad(channel, (pad, pad, pad, pad), mode="reflect")
        channel = F.conv2d(channel, kernel)
        smoothed.append(channel)

    x = torch.cat(smoothed, dim=1)
    x = torch.clamp(x, 0, 1)

    x = x.reshape(orig_shape)
    return x


class AnemoiDataset(Dataset):
    def __init__(
        self,
        zarr_path: list[str],
        group_size: int,
        prognostic_params: list[str],
        forcing_params: list[str],
        static_forcing_path: str | None,
        static_forcing_params: list[str],
        normalization_methods: dict,
        return_metadata: bool = False,
        data_options: dict[str, str | int | list] = {},
        statistics: dict | None = {},
        history_length: int = 2,
    ):
        self.data = open_dataset(zarr_path, **data_options)
        self.group_size = group_size
        self.history_length = history_length
        self.time_steps = len(self.data.dates)

        assert type(prognostic_params) == list
        assert type(forcing_params) == list
        assert type(static_forcing_params) == list

        self.prognostic_params = prognostic_params
        self.forcing_params = forcing_params
        self.static_forcing_params = static_forcing_params
        self.static_forcing_path = static_forcing_path

        self.data_indexes = [self.data.name_to_index[x] for x in self.prognostic_params]
        self.forcings_indexes = [
            self.data.name_to_index[x] for x in self.forcing_params
        ]

        # Initialize static forcings to None, will be loaded once
        self.static_forcings = None
        self.static_forcings_indexes = []
        if self.static_forcing_path and self.static_forcing_params:
            temp_static_data = open_dataset(self.static_forcing_path, **data_options)
            self.static_forcings_indexes = [
                temp_static_data.name_to_index[x] for x in self.static_forcing_params
            ]
            del temp_static_data

        assert (
            self.time_steps >= group_size
        ), "Not enough time steps in data for the specified group size: {} >= {}".format(
            self.time_steps, group_size
        )

        self.missing_indices = set(self.data.missing)
        self.valid_indices = [
            i
            for i in range(self.time_steps - self.group_size + 1)
            if not self._sequence_has_missing_data(i)
        ]

        self.data_options = data_options

        self.input_resolution = self.data.field_shape  # H, W
        self.return_metadata = return_metadata
        self.dates = self.data.dates

        if statistics:
            self.statistics = statistics
            rank_zero_info("Using statistics from checkpoint")
        else:
            self.statistics = self.data.statistics
            rank_zero_info("Using statistics from data")

        assert self.statistics, f"Statistics is: {self.statistics}"

        self.normalization_methods = normalization_methods
        assert self.normalization_methods is not None

        self._setup_normalization()

    def _setup_normalization(self):
        # Pre-compute combined indexes and params for dynamic data
        self.combined_dynamic_indexes = self.data_indexes + self.forcings_indexes
        self.combined_dynamic_params = self.prognostic_params + self.forcing_params

        # Load and normalize static forcings once
        dtype = torch.float32
        if self.static_forcing_path and self.static_forcing_params:
            static_root_group = open_dataset(
                self.static_forcing_path, **self.data_options
            )
            static_data_zarr = static_root_group[:]

            selection_tuple = (
                0,
                np.array(self.static_forcings_indexes),
                slice(None),
                slice(None),
            )

            # Load only the required static forcing channels
            # Assuming static forcings are [C_static, H, W]
            static_data_numpy = static_data_zarr[selection_tuple].astype(np.float32)

            # Make sure it's 4D for consistency (T, C, H, W) where T=1
            # If static data is (C, H, W), add a time dimension of 1

            if static_data_numpy.ndim == 3:
                static_data_numpy = np.expand_dims(static_data_numpy, axis=0)
            elif static_data_numpy.ndim == 4:
                pass  # Already in correct shape
            else:
                raise ValueError(
                    f"Unexpected static forcing data shape: {static_data_numpy.shape}"
                )

            self.static_forcings = np.zeros_like(static_data_numpy)

            for i, param_name in enumerate(self.static_forcing_params):
                method = self.normalization_methods.get(param_name, "none")
                channel_data = static_data_numpy[0, i : i + 1, :, :]

                assert channel_data.ndim == 3

                if method == "none":
                    self.static_forcings[0, i : i + 1, 0, :] = channel_data
                elif method == "minmax":
                    # Calculate min/max for THIS static channel
                    c_min = channel_data.min()
                    c_max = channel_data.max()
                    self.static_forcings[0, i : i + 1, 0, :] = (
                        channel_data - c_min
                    ) / (c_max - c_min + 1e-8)
                elif method == "standard":
                    # Calculate mean/std for THIS static channel
                    c_mean = channel_data.mean()
                    c_std = channel_data.std()
                    self.static_forcings[0, i : i + 1, 0, :] = (
                        channel_data - c_mean
                    ) / (c_std + 1e-8)
                else:
                    raise ValueError(
                        f"Unknown normalization method for static forcing '{param_name}': {method}"
                    )

                min_, max_, mean_ = (
                    np.min(self.static_forcings[0, i, :, :]),
                    np.max(self.static_forcings[0, i, :, :]),
                    np.mean(self.static_forcings[0, i, :, :]),
                )

            # Add time dimension (T=1) at the beginning for consistency (1, C_static, H, W)
            self.static_forcings = (
                torch.tensor(self.static_forcings).unsqueeze(0).to(dtype).squeeze(1)
            )

        # Combine all parameters and their normalization methods for the main normalize function
        self.all_combined_params = self.prognostic_params + self.forcing_params

        self.all_combined_indexes = self.data_indexes + self.forcings_indexes

        self.total_channels = len(self.all_combined_indexes)

        # Pre-reshape statistics for all combined channels (dynamic + static)
        # Assuming self.statistics also contains stats for static forcings if registered
        self.mins = torch.tensor(
            self.statistics["minimum"][self.all_combined_indexes].reshape(
                1, self.total_channels, 1, 1
            ),
            dtype=dtype,
        )

        self.maxs = torch.tensor(
            self.statistics["maximum"][self.all_combined_indexes].reshape(
                1, self.total_channels, 1, 1
            ),
            dtype=dtype,
        )
        self.means = torch.tensor(
            self.statistics["mean"][self.all_combined_indexes].reshape(
                1, self.total_channels, 1, 1
            ),
            dtype=dtype,
        )
        self.stds = torch.tensor(
            self.statistics["stdev"][self.all_combined_indexes].reshape(
                1, self.total_channels, 1, 1
            ),
            dtype=dtype,
        )
        # Create a single method tensor for indexing for all combined parameters
        methods_all = [self.normalization_methods[k] for k in self.all_combined_params]
        self.method_tensor = torch.tensor(
            [0 if m == "none" else 1 if m == "minmax" else 2 for m in methods_all],
            dtype=torch.long,
        ).view(1, self.total_channels, 1, 1)

    def _sequence_has_missing_data(self, start_idx):
        for i in range(start_idx, start_idx + self.group_size):
            if i in self.missing_indices:
                return True
        return False

    def __len__(self):
        return len(self.valid_indices)

    def normalize(self, tensor: torch.tensor, params: list):
        # This normalize method will now work on the combined dynamic + static data.
        # The `self.method_tensor`, `self.mins`, etc., are already setup to include
        # all channels (prognostic, dynamic forcings, static forcings) in their correct order.

        T, C, H, W = tensor.shape
        result = tensor.clone()  # Avoid modifying input tensor directly

        minmax_mask = self.method_tensor.expand(T, -1, H, W) == 1
        std_mask = self.method_tensor.expand(T, -1, H, W) == 2

        if minmax_mask.any():
            result = torch.where(
                minmax_mask,
                (tensor - self.mins.expand(T, -1, H, W))
                / (
                    self.maxs.expand(T, -1, H, W) - self.mins.expand(T, -1, H, W) + 1e-8
                ),
                result,
            )
        if std_mask.any():
            result = torch.where(
                std_mask,
                (tensor - self.means.expand(T, -1, H, W))
                / (self.stds.expand(T, -1, H, W) + 1e-8),
                result,
            )

        return result.to(tensor.dtype)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]

        # Get the combined dynamic data (prognostic + dynamic forcings)
        combined_dynamic = self.data[
            actual_idx : actual_idx + self.group_size,
            self.combined_dynamic_indexes,
            ...,
        ]

        T_dyn, C_dyn_e, H_flat_e, W_flat_e = combined_dynamic.shape

        assert H_flat_e == 1  # Assuming the 'E' dimension is flattened into H if E is 1

        # Reshape dynamic data to (T, C, H, W)
        combined_dynamic = combined_dynamic.reshape(
            T_dyn,
            len(self.combined_dynamic_indexes),
            self.input_resolution[0],
            self.input_resolution[1],
        )
        combined_dynamic = torch.from_numpy(combined_dynamic)

        # Normalize the dynamic data only
        combined_dynamic = self.normalize(
            combined_dynamic, self.combined_dynamic_params
        )

        # Separate prognostic from dynamic forcings (both now normalized)
        prognostic_normalized = combined_dynamic[:, : len(self.data_indexes), :, :]
        dynamic_forcing_normalized = combined_dynamic[:, len(self.data_indexes) :, :, :]

        # If static forcings exist, concatenate them to the dynamic forcings
        if self.static_forcings is not None:
            num_timesteps_in_group = self.group_size

            if self.static_forcings.shape[-1] > 2000:
                s = self.static_forcings.shape
                self.static_forcings = self.static_forcings.reshape(
                    s[0], s[1], self.input_resolution[0], self.input_resolution[1]
                )

            static_forcings_for_group = self.static_forcings.repeat(
                num_timesteps_in_group, 1, 1, 1
            )

            # Concatenate normalized dynamic forcings with normalized static forcings
            all_forcings_combined_normalized = torch.cat(
                (dynamic_forcing_normalized, static_forcings_for_group), dim=1
            )
            # The 'combined' tensor for output will be prognostic + all_forcings_combined_normalized
            combined_output_tensor = torch.cat(
                (prognostic_normalized, all_forcings_combined_normalized), dim=1
            )
        else:
            all_forcings_combined_normalized = dynamic_forcing_normalized
            combined_output_tensor = combined_dynamic

        # Split into data (prognostic) and forcings (dynamic + static)
        data_channels_count = len(self.prognostic_params)
        data = combined_output_tensor[:, :data_channels_count, :, :]
        forcing = combined_output_tensor[:, data_channels_count:, :, :]

        sequence_dates = self.dates[actual_idx : actual_idx + self.group_size]
        t0 = sequence_dates[self.history_length - 1]  # last history step time (x[-1])
        month_idx = int(np.datetime64(t0, "M").astype(int) % 12 + 1)

        if self.return_metadata:
            return data, forcing, month_idx, sequence_dates.astype(np.float64)

        return data, forcing, month_idx


class SplitWrapper:
    def __init__(
        self,
        dataset,
        n_x,
        apply_smoothing: bool = False,
    ):
        self.dataset = dataset
        self.n_x = n_x
        self.apply_smoothing = apply_smoothing

        self.return_metadata = getattr(dataset, "return_metadata", False) or getattr(
            getattr(dataset, "dataset", None), "return_metadata", False
        )

        rank_zero_info(
            "SplitWrapper: blur {}".format(
                "enabled" if self.apply_smoothing else "disabled",
            )
        )

    def __getitem__(self, idx):
        if self.return_metadata:
            data, forcing, month_idx, sequence_dates = self.dataset[idx]
        else:
            data, forcing, month_idx = self.dataset[idx]

        # Split into x and y
        data_x = data[: self.n_x]
        data_y = data[self.n_x :]

        if self.apply_smoothing:
            with torch.no_grad():
                data_x = gaussian_smooth(data_x)
                data_y = gaussian_smooth(data_y)

        assert data_x.ndim == 4, "Invalid data_x shape: {}".format(data_x.shape)
        assert data_y.ndim == 4, "Invalid data_y shape: {}".format(data_y.shape)
        assert forcing.ndim == 4, "Invalid forcing shape: {}".format(forcing.shape)

        if self.return_metadata:
            dates_x = sequence_dates[: self.n_x]
            dates_y = sequence_dates[self.n_x :]

            return [(data_x, data_y), forcing, month_idx, (dates_x, dates_y)]

        return [(data_x, data_y), forcing, month_idx]

    def __len__(self):
        return len(self.dataset)


class cc2DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: list[str],
        prognostic_params: tuple[str, ...],
        forcing_params: tuple[str, ...],
        static_forcing_path: str | None = None,
        static_forcing_params: list[str] = [],
        train_start: str | None = None,
        train_end: str | None = None,
        val_start: str | None = None,
        val_end: str | None = None,
        test_start: str | None = None,
        test_end: str | None = None,
        predict_start: str | None = None,
        predict_end: str | None = None,
        seed: int = 0,
        batch_size: int = 32,
        num_workers: int = 6,
        persistent_workers: bool = True,
        prefetch_factor: int | None = 3,
        pin_memory: bool = True,
        normalization: dict[str, str] | None = None,
        apply_smoothing: bool = False,
        data_options: dict[str, str | int | list] = {},
    ):
        super().__init__()

        self.save_hyperparameters()

        self.ds_train: Subset | None = None
        self.ds_val: Subset | None = None
        self.ds_test: Subset | None = None
        self._full_dataset: AnemoiDataset | None = None  # Cache the full dataset
        self._forced_statistics: dict | None = None

    @property
    def input_resolution(self) -> tuple[int, int]:
        """Get input resolution from the dataset."""
        if self._full_dataset is None:
            # Trigger dataset creation if not already done
            self._get_or_create_full_dataset()
        return self._full_dataset.input_resolution

    def inject_statistics(self, stats: dict | None):
        self._forced_statistics = stats

    def _get_or_create_full_dataset(self) -> AnemoiDataset:
        if self._full_dataset is None:
            model = self.trainer.model
            model = model.module if hasattr(model, "module") else model

            self.history_length = model.hparams.history_length
            self.rollout_length = model.hparams.rollout_length

            group_size = self.history_length + self.rollout_length

            self._full_dataset = AnemoiDataset(
                zarr_path=self.hparams.data_path,
                group_size=group_size,
                prognostic_params=self.hparams.prognostic_params,
                forcing_params=self.hparams.forcing_params,
                static_forcing_path=self.hparams.static_forcing_path,
                static_forcing_params=self.hparams.static_forcing_params,
                normalization_methods=get_default_normalization_methods(
                    self.hparams.normalization
                ),
                data_options=self.hparams.data_options,
                statistics=self._forced_statistics,
            )

        return self._full_dataset

    def setup(self, stage: str | None = None):
        if stage == "fit" and self.ds_train is not None and self.ds_val is not None:
            return
        if stage == "test" and self.ds_test is not None:
            return
        if stage == "predict" and hasattr(self, "ds_predict"):
            return  # Use test for predict if no specific ds_predict

        ds_full = self._get_or_create_full_dataset()

        # Build splits aligned to valid_indices
        # 1) positions  into valid_indices
        valid_pos = np.arange(len(ds_full.valid_indices))
        # 2) actual start dates for each valid window
        all_dates = np.array(ds_full.dates, dtype="datetime64[ns]")
        actual_pos = np.array(ds_full.valid_indices, dtype=int)
        date_starts = all_dates[actual_pos]

        assert len(date_starts), "No valid dates found from dataset!"

        if (
            self.hparams.train_start
            and self.hparams.train_end
            and self.hparams.val_start
            and self.hparams.val_end
        ):
            # parse to numpy datetimes
            t0 = np.datetime64(self.hparams.train_start)
            t1 = np.datetime64(self.hparams.train_end)
            v0 = np.datetime64(self.hparams.val_start)
            v1 = np.datetime64(self.hparams.val_end)

            # only keep windows whose first date falls into each span
            train_mask = (date_starts >= t0) & (date_starts < t1)
            val_mask = (date_starts >= v0) & (date_starts < v1)

            train_indices = valid_pos[train_mask]
            val_indices = valid_pos[val_mask]

            num_training_samples = train_indices.shape[0]
            num_val_samples = val_indices.shape[0]

            if num_training_samples == 0 or num_val_samples == 0:
                raise ValueError(
                    f"Number of training samples={num_training_samples}, val samples={num_val_samples}"
                )
            rank_zero_info(
                "Training dataset ({} to {}) contains {} samples".format(
                    self.hparams.train_start,
                    self.hparams.train_end,
                    num_training_samples,
                )
            )
            rank_zero_info(
                "Val dataset ({} to {}) contains {} samples".format(
                    self.hparams.val_start, self.hparams.val_end, num_val_samples
                )
            )

        if self.hparams.test_start and self.hparams.test_end:
            t0 = np.datetime64(self.hparams.test_start)
            t1 = np.datetime64(self.hparams.test_end)

            test_mask = (date_starts >= t0) & (date_starts < t1)
            test_indices = valid_pos[test_mask]

            rank_zero_info(
                "Test dataset contains {} samples".format(test_indices.shape[0])
            )

        elif self.hparams.predict_start and self.hparams.predict_end:
            t0 = np.datetime64(self.hparams.predict_start)
            t1 = np.datetime64(self.hparams.predict_end)

            predict_mask = (date_starts >= t0) & (date_starts <= t1)
            predict_indices = valid_pos[predict_mask]

            rank_zero_info(
                "Predict dataset contains {} samples".format(predict_indices.shape[0])
            )

            assert (
                predict_indices.shape[0] > 0
            ), f"No samples found for prediction range {t0} to {t1}"

        # Create Subset datasets based on the stage
        # stage='fit' or stage=None will setup train and val
        if stage == "fit":
            self.ds_train = Subset(ds_full, train_indices)
            self.ds_val = Subset(ds_full, val_indices)

        elif stage == "validate":
            self.ds_val = Subset(ds_full, val_indices)

        # stage='test' or stage=None will setup test
        elif stage == "test" or stage is None:
            self.ds_test = Subset(ds_full, test_indices)

        # stage='predict' or stage=None will setup predict (using test set here)
        elif stage == "predict":
            self.ds_predict = Subset(ds_full, predict_indices)

        else:
            raise NotImplementedError(f"Unknown stage: {stage}")

    def _get_dataloader(
        self, dataset: Subset | None, shuffle: bool = False, stage: str | None = None
    ) -> DataLoader:
        if dataset is None:
            raise ValueError(
                "Dataset not available. Ensure setup() has been called for the correct stage."
            )

        if hasattr(dataset, "dataset") and isinstance(dataset.dataset, AnemoiDataset):
            # Set flag based on stage
            is_test_or_predict = stage == "test" or stage == "predict"
            dataset.dataset.return_metadata = is_test_or_predict

        wrapped_dataset = SplitWrapper(
            dataset=dataset,
            n_x=self.history_length,
            apply_smoothing=self.hparams.apply_smoothing,
        )

        return DataLoader(
            wrapped_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers
            and self.hparams.num_workers > 0,
            prefetch_factor=(
                self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None
            ),
        )

    def train_dataloader(self):
        return self._get_dataloader(self.ds_train, shuffle=True, stage="fit")

    def val_dataloader(self):
        if self.ds_val is None:
            # ensure validation split exists when validate-only runs
            self.setup(stage="validate")
        return self._get_dataloader(self.ds_val, shuffle=False, stage="fit")

    def test_dataloader(self):
        return self._get_dataloader(self.ds_test, shuffle=False, stage="test")

    def predict_dataloader(self):
        return self._get_dataloader(self.ds_predict, shuffle=False, stage="predict")
