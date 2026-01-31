import torch
import numpy as np


class PSDStyleCalibrator:
    """Phase-preserving PSD style calibrator (NWCSAF -> CERRA) in logit space."""

    def __init__(self, params_path: str):
        params = torch.load(params_path, map_location="cpu")
        self.psd_target = params["psd_target"].double()
        self.rbin = params["rbin"].long()
        self.bincount = params["bincount"].double()
        self.window = params["window"].double()
        self.kernel = params["kernel"].double()
        self.n_iter = int(params["n_iter"])
        self.eps = float(params["eps"])
        self.gain_clip = float(params["gain_clip"])
        self.k_freeze = int(params["k_freeze"])
        self.k_taper_lo = int(params["k_taper_lo"])
        self.beta = float(params["beta"])
        self.device = torch.device("cpu")

        assert self.rbin.min().item() >= 0
        assert self.rbin.max().item() < self.psd_target.numel()

    def to(self, device):
        self.device = torch.device(device)
        self.psd_target = self.psd_target.to(self.device)
        self.rbin = self.rbin.to(self.device)
        self.bincount = self.bincount.to(self.device)
        self.window = self.window.to(self.device)
        self.kernel = self.kernel.to(self.device)
        return self

    @staticmethod
    def _logit(x, eps):
        x = torch.clamp(x, eps, 1.0 - eps)
        return torch.log(x / (1.0 - x))

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + torch.exp(-x))

    def _radial_psd(self, field):
        field_w = field * self.window
        fft = torch.fft.fft2(field_w)
        psd2 = (fft.abs() ** 2).double()
        rad = torch.bincount(
            self.rbin.flatten(),
            weights=psd2.flatten(),
            minlength=self.bincount.numel(),
        )
        return rad / torch.clamp(self.bincount, min=1.0)

    def _smooth_1d(self, arr):
        if self.kernel.numel() == 1:
            return arr
        x = arr.view(1, 1, -1)
        k = self.kernel.view(1, 1, -1)
        pad = self.kernel.numel() // 2
        return torch.nn.functional.conv1d(x, k, padding=pad).view(-1)

    def _soft_tail_weight(self, n):
        k = torch.arange(n, device=self.device, dtype=torch.float64)
        kmax = n - 1
        denom = 0.85 * kmax if kmax > 0 else 1.0
        return 1.0 / (1.0 + (k / denom) ** self.beta)

    def _gain_field(self, log_gain_1d):
        gain = torch.exp(log_gain_1d).float()
        return gain[self.rbin]  # [H,W]

    def __call__(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Accept [B,T,C,H,W] only for now.
        assert x.ndim == 5, f"Expected [B,T,C,H,W], got {x.shape}"
        y = x #.to(self.device, dtype=torch.float32)
        self.to(y.device)
        B, T, C, H, W = y.shape
        assert C == 1, "This calibrator currently assumes C==1."

        y_flat = y.view(B * T * C, H, W)
        y_work = self._logit(y_flat, self.eps)

        # Precompute weights (n is fixed = number of radial bins)
        n = self.psd_target.numel()
        w_tail = self._soft_tail_weight(n).view(1, -1)
        w_lo = torch.ones(n, device=self.device, dtype=torch.float64).view(1, -1)
        w_lo[:, : self.k_freeze] = 0.0
        lo_end = min(n, self.k_freeze + self.k_taper_lo)
        if lo_end > self.k_freeze:
            w_lo[:, self.k_freeze : lo_end] = torch.linspace(
                0.0,
                1.0,
                lo_end - self.k_freeze,
                device=self.device,
                dtype=torch.float64,
            ).view(1, -1)

        for _ in range(self.n_iter):
            psd_y = torch.stack(
                [self._radial_psd(y_work[i]) for i in range(y_work.shape[0])], dim=0
            )
            log_gain = 0.5 * (
                torch.log(self.psd_target + self.eps) - torch.log(psd_y + self.eps)
            )

            log_gain = torch.stack(
                [self._smooth_1d(log_gain[i]) for i in range(log_gain.shape[0])], dim=0
            )
            log_gain = torch.clamp(
                log_gain, -np.log(self.gain_clip), np.log(self.gain_clip)
            )
            log_gain = log_gain * w_tail * w_lo

            gain2d = torch.stack(
                [self._gain_field(log_gain[i]) for i in range(log_gain.shape[0])], dim=0
            )
            y_fft = torch.fft.fft2(y_work)
            y_work = torch.fft.ifft2(y_fft * gain2d).real

        out = torch.clamp(self._sigmoid(y_work), 0.0, 1.0)
        return out.view(B, T, C, H, W)
