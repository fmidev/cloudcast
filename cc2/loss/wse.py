import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from pytorch_wavelets import DTCWTForward


class WaveletSpectralLoss(nn.Module):
    """
    Local, banded wavelet spectral loss.

    Concept:
      - compute wavelet detail coefficients at each level j
      - form local energy maps P_j(x,y) = sum_orient |d_{j,o}(x,y)|^2
      - group levels into bands (e.g. small/medium/large)
      - for each band, penalize mismatch between local band power maps
        using a sqrt-energy (amplitude) loss.
    """

    def __init__(
        self,
        J: int = 6,  # number of wavelet levels
        bands: dict = {
            "small": [1, 2],
            "medium": [3],
            "large": [4, 5, 6],
        },  # < 30km, ~30-60km, >60-100km
        band_weights: dict = {"small": 0.5, "medium": 2, "large": 1},
    ):
        super().__init__()
        self.J = J
        self.dtcw = DTCWTForward(J=J)
        self.bands = bands
        self.band_weights = band_weights

        # convenience: map each level -> band name(s)
        # (levels are 1..J, Yh index is 0..J-1)
        level_to_bands = {j: [] for j in range(1, J + 1)}
        for bname, lvls in self.bands.items():
            for j in lvls:
                if j < 1 or j > J:
                    raise ValueError(f"Level {j} in band '{bname}' is outside [1, {J}]")
                level_to_bands[j].append(bname)
        self.level_to_bands = level_to_bands

    def _level_powers(self, Yh_pred, Yh_true):
        """
        Compute local power maps at each level:
           P_j_pred, P_j_true : list length J
           each element has shape [BT, C, Hj, Wj]
        """
        Pj_pred = []
        Pj_true = []

        for j in range(self.J):
            # Yh[j]: [BT, C, n_orient, Hj, Wj]
            yh_p = Yh_pred[j]
            yh_t = Yh_true[j]

            # Sum over orientations to get local energy at this scale
            # shape: [BT, C, Hj, Wj]
            Pp = (yh_p**2).sum(dim=2)
            Pt = (yh_t**2).sum(dim=2)

            Pj_pred.append(Pp)
            Pj_true.append(Pt)

        return Pj_pred, Pj_true

    @staticmethod
    def _amplitude_loss(P_pred, P_true, eps: float = 1e-8):
        """
        Local amplitude mismatch:
            E[ (sqrt(P_pred) - sqrt(P_true))^2 ]
        P_*: tensor [BT, C, Hj, Wj]
        """
        a_pred = (P_pred + eps).sqrt()
        a_true = (P_true + eps).sqrt()
        return ((a_pred - a_true) ** 2).mean()

    def _band_losses(self, Pj_pred, Pj_true):
        """
        Aggregate per-level local power into band-wise losses.
        Pj_pred, Pj_true: lists length J of [BT, C, Hj, Wj]

        Returns:
            band_losses: dict[band_name] = scalar tensor
            level_losses: dict[(band_name, level)] = scalar tensor
        """
        band_losses = {bname: [] for bname in self.bands.keys()}
        level_losses = {}

        for level in range(1, self.J + 1):
            Pp = Pj_pred[level - 1]
            Pt = Pj_true[level - 1]
            L_level = self._amplitude_loss(Pp, Pt)
            # assign this level's loss to all bands that include it
            for bname in self.level_to_bands[level]:
                band_losses[bname].append(L_level)
                level_losses[(bname, level)] = L_level.detach()

        # average contributions within each band
        for bname, losses in band_losses.items():
            if len(losses) == 0:
                # possible if a band has no levels assigned
                band_losses[bname] = torch.zeros((), device=Pj_pred[0].device)
            else:
                band_losses[bname] = torch.stack(losses).mean()

        return band_losses, level_losses

    def forward(self, y_pred_full: torch.Tensor, y_true_full: torch.Tensor, **kwargs):
        y_true = y_true_full
        y_pred = y_pred_full

        # Normalise to [B, T, C, H, W]
        if y_true.dim() == 4:
            y_true = y_true.unsqueeze(1)
            y_pred = y_pred.unsqueeze(1)

        B, T, C, H, W = y_pred.shape
        assert C >= 1, f"Expected at least one channel, got C={C}"

        # merge B and T for transform: [BT, C, H, W]
        y_pred_bt = y_pred.reshape(B * T, C, H, W)
        y_true_bt = y_true.reshape(B * T, C, H, W)

        device = y_pred_bt.device
        self.dtcw = self.dtcw.to(device)

        with autocast(enabled=False):
            y_pred_bt = y_pred_bt.float()
            y_true_bt = y_true_bt.float()

            Yl_pred, Yh_pred = self.dtcw(y_pred_bt)
            Yl_true, Yh_true = self.dtcw(y_true_bt)

            # per-level local power maps
            Pj_pred, Pj_true = self._level_powers(Yh_pred, Yh_true)

            # band-wise losses from local power
            band_losses, level_losses = self._band_losses(Pj_pred, Pj_true)

            # combine bands with weights
            wavelet_loss = 0.0
            for bname, Lb in band_losses.items():
                w = self.band_weights.get(bname, 1.0)
                wavelet_loss = wavelet_loss + w * Lb

        assert torch.isfinite(wavelet_loss), f"Non-finite loss: {wavelet_loss}"

        # diagnostics
        metrics = {
            "loss": wavelet_loss,
        }

        # per-band diagnostics
        for bname, Lb in band_losses.items():
            metrics[f"wav_band_loss_{bname}"] = Lb.detach()

        # (optional) per-level diagnostics
        for (bname, level), L_level in level_losses.items():
            metrics[f"wav_level_loss_{bname}_L{level}"] = L_level

        return metrics
