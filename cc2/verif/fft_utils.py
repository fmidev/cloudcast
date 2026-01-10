# fft_utils.py
import torch
from torch.fft import rfftfreq, fftfreq


def ensure_btchw(x: torch.Tensor) -> torch.Tensor:
    """
    Accept [B,T,C,H,W] or [T,C,H,W] or [B,C,H,W].
    Return [B,T,C,H,W].
    """
    if x.dim() == 5:
        return x
    if x.dim() == 4:
        # Assume [B,C,H,W] -> T=1
        B, C, H, W = x.shape
        return x.view(B, 1, C, H, W)
    if x.dim() == 3:
        # Assume [T,C,H,W] -> B=1
        T, C, H, W = x.shape
        return x.view(1, T, C, H, W)
    raise ValueError(f"Unsupported shape {tuple(x.shape)}")


def radial_bins_rfft(Hf: int, Wf: int, device, n_bins=None, return_meta: bool = False):
    """
    Build radial bins for an rfft2 grid of shape [Hf, Wf], where Wf = W//2 + 1.
    Uses frequency units of cycles/cell (because d=1.0).
    Returns:
      bin_index [Hf,Wf], counts [n_bins], n_bins
    If return_meta:
      also returns edges [n_bins+1] in r_norm space and rmax (scalar) in cycles/cell.
    """
    fy = fftfreq(Hf, d=1.0, device=device)  # cycles/cell
    fx = rfftfreq(2 * (Wf - 1), d=1.0, device=device)  # cycles/cell
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")

    r_raw = torch.sqrt(FY**2 + FX**2)  # cycles/cell
    rmax = r_raw.max().clamp(min=1e-8)  # scalar
    r = r_raw / rmax  # normalized to [0,1]

    if n_bins is None:
        n_bins = max(8, min(Hf, 2 * (Wf - 1)) // 2)

    edges = torch.linspace(0, 1.0000001, n_bins + 1, device=device)
    bin_index = torch.bucketize(r.reshape(-1), edges) - 1
    bin_index = bin_index.reshape(Hf, Wf).clamp(0, n_bins - 1)

    counts = torch.bincount(bin_index.flatten(), minlength=n_bins).clamp(min=1)

    if return_meta:
        return bin_index, counts, n_bins, edges, rmax
    return bin_index, counts, n_bins


def band_masks_from_wavelengths(
    edges: torch.Tensor,
    rmax: torch.Tensor,
    dx_km: float,
    mid_km=(30.0, 60.0),
    high_km=30.0,
):
    """
    edges: [nb+1] in r_norm (0..1)
    rmax: scalar in cycles/cell (from radial_bins_rfft)
    dx_km: grid spacing in km
    mid_km: (lo, hi) wavelengths, e.g. (30,60) => 30-60 km band
    high_km: wavelength threshold for "high-k", e.g. 30 => <30 km
    Returns: mid_mask [nb], high_mask [nb], plus the corresponding r_norm cutoffs.
    """
    centers = 0.5 * (edges[:-1] + edges[1:])  # [nb], r_norm

    # Convert wavelength (km) -> cycles/cell: f_cell = dx / lambda
    # Then convert to r_norm by dividing by rmax (cycles/cell).
    def rnorm_from_lambda(lam_km: float) -> float:
        f_cell = dx_km / lam_km
        return f_cell / float(rmax)

    lam_lo, lam_hi = mid_km  # e.g. 30, 60
    r_mid_lo = rnorm_from_lambda(lam_hi)  # 60 km -> lower freq -> lower r_norm
    r_mid_hi = rnorm_from_lambda(lam_lo)  # 30 km -> higher freq -> higher r_norm
    r_high_lo = rnorm_from_lambda(high_km)

    mid_mask = (centers >= r_mid_lo) & (centers < r_mid_hi)
    high_mask = centers >= r_high_lo

    return mid_mask, high_mask, (r_mid_lo, r_mid_hi, r_high_lo)
