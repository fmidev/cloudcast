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

def radial_bins_rfft(Hf: int, Wf: int, device, n_bins=None):
    """
    Build radial bins for an rfft2 grid of shape [Hf, Wf], where Wf = W//2 + 1.
    Returns (bin_index [Hf,Wf], counts [n_bins], n_bins).
    """
    fy = fftfreq(Hf, d=1.0, device=device)           # [-0.5,0.5)
    fx = rfftfreq(2*(Wf-1), d=1.0, device=device)    # [0,0.5]
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(FY**2 + FX**2)
    r = r / r.max().clamp(min=1e-8)

    if n_bins is None:
        n_bins = max(8, min(Hf, 2*(Wf-1)) // 2)

    edges = torch.linspace(0, 1.0000001, n_bins+1, device=device)
    bin_index = torch.bucketize(r.reshape(-1), edges) - 1
    bin_index = bin_index.reshape(Hf, Wf).clamp(0, n_bins-1)

    counts = torch.bincount(bin_index.flatten(), minlength=n_bins).clamp(min=1)
    return bin_index, counts, n_bins
