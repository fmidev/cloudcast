import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_groups: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class LatentObsUNet2(nn.Module):
    """
    Latent-space mini U-Net that maps latent tokens -> latent tokens.
    Designed to act as an "observation operator" correction in latent space.

    Input/Output:
      - x_tokens: [B, T, P, D]  (T typically 1, P=H*W)
      - returns:  [B, T, P, D]

    Notes:
      - Applies a residual update: x + alpha * f(x)
      - alpha is tanh-bounded for stability
      - out_conv is zero-initialized => start as identity (no-op)
    """

    def __init__(
        self,
        latent_dim: int,
        base_channels: int = 64,
        num_groups: int = 8,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.base = base_channels
        self.num_groups = num_groups
        self.alpha_bound = 0.1
        self.alpha_init = 1e-3

        # alpha = bound * tanh(alpha_raw)
        # initialize so alpha ~= alpha_init
        # For small values: tanh(z) ~ z, so alpha_raw ~ alpha_init/bound
        self.alpha_raw = nn.Parameter(
            torch.tensor(self.alpha_init / self.alpha_bound, dtype=torch.float32)
        )

        # Encoder
        self.in0 = ConvBlock(latent_dim, base_channels, num_groups)
        self.in1 = ConvBlock(base_channels, base_channels, num_groups)

        self.down1 = nn.Conv2d(
            base_channels,
            base_channels * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.d10 = ConvBlock(base_channels * 2, base_channels * 2, num_groups)
        self.d11 = ConvBlock(base_channels * 2, base_channels * 2, num_groups)

        self.down2 = nn.Conv2d(
            base_channels * 2,
            base_channels * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.d20 = ConvBlock(base_channels * 4, base_channels * 4, num_groups)
        self.d21 = ConvBlock(base_channels * 4, base_channels * 4, num_groups)

        # Decoder
        self.up1 = nn.Conv2d(
            base_channels * 4, base_channels * 2, kernel_size=1, bias=False
        )
        self.u10 = ConvBlock(base_channels * 4, base_channels * 2, num_groups)  # concat
        self.u11 = ConvBlock(base_channels * 2, base_channels * 2, num_groups)

        self.up0 = nn.Conv2d(
            base_channels * 2, base_channels, kernel_size=1, bias=False
        )
        self.u00 = ConvBlock(base_channels * 2, base_channels, num_groups)  # concat
        self.u01 = ConvBlock(base_channels, base_channels, num_groups)

        # --- LF/HF residual heads + gate ---
        # HF: full-res residual in latent space
        self.out_hf = nn.Conv2d(
            base_channels, latent_dim, kernel_size=3, padding=1, bias=True
        )
        # LF: compute on pooled features -> upsample -> latent residual
        self.out_lf = nn.Conv2d(
            base_channels, latent_dim, kernel_size=3, padding=1, bias=True
        )
        # Gate: spatial confidence in [0,1]
        self.gate_head = nn.Conv2d(
            base_channels, 1, kernel_size=3, padding=1, bias=True
        )
        # Coarse pooling factor for LF branch (2 is conservative; 4 is stronger)
        self.lf_pool = 4

        # Identity init: start as near no-op
        nn.init.zeros_(self.out_hf.weight)
        nn.init.zeros_(self.out_hf.bias)
        nn.init.zeros_(self.out_lf.weight)
        nn.init.zeros_(self.out_lf.bias)
        # Gate starts small-ish (but alpha is already tiny); bias<0 pushes sigmoid->small
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -2.0)

    def forward(
        self, x_tokens: torch.Tensor, H: int, W: int, return_diag: bool = False
    ) -> torch.Tensor | tuple:
        if x_tokens.dim() != 4:
            raise ValueError(f"Expected [B,T,P,D], got {tuple(x_tokens.shape)}")
        B, T, P, D = x_tokens.shape
        if P != H * W:
            raise ValueError(f"P mismatch: P={P} vs H*W={H*W} (H={H}, W={W})")
        if D != self.latent_dim:
            raise ValueError(f"D mismatch: D={D} vs latent_dim={self.latent_dim}")

        xt = x_tokens.reshape(B * T, P, D).transpose(1, 2).contiguous()  # [B*T, D, P]
        x = xt.view(B * T, D, H, W)  # [B*T, D, H, W]

        # If H/W not divisible by 4, pad minimally (rare, but safe)
        pad_h = (4 - (H % 4)) % 4
        pad_w = (4 - (W % 4)) % 4
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[-2], x.shape[-1]

        # Encoder
        x0 = self.in1(self.in0(x))  # [BT, C, Hp, Wp]
        x1 = self.down1(x0)  # [BT, 2C, Hp/2, Wp/2]
        x1 = self.d11(self.d10(x1))
        x2 = self.down2(x1)  # [BT, 4C, Hp/4, Wp/4]
        x2 = self.d21(self.d20(x2))

        # Decoder
        u1 = F.interpolate(x2, scale_factor=2, mode="nearest")
        u1 = self.up1(u1)
        # crop skip if needed (due to padding)
        s1 = x1
        if u1.shape[-2:] != s1.shape[-2:]:
            u1 = u1[..., : s1.shape[-2], : s1.shape[-1]]
        u1 = torch.cat([u1, s1], dim=1)
        u1 = self.u11(self.u10(u1))

        u0 = F.interpolate(u1, scale_factor=2, mode="nearest")
        u0 = self.up0(u0)
        s0 = x0
        if u0.shape[-2:] != s0.shape[-2:]:
            u0 = u0[..., : s0.shape[-2], : s0.shape[-1]]
        u0 = torch.cat([u0, s0], dim=1)
        u0 = self.u01(self.u00(u0))

        # --- LF branch: pooled -> residual -> upsample ---
        if self.lf_pool > 1:
            u0_lf = F.avg_pool2d(
                u0, kernel_size=self.lf_pool, stride=self.lf_pool, ceil_mode=False
            )
            r_lf = self.out_lf(u0_lf)
            r_lf = F.interpolate(r_lf, size=u0.shape[-2:], mode="nearest")
        else:
            r_lf = self.out_lf(u0)

        # --- HF branch: full-res residual ---
        r_hf = self.out_hf(u0)

        # Spatial gate in [0,1]
        gate = torch.sigmoid(self.gate_head(u0))  # [BT,1,Hp,Wp]

        # Combine residuals (gate applied to both)
        r = gate * (r_lf + r_hf)  # [BT,D,Hp,Wp]

        # unpad
        if pad_h or pad_w:
            r = r[..., :H, :W]
            gate = gate[..., :H, :W]

        alpha = self.alpha_bound * torch.tanh(self.alpha_raw)
        y = x[..., :H, :W] + alpha * r  # residual in latent space (gated LF+HF)
        yt = y.view(B * T, D, H * W).transpose(1, 2).contiguous()  # [BT, P, D]
        # back to tokens: [BT, D, H, W] -> [B,T,P,D]
        y_tokens = yt.view(B, T, H * W, D)

        if return_diag:
            with torch.no_grad():
                diag = {
                    "alpha": float(alpha.detach().cpu()),
                    "gate_mean": float(gate.mean().detach().cpu()),
                    "gate_p90": float(
                        torch.quantile(gate.detach().flatten().float(), 0.90).cpu()
                    ),
                    "r_lf_rms": float(r_lf.pow(2).mean().sqrt().detach().cpu()),
                    "r_hf_rms": float(r_hf.pow(2).mean().sqrt().detach().cpu()),
                    "r_rms": float(r.pow(2).mean().sqrt().detach().cpu()),
                }
            return y_tokens, diag

        return y_tokens
