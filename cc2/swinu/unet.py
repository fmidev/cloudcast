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
        ctx_channels: int = 32,
        ctx_detach: bool = True,  # for rollout_length=1
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.base = base_channels
        self.num_groups = num_groups
        self.ctx_channels = ctx_channels
        self.ctx_detach = ctx_detach

        self.alpha_bound = 0.1
        self.alpha_init = 1e-3
        self.alpha_raw = nn.Parameter(
            torch.tensor(self.alpha_init / self.alpha_bound, dtype=torch.float32)
        )

        self.ctx_1x1 = nn.Conv2d(latent_dim, ctx_channels, kernel_size=1, bias=False)

        # Encoder
        in_ch = latent_dim + ctx_channels

        self.in0 = ConvBlock(in_ch, base_channels, num_groups)
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
        self,
        x_tokens: torch.Tensor,
        H: int,
        W: int,
        ctx_tokens: torch.Tensor | None = None,
        return_diag: bool = False,
    ) -> torch.Tensor | tuple:
        assert x_tokens.dim() == 4, f"Expected [B,T,P,D], got {tuple(x_tokens.shape)}"

        B, T, P, D = x_tokens.shape

        assert P == H * W, "P mismatch: P={P} vs H*W={H*W} (H={H}, W={W})"
        assert D == self.latent_dim, "D mismatch: D={D} vs latent_dim={self.latent_dim}"

        xt = x_tokens.reshape(B * T, P, D).transpose(1, 2).contiguous()  # [B*T, D, P]
        x = xt.view(B * T, D, H, W)  # [B*T, D, H, W]

        # If H/W not divisible by 4, pad minimally (rare, but safe)
        pad_h = (4 - (H % 4)) % 4
        pad_w = (4 - (W % 4)) % 4
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[-2], x.shape[-1]

        # Context

        if ctx_tokens is None:
            ctx = torch.zeros(
                (B * T, self.ctx_channels, Hp, Wp), device=x.device, dtype=x.dtype
            )
        else:
            assert (
                ctx_tokens.dim() == 3
            ), f"ctx_tokens must be [B,P,D], got {tuple(ctx_tokens.shape)}"
            assert (
                ctx_tokens.shape[1] == H * W and ctx_tokens.shape[2] == D
            ), f"ctx_tokens mismatch: expected [B,{H*W},{D}], got {tuple(ctx_tokens.shape)}"

            ct = ctx_tokens
            if self.ctx_detach:
                ct = ct.detach()

            assert T == 1
            ct = (
                ct.reshape(B * T, H * W, D)
                .transpose(1, 2)
                .contiguous()
                .view(B * T, D, H, W)
            )
            if pad_h or pad_w:
                ct = F.pad(ct, (0, pad_w, 0, pad_h))
            ctx = self.ctx_1x1(ct)  # [BT, ctx_channels, Hp, Wp]

        xin = torch.cat([x, ctx], dim=1)  # [BT, D+ctx_channels, Hp, Wp]

        # Encoder
        x0 = self.in1(self.in0(xin))  # [BT, C, Hp, Wp]
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
                    "ctx_rms": (
                        float(ctx[..., :H, :W].pow(2).mean().sqrt().detach().cpu())
                        if ctx_tokens is not None
                        else 0.0
                    ),
                }
            return y_tokens, diag

        return y_tokens


class ObsStateUNetResidual(nn.Module):
    """
    State-space mini U-Net that maps core *full state* -> obs *full state*.

    Intended usage:
      x_core_next = core_state_{t+1}   # [B,T,C,H,W] (usually T=1 inside rollout loop)
      x_obs_next  = obs_head(x_core_next, ctx=..., ...)
    It does NOT feed back into the AR state updateI; it only produces the obs-space output.

    Design:
      - Residual path: y_full = x + alpha * r(x, ctx)
      - Optional overwrite via gate: y = x + gate * (y_full - x) = x + gate * alpha * r
        (gate is spatial [0,1], init biased to ~0 => starts as identity)
      - out_conv is zero-initialized => starts as (near) no-op
    """

    def __init__(
        self,
        base_channels: int = 64,
        num_groups: int = 8,
        ctx_channels: int = 32,  # extra context image channels (optional)
        ctx_token_dim: int | None = None,  # Dctx, if you will pass ctx_tokens
        ctx_detach: bool = True,
        use_gate: bool = True,
    ):
        super().__init__()
        self.base = base_channels
        self.num_groups = num_groups
        self.ctx_channels = ctx_channels
        self.ctx_token_dim = ctx_token_dim
        self.ctx_detach = ctx_detach
        self.use_gate = use_gate

        # residual scale (tanh bounded)
        self.alpha_bound = 0.1
        self.alpha_init = 1e-3
        self.alpha_raw = nn.Parameter(
            torch.tensor(self.alpha_init / self.alpha_bound, dtype=torch.float32)
        )

        # optional ctx compression (keeps concatenation small & consistent)
        # If ctx_channels == 0, ctx is ignored.
        self.ctx_1x1 = None
        if ctx_channels > 0:
            if self.ctx_token_dim is None:
                raise ValueError("ctx_token_dim must be set when ctx_channels > 0")
            self.ctx_1x1 = nn.Conv2d(
                self.ctx_token_dim, self.ctx_channels, kernel_size=1, bias=False
            )

        # Encoder
        in_ch = 1 + (ctx_channels if ctx_channels > 0 else 0)
        self.in0 = ConvBlock(in_ch, base_channels, num_groups)
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

        # Residual output head (predict residual in state space)
        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1, bias=True)

        # Optional spatial overwrite gate in [0,1]
        self.gate_head = None
        if use_gate:
            self.gate_head = nn.Conv2d(
                base_channels, 1, kernel_size=3, padding=1, bias=True
            )

        # Identity init: start as near no-op
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        if self.gate_head is not None:
            nn.init.zeros_(self.gate_head.weight)
            nn.init.constant_(
                self.gate_head.bias, -4.0
            )  # sigmoid ~ 0.018 => almost identity

    def forward(
        self,
        x_state: torch.Tensor,  # [B,T,C,H,W]
        ctx_tokens: torch.Tensor | None = None,  # [B,T,Cctx,H,W] or None
        return_diag: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        assert x_state.ndim == 5, f"Expected [B,T,C,H,W], got {tuple(x_state.shape)}"
        B, T, C, H, W = x_state.shape

        # We support T>1 by folding into batch (same as your latent head)
        x = x_state.reshape(B * T, C, H, W)

        # minimal padding so H,W divisible by 4 (2 downsamples)
        pad_h = (4 - (H % 4)) % 4
        pad_w = (4 - (W % 4)) % 4
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[-2], x.shape[-1]

        # Context handling
        if self.ctx_channels > 0:
            if ctx_tokens is None:
                ctx = torch.zeros(
                    (B * T, self.ctx_channels, Hp, Wp), device=x.device, dtype=x.dtype
                )
            else:
                assert (
                    ctx_tokens.ndim == 3
                ), f"ctx_tokens must be [B,P,Dctx], got {tuple(ctx_tokens.shape)}"
                assert ctx_tokens.shape[0] == B, "ctx_tokens B mismatch"
                assert (
                    ctx_tokens.shape[1] == H * W
                ), f"ctx_tokens P mismatch: got {ctx_tokens.shape[1]}, expected {H*W}"
                assert (
                    ctx_tokens.shape[2] == self.ctx_token_dim
                ), f"ctx_tokens D mismatch: got {ctx_tokens.shape[2]}, expected {self.ctx_token_dim}"
                ct = ctx_tokens.detach() if self.ctx_detach else ctx_tokens

                # If T>1, reuse same ctx for each time slice
                ct = (
                    ct.unsqueeze(1)
                    .expand(B, T, H * W, self.ctx_token_dim)
                    .reshape(B * T, H * W, self.ctx_token_dim)
                )
                ct = (
                    ct.transpose(1, 2)
                    .contiguous()
                    .view(B * T, self.ctx_token_dim, H, W)
                )

                if pad_h or pad_w:
                    ct = F.pad(ct, (0, pad_w, 0, pad_h))

                ctx = self.ctx_1x1(ct)  # [BT, ctx_channels, Hp, Wp]

            xin = torch.cat([x, ctx], dim=1)
        else:
            ctx = None
            xin = x

        # Encoder
        x0 = self.in1(self.in0(xin))  # [BT, Cb, Hp, Wp]
        x1 = self.down1(x0)  # [BT, 2Cb, Hp/2, Wp/2]
        x1 = self.d11(self.d10(x1))
        x2 = self.down2(x1)  # [BT, 4Cb, Hp/4, Wp/4]
        x2 = self.d21(self.d20(x2))

        # Decoder
        u1 = F.interpolate(x2, scale_factor=2, mode="nearest")
        u1 = self.up1(u1)
        if u1.shape[-2:] != x1.shape[-2:]:
            u1 = u1[..., : x1.shape[-2], : x1.shape[-1]]
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.u11(self.u10(u1))

        u0 = F.interpolate(u1, scale_factor=2, mode="nearest")
        u0 = self.up0(u0)
        if u0.shape[-2:] != x0.shape[-2:]:
            u0 = u0[..., : x0.shape[-2], : x0.shape[-1]]
        u0 = torch.cat([u0, x0], dim=1)
        u0 = self.u01(self.u00(u0))

        r = self.out_conv(u0)  # [BT, C, Hp, Wp]

        # unpad residual + (optional) gate
        if pad_h or pad_w:
            r = r[..., :H, :W]

        alpha = self.alpha_bound * torch.tanh(self.alpha_raw)

        # gated residual update in state space
        if self.gate_head is not None:
            gate = torch.sigmoid(self.gate_head(u0))  # [BT,1,Hp,Wp]
            if pad_h or pad_w:
                gate = gate[..., :H, :W]
            y = x[..., :H, :W] + gate * (alpha * r)
        else:
            gate = None
            y = x[..., :H, :W] + alpha * r

        y_state = y.view(B, T, C, H, W)

        if return_diag:
            with torch.no_grad():
                diag = {
                    "alpha": float(alpha.detach().cpu()),
                    "r_rms": float(r.pow(2).mean().sqrt().detach().cpu()),
                }
                if gate is not None:
                    diag.update(
                        {
                            "gate_mean": float(gate.mean().detach().cpu()),
                            "gate_p90": float(
                                torch.quantile(gate.flatten().float(), 0.90)
                                .detach()
                                .cpu()
                            ),
                        }
                    )
                if ctx is not None:
                    diag["ctx_rms"] = float(
                        ctx[..., :H, :W].pow(2).mean().sqrt().detach().cpu()
                    )
                else:
                    diag["ctx_rms"] = 0.0
            return y_state, diag

        return y_state
