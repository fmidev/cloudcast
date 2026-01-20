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
        use_obs_deep_net: bool = False,
    ):
        super().__init__()
        self.base = base_channels
        self.num_groups = num_groups
        self.ctx_channels = ctx_channels
        self.ctx_token_dim = ctx_token_dim
        self.ctx_detach = ctx_detach
        self.use_gate = use_gate
        self.use_obs_deep_net = use_obs_deep_net

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

        if self.use_obs_deep_net:
            bott_ch = base_channels * 4
            self.down3 = nn.Conv2d(
                bott_ch,
                bott_ch,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
            self.d30 = ConvBlock(bott_ch, bott_ch, num_groups)
            self.d31 = ConvBlock(bott_ch, bott_ch, num_groups)

            self.up2 = nn.Conv2d(bott_ch, bott_ch, kernel_size=1, bias=False)
            self.u20 = ConvBlock(
                bott_ch + bott_ch, bott_ch, num_groups
            )  # concat with x2
            self.u21 = ConvBlock(bott_ch, bott_ch, num_groups)

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
        divisor = 4
        if self.use_obs_deep_net:
            divisor = 8
        pad_h = (divisor - (H % divisor)) % divisor
        pad_w = (divisor - (W % divisor)) % divisor

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

        if self.use_obs_deep_net:
            x3 = self.down3(x2)  # [BT, 4Cb, Hp/8, Wp/8]
            x3 = self.d31(self.d30(x3))

            u2 = F.interpolate(x3, scale_factor=2, mode="nearest")
            u2 = self.up2(u2)
            if u2.shape[-2:] != x2.shape[-2:]:
                u2 = u2[..., : x2.shape[-2], : x2.shape[-1]]
            u2 = torch.cat([u2, x2], dim=1)
            u2 = self.u21(self.u20(u2))
            x2 = u2

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
