import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from math import sqrt, ceil
from timm.models.layers import DropPath


def make_window_key_mask(H, W, Hpad, Wpad, ws, device, repeat_factor):
    # 1 inside real image, 0 in padded area
    base = torch.zeros((Hpad, Wpad), dtype=torch.bool, device=device)
    base[:H, :W] = False  # valid -> False (not masked)
    base[H:, :] = True  # pad -> True
    base[:, W:] = True
    nH, nW = Hpad // ws, Wpad // ws
    mask = (
        base.view(nH, ws, nW, ws).permute(0, 2, 1, 3).reshape(nH * nW, ws * ws)
    )  # [nWin, L]
    mask = mask.unsqueeze(0).repeat(repeat_factor, 1, 1)  # [B*T, nWin, L]
    mask = mask.reshape(-1, ws * ws)  # [Bwin, L]
    return mask


def get_padded_size(H, W, patch_size, num_merges=1):
    # Calculate required factor for divisibility
    required_factor = patch_size * (2**num_merges)

    # Calculate target dimensions (must be divisible by required_factor)
    target_h = ((H + required_factor - 1) // required_factor) * required_factor
    target_w = ((W + required_factor - 1) // required_factor) * required_factor

    return target_h, target_w


def pad_tensor(tensor, patch_size, num_merges=1):
    H, W = tensor.shape[-2:]
    target_h, target_w = get_padded_size(H, W, patch_size, num_merges)

    # Calculate padding needed
    pad_h = target_h - H
    pad_w = target_w - W

    # Create padding configuration (left, right, top, bottom)
    pad_left = 0
    pad_right = pad_w
    pad_top = 0
    pad_bottom = pad_h

    # Apply padding
    if pad_h > 0 or pad_w > 0:
        padded_tensor = F.pad(
            tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0
        )
    else:
        padded_tensor = tensor

    # Store padding info for later de-padding
    padding_info = {
        "original_size": (H, W),
        "padded_size": (target_h, target_w),
        "pad_h": pad_h,
        "pad_w": pad_w,
    }

    return padded_tensor, padding_info


def pad_tensors(tensors, patch_size, num_merges=1):
    padded_list = []
    for t in tensors:
        padded, pad_info = pad_tensor(t, patch_size, num_merges)
        padded_list.append(padded)

    return padded_list, pad_info


def depad_tensor(tensor, padding_info):
    if padding_info["pad_h"] == 0 and padding_info["pad_w"] == 0:
        return tensor

    H, W = tensor.shape[-2:]
    original_h, original_w = padding_info["original_size"]

    # Handle case where tensor has been processed and dimensions changed
    # Scale the original dimensions to match current tensor's scale
    scale_h = H / padding_info["padded_size"][0]
    scale_w = W / padding_info["padded_size"][1]

    target_h = int(original_h * scale_h)
    target_w = int(original_w * scale_w)

    # Extract the unpadded region
    depadded_tensor = tensor[:, :, :, :target_h, :target_w]

    return depadded_tensor


class PatchEmbedLossless(nn.Module):
    def __init__(
        self,
        input_resolution,
        patch_size,
        data_channels,
        forcing_channels,
        embed_dim,
        data_dim_override=None,
    ):
        super().__init__()
        assert type(input_resolution) in (list, tuple)
        H, W = input_resolution
        if isinstance(patch_size, (list, tuple)):
            ps = int(patch_size[0])
            assert patch_size[0] == patch_size[1], "use square patches"
        else:
            ps = int(patch_size)
        self.input_resolution = (H, W)
        self.patch_size = (ps, ps)

        # patch grid
        self.h_patches = ceil(H / ps)
        self.w_patches = ceil(W / ps)
        self.num_patches = self.h_patches * self.w_patches

        self.Cd = int(data_channels)
        self.Cf = int(forcing_channels)
        self.embed_dim = int(embed_dim)

        self.unfold_data = nn.Unfold(kernel_size=ps, stride=ps)  # exact

        # Choose d_data so we can reconstruct data exactly (start with full C*ps*ps, but cap by embed_dim-1)
        full_data_dim = self.Cd * ps * ps
        if data_dim_override is None:
            self.d_data = (
                min(full_data_dim, self.embed_dim - 1)
                if self.embed_dim >= 2
                else self.embed_dim
            )
        else:
            self.d_data = int(data_dim_override)
            assert 1 <= self.d_data <= self.embed_dim, "invalid data_dim_override"

        # If we reduced data dimension, use a learnable projection. If equal, start as identity.
        self.proj_data = nn.Linear(full_data_dim, self.d_data, bias=True)
        with torch.no_grad():
            nn.init.zeros_(self.proj_data.bias)
            # identity-ish init on the first d_data rows
            eye = torch.eye(full_data_dim, self.d_data)
            self.proj_data.weight.copy_(eye.t())

        # --- Forcings: also patchify (exact), then compress to remaining dims ---
        self.unfold_force = nn.Unfold(kernel_size=ps, stride=ps)
        self.d_force = max(0, self.embed_dim - self.d_data)
        if self.d_force > 0:
            self.proj_force = nn.Linear(self.Cf * ps * ps, self.d_force, bias=True)
            nn.init.xavier_uniform_(self.proj_force.weight)
            nn.init.zeros_(self.proj_force.bias)
        else:
            self.proj_force = None

        # Optional: a light fusion to keep interface identical (no change in dim)
        self.fusion = (
            nn.Identity()
        )  # keep simple; you can swap to nn.Linear(embed_dim, embed_dim)

    def forward(self, data, forcing):
        """
        data, forcing: [B,T,C,H,W]
        returns: x_tokens [B,T,P,embed_dim], f_future [B,1,P,embed_dim] or None
        """
        assert data.ndim == 5 and forcing.ndim == 5
        B, T, Cd, H, W = data.shape
        _, Tf, Cf, _, _ = forcing.shape
        assert Cd == self.Cd and Cf == self.Cf

        tokens_per_t = []
        for t in range(T):
            # exact patchify
            u_d = self.unfold_data(data[:, t])  # [B, Cd*ps*ps, P]
            u_f = self.unfold_force(forcing[:, t])  # [B, Cf*ps*ps, P]
            P = u_d.shape[-1]
            # to [B,P,*]
            u_d = u_d.transpose(1, 2)
            u_f = u_f.transpose(1, 2)

            # project to token dims
            td = self.proj_data(u_d)  # [B,P,d_data]
            if self.d_force > 0:
                tf = self.proj_force(u_f)  # [B,P,d_force]
                tok = torch.cat([td, tf], dim=-1)  # [B,P,embed_dim]
            else:
                tok = td
            # optional fusion (kept Identity)
            tok = self.fusion(tok)
            tokens_per_t.append(tok)

        x_tokens = torch.stack(tokens_per_t, dim=1)  # [B,T,P,D]

        # future forcing token (kept for API compatibility)
        f_future = None
        if Tf > T:
            u_ff = self.unfold_force(forcing[:, T]).transpose(1, 2)  # [B,P,Cf*ps*ps]
            if self.d_force > 0:
                tfut = self.proj_force(u_ff)  # [B,P,d_force]
                # pad zeros for the data part to keep same D
                zeros = torch.zeros(
                    B, u_ff.shape[1], self.d_data, device=u_ff.device, dtype=u_ff.dtype
                )
                f_future = torch.cat([zeros, tfut], dim=-1)  # [B,P,D]
            else:
                f_future = torch.zeros(
                    B,
                    u_ff.shape[1],
                    self.embed_dim,
                    device=u_ff.device,
                    dtype=u_ff.dtype,
                )
            f_future = self.fusion(f_future).unsqueeze(1)  # [B,1,P,D]

        return x_tokens, f_future


class ProjectToImageFold(nn.Module):
    def __init__(
        self,
        input_resolution,
        patch_size,
        out_channels,
        d_data,
        embed_dim,
        undo_scale: bool = False,
    ):
        super().__init__()
        H, W = input_resolution
        if isinstance(patch_size, (list, tuple)):
            ps = int(patch_size[0])
        else:
            ps = int(patch_size)
        self.ps = ps
        self.input_resolution = (H, W)
        self.out_channels = int(out_channels)
        self.embed_dim = int(embed_dim)
        self.d_data = int(d_data)  # how many leading dims are the data sub-token

        self.norm_final = nn.LayerNorm(self.embed_dim)  # keep your interface
        self.unproj = nn.Linear(self.d_data, self.out_channels * ps * ps, bias=True)
        with torch.no_grad():
            nn.init.zeros_(self.unproj.bias)
            # start close to identity on the first channels if sizes match
            eye = torch.eye(self.out_channels * ps * ps, self.d_data)
            self.unproj.weight.copy_(eye)

        self.undo_scale = bool(undo_scale)

    def forward(self, x_tokens_bt_p_d):
        """
        x_tokens: [B,T,P,D] (D==embed_dim). We only use the first d_data dims to reconstruct.
        returns: [B,T,out_channels,H,W]
        """
        B, T, P, D = x_tokens_bt_p_d.shape
        assert D == self.embed_dim, f"expected token dim {self.embed_dim}, got {D}"

        # layernorm to match your original head contract
        x = self.norm_final(x_tokens_bt_p_d)

        # split off the data sub-token
        x_data = x[..., : self.d_data]  # [B,T,P,d_data]

        # unproject per-patch to pixels
        flat = x_data.reshape(B * T, P, self.d_data)  # [B*T,P,d_data]
        pix = self.unproj(flat)  # [B*T,P,C*ps*ps]
        if self.undo_scale:
            pix = pix / float(
                self.ps * self.ps
            )  # undo any patch_area_scale, if you keep it elsewhere
        pix = pix.transpose(1, 2)  # [B*T,C*ps*ps,P]

        # exact unpatch
        H, W = self.input_resolution
        Ho = ceil(H / self.ps) * self.ps
        Wo = ceil(W / self.ps) * self.ps
        y = F.fold(
            pix, output_size=(Ho, Wo), kernel_size=self.ps, stride=self.ps
        )  # [B*T,C,Ho,Wo]
        y = y.view(B, T, self.out_channels, Ho, Wo)
        # If you pad elsewhere, your outer forward will depad to real size.
        return y


class DWConvResidual3D(nn.Module):
    """
    Drop-in local mixing for token sequences.
    - Input:  [B, L, C] where L == T * (h*w)
    - Output: [B, L, C] (same shape)
    Internally reshapes to [B*T, C, h, w], applies DW/PW convs, and reshapes back.
    """

    def __init__(
        self,
        C: int,
        grid_hw: tuple[int, int],
        time_dim: int,
        expand: float = 2.0,
        dilation: int = 1,
        ls_init: float = 1e-2,
    ):
        super().__init__()
        self.C = int(C)
        self.h, self.w = int(grid_hw[0]), int(grid_hw[1])
        self.T = int(time_dim)  # history_length (or the T at this stage)
        hidden = max(1, int(self.C * expand))

        self.dw = nn.Conv2d(
            self.C,
            self.C,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=self.C,
            bias=True,
        )
        self.pw1 = nn.Conv2d(self.C, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, self.C, kernel_size=1, bias=True)

        # tiny LayerScale so it's near-identity at init
        self.ls = nn.Parameter(torch.ones(self.C) * ls_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        assert x.dim() == 3, f"DWConvResidual3D expects [B,L,C], got {tuple(x.shape)}"
        B, L, C = x.shape
        assert C == self.C, f"C={C} != {self.C}"
        P = self.h * self.w
        expected_L = self.T * P
        assert L == expected_L, f"L={L} != T*P={self.T}*{P}={expected_L}"

        # [B, L(=T*P), C] -> [B, T, P, C] -> [B*T, C, h, w]
        x_btpc = x.view(B, self.T, P, C)
        x_2d = (
            x_btpc.view(B * self.T, self.h, self.w, C).permute(0, 3, 1, 2).contiguous()
        )

        y = self.dw(x_2d)
        y = self.pw2(self.act(self.pw1(y)))
        y = y * self.ls.view(1, -1, 1, 1) + x_2d

        # back to [B, L, C]
        y_btpc = y.permute(0, 2, 3, 1).contiguous().view(B, self.T, P, C)
        y_seq = y_btpc.view(B, L, C)
        return y_seq


class PatchExpand(nn.Module):
    def __init__(self, input_dim, output_dim, scale_factor=2):
        super().__init__()
        self.dim = input_dim
        self.expand = nn.Linear(input_dim, output_dim * scale_factor**2)
        self.scale_factor = scale_factor

    def forward(self, x, H, W):
        # x: [B, H*W, C]
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size: {} vs {}".format(L, H * W)

        x = self.expand(x)  # B, H*W, C*scale_factor^2

        # Reshape to spatial format for upsampling
        x = x.view(B, H, W, -1)
        x = rearrange(
            x,
            "b h w (p1 p2 c) -> b (h p1) (w p2) c",
            p1=self.scale_factor,
            p2=self.scale_factor,
        )
        x = rearrange(x, "b h w c -> b (h w) c")

        return x, H * self.scale_factor, W * self.scale_factor


class PatchMerge(nn.Module):
    def __init__(self, input_resolution, dim, time_dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
        self.time_dim = time_dim

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        T = self.time_dim

        assert L == T * H * W, f"input feature has wrong size: {L} != {T * H * W}"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.reshape(B, T, H * W, C)

        # Process each time step separately
        output_time_steps = []

        for t in range(T):
            xt = x[:, t, :, :]  # B H*W C
            xt = xt.reshape(B, H, W, C)

            x0 = xt[:, 0::2, 0::2, :]  # B H/2 W/2 C
            x1 = xt[:, 1::2, 0::2, :]  # B H/2 W/2 C
            x2 = xt[:, 0::2, 1::2, :]  # B H/2 W/2 C
            x3 = xt[:, 1::2, 1::2, :]  # B H/2 W/2 C
            xt = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
            xt = self.norm(xt)
            xt = self.reduction(xt)
            xt = xt.reshape(B, (H // 2) * (W // 2), 2 * C)  # [B, H/2*W/2, 2*C]
            output_time_steps.append(xt)

        # Recombine time steps
        x = torch.cat(
            [step.unsqueeze(1) for step in output_time_steps], dim=1
        )  # [B, T, H/2*W/2, 2*C]
        x = x.reshape(B, T * (H // 2) * (W // 2), 2 * C)  # [B, T*H/2*W/2, 2*C]
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SwinEncoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio,
        qkv_bias,
        drop,
        attn_drop,
        drop_path_rate,
        window_size,
        shift_size,
        H,
        W,
        T,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.H = H
        self.W = W
        self.T = T

        assert 0 <= self.shift_size < self.window_size, "shift must be < window_size"

        self.norm1 = nn.LayerNorm(dim)

        self.attn = WindowAttentionRPB(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=qkv_bias,
            batch_first=True,
            window_size=window_size,
        )

        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, hidden_dim=int(dim * mlp_ratio), dropout=drop)

        self.gamma_attn = nn.Parameter(torch.ones(1))
        self.gamma_mlp = nn.Parameter(torch.ones(1))

    def _partition_windows(self, x4):
        """
        x4: [B*, Hpad, Wpad, C]
        returns windows: [Bwin, Ws*Ws, C], plus Hpad, Wpad to reverse later
        """
        B_, H, W, C = x4.shape
        ws = self.window_size
        assert (
            H % ws == 0 and W % ws == 0
        ), f"Input ({H},{W}) not divisible by window_size {ws}"
        nH, nW = H // ws, W // ws

        # [B, H, W, C] -> [B, nH, ws, nW, ws, C]
        xw = x4.view(B_, nH, ws, nW, ws, C)
        # [B, nH, nW, ws, ws, C]
        xw = xw.permute(0, 1, 3, 2, 4, 5).contiguous()
        # flatten windows
        xw = xw.view(B_ * nH * nW, ws * ws, C)
        return xw, nH, nW

    def _reverse_windows(self, xw, nH, nW):
        """
        xw: [Bwin, Ws*Ws, C] -> [B*, Hpad, Wpad, C]
        """
        ws = self.window_size
        Bwin, L, C = xw.shape
        B_ = Bwin // (nH * nW)
        xw = xw.view(B_, nH, nW, ws, ws, C)
        # [B, nH, ws, nW, ws, C] -> [B, H, W, C]
        x4 = xw.permute(0, 1, 3, 2, 4, 5).contiguous().view(B_, nH * ws, nW * ws, C)
        return x4

    def forward(self, x):
        """
        x: [B, T*P, C] where P = H*W tokens
        """
        B, N, C = x.shape
        H, W, T = self.H, self.W, self.T
        P = H * W

        assert N % P == 0, f"Expected N multiple of P=H*W={P}, got N={N}"
        assert (N // P) == T, f"N={N} corresponds to T={N//P}, expected T={T}"

        x_attn_in = x  # save residuals
        y = self.norm1(x)

        # [B, T*P, C] -> [B, T, P, C] -> [B*T, P, C] -> [B*T, H, W, C]
        y = y.view(B, T, P, C).reshape(B * T, P, C).view(B * T, H, W, C)

        # cyclic shift (on spatial dims)
        if self.shift_size > 0:
            y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # pad H/W to multiples of window_size if needed
        ws = self.window_size
        pad_h = (ws - (H % ws)) % ws
        pad_w = (ws - (W % ws)) % ws

        if pad_h or pad_w:
            y = F.pad(y, (0, 0, 0, pad_w, 0, pad_h))
        Hpad, Wpad = H + pad_h, W + pad_w
        nH, nW = Hpad // ws, Wpad // ws
        Bmul = B * T
        Ppad = Hpad * Wpad

        # build per-window key mask (no cost when no pad)
        key_mask = None
        if (Hpad != H) or (Wpad != W):
            key_mask = make_window_key_mask(
                H, W, Hpad, Wpad, ws, y.device, repeat_factor=Bmul
            )  # [Bwin, L]

        # partition -> [Bwin, L, C]
        yw = (
            y.view(Bmul, nH, ws, nW, ws, C)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(Bmul * nH * nW, ws * ws, C)
        )

        # self-attention with RPB + mask
        attn_out, _ = self.attn(yw, key_padding_mask=key_mask)  # drop-in replacement

        # reverse windows -> [B*T, Hpad, Wpad, C]
        y = (
            attn_out.view(Bmul, nH, nW, ws, ws, C)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(Bmul, Hpad, Wpad, C)
        )

        # unpad, reverse shift, reshape back (as you already do)
        if (Hpad != H) or (Wpad != W):
            y = y[:, :H, :W, :]
        if self.shift_size > 0:
            y = torch.roll(y, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        y = y.reshape(B, T, P, C).reshape(B, T * P, C)

        # residual + MLP
        x = x + x_attn_in + self.drop_path(self.gamma_attn * y)
        x = x + self.drop_path(self.gamma_mlp * self.mlp(self.norm2(x)))
        return x


class SwinDecoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio,
        qkv_bias,
        drop,
        attn_drop,
        drop_path_rate,
        window_size,
        shift_size,
        H,
        W,
        T,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.H = H
        self.W = W
        self.T = T
        self.P = H * W

        assert 0 <= self.shift_size < self.window_size, "shift must be < window_size"

        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = WindowAttentionRPB(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=qkv_bias,
            batch_first=True,
            window_size=window_size,
        )

        self.drop_path1 = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.gamma_attn1 = nn.Parameter(torch.ones(1))

        self.norm2 = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)

        self.cross_attn = WindowAttentionRPB(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=qkv_bias,
            batch_first=True,
            window_size=window_size,
        )
        self.drop_path2 = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.gamma_attn2 = nn.Parameter(torch.ones(1))

        # --- 3. MLP ---
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, hidden_dim=int(dim * mlp_ratio), dropout=drop)
        self.drop_path3 = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.gamma_mlp = nn.Parameter(torch.ones(1) * 0.1)

    def _partition_windows(self, x4):
        B_, H, W, C = x4.shape
        ws = self.window_size
        assert (
            H % ws == 0 and W % ws == 0
        ), f"Input ({H},{W}) not divisible by window_size {ws}"
        nH, nW = H // ws, W // ws

        # [B, H, W, C] -> [B, nH, ws, nW, ws, C]
        xw = x4.view(B_, nH, ws, nW, ws, C)
        # [B, nH, nW, ws, ws, C]
        xw = xw.permute(0, 1, 3, 2, 4, 5).contiguous()
        # flatten windows
        xw = xw.view(B_ * nH * nW, ws * ws, C)
        return xw, nH, nW

    def _reverse_windows(self, xw, nH, nW):
        ws = self.window_size
        Bwin, L, C = xw.shape
        B_ = Bwin // (nH * nW)
        xw = xw.view(B_, nH, nW, ws, ws, C)
        # [B, nH, ws, nW, ws, C] -> [B, H, W, C]
        x4 = xw.permute(0, 1, 3, 2, 4, 5).contiguous().view(B_, nH * ws, nW * ws, C)
        return x4

    def _window_attn(self, x, attn_layer, kv_in=None):
        B, N, C = x.shape
        H, W, T, P = self.H, self.W, self.T, self.P
        ws = self.window_size

        # reshape Q and KV to [B*T, H, W, C]
        q = x.view(B, T, P, C).reshape(B * T, P, C).view(B * T, H, W, C)
        kv = (
            q
            if kv_in is None
            else kv_in.view(B, T, P, C).reshape(B * T, P, C).view(B * T, H, W, C)
        )

        # cyclic shift
        if self.shift_size > 0:
            q = torch.roll(q, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            kv = torch.roll(
                kv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )

        # pad
        pad_h = (ws - (H % ws)) % ws
        pad_w = (ws - (W % ws)) % ws
        if pad_h or pad_w:
            q = F.pad(q, (0, 0, 0, pad_w, 0, pad_h))
            kv = F.pad(kv, (0, 0, 0, pad_w, 0, pad_h))
        Hpad, Wpad = H + pad_h, W + pad_w
        nH, nW = Hpad // ws, Wpad // ws
        Bmul = B * T

        # build mask (shared for self/cross; it applies to keys)
        key_mask = None
        if (Hpad != H) or (Wpad != W):
            key_mask = make_window_key_mask(
                H, W, Hpad, Wpad, ws, q.device, repeat_factor=Bmul
            )

        # windows -> [Bwin, L, C]
        q_win = (
            q.view(Bmul, nH, ws, nW, ws, C)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(Bmul * nH * nW, ws * ws, C)
        )
        kv_win = (
            kv.view(Bmul, nH, ws, nW, ws, C)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(Bmul * nH * nW, ws * ws, C)
        )

        # RPB attention (+mask)
        out, _ = attn_layer(q_win, kv_win, kv_win, key_padding_mask=key_mask)  # drop-in

        # merge, unpad, reverse shift, reshape back
        y = (
            out.view(Bmul, nH, nW, ws, ws, C)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(Bmul, Hpad, Wpad, C)
        )
        if (Hpad != H) or (Wpad != W):
            y = y[:, :H, :W, :]
        if self.shift_size > 0:
            y = torch.roll(y, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        y = y.reshape(B, T, P, C).reshape(B, T * P, C)
        return y

    def forward(self, x, context):
        """
        x: [B, N, C] (from previous decoder layer) N = T*P
        context: [B, N, C] (from encoder skip connection) N = T*P
        """

        # --- 1. Windowed Self-Attention ---
        # x = x + drop_path(gamma * attn(norm(x)))
        x_res = x
        y_s = self._window_attn(self.norm1(x), self.self_attn, kv_in=None)
        x = x_res + self.drop_path1(self.gamma_attn1 * y_s)

        # --- 2. Windowed Cross-Attention ---
        # x = x + drop_path(gamma * attn(norm(x), norm(context)))
        x_res = x
        y_c = self._window_attn(
            self.norm2(x), self.cross_attn, kv_in=self.norm_context(context)
        )
        x = x_res + self.drop_path2(self.gamma_attn2 * y_c)

        # --- 3. MLP ---
        # x = x + drop_path(gamma * mlp(norm(x)))
        x_res = x
        y_m = self.mlp(self.norm3(x))
        x = x_res + self.drop_path3(self.gamma_mlp * y_m)

        return x


class WindowAttentionRPB(nn.Module):
    """
    Self/Cross window attention with learnable Relative Position Bias (RPB)
    and optional per-window key mask for padded tokens.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        window_size: int,
        bias: bool,
        batch_first: bool,
    ):
        super().__init__()
        assert batch_first
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        self.scale = self.head_dim**-0.5

        # qkv projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # --- Relative Position Bias table ---
        ws = window_size
        self.bias_table = nn.Parameter(
            torch.zeros((2 * ws - 1) * (2 * ws - 1), num_heads)
        )
        coords = torch.stack(
            torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij")
        )  # [2, ws, ws]
        coords_flat = coords.flatten(1)  # [2, ws*ws]
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # [2, L, L]
        rel = rel.permute(1, 2, 0) + ws - 1  # [L, L, 2]
        rel_idx = rel[..., 0] * (2 * ws - 1) + rel[..., 1]  # [L, L]
        self.register_buffer("rel_index", rel_idx, persistent=False)
        nn.init.trunc_normal_(self.bias_table, std=0.02)

    def forward(
        self,
        query,
        key=None,
        value=None,
        key_padding_mask=None,
        need_weights=False,
        attn_mask=None,
    ):
        if key is None:
            key = query
        if value is None:
            value = key
        B, L, C = query.shape
        H = self.num_heads
        D = self.head_dim

        q = self.q_proj(query).reshape(B, L, H, D).transpose(1, 2)  # [B, H, L, D]
        k = self.k_proj(key).reshape(B, L, H, D).transpose(1, 2)
        v = self.v_proj(value).reshape(B, L, H, D).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, L, L]

        ws = self.window_size
        Lr = ws * ws
        if L == Lr:  # Only add bias if this is windowed attention
            bias = (
                self.bias_table[self.rel_index.reshape(-1)]
                .reshape(Lr, Lr, H)
                .permute(2, 0, 1)
            )
            attn = attn + bias.unsqueeze(0)

        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        if attn_mask is not None:
            attn = attn + attn_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        if need_weights:
            attn_weights = attn.mean(dim=1)
            return out, attn_weights
        else:
            return out, None


class RefineBlock(nn.Module):
    """
    Local residual block with mixed dilations.
    Keeps low-k content, boosts mid/high-k.
    """

    def __init__(self, C: int, dilation: int = 1):
        super().__init__()
        padding1 = 1
        padding2 = dilation

        self.conv1 = nn.Conv2d(C, C, 3, padding=padding1, dilation=1)
        self.conv2 = nn.Conv2d(C, C, 3, padding=padding2, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, C)
        self.norm2 = nn.GroupNorm(8, C)
        self.act = nn.GELU()

        # LayerScale for the block
        self.gamma = nn.Parameter(torch.zeros(1, C, 1, 1))

    def forward(self, x):
        y = self.norm1(x)
        y = self.act(y)
        y = self.conv1(y)

        y = self.norm2(y)
        y = self.act(y)
        y = self.conv2(y)

        return x + self.gamma * y


class MultiScaleRefinementHead(nn.Module):
    """
    Strong refinement head operating at full resolution.

    - in_channels/out_channels: prognostic channels (1 in your current setup)
    - base_channels: width of the refinement path
    - blocks use dilations [1, 2, 3, 1] by default
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        dilations=(1, 2, 3, 1),
    ):
        super().__init__()
        C = base_channels

        self.conv_in = nn.Conv2d(in_channels, C, 3, padding=1)

        self.blocks = nn.ModuleList([RefineBlock(C, d) for d in dilations])

        self.conv_out = nn.Conv2d(C, out_channels, 3, padding=1)

        # Global scale so head starts near-zero (i.e. network near identity)
        self.global_gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """
        x: [B, C, H, W]
        returns: refinement delta, same shape
        """
        y = self.conv_in(x)
        for blk in self.blocks:
            y = blk(y)
        y = self.conv_out(y)
        return self.global_gamma * y
