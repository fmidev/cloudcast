import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from math import sqrt, ceil
from timm.models.layers import DropPath


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
    """
    Drop-in replacement for your PatchEmbed.
    - Prognostic data is patchified with Unfold (exact, invertible).
    - Forcings are embedded separately and compressed to fill (embed_dim - d_data).
    - A small fusion Linear keeps the output shape [B,T,P,embed_dim] identical.
    Exposes: patch_size, h_patches, w_patches, num_patches
    """

    def __init__(
        self,
        input_resolution,  # (H, W)
        patch_size,  # int or (ph,pw)
        data_channels,  # C_data (e.g., 1)
        forcing_channels,  # C_forc (e.g., 32)
        embed_dim,  # D (e.g., 128)
        data_dim_override=None,  # optionally force d_data; default = C_data*ps*ps (clipped to embed_dim-1)
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

        # --- Lossless (exact) patchify for data ---
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
    """
    Drop-in head that reconstructs the image from the *data part* of tokens.
    Uses a Linear "unproj" to C*ps*ps per patch + Fold (exact unpatch).
    Ignoring the forcing sub-token here is intentional: forcings help forecast
    via the backbone, not via the pixel reconstruction path.
    """

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


import torch
import torch.nn as nn


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


class EncoderBlock(nn.Module):
    def __init__(
        self, dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path_rate
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=qkv_bias,
            batch_first=True,
        )
        self.drop_path1 = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, hidden_dim=int(dim * mlp_ratio), dropout=drop)
        self.drop_path2 = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path1(attn)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class DecoderBlock(nn.Module):
    """Transformer decoder block with self-attention and cross-attention"""

    def __init__(
        self, dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path_rate
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=qkv_bias,
            batch_first=True,
        )
        self.drop_path1 = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

        self.norm2 = nn.LayerNorm(dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=qkv_bias,
            batch_first=True,
        )
        self.drop_path2 = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

        self.norm3 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, hidden_dim=int(dim * mlp_ratio), dropout=drop)
        self.drop_path3 = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x, context):
        # Self-attention (without mask for now to avoid shape issues)
        x_norm1 = self.norm1(x)
        self_attn, _ = self.self_attn(x_norm1, x_norm1, x_norm1)

        x = x + self.drop_path1(self_attn)

        # Cross-attention to encoder outputs
        x_norm2 = self.norm2(x)

        cross_attn, _ = self.cross_attn(x_norm2, context, context)

        x = x + self.drop_path2(cross_attn)

        # Feedforward
        x = x + self.drop_path3(self.mlp(self.norm3(x)))

        return x


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


class SkipConnection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Main feature transform
        self.transform = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )

    def forward(self, x_encoder, x_decoder):
        # x_encoder: [B, L, C]
        # x_decoder: [B, L, C]
        # noise_embedding: [B, noise_dim]

        # Transform encoder features
        x_skip = self.transform(x_encoder)  # [B, L, C]

        # Combine
        x_combined = x_skip

        return x_decoder + x_combined


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


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape

        assert L == H * W, f"input feature has wrong size: {L} != {H * W}"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class DualStem(nn.Module):
    """
    Apply ConvStem separately to prognostic and forcing inputs.
    Outputs stemmed feature maps for both, still at full resolution.
    """

    def __init__(self, prog_ch, forcing_ch, stem_ch):
        super().__init__()
        self.stem_prognostic = nn.Sequential(
            nn.Conv2d(prog_ch, stem_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(stem_ch, stem_ch, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.stem_forcing = nn.Sequential(
            nn.Conv2d(forcing_ch, stem_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(stem_ch, stem_ch, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x, forcing):
        """
        x: [B, T, C_prog, H, W]
        forcing: [B, T_f, C_force, H, W]
        returns:
            x_stem: [B, T, stem_ch, H, W]
            f_stem: [B, T_f, stem_ch, H, W]
        """
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x_stem = self.stem_prognostic(x).reshape(B, T, -1, H, W)

        Bf, Tf, Cf, H, W = forcing.shape
        f = forcing.reshape(Bf * Tf, Cf, H, W)
        f_stem = self.stem_forcing(f).reshape(Bf, Tf, -1, H, W)

        return x_stem, f_stem


class PatchEmbed(nn.Module):
    def __init__(
        self,
        input_resolution,
        patch_size,
        data_channels,
        forcing_channels,
        embed_dim,
    ):
        super().__init__()
        assert type(input_resolution) in (list, tuple)
        self.input_resolution = input_resolution
        self.patch_size = (
            patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        )
        self.num_patches = (self.input_resolution[0] // self.patch_size[0]) * (
            self.input_resolution[1] // self.patch_size[1]
        )

        # Separate projections for data and forcings
        self.data_proj = nn.Conv2d(
            data_channels,
            embed_dim // 2,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.forcing_proj = nn.Conv2d(
            forcing_channels,
            embed_dim // 2,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # Optional: fusion layer to combine the embeddings
        self.fusion = nn.Linear(embed_dim, embed_dim)

        self.h_patches = input_resolution[0] // self.patch_size[0]
        self.w_patches = input_resolution[1] // self.patch_size[1]
        self.num_patches = self.h_patches * self.w_patches

        self.patch_area_scale = sqrt(self.patch_size[0] * self.patch_size[1])

    def forward(self, data, forcing):
        assert data.ndim == 5, "x dims should be B, T, C, H, W, found: {}".format(
            data.shape
        )
        assert (
            forcing.ndim == 5
        ), "forcing dims should be B, T, C, H, W, found: {}".format(forcing.shape)
        B, T, C_data, H, W = data.shape
        _, T_forcing, C_forcing, _, _ = forcing.shape

        # Process each time step
        embeddings = []
        for t in range(T):
            # Get embeddings for data and forcings
            data_emb = (
                self.data_proj(data[:, t]).flatten(2).transpose(1, 2)
            )  # [B, patches, embed_dim//2]
            forcing_emb = (
                self.forcing_proj(forcing[:, t]).flatten(2).transpose(1, 2)
            )  # [B, patches, embed_dim//2]

            # With patch size k each output token is an average over k pixels
            # ie. the activation magnitude is reduced to 1/k. To counter this
            # we multiply the activation with the patch size
            data_emb = data_emb * self.patch_area_scale
            forcing_emb = forcing_emb * self.patch_area_scale

            # Concatenate along embedding dimension
            combined_emb = torch.cat(
                [data_emb, forcing_emb], dim=2
            )  # [B, patches, embed_dim]

            # apply fusion layer
            combined_emb = self.fusion(combined_emb)

            embeddings.append(combined_emb)

        # [B, T, P, D]
        x_tokens = torch.stack(embeddings, dim=1)

        # keep the future forcing (if provided)
        f_future = None
        if T_forcing > T:
            f_future = (
                self.forcing_proj(forcing[:, T])  # use timestep T as the "future"
                .flatten(2)
                .transpose(1, 2)  # [B, P, D/2]
            )
            # pad to full D by concatenating zeros for the prognostic half
            zeros = torch.zeros_like(f_future)
            f_future = torch.cat([zeros, f_future], dim=2)  # [B, P, D]
            f_future = self.fusion(f_future)  # [B, P, D]
            f_future = f_future.unsqueeze(1)  # [B, 1, P, D]

        return x_tokens, f_future


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
