import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from swinu.layers import (
    PatchMerge,
    PatchExpand,
    SwinDecoderBlock,
    SwinEncoderBlock,
    ProjectToImageFold,
    PatchEmbedLossless,
    DWConvResidual3D,
    MultiScaleRefinementHead,
    get_padded_size,
    pad_tensors,
    pad_tensor,
    depad_tensor,
)
from types import SimpleNamespace
from torch.utils.checkpoint import checkpoint


class cc2model(nn.Module):
    def __init__(self, config):
        super().__init__()
        config = SimpleNamespace(**config)

        self.model_class = "swinu"
        self.patch_size = config.patch_size
        self.embed_dim = config.hidden_dim

        self.real_input_resolution = config.input_resolution

        input_resolution = get_padded_size(
            config.input_resolution[0], config.input_resolution[1], self.patch_size, 1
        )

        self.num_prognostic = len(config.prognostic_params)
        self.num_forcings = len(config.forcing_params) + len(
            config.static_forcing_params
        )

        self.patch_embed = PatchEmbedLossless(
            input_resolution=input_resolution,
            patch_size=self.patch_size,
            data_channels=self.num_prognostic,
            forcing_channels=self.num_forcings,
            embed_dim=self.embed_dim,
            data_dim_override=self.embed_dim // 2,
        )
        self.h_patches = self.patch_embed.h_patches
        self.w_patches = self.patch_embed.w_patches
        self.num_patches = self.patch_embed.num_patches

        self.post_pe_norm = nn.LayerNorm(self.embed_dim)
        self.post_pe_gain = nn.Parameter(torch.ones(1))

        # Spatial position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))

        # Input layer norm and dropout
        self.norm_input = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(config.drop_rate)

        if not isinstance(config.drop_path_rate, list):
            config.drop_path_rate = [config.drop_path_rate] * 4

        def _get_dilation(block, num):
            if block in ("enc2", "dec1"):
                return 1
            return 1 if num % 2 == 0 else 2

        # Transformer encoder blocks
        self.encoder1 = nn.ModuleList(
            [
                SwinEncoderBlock(
                    dim=self.embed_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=True,
                    drop=config.drop_rate,
                    attn_drop=config.attn_drop_rate,
                    drop_path_rate=config.drop_path_rate[0],
                    window_size=config.window_size,
                    shift_size=0 if i % 2 == 0 else config.window_size // 2,
                    H=self.h_patches,
                    W=self.w_patches,
                    T=config.history_length,
                )
                for i in range(config.encoder1_depth)
            ]
        )

        h0, w0 = self.h_patches, self.w_patches

        self.dwres_e1 = nn.ModuleList(
            [
                DWConvResidual3D(
                    self.embed_dim,
                    (h0, w0),
                    time_dim=config.history_length,
                    dilation=_get_dilation("enc1", i),
                )
                for i in range(config.encoder1_depth)
            ]
        )

        self.input_resolution_halved = (
            input_resolution[0] // self.patch_size,
            input_resolution[1] // self.patch_size,
        )
        self.downsample = PatchMerge(
            self.input_resolution_halved, self.embed_dim, time_dim=config.history_length
        )

        self.post_merge_norm = nn.LayerNorm(self.embed_dim * 2)

        self.encoder2 = nn.ModuleList(
            [
                SwinEncoderBlock(
                    dim=self.embed_dim * 2,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=True,
                    drop=config.drop_rate,
                    attn_drop=config.attn_drop_rate,
                    drop_path_rate=config.drop_path_rate[1],
                    window_size=config.window_size_deep,
                    shift_size=0 if i % 2 == 0 else config.window_size_deep // 2,
                    H=self.h_patches // 2,
                    W=self.w_patches // 2,
                    T=config.history_length,
                )
                for i in range(config.encoder2_depth)
            ]
        )

        h1, w1 = h0 // 2, w0 // 2

        ls_init = 1e-3
        expand = 1

        self.dwres_e2 = nn.ModuleList(
            [
                DWConvResidual3D(
                    self.embed_dim * 2,
                    (h1, w1),
                    time_dim=config.history_length,
                    expand=expand,
                    dilation=_get_dilation("enc2", i),
                    ls_init=ls_init,
                )
                for i in range(config.encoder2_depth)
            ]
        )

        self.pre_dec1_norm = nn.LayerNorm(self.embed_dim * 2)

        self.decoder1 = nn.ModuleList(
            [
                SwinDecoderBlock(
                    dim=self.embed_dim * 2,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=True,
                    drop=config.drop_rate,
                    attn_drop=config.attn_drop_rate,
                    drop_path_rate=config.drop_path_rate[2],
                    window_size=config.window_size_deep,
                    shift_size=0 if i % 2 == 0 else config.window_size_deep // 2,
                    H=self.h_patches // 2,
                    W=self.w_patches // 2,
                    T=config.history_length,
                )
                for i in range(config.decoder1_depth)
            ]
        )

        self.dwres_d1 = nn.ModuleList(
            [
                DWConvResidual3D(
                    self.embed_dim * 2,
                    (h1, w1),
                    time_dim=config.history_length,
                    expand=expand,
                    dilation=_get_dilation("dec1", i),
                    ls_init=ls_init,
                )
                for i in range(config.decoder1_depth)
            ]
        )

        self.upsample = PatchExpand(
            self.embed_dim * 2, self.embed_dim * 2, scale_factor=2
        )

        self.pre_dec2_norm = nn.LayerNorm(self.embed_dim * 2)

        self.decoder2 = nn.ModuleList(
            [
                SwinDecoderBlock(
                    dim=self.embed_dim * 2,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=True,
                    drop=config.drop_rate,
                    attn_drop=config.attn_drop_rate,
                    drop_path_rate=config.drop_path_rate[3],
                    window_size=config.window_size,
                    shift_size=0 if i % 2 == 0 else config.window_size // 2,
                    H=self.h_patches,
                    W=self.w_patches,
                    T=1,
                )
                for i in range(config.decoder2_depth)
            ]
        )

        expand = 2.0
        ls_init = 1e-2

        if config.use_deep_refinement_head:
            expand = 4.0
            ls_init = 5e-2

        self.dwres_d2 = nn.ModuleList(
            [
                DWConvResidual3D(
                    self.embed_dim * 2,
                    (h0, w0),
                    time_dim=1,
                    dilation=_get_dilation("dec2", i),
                    expand=expand,
                    ls_init=ls_init,
                )
                for i in range(config.decoder2_depth)
            ]
        )

        # Final norm and output projection
        self.norm_final = nn.LayerNorm(self.embed_dim * 2)

        self.project_to_image = ProjectToImageFold(
            input_resolution=input_resolution,
            patch_size=self.patch_size,
            out_channels=self.num_prognostic,  # 1
            d_data=self.patch_embed.d_data,
            embed_dim=self.embed_dim * 2,
            undo_scale=False,
        )
        # Patch expansion for upsampling to original resolution

        self.patch_expand = PatchExpand(
            self.embed_dim * 2, self.embed_dim // 4, scale_factor=1
        )

        self.final_expand = nn.Linear(
            self.embed_dim // 4,
            self.patch_size**2 * len(config.prognostic_params),
        )

        # Step identification embeddings
        self.step_id_embeddings = nn.Parameter(torch.randn(2, self.embed_dim * 2))
        nn.init.normal_(self.step_id_embeddings, std=0.02)

        # Initialize weights
        # self.apply(self._init_weights)
        # nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.skip_proj = nn.Linear(self.embed_dim, self.embed_dim * 2)

        self.use_hard_skip = config.use_hard_skip

        if self.use_hard_skip:
            self.skip_fuse = nn.Linear(self.embed_dim * 4, self.embed_dim * 2)

        self.use_gradient_checkpointing = config.use_gradient_checkpointing
        self.use_scheduled_sampling = config.use_scheduled_sampling

        if config.use_deep_refinement_head:
            self.refinement_head = MultiScaleRefinementHead(
                in_channels=1,
                out_channels=1,
                base_channels=64,
                dilations=(1, 2, 3, 1, 4),
            )
        else:
            self.refinement_head = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 1, 3, padding=1),
            )

        self.autoregressive_mode = config.autoregressive_mode

        if self.autoregressive_mode is False:
            self.max_step = 12
            self.step_embedding_direct = nn.Embedding(
                self.max_step + 1, self.embed_dim * 2
            )

        # Layers to process future forcings
        # project a [B, P, D] future-forcing token into the decoder space
        self.future_force_proj = nn.Sequential(
            nn.LayerNorm(self.embed_dim * 2),
            nn.Linear(self.embed_dim * 2, self.embed_dim * 2),  # lift to decoder dim
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim * 2),  # refine
        )

        # Simple downsampler to match spatial dimesions
        self.future_force_down = nn.Conv2d(
            in_channels=self.embed_dim,
            out_channels=self.embed_dim * 2,
            kernel_size=2,
            stride=2,
        )

        self.use_residual_adapter_head = config.use_residual_adapter_head

        if self.use_residual_adapter_head:
            self.residual_alpha = nn.Parameter(torch.tensor(0.0))

            self.residual_adapter_head = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(8, 1, kernel_size=3, padding=1),
            )

            nn.init.zeros_(self.residual_adapter_head[-1].weight)
            nn.init.zeros_(self.residual_adapter_head[-1].bias)

    def patch_embedding(self, x, forcing):
        x_tokens, f_future = self.patch_embed(x, forcing)  # [B, T, patches, embed_dim]

        B, T, P, D = x_tokens.shape

        # Add positional embedding to each patch
        for t in range(T):
            x_tokens[:, t] = x_tokens[:, t] + self.pos_embed

        if hasattr(self, "post_pe_norm"):
            x_tokens = self.post_pe_norm(x_tokens)
            x_tokens = x_tokens * self.post_pe_gain

        return x_tokens, f_future

    def encode(self, x):
        """Encode input sequence into latent representation"""
        B, T, P, D = x.shape

        # Apply input normalization and dropout
        x = x.reshape(B, T * P, D)
        x = self.norm_input(x)
        x = self.dropout(x)

        # Pass through encoder blocks
        if self.use_gradient_checkpointing:
            for block, dwres in zip(self.encoder1, self.dwres_e1):
                x = checkpoint(block, x, use_reentrant=False)
                x = dwres(x)
        else:
            for block, dwres in zip(self.encoder1, self.dwres_e1):
                x = block(x)
                x = dwres(x)

        skip = x.clone()  # skip connection, B, T*P, D
        skip = skip.reshape(B, T, -1, D)

        # Downsample
        x = self.downsample(x)
        x = self.post_merge_norm(x)

        # Pass through encoder blocks
        if self.use_gradient_checkpointing:
            for block, dwres in zip(self.encoder2, self.dwres_e2):
                x = checkpoint(block, x, use_reentrant=False)
                x = dwres(x)
        else:
            for block, dwres in zip(self.encoder2, self.dwres_e2):
                x = block(x)
                x = dwres(x)

        # Reshape back to separate time and space dimensions
        x = x.reshape(B, T, -1, D * 2)

        return x, skip

    def decode(self, encoded, step, skip, f_future=None):
        B, T, P, D = encoded.shape
        outputs = []

        # Initial input is the encoded sequence
        decoder_input = encoded

        # Keep track of the latest state
        latest_state = encoded[:, -1:, :, :]  # Just the last time step

        encoded_flat = encoded.reshape(B, -1, D)
        decoder_in = decoder_input.reshape(B, -1, D)

        # Determine step id (0 for first step using ground truth, 1 for subsequent steps)
        if self.autoregressive_mode:
            step_id = 0
            if not self.use_scheduled_sampling and step > 0:
                step_id = 1
            # Get appropriate step embedding
            step_embedding = self.step_id_embeddings[step_id]  # [D]
        else:
            step_id = step + 1
            if step_id > self.max_step:
                step_id = self.max_step
            step_tensor = torch.tensor([step_id], device=encoded.device)
            step_embedding = self.step_embedding_direct(step_tensor).squeeze(0)

        # Add step embedding to decoder input
        # For first token in each sequence (acts as a "step type" token)
        # We'll add it to the last P tokens which represent our current state
        decoder_in_with_id = decoder_in.clone()
        decoder_in_with_id[:, -P:] = decoder_in[:, -P:] + step_embedding.unsqueeze(
            0
        ).unsqueeze(1)

        if f_future is not None:
            # f_future: [B, 1, P, D_enc]  (from patch embed)
            # Map it to decoder dim and add to the last P tokens.
            f_add = f_future.squeeze(1)  # [B, P, D_enc]

            # reshape back to image
            B, P_enc, D_enc = f_add.shape
            H, W = self.h_patches, self.w_patches
            f_img = f_add.view(B, H, W, D_enc).permute(0, 3, 1, 2)  # [B, D_enc, H, W]

            # downsample spatially
            f_img_ds = self.future_force_down(f_img)  # [B, D_dec, H/2, W/2]

            # flatten back to tokens
            H2, W2 = f_img_ds.shape[2], f_img_ds.shape[3]
            f_add = f_img_ds.flatten(2).transpose(1, 2)  # [B, P_dec, D_dec]

            # refine
            f_add = self.future_force_proj(f_add)

            # Add into the current state tokens
            decoder_in_with_id[:, -P:, :] = decoder_in_with_id[:, -P:, :] + f_add

        # Process through decoder blocks
        x = decoder_in_with_id
        x = self.pre_dec1_norm(x)

        # Process through decoder blocks
        if self.use_gradient_checkpointing:
            for block, dwres in zip(self.decoder1, self.dwres_d1):
                x = checkpoint(block, x, encoded_flat, use_reentrant=False)
                x = dwres(x)
        else:
            for block, dwres in zip(self.decoder1, self.dwres_d1):
                x = block(x, encoded_flat)
                x = dwres(x)

        # Get the delta prediction
        delta_pred1 = x[:, -P:].reshape(B, 1, P, D)

        new_H, new_W = self.input_resolution_halved
        new_H = new_H // 2  # 2 = num_times
        new_W = new_W // 2  # 2 = num_times

        with torch.amp.autocast("cuda", enabled=False):
            upsampled_delta, P_new, D_new = self.upsample(
                delta_pred1.reshape(B, -1, D).float(), H=new_H, W=new_W
            )

        P_new, D_new = upsampled_delta.shape[1], upsampled_delta.shape[2]

        upsampled_delta = upsampled_delta.reshape(B, 1, P_new, D_new)
        P_new, D_new = upsampled_delta.shape[2], upsampled_delta.shape[3]
        x2 = upsampled_delta.reshape(B, -1, D_new)

        x2 = self.pre_dec2_norm(x2)

        skip_token = skip[:, -1, :, :]  # shape: [B, num_tokens, embed_dim]
        assert skip_token.ndim == 3
        skip_proj = self.skip_proj(skip_token)  # [B, num_tokens, embed_dim*2]

        if self.use_hard_skip:
            # Hard skip fusion path
            assert (
                x2.shape[1] == skip_proj.shape[1]
            ), f"Shape mismatch: x2 {x2.shape} vs skip_proj {skip_proj.shape}"

            # Concatenate along channel dim: [B, P, 4D]
            x2 = torch.cat([x2, skip_proj], dim=-1)

            # Fuse back to decoder2 dim: [B, P, 2D]
            x2 = self.skip_fuse(x2)

        if self.use_gradient_checkpointing:
            for block, dwres in zip(self.decoder2, self.dwres_d2):
                x2 = checkpoint(block, x2, skip_proj, use_reentrant=False)
                x2 = dwres(x2)
        else:
            for block, dwres in zip(self.decoder2, self.dwres_d2):
                x2 = block(x2, skip_proj)
                x2 = dwres(x2)

        delta_pred2 = x2.reshape(B, 1, P_new, D_new)

        return delta_pred2

    def forward(self, data, forcing, step):
        assert (
            data.ndim == 5
        ), "Input data tensor shape should be [B, T, C, H, W], is: {}".format(
            data.shape
        )

        data, padding_info = pad_tensor(data, self.patch_size, 1)
        forcing, _ = pad_tensor(forcing, self.patch_size, 1)

        x, f_future = self.patch_embedding(data, forcing)
        x, skip = self.encode(x)
        x = self.decode(x, step, skip, f_future)

        output = self.project_to_image(x)
        output_ref = self.refinement_head(output.squeeze(2))
        output = output + output_ref.unsqueeze(2)

        if self.use_residual_adapter_head:
            B, T, C, H, W = output.shape
            output = output.view(B * T, C, H, W)
            output = output + self.residual_alpha * self.residual_adapter_head(output)
            output = output.view(B, T, C, H, W)

        output = depad_tensor(output, padding_info)

        assert list(output.shape[-2:]) == list(
            self.real_input_resolution
        ), f"Output shape {output.shape[-2:]} does not match real input resolution {self.real_input_resolution}"
        return output
