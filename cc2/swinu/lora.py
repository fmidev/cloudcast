import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    nn.Linear + low-rank trainable update.
    Base weight/bias are frozen. Only A and B are trainable.
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 4,
        alpha: float | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(base)}")
        if r <= 0:
            raise ValueError("r must be > 0")

        self.base = base
        self.r = r
        self.alpha = float(alpha if alpha is not None else r)
        self.scaling = self.alpha / self.r
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze base params
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        # Low-rank factors
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)

        # Init: start as no-op (B=0)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.B(self.A(self.drop(x)))


def apply_lora_to_window_attention(attn_module, r=4, alpha=4, dropout=0.0, which="qv"):
    """
    attn_module: WindowAttentionRPB
    which: "qv" (conservative), "qkv", or "all" (qkv+proj)
    """
    if which in ("qv", "qkv", "all"):
        attn_module.q_proj = LoRALinear(
            attn_module.q_proj, r=r, alpha=alpha, dropout=dropout
        )
        attn_module.v_proj = LoRALinear(
            attn_module.v_proj, r=r, alpha=alpha, dropout=dropout
        )
    if which in ("qkv", "all"):
        attn_module.k_proj = LoRALinear(
            attn_module.k_proj, r=r, alpha=alpha, dropout=dropout
        )
    if which in ("all",):
        attn_module.proj = LoRALinear(
            attn_module.proj, r=r, alpha=alpha, dropout=dropout
        )


def apply_lora_to_feedforward(ff_module, r=4, alpha=4, dropout=0.0, which="down"):
    """
    ff_module: FeedForward where ff_module.net = [Linear, GELU, Dropout, Linear, Dropout]
    which: "down" (conservative), "both"
    """
    if which in ("both",):
        ff_module.net[0] = LoRALinear(
            ff_module.net[0], r=r, alpha=alpha, dropout=dropout
        )
    if which in ("down", "both"):
        ff_module.net[3] = LoRALinear(
            ff_module.net[3], r=r, alpha=alpha, dropout=dropout
        )


def freeze_all_except_lora(model: nn.Module):
    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze only LoRA A/B
    for m in model.modules():
        if isinstance(m, LoRALinear):
            for p in m.A.parameters():
                p.requires_grad = True
            for p in m.B.parameters():
                p.requires_grad = True


def inject_lora_decoder2(
    model: nn.Module,
    r: int = 4,
    alpha: int | None = None,
    lora_dropout: float = 0.0,
    attn_where: str = "cross",  # "self" | "cross" | "both"
    attn_which: str = "qv",  # "qv" | "qkv" | "all"
    mlp_which: str = "down",  # "none" | "down" | "both"
    disable_stochastic: bool = True,
):
    """
    Inject LoRA into decoder2 blocks of a swinu cc2model.
    Expects model.decoder2 to be a ModuleList of SwinDecoderBlock.
    """

    if alpha is None:
        alpha = r

    if not hasattr(model, "decoder2"):
        raise AttributeError("Model has no attribute 'decoder2'")

    for blk in model.decoder2:
        if disable_stochastic:
            # DropPath in SwinDecoderBlock
            blk.drop_path1 = nn.Identity()
            blk.drop_path2 = nn.Identity()
            blk.drop_path3 = nn.Identity()

            # Attention dropouts inside WindowAttentionRPB
            blk.self_attn.attn_drop = nn.Identity()
            blk.self_attn.proj_drop = nn.Identity()
            blk.cross_attn.attn_drop = nn.Identity()
            blk.cross_attn.proj_drop = nn.Identity()

            # MLP dropouts (FeedForward.net = [Linear, GELU, Dropout, Linear, Dropout])
            blk.mlp.net[2] = nn.Identity()
            blk.mlp.net[4] = nn.Identity()

        if attn_where in ("self", "both"):
            apply_lora_to_window_attention(
                blk.self_attn, r=r, alpha=alpha, dropout=lora_dropout, which=attn_which
            )
        if attn_where in ("cross", "both"):
            apply_lora_to_window_attention(
                blk.cross_attn, r=r, alpha=alpha, dropout=lora_dropout, which=attn_which
            )

        if mlp_which != "none":
            apply_lora_to_feedforward(
                blk.mlp, r=r, alpha=alpha, dropout=lora_dropout, which=mlp_which
            )
