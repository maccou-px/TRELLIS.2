from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from ...modules import sparse as sp
from ...modules.sparse.transformer import SparseTransformerBlock, ModulatedSparseTransformerBlock
from ...modules.utils import convert_module_to, manual_cast, str_to_dtype


class Slat2Scalar(nn.Module):
    """
    Args:
        out_channels (int): Number of scalar targets.
        latent_channels (int): Input latent feature dim.
        model_channels (int): Hidden dim for transformer blocks.
        num_blocks (int): Number of transformer blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): FFN hidden dim multiplier.
        attn_mode (str): Attention mode ('full', 'swin').
        use_rope (bool): Use rotary position embeddings.
        qk_rms_norm (bool): RMS norm on Q/K.
        use_checkpoint (bool): Gradient checkpointing.
        pool_channels (int): MLP head hidden dim.
        mlp_layers (int): Number of MLP head layers.
        dropout (float): Dropout in MLP head.
    """
    def __init__(
        self,
        out_channels: int,
        cond_dim: int,
        latent_channels: int,
        model_channels: int,
        num_blocks: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: str = 'full',
        use_rope: bool = True,
        qk_rms_norm: bool = False,
        use_checkpoint: bool = False,
        dtype: str = 'bfloat16',
        dropout: float = 0.0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.dtype = str_to_dtype(dtype)
        self.dropout = nn.Dropout(dropout)

        # Input projection
        self.input_layer = sp.SparseLinear(latent_channels, model_channels)
        self.cond_proj = nn.Linear(cond_dim, model_channels) if cond_dim > 0 else None

        # Transformer blocks
        _block = ModulatedSparseTransformerBlock if cond_dim > 0 else SparseTransformerBlock
        self.blocks = nn.ModuleList([
            _block(
                channels=model_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_mode=attn_mode,
                use_checkpoint=use_checkpoint,
                use_rope=use_rope,
                qk_rms_norm=qk_rms_norm,
            )
            for _ in range(num_blocks)
        ])

        # mean + max pool → 2x model_channels → FFN → scalars
        self.head = nn.Linear(model_channels * 2, out_channels)

        self.convert_to(self.dtype)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def convert_to(self, dtype: torch.dtype) -> None:
        """
        Convert the torso of the model to the specified dtype.
        """
        self.dtype = dtype
        self.blocks.apply(partial(convert_module_to, dtype=dtype))

    def sparse_global_pool(self, x: sp.SparseTensor) -> torch.Tensor:
        """Pool batched SparseTensor to [N, 2*C] via mean + max."""
        batch_size = x.shape[0]
        feats = x.feats
        batch_idx = x.coords[:, 0]

        pooled = []
        for b in range(batch_size):
            mask = batch_idx == b
            if mask.any():
                f = feats[mask]
                pooled.append(torch.cat([f.mean(dim=0), f.max(dim=0).values], dim=-1))
            else:
                pooled.append(torch.zeros(feats.shape[-1] * 2, device=feats.device))
        return torch.stack(pooled, dim=0)

    def forward(self, x: sp.SparseTensor, **kwargs) -> torch.Tensor:
        h = self.input_layer(x)
        h = manual_cast(h, self.dtype)

        mod = None
        if self.cond_proj is not None:
            mod = manual_cast(self.cond_proj(kwargs["mod"]), self.dtype)

        for block in self.blocks:
            h = block(h, mod=mod)

        h = manual_cast(h, x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = h.replace(self.dropout(h.feats))
        g = self.sparse_global_pool(h)
        return self.head(g)