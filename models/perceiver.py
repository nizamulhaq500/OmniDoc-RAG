"""
Perceiver Resampler Module for OmniDoc-RAG.

Compresses high-resolution visual document tokens (e.g., N=1024) into a fixed budget
of dense semantic visual latents (e.g., K=64) using learned latent query cross-attention,
integrated with optional 2D-RoPE spatial encoding.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange, repeat

from .rope2d import RotaryEmbedding2D, apply_rotary_emb_2d


class FeedForward(nn.Module):
    """Two-layer MLP with GELU activation and residual connections."""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceiverAttention(nn.Module):
    """
    Multi-Head Cross-Attention / Self-Attention layer for Perceiver Resampler.
    Supports optional 2D-RoPE applied to the Key representations.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        inner_dim = heads * head_dim

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        q_input: torch.Tensor,
        kv_input: torch.Tensor,
        rope_2d: Optional[RotaryEmbedding2D] = None,
        grid_hw: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Args:
            q_input: Query source (latents) of shape (B, K, D)
            kv_input: Key/Value source (visual patches) of shape (B, N, D)
            rope_2d: Optional RotaryEmbedding2D module
            grid_hw: Optional tuple (height, width) of the visual patch grid
        """
        q_norm = self.norm_q(q_input)
        kv_norm = self.norm_kv(kv_input)

        # Linear projections
        q = self.to_q(q_norm)
        k = self.to_k(kv_norm)
        v = self.to_v(kv_norm)

        # Multi-head reshape: (B, N, H * D_h) -> (B, H, N, D_h)
        q = rearrange(q, "b k (h d) -> b h k d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        # Apply 2D-RoPE to keys if provided
        if rope_2d is not None and grid_hw is not None:
            h_grid, w_grid = grid_hw
            cos, sin = rope_2d.get_cos_sin(h_grid, w_grid, device=k.device, dtype=k.dtype)
            k = apply_rotary_emb_2d(k, cos, sin)

        # Fast and memory-efficient scaled dot product attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.0 if not self.training else 0.1
        )

        # Recombine heads: (B, H, K, D_h) -> (B, K, H * D_h)
        out = rearrange(out, "b h k d -> b k (h d)")
        return self.to_out(out)


class PerceiverBlock(nn.Module):
    """
    A single Perceiver layer composed of:
      1. Cross-Attention: Learned Latents (Q) attend to Visual Tokens (K, V)
      2. Latent Self-Attention: Latents attend to themselves (Q, K, V from latents)
      3. Feed-Forward Network: MLP with residual connection
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        head_dim: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        self.cross_attn = PerceiverAttention(dim, heads=heads, head_dim=head_dim, dropout=dropout)
        self.self_attn = PerceiverAttention(dim, heads=heads, head_dim=head_dim, dropout=dropout)
        self.ffn = FeedForward(dim, mult=ff_mult, dropout=dropout)

    def forward(
        self,
        latents: torch.Tensor,
        x: torch.Tensor,
        rope_2d: Optional[RotaryEmbedding2D] = None,
        grid_hw: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        # Cross-Attention: latents attend over visual tokens x
        latents = latents + self.cross_attn(latents, x, rope_2d=rope_2d, grid_hw=grid_hw)
        # Self-Attention: latents interact among themselves
        latents = latents + self.self_attn(latents, latents)
        # Feed-Forward
        latents = latents + self.ffn(latents)
        return latents


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler for OCR-Free Document Retrieval.
    
    Compresses variable-length visual patches X (B, N, D) into a fixed sequence
    of K dense latent tokens Z_out (B, K, D).
    """

    def __init__(
        self,
        dim: int = 768,
        depth: int = 2,
        num_latents: int = 64,
        heads: int = 8,
        head_dim: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.0,
        use_rope2d: bool = True,
        max_grid_h: int = 64,
        max_grid_w: int = 64
    ):
        super().__init__()
        self.dim = dim
        self.num_latents = num_latents

        # Learnable latent queries Z: (K, D)
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)

        # Optional 2D Spatial Rotary Position Embedding
        self.rope_2d = RotaryEmbedding2D(
            dim=head_dim,
            max_height=max_grid_h,
            max_width=max_grid_w
        ) if use_rope2d else None

        # Perceiver Resampler transformer blocks
        self.layers = nn.ModuleList([
            PerceiverBlock(
                dim=dim,
                heads=heads,
                head_dim=head_dim,
                ff_mult=ff_mult,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        grid_hw: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Visual patch features of shape (Batch, N, Dim)
            grid_hw: Tuple of (height, width) specifying 2D patch layout.
                     If None and sqrt(N) is an integer, inferred automatically.
                     
        Returns:
            Compressed visual latents of shape (Batch, num_latents, Dim)
        """
        b, n, d = x.shape

        # Automatically infer square grid if grid_hw is omitted
        if grid_hw is None and self.rope_2d is not None:
            side = int(math.isqrt(n))
            if side * side == n:
                grid_hw = (side, side)

        # Broadcast learned latents to batch: (K, D) -> (B, K, D)
        latents = repeat(self.latents, "k d -> b k d", b=b)

        # Pass through Perceiver blocks
        for layer in self.layers:
            latents = layer(latents, x, rope_2d=self.rope_2d, grid_hw=grid_hw)

        return self.norm(latents)
