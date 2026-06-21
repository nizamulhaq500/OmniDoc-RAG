"""
2D Spatial Rotary Position Embedding (2D-RoPE) for Document Vision Transformers.

Decomposes the feature dimension into vertical (y) and horizontal (x) rotational frequency
manifolds to preserve 2D relative Euclidean distances across document tokens.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dimensions of the input tensor:
    [-x2, x1] where x = [x1, x2].
    """
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb_2d(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> torch.Tensor:
    """
    Applies 2D Rotary Position Embedding to input tensor x.
    
    Args:
        x: Tensor of shape (Batch, Heads, Num_Patches, Head_Dim)
        cos: Cosine frequencies of shape (1, 1, Num_Patches, Head_Dim)
        sin: Sine frequencies of shape (1, 1, Num_Patches, Head_Dim)
        
    Returns:
        Rotated tensor with same shape as x.
    """
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding2D(nn.Module):
    """
    2D Spatial Rotary Position Embedding Layer.
    
    Splits the head dimension D_h into:
      - D_h / 2 dimensions for vertical (y-axis) frequencies
      - D_h / 2 dimensions for horizontal (x-axis) frequencies
      
    Guarantees that attention dot products satisfy:
      <R_2D(y1, x1) q, R_2D(y2, x2) k> = g(q, k, y1 - y2, x1 - x2)
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        max_height: int = 64,
        max_width: int = 64
    ):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f"RotaryEmbedding2D dimension must be divisible by 4, got {dim}")
        
        self.dim = dim
        self.base = base
        self.max_height = max_height
        self.max_width = max_width
        
        # Dimension per spatial axis (half for Y, half for X)
        self.dim_axis = dim // 2  # e.g., 64 -> 32
        
        # Inverse frequencies for half-dimension: theta_k = base^(-4k / dim)
        # Length = dim_axis // 2 = dim // 4
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim_axis, 2).float() / self.dim_axis))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cached cos/sin tensors
        self._cached_h = 0
        self._cached_w = 0
        self._cached_cos = None
        self._cached_sin = None

    def _compute_grid_frequencies(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates 2D frequency meshgrid for grid of size (height, width).
        """
        # Ensure inv_freq is on the target device
        inv_freq = self.inv_freq.to(device=device, dtype=dtype)
        
        # 1D coordinates
        y_pos = torch.arange(height, device=device, dtype=dtype)  # (H,)
        x_pos = torch.arange(width, device=device, dtype=dtype)   # (W,)
        
        # Outer product to get phase angles
        freqs_y = torch.outer(y_pos, inv_freq)  # (H, dim_axis // 2)
        freqs_x = torch.outer(x_pos, inv_freq)  # (W, dim_axis // 2)
        
        # Expand across 2D grid
        freqs_y = freqs_y.view(height, 1, -1).expand(height, width, -1)  # (H, W, dim_axis // 2)
        freqs_x = freqs_x.view(1, width, -1).expand(height, width, -1)   # (H, W, dim_axis // 2)
        
        # Concatenate Y and X frequency manifolds
        freqs_2d = torch.cat([freqs_y, freqs_x], dim=-1)  # (H, W, dim // 2)
        
        # Flatten spatial grid into sequence: (H*W, dim // 2)
        freqs_2d = freqs_2d.view(height * width, -1)
        
        # Duplicate frequencies to match full dimension D_h for rotate_half pairing
        emb = torch.cat([freqs_2d, freqs_2d], dim=-1)  # (H*W, dim)
        
        # Reshape for broadcasting with (Batch, Heads, Num_Patches, Head_Dim)
        cos = emb.cos().view(1, 1, height * width, self.dim)
        sin = emb.sin().view(1, 1, height * width, self.dim)
        
        return cos, sin

    def get_cos_sin(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cached or newly computed (cos, sin) frequency tensors.
        """
        if (
            self._cached_cos is None
            or self._cached_h != height
            or self._cached_w != width
            or self._cached_cos.device != device
            or self._cached_cos.dtype != dtype
        ):
            cos, sin = self._compute_grid_frequencies(height, width, device, dtype)
            self._cached_h = height
            self._cached_w = width
            self._cached_cos = cos
            self._cached_sin = sin
            
        return self._cached_cos, self._cached_sin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        height: int,
        width: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies 2D-RoPE to query and key tensors.
        
        Args:
            q: Query tensor of shape (Batch, Heads, H*W, Head_Dim)
            k: Key tensor of shape (Batch, Heads, H*W, Head_Dim)
            height: Spatial grid height (number of vertical patches)
            width: Spatial grid width (number of horizontal patches)
            
        Returns:
            Tuple of rotated (q_rot, k_rot) tensors.
        """
        num_patches = q.shape[2]
        if num_patches != height * width:
            raise ValueError(
                f"Patch sequence length ({num_patches}) does not match height*width ({height}*{width} = {height*width})"
            )
            
        cos, sin = self.get_cos_sin(height, width, device=q.device, dtype=q.dtype)
        
        q_rot = apply_rotary_emb_2d(q, cos, sin)
        k_rot = apply_rotary_emb_2d(k, cos, sin)
        
        return q_rot, k_rot
