"""
Unit tests for 2D Spatial Rotary Position Embeddings (2D-RoPE).
Tests shape preservation and mathematical 2D relative translation invariance on CPU.
"""

import torch
import pytest
from models.rope2d import RotaryEmbedding2D, apply_rotary_emb_2d


def test_rope2d_output_shapes():
    """Verify that 2D-RoPE preserves exact tensor shapes."""
    batch_size = 1
    num_heads = 2
    height, width = 4, 4
    num_patches = height * width  # 16
    head_dim = 32  # Divisible by 4

    rope = RotaryEmbedding2D(dim=head_dim)

    q = torch.randn(batch_size, num_heads, num_patches, head_dim)
    k = torch.randn(batch_size, num_heads, num_patches, head_dim)

    q_rot, k_rot = rope(q, k, height=height, width=width)

    assert q_rot.shape == (batch_size, num_heads, num_patches, head_dim)
    assert k_rot.shape == (batch_size, num_heads, num_patches, head_dim)
    assert q_rot.dtype == q.dtype


def test_rope2d_translation_invariance():
    """
    Mathematical Proof Verification:
    Relative attention scores <R(y1, x1) q, R(y2, x2) k> must equal
    <R(y1 + dy, x1 + dx) q, R(y2 + dy, x2 + dx) k> for any 2D shift (dy, dx).
    """
    head_dim = 32
    rope = RotaryEmbedding2D(dim=head_dim)

    # Consider an 8x8 document grid
    h, w = 8, 8
    cos, sin = rope.get_cos_sin(height=h, width=w, device=torch.device("cpu"))

    # Vector q and vector k
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    # Position 1: (y=1, x=2) -> index = 1*8 + 2 = 10
    # Position 2: (y=3, x=5) -> index = 3*8 + 5 = 29
    idx_q1 = 1 * w + 2
    idx_k1 = 3 * w + 5

    cos_q1 = cos[:, :, idx_q1:idx_q1+1, :]
    sin_q1 = sin[:, :, idx_q1:idx_q1+1, :]
    cos_k1 = cos[:, :, idx_k1:idx_k1+1, :]
    sin_k1 = sin[:, :, idx_k1:idx_k1+1, :]

    q_rot1 = apply_rotary_emb_2d(q, cos_q1, sin_q1)
    k_rot1 = apply_rotary_emb_2d(k, cos_k1, sin_k1)
    dot_product_original = torch.sum(q_rot1 * k_rot1).item()

    # Shift both tokens by 2D vector (dy=+2, dx=+1)
    # Position 1 shifted: (y=1+2=3, x=2+1=3) -> index = 3*8 + 3 = 27
    # Position 2 shifted: (y=3+2=5, x=5+1=6) -> index = 5*8 + 6 = 46
    idx_q2 = (1 + 2) * w + (2 + 1)
    idx_k2 = (3 + 2) * w + (5 + 1)

    cos_q2 = cos[:, :, idx_q2:idx_q2+1, :]
    sin_q2 = sin[:, :, idx_q2:idx_q2+1, :]
    cos_k2 = cos[:, :, idx_k2:idx_k2+1, :]
    sin_k2 = sin[:, :, idx_k2:idx_k2+1, :]

    q_rot2 = apply_rotary_emb_2d(q, cos_q2, sin_q2)
    k_rot2 = apply_rotary_emb_2d(k, cos_k2, sin_k2)
    dot_product_shifted = torch.sum(q_rot2 * k_rot2).item()

    # Check numerical equivalence under translation
    diff = abs(dot_product_original - dot_product_shifted)
    assert diff < 1e-5, f"Translation invariance violated! Diff: {diff}"


def test_rope2d_axis_orthogonality():
    """
    Verify that vertical displacement and horizontal displacement produce
    distinct rotational effects on asymmetric vectors.
    """
    head_dim = 32
    rope = RotaryEmbedding2D(dim=head_dim)
    h, w = 6, 6
    cos, sin = rope.get_cos_sin(height=h, width=w, device=torch.device("cpu"))

    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    # Base at (0, 0)
    idx_origin = 0
    q_rot0 = apply_rotary_emb_2d(q, cos[:, :, idx_origin:1, :], sin[:, :, idx_origin:1, :])

    # Vertical neighbor at (1, 0) -> idx = 1*6 + 0 = 6
    idx_vert = 1 * w + 0
    k_rot_vert = apply_rotary_emb_2d(k, cos[:, :, idx_vert:idx_vert+1, :], sin[:, :, idx_vert:idx_vert+1, :])
    dot_vert = torch.sum(q_rot0 * k_rot_vert).item()

    # Horizontal neighbor at (0, 1) -> idx = 0*6 + 1 = 1
    idx_horiz = 0 * w + 1
    k_rot_horiz = apply_rotary_emb_2d(k, cos[:, :, idx_horiz:idx_horiz+1, :], sin[:, :, idx_horiz:idx_horiz+1, :])
    dot_horiz = torch.sum(q_rot0 * k_rot_horiz).item()

    # Dot products should be distinguishable (axes are independent)
    assert abs(dot_vert - dot_horiz) > 1e-4
