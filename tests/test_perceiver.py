"""
Unit tests for PerceiverResampler.
Tests compression factor, gradient flow, and 2D-RoPE integration on CPU.
"""

import torch
import pytest
from models.perceiver import PerceiverResampler


def test_perceiver_compression_shape():
    """Verify that input sequence of N tokens is compressed to exactly K latents."""
    batch_size = 2
    n_tokens = 64  # e.g., 8x8 patch grid
    k_latents = 16
    dim = 64
    heads = 4
    head_dim = 16

    resampler = PerceiverResampler(
        dim=dim,
        depth=2,
        num_latents=k_latents,
        heads=heads,
        head_dim=head_dim,
        use_rope2d=True
    )

    x = torch.randn(batch_size, n_tokens, dim)
    out = resampler(x, grid_hw=(8, 8))

    assert out.shape == (batch_size, k_latents, dim)
    assert not torch.isnan(out).any()


def test_perceiver_gradient_backprop():
    """Verify clean gradient backpropagation through learned latents and attention layers."""
    batch_size = 1
    n_tokens = 16
    k_latents = 4
    dim = 32
    heads = 2
    head_dim = 16

    resampler = PerceiverResampler(
        dim=dim,
        depth=1,
        num_latents=k_latents,
        heads=heads,
        head_dim=head_dim,
        use_rope2d=False
    )

    x = torch.randn(batch_size, n_tokens, dim, requires_grad=True)
    out = resampler(x)
    loss = (out ** 2).sum()
    loss.backward()

    # Verify gradients reach input x
    assert x.grad is not None
    assert torch.norm(x.grad) > 0

    # Verify gradients update learnable latent queries
    assert resampler.latents.grad is not None
    assert torch.norm(resampler.latents.grad) > 0


def test_perceiver_variable_resolution():
    """Verify that resampler can process variable patch lengths dynamically."""
    resampler = PerceiverResampler(
        dim=32,
        depth=1,
        num_latents=8,
        heads=2,
        head_dim=16,
        use_rope2d=True
    )

    # Resolution 1: 4x4 = 16 tokens
    x1 = torch.randn(1, 16, 32)
    out1 = resampler(x1, grid_hw=(4, 4))
    assert out1.shape == (1, 8, 32)

    # Resolution 2: 6x6 = 36 tokens
    x2 = torch.randn(1, 36, 32)
    out2 = resampler(x2, grid_hw=(6, 6))
    assert out2.shape == (1, 8, 32)
