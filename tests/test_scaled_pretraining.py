"""
Unit Tests for ScaledOmniDocDualEncoder.
Verifies micro-forward, gradient backprop, and late-interaction MaxSim.
"""

import pytest
import torch
from models.scaled_omni_encoder import ScaledOmniDocDualEncoder


def test_scaled_encoder_forward_pass():
    """Verify micro forward pass of Scaled Dual-Encoder."""
    model = ScaledOmniDocDualEncoder(
        text_backbone="bert-base-uncased",
        embed_dim=128,
        patch_size=32,
        num_latents=16,
        perceiver_depth=1,
        heads=4,
        head_dim=32
    )
    model.eval()

    # Micro-batch: B=2, C=3, H=128, W=128
    images = torch.randn(2, 3, 128, 128)
    input_ids = torch.tensor([[101, 2054, 2024, 102], [101, 2008, 4839, 102]]) # (2, 4)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]])

    with torch.no_grad():
        doc_latents = model.encode_document(images)
        query_embeds = model.encode_query(input_ids, attention_mask=mask)
        loss, metrics = model(images, input_ids, attention_mask=mask)

    assert doc_latents.shape == (2, 16, 128), f"Expected (2, 16, 128), got {doc_latents.shape}"
    assert query_embeds.shape == (2, 4, 128), f"Expected (2, 4, 128), got {query_embeds.shape}"
    assert not torch.isnan(loss), "Loss contains NaN"
    assert loss.item() > 0.0, "Loss must be positive"


def test_scaled_encoder_gradient_backprop():
    """Verify unbroken gradient backpropagation through 2D-RoPE and Perceiver."""
    model = ScaledOmniDocDualEncoder(
        text_backbone="bert-base-uncased",
        embed_dim=64,
        patch_size=32,
        num_latents=8,
        perceiver_depth=1,
        heads=2,
        head_dim=32
    )
    model.train()

    images = torch.randn(2, 3, 64, 64, requires_grad=True)
    input_ids = torch.tensor([[101, 2054, 102], [101, 2008, 102]])

    loss, _ = model(images, input_ids)
    loss.backward()

    # Verify gradients exist on patch projection and latents
    assert model.patch_proj.weight.grad is not None, "Patch projection missing gradient"
    assert model.perceiver.latents.grad is not None, "Perceiver latents missing gradient"
    assert images.grad is not None, "Input images missing gradient"
