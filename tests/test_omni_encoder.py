"""
Unit tests for OmniDocDualEncoder architecture.
Tests end-to-end forward pass, image/text encoding, retrieval scoring, and gradient flow on CPU.
"""

import torch
import pytest
from models.omni_encoder import OmniDocDualEncoder


def test_omni_encoder_forward_pass():
    """Verify complete end-to-end forward pass and loss computation."""
    batch_size = 2
    img_h, img_w = 64, 64
    patch_size = 16  # -> 4x4 = 16 patches
    num_latents = 8  # Compress 16 -> 8 latents
    embed_dim = 32
    seq_len = 6
    vocab_size = 1000

    model = OmniDocDualEncoder(
        embed_dim=embed_dim,
        patch_size=patch_size,
        num_latents=num_latents,
        perceiver_depth=1,
        heads=2,
        head_dim=16,
        vocab_size=vocab_size,
        use_rope2d=True
    )

    images = torch.randn(batch_size, 3, img_h, img_w)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len)

    loss, metrics = model(images, input_ids, attention_mask=mask)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() > 0
    assert "loss_q2d" in metrics
    assert "loss_d2q" in metrics


def test_omni_encoder_separate_encoding():
    """Verify separate document and query encoding for offline indexing."""
    model = OmniDocDualEncoder(
        embed_dim=32,
        patch_size=16,
        num_latents=8,
        perceiver_depth=1,
        heads=2,
        head_dim=16,
        vocab_size=1000,
        use_rope2d=True
    )

    images = torch.randn(2, 3, 64, 64)
    input_ids = torch.randint(0, 1000, (2, 5))

    doc_latents = model.encode_document(images)
    query_embeds = model.encode_query(input_ids)

    assert doc_latents.shape == (2, 8, 32)
    assert query_embeds.shape == (2, 5, 32)

    # Compute MaxSim retrieval matrix
    scores = model.compute_retrieval_scores(query_embeds, doc_latents)
    assert scores.shape == (2, 2)


def test_omni_encoder_end_to_end_gradients():
    """Verify that full backprop updates vision, perceiver, and query weights."""
    model = OmniDocDualEncoder(
        embed_dim=32,
        patch_size=16,
        num_latents=4,
        perceiver_depth=1,
        heads=2,
        head_dim=16,
        vocab_size=500
    )

    images = torch.randn(2, 3, 32, 32)
    input_ids = torch.randint(0, 500, (2, 4))

    loss, _ = model(images, input_ids)
    loss.backward()

    # Vision patch projection gradients
    assert model.patch_extractor.proj.weight.grad is not None
    assert torch.norm(model.patch_extractor.proj.weight.grad) > 0

    # Perceiver latent parameter gradients
    assert model.perceiver.latents.grad is not None
    assert torch.norm(model.perceiver.latents.grad) > 0

    # Query embedding gradients
    assert model.query_encoder.token_embeddings.weight.grad is not None
    assert torch.norm(model.query_encoder.token_embeddings.weight.grad) > 0
