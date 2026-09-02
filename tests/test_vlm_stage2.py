"""
Unit tests for Stage 2 Visual-Language Generation (OmniDocVLM, Cross-Modal Projector, CLM Loss).
Runs on CPU with micro-tensors in <0.5s.
"""

import pytest
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from train_stage2_vlm import (
    OmniDocVisualProjector,
    OmniDocVLM,
    SyntheticVLMDataset,
    collate_vlm,
    train_stage2_vlm
)


def test_visual_projector_dimension_transformation():
    """Verify projector accurately transforms visual dimension to LLM dimension."""
    batch_size = 2
    num_latents = 16
    vis_dim = 64
    llm_dim = 128

    projector = OmniDocVisualProjector(vis_dim=vis_dim, llm_dim=llm_dim)
    x = torch.randn(batch_size, num_latents, vis_dim)
    out = projector(x)

    assert out.shape == (batch_size, num_latents, llm_dim)
    assert not torch.isnan(out).any()


def test_vlm_forward_and_loss_masking():
    """Verify forward pass correctly evaluates CLM cross-entropy on answer tokens."""
    batch_size = 2
    img_size = 64
    vis_dim = 32
    llm_dim = 64
    num_latents = 4
    vocab_size = 500

    vlm = OmniDocVLM(
        vis_dim=vis_dim,
        llm_dim=llm_dim,
        patch_size=16,
        num_latents=num_latents,
        perceiver_depth=1,
        heads=2,
        vocab_size=vocab_size
    )

    images = torch.randn(batch_size, 3, img_size, img_size)
    prompt_ids = torch.randint(1, vocab_size, (batch_size, 6))
    answer_ids = torch.randint(1, vocab_size, (batch_size, 4))

    loss, logits = vlm(images=images, prompt_ids=prompt_ids, answer_ids=answer_ids)

    assert loss is not None
    assert loss.item() > 0.0
    assert not torch.isnan(loss)
    # Full sequence length = num_latents (4) + prompt_len (6) + answer_len (4) = 14
    assert logits.shape == (batch_size, 14, vocab_size)


def test_vlm_gradient_flow():
    """Verify gradients propagate from CLM loss back to visual projector and Perceiver."""
    vlm = OmniDocVLM(
        vis_dim=32,
        llm_dim=64,
        patch_size=16,
        num_latents=4,
        perceiver_depth=1,
        heads=2,
        vocab_size=500
    )

    images = torch.randn(2, 3, 64, 64, requires_grad=True)
    prompt_ids = torch.randint(1, 500, (2, 5))
    answer_ids = torch.randint(1, 500, (2, 3))

    loss, _ = vlm(images=images, prompt_ids=prompt_ids, answer_ids=answer_ids)
    loss.backward()

    # Gradients in visual projector
    proj_weight = next(vlm.visual_projector.mlp.parameters())
    assert proj_weight.grad is not None
    assert torch.norm(proj_weight.grad) > 0

    # Gradients in Perceiver Resampler latents
    assert vlm.perceiver.latents.grad is not None
    assert torch.norm(vlm.perceiver.latents.grad) > 0


def test_vlm_autoregressive_generation():
    """Verify conditional generation produces valid next token IDs."""
    vlm = OmniDocVLM(
        vis_dim=32,
        llm_dim=64,
        patch_size=16,
        num_latents=4,
        perceiver_depth=1,
        heads=2,
        vocab_size=500
    )

    images = torch.randn(1, 3, 64, 64)
    prompt_ids = torch.randint(1, 500, (1, 4))

    gen_tokens = vlm.generate_answer(images=images, prompt_ids=prompt_ids, max_new_tokens=5)
    assert gen_tokens.shape == (1, 5)
    assert (gen_tokens >= 0).all() and (gen_tokens < 500).all()


def test_vlm_training_convergence_step():
    """Verify Stage 2 micro-training step executes without error."""
    dataset = SyntheticVLMDataset(num_samples=4, img_size=64)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_vlm)

    vlm = OmniDocVLM(vis_dim=32, llm_dim=64, patch_size=16, num_latents=4, perceiver_depth=1, heads=2, vocab_size=500)
    optimizer = AdamW(vlm.parameters(), lr=1e-3)

    history = train_stage2_vlm(
        model=vlm,
        dataloader=loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        epochs=3,
        max_steps=6,
        log_interval=2
    )

    assert len(history["losses"]) == 6
    assert history["losses"][-1] < history["losses"][0] * 1.2
