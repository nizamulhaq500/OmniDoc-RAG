"""
Unit tests for SymmetricPatchInfoNCELoss.
Tests MaxSim computation, symmetric cross-entropy, query masking, and gradient flow on CPU.
"""

import torch
import pytest
from losses.contrastive_loss import SymmetricPatchInfoNCELoss


def test_contrastive_loss_forward():
    """Verify loss computation and metrics output on CPU."""
    batch_size = 4
    seq_len = 10
    k_latents = 16
    dim = 32

    criterion = SymmetricPatchInfoNCELoss(init_temperature=0.07)

    queries = torch.randn(batch_size, seq_len, dim)
    documents = torch.randn(batch_size, k_latents, dim)

    loss, metrics = criterion(queries, documents)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() > 0
    assert "loss_q2d" in metrics
    assert "loss_d2q" in metrics
    assert "temperature" in metrics


def test_contrastive_loss_gradient_flow():
    """Verify that gradients properly flow back to queries, documents, and temperature."""
    batch_size = 2
    seq_len = 5
    k_latents = 8
    dim = 16

    criterion = SymmetricPatchInfoNCELoss(init_temperature=0.07, learnable_temperature=True)

    queries = torch.randn(batch_size, seq_len, dim, requires_grad=True)
    documents = torch.randn(batch_size, k_latents, dim, requires_grad=True)

    loss, _ = criterion(queries, documents)
    loss.backward()

    assert queries.grad is not None
    assert torch.norm(queries.grad) > 0

    assert documents.grad is not None
    assert torch.norm(documents.grad) > 0

    assert criterion.log_tau.grad is not None


def test_contrastive_loss_query_masking():
    """Verify that zeroed mask tokens do not affect the MaxSim score."""
    batch_size = 2
    seq_len = 6
    k_latents = 4
    dim = 16

    criterion = SymmetricPatchInfoNCELoss()

    queries = torch.randn(batch_size, seq_len, dim)
    documents = torch.randn(batch_size, k_latents, dim)

    # Mask with only first 3 tokens active
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.float32)

    # Compute masked score
    scores1 = criterion.compute_maxsim_scores(queries, documents, query_mask=mask)

    # Modify the masked out query positions arbitrarily
    queries_perturbed = queries.clone()
    queries_perturbed[:, 3:, :] = torch.randn_like(queries[:, 3:, :]) * 10.0

    scores2 = criterion.compute_maxsim_scores(queries_perturbed, documents, query_mask=mask)

    # Scores must be mathematically identical since tokens 3..5 are masked
    assert torch.allclose(scores1, scores2, atol=1e-5)
