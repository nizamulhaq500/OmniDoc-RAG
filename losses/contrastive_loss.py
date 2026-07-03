"""
Symmetric Multi-Scale Patch-InfoNCE Loss with MaxSim Late Interaction.

Computes fine-grained contrastive alignment between variable-length text queries
and dense visual document latents produced by the Perceiver Resampler.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class SymmetricPatchInfoNCELoss(nn.Module):
    """
    Symmetric Patch-InfoNCE Contrastive Loss using Late-Interaction MaxSim operator.
    
    Formula:
      Score(Q_i, D_j) = sum_{l=1}^L max_{k=1}^K (q_{i,l} · d_{j,k})
      L_{Q->D} = CrossEntropy(Score / tau, targets)
      L_{D->Q} = CrossEntropy((Score / tau)^T, targets)
      L_total  = 0.5 * (L_{Q->D} + L_{D->Q})
    """

    def __init__(
        self,
        init_temperature: float = 0.07,
        learnable_temperature: bool = True,
        min_temperature: float = 0.01,
        max_temperature: float = 1.0
    ):
        super().__init__()
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature

        # Learnable log-temperature parameter (log(tau))
        init_log_tau = math.log(init_temperature)
        if learnable_temperature:
            self.log_tau = nn.Parameter(torch.tensor(init_log_tau, dtype=torch.float32))
        else:
            self.register_buffer("log_tau", torch.tensor(init_log_tau, dtype=torch.float32))

    @property
    def temperature(self) -> torch.Tensor:
        """Returns the current temperature clamped to stable bounds."""
        return torch.clamp(torch.exp(self.log_tau), min=self.min_temperature, max=self.max_temperature)

    def compute_maxsim_scores(
        self,
        queries: torch.Tensor,
        documents: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes the pairwise MaxSim late-interaction score matrix.
        
        Args:
            queries: Tensor of shape (B_q, L, D) - Query token embeddings
            documents: Tensor of shape (B_d, K, D) - Visual document latents
            query_mask: Optional binary mask (B_q, L) - 1 for valid tokens, 0 for pad
            
        Returns:
            Score matrix of shape (B_q, B_d)
        """
        # L2-normalize vectors along feature dimension D
        q_norm = F.normalize(queries, p=2, dim=-1)      # (B_q, L, D)
        d_norm = F.normalize(documents, p=2, dim=-1)    # (B_d, K, D)

        # Pairwise inner products: (B_q, B_d, L, K)
        # sim_matrix[i, j, l, k] = <q_{i, l}, d_{j, k}>
        sim_matrix = torch.einsum("b l d, c k d -> b c l k", q_norm, d_norm)

        # MaxSim operator: For each query token l, take max over document latents K
        max_sim = sim_matrix.max(dim=-1).values  # (B_q, B_d, L)

        # Mask out padded query tokens
        if query_mask is not None:
            # Expand mask: (B_q, 1, L)
            mask = query_mask.unsqueeze(1).to(dtype=max_sim.dtype)
            max_sim = max_sim * mask

        # Sum over query tokens L: (B_q, B_d)
        scores = max_sim.sum(dim=-1)
        return scores

    def forward(
        self,
        queries: torch.Tensor,
        documents: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes symmetric Patch-InfoNCE loss over batch.
        
        Args:
            queries: (B, L, D)
            documents: (B, K, D)
            query_mask: Optional (B, L)
            
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        batch_size = queries.shape[0]
        if batch_size != documents.shape[0]:
            raise ValueError(
                f"Batch size mismatch: queries ({batch_size}) vs documents ({documents.shape[0]})"
            )

        # Compute raw MaxSim scores
        raw_scores = self.compute_maxsim_scores(queries, documents, query_mask=query_mask)

        # Scale by temperature
        tau = self.temperature
        scaled_scores = raw_scores / tau

        # Ground truth diagonal targets: pair (i, i) is positive
        targets = torch.arange(batch_size, device=queries.device, dtype=torch.long)

        # Query-to-Document directional loss
        loss_q2d = F.cross_entropy(scaled_scores, targets)

        # Document-to-Query directional loss
        loss_d2q = F.cross_entropy(scaled_scores.transpose(0, 1), targets)

        # Symmetric total loss
        total_loss = 0.5 * (loss_q2d + loss_d2q)

        # Diagnostic metrics
        metrics = {
            "loss": total_loss.detach(),
            "loss_q2d": loss_q2d.detach(),
            "loss_d2q": loss_d2q.detach(),
            "temperature": tau.detach(),
            "mean_pos_score": torch.diag(raw_scores).mean().detach(),
            "mean_neg_score": (raw_scores.sum() - torch.diag(raw_scores).sum()).detach() / (batch_size * (batch_size - 1) + 1e-8)
        }

        return total_loss, metrics
