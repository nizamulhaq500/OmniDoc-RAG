"""
OmniDocDualEncoder: End-to-End Vision Document & Query Dual-Encoder Architecture.

Binds the Vision Patch Extractor, 2D-RoPE, Perceiver Resampler, and Text Query Projection
into a unified differentiable retrieval model.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from einops import rearrange

from .rope2d import RotaryEmbedding2D
from .perceiver import PerceiverResampler
from losses.contrastive_loss import SymmetricPatchInfoNCELoss


class VisionPatchExtractor(nn.Module):
    """
    Extracts spatial 2D patch tokens from high-resolution document images.
    Converts (B, C, H, W) -> (B, (H/P)*(W/P), D).
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 768,
        patch_size: int = 32
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Args:
            x: Document images of shape (B, C, H, W)
            
        Returns:
            Tuple of (patch_tokens: (B, N, D), grid_hw: (H_patches, W_patches))
        """
        b, c, h, w = x.shape
        # Conv2D: (B, D, H//P, W//P)
        feat = self.proj(x)
        h_grid, w_grid = feat.shape[2], feat.shape[3]
        
        # Flatten spatial grid: (B, D, H_g, W_g) -> (B, H_g * W_g, D)
        patches = rearrange(feat, "b d h w -> b (h w) d")
        patches = self.norm(patches)
        return patches, (h_grid, w_grid)


class QueryEmbedding(nn.Module):
    """
    Lightweight Text Query Token Encoder.
    Maps token IDs to dense token vectors in the shared latent space D.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        embed_dim: int = 768,
        max_seq_len: int = 128
    ):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(max_seq_len, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape (B, L)
            attention_mask: Optional tensor of shape (B, L)
            
        Returns:
            Query token embeddings of shape (B, L, D)
        """
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        tokens = self.token_embeddings(input_ids)
        pos = self.position_embeddings(positions)
        
        x = self.norm(tokens + pos)
        return x


class OmniDocDualEncoder(nn.Module):
    """
    Unified OmniDoc-RAG Dual-Encoder Architecture.
    
    Encodes:
      1. Document Images -> 2D-RoPE + Perceiver Resampler -> K=64 Document Latents
      2. Text Queries -> Query Embedding -> L Query Token Vectors
      3. Computes Symmetric Multi-Scale Patch-InfoNCE Contrastive Loss
    """

    def __init__(
        self,
        embed_dim: int = 768,
        patch_size: int = 32,
        num_latents: int = 64,
        perceiver_depth: int = 2,
        heads: int = 8,
        head_dim: int = 64,
        vocab_size: int = 30522,
        init_temperature: float = 0.07,
        use_rope2d: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_latents = num_latents

        # Document Vision Pathway
        self.patch_extractor = VisionPatchExtractor(
            in_channels=3,
            embed_dim=embed_dim,
            patch_size=patch_size
        )
        self.perceiver = PerceiverResampler(
            dim=embed_dim,
            depth=perceiver_depth,
            num_latents=num_latents,
            heads=heads,
            head_dim=head_dim,
            use_rope2d=use_rope2d
        )

        # Text Query Pathway
        self.query_encoder = QueryEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim
        )

        # Loss Criterion
        self.loss_fn = SymmetricPatchInfoNCELoss(
            init_temperature=init_temperature,
            learnable_temperature=True
        )

    def encode_document(
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        Encodes document image batch into compressed visual latents.
        
        Args:
            images: Tensor of shape (B, 3, H, W)
            
        Returns:
            Document latents of shape (B, num_latents, D)
        """
        raw_patches, grid_hw = self.patch_extractor(images)
        doc_latents = self.perceiver(raw_patches, grid_hw=grid_hw)
        return doc_latents

    def encode_query(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encodes text query tokens into query embeddings.
        
        Args:
            input_ids: Tensor of shape (B, L)
            attention_mask: Optional tensor of shape (B, L)
            
        Returns:
            Query token embeddings of shape (B, L, D)
        """
        return self.query_encoder(input_ids, attention_mask=attention_mask)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        End-to-end forward pass computing contrastive loss.
        """
        doc_latents = self.encode_document(images)
        query_embeds = self.encode_query(input_ids, attention_mask=attention_mask)

        loss, metrics = self.loss_fn(
            queries=query_embeds,
            documents=doc_latents,
            query_mask=attention_mask
        )
        return loss, metrics

    def compute_retrieval_scores(
        self,
        queries: torch.Tensor,
        documents: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes MaxSim late-interaction retrieval scores between query and document representations.
        """
        return self.loss_fn.compute_maxsim_scores(queries, documents, query_mask=query_mask)
