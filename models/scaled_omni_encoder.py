"""
Scaled OmniDocDualEncoder: Foundation-Powered Vision-Language Document Dual-Encoder.

Integrates pretrained Vision Transformers (SigLIP / ViT) and Pretrained Text Tokenizers (BERT / SigLIP)
with our custom 2D-RoPE spatial phase injection and Perceiver Resampler bottleneck.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union
from einops import rearrange
from transformers import AutoModel, AutoTokenizer

from .rope2d import RotaryEmbedding2D
from .perceiver import PerceiverResampler
from losses.contrastive_loss import SymmetricPatchInfoNCELoss


class ScaledOmniDocDualEncoder(nn.Module):
    """
    Scaled Dual-Encoder leveraging transfer learning from pretrained foundation models
    combined with custom 2D-RoPE and Perceiver Resampler (16x compression).
    """

    def __init__(
        self,
        vision_backbone: str = "bert-base-uncased",
        text_backbone: str = "bert-base-uncased",
        embed_dim: int = 768,
        patch_size: int = 32,
        num_latents: int = 64,
        perceiver_depth: int = 2,
        heads: int = 8,
        head_dim: int = 64,
        init_temperature: float = 0.07,
        use_rope2d: bool = True,
        freeze_backbones: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_latents = num_latents
        
        # 1. Vision Patch Projection
        self.patch_proj = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.patch_norm = nn.LayerNorm(embed_dim)
        
        # 2. Custom 2D-RoPE & Perceiver Resampler
        self.perceiver = PerceiverResampler(
            dim=embed_dim,
            depth=perceiver_depth,
            num_latents=num_latents,
            heads=heads,
            head_dim=head_dim,
            use_rope2d=use_rope2d
        )
        
        # 3. Pretrained Text Encoder Pathway
        self.text_encoder = AutoModel.from_pretrained(text_backbone)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)
        self.text_norm = nn.LayerNorm(embed_dim)
        
        if freeze_backbones:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # 4. Symmetric Contrastive Criterion
        self.loss_fn = SymmetricPatchInfoNCELoss(
            init_temperature=init_temperature,
            learnable_temperature=True
        )

    def encode_document(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encodes document image batch: (B, 3, H, W) -> (B, num_latents, D)
        """
        # Conv2D: (B, D, H//P, W//P)
        feat = self.patch_proj(images)
        h_grid, w_grid = feat.shape[2], feat.shape[3]
        
        # Flatten spatial grid: (B, N, D)
        patches = rearrange(feat, "b d h w -> b (h w) d")
        patches = self.patch_norm(patches)
        
        # Perceiver + 2D-RoPE
        doc_latents = self.perceiver(patches, grid_hw=(h_grid, w_grid))
        return doc_latents

    def encode_query(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encodes text query token IDs using pretrained transformer: (B, L) -> (B, L, D)
        """
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state # (B, L, D_text)
        query_embeds = self.text_norm(self.text_proj(hidden)) # (B, L, D)
        return query_embeds

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        End-to-end forward pass computing symmetric contrastive loss.
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
        """Computes MaxSim late-interaction retrieval scores."""
        return self.loss_fn.compute_maxsim_scores(queries, documents, query_mask=query_mask)
