"""
OmniDoc-RAG Stage 2: Visual-Language Generative Alignment Engine.
Connects Stage 1 trained 2D-RoPE + Perceiver Resampler visual latents (K=64)
to a compact Causal Language Model (e.g. Qwen2.5) via a Cross-Modal Projector and LoRA adapters.
"""

import os
import math
import time
import argparse
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange

from models.rope2d import RotaryEmbedding2D
from models.perceiver import PerceiverResampler
from models.omni_encoder import VisionPatchExtractor


class OmniDocVisualProjector(nn.Module):
    """
    Cross-Modal Projector:
    Transforms spatial document latents (B, K=64, D_vis) into the continuous token
    embedding space of the Language Model (B, K=64, D_llm) using a 2-layer MLP with GELU.
    """
    def __init__(self, vis_dim: int = 512, llm_dim: int = 896, dropout: float = 0.05):
        super().__init__()
        self.norm = nn.LayerNorm(vis_dim)
        self.mlp = nn.Sequential(
            nn.Linear(vis_dim, llm_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim, llm_dim, bias=True),
            nn.Dropout(dropout)
        )

    def forward(self, visual_latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_latents: (B, K, D_vis)
        Returns:
            visual_prefix: (B, K, D_llm)
        """
        return self.mlp(self.norm(visual_latents))


class OmniDocVLM(nn.Module):
    """
    End-to-End OCR-Free Visual-Language Document Generation Model.
    Combines:
      1. Vision Backbone: Patch Extractor + 2D-RoPE + Perceiver Resampler (Stage 1)
      2. Cross-Modal Linear/MLP Projector
      3. Language Decoder (e.g. Qwen2.5 / Mock Causal LM for micro-testing)
    """
    def __init__(
        self,
        vis_dim: int = 512,
        llm_dim: int = 896,
        patch_size: int = 32,
        num_latents: int = 64,
        perceiver_depth: int = 2,
        heads: int = 8,
        vocab_size: int = 30522,
        lm_backbone: Optional[nn.Module] = None
    ):
        super().__init__()
        self.vis_dim = vis_dim
        self.llm_dim = llm_dim
        self.num_latents = num_latents
        head_dim = vis_dim // heads

        # 1. Vision Document Encoder (Stage 1)
        self.patch_extractor = VisionPatchExtractor(in_channels=3, embed_dim=vis_dim, patch_size=patch_size)
        self.perceiver = PerceiverResampler(
            dim=vis_dim,
            depth=perceiver_depth,
            num_latents=num_latents,
            heads=heads,
            head_dim=head_dim,
            use_rope2d=True
        )

        # 2. Cross-Modal Projector
        self.visual_projector = OmniDocVisualProjector(vis_dim=vis_dim, llm_dim=llm_dim)

        # 3. Language Decoder Backbone
        if lm_backbone is not None:
            self.lm = lm_backbone
            self.embed_tokens = self.lm.get_input_embeddings()
        else:
            # Standalone lightweight causal decoder for CPU tests & fast validation
            self.embed_tokens = nn.Embedding(vocab_size, llm_dim)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=llm_dim,
                nhead=2,
                dim_feedforward=llm_dim * 2,
                batch_first=True
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
            self.lm_head = nn.Linear(llm_dim, vocab_size, bias=False)
            self.lm = None

    def encode_document(self, images: torch.Tensor) -> torch.Tensor:
        """
        Passes document images through Stage 1 vision pipeline + visual projector.
        Returns visual prefix embeddings: (B, K=64, D_llm)
        """
        patches, grid_hw = self.patch_extractor(images)
        latents = self.perceiver(patches, grid_hw=grid_hw)  # (B, K, D_vis)
        return self.visual_projector(latents)                # (B, K, D_llm)

    def forward(
        self,
        images: torch.Tensor,
        prompt_ids: torch.Tensor,
        answer_ids: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        answer_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with Teacher-Forcing next-token cross-entropy on answer tokens only.
        
        Args:
            images: (B, 3, H, W)
            prompt_ids: (B, L_prompt)
            answer_ids: (B, L_answer)
        Returns:
            loss: scalar CLM loss
            logits: (B, Total_Seq_Len, Vocab_Size)
        """
        device = images.device
        b = images.shape[0]

        # 1. Obtain visual prefix: (B, K, D_llm)
        visual_prefix = self.encode_document(images)
        k = visual_prefix.shape[1]

        # 2. Obtain prompt text embeddings: (B, L_prompt, D_llm)
        prompt_embeds = self.embed_tokens(prompt_ids)
        l_prompt = prompt_embeds.shape[1]

        if answer_ids is not None:
            # Training Mode: Prepend visual prefix to prompt + answer sequence
            answer_embeds = self.embed_tokens(answer_ids)
            l_answer = answer_embeds.shape[1]

            # Full multimodal embedding sequence: (B, K + L_p + L_a, D_llm)
            inputs_embeds = torch.cat([visual_prefix, prompt_embeds, answer_embeds], dim=1)

            # Construct labels: -100 masking on visual prefix and prompt tokens
            labels = torch.full((b, k + l_prompt + l_answer), -100, dtype=torch.long, device=device)
            # Assign true target tokens for answer positions
            if answer_mask is not None:
                for i in range(b):
                    valid_len = int(answer_mask[i].sum().item())
                    labels[i, k + l_prompt : k + l_prompt + valid_len] = answer_ids[i, :valid_len]
            else:
                labels[:, k + l_prompt :] = answer_ids

            if self.lm is not None and hasattr(self.lm, "forward"):
                outputs = self.lm(inputs_embeds=inputs_embeds, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
            else:
                seq_len = inputs_embeds.shape[1]
                causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
                hidden = self.decoder(inputs_embeds, inputs_embeds, tgt_mask=causal_mask, memory_mask=causal_mask)
                logits = self.lm_head(hidden)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1), ignore_index=-100)

            return loss, logits
        else:
            # Inference Mode: Prompt prefix only
            inputs_embeds = torch.cat([visual_prefix, prompt_embeds], dim=1)
            if self.lm is not None and hasattr(self.lm, "forward"):
                outputs = self.lm(inputs_embeds=inputs_embeds)
                logits = outputs.logits
            else:
                seq_len = inputs_embeds.shape[1]
                causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
                hidden = self.decoder(inputs_embeds, inputs_embeds, tgt_mask=causal_mask, memory_mask=causal_mask)
                logits = self.lm_head(hidden)
            dummy_loss = torch.tensor(0.0, device=device)
            return dummy_loss, logits

    @torch.no_grad()
    def generate_answer(
        self,
        images: torch.Tensor,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 32,
        eos_token_id: int = 102
    ) -> torch.Tensor:
        """
        Autoregressive greedy decoding conditioned on visual prefix.
        """
        self.eval()
        device = images.device
        visual_prefix = self.encode_document(images)  # (B, K, D_llm)
        curr_embeds = torch.cat([visual_prefix, self.embed_tokens(prompt_ids)], dim=1)

        generated_tokens = []
        for _ in range(max_new_tokens):
            if self.lm is not None:
                outputs = self.lm(inputs_embeds=curr_embeds)
                next_token_logits = outputs.logits[:, -1, :]
            else:
                seq_len = curr_embeds.shape[1]
                causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
                hidden = self.decoder(curr_embeds, curr_embeds, tgt_mask=causal_mask, memory_mask=causal_mask)
                next_token_logits = self.lm_head(hidden)[:, -1, :]

            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_tokens.append(next_token)

            if (next_token == eos_token_id).all():
                break

            next_embed = self.embed_tokens(next_token)
            curr_embeds = torch.cat([curr_embeds, next_embed], dim=1)

        return torch.cat(generated_tokens, dim=-1)


class SyntheticVLMDataset(Dataset):
    """Generates synthetic multi-modal document QA triplets for fast CPU sanity testing."""
    def __init__(self, num_samples: int = 8, img_size: int = 64, vocab_size: int = 500):
        self.num_samples = num_samples
        self.img_size = img_size
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        image = torch.randn(3, self.img_size, self.img_size)
        prompt_ids = torch.randint(1, self.vocab_size, (8,))
        answer_ids = torch.randint(1, self.vocab_size, (6,))
        return {
            "image": image,
            "prompt_ids": prompt_ids,
            "answer_ids": answer_ids,
            "prompt_mask": torch.ones(8, dtype=torch.float32),
            "answer_mask": torch.ones(6, dtype=torch.float32)
        }


def collate_vlm(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    prompt_ids = torch.stack([item["prompt_ids"] for item in batch], dim=0)
    answer_ids = torch.stack([item["answer_ids"] for item in batch], dim=0)
    prompt_mask = torch.stack([item["prompt_mask"] for item in batch], dim=0)
    answer_mask = torch.stack([item["answer_mask"] for item in batch], dim=0)
    return {
        "images": images,
        "prompt_ids": prompt_ids,
        "answer_ids": answer_ids,
        "prompt_mask": prompt_mask,
        "answer_mask": answer_mask
    }


def train_stage2_vlm(
    model: OmniDocVLM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    device: torch.device = torch.device("cpu"),
    epochs: int = 1,
    max_steps: Optional[int] = None,
    log_interval: int = 5
) -> Dict[str, List[float]]:
    """
    Stage 2 Training Loop for Visual-Language Alignment.
    """
    model.to(device)
    model.train()
    history = {"losses": []}
    global_step = 0
    start_time = time.time()

    print(f"\n[OmniDoc-RAG] Starting Stage 2 VLM Fine-Tuning on {device}")
    for epoch in range(epochs):
        for batch in dataloader:
            images = batch["images"].to(device)
            prompt_ids = batch["prompt_ids"].to(device)
            answer_ids = batch["answer_ids"].to(device)
            prompt_mask = batch["prompt_mask"].to(device)
            answer_mask = batch["answer_mask"].to(device)

            optimizer.zero_grad()
            loss, _ = model(
                images=images,
                prompt_ids=prompt_ids,
                answer_ids=answer_ids,
                prompt_mask=prompt_mask,
                answer_mask=answer_mask
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            global_step += 1
            loss_val = loss.item()
            history["losses"].append(loss_val)

            if global_step % log_interval == 0 or global_step == 1:
                print(f"Epoch [{epoch+1}/{epochs}] | Step [{global_step}] | CLM Loss: {loss_val:.4f}")

            if max_steps is not None and global_step >= max_steps:
                break
        if max_steps is not None and global_step >= max_steps:
            break

    print(f"[OmniDoc-RAG] Stage 2 Training completed in {time.time() - start_time:.2f}s | Final Loss: {history['losses'][-1]:.4f}\n")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniDoc-RAG Stage 2 VLM Training")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_steps", type=int, default=10)
    args = parser.parse_args()

    dev = torch.device(args.device)
    dataset = SyntheticVLMDataset(num_samples=16, img_size=64)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_vlm)

    vlm = OmniDocVLM(vis_dim=64, llm_dim=128, patch_size=16, num_latents=8, perceiver_depth=1, heads=2)
    opt = AdamW(vlm.parameters(), lr=args.lr)

    train_stage2_vlm(vlm, loader, opt, device=dev, epochs=args.epochs, max_steps=args.max_steps)
