"""
Stage 1: Multi-Scale Contrastive Training Script for OmniDoc-RAG.

Trains the custom 2D-RoPE Projection Layer and Perceiver Resampler end-to-end
using Symmetric Patch-InfoNCE Loss with Late Interaction (MaxSim).
"""

import os
import sys
import math
import time
import argparse
from typing import Dict, Any, Optional
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from models.omni_encoder import OmniDocDualEncoder
from data.docvqa_dataset import DocVQADataset, OmniDocCollate
from data.pdf_processor import PDFProcessor


def build_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01
) -> LambdaLR:
    """
    Creates a learning rate schedule with linear warmup and cosine annealing.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return LambdaLR(optimizer, lr_lambda)


class SyntheticDocumentDataset(Dataset):
    """
    Lightweight Synthetic Dataset generator for local debugging and unit verification.
    """

    def __init__(self, num_samples: int = 16, img_size: int = 128):
        self.num_samples = num_samples
        self.img_size = img_size
        self.samples = [
            {
                "image": Image.new("RGB", (img_size, img_size), color=(i * 15 % 255, (i * 35) % 255, 200)),
                "question": f"What is the financial metric for section {i}?",
                "answers": [f"Value_{i}"],
                "doc_id": f"synthetic_doc_{i}"
            }
            for i in range(num_samples)
        ]
        self.processor = PDFProcessor(target_size=(img_size, img_size))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        padded_img, _ = self.processor.preprocess_image(item["image"], target_size=(self.img_size, self.img_size))
        image_tensor = self.processor.image_to_tensor(padded_img)
        return {
            "image": image_tensor,
            "question": item["question"],
            "answers": item["answers"],
            "doc_id": item["doc_id"]
        }


def train_stage1(
    model: OmniDocDualEncoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
    epochs: int = 1,
    max_steps: Optional[int] = None,
    grad_clip: float = 1.0,
    log_interval: int = 5
) -> Dict[str, float]:
    """
    Executes Stage 1 Contrastive Pretraining Loop.
    """
    model.to(device)
    model.train()

    global_step = 0
    total_loss_accum = 0.0
    start_time = time.time()
    history = {"losses": [], "temperatures": []}

    print(f"\n[OmniDoc-RAG] Starting Stage 1 Pretraining on device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable\n")

    for epoch in range(epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader):
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()

            loss, metrics = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()
            total_loss_accum += loss.item()

            history["losses"].append(loss.item())
            history["temperatures"].append(metrics["temperature"].item())

            if global_step % log_interval == 0 or global_step == 1:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"Epoch [{epoch+1}/{epochs}] | Step [{global_step}] | "
                    f"Loss: {loss.item():.4f} | Q->D: {metrics['loss_q2d']:.4f} | "
                    f"D->Q: {metrics['loss_d2q']:.4f} | Tau: {metrics['temperature']:.4f} | "
                    f"LR: {lr:.2e}"
                )

            if max_steps is not None and global_step >= max_steps:
                break

        if max_steps is not None and global_step >= max_steps:
            break

    elapsed = time.time() - start_time
    avg_loss = total_loss_accum / max(1, global_step)
    print(f"\n[OmniDoc-RAG] Pretraining completed in {elapsed:.2f}s | Final Avg Loss: {avg_loss:.4f}")
    return history


def main():
    parser = argparse.ArgumentParser(description="OmniDoc-RAG Stage 1 Contrastive Training")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--patch_size", type=int, default=32, help="Patch size")
    parser.add_argument("--num_latents", type=int, default=32, help="Number of Perceiver latents")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, mps, cuda)")
    parser.add_argument("--max_steps", type=int, default=None, help="Max steps for debugging")
    parser.add_argument("--output_dir", type=str, default="checkpoints/stage1", help="Checkpoint dir")
    args = parser.parse_args()

    # Determine device
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = torch.device("cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Initialize synthetic dataset and collator
    dataset = SyntheticDocumentDataset(num_samples=32, img_size=128)
    collate_fn = OmniDocCollate(max_query_len=32)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Initialize Dual-Encoder Model
    model = OmniDocDualEncoder(
        embed_dim=args.embed_dim,
        patch_size=args.patch_size,
        num_latents=args.num_latents,
        perceiver_depth=2,
        heads=4,
        head_dim=args.embed_dim // 4,
        use_rope2d=True
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(dataloader) * args.epochs if args.max_steps is None else args.max_steps
    scheduler = build_cosine_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=total_steps)

    # Run training
    history = train_stage1(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        max_steps=args.max_steps
    )

    # Save checkpoint
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "omnidoc_stage1_latest.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args)
    }, save_path)
    print(f"[OmniDoc-RAG] Checkpoint saved to: {save_path}")


if __name__ == "__main__":
    main()
