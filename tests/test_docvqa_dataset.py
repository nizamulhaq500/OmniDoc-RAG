"""
Unit tests for data/docvqa_dataset.py.
Tests dataset item processing, collate batching, and integration with OmniDocDualEncoder on CPU.
"""

import pytest
import torch
from torch.utils.data import DataLoader
from PIL import Image

from data.docvqa_dataset import DocVQADataset, OmniDocCollate
from models.omni_encoder import OmniDocDualEncoder


def test_docvqa_dataset_getitem():
    """Verify single-sample extraction and preprocessing."""
    # Synthetic samples with PIL images
    sample_images = [
        Image.new("RGB", (300, 400), color=(200, 200, 200)),
        Image.new("RGB", (500, 300), color=(220, 220, 220))
    ]
    samples = [
        {"image": sample_images[0], "question": "What is the total revenue?", "answers": ["$5.2M"], "doc_id": "doc_01"},
        {"image": sample_images[1], "question": "Who signed the document?", "answers": ["John Doe"], "doc_id": "doc_02"}
    ]

    dataset = DocVQADataset(samples, target_size=(256, 256))
    assert len(dataset) == 2

    item0 = dataset[0]
    assert item0["image"].shape == (3, 256, 256)
    assert item0["question"] == "What is the total revenue?"
    assert item0["doc_id"] == "doc_01"


def test_omnidoc_collate_batching():
    """Verify dynamic padding and attention mask construction across variable queries."""
    sample_images = [Image.new("RGB", (200, 200), color=(250, 250, 250)) for _ in range(3)]
    samples = [
        {"image": sample_images[0], "question": "Short query", "doc_id": "1"},
        {"image": sample_images[1], "question": "A much longer document analytical query string", "doc_id": "2"},
        {"image": sample_images[2], "question": "Medium query here", "doc_id": "3"}
    ]

    dataset = DocVQADataset(samples, target_size=(128, 128))
    collate_fn = OmniDocCollate(max_query_len=32)

    loader = DataLoader(dataset, batch_size=3, collate_fn=collate_fn)
    batch = next(iter(loader))

    assert batch["images"].shape == (3, 3, 128, 128)
    assert batch["input_ids"].shape[0] == 3
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    # Mask should have 1.0 for valid tokens
    assert batch["attention_mask"].sum() > 0


def test_dataloader_to_omni_encoder_pipeline():
    """Verify full end-to-end forward pass from DataLoader through OmniDocDualEncoder."""
    samples = [
        {"image": Image.new("RGB", (100, 100)), "question": f"Question number {i}", "doc_id": f"id_{i}"}
        for i in range(4)
    ]

    dataset = DocVQADataset(samples, target_size=(64, 64))
    collate_fn = OmniDocCollate(max_query_len=16)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    model = OmniDocDualEncoder(
        embed_dim=32,
        patch_size=16,
        num_latents=4,
        perceiver_depth=1,
        heads=2,
        head_dim=16,
        vocab_size=30522,
        use_rope2d=True
    )

    batch = next(iter(loader))
    loss, metrics = model(
        images=batch["images"],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"]
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0
    assert "loss_q2d" in metrics
    assert "temperature" in metrics
