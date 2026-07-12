"""
Unit tests for data/pdf_processor.py.
Tests isotropic image resizing, white padding canvas, tensor conversion, and PDF rasterization.
"""

import io
import pytest
import torch
import numpy as np
from PIL import Image
import pymupdf

from data.pdf_processor import PDFProcessor


def test_preprocess_image_aspect_ratio_and_padding():
    """Verify isotropic scaling and neutral white padding for tall portrait image."""
    processor = PDFProcessor(target_size=(1024, 1024))

    # Create a non-square portrait image (400 x 800)
    raw_img = Image.new("RGB", (400, 800), color=(100, 150, 200))

    padded_img, meta = processor.preprocess_image(raw_img, target_size=(1024, 1024), pad_value=255)

    assert padded_img.size == (1024, 1024)
    # Height should scale to 1024, width should scale to 512 (preserving 1:2 ratio)
    assert meta["scaled_size"] == (512, 1024)
    assert meta["scale"] == pytest.approx(1024 / 800, rel=1e-3)

    # Check that padded right margin is white (255, 255, 255)
    padded_arr = np.array(padded_img)
    # Right border pixel (x=1000, y=500) must be white canvas
    assert (padded_arr[500, 1000] == [255, 255, 255]).all()


def test_image_to_tensor_normalization():
    """Verify tensor conversion and statistical normalization."""
    processor = PDFProcessor(normalize=True)

    img = Image.new("RGB", (256, 256), color=(255, 255, 255))
    tensor = processor.image_to_tensor(img, normalize=False)

    assert tensor.shape == (3, 256, 256)
    assert tensor.dtype == torch.float32
    assert torch.allclose(tensor, torch.tensor(1.0), atol=1e-4)

    # Test ImageNet normalized version
    norm_tensor = processor.image_to_tensor(img, normalize=True)
    assert norm_tensor.shape == (3, 256, 256)
    # For white (1.0), normalized value should be (1.0 - mean) / std
    expected_r = (1.0 - 0.485) / 0.229
    assert norm_tensor[0, 0, 0].item() == pytest.approx(expected_r, rel=1e-3)


def test_synthetic_pdf_rasterization_pipeline():
    """Verify end-to-end PDF processing using an in-memory synthetic PDF document."""
    # Programmatically create a synthetic PDF using PyMuPDF
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # A4 dimensions in points
    page.insert_text((50, 72), "OmniDoc-RAG Synthetic Test Document", fontsize=18)
    page.insert_text((50, 120), "Table 1: Financial Performance Analysis", fontsize=14)
    pdf_bytes = doc.tobytes()
    doc.close()

    processor = PDFProcessor(default_dpi=150, target_size=(512, 512))

    tensor, meta = processor.process_pdf(pdf_bytes, page_idx=0, dpi=150, target_size=(512, 512))

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 512, 512)
    assert meta["target_size"] == (512, 512)
    assert meta["orig_size"][0] > 0
    assert meta["orig_size"][1] > 0
