"""
High-DPI PDF Rasterization & Visual Preprocessing Pipeline for OmniDoc-RAG.

Renders vector PDF document pages into high-resolution RGB images using PyMuPDF,
applies isotropic aspect-ratio preserving scaling with neutral white canvas padding,
and converts to normalized PyTorch vision tensors.
"""

import io
from typing import Union, Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image
import torch
import pymupdf


class PDFProcessor:
    """
    High-DPI PDF Document Processor & Visual Transformer Preprocessor.
    """

    # ImageNet normalization statistics
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        default_dpi: int = 150,
        target_size: Tuple[int, int] = (1024, 1024),
        normalize: bool = True
    ):
        self.default_dpi = default_dpi
        self.target_size = target_size
        self.normalize = normalize

        # Precompute mean and std tensors for fast broadcasting: (3, 1, 1)
        self._mean_tensor = torch.tensor(self.IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self._std_tensor = torch.tensor(self.IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    def render_pdf_page(
        self,
        pdf_source: Union[str, bytes, io.BytesIO],
        page_idx: int = 0,
        dpi: Optional[int] = None
    ) -> Image.Image:
        """
        Renders a specific PDF page to a high-resolution PIL Image.
        
        Args:
            pdf_source: File path, raw bytes, or BytesIO stream of the PDF.
            page_idx: 0-indexed page number to render.
            dpi: Rendering DPI (defaults to self.default_dpi).
            
        Returns:
            PIL Image in RGB format.
        """
        target_dpi = dpi or self.default_dpi
        zoom = target_dpi / 72.0  # 72 points per inch standard

        if isinstance(pdf_source, (bytes, io.BytesIO)):
            raw_bytes = pdf_source.getvalue() if isinstance(pdf_source, io.BytesIO) else pdf_source
            doc = pymupdf.open(stream=raw_bytes, filetype="pdf")
        else:
            doc = pymupdf.open(pdf_source)

        if page_idx < 0 or page_idx >= len(doc):
            raise IndexError(f"Page index {page_idx} out of range for PDF with {len(doc)} pages.")

        page = doc[page_idx]
        mat = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Convert raw pixmap samples to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img

    def preprocess_image(
        self,
        image: Image.Image,
        target_size: Optional[Tuple[int, int]] = None,
        pad_value: int = 255,
        center: bool = False
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Isotropically rescales a document image and pads with neutral background.
        
        Args:
            image: Input PIL Image.
            target_size: Target (width, height) canvas.
            pad_value: Canvas padding color (default 255 = neutral white).
            center: If True, center image on canvas; if False, align top-left.
            
        Returns:
            Tuple of (padded PIL Image, metadata dictionary).
        """
        target_w, target_h = target_size or self.target_size
        orig_w, orig_h = image.size

        # Compute isotropic scale factor
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))

        # High-quality Lanczos antialiasing resize
        resized_img = image.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

        # Create neutral white canvas
        canvas = Image.new("RGB", (target_w, target_h), (pad_value, pad_value, pad_value))

        # Position on canvas
        if center:
            offset_x = (target_w - new_w) // 2
            offset_y = (target_h - new_h) // 2
        else:
            offset_x = 0
            offset_y = 0

        canvas.paste(resized_img, (offset_x, offset_y))

        meta = {
            "orig_size": (orig_w, orig_h),
            "scaled_size": (new_w, new_h),
            "scale": scale,
            "offset": (offset_x, offset_y),
            "target_size": (target_w, target_h)
        }
        return canvas, meta

    def image_to_tensor(
        self,
        image: Image.Image,
        normalize: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Converts a PIL Image to a PyTorch FloatTensor of shape (3, H, W).
        
        Args:
            image: PIL Image in RGB format.
            normalize: Whether to apply ImageNet normalization.
            
        Returns:
            Tensor of shape (3, H, W).
        """
        should_norm = self.normalize if normalize is None else normalize

        # Convert PIL -> NumPy array -> FloatTensor [0.0, 1.0]
        arr = np.array(image, dtype=np.float32) / 255.0  # (H, W, 3)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)

        if should_norm:
            tensor = (tensor - self._mean_tensor) / self._std_tensor

        return tensor

    def process_pdf(
        self,
        pdf_source: Union[str, bytes, io.BytesIO],
        page_idx: int = 0,
        dpi: Optional[int] = None,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        End-to-end processing pipeline:
          PDF -> High-DPI Render -> Isotropic Resize & White Pad -> Normalized Tensor.
          
        Returns:
            Tuple of (tensor: (3, H, W), metadata dictionary).
        """
        rendered_img = self.render_pdf_page(pdf_source, page_idx=page_idx, dpi=dpi)
        padded_img, meta = self.preprocess_image(rendered_img, target_size=target_size)
        tensor = self.image_to_tensor(padded_img)
        return tensor, meta
