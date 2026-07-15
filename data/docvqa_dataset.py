"""
DocVQA Multi-Modal Dataset and Dynamic Batching Collate for OmniDoc-RAG.

Synchronizes high-resolution document image tensors with variable-length text queries,
providing in-batch negative mining and padding masks for MaxSim late-interaction loss.
"""

from typing import List, Dict, Any, Optional, Union
from PIL import Image
import torch
from torch.utils.data import Dataset

from .pdf_processor import PDFProcessor


class DocVQADataset(Dataset):
    """
    PyTorch Dataset for Document Visual Question Answering & Retrieval Pairs.
    
    Accepts:
      - In-memory list of document-question dictionaries
      - Streaming Hugging Face DocVQA datasets
      - Local multi-page PDF files
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        processor: Optional[PDFProcessor] = None,
        target_size: tuple = (1024, 1024),
        dpi: int = 150
    ):
        """
        Args:
            samples: List of dicts, each containing:
                     - 'image': PIL Image, file path, or bytes
                     - 'question': Query string
                     - 'answers': List of answer strings (optional)
                     - 'doc_id': Unique document identifier (optional)
            processor: PDFProcessor instance (created if None)
            target_size: Target image canvas dimensions (H, W)
            dpi: Rendering DPI if source is a PDF
        """
        self.samples = samples
        self.processor = processor or PDFProcessor(default_dpi=dpi, target_size=target_size)
        self.target_size = target_size
        self.dpi = dpi

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]

        image_source = item["image"]
        
        # Handle query string or multilingual query dict (e.g. {'en': '...'})
        raw_query = item.get("question", item.get("query", "What is shown in this document?"))
        if isinstance(raw_query, dict):
            question = raw_query.get("en", next(iter(raw_query.values()), ""))
        elif isinstance(raw_query, (list, tuple)):
            question = str(raw_query[0]) if len(raw_query) > 0 else ""
        else:
            question = str(raw_query)

        answers = item.get("answers", item.get("answer", []))
        doc_id = str(item.get("doc_id", item.get("id", str(idx))))

        # Process image source into normalized PyTorch tensor: (3, H, W)
        if isinstance(image_source, Image.Image):
            padded_img, meta = self.processor.preprocess_image(image_source, target_size=self.target_size)
            image_tensor = self.processor.image_to_tensor(padded_img)
        elif isinstance(image_source, (str, bytes)):
            if isinstance(image_source, str) and (image_source.endswith(".png") or image_source.endswith(".jpg")):
                raw_img = Image.open(image_source).convert("RGB")
                padded_img, meta = self.processor.preprocess_image(raw_img, target_size=self.target_size)
                image_tensor = self.processor.image_to_tensor(padded_img)
            else:
                # PDF file path or raw bytes
                page_idx = item.get("page_idx", 0)
                image_tensor, meta = self.processor.process_pdf(
                    image_source, page_idx=page_idx, dpi=self.dpi, target_size=self.target_size
                )
        elif isinstance(image_source, torch.Tensor):
            image_tensor = image_source
        else:
            raise ValueError(f"Unsupported image source type: {type(image_source)}")

        return {
            "image": image_tensor,
            "question": str(question),
            "answers": answers,
            "doc_id": doc_id
        }


class OmniDocCollate:
    """
    Collate function for dynamic multi-modal batch construction.
    Stacks image tensors and tokenizes/pads query text with binary attention masks.
    """

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        max_query_len: int = 64
    ):
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Stack document images: (B, 3, H, W)
        images = torch.stack([item["image"] for item in batch], dim=0)

        questions = [item["question"] for item in batch]
        answers = [item["answers"] for item in batch]
        doc_ids = [item["doc_id"] for item in batch]

        # Tokenize queries
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                questions,
                padding=True,
                truncation=True,
                max_length=self.max_query_len,
                return_tensors="pt"
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
        else:
            # Standalone fallback: word-level hashing tokenizer for testing without external models
            max_len = min(self.max_query_len, max(len(q.split()) for q in questions) + 2)
            b_size = len(questions)
            input_ids = torch.zeros((b_size, max_len), dtype=torch.long)
            attention_mask = torch.zeros((b_size, max_len), dtype=torch.float32)

            for i, q in enumerate(questions):
                tokens = [abs(hash(w)) % 30000 + 1 for w in q.split()][:max_len - 1]
                input_ids[i, 0] = 101  # [CLS] token
                input_ids[i, 1:len(tokens) + 1] = torch.tensor(tokens, dtype=torch.long)
                attention_mask[i, :len(tokens) + 1] = 1.0

        return {
            "images": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "questions": questions,
            "answers": answers,
            "doc_ids": doc_ids
        }
