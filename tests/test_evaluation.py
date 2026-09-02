"""
Unit tests for the evaluation and benchmarking suite (tests/test_evaluation.py).
Verifies Recall@K, MRR, Exact Match, and Token F1 calculations on CPU in <0.1s.
"""

import pytest
import torch

from evaluate_benchmarks import (
    compute_retrieval_metrics,
    compute_vqa_f1_and_em,
    generate_benchmark_summary
)


def test_retrieval_metrics_exact_recovery():
    """Verify Recall@1 and MRR equal 100% when queries and docs are aligned."""
    num_samples = 4
    num_query_tokens = 5
    num_doc_latents = 4
    dim = 16

    # Construct orthogonal / distinct queries and documents
    queries = torch.eye(num_samples, dim).unsqueeze(1).repeat(1, num_query_tokens, 1)
    docs = torch.eye(num_samples, dim).unsqueeze(1).repeat(1, num_doc_latents, 1)

    metrics = compute_retrieval_metrics(queries, docs, k_values=(1, 2))

    assert metrics["Recall@1"] == 100.0
    assert metrics["Recall@2"] == 100.0
    assert metrics["MRR"] == 1.0


def test_vqa_exact_match_and_token_f1():
    """Verify token F1 and Exact Match on sample QA predictions."""
    preds = ["net revenue is $14.2M", "table 3", "April 1998", "wrong answer"]
    gts = ["net revenue is $14.2M", "table 3", "April 1998", "correct answer"]

    results = compute_vqa_f1_and_em(preds, gts)

    # 3 out of 4 are exact matches -> EM = 75.0%
    assert results["ExactMatch"] == 75.0
    assert results["TokenF1"] > 70.0


def test_benchmark_summary_generation():
    """Verify benchmark summary report string formatting."""
    omni = {"Recall@1": "72.40%", "Recall@5": "91.20%", "MRR": "0.8120", "ExactMatch": "65.10%", "TokenF1": "81.00%"}
    ocr = {"Recall@1": "14.20%", "Recall@5": "38.60%", "MRR": "0.2240", "ExactMatch": "21.50%", "TokenF1": "34.80%"}

    report = generate_benchmark_summary(omni, ocr)

    assert "OmniDoc-RAG vs. Traditional OCR RAG" in report
    assert "72.40%" in report
    assert "Zero OCR Loss" in report
