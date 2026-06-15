# OmniDoc-RAG: OCR-Free Multi-Modal Vision Document Retrieval & Reasoning Engine

[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-12%2F12%20Passing-brightgreen.svg)]()

> **OmniDoc-RAG** is an OCR-free, multi-modal document retrieval and question-answering architecture mathematically engineered from scratch in PyTorch. It eliminates fragile text-extraction pipelines by operating directly on high-resolution visual document patch grids via custom 2D Spatial Rotary Embeddings and learned latent cross-attention bottlenecks.

---

## 🏛️ Architectural Overview

```
[Document Page Image (1024x1024)]                        [Text Query ("Total Revenue?")]
               │                                                          │
               ▼                                                          ▼
    [Vision Patch Extractor]                                    [Query Token Encoder]
  (N=1024 Spatial Tokens, D=768)                               (L Query Vectors, D=768)
               │                                                          │
               ▼                                                          │
    [Custom 2D-RoPE Module]                                               │
(Phase Rotation in Complex Frequency Space)                               │
               │                                                          │
               ▼                                                          │
  [Custom Perceiver Resampler]                                            │
  (K=64 Latent Cross-Attention Bottleneck)                                │
               │                                                          │
               ▼                                                          │
    Normalized Visual Latents                                  Normalized Query Tokens
         (B, K=64, D)                                               (B, L, D)
               │                                                          │
               └────────────────────────┬─────────────────────────────────┘
                                        ▼
                          [Late-Interaction MaxSim]
                    Score = Σ_{l=1}^L max_{k=1}^K (q_l · d_k)
                                        │
                                        ▼
                   [Symmetric Multi-Scale Patch-InfoNCE]
```

---

## 🔬 Core Mathematical Innovations

### 1. 2D Spatial Rotary Position Embedding (`models/rope2d.py`)
Standard 1D raster-scan embeddings artificially distort 2D geometry (vertical neighbors on a $32 \times 32$ grid appear 32 tokens apart). **2D-RoPE** decomposes the hidden dimension into orthogonal vertical ($y$) and horizontal ($x$) frequency manifolds:

$$\langle R^{2D}(y_1, x_1) \mathbf{q}, \, R^{2D}(y_2, x_2) \mathbf{k} \rangle = g\Big(\mathbf{q}, \mathbf{k}, \, \underbrace{y_1 - y_2}_{\Delta y}, \, \underbrace{x_1 - x_2}_{\Delta x}\Big)$$

* **Translation Invariance:** Exact 2D relative displacement $(\Delta y, \Delta x)$ preserved under dot-products.
* **Zero Additional Parameters:** Analytic continuous frequency rotation.

### 2. Custom Perceiver Resampler Bottleneck (`models/perceiver.py`)
Compresses high-resolution visual context ($N=1024$ patches) down to $K=64$ dense visual latents ($16\times$ compression) using multi-head cross-attention with learned latent queries:
* **Compute Reduction:** Replaces quadratic $O(N^2)$ self-attention with $O(K \cdot N)$ cross-attention.
* **Layout Preservation:** Dynamic content-aware routing preserves fine table borders, superscript digits, and dense financial data.

### 3. Symmetric Multi-Scale Patch-InfoNCE Loss (`losses/contrastive_loss.py`)
Computes fine-grained token-to-patch alignment via the **MaxSim Late-Interaction operator**:

$$\text{Score}(\mathbf{Q}, \mathbf{D}) = \sum_{l=1}^L \max_{k \in [1, K]} \left( \mathbf{q}_l \cdot \mathbf{d}_k \right)$$

$$\mathcal{L}_{\text{Patch-InfoNCE}} = \frac{1}{2} \left( \mathcal{L}_{Q \to D} + \mathcal{L}_{D \to Q} \right)$$

---

## 📁 Repository Structure

```
OmniDoc-RAG/
├── docs/
│   └── MATHEMATICAL_FOUNDATIONS.md  # PhD-grade derivations, proofs & complexity analysis
├── models/
│   ├── __init__.py
│   ├── rope2d.py                   # Custom 2D Spatial Rotary Position Embeddings
│   ├── perceiver.py                # Custom Perceiver Resampler Cross-Attention Bottleneck
│   └── omni_encoder.py             # Unified Dual-Encoder Architecture
├── losses/
│   ├── __init__.py
│   └── contrastive_loss.py         # Symmetric Multi-Scale Patch-InfoNCE with MaxSim
├── data/                           # High-DPI PDF rasterization & DocVQA datasets
├── tests/
│   ├── test_rope2d.py              # Translation invariance & orthogonality tests
│   ├── test_perceiver.py           # Compression & gradient backprop tests
│   ├── test_contrastive_loss.py    # MaxSim & symmetric InfoNCE loss tests
│   └── test_omni_encoder.py        # End-to-end forward & backward pass tests
├── requirements.txt
└── README.md
```

---

## ⚡ Quickstart & Testing

```bash
# Clone the repository
git clone https://github.com/nizamulhaq500/OmniDoc-RAG.git
cd OmniDoc-RAG

# Create virtual environment & install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run complete unit test suite (CPU/MPS verified)
pytest tests/ -v
```

---

## 📖 Theoretical Documentation
For complete mathematical proofs, tensor maps, and complexity comparisons, see [`docs/MATHEMATICAL_FOUNDATIONS.md`](docs/MATHEMATICAL_FOUNDATIONS.md).
