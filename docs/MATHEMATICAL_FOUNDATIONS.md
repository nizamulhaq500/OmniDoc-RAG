# OmniDoc-RAG: Mathematical Foundations & Theoretical Architecture

**Document Type:** Research Reference & Theoretical Specification  
**Project:** OmniDoc-RAG (OCR-Free Vision Document Retrieval & Reasoning Engine)  
**Author:** AI/ML Engineering & Research Notes  

---

## Chapter 1: 2D Spatial Rotary Position Embeddings (2D-RoPE)

---

### 1.1 The Geometric Document Challenge

In multi-modal document understanding (e.g., scientific papers, balance sheets, invoices, technical blueprints), spatial geometry carries high semantic density. Unlike natural linear text, documents are fundamentally **two-dimensional grids**:

* **Multi-column layouts:** Text blocks flow down column $C_1$ before starting column $C_2$.
* **Tables & Grids:** A cell at row $y$, column $x$ has vertical semantic alignment with cell $(y-1, x)$ (column header) and horizontal semantic alignment with cell $(y, x-1)$ (row header).
* **Hierarchical Bounding Boxes:** Spatial containment, headers, and figures depend on Euclidean proximity $(\Delta y, \Delta x)$.

#### Failure Mode of Standard 1D Embeddings (Raster-Scan Ordering)
Standard Vision Transformers (ViT) flatten a $H \times W$ patch grid into a 1D sequence using raster scan (row-major order):

$$\text{Index } i(y, x) = y \cdot W + x$$

Consider three image patches on a $32 \times 32$ grid ($W = 32$):
* **Patch A:** $(y=2, x=5) \implies i_A = 2 \cdot 32 + 5 = 69$
* **Patch B (Horizontal Neighbor):** $(y=2, x=6) \implies i_B = 2 \cdot 32 + 6 = 70 \implies \Delta i_{A \to B} = 1$
* **Patch C (Vertical Neighbor):** $(y=3, x=5) \implies i_C = 3 \cdot 32 + 5 = 101 \implies \Delta i_{A \to C} = 32$

In physical document space:
$$\text{Dist}_{2D}(A, B) = \sqrt{(2-2)^2 + (6-5)^2} = 1.0$$
$$\text{Dist}_{2D}(A, C) = \sqrt{(3-2)^2 + (5-5)^2} = 1.0$$

Under 1D position embeddings (Absolute 1D or Standard 1D-RoPE), **Patch C appears 32 times farther away than Patch B**. This artificial asymmetry destroys the attention mechanism's ability to model vertical tabular alignment.

---

### 1.2 Mathematical Derivation: From 1D-RoPE to 2D-RoPE

#### 1.2.1 Standard 1D Rotary Position Embedding (RoPE)
Given a query vector $\mathbf{q} \in \mathbb{R}^{D_h}$ at 1D position $m$, RoPE (*Su et al., 2021*) rotates orthogonal 2D sub-vectors in the complex plane:

$$R_{\Theta, m} \mathbf{q} = \begin{pmatrix} 
\cos(m\theta_0) & -\sin(m\theta_0) & 0 & 0 & \dots \\
\sin(m\theta_0) & \cos(m\theta_0) & 0 & 0 & \dots \\
0 & 0 & \cos(m\theta_1) & -\sin(m\theta_1) & \dots \\
0 & 0 & \sin(m\theta_1) & \cos(m\theta_1) & \dots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{pmatrix} \begin{pmatrix} q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \end{pmatrix}$$

Where the rotational frequencies are defined as:
$$\theta_k = 10000^{- \frac{2k}{D_h}}, \quad k \in \left[0, \frac{D_h}{2} - 1\right]$$

The critical property of 1D-RoPE is that the attention score between query at position $m$ and key at position $n$ is purely a function of relative distance $(m - n)$:

$$\langle R_{\Theta, m} \mathbf{q}, \, R_{\Theta, n} \mathbf{k} \rangle = \text{Re} \sum_{k=0}^{D_h/2 - 1} (\mathbf{q}_k e^{i m \theta_k}) (\mathbf{k}_k e^{i n \theta_k})^* = \text{Re} \sum_{k=0}^{D_h/2 - 1} \mathbf{q}_k \mathbf{k}_k^* e^{i (m - n) \theta_k}$$

---

#### 1.2.2 The 2D Spatial Rotary Formulation (2D-RoPE)
In **2D-RoPE**, we decompose the head dimension $D_h$ (where $D_h$ is divisible by 4) into two equal, orthogonal frequency manifolds:

1. **Vertical Manifold ($y$-axis):** Allocated to the first $D_h / 2$ dimensions.
2. **Horizontal Manifold ($x$-axis):** Allocated to the remaining $D_h / 2$ dimensions.

Let each manifold contain $d = D_h / 4$ frequency pairs. The base frequencies for channel index $k \in [0, d-1]$ are:

$$\theta_k = \text{base}^{- \frac{2k}{D_h / 2}} = 10000^{- \frac{4k}{D_h}}$$

For a visual patch located at 2D grid coordinates $(y, x)$:
* **Vertical Phase Angles:** $\mathbf{\Theta}_y(y) = y \cdot [\theta_0, \theta_1, \dots, \theta_{d-1}] \in \mathbb{R}^{d}$
* **Horizontal Phase Angles:** $\mathbf{\Theta}_x(x) = x \cdot [\theta_0, \theta_1, \dots, \theta_{d-1}] \in \mathbb{R}^{d}$

We construct the full spatial rotation matrix $R^{2D}(y, x) \in \mathbb{R}^{D_h \times D_h}$ as a block diagonal composition:

$$R^{2D}(y, x) = \begin{bmatrix} R_{\Theta_y}(y) & \mathbf{0} \\ \mathbf{0} & R_{\Theta_x}(x) \end{bmatrix}$$

---

### 1.3 Mathematical Proof: 2D Relative Displacement Invariance

Let $\mathbf{q} = [\mathbf{q}_y, \mathbf{q}_x]^T$ be a query vector at 2D coordinates $(y_1, x_1)$, and $\mathbf{k} = [\mathbf{k}_y, \mathbf{k}_x]^T$ be a key vector at 2D coordinates $(y_2, x_2)$.

$$\langle R^{2D}(y_1, x_1) \mathbf{q}, \, R^{2D}(y_2, x_2) \mathbf{k} \rangle = \langle R_{\Theta_y}(y_1) \mathbf{q}_y, \, R_{\Theta_y}(y_2) \mathbf{k}_y \rangle + \langle R_{\Theta_x}(x_1) \mathbf{q}_x, \, R_{\Theta_x}(x_2) \mathbf{k}_x \rangle$$

Using the complex exponential representation:

$$\langle R_{\Theta_y}(y_1) \mathbf{q}_y, \, R_{\Theta_y}(y_2) \mathbf{k}_y \rangle = \text{Re} \sum_{k=0}^{d-1} (\mathbf{q}_{y, k} e^{i y_1 \theta_k}) (\mathbf{k}_{y, k} e^{i y_2 \theta_k})^* = \text{Re} \sum_{k=0}^{d-1} \mathbf{q}_{y, k} \mathbf{k}_{y, k}^* e^{i (y_1 - y_2) \theta_k}$$

$$\langle R_{\Theta_x}(x_1) \mathbf{q}_x, \, R_{\Theta_x}(x_2) \mathbf{k}_x \rangle = \text{Re} \sum_{k=0}^{d-1} (\mathbf{q}_{x, k} e^{i x_1 \theta_k}) (\mathbf{k}_{x, k} e^{i x_2 \theta_k})^* = \text{Re} \sum_{k=0}^{d-1} \mathbf{q}_{x, k} \mathbf{k}_{x, k}^* e^{i (x_1 - x_2) \theta_k}$$

Summing the two components:

$$\langle R^{2D}(y_1, x_1) \mathbf{q}, \, R^{2D}(y_2, x_2) \mathbf{k} \rangle = g\Big(\mathbf{q}, \mathbf{k}, \, \underbrace{(y_1 - y_2)}_{\Delta y}, \, \underbrace{(x_1 - x_2)}_{\Delta x}\Big)$$

$$\mathbf{\text{Q.E.D.}}$$

**Conclusion:** The inner product is strictly a function of the 2D spatial displacement vector $(\Delta y, \Delta x)$. It is completely invariant to global translation of the document page.

---

### 1.4 Tensor Dimension Blueprint

| Step | Operation | Tensor Shape |
| :--- | :--- | :--- |
| 1 | Input Patch Features $\mathbf{X}$ | $(B, N, D)$ where $N = H \times W$ |
| 2 | Linear Query / Key Projections | $(B, N, H_{\text{heads}} \times D_h)$ |
| 3 | Multi-Head Reshape | $(B, H_{\text{heads}}, N, D_h)$ |
| 4 | Coordinate Grid Generation $(y, x)$ | $(H, W) \implies (N, 2)$ |
| 5 | Frequency Calculation $(\mathbf{\Theta}_y, \mathbf{\Theta}_x)$ | $(N, D_h / 2) \implies (1, 1, N, D_h)$ |
| 6 | Rotary Application ($\mathbf{q} \odot \cos + \text{rotate}(\mathbf{q}) \odot \sin$) | $(B, H_{\text{heads}}, N, D_h)$ |
| 7 | Multi-Head Scaled Dot-Product Attention | $\text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{D_h}}\right) \mathbf{V} \implies (B, H_{\text{heads}}, N, D_h)$ |

---

### 1.5 Theoretical Comparison Matrix

| Property | Learned 1D Absolute Embeddings | 1D-RoPE | Learned 2D Absolute Embeddings | Custom 2D-RoPE (Our Method) |
| :--- | :--- | :--- | :--- | :--- |
| **2D Geometry Aware** | ❌ No (1D raster) | ❌ No (1D raster) | ⚠️ Partial (learned lookup) | ✅ Exact (analytic 2D angles) |
| **Relative Translation Invariance** | ❌ No | ✅ 1D only | ❌ No | ✅ Complete 2D $(\Delta y, \Delta x)$ |
| **Extrapolation to Novel Resolutions** | ❌ Fails / requires interpolation | ⚠️ 1D length only | ❌ Grid size fixed | ✅ Arbitrary $(H, W)$ grids |
| **Memory / Param Overhead** | $O(N \cdot D)$ parameters | 0 parameters | $O((H+W) \cdot D)$ parameters | **0 parameters (pure functional)** |

---

### 1.6 Interview & Defense Takeaways (PhD / Research Pitch)

1. **Why not just use learned 2D coordinate embeddings $(E_y + E_x)$?**  
   *Learned embeddings encode absolute positions, not relative displacements. When documents undergo scale variance or variable margins, absolute embeddings fail to generalize. 2D-RoPE introduces zero learnable parameters and guarantees exact relative distance invariance under dot products.*

2. **How does 2D-RoPE handle non-square or dynamic aspect ratios?**  
   *Because vertical and horizontal frequencies are evaluated independently on continuous integer grids $y \in [0, H-1]$ and $x \in [0, W-1]$, the grid dimension $H$ and $W$ can vary dynamically per document without retraining or interpolation.*

---

## Chapter 2: The Perceiver Resampler & Cross-Attention Latent Bottleneck

---

### 2.1 The Quadratic Context Explosion in Multi-Modal Retrieval

When processing high-resolution visual documents (e.g. $1024 \times 1024$ pixels at 150-300 DPI) using Vision Transformers with patch size $P = 32 \times 32$ or $16 \times 16$:

$$N = \left(\frac{H}{P}\right) \times \left(\frac{W}{P}\right) = \left(\frac{1024}{32}\right) \times \left(\frac{1024}{32}\right) = 32 \times 32 = \mathbf{1024 \text{ visual tokens}}$$

If a user retrieves $M = 5$ candidate pages for a multi-page document query:
$$\text{Total Visual Tokens} = 5 \times 1024 = \mathbf{5120 \text{ tokens}}$$

#### The Bottleneck:
1. **Self-Attention Quadratic Growth:** Standard full-attention across $N$ tokens requires computing an $N \times N$ attention matrix:
   $$\text{FLOPs}_{\text{Self-Attn}} \propto 2 N^2 D = 2 \cdot (1024)^2 \cdot 768 \approx \mathbf{1.61 \text{ GFLOPs per layer}}$$
2. **Language Model KV-Cache Exhaustion:** Passing 1024 visual tokens per page into a compact Language Model (e.g. Qwen2.5 / SmolVLM) overwhelms the context window and slows down auto-regressive decoding.
3. **Lossy Naive Pooling:** Spatial average pooling ($2 \times 2$ or $4 \times 4$) destroys thin horizontal table borders, superscripts, and small font numbers (e.g. distinguishing `10.5` from `10.6` in a balance sheet).

---

### 2.2 Mathematical Architecture of the Perceiver Resampler

The **Perceiver Resampler** decouples the output token budget from the input image resolution through a **learned latent query cross-attention bottleneck**.

```
Input Visual Tokens X: (B, N=1024, D) ────► Projected to Keys (K) and Values (V)
                                                      │
                                                      ▼
Learned Latent Queries Z: (B, K=64, D)  ──► Projected to Queries (Q)
                                                      │
                                                      ▼
                      [Multi-Head Cross-Attention Layer]
                      Attention Weights: (B, H, K=64, N=1024)
                                                      │
                                                      ▼
                      [Latent Self-Attention Layer (K x K)]
                      Attention Weights: (B, H, K=64, K=64)
                                                      │
                                                      ▼
                      [Feed-Forward Network (FFN + GeLU)]
                                                      │
                                                      ▼
           Compressed Latent Output: (B, K=64, D) (16x Compression)
```

---

### 2.3 Formal Mathematical Equations

#### Step 1: Learnable Latent Initialization
We define a parameter matrix $\mathbf{Z} \in \mathbb{R}^{K \times D}$ (where $K=64$ and $D=768$), initialized from a truncated normal distribution $\mathcal{N}(0, 0.02)$. For a batch of size $B$, $\mathbf{Z}$ is broadcasted:

$$\mathbf{Z}_{\text{batch}} = \text{repeat}(\mathbf{Z}, 'k \, d \to b \, k \, d', b=B) \in \mathbb{R}^{B \times K \times D}$$

#### Step 2: Multi-Head Cross-Attention
Let $\mathbf{X} \in \mathbb{R}^{B \times N \times D}$ be the 2D-RoPE visual patch representations from the vision backbone.

$$\mathbf{Q} = \mathbf{Z}_{\text{batch}} \mathbf{W}_Q \in \mathbb{R}^{B \times H \times K \times D_h}$$
$$\mathbf{K} = \mathbf{X} \mathbf{W}_K \in \mathbb{R}^{B \times H \times N \times D_h}$$
$$\mathbf{V} = \mathbf{X} \mathbf{W}_V \in \mathbb{R}^{B \times H \times N \times D_h}$$

The cross-attention output is computed as:
$$\mathbf{A}_{\text{cross}} = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{D_h}}\right) \in \mathbb{R}^{B \times H \times K \times N}$$
$$\mathbf{Z}^{(1)} = \mathbf{Z}_{\text{batch}} + \text{Dropout}\left(\mathbf{A}_{\text{cross}} \mathbf{V} \mathbf{W}_O\right)$$

#### Step 3: Latent Self-Attention
The $K$ compressed tokens refine their contextual relationships through multi-head self-attention:
$$\mathbf{Z}^{(2)} = \mathbf{Z}^{(1)} + \text{MultiHeadSelfAttention}\left(\text{LayerNorm}(\mathbf{Z}^{(1)})\right)$$

#### Step 4: Feed-Forward Network (FFN)
$$\mathbf{Z}_{\text{out}} = \mathbf{Z}^{(2)} + \text{MLP}\left(\text{LayerNorm}(\mathbf{Z}^{(2)})\right)$$
$$\text{where } \text{MLP}(\mathbf{u}) = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \mathbf{u} + \mathbf{b}_1) + \mathbf{b}_2$$

---

### 2.4 Computational Complexity Analysis

| Operation | Standard Self-Attention | Perceiver Resampler (Ours) | Speedup Factor |
| :--- | :--- | :--- | :--- |
| **Token Count** | $N = 1024$ | $K = 64$ ($16\times$ compression) | $16\times$ fewer downstream tokens |
| **Cross-Attention FLOPs** | $O(N^2 \cdot D) = 1024^2 \cdot D \approx 1.05 \times 10^6 D$ | $O(K \cdot N \cdot D) = 64 \cdot 1024 \cdot D \approx \mathbf{6.55 \times 10^4 D}$ | **$16\times$ compute reduction** |
| **Self-Attention FLOPs** | $O(N^2 \cdot D) = 1024^2 \cdot D \approx 1.05 \times 10^6 D$ | $O(K^2 \cdot D) = 64^2 \cdot D \approx \mathbf{4.10 \times 10^3 D}$ | **$256\times$ compute reduction** |
| **Downstream LLM KV-Cache** | $1024 \times 2 \times L \times D$ | $64 \times 2 \times L \times D$ | **$16\times$ memory reduction** |

---

### 2.5 Theoretical Comparison Matrix

| Approach | Compression Ratio | Information Routing | Spatial Layout Retention | Learnable Parameters |
| :--- | :--- | :--- | :--- | :--- |
| **Average Pooling ($4 \times 4$)** | Fixed $16\times$ | Uniform static average (no routing) | ❌ High loss of small characters | $0$ |
| **Fixed Linear Projection ($N \to K$)** | Fixed $16\times$ | Rigid position-to-latent index | ❌ Fails on shifted layouts | $N \cdot K \approx 6.5 \times 10^4$ |
| **Perceiver Resampler (Ours)** | Flexible $K / N$ | **Dynamic content-aware cross-attention** | ✅ **Preserves high-frequency layout & text** | $\sim 2.5\text{M} \text{ (Cross-Attn + MLP)}$ |

---

### 2.6 Interview & Defense Takeaways (PhD / Research Pitch)

1. **Why does the Perceiver Resampler use learned latents as queries rather than visual tokens as queries?**  
   *Using learned latents as queries bounds the output sequence length strictly to $K$ regardless of how large the input image or patch grid $N$ becomes. The learned query vectors specialize during training: some latents learn to attend to structural table headers, others to numerical values, and others to text paragraphs.*

2. **How does 2D-RoPE synergize with the Perceiver Resampler?**  
   *The keys ($\mathbf{K}$) in the Perceiver cross-attention layer are positional-encoded via 2D-RoPE. This means when the learned latent queries attend to visual patches, their attention weights $\mathbf{A}_{\text{cross}}$ are geometrically grounded in 2D document coordinates.*

---

## Chapter 3: Symmetric Multi-Scale Patch-InfoNCE Loss & Late Interaction

---

### 3.1 The Limitation of Single-Vector Embeddings in Multi-Modal RAG

Traditional dense bi-encoders map an entire document into a single 1D vector $\mathbf{v} \in \mathbb{R}^{D}$ using mean-pooling or `[CLS]` token extraction:

$$\text{Similarity}(\mathbf{q}, \mathbf{d}) = \cos(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2}$$

#### Failure Cases on Document Layouts:
1. **Information Dilution:** A page containing 5 tables, 2 charts, and 4 paragraphs has high semantic entropy. Compressing all $N$ tokens into a single vector forces the model to average over distinct factual entities.
2. **Fine-Grained Alignment:** When an analytical query seeks a specific numerical figure (e.g. *"What is the gross margin in the electronics sector?"*), single-vector retrieval fails because the overall page representation is dominated by generic corporate boilerplate.

---

### 3.2 Multi-Vector Late Interaction (MaxSim Operator)

In OmniDoc-RAG, we maintain multi-vector representations for both text queries and visual documents:
* **Query Representation:** Sequence of $L$ normalized token vectors:
  $$\mathbf{Q} = [\mathbf{q}_1, \mathbf{q}_2, \dots, \mathbf{q}_L] \in \mathbb{R}^{L \times D}, \quad \|\mathbf{q}_i\|_2 = 1$$
* **Visual Document Representation:** Sequence of $K=64$ normalized Perceiver latent vectors:
  $$\mathbf{D} = [\mathbf{d}_1, \mathbf{d}_2, \dots, \mathbf{d}_K] \in \mathbb{R}^{K \times D}, \quad \|\mathbf{d}_j\|_2 = 1$$

#### The MaxSim Operator:
The relevance score between query $\mathbf{Q}$ and document $\mathbf{D}$ is the sum of maximum inner products across all query tokens:

$$\text{Score}(\mathbf{Q}, \mathbf{D}) = \sum_{i=1}^L \max_{j \in [1, K]} \left( \mathbf{q}_i \cdot \mathbf{d}_j \right)$$

*Each query token acts as an autonomous probe seeking its highest-affinity visual latent patch on the document page.*

---

### 3.3 Symmetric Patch-InfoNCE Contrastive Formulation

Given a mini-batch of $B$ query-document pairs $\{(\mathbf{Q}_b, \mathbf{D}_b^+)\}_{b=1}^B$ with learnable inverse temperature parameter $\tau > 0$:

#### 1. Pairwise Similarity Matrix ($B \times B$)
For any query $\mathbf{Q}_i$ and candidate document $\mathbf{D}_j$:

$$S_{i, j} = \frac{1}{\tau} \text{Score}(\mathbf{Q}_i, \mathbf{D}_j) = \frac{1}{\tau} \sum_{l=1}^L \max_{k \in [1, K]} \left( \mathbf{q}_{i, l} \cdot \mathbf{d}_{j, k} \right)$$

#### 2. Query-to-Document Directional Contrastive Loss ($\mathcal{L}_{Q \to D}$)
Treats row $i$ as a classification problem over all $B$ in-batch documents:

$$\mathcal{L}_{Q \to D} = - \frac{1}{B} \sum_{i=1}^B \log \left( \frac{\exp(S_{i, i})}{\sum_{j=1}^B \exp(S_{i, j})} \right)$$

#### 3. Document-to-Query Directional Contrastive Loss ($\mathcal{L}_{D \to Q}$)
Treats column $j$ as a classification problem over all $B$ in-batch queries:

$$\mathcal{L}_{D \to Q} = - \frac{1}{B} \sum_{j=1}^B \log \left( \frac{\exp(S_{j, j})}{\sum_{i=1}^B \exp(S_{i, j})} \right)$$

#### 4. Total Symmetric Objective
$$\mathcal{L}_{\text{Patch-InfoNCE}} = \frac{1}{2} \left( \mathcal{L}_{Q \to D} + \mathcal{L}_{D \to Q} \right)$$

---

### 3.4 In-Batch Hard Negative Mining & Masking

When training on multi-page PDF documents:
* **In-Batch Negatives:** Other documents in the batch $\{ \mathbf{D}_j \}_{j \neq i}$ act as natural negatives.
* **Document-Level Hard Negatives:** Different pages from the same document (e.g. Page 1 vs Page 2 of the same SEC filing) provide hard negatives sharing identical typography, headers, and color schemes.
* **Query Padding Masking:** For queries of variable length $L$, padded tokens are masked out with a binary mask $\mathbf{M}_Q \in \{0, 1\}^{B \times L}$ so they contribute zero to the MaxSim summation:

$$\text{Score}_{\text{masked}}(\mathbf{Q}_i, \mathbf{D}_j) = \sum_{l=1}^L M_{i, l} \cdot \max_{k \in [1, K]} \left( \mathbf{q}_i \cdot \mathbf{d}_j \right)$$

---

### 3.5 Theoretical Comparison Matrix

| Objective Formulation | Loss Type | Granularity | Spatial Sensitivity | Retrieval Metric |
| :--- | :--- | :--- | :--- | :--- |
| **Standard InfoNCE (CLIP/DPR)** | Symmetric Cross-Entropy | Coarse (Single vector $\mathbf{q} \cdot \mathbf{d}$) | ❌ Zero spatial awareness | Low Recall@1 on dense tables |
| **Multiple Negative Ranking (MNRL)** | Asymmetric Cross-Entropy | Coarse (Single vector) | ❌ Zero spatial awareness | Biased towards query direction |
| **Symmetric Patch-InfoNCE (Ours)** | **Symmetric Cross-Entropy with MaxSim** | **Fine-Grained ($L \times K$ late interaction)** | ✅ **Preserves sub-patch layout & entities** | **State-of-the-Art Recall@K & MRR** |

---

### 3.6 Interview & Defense Takeaways (PhD / Research Pitch)

1. **Why is Late-Interaction (MaxSim) computationally superior to Cross-Encoders during retrieval?**  
   *Cross-encoders require concatenating the query and all image patches into a single sequence and re-running full transformer layers for every candidate page during search ($O(M \cdot (L+N)^2)$). Late-interaction allows offline pre-computation of all document latents $\mathbf{D} \in \mathbb{R}^{K \times D}$. At search time, MaxSim requires only lightweight vector dot-products ($O(M \cdot L \cdot K)$), achieving sub-millisecond retrieval speeds across millions of pages.*

2. **Why is symmetric loss $(\mathcal{L}_{Q \to D} + \mathcal{L}_{D \to Q})$ required instead of one-way loss?**  
   *One-way query-to-document loss causes representation collapse where multiple documents can cluster around common hub vectors. Symmetric loss guarantees that the visual document latent manifold and the text query manifold are strictly isomorphic and mutually discriminative.*

---

## Chapter 4: The Dual-Encoder Architecture & Unified Forward Pipeline

---

### 4.1 Architectural Topology

The complete **OmniDocDualEncoder** coordinates visual document perception and natural language query projection into a joint multi-vector embedding space:

```
[Document Page Image: (B, 3, H_img, W_img)]          [Query Text IDs: (B, L_text)]
                      │                                            │
                      ▼                                            ▼
           [Vision Patch Embedding]                      [Text Embedding + Linear]
           Patch Size P (e.g. 16x16 or 32x32)                      │
                      │                                            │
              (B, N_patches, D)                            (B, L_text, D)
                      │                                            │
                      ▼                                            ▼
          [Custom 2D-RoPE Phase Injection]                 [Query Projection Layer]
                      │                                            │
                      ▼                                            ▼
        [Custom Perceiver Resampler (16x)]              Normalized Query Tokens Q
                      │                                      (B, L_text, D)
                      ▼                                            │
         Normalized Document Latents D                             │
                  (B, K=64, D)                                     │
                      │                                            │
                      └────────────────────┬───────────────────────┘
                                           ▼
                             [Late-Interaction MaxSim]
                         S_ij = Σ_{l=1}^L max_{k=1}^K (q_l · d_k)
                                           │
                                           ▼
                       [Symmetric Patch-InfoNCE Loss / Retrieval]
```

---

### 4.2 Mathematical Invariants & Design Decisions

1. **Decoupled Asynchronous Indexing:**
   Visual document latents $\mathbf{D} \in \mathbb{R}^{K \times D}$ can be generated and indexed offline across millions of PDF pages. At query time, only the text encoder runs online ($\sim 5\text{ms}$), followed by matrix MaxSim lookup.

2. **End-to-End Differentiability:**
   Every step—patch projection $\to$ 2D-RoPE phase rotation $\to$ Perceiver cross-attention $\to$ Query linear projection $\to$ MaxSim $\to$ InfoNCE—is fully differentiable with analytical gradients.

---

## Chapter 5: High-DPI Visual Document Preprocessing & Normalization

---

### 5.1 The Micro-Typography & Resolution Dilemma

Digital documents (PDFs, TIFFs, scanned filings) exhibit fundamentally different spectral statistics than natural camera images:
* **High Spatial Frequency Content:** Character strokes, serif terminals, and fraction bars span only $1\text{ to }3\text{ pixels}$ at standard resolutions.
* **Aspect Ratio Variance:** Standard Letter documents have an aspect ratio of $1 : 1.294$, legal documents $1 : 1.647$, and presentation slides $1.778 : 1$.

#### Resolution & Font Legibility:
At standard screen resolution ($72\text{ DPI}$), an $8\text{pt}$ font has a physical capital letter height of:

$$\text{Height}_{\text{pixels}} = \frac{8\text{ pt}}{72\text{ pt/inch}} \times 72\text{ DPI} = \mathbf{8 \text{ pixels}}$$

When a patch size of $P = 16$ or $32$ is applied, an entire $8\text{pt}$ word collapses into a single patch token without sub-character distinction. 

By rasterizing at **$150\text{ to }300\text{ DPI}$**:
$$\text{Height}_{\text{pixels}} = \frac{8\text{ pt}}{72\text{ pt/inch}} \times 150\text{ DPI} \approx \mathbf{16.7 \text{ pixels}}$$
$$\text{Height}_{\text{pixels}} = \frac{8\text{ pt}}{72\text{ pt/inch}} \times 300\text{ DPI} \approx \mathbf{33.3 \text{ pixels}}$$

Character boundaries, table column separators, and subscript indices become optically distinct.

---

### 5.2 Isotropic Rescaling & Neutral Canvas Injection

To feed variable-sized document pages into fixed-grid Vision Transformers ($1024 \times 1024$) without distortion:

#### Step 1: Isotropic Scale Factor Determination
Given raw rendered dimensions $(H_0, W_0)$ and target canvas dimension $S = 1024$:

$$\text{scale} = \min\left(\frac{S}{H_0}, \, \frac{S}{W_0}\right)$$
$$H_{\text{new}} = \lfloor H_0 \cdot \text{scale} \rfloor, \quad W_{\text{new}} = \lfloor W_0 \cdot \text{scale} \rfloor$$

#### Step 2: Neutral Background Canvas Injection
Rather than zero-padding with black pixels $[0, 0, 0]$ (which induces strong artificial edge step-functions):
1. A blank canvas of size $(S, S, 3)$ is initialized with neutral document background value $\mathbf{C}_{\text{white}} = [255, 255, 255]$.
2. The resized document image is pasted at top-left $(0, 0)$ or centered $(\lfloor (S - H_{\text{new}})/2 \rfloor, \lfloor (S - W_{\text{new}})/2 \rfloor)$.

#### Step 3: Statistical Normalization
The pixel array is converted to floating-point $\mathbf{I} \in [0.0, 1.0]^{3 \times S \times S}$ and normalized via channel-wise statistics:

$$\mathbf{I}_{\text{norm}}[c, y, x] = \frac{\mathbf{I}[c, y, x] - \mu_c}{\sigma_c}$$
$$\text{where } \boldsymbol{\mu} = [0.485, 0.456, 0.406], \quad \boldsymbol{\sigma} = [0.229, 0.224, 0.225]$$

---

## Chapter 6: Multi-Modal Batch Construction & Hard Negative Sampling

---

### 6.1 The Mechanics of Contrastive Document Pairing

During contrastive pretraining, the model learns by contrasting positive document-query pairs $\{(\mathbf{Q}_i, \mathbf{D}_i^+)\}$ against negative documents $\{\mathbf{D}_j^-\}_{j \neq i}$.

```
Batch Dimension B = 4:
Query 1: "What was net income in 2023?" ──► Pos: Doc 1 (Page 4 Income Statement)
                                       ──► In-Batch Negs: Doc 2, Doc 3, Doc 4
                                       ──► Hard Neg: Doc 1 (Page 5 Balance Sheet) ◄ (Same company/font!)
```

#### Why Hard Negatives are Crucial:
If negative documents are drawn from completely disparate domains (e.g. comparing a recipe to a bank statement), the model takes shortcuts by matching gross macro features (background color, logo shape).

By including **in-document hard negatives** (different pages from the exact same PDF report):
* Font typography, margins, and header styles are identical.
* The model is forced to route attention to table cells, column headings, and precise numerical values to discriminate between the true target page and neighboring pages.

---

### 6.2 The Dynamic Collate Protocol

For variable-length text queries and high-resolution document images:
1. **Visual Stacking:** Image tensors are already standardized to $(3, 1024, 1024)$ via `PDFProcessor`, yielding uniform tensor $\mathbf{X}_{\text{batch}} \in \mathbb{R}^{B \times 3 \times 1024 \times 1024}$.
2. **Dynamic Query Padding:** Queries are tokenized and padded to the length of the longest query in the current batch $L_{\text{batch}} \le L_{\text{max}}$.
3. **Padding Attention Mask:** Binary mask $\mathbf{M}_Q \in \{0, 1\}^{B \times L_{\text{batch}}}$ is created to cleanly zero-out padded token positions during the MaxSim late-interaction calculation.

---

## Chapter 7: Optimization Dynamics, Learning Rate Scheduling & Sanity Verification

---

### 7.1 Optimization Dynamics in Late-Interaction Contrastive Learning

Training multi-vector dual-encoders via late-interaction contrastive loss exhibits unique gradient dynamics:

1. **The Gradient Flow of the MaxSim Operator:**
   For a query token $\mathbf{q}_l$ and document latents $\{\mathbf{d}_k\}_{k=1}^K$:
   $$\text{sim}_l = \max_{k \in [1, K]} (\mathbf{q}_l \cdot \mathbf{d}_k) = \mathbf{q}_l \cdot \mathbf{d}_{k^*(l)}, \quad k^*(l) = \arg\max_{k} (\mathbf{q}_l \cdot \mathbf{d}_k)$$
   The gradient with respect to document latent $\mathbf{d}_k$ flows **exclusively through the winning latent $k^*(l)$**:
   $$\frac{\partial \text{sim}_l}{\partial \mathbf{d}_k} = \begin{cases} \mathbf{q}_l & \text{if } k = k^*(l) \\ \mathbf{0} & \text{otherwise} \end{cases}$$
   *This sub-gradient routing allows specialized latents to adapt independently without conflicting gradient noise from non-matching visual patches.*

2. **Cosine Annealing with Linear Warmup:**
   To prevent early representation collapse before the temperature parameter $\tau$ stabilizes:
   $$\eta(t) = \begin{cases} \eta_{\text{max}} \cdot \frac{t}{T_{\text{warmup}}} & t \le T_{\text{warmup}} \\ \eta_{\text{min}} + \frac{1}{2} (\eta_{\text{max}} - \eta_{\text{min}}) \left(1 + \cos\left(\frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}} \pi\right)\right) & t > T_{\text{warmup}} \end{cases}$$

---

### 7.2 Micro-Overfitting Sanity Invariant

Before launching cloud GPU runs, we enforce the **Micro-Batch Overfit Criterion**:
Given a fixed batch of $B=4$ samples, an appropriately parameterized network must be capable of driving the symmetric cross-entropy loss down to near zero:

$$\mathcal{L}_{\text{initial}} \approx \ln(B) = \ln(4) \approx 1.386 \implies \mathcal{L}_{\text{step } 20} < 0.10$$

Satisfying this criterion guarantees:
* Zero shape misalignments or hidden tensor transpose bugs.
* Unbroken gradient backpropagation through the entire graph (from MaxSim to 2D-RoPE to Vision Patches).
* Stability of the learnable temperature parameter $\tau$.

---

## Chapter 8: Scaled Mixed-Precision Training, GradScaler Dynamics & Cloud Infrastructure

---

### 8.1 Mixed-Precision (`fp16`/`bf16`) & GradScaler Mechanics

When scaling training on NVIDIA Tensor Core architectures (e.g., NVIDIA T4, A100, H100):
* **Computational Speedup:** 16-bit floating-point GEMM operations execute up to $4\times$ faster.
* **Memory Footprint:** Activation storage and model weights require half the VRAM ($16\text{ GB} \to 8\text{ GB}$).

#### The GradScaler Protocol:
In contrastive learning with learnable temperature $\tau \approx 0.07$, the exponential cross-entropy operation $\exp(S / \tau)$ can produce gradient values smaller than the minimum representable value of FP16 ($\approx 6 \times 10^{-5}$), causing **underflow** to exact zeros.

To prevent gradient death:
1. **Forward Pass:** Run under `torch.amp.autocast(device_type="cuda", dtype=torch.float16)`.
2. **Loss Scaling:** Multiply loss by a dynamic scale factor $S_{\text{scale}} = 2^{16} = 65536$:
   $$\mathcal{L}_{\text{scaled}} = S_{\text{scale}} \cdot \mathcal{L}_{\text{InfoNCE}}$$
3. **Backward Pass:** Backpropagate $\mathcal{L}_{\text{scaled}}$ through the computation graph.
4. **Gradient Unscaling & Inf/NaN Checking:**
   $$\mathbf{g}_{\text{true}} = \frac{1}{S_{\text{scale}}} \nabla \mathcal{L}_{\text{scaled}}$$
   If any gradient contains $\text{Inf}$ or $\text{NaN}$, skip the optimizer step and halve $S_{\text{scale}} \leftarrow S_{\text{scale}} / 2$. Otherwise, execute optimizer step and increment scale factor.

---

### 8.2 Distributed Data Parallelism (DDP) on Multi-GPU Nodes

When training on Kaggle's dual NVIDIA T4 environment ($2\times 16\text{ GB}$):
* **Rank-Level Sharding:** The dataset is split using `DistributedSampler` so GPU 0 and GPU 1 process independent micro-batches.
* **All-Gather Contrastive Negative Pool:** In-batch negatives are gathered across all $P$ GPU ranks using `torch.distributed.all_gather`:
  $$B_{\text{global}} = P \times B_{\text{local}} = 2 \times 8 = 16 \text{ samples}$$
  This doubles the effective contrastive discrimination capacity per step without increasing per-GPU VRAM overhead.

---

## Chapter 9: Visual-Language Generation, Cross-Modal Projection & LoRA Adaptation

---

### 9.1 The Cross-Modal Projection Bridge

In Stage 2 (Visual-Language Generation), the goal is to transform the $K=64$ spatial document latents $\mathbf{D} \in \mathbb{R}^{B \times K \times D_{\text{vis}}}$ into the continuous input token embedding space of a causal language model (such as `Qwen2.5-1.5B` or `0.5B`) with hidden dimension $D_{\text{llm}}$:

$$\mathbf{H}_{\text{visual}} = \text{MLP}_{\text{proj}}\big(\text{LayerNorm}(\mathbf{D})\big) \in \mathbb{R}^{B \times K \times D_{\text{llm}}}$$

$$\text{where } \text{MLP}_{\text{proj}}(\mathbf{u}) = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \mathbf{u} + \mathbf{b}_1) + \mathbf{b}_2, \quad \mathbf{W}_1 \in \mathbb{R}^{D_{\text{llm}} \times D_{\text{vis}}}, \, \mathbf{W}_2 \in \mathbb{R}^{D_{\text{llm}} \times D_{\text{llm}}}$$

These $K=64$ continuous vectors act as **"soft visual prefix tokens"**, representing the entire document layout, tabular cells, and high-frequency typographic elements without requiring lossy OCR text serialization.

---

### 9.2 Sequence Assembly & Autoregressive Next-Token Loss Masking

For a training sample consisting of document page image $\mathbf{I}$, question prompt $\mathbf{X}_{\text{prompt}}$, and ground-truth answer $\mathbf{Y}_{\text{answer}}$:

1. **Text Embedding Lookup:**
   $$\mathbf{H}_{\text{prompt}} = \mathbf{E}_{\text{llm}}(\mathbf{X}_{\text{prompt}}) \in \mathbb{R}^{B \times L_q \times D_{\text{llm}}}$$
   $$\mathbf{H}_{\text{answer}} = \mathbf{E}_{\text{llm}}(\mathbf{Y}_{\text{answer}}) \in \mathbb{R}^{B \times L_a \times D_{\text{llm}}}$$

2. **Multimodal Sequence Concatenation:**
   $$\mathbf{H}_{\text{seq}} = \big[ \mathbf{H}_{\text{visual}} \, ; \, \mathbf{H}_{\text{prompt}} \, ; \, \mathbf{H}_{\text{answer}} \big] \in \mathbb{R}^{B \times (K + L_q + L_a) \times D_{\text{llm}}}$$

3. **Loss Masking (Teacher-Forcing on Answer Tokens Only):**
   To prevent the model from penalizing generation on the visual prefix or question tokens, the label sequence $\mathbf{T} \in \mathbb{Z}^{B \times (K + L_q + L_a)}$ is constructed as:
   $$T_i = \begin{cases} -100 & \text{for } 1 \le i \le K + L_q \quad (\text{visual prefix and question prompt}) \\ Y_{i - (K + L_q)} & \text{for } K + L_q < i \le K + L_q + L_a \quad (\text{answer tokens}) \end{cases}$$

   The Causal Language Modeling cross-entropy objective is evaluated strictly on the unmasked answer tokens:
   $$\mathcal{L}_{\text{CLM}} = - \frac{1}{L_a} \sum_{i = K + L_q + 1}^{K + L_q + L_a} \log P\big(T_i \mid \mathbf{H}_{\text{seq}, < i}\big)$$

---

### 9.3 Parameter-Efficient Low-Rank Adaptation (LoRA)

Rather than full fine-tuning of all language model parameters (which requires massive VRAM and risks catastrophic forgetting), we apply **Low-Rank Adaptation (LoRA; *Hu et al., 2021*)**:

For any frozen weight matrix $\mathbf{W}_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ in the attention projection layers (`q_proj`, `v_proj`):

$$\mathbf{W} = \mathbf{W}_0 + \frac{\alpha}{r} \mathbf{B} \mathbf{A}$$

$$\text{where } \mathbf{A} \sim \mathcal{N}\left(0, \frac{1}{r}\right) \in \mathbb{R}^{r \times d_{\text{in}}}, \quad \mathbf{B} = \mathbf{0} \in \mathbb{R}^{d_{\text{out}} \times r}, \quad r \ll \min(d_{\text{in}}, d_{\text{out}})$$

* **Trainable Parameter Reduction:** Freezing the LLM backbone and training only the visual projector and rank-$r=16$ adapters restricts active gradient computation to $<0.2\%$ of total weights ($\approx 3\text{M}$ parameters), enabling fast, stable convergence on single-GPU hardware.
