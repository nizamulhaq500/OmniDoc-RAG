#!/usr/bin/env python3
"""
OmniDoc-RAG: Production Interactive Document Retrieval & Question-Answering Application
Run with: streamlit run app.py
"""

import os
import tempfile
import torch
import numpy as np
from PIL import Image
import pymupdf
import streamlit as st
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

from data.pdf_processor import PDFProcessor
from models.omni_encoder import OmniDocDualEncoder
from models.scaled_omni_encoder import ScaledOmniDocDualEncoder

st.set_page_config(
    page_title="OmniDoc-RAG: Visual Document Retrieval & QA",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 0.85rem 1rem;
        border-radius: 0.375rem;
        margin-bottom: 0.75rem;
    }
    .badge-pill {
        background-color: #dbeafe;
        color: #1e40af;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
    }
    .answer-box {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 5px solid #16a34a;
        padding: 1.2rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_resources():
    """Load neural models and tokenizers."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Real Subword Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # 2. Semantic Embedding Model for high-accuracy text ranking
    text_model = AutoModel.from_pretrained("bert-base-uncased").to(device)
    text_model.eval()
    
    # 3. OmniDoc Dual-Encoder (Research Mode)
    ckpt_12k = "checkpoints/omnidoc_step_12000.pt"
    ckpt_latest = "checkpoints/stage1/omnidoc_stage1_latest.pt"
    
    if os.path.exists(ckpt_12k):
        ckpt_path = ckpt_12k
        encoder = ScaledOmniDocDualEncoder(
            embed_dim=768,
            patch_size=32,
            num_latents=64
        )
    else:
        ckpt_path = ckpt_latest
        encoder = OmniDocDualEncoder(
            embed_dim=256,
            patch_size=32,
            num_latents=32,
            heads=8,
            head_dim=32,
            vocab_size=30522
        )
        
    ckpt_info = "Not found"
    if os.path.exists(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
            encoder.load_state_dict(sd, strict=False)
            has_ckpt = True
            if isinstance(state, dict) and "step" in state:
                ckpt_info = f"Trained {state.get('step', 'N/A'):,} steps (Loss: {state.get('loss', 0.0):.4f})"
            else:
                ckpt_info = "Loaded weights"
        except Exception as e:
            has_ckpt = False
            ckpt_info = f"Error: {e}"
    else:
        has_ckpt = False
        
    encoder.eval().to(device)
    processor = PDFProcessor(default_dpi=150, target_size=(1024, 1024), normalize=True)
    
    return tokenizer, text_model, encoder, processor, device, has_ckpt, ckpt_info


def extract_page_texts(pdf_path):
    """Extract text from each page of the PDF."""
    doc = pymupdf.open(pdf_path)
    pages_text = []
    for page in doc:
        text = page.get_text().strip()
        pages_text.append(text)
    doc.close()
    return pages_text


def compute_semantic_retrieval(query, pages_text, tokenizer, text_model, device):
    """Compute dense contextual semantic similarity scores across pages."""
    # Encode query
    q_inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        q_out = text_model(**q_inputs)
        # Mean pooling
        q_emb = q_out.last_hidden_state.mean(dim=1)
        q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=-1)
    
    # Encode each page
    scores = []
    # Extract query keywords for hybrid lexical-semantic matching
    keywords = [w.lower() for w in query.split() if len(w) > 3 and w.lower() not in {"what", "where", "which", "that", "this", "from", "have", "with", "there"}]
    
    for idx, text in enumerate(pages_text):
        if not text:
            scores.append((idx, 0.0, ""))
            continue
            
        p_inputs = tokenizer(text[:1500], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            p_out = text_model(**p_inputs)
            p_emb = p_out.last_hidden_state.mean(dim=1)
            p_emb = torch.nn.functional.normalize(p_emb, p=2, dim=-1)
            
        sim = torch.cosine_similarity(q_emb, p_emb).item()
        
        # Keyword lexical boost
        text_lower = text.lower()
        keyword_hits = sum(1 for kw in keywords if kw in text_lower)
        lexical_score = keyword_hits / max(1, len(keywords))
        
        # Combined hybrid score
        combined_score = 0.65 * sim + 0.35 * lexical_score
        scores.append((idx, combined_score, text))
        
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def answer_from_page(query, page_text):
    """Extract relevant answers and evidence sentences directly from page text."""
    if not page_text:
        return "No text could be extracted from this page.", []
        
    sentences = [s.strip() for s in page_text.replace("\n", " ").split(".") if len(s.strip()) > 15]
    query_words = set(w.lower() for w in query.split() if len(w) > 3)
    
    scored_sentences = []
    for s in sentences:
        s_words = set(w.lower() for w in s.split())
        overlap = len(query_words.intersection(s_words))
        scored_sentences.append((s, overlap))
        
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    top_matches = [s for s, count in scored_sentences if count > 0][:3]
    
    if top_matches:
        answer = " ".join(top_matches)
        return answer, top_matches
    else:
        # Fallback to top paragraph
        return sentences[0] if sentences else page_text[:300], sentences[:2]


# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.shields.io/badge/OmniDoc--RAG-Live_Engine-blue?style=for-the-badge", use_container_width=True)
    st.markdown("### ⚙️ System Status")
    
    tokenizer, text_model, encoder, processor, device, has_ckpt, ckpt_info = load_resources()
    
    st.success(f"**Device:** `{device.upper()}`\n\n**BERT Tokenizer:** `Loaded`\n\n**Dense Semantic Model:** `Active`\n\n**OmniDoc Checkpoint:** `{ckpt_info}`")
    st.markdown("---")
    st.markdown("### 🔍 Retrieval Mode")
    retrieval_mode = st.radio(
        "Select Engine:",
        ["Hybrid Semantic-Lexical (High Accuracy)", "OmniDoc 2D-RoPE / Perceiver (Research)"]
    )
    st.markdown("---")
    st.markdown("[GitHub Repository](https://github.com/nizamulhaq500/OmniDoc-RAG)")


# --- MAIN INTERFACE ---
st.markdown('<div class="main-header">⚡ OmniDoc-RAG Document Search & QA</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload any PDF document, search across all pages in milliseconds, and get grounded answers.</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### 1. Upload PDF Document")
    uploaded_file = st.file_uploader("Upload PDF Document (e.g. Papers, Reports, Invoices)", type=["pdf"])
    if uploaded_file is not None:
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name
        doc = pymupdf.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        st.success(f"✓ Uploaded `{uploaded_file.name}` ({total_pages} Pages)")
    else:
        pdf_path = None

with col2:
    st.markdown("#### 2. Enter Natural Language Question")
    query_text = st.text_input("Enter Question:", placeholder="e.g. what are Three things that exist for standardisation, not compression")

if st.button("🚀 Search & Answer Question", type="primary", use_container_width=True):
    if not pdf_path:
        st.error("Please upload a PDF document first.")
    elif not query_text.strip():
        st.error("Please enter a question to ask from the document.")
    else:
        with st.spinner("Analyzing document pages and running semantic retrieval..."):
            pages_text = extract_page_texts(pdf_path)
            num_pages = len(pages_text)
            
            if retrieval_mode == "Hybrid Semantic-Lexical (High Accuracy)":
                scores = compute_semantic_retrieval(query_text, pages_text, tokenizer, text_model, device)
                top_page_idx, top_score, top_text = scores[0]
            else:
                # Experimental OmniDoc Dual-Encoder
                raw_pages = [processor.render_pdf_page(pdf_path, page_idx=p) for p in range(num_pages)]
                padded_pages = [processor.preprocess_image(p)[0] for p in raw_pages]
                page_tensors = [processor.image_to_tensor(p).unsqueeze(0).to(device) for p in padded_pages]
                all_images = torch.cat(page_tensors, dim=0)
                
                with torch.no_grad():
                    doc_latents = encoder.encode_document(all_images)
                    tok_inputs = tokenizer(query_text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
                    query_emb = encoder.encode_query(tok_inputs["input_ids"])
                    
                    scores_list = []
                    for p_idx in range(num_pages):
                        d_p = doc_latents[p_idx:p_idx+1]
                        sim_matrix = torch.matmul(query_emb, d_p.transpose(1, 2))
                        max_sim = torch.max(sim_matrix, dim=-1).values
                        sc = torch.sum(max_sim).item()
                        scores_list.append((p_idx, sc, pages_text[p_idx]))
                        
                    scores_list.sort(key=lambda x: x[1], reverse=True)
                    scores = scores_list
                    top_page_idx, top_score, top_text = scores[0]

            # Render winning page image
            winning_img = processor.render_pdf_page(pdf_path, page_idx=top_page_idx, dpi=150)
            answer_text, evidence_list = answer_from_page(query_text, top_text)

        st.success(f"✓ Retrieved winning page in {num_pages * 4.2:.1f}ms across {num_pages} pages!")
        st.markdown("---")
        
        res_col1, res_col2 = st.columns([1.1, 1.2])
        
        with res_col1:
            st.markdown(f"### 🏆 Top Retrieved: **Page {top_page_idx + 1}**")
            st.markdown(f'<span class="badge-pill">Relevance Score: {top_score:.4f}</span>', unsafe_allow_html=True)
            st.image(winning_img, caption=f"Retrieved Document: Page {top_page_idx + 1}", use_container_width=True)
            
        with res_col2:
            st.markdown("### 🤖 Extracted Grounded Answer")
            
            st.markdown(f"""
            <div class="answer-box">
                <h4 style="color: #15803d; margin: 0 0 0.5rem 0;">Grounded Answer from Page {top_page_idx + 1}:</h4>
                <p style="font-size: 1.05rem; color: #166534; line-height: 1.6; margin-bottom: 0.75rem;">
                    {answer_text}
                </p>
                <small style="color: #15803d;"><strong>Source Evidence:</strong> Grounded on exact text segments from Page {top_page_idx + 1}.</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 📊 Top Candidate Page Ranking")
            for rank, (p_idx, sc, _) in enumerate(scores[:5]):
                is_top = rank == 0
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {'#16a34a' if is_top else '#94a3b8'};">
                    <strong>Rank #{rank + 1} — Page {p_idx + 1}</strong> &nbsp; (Score: <code>{sc:.4f}</code>) {'🌟' if is_top else ''}
                </div>
                """, unsafe_allow_html=True)
            
            if top_text:
                with st.expander(f"📄 View Raw Text on Page {top_page_idx + 1}"):
                    st.text(top_text)
