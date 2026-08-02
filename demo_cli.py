#!/usr/bin/env python3
"""
OmniDoc-RAG: Command-Line Interactive Demonstration
Run with: python demo_cli.py
"""

import os
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from data.pdf_processor import PDFProcessor
from models.omni_encoder import OmniDocDualEncoder

console = Console()

def run_demo():
    console.print(Panel.fit(
        "[bold blue]⚡ OmniDoc-RAG: OCR-Free Visual Retrieval & Reasoning Engine[/bold blue]\n"
        "[dim]2D-RoPE + Perceiver Resampler (16x) + MaxSim Late-Interaction[/dim]",
        border_style="blue"
    ))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[bold green]✓[/bold green] Using device: [bold cyan]{device.upper()}[/bold cyan]")
    
    # 1. Initialize Dual-Encoder
    console.print("[yellow]Initializing OmniDocDualEncoder...[/yellow]")
    encoder = OmniDocDualEncoder(
        embed_dim=768,
        patch_size=32,
        num_latents=64,
        vocab_size=30522
    )
    
    ckpt = "checkpoints/omnidoc_stage1_best.pt"
    if os.path.exists(ckpt):
        state_dict = torch.load(ckpt, map_location=device, weights_only=True)
        encoder.load_state_dict(state_dict, strict=False)
        console.print(f"[bold green]✓[/bold green] Loaded trained checkpoint: [bold]{ckpt}[/bold]")
    else:
        console.print("[bold yellow]![/bold yellow] Using initialized weights")
        
    encoder.eval().to(device)
    
    # 2. Simulate 3 Multi-Modal Document Pages
    console.print("\n[bold]Simulating 3 Document Pages (Financial, Technical, Balance Sheet)...[/bold]")
    sample_images = torch.randn(3, 3, 1024, 1024, device=device)
    
    with torch.no_grad():
        # Encode Document Pages -> 64 latents each
        doc_latents = encoder.encode_document(sample_images) # (3, 64, 768)
        console.print(f"[bold green]✓[/bold green] Encoded 3 pages into visual latents tensor: [bold cyan]{tuple(doc_latents.shape)}[/bold cyan]")
        
        # Test Query
        query_text = "What was the annual net revenue in fiscal year 2024?"
        console.print(f"\n[bold]Query:[/bold] [italic]\"{query_text}\"[/italic]")
        
        # Tokenize Query
        token_ids = torch.tensor([[ord(c) % 30522 for c in query_text[:32]]], device=device)
        query_emb = encoder.encode_query(token_ids) # (1, L, 768)
        console.print(f"[bold green]✓[/bold green] Encoded query tokens tensor: [bold cyan]{tuple(query_emb.shape)}[/bold cyan]")
        
        # Compute MaxSim Late-Interaction across pages
        # query_emb: (1, L, D), doc_latents: (3, K, D)
        scores = []
        for p in range(3):
            d_p = doc_latents[p:p+1] # (1, K, D)
            sim_matrix = torch.matmul(query_emb, d_p.transpose(1, 2)) # (1, L, K)
            max_sim = torch.max(sim_matrix, dim=-1).values # (1, L)
            score = torch.sum(max_sim).item()
            scores.append((p+1, score))
            
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Leaderboard Table
    table = Table(title="OmniDoc-RAG Retrieval Leaderboard", border_style="blue")
    table.add_column("Rank", justify="center", style="bold")
    table.add_column("Document Page", justify="left")
    table.add_column("MaxSim Score", justify="right", style="cyan")
    table.add_column("Status", justify="center")
    
    for rank, (page_num, score) in enumerate(scores):
        is_winner = rank == 0
        table.add_row(
            f"#{rank + 1}",
            f"Document Page {page_num}",
            f"{score:.4f}",
            "[bold green]🏆 TOP RETRIEVED[/bold green]" if is_winner else "[dim]Candidate[/dim]"
        )
        
    console.print(table)
    console.print("\n[bold green]✓ Real-time MaxSim Retrieval Successful![/bold green]")
    console.print("To run the full visual browser interface, execute: [bold cyan]streamlit run app.py[/bold cyan]\n")

if __name__ == "__main__":
    run_demo()
