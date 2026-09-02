"""
Unit test for notebooks/kaggle_stage1_training.ipynb.
Validates JSON structure, Python AST syntax across code cells, and completeness of neural modules.
"""

import os
import json
import ast
import pytest

NOTEBOOK_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "notebooks", "kaggle_stage1_training.ipynb")


def test_kaggle_notebook_valid_json():
    """Verify notebook is valid JSON with nbformat 4."""
    assert os.path.exists(NOTEBOOK_PATH), f"Notebook missing at {NOTEBOOK_PATH}"
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb
    assert nb.get("nbformat") == 4
    assert len(nb["cells"]) >= 4


def test_kaggle_notebook_ast_syntax():
    """Verify that all Python code cells contain syntactically valid code."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    code_cells = [cell for cell in nb["cells"] if cell.get("cell_type") == "code"]
    assert len(code_cells) >= 3

    combined_code = []
    for cell in code_cells:
        lines = cell.get("source", [])
        # Filter out shell commands (e.g. !pip install) for AST parsing
        filtered_lines = [l for l in lines if not l.strip().startswith("!")]
        code_str = "".join(filtered_lines)
        # Parse individual cell
        ast.parse(code_str)
        combined_code.append(code_str)

    full_script = "\n".join(combined_code)
    # Verify presence of core architectural components
    assert "class RotaryEmbedding2D" in full_script
    assert "class PerceiverResampler" in full_script
    assert "class SymmetricPatchInfoNCELoss" in full_script
    assert "class OmniDocDualEncoder" in full_script
    assert "def evaluate_retrieval_benchmarks" in full_script
