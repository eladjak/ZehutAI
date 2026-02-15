"""Re-export from the canonical embeddings_comparison module.

This avoids maintaining duplicate code. The canonical version lives at
the project root: ``embeddings_comparison.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so the canonical module is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Re-export public API
from embeddings_comparison import (  # noqa: E402
    MODEL_NAME,
    compare_sentences,
)

__all__ = ["compare_sentences", "MODEL_NAME"]
