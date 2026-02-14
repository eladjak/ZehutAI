"""Sentence comparison utility using multilingual sentence-transformers."""

from __future__ import annotations

import warnings

from sentence_transformers import SentenceTransformer

warnings.simplefilter(action="ignore", category=FutureWarning)

# Cache model instance at module level to avoid re-loading on every call
_model: SentenceTransformer | None = None

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def _get_model() -> SentenceTransformer:
    """Return cached SentenceTransformer model, loading it on first call."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def compare_sentences(sentences: list[str]) -> float:
    """Compare two sentences and return their cosine similarity.

    Uses sentence-transformers/paraphrase-multilingual-mpnet-base-v2 for
    multilingual embedding, supporting 50+ languages including Hebrew.

    Args:
        sentences: List of exactly 2 sentences to compare.

    Returns:
        Cosine similarity score between -1.0 and 1.0.
    """
    model = _get_model()
    embeddings = model.encode(sentences)
    similarities = model.similarity(embeddings, embeddings)
    return float(similarities[0, 1])
