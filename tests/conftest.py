"""Shared fixtures for ZehutAI tests.

Provides mock objects and sample data so that unit tests can run
without downloading large ML models or having torch/transformers installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

# Ensure project root is on sys.path
_root = str(Path(__file__).resolve().parent.parent)
_sub = str(Path(__file__).resolve().parent.parent / "YehoshuaSimilarityComparisons")
for p in (_root, _sub):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Mock heavy ML imports so tests can run without torch/transformers/etc.
# These mocks are installed BEFORE any project module is imported.
# ---------------------------------------------------------------------------

def _install_mock_ml_modules() -> dict[str, MagicMock]:
    """Install mock modules for torch, transformers, sentence_transformers, etc."""
    mocks: dict[str, MagicMock] = {}

    modules_to_mock = [
        "torch",
        "torch.cuda",
        "transformers",
        "sentence_transformers",
        "gensim",
        "gensim.models",
        "gensim.models.doc2vec",
        "nltk",
        "nltk.tokenize",
    ]

    for mod_name in modules_to_mock:
        if mod_name not in sys.modules:
            mock = MagicMock()
            sys.modules[mod_name] = mock  # type: ignore[assignment]
            mocks[mod_name] = mock

    # Make torch.cuda.is_available() return False (CPU mode)
    torch_mock = sys.modules["torch"]
    torch_mock.cuda.is_available.return_value = False  # type: ignore[union-attr]
    torch_mock.bfloat16 = "bfloat16"

    # Create a real class for Tensor so that scipy's issubclass() checks work
    class _FakeTensor:
        pass

    torch_mock.Tensor = _FakeTensor  # type: ignore[union-attr]

    # Make sentence_transformers.SentenceTransformer a proper mock class
    st_mock = sys.modules["sentence_transformers"]
    st_mock.SentenceTransformer = MagicMock  # type: ignore[union-attr]

    # Make transformers submodules accessible
    tf_mock = sys.modules["transformers"]
    tf_mock.BertTokenizer = MagicMock()  # type: ignore[union-attr]
    tf_mock.RobertaTokenizer = MagicMock()  # type: ignore[union-attr]
    tf_mock.BertModel = MagicMock()  # type: ignore[union-attr]
    tf_mock.RobertaModel = MagicMock()  # type: ignore[union-attr]
    tf_mock.AutoTokenizer = MagicMock()  # type: ignore[union-attr]
    tf_mock.AutoModelForCausalLM = MagicMock()  # type: ignore[union-attr]
    tf_mock.AutoModelForSeq2SeqLM = MagicMock()  # type: ignore[union-attr]

    # gensim doc2vec
    gensim_d2v = sys.modules["gensim.models.doc2vec"]
    gensim_d2v.Doc2Vec = MagicMock  # type: ignore[union-attr]
    gensim_d2v.TaggedDocument = MagicMock  # type: ignore[union-attr]

    # nltk tokenize
    nltk_tok = sys.modules["nltk.tokenize"]
    nltk_tok.word_tokenize = MagicMock(  # type: ignore[union-attr]
        side_effect=lambda text: text.lower().split()
    )

    return mocks


# Install mocks at import time (before test collection imports project modules)
_ml_mocks = _install_mock_ml_modules()


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_sentences() -> list[str]:
    """Two English sentences for similarity comparison."""
    return [
        "The cat sat on the mat",
        "A kitten was sitting on a rug",
    ]


@pytest.fixture
def sample_hebrew_sentences() -> list[str]:
    """Two Hebrew sentences for similarity comparison."""
    return [
        "החתול ישב על השטיח",
        "חתלתול ישב על המזרן",
    ]


@pytest.fixture
def sample_documents() -> dict[str, str]:
    """Small set of documents for RAG tests."""
    return {
        "doc1": "Climate change and economic impact.",
        "doc2": "Public health concerns due to climate change.",
        "doc3": "Technological solutions to climate change.",
        "doc4": "The history of ancient Rome.",
        "doc5": "Introduction to machine learning.",
    }


@pytest.fixture
def sample_data() -> list[str]:
    """Default corpus for Similarity class tests."""
    return [
        "The movie is awesome. It was a good thriller",
        "We are learning NLP through GeeksforGeeks",
        "The baby learned to walk in the 5th month itself",
    ]


@pytest.fixture
def sample_query() -> str:
    """Default query string for Similarity class tests."""
    return "The baby was laughing and playing"


# ---------------------------------------------------------------------------
# Mock model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_sentence_transformer(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch the cached model in embeddings_comparison so no real model is used.

    The mock encode() returns deterministic normalized random embeddings.
    The mock similarity() computes actual cosine similarity on those embeddings.
    """
    import embeddings_comparison as ec

    mock_model = MagicMock()

    def _encode(sentences: list[str], **kwargs: Any) -> np.ndarray:
        rng = np.random.default_rng(hash(tuple(sentences)) % (2**31))
        n = len(sentences)
        embs = rng.standard_normal((n, 768)).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / norms

    mock_model.encode = _encode

    def _similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        return a_norm @ b_norm.T

    mock_model.similarity = _similarity

    monkeypatch.setattr(ec, "_model", mock_model)
    return mock_model
