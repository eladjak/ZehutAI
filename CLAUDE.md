# ZehutAI - Development Instructions

## Project
Hebrew/multilingual AI text analysis and semantic similarity research project.
Uses embeddings (sentence-transformers, BERT, RoBERTa) and RAG pipelines.

## Tech Stack
- Python 3.11+
- sentence-transformers (paraphrase-multilingual-mpnet-base-v2)
- transformers (BERT, RoBERTa, DictaLM 2.0, mt5-xl-heq)
- torch (CUDA when available, CPU fallback)
- gensim (Doc2Vec), nltk, scikit-learn, scipy, numpy, matplotlib

## Principles
- Security first: NEVER hardcode API keys or secrets
- Use environment variables for all credentials
- Type hints on all functions
- Docstrings on all public functions (Google-style)
- Keep model loading separate from comparison logic
- Cache models at module level to avoid re-instantiation
- Support both CUDA and CPU execution
- Input validation on public API functions

## Commands
- `python -m pytest` - Run tests
- `python -m pytest -m "not slow"` - Run fast tests only (skip model downloads)
- `python -m pytest --cov=. --cov-report=term-missing` - Run tests with coverage
- `python -m mypy .` - Type checking
- `ruff check .` - Linting
- `ruff format .` - Format code
- `pip install -r requirements.txt` - Install dependencies
- `pip install -e ".[dev]"` - Install with dev dependencies

## Project Structure
```
ZehutAI/
  embeddings_comparison.py          # Canonical sentence similarity (single source of truth)
  pyproject.toml                    # Modern Python packaging config
  tests/
    conftest.py                     # Shared fixtures (mock models, sample data)
    test_embeddings_comparison.py   # Tests for compare_sentences()
    test_similarity.py              # Tests for Similarity class methods
    test_rag.py                     # Tests for RAG pipeline (vector_search, RRF)
  YehoshuaSimilarityComparisons/
    embeddings_comparison.py        # Re-exports from root (no duplicate code)
    sim.py                          # Similarity class: NLTK/Doc2Vec, TF-IDF, BERT, RoBERTa
    rag.py                          # RAG fusion pipeline: query expansion + vector search + RRF
    plotting.py                     # Model runner (BERT similarity)
    main.py                         # Entry point for similarity comparisons
    ragtest.py                      # Demo script for DictaLM 2.0 Hebrew model
    simple_rag.py                   # Hebrew RAG with DictaLM/mt5-xl-heq and political docs
```

## Key Files
- `embeddings_comparison.py` - Core sentence comparison utility (canonical location)
- `YehoshuaSimilarityComparisons/sim.py` - Multi-method similarity class
- `YehoshuaSimilarityComparisons/rag.py` - RAG fusion pipeline

## Testing
- Tests live in `tests/` directory
- Use `@pytest.mark.slow` for tests that download models
- Use `unittest.mock.patch` to mock heavy model loads in unit tests
- Run `python -m pytest -m "not slow"` for CI/fast iteration
