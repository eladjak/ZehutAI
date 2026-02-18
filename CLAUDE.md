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
- `python -m pytest` - Run tests (slow tests skipped by default via addopts)
- `python -m pytest -m slow --override-ini="addopts="` - Run slow integration tests only
- `python -m pytest --cov=. --cov-report=term-missing` - Run tests with coverage
- `python -m mypy .` - Type checking
- `ruff check .` - Linting
- `ruff format .` - Format code
- `pip install -r requirements.txt` - Install dependencies
- `pip install -e ".[dev]"` - Install with dev dependencies
- `pre-commit install` - Activate pre-commit hooks (.pre-commit-config.yaml)
- `pre-commit run --all-files` - Run all pre-commit hooks manually

## Project Structure
```
ZehutAI/
  embeddings_comparison.py          # Canonical sentence similarity (single source of truth)
  pyproject.toml                    # Modern Python packaging config
  .pre-commit-config.yaml           # Shareable pre-commit hooks (ruff, detect-secrets)
  .github/workflows/ci.yml          # CI pipeline: pytest, ruff, secret scan
  tests/
    conftest.py                     # Shared fixtures (mock models, sample data)
    test_embeddings_comparison.py   # Tests for compare_sentences() (10 tests)
    test_similarity.py              # Tests for Similarity class + model caching (37 tests)
    test_rag.py                     # Tests for RAG pipeline + chunking (40 tests)
    test_hebrew_benchmark.py        # Hebrew NLP benchmarks, 10 sentence pairs (17 tests)
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

## Key Functions

### sim.py
- `_get_or_load_model(tokenizer_cls, model_cls, model_weights)` - Module-level cache for tokenizer/model pairs. Loads once, returns cached on subsequent calls.
- `Similarity.methodBert(data, query)` - BERT cosine similarity (cached model)
- `Similarity.methodRoBERTa(data, query)` - RoBERTa cosine similarity (cached model)
- `Similarity.methodNNEmbeddings(tokenizer, model, model_weights, data, query)` - Generic NN embedding similarity (cached model)
- `Similarity.methodScikitlearn(data, query)` - TF-IDF cosine similarity (no model download)
- `Similarity.methodNLTK(data, query)` - Doc2Vec similarity

### rag.py
- `preprocess_text(text)` - Normalize whitespace, strip edges
- `chunk_documents(documents, chunk_size=200, overlap=50)` - Split documents into overlapping chunks with source tracking
- `generate_queries(original_query)` - Expand query via DictaLM 2.0
- `vector_search(query, all_documents, top_k=None)` - Cosine similarity search with optional top-k limit
- `reciprocal_rank_fusion(search_results_dict, k=60)` - Fuse ranked lists (k=60 is standard from SIGIR 2009 paper)
- `generate_output(reranked_results, queries)` - Format final output

### embeddings_comparison.py
- `compare_sentences(sentences)` - Compare exactly 2 sentences using multilingual mpnet (cached model)

## Testing
- 101 tests across 4 test files, all passing (~2s)
- 3 additional `@pytest.mark.slow` integration stubs (skipped by default)
- Slow tests require real model downloads (BERT/RoBERTa)
- Tests live in `tests/` directory
- Use `@pytest.mark.slow` for tests that download models
- Use `unittest.mock.patch` to mock heavy model loads in unit tests
- `conftest.py` mocks torch/transformers/gensim/nltk at sys.modules level
- Run `python -m pytest` for fast iteration (slow auto-skipped)
