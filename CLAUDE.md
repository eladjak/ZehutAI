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
- `python -m pytest --cov=. --cov=YehoshuaSimilarityComparisons --cov-report=term-missing --cov-fail-under=70` - Coverage
- `python -m mypy .` - Type checking
- `ruff check .` - Linting
- `ruff format .` - Format code
- `pip install -r requirements.txt` - Install dependencies
- `pip install -e ".[dev]"` - Install with dev dependencies
- `pre-commit install` - Activate pre-commit hooks
- `pre-commit run --all-files` - Run all pre-commit hooks manually
- `python cli.py compare "sent1" "sent2"` - CLI: compare two sentences
- `python cli.py similarity --method tfidf --data "t1" "t2" --query "q"` - CLI: similarity
- `python cli.py rag --query "query" --top-k 5` - CLI: RAG search
- `python cli.py benchmark --format table` - CLI: run Hebrew benchmark

## Project Structure
```
ZehutAI/
  embeddings_comparison.py          # Canonical sentence similarity (single source of truth)
  cli.py                            # CLI interface (argparse): compare, similarity, rag, benchmark
  pyproject.toml                    # Modern Python packaging config + coverage + ruff
  .pre-commit-config.yaml           # Shareable pre-commit hooks (ruff, detect-secrets)
  .github/workflows/ci.yml          # CI: pytest+coverage, ruff lint, secret scan
  .omc/plans/src-layout-plan.md     # Migration plan for src/ layout (not yet executed)
  tests/
    conftest.py                     # Shared fixtures (mock models, sample data)
    test_embeddings_comparison.py   # Tests for compare_sentences() (10 tests)
    test_similarity.py              # Tests for Similarity class + model caching (37 tests)
    test_rag.py                     # Tests for RAG pipeline + chunking + Hebrew (62 tests)
    test_evaluation.py              # Tests for IR evaluation metrics (50 tests)
    test_cli.py                     # Tests for CLI interface (25 tests)
    test_hebrew_benchmark.py        # Hebrew NLP benchmarks, 10 sentence pairs (17 tests)
    test_hebrew_benchmark_module.py # Tests for benchmark module + corpus (33 tests)
    test_hebrew_model_benchmark.py  # Tests for model benchmark harness (57 tests)
  YehoshuaSimilarityComparisons/
    embeddings_comparison.py        # Re-exports from root (no duplicate code)
    sim.py                          # Similarity class: NLTK/Doc2Vec, TF-IDF, BERT, RoBERTa
    rag.py                          # RAG pipeline: preprocessing, chunking, search, RRF, e2e
    evaluation.py                   # IR metrics: precision@k, recall@k, AP, MRR, NDCG@k
    hebrew_benchmark.py             # Hebrew benchmark: 20 pairs, run_benchmark, report
    plotting.py                     # Model runner (BERT similarity)
    main.py                         # Entry point for similarity comparisons
    ragtest.py                      # Demo script for DictaLM 2.0 Hebrew model
    simple_rag.py                   # Hebrew RAG with DictaLM/mt5-xl-heq and political docs
```

## Key Files
- `embeddings_comparison.py` - Core sentence comparison utility (canonical location)
- `YehoshuaSimilarityComparisons/sim.py` - Multi-method similarity class
- `YehoshuaSimilarityComparisons/rag.py` - RAG fusion pipeline
- `YehoshuaSimilarityComparisons/evaluation.py` - IR evaluation metrics
- `YehoshuaSimilarityComparisons/hebrew_benchmark.py` - Hebrew benchmark suite
- `cli.py` - Command-line interface

## Key Functions

### sim.py
- `_get_or_load_model(tokenizer_cls, model_cls, model_weights)` - Module-level cache for tokenizer/model pairs
- `Similarity.methodBert(data, query)` - BERT cosine similarity (cached model)
- `Similarity.methodRoBERTa(data, query)` - RoBERTa cosine similarity (cached model)
- `Similarity.methodNNEmbeddings(tokenizer, model, model_weights, data, query)` - Generic NN embedding similarity
- `Similarity.methodScikitlearn(data, query)` - TF-IDF cosine similarity (no model download)
- `Similarity.methodNLTK(data, query)` - Doc2Vec similarity

### rag.py
- `preprocess_text(text)` - Normalize whitespace, strip edges
- `preprocess_hebrew(text, normalize_finals=False)` - Remove niqqud, optionally normalize final letters
- `_get_or_load_rag_model(model_name)` - Cache DictaLM/tokenizer pairs
- `chunk_documents(documents, chunk_size=200, overlap=50)` - Split into overlapping chunks
- `generate_queries(original_query)` - Expand query via DictaLM 2.0 (cached model)
- `vector_search(query, all_documents, top_k=None)` - Cosine similarity search
- `reciprocal_rank_fusion(search_results_dict, k=60)` - Fuse ranked lists (SIGIR 2009)
- `rag_pipeline(query, documents, ...)` - End-to-end: chunk -> search -> RRF -> output

### evaluation.py
- `precision_at_k(retrieved, relevant, k)` - Precision at k
- `recall_at_k(retrieved, relevant, k)` - Recall at k
- `average_precision(retrieved, relevant)` - Average precision
- `mean_reciprocal_rank(queries_results)` - MRR across queries
- `ndcg_at_k(retrieved, relevant, k)` - Normalized DCG
- `evaluate_retrieval(retrieved, relevant, k_values)` - All metrics at once

### hebrew_benchmark.py
- `HEBREW_SENTENCE_PAIRS` - 20 Hebrew pairs (7 high / 7 medium / 6 low similarity)
- `run_benchmark(model_name=None)` - Run benchmark on all pairs
- `format_benchmark_report(results)` - Pretty-print results

### embeddings_comparison.py
- `compare_sentences(sentences)` - Compare exactly 2 sentences using multilingual mpnet

## Testing
- 288 tests across 8 test files, all passing (~5s)
- 5 additional `@pytest.mark.slow` integration stubs (skipped by default)
- Coverage threshold: 70% (enforced in CI)
- Tests live in `tests/` directory
- Use `@pytest.mark.slow` for tests that download models
- `conftest.py` mocks torch/transformers/gensim/nltk at sys.modules level
- Run `python -m pytest` for fast iteration (slow auto-skipped)
