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
- Docstrings on all public functions
- Keep model loading separate from comparison logic
- Support both CUDA and CPU execution

## Commands
- `python -m pytest` - Run tests
- `python -m mypy .` - Type checking
- `ruff check .` - Linting
- `pip install -r requirements.txt` - Install dependencies

## Key Files
- `embeddings_comparison.py` - Core sentence comparison utility
- `YehoshuaSimilarityComparisons/sim.py` - Multi-method similarity class
- `YehoshuaSimilarityComparisons/rag.py` - RAG fusion pipeline

## Known Issues
- Model is re-instantiated on every call to compare_sentences() (should be cached)
- sim.py has hardcoded test data instead of accepting parameters
- rag.py has a bug: `doc` variable used but `docc` is defined in reciprocal_rank_fusion()
- rag.py generate_queries() passes undefined `input_ids` to model.generate()
- sim.py removeModel() has a bug: compares model_weights to itself
- No tests exist yet
- No type hints on most functions
