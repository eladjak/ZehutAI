# ZehutAI - Progress

## Status: Active (Research/Prototype)
## Last Updated: 2026-02-15

## Current State
Research project for Hebrew/multilingual text similarity and RAG pipelines.
All 3 original runtime bugs fixed. Comprehensive test suite added (60 tests, all passing).
Duplicate code consolidated. Modern Python packaging via pyproject.toml.
Input validation added to public API. Code quality significantly improved.

## What Was Done
- [x] Initial repo setup with GitHub remote
- [x] Core sentence comparison using sentence-transformers (multilingual mpnet)
- [x] Similarity class comparing NLTK/Doc2Vec, TF-IDF, BERT, RoBERTa
- [x] RAG fusion pipeline with query expansion + vector search + RRF
- [x] Hebrew LLM experiments with DictaLM 2.0 and mt5-xl-heq
- [x] Project structure review and documentation (2026-02-13)
- [x] Created .gitignore for Python
- [x] Created requirements.txt with all dependencies
- [x] Updated README.md with full setup instructions
- [x] Updated CLAUDE.md with project-specific dev instructions
- [x] SECURITY FIX: Removed leaked OpenAI API key from rag.py (2026-02-13)
- [x] Fixed all 3 runtime bugs (2026-02-14)
- [x] Added type hints and docstrings to all functions (2026-02-14)
- [x] Cached model instances in embeddings_comparison.py (2026-02-14)
- [x] Removed unused tensorflow import from sim.py (2026-02-14)
- [x] Removed module-level side effects from sim.py (2026-02-14)
- [x] Fixed wildcard import in main.py (2026-02-14)
- [x] Cleaned up debug print statements from simple_rag.py (2026-02-14)
- [x] Added __init__.py to make YehoshuaSimilarityComparisons a package (2026-02-14)
- [x] Created .env.example with documented environment variables (2026-02-14)
- [x] Added python-dotenv to requirements.txt (2026-02-14)
- [x] Fixed .gitignore duplicate .env entry (2026-02-14)
- [x] Proper device detection pattern (module-level) (2026-02-14)
- [x] **Created pyproject.toml** with modern Python packaging config (2026-02-15)
- [x] **Added comprehensive test suite** - 60 tests across 3 test files (2026-02-15)
- [x] **Fixed cosine similarity bug in sim.py** - methodBert/methodNNEmbeddings used `.T` inconsistently (2026-02-15)
- [x] **Consolidated duplicate embeddings_comparison.py** - subpackage now re-exports from root (2026-02-15)
- [x] **Added input validation** to compare_sentences() - raises ValueError for wrong number of sentences (2026-02-15)
- [x] **Removed unused import** in main.py - `runModels` from plotting was imported but never used (2026-02-15)
- [x] **Added main() function** to main.py entry point (2026-02-15)
- [x] **Updated CLAUDE.md** - removed stale "Known Issues" that were already fixed, added test documentation (2026-02-15)

## Bug Fixes Applied (2026-02-14)

### CRITICAL - rag.py `reciprocal_rank_fusion()` (line 52-56)
- **Problem:** Loop variable was `docc` but code referenced `doc` -> NameError at runtime
- **Fix:** Changed loop variable to `doc` to match all references

### CRITICAL - rag.py `generate_queries()` (line 30)
- **Problem:** Passed undefined `input_ids` as second argument to `model.generate()`
- **Fix:** Removed the undefined `input_ids` argument; `encoded` already contains the input

### HIGH - sim.py `removeModel()` (line 31-32)
- **Problem:** Compared `model_weights == model_weights` (self-comparison, always True)
  and also shadowed the parameter name with the unpacked tuple variable
- **Fix:** Renamed parameter to `target_weights`, compare `model_weights == target_weights`

## Bug Fixes Applied (2026-02-15)

### MEDIUM - sim.py `methodBert()` and `methodNNEmbeddings()` cosine similarity
- **Problem:** Used `embedding1.T` in `np.dot()` and `np.linalg.norm()` while
  `methodRoBERTa()` used `embedding1` (without `.T`). For 1D arrays, `.T` is a no-op,
  but this was inconsistent and would be wrong for 2D embeddings.
- **Fix:** Changed both methods to use `embedding1` (without `.T`), consistent with RoBERTa

## Test Suite (2026-02-15)

### Structure
```
tests/
  __init__.py                     # Package marker
  conftest.py                     # Mock ML imports + shared fixtures
  test_embeddings_comparison.py   # 10 tests - compare_sentences, caching, constants
  test_similarity.py              # 28 tests - Similarity class, TF-IDF, utilities
  test_rag.py                     # 22 tests - RRF, vector_search, generate_output
```

### Key Design Decisions
- **No ML dependencies required**: All tests mock torch/transformers/sentence-transformers
  at the `sys.modules` level in conftest.py, so tests run in ~1 second without any GPU
- **TF-IDF tests use real scikit-learn**: Since sklearn is lightweight, we test the actual
  TF-IDF cosine similarity computation
- **RRF tests verify exact formula**: Tests validate the mathematical correctness of
  `1/(rank + k)` accumulation
- **Pytest markers**: `@pytest.mark.slow` available for future integration tests

### Running Tests
```bash
python -m pytest tests/ -v           # All tests (~1s)
python -m pytest tests/ -v --tb=short  # Compact output on failure
```

## Code Quality Improvements (2026-02-14)

### Security Hardening
- Verified no API keys/secrets remain anywhere in source code
- Created `.env.example` documenting required environment variables
- Added `python-dotenv` to requirements for proper env var loading
- Fixed `.gitignore` duplicate `.env` entry

### Type Hints & Documentation
- Added `from __future__ import annotations` to all files
- Added type hints to all function signatures
- Added Google-style docstrings to all functions
- Added module-level docstrings to all files

### Model Caching
- Both `embeddings_comparison.py` files now cache the SentenceTransformer model
  at module level, avoiding re-instantiation on every `compare_sentences()` call

### Code Cleanup
- Removed unused `tensorflow` import from sim.py
- Removed module-level side effects from sim.py (was auto-running on import)
- Replaced wildcard import `from plotting import *` with explicit import
- Removed debug print statements (`print('&&&&&&&&&&&&')`, `print('******************')`)
- Removed commented-out code blocks
- Removed unused imports (BertTokenizerFast, BertModel, AutoModelForCausalLM from simple_rag.py)
- Extracted hardcoded sample data into module-level constants (DEFAULT_DATA, DEFAULT_QUERY)
- All methods now accept optional `data` and `query` parameters instead of hardcoding

### Device Handling
- Replaced fragmented device detection (torch.device() without assignment) with
  clean module-level `device: str = "cuda" if torch.cuda.is_available() else "cpu"`

## Improvements (2026-02-15)

### Code Consolidation
- Duplicate `embeddings_comparison.py` eliminated - subpackage version now
  re-exports from the canonical root module (single source of truth)
- Removed unused `runModels` import from `main.py`
- Added proper `main()` function to entry point

### Input Validation
- `compare_sentences()` now raises `ValueError` if given != 2 sentences
- Clear error message: "Expected exactly 2 sentences, got N"

### Modern Packaging
- Added `pyproject.toml` with:
  - Build system (setuptools)
  - Project metadata (name, version, description, classifiers)
  - Optional `[dev]` dependency group for test/lint tools
  - pytest configuration (testpaths, pythonpath, markers, filter warnings)
  - mypy configuration (strict type checking)
  - ruff configuration (linting rules, naming exceptions for camelCase methods)

## Next Steps
1. **URGENT: Rotate the leaked OpenAI API key** (still in git history!)
2. Consider using BFG Repo Cleaner to remove the key from git history
3. Add model caching to sim.py methodBert/methodRoBERTa (currently re-download on every call)
4. Consider restructuring into a proper `src/` layout
5. Set up CI/CD (GitHub Actions for pytest + ruff + mypy)
6. Add integration tests with `@pytest.mark.slow` for real model evaluation
7. Add Hebrew-specific test data and evaluation benchmarks
8. Improve RAG pipeline: add chunk overlap, configurable top-k, document preprocessing

## Key Decisions Made
- Using sentence-transformers/paraphrase-multilingual-mpnet-base-v2 as primary model (good Hebrew support)
- Exploring DictaLM 2.0 for Hebrew text generation
- RAG approach: query expansion -> vector search -> reciprocal rank fusion
- Module-level model caching over class-based singleton (simpler for research code)
- Google-style docstrings for consistency
- Mock ML imports at sys.modules level for fast tests (no GPU/model downloads needed)
- Single canonical embeddings_comparison.py at project root (subpackage re-exports)

## Files Modified (2026-02-15)
- `embeddings_comparison.py` - Added input validation, FutureWarning filter, enhanced docstrings
- `YehoshuaSimilarityComparisons/embeddings_comparison.py` - Replaced duplicate with re-export
- `YehoshuaSimilarityComparisons/sim.py` - Fixed cosine similarity `.T` inconsistency in methodBert/methodNNEmbeddings
- `YehoshuaSimilarityComparisons/main.py` - Removed unused import, added main() function
- `CLAUDE.md` - Updated with current state, removed stale known issues, added test docs
- `pyproject.toml` - Created (new file)
- `tests/__init__.py` - Created (new file)
- `tests/conftest.py` - Created (new file) - Mock ML imports + shared fixtures
- `tests/test_embeddings_comparison.py` - Created (new file) - 10 tests
- `tests/test_similarity.py` - Created (new file) - 28 tests
- `tests/test_rag.py` - Created (new file) - 22 tests
- `PROGRESS.md` - Updated with all changes

## Notes for Next Session
- The leaked API key is still in git history - consider using `git filter-branch` or BFG Repo Cleaner
- sim.py methodBert/methodRoBERTa still create models on every call - could add caching
- Consider adding GitHub Actions CI with the test suite
- Hebrew-specific evaluation data would strengthen the project significantly
