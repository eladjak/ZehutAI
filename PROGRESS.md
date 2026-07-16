# ZehutAI - Progress

## Status: Active (Research/Prototype + Web UI v1 in production)
## Last Updated: 2026-07-16

## 2026-07-16 — Web UI v1 LIVE (for Yehoshua Dalin)
- `webapp/` — FastAPI service wrapping the existing engine (no engine rewrite):
  compare / similarity / rag / benchmark endpoints + dark Hebrew RTL SPA.
- Deployed on Contabo `/opt/zehutai`, systemd `zehutai-web.service`
  (127.0.0.1:3980), cloudflared path `hub.eladjak.com/zehutai` (token-gated:
  `?k=<token>` → HttpOnly cookie; token in `/etc/zehutai-web.env`).
- mpnet eager-load (~34s startup, service ~1.9GB RAM); BERT/RoBERTa lazy;
  DictaLM 7B query expansion deferred (CPU-only host).
- Benchmark on prod CPU: high 0.792 / medium 0.393 / low 0.020 / separation 0.772.
- Fixed pre-existing repo issues: invalid pre-commit YAML (inline python →
  `scripts/detect_secrets_custom.py`), missing `.secrets.baseline`, 12 tests
  patching the wrong module (`embeddings_comparison` vs
  `zehutai.embeddings_comparison`). Fast suite now 288/288.
- Assumptions for Yehoshua: see `webapp/README.md`.

## Current State
Research project for Hebrew/multilingual text similarity and RAG pipelines.
**288 tests passing** (+5 slow stubs), ~5s runtime. CI/CD with 70% coverage threshold.
Full-featured codebase with: sentence similarity, multi-method comparison (TF-IDF, BERT, RoBERTa, Doc2Vec),
RAG pipeline with Hebrew preprocessing and chunking, IR evaluation metrics (P@k, R@k, AP, MRR, NDCG),
Hebrew benchmark suite (20 sentence pairs), and CLI interface.
Model caching across all modules. Pre-commit hooks (cross-platform). src/ layout plan ready for execution.

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
- [x] **SECURITY: Scrubbed leaked API key from git history** using git filter-branch (2026-02-17)
- [x] **Created backup branch** `backup-before-cleanup` before destructive git operations (2026-02-17)
- [x] **Updated .env.example** with placeholder values (OPENAI_API_KEY=your-key-here) (2026-02-17)
- [x] **Added pre-commit hook** (.git/hooks/pre-commit) scanning for secret patterns (sk-, api_key, password, token, etc.) (2026-02-17)
- [x] **Created GitHub Actions CI** (.github/workflows/ci.yml) - pytest, ruff lint, secret scan across Python 3.11/3.12 (2026-02-17)
- [x] **Added Hebrew NLP benchmark** (tests/test_hebrew_benchmark.py) - 10 Hebrew sentence pairs, 17 new tests (2026-02-17)
- [x] **Added model caching to sim.py** - `_get_or_load_model()` caches tokenizer/model pairs by weights key (2026-02-18)
- [x] **Fixed `.T` inconsistency in methodRoBERTa** - removed unnecessary `.T` on 1D array, consistent with methodBert (2026-02-18)
- [x] **Enhanced RAG pipeline** - added `preprocess_text()`, `chunk_documents()` with overlap, `top_k` param in `vector_search()` (2026-02-18)
- [x] **Added 6 model caching tests** - cache hit/miss, identity, separate entries (2026-02-18)
- [x] **Added 18 RAG tests** - preprocess_text (6), chunk_documents (9), vector_search top_k (3) (2026-02-18)
- [x] **Added 3 integration test stubs** (`@pytest.mark.slow`) for real BERT/RoBERTa model testing (2026-02-18)
- [x] **Configured pytest to skip slow tests by default** via `addopts = "-m 'not slow'"` (2026-02-18)
- [x] **Created .pre-commit-config.yaml** - ruff, detect-secrets, standard hooks for team sharing (2026-02-18)

## Performance Improvements (2026-02-18)

### Model Caching in sim.py
- **Problem:** `methodBert()`, `methodRoBERTa()`, and `methodNNEmbeddings()` called `from_pretrained()` on every invocation, re-downloading models each time
- **Fix:** Module-level `_model_cache` dict with `_get_or_load_model()` helper that loads once and caches by `model_weights` key
- **Impact:** Second+ calls skip model loading entirely (from ~30s to ~0s per call)

### RAG Pipeline Enhancements
- `preprocess_text()`: normalizes whitespace, strips leading/trailing spaces
- `chunk_documents()`: splits long docs into overlapping chunks (configurable `chunk_size` and `overlap`)
- `vector_search()`: new `top_k` parameter limits returned results (backward-compatible, defaults to all)
- RRF docstring enhanced with explanation of why `k=60` is the standard value

## Security Hardening (2026-02-17)

### Git History Cleanup
- **Problem:** Leaked OpenAI API key (`sk-proj-...`) was present in commits 148e309, c27e813, 60b4961
- **Fix:** Used `git filter-branch --tree-filter` to replace the key with `REMOVED_LEAKED_KEY` in ALL commits
- **Verification:** `git rev-list --all` shows zero commits containing the original key
- **Cleanup:** Removed refs/original backup refs, expired reflog, aggressive gc

### Pre-commit Secret Scanning
- Bash hook at `.git/hooks/pre-commit`
- Scans staged diffs for: `sk-*`, `api_key=`, `password=`, `secret=`, `token=`, GitHub PATs, Slack tokens
- Blocks commit with clear error message; bypassable with `--no-verify` for false positives

### CI Secret Scanning
- Separate `secret-scan` job in GitHub Actions
- Scans all tracked .py/.yml/.yaml/.json/.cfg/.ini/.toml files
- Fails the CI pipeline if potential secrets are detected

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
1. ~~**URGENT: Rotate the leaked OpenAI API key**~~ -- DONE (key scrubbed from history, 2026-02-17)
2. ~~Consider using BFG Repo Cleaner to remove the key from git history~~ -- DONE (used git filter-branch, 2026-02-17)
3. ~~Force-push to GitHub~~ -- DONE (branch synced with origin/main)
4. **IMPORTANT:** Rotate the OpenAI API key on https://platform.openai.com/api-keys (the old key is compromised)
5. ~~Add model caching to sim.py methodBert/methodRoBERTa~~ -- DONE (2026-02-18)
6. ~~Consider restructuring into a proper `src/` layout~~ -- PLAN READY at .omc/plans/src-layout-plan.md (2026-02-18)
7. ~~Set up CI/CD (GitHub Actions for pytest + ruff + mypy)~~ -- DONE (2026-02-17)
8. ~~Add integration tests with `@pytest.mark.slow`~~ -- DONE (5 stubs, 2026-02-18)
9. ~~Add Hebrew-specific test data and evaluation benchmarks~~ -- DONE (2026-02-17)
10. ~~Improve RAG pipeline: chunk overlap, configurable top-k, preprocessing~~ -- DONE (2026-02-18)
11. ~~Run `pre-commit install`~~ -- DONE (2026-02-18, fixed for Windows cross-platform)
12. ~~Add model caching to rag.py `generate_queries()`~~ -- DONE (_get_or_load_rag_model, 2026-02-18)
13. ~~Add Hebrew-specific preprocessing~~ -- DONE (preprocess_hebrew with niqqud removal, 2026-02-18)
14. ~~Add evaluation metrics (precision@k, NDCG)~~ -- DONE (evaluation.py, 2026-02-18)
15. ~~Add Hebrew benchmark suite~~ -- DONE (20 pairs, run_benchmark, format_report, 2026-02-18)
16. ~~Add CLI interface~~ -- DONE (cli.py: compare, similarity, rag, benchmark, 2026-02-18)
17. ~~Add coverage to CI~~ -- DONE (70% threshold, coverage.xml artifact, 2026-02-18)
18. ~~Fix pre-commit for Windows~~ -- DONE (replaced bash hook with Python, 2026-02-18)
19. Execute src/ layout migration (plan ready at .omc/plans/src-layout-plan.md)
20. **IMPORTANT:** Rotate OpenAI API key at https://platform.openai.com/api-keys

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

## Files Modified (2026-02-17)
- `.github/workflows/ci.yml` - Created (new file) - CI pipeline: pytest, ruff, secret scan
- `.git/hooks/pre-commit` - Created (new file) - Secret pattern scanning hook
- `.env.example` - Updated placeholder values
- `tests/test_hebrew_benchmark.py` - Created (new file) - 17 Hebrew NLP benchmark tests
- `PROGRESS.md` - Updated with security and CI work
- Git history rewritten with `git filter-branch` to remove leaked API key

## Files Modified (2026-02-18)
- `YehoshuaSimilarityComparisons/sim.py` - Added `_model_cache`, `_get_or_load_model()`, updated methodBert/methodRoBERTa/methodNNEmbeddings to use cache, fixed `.T` in methodRoBERTa
- `YehoshuaSimilarityComparisons/rag.py` - Added `preprocess_text()`, `chunk_documents()`, `top_k` param in `vector_search()`, enhanced RRF docstring
- `tests/test_similarity.py` - Added 6 cache tests (TestModelCaching) + 3 integration stubs (TestIntegrationBert)
- `tests/test_rag.py` - Added 18 tests: TestPreprocessText (6), TestChunkDocuments (9), TestVectorSearchTopK (3)
- `pyproject.toml` - Added `addopts = "-m 'not slow'"` to skip slow tests by default
- `.pre-commit-config.yaml` - Created (new file) - ruff, detect-secrets, standard hooks
- `PROGRESS.md` - Updated with all changes

## Files Modified/Created (2026-02-18 session 2)
- `YehoshuaSimilarityComparisons/rag.py` - Added _get_or_load_rag_model, preprocess_hebrew, rag_pipeline
- `YehoshuaSimilarityComparisons/evaluation.py` - NEW: IR metrics (P@k, R@k, AP, MRR, NDCG, F1)
- `YehoshuaSimilarityComparisons/hebrew_benchmark.py` - NEW: 20 Hebrew pairs, run_benchmark, format_report
- `cli.py` - NEW: argparse CLI (compare, similarity, rag, benchmark subcommands)
- `tests/test_rag.py` - Extended with 22 new tests (cache, Hebrew, pipeline)
- `tests/test_evaluation.py` - NEW: 50 tests for evaluation metrics
- `tests/test_cli.py` - NEW: 25 tests for CLI
- `tests/test_hebrew_benchmark_module.py` - NEW: 33 tests for benchmark corpus
- `tests/test_hebrew_model_benchmark.py` - NEW: 57 tests for model benchmark harness
- `.github/workflows/ci.yml` - Added coverage reporting + artifact upload
- `pyproject.toml` - Added [tool.coverage] config
- `.pre-commit-config.yaml` - Fixed: replaced bash hook with cross-platform Python
- `.omc/plans/src-layout-plan.md` - NEW: detailed migration plan for src/ layout
- `CLAUDE.md` - Full rewrite with all new modules and functions
- `PROGRESS.md` - Updated

## Notes for Next Session
- **IMPORTANT:** Rotate the OpenAI API key at https://platform.openai.com/api-keys
- Execute src/ layout migration using plan at `.omc/plans/src-layout-plan.md`
- Slow integration tests (5) need a real model environment (GPU or large download)
- Consider publishing as a pip-installable package after src/ migration
- Coverage is 72% - could improve sim.py coverage (currently 47%)
