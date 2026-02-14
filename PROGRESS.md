# ZehutAI - Progress

## Status: Active (Research/Prototype)
## Last Updated: 2026-02-14

## Current State
Early-stage research project exploring Hebrew/multilingual text similarity and RAG pipelines.
Major code quality improvements applied: all 3 runtime bugs fixed, type hints added,
module-level side effects removed, model caching implemented, debug prints cleaned up,
and proper environment variable handling added. No secrets remain in source code.

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
- [x] Proper device detection pattern (module-level, no dangling torch.device) (2026-02-14)

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

## Next Steps
1. **URGENT: Rotate the leaked OpenAI API key** (still in git history!)
2. Consider using BFG Repo Cleaner to remove the key from git history
3. Add unit tests for `compare_sentences()`, `vector_search()`, `reciprocal_rank_fusion()`
4. Remove duplicate `embeddings_comparison.py` (keep one canonical location, import from there)
5. Consider restructuring into a proper Python package with `src/` layout
6. Add a `pyproject.toml` for modern Python packaging
7. Set up CI/CD (GitHub Actions for linting + type checking)

## Key Decisions Made
- Using sentence-transformers/paraphrase-multilingual-mpnet-base-v2 as primary model (good Hebrew support)
- Exploring DictaLM 2.0 for Hebrew text generation
- RAG approach: query expansion -> vector search -> reciprocal rank fusion
- Module-level model caching over class-based singleton (simpler for research code)
- Google-style docstrings for consistency

## Files Modified (2026-02-14)
- `embeddings_comparison.py` - Added model caching, type hints, docstrings
- `YehoshuaSimilarityComparisons/embeddings_comparison.py` - Same improvements
- `YehoshuaSimilarityComparisons/rag.py` - Fixed 2 runtime bugs, added type hints/docstrings, cleaned up
- `YehoshuaSimilarityComparisons/sim.py` - Fixed removeModel bug, removed tensorflow, removed side effects, added types
- `YehoshuaSimilarityComparisons/main.py` - Fixed wildcard import, added docstring
- `YehoshuaSimilarityComparisons/plotting.py` - Cleaned up, added docstring and type hints
- `YehoshuaSimilarityComparisons/simple_rag.py` - Removed debug prints, cleaned up imports/device handling
- `YehoshuaSimilarityComparisons/ragtest.py` - Cleaned up device handling, added docstring
- `YehoshuaSimilarityComparisons/__init__.py` - Created (new file)
- `.env.example` - Created (new file)
- `.gitignore` - Fixed duplicate .env entry
- `requirements.txt` - Added python-dotenv
- `PROGRESS.md` - Updated with all changes

## Notes for Next Session
- The leaked API key is still in git history - consider using `git filter-branch` or BFG Repo Cleaner
- The project still has two copies of embeddings_comparison.py - consolidate in next session
- Consider adding a conftest.py and basic pytest tests
- sim.py methods still create models on every call (BERT, RoBERTa) - could add caching
