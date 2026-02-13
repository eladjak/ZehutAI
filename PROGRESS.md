# ZehutAI - Progress

## Status: Active (Research/Prototype)
## Last Updated: 2026-02-13

## Current State
Early-stage research project exploring Hebrew/multilingual text similarity and RAG pipelines.
Code exists but has several bugs, no tests, no type hints, and needs structural cleanup.
Git repo exists with remote at github.com/eladjak/ZehutAI.git.

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
- [x] SECURITY FIX: Removed leaked OpenAI API key from rag.py

## Code Review Findings (2026-02-13)

### CRITICAL - Security
1. **LEAKED API KEY** in `YehoshuaSimilarityComparisons/rag.py` line 4
   - OpenAI API key was committed (commented out but visible in git history)
   - **ACTION REQUIRED: Rotate this key immediately on OpenAI dashboard**
   - Key removed from source code in this session

### HIGH - Bugs
1. **rag.py:55** - `reciprocal_rank_fusion()` uses variable `doc` but the loop variable is `docc`
   - Will cause `NameError` at runtime
2. **rag.py:32** - `generate_queries()` passes undefined `input_ids` to `model.generate()`
   - Will cause `NameError` at runtime
3. **sim.py:32** - `removeModel()` compares `model_weights` to itself (`model_weights == model_weights`)
   - Should compare `model` parameter to `model_weights` in tuple - always evaluates True

### MEDIUM - Code Quality
1. **Model re-instantiation** - `compare_sentences()` creates a new SentenceTransformer on every call
   - Should cache/reuse the model instance
2. **Hardcoded test data** - `sim.py` methods have hardcoded sample texts
   - Should accept parameters
3. **Duplicate code** - `embeddings_comparison.py` exists in both root and subdirectory
4. **No type hints** on any functions
5. **No docstrings** on most functions (only `compare_sentences` and `methodNNEmbeddings` have them)
6. **Wildcard import** - `main.py` uses `from plotting import *`
7. **Unused imports** - `tensorflow` imported in sim.py but never used
8. **Module-level side effects** - `sim.py` runs code at module level (line 247-254)
9. **torch.device()` calls without assignment** - Device objects created but not stored in rag.py and simple_rag.py

### LOW - Style
1. No consistent naming convention (camelCase mixed with snake_case)
2. Debug print statements throughout (`print('&&&&&&&&&&&&')` in simple_rag.py)
3. Commented-out code blocks in multiple files
4. Typos in sample data ("palying", "throughg")

## Next Steps
1. **URGENT: Rotate the leaked OpenAI API key**
2. Fix the 3 runtime bugs (rag.py variables, sim.py removeModel)
3. Add type hints to all functions
4. Cache model instances (singleton pattern or module-level)
5. Extract hardcoded test data into configurable parameters
6. Remove duplicate embeddings_comparison.py (keep one, import from there)
7. Add `__init__.py` to make YehoshuaSimilarityComparisons a proper package
8. Add unit tests (at least for compare_sentences and vector_search)
9. Remove unused imports (tensorflow)
10. Clean up debug print statements
11. Consider restructuring into a proper Python package with src/ layout

## Key Decisions Made
- Using sentence-transformers/paraphrase-multilingual-mpnet-base-v2 as primary model (good Hebrew support)
- Exploring DictaLM 2.0 for Hebrew text generation
- RAG approach: query expansion -> vector search -> reciprocal rank fusion

## Files Modified (2026-02-13)
- `.gitignore` - Created (Python-specific ignores)
- `requirements.txt` - Created (all project dependencies)
- `README.md` - Rewritten with full documentation
- `CLAUDE.md` - Updated with accurate project info and known issues
- `PROGRESS.md` - Full rewrite with code review findings
- `YehoshuaSimilarityComparisons/rag.py` - Removed leaked API key

## Notes for Next Session
- The leaked API key is still in git history - consider using `git filter-branch` or BFG Repo Cleaner
- Start with fixing the 3 runtime bugs before adding new features
- The project mixes research/experimentation scripts - consider separating library code from experiments
