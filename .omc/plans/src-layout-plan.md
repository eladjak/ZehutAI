# ZehutAI: Migration to `src/` Layout Plan

**Created:** 2026-02-18
**Status:** PLAN ONLY — not yet executed
**Author:** Planning agent (claude-sonnet-4-5)

---

## 1. Current State Summary

### Source layout (flat)
```
ZehutAI/
  embeddings_comparison.py          # canonical; compare_sentences(), _model, MODEL_NAME
  pyproject.toml
  tests/
    __init__.py
    conftest.py
    test_embeddings_comparison.py
    test_evaluation.py
    test_hebrew_benchmark.py
    test_hebrew_model_benchmark.py
    test_rag.py
    test_similarity.py
  YehoshuaSimilarityComparisons/
    __init__.py                      # docstring only
    embeddings_comparison.py         # re-export shim → root embeddings_comparison
    evaluation.py                    # IR metrics (precision, recall, MRR, NDCG, F1)
    hebrew_benchmark.py              # benchmark suite + MODEL_CONFIGS + evaluate_benchmark
    main.py                          # entry point; imports `from sim import Similarity`
    plotting.py                      # BERT runner; imports `from sim import Similarity`
    rag.py                           # RAG pipeline; imports `from embeddings_comparison import compare_sentences`
    ragtest.py                       # standalone demo script (module-level model load)
    sim.py                           # Similarity class; BERT, RoBERTa, TF-IDF, Doc2Vec
    simple_rag.py                    # standalone demo script (module-level model load)
```

### `pyproject.toml` package discovery
- No explicit `[tool.setuptools.packages]` — setuptools auto-discovers.
- `[tool.pytest.ini_options] pythonpath = [".", "YehoshuaSimilarityComparisons"]`
- Tests import modules by bare name (`import embeddings_comparison`, `from rag import ...`,
  `from sim import ...`, `from YehoshuaSimilarityComparisons.evaluation import ...`,
  `from hebrew_benchmark import ...`).

### CI (`ci.yml`)
- `--cov=. --cov=YehoshuaSimilarityComparisons`
- Installs via `pip install -r requirements.txt` (not editable install).

---

## 2. Target Structure

```
src/
  zehutai/
    __init__.py              # package root; re-exports public API
    embeddings_comparison.py # moved from root
    cli.py                   # optional entry-point wrapper (new file)
    similarity/
      __init__.py            # re-exports Similarity
      sim.py                 # moved from YehoshuaSimilarityComparisons/sim.py
      plotting.py            # moved from YehoshuaSimilarityComparisons/plotting.py
    rag/
      __init__.py            # re-exports public RAG API
      rag.py                 # moved from YehoshuaSimilarityComparisons/rag.py
      simple_rag.py          # moved from YehoshuaSimilarityComparisons/simple_rag.py
    hebrew/
      __init__.py            # re-exports benchmark public API
      hebrew_benchmark.py    # moved from YehoshuaSimilarityComparisons/hebrew_benchmark.py
    evaluation/
      __init__.py            # re-exports evaluation metrics
      evaluation.py          # moved from YehoshuaSimilarityComparisons/evaluation.py
```

**Demo/entry-point scripts** (these are not library code; they are standalone runners):
- `YehoshuaSimilarityComparisons/main.py`  → `src/zehutai/cli.py` (converted to proper CLI)
- `YehoshuaSimilarityComparisons/ragtest.py`  → `scripts/ragtest.py` (standalone script, no package import needed)
- `YehoshuaSimilarityComparisons/simple_rag.py` → `src/zehutai/rag/simple_rag.py` OR `scripts/simple_rag.py`
  - Because `simple_rag.py` executes model loading at module import time (no `if __name__ == "__main__":` guard), it is safest to move it to `scripts/` to keep it out of the importable package.

---

## 3. Files to Move — Exact Source → Destination Mapping

| # | Source (relative to repo root) | Destination (relative to repo root) | Notes |
|---|-------------------------------|--------------------------------------|-------|
| 1 | `embeddings_comparison.py` | `src/zehutai/embeddings_comparison.py` | Canonical module |
| 2 | `YehoshuaSimilarityComparisons/sim.py` | `src/zehutai/similarity/sim.py` | |
| 3 | `YehoshuaSimilarityComparisons/plotting.py` | `src/zehutai/similarity/plotting.py` | |
| 4 | `YehoshuaSimilarityComparisons/rag.py` | `src/zehutai/rag/rag.py` | |
| 5 | `YehoshuaSimilarityComparisons/evaluation.py` | `src/zehutai/evaluation/evaluation.py` | |
| 6 | `YehoshuaSimilarityComparisons/hebrew_benchmark.py` | `src/zehutai/hebrew/hebrew_benchmark.py` | |
| 7 | `YehoshuaSimilarityComparisons/main.py` | `src/zehutai/cli.py` | Convert to `__main__` entry point |
| 8 | `YehoshuaSimilarityComparisons/simple_rag.py` | `scripts/simple_rag.py` | Module-level side effects; keep out of package |
| 9 | `YehoshuaSimilarityComparisons/ragtest.py` | `scripts/ragtest.py` | Module-level side effects; keep out of package |
| 10 | `YehoshuaSimilarityComparisons/__init__.py` | DELETE (old sub-package removed) | Contents are a docstring only |
| 11 | `YehoshuaSimilarityComparisons/embeddings_comparison.py` | DELETE (was a shim; no longer needed) | |

**New files to create:**
| # | Path | Contents |
|---|------|----------|
| A | `src/zehutai/__init__.py` | Package docstring + public API re-exports |
| B | `src/zehutai/similarity/__init__.py` | Re-export `Similarity` |
| C | `src/zehutai/rag/__init__.py` | Re-export key RAG functions |
| D | `src/zehutai/evaluation/__init__.py` | Re-export metric functions |
| E | `src/zehutai/hebrew/__init__.py` | Re-export benchmark functions |
| F | `scripts/` directory | Created to hold standalone demo scripts |

---

## 4. Import Changes Required — File by File

### 4.1 `src/zehutai/embeddings_comparison.py` (was root `embeddings_comparison.py`)
**No import changes required.** This file only imports from `sentence_transformers` and standard library. It has no intra-project imports.

---

### 4.2 `src/zehutai/rag/rag.py` (was `YehoshuaSimilarityComparisons/rag.py`)

**Current import (line 13):**
```python
from embeddings_comparison import compare_sentences
```

**New import:**
```python
from zehutai.embeddings_comparison import compare_sentences
```

**Also update `@patch` targets in `tests/test_rag.py`:**
All `@patch("rag.compare_sentences")` → `@patch("zehutai.rag.rag.compare_sentences")`
All `@patch("rag.AutoModelForCausalLM")` → `@patch("zehutai.rag.rag.AutoModelForCausalLM")`
All `@patch("rag.AutoTokenizer")` → `@patch("zehutai.rag.rag.AutoTokenizer")`
All `@patch("rag.generate_queries")` → `@patch("zehutai.rag.rag.generate_queries")`
And `import rag` → `from zehutai.rag import rag` (or `import zehutai.rag.rag as rag`)

---

### 4.3 `src/zehutai/similarity/plotting.py` (was `YehoshuaSimilarityComparisons/plotting.py`)

**Current import (line 6):**
```python
from sim import Similarity
```

**New import:**
```python
from zehutai.similarity.sim import Similarity
```

---

### 4.4 `src/zehutai/cli.py` (was `YehoshuaSimilarityComparisons/main.py`)

**Current import (line 5):**
```python
from sim import Similarity
```

**New import:**
```python
from zehutai.similarity.sim import Similarity
```

Also add `if __name__ == "__main__": main()` guard (already present; keep it).

---

### 4.5 `src/zehutai/similarity/sim.py` (was `YehoshuaSimilarityComparisons/sim.py`)
**No intra-project imports.** All imports are from external packages (`numpy`, `transformers`, `gensim`, `nltk`, `sklearn`). No changes needed.

---

### 4.6 `src/zehutai/evaluation/evaluation.py` (was `YehoshuaSimilarityComparisons/evaluation.py`)
**No intra-project imports.** Only `import math`. No changes needed.

---

### 4.7 `src/zehutai/hebrew/hebrew_benchmark.py` (was `YehoshuaSimilarityComparisons/hebrew_benchmark.py`)
**No intra-project imports.** Only `importlib`, `time`, `numpy`. No changes needed.

---

### 4.8 `scripts/simple_rag.py` (was `YehoshuaSimilarityComparisons/simple_rag.py`)
**No intra-project imports.** Only `os`, `torch`, `transformers`. No changes needed.

---

### 4.9 `scripts/ragtest.py` (was `YehoshuaSimilarityComparisons/ragtest.py`)
**No intra-project imports.** Only `torch`, `transformers`. No changes needed.

---

### 4.10 `src/zehutai/__init__.py` (NEW)
```python
"""ZehutAI — Hebrew/multilingual AI text analysis, semantic similarity, and RAG pipelines."""

from zehutai.embeddings_comparison import MODEL_NAME, compare_sentences

__all__ = ["compare_sentences", "MODEL_NAME"]
```

---

### 4.11 `src/zehutai/similarity/__init__.py` (NEW)
```python
"""Similarity sub-package: multi-method text similarity."""

from zehutai.similarity.sim import Similarity

__all__ = ["Similarity"]
```

---

### 4.12 `src/zehutai/rag/__init__.py` (NEW)
```python
"""RAG sub-package: retrieval-augmented generation pipeline."""

from zehutai.rag.rag import (
    chunk_documents,
    preprocess_hebrew,
    preprocess_text,
    rag_pipeline,
    reciprocal_rank_fusion,
    vector_search,
)

__all__ = [
    "chunk_documents",
    "preprocess_hebrew",
    "preprocess_text",
    "rag_pipeline",
    "reciprocal_rank_fusion",
    "vector_search",
]
```

---

### 4.13 `src/zehutai/evaluation/__init__.py` (NEW)
```python
"""Evaluation sub-package: IR/NLP metrics."""

from zehutai.evaluation.evaluation import (
    average_precision,
    f1_score,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    "average_precision",
    "f1_score",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]
```

---

### 4.14 `src/zehutai/hebrew/__init__.py` (NEW)
```python
"""Hebrew sub-package: embedding model benchmark suite."""

from zehutai.hebrew.hebrew_benchmark import (
    HEBREW_BENCHMARK_PAIRS,
    MODEL_CONFIGS,
    evaluate_benchmark,
    format_benchmark_report,
    run_benchmark,
)

__all__ = [
    "HEBREW_BENCHMARK_PAIRS",
    "MODEL_CONFIGS",
    "evaluate_benchmark",
    "format_benchmark_report",
    "run_benchmark",
]
```

---

## 5. `pyproject.toml` Changes

### 5.1 Package discovery — add explicit src layout configuration

**Current:** no explicit `[tool.setuptools.packages]` or `[tool.setuptools.package-dir]`

**Add after `[build-system]`:**
```toml
[tool.setuptools.package-dir]
"" = "src"

[tool.setuptools.packages.find]
where = ["src"]
```

This tells setuptools that all packages live under `src/` and that the root of the distribution is `src/`.

### 5.2 Entry points — add CLI entry point for `cli.py`

**Add to `[project]`:**
```toml
[project.scripts]
zehutai = "zehutai.cli:main"
```

### 5.3 pytest pythonpath — replace with `src` layout path

**Current:**
```toml
[tool.pytest.ini_options]
pythonpath = [".", "YehoshuaSimilarityComparisons"]
```

**New:**
```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
```

Adding `src` to `pythonpath` allows tests to `import zehutai` without an editable install. Remove `.` (no longer needed since nothing imports bare module names from root) and remove `YehoshuaSimilarityComparisons` (that directory will be deleted).

### 5.4 Coverage source path — update in pyproject.toml if present
Currently coverage config is only in the CI command line (`--cov=. --cov=YehoshuaSimilarityComparisons`). If a `[tool.coverage]` section is added later, it should point to `src/zehutai`.

---

## 6. Test Changes — File by File

### 6.1 `tests/conftest.py`

**Current sys.path manipulation (lines 17–22):**
```python
_root = str(Path(__file__).resolve().parent.parent)
_sub = str(Path(__file__).resolve().parent.parent / "YehoshuaSimilarityComparisons")
for p in (_root, _sub):
    if p not in sys.path:
        sys.path.insert(0, p)
```

**New:**
```python
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
```

(This is a belt-and-suspenders guard; `pythonpath = ["src"]` in `pyproject.toml` handles it for pytest, but the explicit insertion keeps the conftest self-contained for editors and direct `python` invocations.)

**Mock fixture `mock_sentence_transformer` (lines 156–177):**
```python
import embeddings_comparison as ec
```
Must change to:
```python
import zehutai.embeddings_comparison as ec
```
And `monkeypatch.setattr(ec, "_model", mock_model)` remains correct since it patches the attribute on the module object.

---

### 6.2 `tests/test_embeddings_comparison.py`

**Current (line 15):**
```python
import embeddings_comparison as ec
```

**New:**
```python
import zehutai.embeddings_comparison as ec
```

All `ec.compare_sentences(...)`, `ec._get_model()`, `ec._model`, `ec.SentenceTransformer`, `ec.MODEL_NAME` references remain valid (same attribute names on the module).

**Patch strings — update:**
- `monkeypatch.setattr(ec, "_model", mock)` — unchanged (works on module object)
- `monkeypatch.setattr(ec, "SentenceTransformer", mock_cls)` — unchanged

---

### 6.3 `tests/test_rag.py`

**Current imports (lines 14–25):**
```python
from rag import (
    ALL_DOCUMENTS,
    _get_or_load_rag_model,
    _rag_model_cache,
    chunk_documents,
    generate_output,
    preprocess_hebrew,
    preprocess_text,
    rag_pipeline,
    reciprocal_rank_fusion,
    vector_search,
)
```

**New:**
```python
from zehutai.rag.rag import (
    ALL_DOCUMENTS,
    _get_or_load_rag_model,
    _rag_model_cache,
    chunk_documents,
    generate_output,
    preprocess_hebrew,
    preprocess_text,
    rag_pipeline,
    reciprocal_rank_fusion,
    vector_search,
)
```

**Patch strings — update all occurrences:**

| Current | New |
|---------|-----|
| `@patch("rag.compare_sentences")` | `@patch("zehutai.rag.rag.compare_sentences")` |
| `@patch("rag.AutoModelForCausalLM")` | `@patch("zehutai.rag.rag.AutoModelForCausalLM")` |
| `@patch("rag.AutoTokenizer")` | `@patch("zehutai.rag.rag.AutoTokenizer")` |
| `@patch("rag.generate_queries")` | `@patch("zehutai.rag.rag.generate_queries")` |
| `import rag` (inside tests) | `from zehutai.rag import rag` (then use `rag._rag_model_cache.pop(...)`) |

---

### 6.4 `tests/test_similarity.py`

**Current import (line 13):**
```python
from sim import DEFAULT_DATA, DEFAULT_QUERY, Similarity, _get_or_load_model, _model_cache
```

**New:**
```python
from zehutai.similarity.sim import DEFAULT_DATA, DEFAULT_QUERY, Similarity, _get_or_load_model, _model_cache
```

No other changes needed (no patch strings referencing `sim.`).

---

### 6.5 `tests/test_evaluation.py`

**Current import (lines 9–16):**
```python
from YehoshuaSimilarityComparisons.evaluation import (
    average_precision,
    f1_score,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
```

**New:**
```python
from zehutai.evaluation.evaluation import (
    average_precision,
    f1_score,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
```

---

### 6.6 `tests/test_hebrew_benchmark.py` and `tests/test_hebrew_model_benchmark.py`

**Current sys.path manipulation (test_hebrew_benchmark.py lines 19–23):**
```python
_sub = str(Path(__file__).resolve().parent.parent / "YehoshuaSimilarityComparisons")
if _sub not in sys.path:
    sys.path.insert(0, _sub)
```

**Remove** this block entirely (conftest.py now handles path).

**Current import (lines 25–30):**
```python
from hebrew_benchmark import (
    HEBREW_BENCHMARK_PAIRS,
    MODEL_CONFIGS,
    _cosine_similarity,
    evaluate_benchmark,
    format_benchmark_report,
    ...
)
```

**New:**
```python
from zehutai.hebrew.hebrew_benchmark import (
    HEBREW_BENCHMARK_PAIRS,
    MODEL_CONFIGS,
    _cosine_similarity,
    evaluate_benchmark,
    format_benchmark_report,
    ...
)
```

**`tests/test_hebrew_model_benchmark.py`:**
```python
import embeddings_comparison as ec
```
→
```python
import zehutai.embeddings_comparison as ec
```

---

### 6.7 `tests/__init__.py`
Currently empty or minimal. No changes needed.

---

## 7. CI Changes (`.github/workflows/ci.yml`)

### 7.1 Install step — switch to editable install

**Current:**
```yaml
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -r requirements.txt
```

**New (recommended for src layout):**
```yaml
- name: Install package and dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -e ".[dev]"
```

Using `pip install -e ".[dev]"` installs the package in editable mode, which makes `import zehutai` work without needing to manipulate `PYTHONPATH`. This also installs all `dev` extras (pytest, ruff, mypy).

If `requirements.txt` must be kept for other reasons, add:
```yaml
    pip install -r requirements.txt
    pip install -e .
```

### 7.2 Coverage command — update `--cov` paths

**Current:**
```yaml
python -m pytest tests/ --cov=. --cov=YehoshuaSimilarityComparisons --cov-report=term-missing --cov-report=xml --cov-fail-under=70
```

**New:**
```yaml
python -m pytest tests/ --cov=zehutai --cov-report=term-missing --cov-report=xml --cov-fail-under=70
```

The `--cov=zehutai` flag tells pytest-cov to measure coverage on the `zehutai` package (found automatically under `src/` once the package is installed). Remove the old `--cov=.` and `--cov=YehoshuaSimilarityComparisons`.

### 7.3 Ruff lint step — no changes needed
`ruff check .` will discover `pyproject.toml` and lint `src/` correctly.

### 7.4 pip cache key — update if switching from requirements.txt
If switching to `pyproject.toml`-based install, update the cache key:
```yaml
key: ${{ runner.os }}-pip-${{ hashFiles('pyproject.toml') }}
```

---

## 8. Backward Compatibility

### 8.1 Root `embeddings_comparison.py` shim

After moving the file to `src/zehutai/embeddings_comparison.py`, any external code or scripts that do `import embeddings_comparison` will break. To provide a backward compatibility shim during transition, leave a thin shim at the old location:

```python
# embeddings_comparison.py  (ROOT — backward-compat shim, delete after transition)
"""Backward compatibility shim. Import from zehutai.embeddings_comparison instead."""
from zehutai.embeddings_comparison import MODEL_NAME, compare_sentences  # noqa: F401
__all__ = ["compare_sentences", "MODEL_NAME"]
```

Remove this shim once all callers are updated.

### 8.2 `YehoshuaSimilarityComparisons/` package shim

The old sub-package can be replaced with a thin backward-compat shim package that re-exports from the new locations. This is useful if any external code imports from `YehoshuaSimilarityComparisons.*`.

```python
# YehoshuaSimilarityComparisons/__init__.py  (shim)
"""Backward-compat shim. Import from zehutai.* instead."""
from zehutai.evaluation.evaluation import *  # noqa: F401, F403
from zehutai.hebrew.hebrew_benchmark import *  # noqa: F401, F403
from zehutai.similarity.sim import *  # noqa: F401, F403
```

Individual module shims:
```python
# YehoshuaSimilarityComparisons/sim.py  (shim)
from zehutai.similarity.sim import *  # noqa: F401, F403

# YehoshuaSimilarityComparisons/rag.py  (shim)
from zehutai.rag.rag import *  # noqa: F401, F403

# YehoshuaSimilarityComparisons/evaluation.py  (shim)
from zehutai.evaluation.evaluation import *  # noqa: F401, F403

# YehoshuaSimilarityComparisons/hebrew_benchmark.py  (shim)
from zehutai.hebrew.hebrew_benchmark import *  # noqa: F401, F403
```

**Recommendation:** Since this is a research project with no known external consumers, skip the shims and do a hard cut-over. The shim approach is recommended only if there are notebooks, scripts, or external consumers that cannot be updated simultaneously.

### 8.3 `ragtest.py` and `simple_rag.py`

Both scripts execute model loading at module import time (no `if __name__ == "__main__":` guard in `simple_rag.py`). Moving them to `scripts/` keeps them outside the importable package, preventing accidental side effects during test collection or imports. They do not need backward compatibility shims since they are standalone runner scripts, not library modules.

---

## 9. Risk Assessment

### High Risk

| Risk | Description | Mitigation |
|------|-------------|------------|
| `@patch` target strings in tests | 16+ `@patch(...)` calls in `test_rag.py` reference bare module names like `"rag.compare_sentences"`. After migration these must reference the full dotted path `"zehutai.rag.rag.compare_sentences"`. Forgetting to update any one of them will cause silent test failures (the mock won't apply). | Do a `grep -r 'patch("rag\.' tests/` after migration and verify all are updated. |
| `import rag` inside test methods | `test_rag.py` has `import rag` inside test method bodies (for `rag._rag_model_cache.pop(...)`). These will raise `ModuleNotFoundError` after migration. | Change to `from zehutai.rag import rag as rag_module`. |
| sys.path manipulation in conftest | `conftest.py` manually inserts both root and `YehoshuaSimilarityComparisons/` into `sys.path`. After migration, if the old path is left and the new `src/` path is not added, imports will silently fall back to the old (deleted) location if shims are not present, or fail with `ModuleNotFoundError`. | Update `conftest.py` as described in section 6.1 as the very first step. |

### Medium Risk

| Risk | Description | Mitigation |
|------|-------------|------------|
| setuptools auto-discovery picks up both old and new locations | If `YehoshuaSimilarityComparisons/` is not removed before updating `pyproject.toml`, setuptools may discover both packages. | Remove the old directory (or replace with shims) before running `pip install -e .`. |
| `simple_rag.py` module-level model load | `simple_rag.py` loads a 4GB+ model at import time. If accidentally imported during test collection, tests will hang or fail with OOM. | Keep in `scripts/` (outside package), never in `src/zehutai/`. |
| `ragtest.py` module-level model load | Same issue. | Same mitigation. |
| Coverage drop below 70% | If `--cov=zehutai` misses some files or scripts are no longer counted, coverage might dip below the enforced 70% threshold. | Run coverage check locally after migration; adjust `--cov-fail-under` if scripts are intentionally excluded. |

### Low Risk

| Risk | Description | Mitigation |
|------|-------------|------------|
| `YehoshuaSimilarityComparisons/embeddings_comparison.py` shim path | The old shim manipulates `sys.path` to re-import from root. After the root file is moved, this shim will fail. | Delete this shim (it will be unreachable via the new import path). |
| ruff finds issues in `scripts/` | Standalone scripts may have patterns (module-level side effects, missing type hints) that ruff flags. | Either fix the issues, add `# noqa` annotations, or exclude `scripts/` from ruff via `[tool.ruff] exclude = ["scripts/"]`. |
| mypy path configuration | mypy currently has no `mypy_path` configured; it discovers `YehoshuaSimilarityComparisons` via `sys.path`. After migration, mypy may not find `zehutai`. | Add `mypy_path = "src"` to `[tool.mypy]` in `pyproject.toml`. |

### Rollback Strategy

1. The migration is a sequence of file moves and import string edits. Git provides full rollback:
   ```bash
   git stash        # if changes are unstaged
   # or
   git revert HEAD  # if committed
   ```
2. Before starting, create a checkpoint commit: `git commit -m "chore: pre-src-layout checkpoint"`.
3. After each step (see section 10), run `python -m pytest -m "not slow"` to verify no regressions before proceeding to the next step.
4. If tests break mid-migration, `git stash` or `git checkout -- .` restores the previous working state.

---

## 10. Execution Order (Step-by-Step)

Execute steps in this order to minimize time spent in a broken state. Each step ends with a test run to verify.

### Step 1 — Prepare `pyproject.toml` for src layout
1. Add `[tool.setuptools.package-dir]` and `[tool.setuptools.packages.find]` sections.
2. Change `pythonpath` in `[tool.pytest.ini_options]` from `[".", "YehoshuaSimilarityComparisons"]` to `["src"]`.
3. Add `mypy_path = "src"` to `[tool.mypy]`.
4. **Do NOT run tests yet** (the `src/` directory does not exist yet).

---

### Step 2 — Create `src/zehutai/` directory tree and `__init__.py` files
Create the following empty/stub files:
```
src/zehutai/__init__.py
src/zehutai/similarity/__init__.py
src/zehutai/rag/__init__.py
src/zehutai/evaluation/__init__.py
src/zehutai/hebrew/__init__.py
```
Also create `scripts/` directory.

---

### Step 3 — Move source files (do NOT delete originals yet)

Copy (not move) files to new locations first, then update their internal imports:

3a. Copy `embeddings_comparison.py` → `src/zehutai/embeddings_comparison.py`
  - No import changes needed in this file.

3b. Copy `YehoshuaSimilarityComparisons/sim.py` → `src/zehutai/similarity/sim.py`
  - No import changes needed in this file.

3c. Copy `YehoshuaSimilarityComparisons/evaluation.py` → `src/zehutai/evaluation/evaluation.py`
  - No import changes needed in this file.

3d. Copy `YehoshuaSimilarityComparisons/hebrew_benchmark.py` → `src/zehutai/hebrew/hebrew_benchmark.py`
  - No import changes needed in this file.

3e. Copy `YehoshuaSimilarityComparisons/rag.py` → `src/zehutai/rag/rag.py`
  - **Update import:** `from embeddings_comparison import compare_sentences`
    → `from zehutai.embeddings_comparison import compare_sentences`

3f. Copy `YehoshuaSimilarityComparisons/plotting.py` → `src/zehutai/similarity/plotting.py`
  - **Update import:** `from sim import Similarity`
    → `from zehutai.similarity.sim import Similarity`

3g. Copy `YehoshuaSimilarityComparisons/main.py` → `src/zehutai/cli.py`
  - **Update import:** `from sim import Similarity`
    → `from zehutai.similarity.sim import Similarity`

3h. Copy `YehoshuaSimilarityComparisons/simple_rag.py` → `scripts/simple_rag.py`
  - No import changes needed.

3i. Copy `YehoshuaSimilarityComparisons/ragtest.py` → `scripts/ragtest.py`
  - No import changes needed.

---

### Step 4 — Fill in `__init__.py` content
Populate each `__init__.py` with the re-exports described in section 4.10–4.14.

---

### Step 5 — Install the package in editable mode
```bash
pip install -e ".[dev]"
```
This makes `import zehutai` work from the `src/` directory.

---

### Step 6 — Update all test files

6a. Update `tests/conftest.py`:
  - Replace old `sys.path` manipulation with `src`-based path.
  - Update `import embeddings_comparison as ec` → `import zehutai.embeddings_comparison as ec`.

6b. Update `tests/test_embeddings_comparison.py`:
  - `import embeddings_comparison as ec` → `import zehutai.embeddings_comparison as ec`.

6c. Update `tests/test_rag.py`:
  - Update all `from rag import ...` → `from zehutai.rag.rag import ...`.
  - Update all `@patch("rag....")` → `@patch("zehutai.rag.rag....")`.
  - Update all `import rag` inside test methods → `from zehutai.rag import rag`.

6d. Update `tests/test_similarity.py`:
  - `from sim import ...` → `from zehutai.similarity.sim import ...`.

6e. Update `tests/test_evaluation.py`:
  - `from YehoshuaSimilarityComparisons.evaluation import ...` → `from zehutai.evaluation.evaluation import ...`.

6f. Update `tests/test_hebrew_benchmark.py`:
  - Remove `sys.path` manipulation block.
  - `from hebrew_benchmark import ...` → `from zehutai.hebrew.hebrew_benchmark import ...`.

6g. Update `tests/test_hebrew_model_benchmark.py`:
  - `import embeddings_comparison as ec` → `import zehutai.embeddings_comparison as ec`.

---

### Step 7 — Run full test suite
```bash
python -m pytest -m "not slow" -v
```
**Expected:** All tests pass. If any fail, diagnose by comparing import paths and `@patch` target strings before proceeding.

---

### Step 8 — Remove old source files
Only after step 7 passes:

8a. Delete `embeddings_comparison.py` (root) — OR replace with backward-compat shim (see section 8.1).
8b. Delete `YehoshuaSimilarityComparisons/embeddings_comparison.py` (was already a shim).
8c. Delete `YehoshuaSimilarityComparisons/sim.py`.
8d. Delete `YehoshuaSimilarityComparisons/rag.py`.
8e. Delete `YehoshuaSimilarityComparisons/evaluation.py`.
8f. Delete `YehoshuaSimilarityComparisons/hebrew_benchmark.py`.
8g. Delete `YehoshuaSimilarityComparisons/plotting.py`.
8h. Delete `YehoshuaSimilarityComparisons/main.py`.
8i. Delete `YehoshuaSimilarityComparisons/ragtest.py`.
8j. Delete `YehoshuaSimilarityComparisons/simple_rag.py`.
8k. Delete `YehoshuaSimilarityComparisons/__init__.py`.
8l. Remove the now-empty `YehoshuaSimilarityComparisons/` directory.

---

### Step 9 — Update CI workflow
Update `.github/workflows/ci.yml` as described in section 7.

---

### Step 10 — Run tests again and check coverage
```bash
python -m pytest -m "not slow" --cov=zehutai --cov-report=term-missing -v
```
Verify coverage is at or above 70%.

---

### Step 11 — Run linter
```bash
ruff check . --output-format=github
ruff format --check .
```
Fix any new linting issues introduced by the migration.

---

### Step 12 — Final verification
```bash
python -m mypy src/zehutai/
python -m pytest -m "not slow" -v
```
Commit with message: `refactor: migrate to src/ layout under src/zehutai/`

---

## 11. Summary of All Changed Files

| File | Action | Key Change |
|------|--------|------------|
| `pyproject.toml` | Modify | Add src layout config, update pythonpath, add entry point, add mypy_path |
| `src/zehutai/__init__.py` | Create | Public API re-exports |
| `src/zehutai/embeddings_comparison.py` | Create (moved from root) | No import changes |
| `src/zehutai/cli.py` | Create (moved from `main.py`) | `from sim import` → `from zehutai.similarity.sim import` |
| `src/zehutai/similarity/__init__.py` | Create | Re-export Similarity |
| `src/zehutai/similarity/sim.py` | Create (moved) | No import changes |
| `src/zehutai/similarity/plotting.py` | Create (moved) | `from sim import` → `from zehutai.similarity.sim import` |
| `src/zehutai/rag/__init__.py` | Create | Re-export RAG functions |
| `src/zehutai/rag/rag.py` | Create (moved) | `from embeddings_comparison import` → `from zehutai.embeddings_comparison import` |
| `src/zehutai/evaluation/__init__.py` | Create | Re-export metrics |
| `src/zehutai/evaluation/evaluation.py` | Create (moved) | No import changes |
| `src/zehutai/hebrew/__init__.py` | Create | Re-export benchmark API |
| `src/zehutai/hebrew/hebrew_benchmark.py` | Create (moved) | No import changes |
| `scripts/simple_rag.py` | Create (moved) | No import changes |
| `scripts/ragtest.py` | Create (moved) | No import changes |
| `tests/conftest.py` | Modify | sys.path: root+sub → src; import ec path |
| `tests/test_embeddings_comparison.py` | Modify | `import embeddings_comparison as ec` → `import zehutai.embeddings_comparison as ec` |
| `tests/test_rag.py` | Modify | from/import + all @patch strings |
| `tests/test_similarity.py` | Modify | `from sim import` → `from zehutai.similarity.sim import` |
| `tests/test_evaluation.py` | Modify | `from YehoshuaSimilarityComparisons.evaluation import` → `from zehutai.evaluation.evaluation import` |
| `tests/test_hebrew_benchmark.py` | Modify | Remove sys.path block; update import |
| `tests/test_hebrew_model_benchmark.py` | Modify | `import embeddings_comparison as ec` → `import zehutai.embeddings_comparison as ec` |
| `.github/workflows/ci.yml` | Modify | editable install; --cov=zehutai |
| `embeddings_comparison.py` (root) | Delete (or shim) | After tests pass |
| `YehoshuaSimilarityComparisons/` (entire dir) | Delete | After tests pass |
