"""ZehutAI Web — FastAPI service wrapping the existing research engine.

שכבת web דקה מעל מנוע המחקר הקיים (src/zehutai). לא משכתבת את המנוע —
מייבאת את הפונקציות שלו כפי שהן. משרתת גם את ה-frontend הסטטי.

Routes (all under /zehutai, token-gated except bare /health):
    GET  /health                     -- local liveness (no auth, not tunneled)
    GET  /zehutai                    -- the SPA
    GET  /zehutai/api/status         -- model load states + RAM
    POST /zehutai/api/models/load    -- kick off background model load
    POST /zehutai/api/compare        -- two texts -> per-model similarity
    POST /zehutai/api/similarity     -- query + corpus -> ranked list
    POST /zehutai/api/rag            -- query + docs -> vector search + RRF
    GET  /zehutai/api/benchmark      -- 20-pair Hebrew benchmark (cached)

Auth: single bearer token (env ZEHUTAI_TOKEN). Accept ?k=<token> once ->
sets HttpOnly cookie `zehutk` (Path=/zehutai) and redirects to clean URL.

Design notes:
- Engine model caches are module-level; run with ONE uvicorn worker.
- Heavy inference is serialized with a lock (single user; avoids RAM spikes).
- mpnet loads eagerly in a background thread at startup; BERT/RoBERTa are
  lazy-loaded on first request (English research models -- secondary).
- DictaLM query expansion (generate_queries) is intentionally NOT exposed:
  a 7B causal LM is not practical on this CPU-only host. vector_search+RRF
  (which use mpnet) cover the RAG screen.
- vector_search() compares the dict KEYS to the query, so we pass
  {text: text} mappings (texts as their own IDs).
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import sys
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paths / engine import
# ---------------------------------------------------------------------------

WEBAPP_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEBAPP_DIR.parent
SRC_DIR = REPO_ROOT / "src"
STATIC_DIR = WEBAPP_DIR / "static"
BENCHMARK_CACHE = Path(os.environ.get("ZEHUTAI_DATA_DIR", str(WEBAPP_DIR))) / "benchmark_cache.json"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("zehutai-web")

TOKEN = os.environ.get("ZEHUTAI_TOKEN", "")
COOKIE_NAME = "zehutk"

app = FastAPI(title="ZehutAI Web", docs_url=None, redoc_url=None, openapi_url=None)

# ---------------------------------------------------------------------------
# Model state management
# ---------------------------------------------------------------------------

# tfidf/nltk need no pretrained download -> always "ready"
MODEL_STATES: dict[str, str] = {
    "mpnet": "idle",
    "bert": "idle",
    "roberta": "idle",
    "tfidf": "ready",
    "nltk": "ready",
}
MODEL_ERRORS: dict[str, str] = {}
_state_lock = threading.Lock()
_compute_lock = threading.Lock()  # serialize heavy CPU inference
_started_at = time.time()

MODEL_LABELS = {
    "mpnet": "Multilingual MPNet (sentence-transformers)",
    "bert": "BERT base (English, research)",
    "roberta": "RoBERTa base (English, research)",
    "tfidf": "TF-IDF (scikit-learn)",
    "nltk": "Doc2Vec (gensim)",
}


def _set_state(model: str, state: str, error: str | None = None) -> None:
    with _state_lock:
        MODEL_STATES[model] = state
        if error:
            MODEL_ERRORS[model] = error


def _get_state(model: str) -> str:
    with _state_lock:
        return MODEL_STATES.get(model, "unknown")


def _load_mpnet() -> None:
    """Load + warm the primary multilingual model (module-level cache)."""
    _set_state("mpnet", "loading")
    try:
        t0 = time.time()
        from zehutai.embeddings_comparison import compare_sentences

        # warm-up encode so the first real request is fast
        compare_sentences(["שלום", "שלום"])
        log.info("mpnet ready in %.1fs", time.time() - t0)
        _set_state("mpnet", "ready")
    except Exception as exc:  # pragma: no cover - startup path
        log.exception("mpnet load failed")
        _set_state("mpnet", "error", str(exc))


def _load_bert() -> None:
    _set_state("bert", "loading")
    try:
        t0 = time.time()
        from zehutai.similarity.sim import Similarity

        Similarity().methodBert(data=["בדיקה"], query="warm up")
        log.info("bert ready in %.1fs", time.time() - t0)
        _set_state("bert", "ready")
    except Exception as exc:
        log.exception("bert load failed")
        _set_state("bert", "error", str(exc))


def _load_roberta() -> None:
    _set_state("roberta", "loading")
    try:
        t0 = time.time()
        from zehutai.similarity.sim import Similarity

        Similarity().methodRoBERTa(data=["בדיקה"], query="warm up")
        log.info("roberta ready in %.1fs", time.time() - t0)
        _set_state("roberta", "ready")
    except Exception as exc:
        log.exception("roberta load failed")
        _set_state("roberta", "error", str(exc))


_LOADERS = {"mpnet": _load_mpnet, "bert": _load_bert, "roberta": _load_roberta}


def _ensure_loading(model: str) -> str:
    """Trigger a background load if the model is idle (or errored -> retry).

    A previous "error" state is usually a transient network failure while
    downloading from the HF hub (e.g. 504) -- each new user action gets one
    fresh retry instead of being stuck on a sticky error.
    """
    with _state_lock:
        state = MODEL_STATES.get(model)
        if state in ("idle", "error"):
            MODEL_STATES[model] = "loading"
            MODEL_ERRORS.pop(model, None)
            threading.Thread(target=_LOADERS[model], daemon=True).start()
            return "loading"
        return state or "unknown"


@app.on_event("startup")
def _startup() -> None:
    if not TOKEN:
        log.warning("ZEHUTAI_TOKEN is not set -- ALL requests will be rejected")
    # Eager-load the hero model in the background; /health stays responsive.
    threading.Thread(target=_load_mpnet, daemon=True).start()


# ---------------------------------------------------------------------------
# Auth middleware (token gate)
# ---------------------------------------------------------------------------

_401_HTML = """<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ZehutAI — נדרשת הרשאה</title>
<style>body{margin:0;min-height:100dvh;display:grid;place-items:center;
background:#0b0f1a;color:#e2e8f0;font-family:'Heebo',system-ui,sans-serif;text-align:center}
.c{padding:2rem}h1{font-size:1.5rem;margin:0 0 .5rem}
p{color:#94a3b8;margin:0}.k{color:#22d3ee}</style></head><body><div class="c">
<h1>🔒 נדרשת הרשאה</h1><p>הקישור חסר את מפתח הגישה.<br>
פתחו את הקישור הפרטי המלא שקיבלתם (עם <span class="k">?k=...</span>).</p>
</div></body></html>"""


def _token_ok(candidate: str | None) -> bool:
    return bool(TOKEN) and bool(candidate) and secrets.compare_digest(candidate, TOKEN)


@app.middleware("http")
async def auth_gate(request: Request, call_next: Any) -> Any:
    path = request.url.path
    if path == "/health":
        return await call_next(request)
    if path == "/zehutai" or path.startswith("/zehutai/"):
        # 1) ?k=<token> -> set cookie, redirect to clean URL
        k = request.query_params.get("k")
        if k is not None and _token_ok(k):
            resp = RedirectResponse(url=path, status_code=302)
            resp.set_cookie(
                COOKIE_NAME,
                TOKEN,
                httponly=True,
                secure=True,
                samesite="lax",
                path="/zehutai",
                max_age=60 * 60 * 24 * 365,
            )
            return resp
        # 2) cookie or bearer header
        authz = request.headers.get("authorization", "")
        bearer = authz[7:] if authz.lower().startswith("bearer ") else None
        if _token_ok(request.cookies.get(COOKIE_NAME)) or _token_ok(bearer):
            return await call_next(request)
        # 3) reject
        if path.startswith("/zehutai/api/"):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return HTMLResponse(_401_HTML, status_code=401)
    return await call_next(request)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

SIM_METHODS = ("mpnet", "tfidf", "nltk", "bert", "roberta")


class CompareRequest(BaseModel):
    text1: str = Field(min_length=1, max_length=5000)
    text2: str = Field(min_length=1, max_length=5000)
    models: list[str] = Field(default=["mpnet", "tfidf"])


class SimilarityRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    texts: list[str] = Field(min_length=1, max_length=50)
    method: str = Field(default="mpnet")


class RagRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    documents: list[str] | None = Field(default=None, max_length=50)
    top_k: int | None = Field(default=None, ge=1, le=50)


class LoadRequest(BaseModel):
    model: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mem_available_mb() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        return None
    return None


def _clean_texts(texts: list[str]) -> list[str]:
    """Strip, drop empties, dedupe preserving order (dict-key mapping needs it)."""
    seen: set[str] = set()
    out: list[str] = []
    for t in texts:
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _compare_one(model: str, text1: str, text2: str) -> float:
    """Similarity between two texts with the given model (engine calls only)."""
    if model == "mpnet":
        from zehutai.embeddings_comparison import compare_sentences

        return float(compare_sentences([text1, text2]))
    from zehutai.similarity.sim import Similarity

    sim = Similarity()
    if model == "tfidf":
        return float(sim.methodScikitlearn(data=[text2], query=text1)[0][1])
    if model == "bert":
        return float(sim.methodBert(data=[text2], query=text1)[0][0])
    if model == "roberta":
        return float(sim.methodRoBERTa(data=[text2], query=text1)[0][0])
    raise ValueError(f"unsupported compare model: {model}")


# ---------------------------------------------------------------------------
# API endpoints (sync def -> FastAPI threadpool; _compute_lock serializes)
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    with _state_lock:
        states = dict(MODEL_STATES)
    return {
        "status": "ok",
        "service": "zehutai-web",
        "uptime_s": round(time.time() - _started_at),
        "models": states,
    }


@app.get("/zehutai/api/status")
def api_status() -> dict[str, Any]:
    with _state_lock:
        states = dict(MODEL_STATES)
        errors = dict(MODEL_ERRORS)
    return {
        "models": states,
        "labels": MODEL_LABELS,
        "errors": errors,
        "mem_available_mb": _mem_available_mb(),
        "uptime_s": round(time.time() - _started_at),
    }


@app.post("/zehutai/api/models/load")
def api_load_model(req: LoadRequest) -> dict[str, Any]:
    if req.model not in _LOADERS:
        return JSONResponse({"error": f"unknown model: {req.model}"}, status_code=400)
    return {"model": req.model, "state": _ensure_loading(req.model)}


@app.post("/zehutai/api/compare")
def api_compare(req: CompareRequest) -> dict[str, Any]:
    text1, text2 = req.text1.strip(), req.text2.strip()
    results: dict[str, dict[str, Any]] = {}
    for model in req.models:
        if model not in ("mpnet", "tfidf", "bert", "roberta"):
            results[model] = {"status": "error", "error": "unsupported model"}
            continue
        state = _ensure_loading(model) if model in _LOADERS else "ready"
        if state == "loading":
            results[model] = {"status": "loading"}
            continue
        if state == "error":
            results[model] = {"status": "error", "error": MODEL_ERRORS.get(model, "")}
            continue
        try:
            t0 = time.time()
            with _compute_lock:
                score = _compare_one(model, text1, text2)
            results[model] = {
                "status": "ok",
                "score": round(score, 6),
                "ms": round((time.time() - t0) * 1000),
            }
        except Exception as exc:
            log.exception("compare failed for %s", model)
            results[model] = {"status": "error", "error": str(exc)}
    return {"results": results}


@app.post("/zehutai/api/similarity")
def api_similarity(req: SimilarityRequest) -> Any:
    method = req.method
    if method not in SIM_METHODS:
        return JSONResponse({"error": f"unknown method: {method}"}, status_code=400)
    if method in _LOADERS:
        state = _ensure_loading(method)
        if state == "loading":
            return JSONResponse({"status": "loading", "model": method}, status_code=503)
        if state == "error":
            return JSONResponse(
                {"status": "error", "error": MODEL_ERRORS.get(method, "")},
                status_code=500,
            )

    query = req.query.strip()
    texts = _clean_texts(req.texts)
    if not texts:
        return JSONResponse({"error": "no valid texts"}, status_code=400)

    t0 = time.time()
    try:
        with _compute_lock:
            ranked: list[dict[str, Any]]
            if method == "mpnet":
                from zehutai.rag.rag import vector_search

                scores = vector_search(query, {t: t for t in texts})
                ranked = [{"text": t, "score": round(s, 6)} for t, s in scores.items()]
            else:
                from zehutai.similarity.sim import Similarity

                sim = Similarity()
                if method == "tfidf":
                    res = sim.methodScikitlearn(data=texts, query=query)
                    pairs = [(t, s) for t, s in res]
                elif method == "nltk":
                    res = sim.methodNLTK(data=texts, query=query)
                    pairs = []
                    for tag, s in res:
                        try:
                            pairs.append((texts[int(tag)], s))
                        except (ValueError, IndexError):
                            pairs.append((str(tag), s))
                elif method == "bert":
                    res = sim.methodBert(data=texts, query=query)
                    pairs = [(t, s) for s, t, _q in res]
                else:  # roberta
                    res = sim.methodRoBERTa(data=texts, query=query)
                    pairs = [(t, s) for s, t, _q in res]
                pairs.sort(key=lambda x: x[1], reverse=True)
                ranked = [{"text": t, "score": round(float(s), 6)} for t, s in pairs]
    except Exception as exc:
        log.exception("similarity failed (%s)", method)
        return JSONResponse({"error": str(exc)}, status_code=500)

    return {
        "method": method,
        "query": query,
        "ranked": ranked,
        "ms": round((time.time() - t0) * 1000),
    }


@app.post("/zehutai/api/rag")
def api_rag(req: RagRequest) -> Any:
    state = _ensure_loading("mpnet")
    if state == "loading":
        return JSONResponse({"status": "loading", "model": "mpnet"}, status_code=503)
    if state == "error":
        return JSONResponse(
            {"status": "error", "error": MODEL_ERRORS.get("mpnet", "")}, status_code=500
        )

    from zehutai.rag.rag import ALL_DOCUMENTS, reciprocal_rank_fusion, vector_search

    query = req.query.strip()
    docs = _clean_texts(req.documents) if req.documents else list(ALL_DOCUMENTS.values())
    if not docs:
        return JSONResponse({"error": "no valid documents"}, status_code=400)

    t0 = time.time()
    try:
        with _compute_lock:
            doc_map = {d: d for d in docs}
            cosine = vector_search(query, doc_map)
            fused = reciprocal_rank_fusion({query: cosine})
    except Exception as exc:
        log.exception("rag failed")
        return JSONResponse({"error": str(exc)}, status_code=500)

    items = [
        {
            "rank": rank,
            "text": doc,
            "rrf": round(score, 6),
            "cosine": round(cosine.get(doc, 0.0), 6),
        }
        for rank, (doc, score) in enumerate(fused.items(), 1)
    ]
    if req.top_k is not None:
        items = items[: req.top_k]
    return {
        "query": query,
        "results": items,
        "n_documents": len(docs),
        "default_corpus": req.documents is None,
        "query_expansion": False,  # DictaLM 7B deferred on CPU-only host
        "ms": round((time.time() - t0) * 1000),
    }


@app.get("/zehutai/api/benchmark")
def api_benchmark(refresh: int = 0) -> Any:
    if not refresh and BENCHMARK_CACHE.exists():
        try:
            cached = json.loads(BENCHMARK_CACHE.read_text(encoding="utf-8"))
            cached["cached"] = True
            return cached
        except (OSError, json.JSONDecodeError):
            pass

    state = _ensure_loading("mpnet")
    if state == "loading":
        return JSONResponse({"status": "loading", "model": "mpnet"}, status_code=503)
    if state == "error":
        return JSONResponse(
            {"status": "error", "error": MODEL_ERRORS.get("mpnet", "")}, status_code=500
        )

    from zehutai.hebrew.hebrew_benchmark import run_benchmark

    t0 = time.time()
    try:
        with _compute_lock:
            summary = run_benchmark()  # new-style: 20 Hebrew pairs via mpnet
    except Exception as exc:
        log.exception("benchmark failed")
        return JSONResponse({"error": str(exc)}, status_code=500)

    payload = {
        "model": summary.get("model"),
        "pairs": summary.get("pairs", []),
        "stats": summary.get("stats", {}),
        "ran_ms": round((time.time() - t0) * 1000),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cached": False,
    }
    try:
        BENCHMARK_CACHE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except OSError:
        log.warning("could not write benchmark cache")
    return payload


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------


@app.get("/zehutai")
@app.get("/zehutai/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/zehutai/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "3980")))
