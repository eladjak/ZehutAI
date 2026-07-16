# ZehutAI Web UI

Hebrew RTL single-page web app for Yehoshua Dalin (single user) on top of the
existing research engine (`src/zehutai`). The engine is imported as-is — no
rewrite, the 288 tests are untouched.

## Architecture

```
Browser (Hebrew RTL SPA, vanilla JS)
   │  https://hub.eladjak.com/zehutai?k=<token>
   ▼
cloudflared ingress  path ^/zehutai(/|$)  →  127.0.0.1:3980
   ▼
FastAPI (webapp/server.py, systemd: zehutai-web.service)
   ▼
src/zehutai engine  (module-level model caches, 1 uvicorn worker)
```

- **Auth**: single bearer token (`ZEHUTAI_TOKEN` in `/etc/zehutai-web.env`).
  `?k=<token>` once → HttpOnly cookie `zehutk` (Path=/zehutai) → clean URL.
  No token → 401. API also accepts `Authorization: Bearer <token>`.
- **Models**: mpnet (paraphrase-multilingual-mpnet-base-v2) eager-loads in a
  background thread at startup; BERT/RoBERTa (English research models)
  lazy-load on first use; TF-IDF/Doc2Vec need no download. A global lock
  serializes heavy inference (single user, CPU-only).
- **Deferred**: DictaLM 2.0 query expansion (`generate_queries`) is NOT
  exposed — a 7B causal LM is impractical on a CPU-only host. The RAG screen
  uses `vector_search` + `reciprocal_rank_fusion` (mpnet-based) which is the
  rest of the pipeline.
- **Benchmark**: the 20-pair Hebrew benchmark runs once and is cached to
  `webapp/benchmark_cache.json` (UI has a "re-run" button).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness (local only, no auth) |
| GET | `/zehutai` | the SPA |
| GET | `/zehutai/api/status` | model states + free RAM |
| POST | `/zehutai/api/models/load` | `{model}` — background load |
| POST | `/zehutai/api/compare` | `{text1,text2,models[]}` → per-model scores |
| POST | `/zehutai/api/similarity` | `{query,texts[],method}` → ranked |
| POST | `/zehutai/api/rag` | `{query,documents?,top_k?}` → cosine + RRF |
| GET | `/zehutai/api/benchmark?refresh=0|1` | Hebrew benchmark (cached) |

## Deploy (Contabo, /opt/zehutai)

```bash
git clone https://github.com/eladjak/ZehutAI.git /opt/zehutai
cd /opt/zehutai && python3.12 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt fastapi "uvicorn[standard]"
.venv/bin/python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
echo "ZEHUTAI_TOKEN=<32-char-token>" > /etc/zehutai-web.env && chmod 600 /etc/zehutai-web.env
cp deploy/zehutai-web.service /etc/systemd/system/
systemctl daemon-reload && systemctl enable --now zehutai-web
```

Then add to `/root/.cloudflared/config.yml` (before the hub catch-all):

```yaml
- hostname: hub.eladjak.com
  path: ^/zehutai(/|$)
  service: http://localhost:3980
```

`cloudflared tunnel ingress validate && systemctl restart cloudflared-tunnel`

## Assumptions for Yehoshua to validate (v1 scoping)

1. **Sentence compare is the hero screen** — primary-use answers weren't
   given, so v1 exposes the tool's four capabilities with compare first.
2. Single user, one private tokened link — no accounts.
3. BERT/RoBERTa are the original English research models (`bert-base-uncased`,
   `roberta-base`): on Hebrew input their scores are research-grade curiosities,
   and the UI labels them "אנגלית · מחקרי".
4. Doc2Vec appears only on the ranking screen (training on a 2-text corpus is
   degenerate, so it's excluded from compare).
5. RAG default corpus (when no documents are pasted) is the engine's built-in
   10-doc English climate set; pasting Hebrew documents works fully.
6. Similarity thresholds in the UI (גבוה ≥ 0.65 / בינוני ≥ 0.40) were derived
   from the Hebrew benchmark's separation stats — adjustable.
