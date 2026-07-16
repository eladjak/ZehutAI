/* ZehutAI web UI — vanilla JS, RTL Hebrew */
"use strict";

const API = "/zehutai/api";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));

const MODEL_NAMES = {
  mpnet: ["MPNet רב־לשוני", "ראשי · עברית"],
  tfidf: ["TF-IDF", "חפיפת מילים"],
  bert: ["BERT", "אנגלית · מחקרי"],
  roberta: ["RoBERTa", "אנגלית · מחקרי"],
  nltk: ["Doc2Vec", "אימון על האוסף"],
};

function level(score) {
  if (score >= 0.65) return ["high", "דמיון גבוה"];
  if (score >= 0.4) return ["medium", "דמיון בינוני"];
  return ["low", "דמיון נמוך"];
}

async function api(path, opts = {}) {
  const res = await fetch(API + path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  const data = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, data };
}

/* ---------------- status / model chips ---------------- */

let statusTimer = null;

async function refreshStatus() {
  try {
    const { data } = await api("/status");
    const models = data.models || {};
    $$(".chip").forEach((chip) => {
      const m = chip.dataset.model;
      chip.classList.remove("ready", "loading", "error");
      const st = models[m];
      if (st === "ready") chip.classList.add("ready");
      else if (st === "loading") chip.classList.add("loading");
      else if (st === "error") chip.classList.add("error");
    });
    if (data.mem_available_mb != null) {
      $("#footMem").textContent = `זיכרון פנוי בשרת: ${(data.mem_available_mb / 1024).toFixed(1)} GB`;
    }
    const anyLoading = Object.values(models).some((s) => s === "loading");
    clearTimeout(statusTimer);
    statusTimer = setTimeout(refreshStatus, anyLoading ? 3000 : 20000);
    return models;
  } catch {
    clearTimeout(statusTimer);
    statusTimer = setTimeout(refreshStatus, 8000);
    return {};
  }
}

function waitForModel(model, timeoutMs = 240000) {
  return new Promise((resolve, reject) => {
    const t0 = Date.now();
    const tick = async () => {
      const models = await refreshStatus();
      if (models[model] === "ready") return resolve();
      if (models[model] === "error") return reject(new Error("טעינת המודל נכשלה"));
      if (Date.now() - t0 > timeoutMs) return reject(new Error("טעינת המודל לוקחת יותר מדי זמן — נסו שוב עוד רגע"));
      setTimeout(tick, 3000);
    };
    tick();
  });
}

/* ---------------- tabs ---------------- */

$$(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    $$(".tab").forEach((t) => {
      t.classList.toggle("active", t === tab);
      t.setAttribute("aria-selected", t === tab ? "true" : "false");
    });
    $$(".screen").forEach((s) => s.classList.remove("active"));
    $("#screen-" + tab.dataset.screen).classList.add("active");
  });
});

/* ---------------- compare (hero) ---------------- */

$("#swapBtn").addEventListener("click", () => {
  const a = $("#cmpA"), b = $("#cmpB");
  [a.value, b.value] = [b.value, a.value];
});

function showError(afterEl, msg) {
  const old = afterEl.parentElement.querySelector(".err");
  if (old) old.remove();
  const div = document.createElement("div");
  div.className = "err";
  div.textContent = msg;
  afterEl.insertAdjacentElement("afterend", div);
}

function clearError(container) {
  const old = container.querySelector(".err");
  if (old) old.remove();
}

function setGauge(score) {
  const clamped = Math.max(0, Math.min(1, score));
  const pct = Math.round(clamped * 100);
  const [lvl, verdict] = level(clamped);
  const fill = $("#gaugeFill");
  fill.style.strokeDashoffset = String(283 * (1 - clamped));
  fill.classList.remove("lvl-high", "lvl-medium", "lvl-low");
  fill.classList.add("lvl-" + lvl);
  const v = $("#gaugeVerdict");
  v.textContent = verdict;
  v.className = "gauge-verdict lvl-" + lvl;
  // count-up animation for the number
  const numEl = $("#gaugeNum");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduced) { numEl.textContent = String(pct); return; }
  const t0 = performance.now();
  const dur = 650;
  const step = (t) => {
    const p = Math.min(1, (t - t0) / dur);
    numEl.textContent = String(Math.round(pct * (1 - Math.pow(1 - p, 3))));
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function renderModelBars(results) {
  const wrap = $("#modelBars");
  wrap.innerHTML = "";
  for (const [model, r] of Object.entries(results)) {
    const [name, tag] = MODEL_NAMES[model] || [model, ""];
    const row = document.createElement("div");
    row.className = "mbar";
    if (r.status === "ok") {
      const clamped = Math.max(0, Math.min(1, r.score));
      const [lvl] = level(clamped);
      row.innerHTML = `
        <div class="mbar-head">
          <span class="mname">${esc(name)} <em>${esc(tag)}</em></span>
          <span class="mscore">${r.score.toFixed(3)}</span>
        </div>
        <div class="track"><div class="fill lvl-${lvl}"></div></div>`;
      wrap.appendChild(row);
      const fill = row.querySelector(".fill");
      requestAnimationFrame(() =>
        requestAnimationFrame(() => { fill.style.transform = `scaleX(${clamped})`; })
      );
    } else if (r.status === "loading") {
      row.innerHTML = `
        <div class="mbar-head">
          <span class="mname">${esc(name)} <em>${esc(tag)}</em></span>
          <span class="mscore pending">טוען מודל…</span>
        </div>
        <div class="track"><div class="fill"></div></div>`;
      wrap.appendChild(row);
    } else {
      row.innerHTML = `
        <div class="mbar-head">
          <span class="mname">${esc(name)} <em>${esc(tag)}</em></span>
          <span class="mscore pending">שגיאה</span>
        </div>
        <div class="track"><div class="fill"></div></div>`;
      wrap.appendChild(row);
    }
  }
}

let compareBusy = false;

async function runCompare(retries = 40) {
  if (compareBusy) return;
  const text1 = $("#cmpA").value.trim();
  const text2 = $("#cmpB").value.trim();
  const btn = $("#compareBtn");
  clearError($("#screen-compare"));
  if (!text1 || !text2) {
    showError(btn, "צריך למלא את שני המשפטים.");
    return;
  }
  const models = ["mpnet", ...$$("#screen-compare .pick input:not([disabled])")
    .filter((i) => i.checked).map((i) => i.dataset.cm)];

  compareBusy = true;
  btn.disabled = true;
  $("#compareLoading").hidden = false;
  $("#compareResults").hidden = true;

  try {
    let attempt = 0;
    while (true) {
      const { ok, data } = await api("/compare", {
        method: "POST",
        body: JSON.stringify({ text1, text2, models }),
      });
      if (!ok) throw new Error(data.error || "השרת החזיר שגיאה");
      const results = data.results || {};
      const loadingModels = Object.entries(results)
        .filter(([, r]) => r.status === "loading").map(([m]) => m);
      if (loadingModels.length === 0 || attempt >= retries) {
        $("#compareLoading").hidden = true;
        $("#compareResults").hidden = false;
        if (results.mpnet && results.mpnet.status === "ok") setGauge(results.mpnet.score);
        renderModelBars(results);
        break;
      }
      attempt += 1;
      $("#compareLoadingText").textContent =
        `טוען את מודל ${loadingModels.map((m) => (MODEL_NAMES[m] || [m])[0]).join(", ")} בפעם הראשונה — עד דקה־שתיים…`;
      refreshStatus();
      await new Promise((r) => setTimeout(r, 3500));
    }
  } catch (e) {
    $("#compareLoading").hidden = true;
    showError(btn, "משהו השתבש: " + e.message);
  } finally {
    compareBusy = false;
    btn.disabled = false;
    $("#compareLoadingText").textContent = "מחשב על המעבד… זה לוקח כמה שניות";
  }
}

$("#compareBtn").addEventListener("click", () => runCompare());

/* ---------------- rank ---------------- */

function renderRankList(el, items) {
  el.innerHTML = "";
  el.hidden = false;
  const maxScore = Math.max(0.02, ...items.map((it) => it.score ?? it.cosine ?? 0));
  items.forEach((it, idx) => {
    const li = document.createElement("li");
    li.className = "rank-item" + (idx === 0 ? " top" : "");
    const score = it.score ?? it.cosine ?? 0;
    const frac = Math.max(0, Math.min(1, score / maxScore));
    const scoreHtml = it.rrf != null
      ? `<span class="rank-score">${score.toFixed(3)}<small>RRF ${it.rrf.toFixed(4)}</small></span>`
      : `<span class="rank-score">${score.toFixed(3)}</span>`;
    li.innerHTML = `
      <span class="rank-num">${idx + 1}</span>
      <div class="rank-body">
        <div class="rank-text">${esc(it.text)}</div>
        <div class="rank-bar"><div class="fill"></div></div>
      </div>
      ${scoreHtml}`;
    el.appendChild(li);
    const fill = li.querySelector(".fill");
    requestAnimationFrame(() =>
      requestAnimationFrame(() => { fill.style.transform = `scaleX(${frac})`; })
    );
  });
}

$("#rankBtn").addEventListener("click", async () => {
  const btn = $("#rankBtn");
  clearError($("#screen-rank"));
  const query = $("#rankQuery").value.trim();
  const texts = $("#rankTexts").value.split("\n").map((t) => t.trim()).filter(Boolean);
  if (!query || texts.length === 0) {
    showError(btn, "צריך שאילתה ולפחות טקסט אחד.");
    return;
  }
  const method = $("#rankMethod").value;
  btn.disabled = true;
  $("#rankLoading").hidden = false;
  $("#rankList").hidden = true;
  try {
    let res = await api("/similarity", {
      method: "POST",
      body: JSON.stringify({ query, texts, method }),
    });
    if (res.status === 503) {
      $("#rankLoadingText").textContent = "טוען את המודל בפעם הראשונה — עד דקה־שתיים…";
      await waitForModel(res.data.model || method);
      res = await api("/similarity", {
        method: "POST",
        body: JSON.stringify({ query, texts, method }),
      });
    }
    if (!res.ok) throw new Error(res.data.error || "השרת החזיר שגיאה");
    renderRankList($("#rankList"), res.data.ranked || []);
  } catch (e) {
    showError(btn, "משהו השתבש: " + e.message);
  } finally {
    btn.disabled = false;
    $("#rankLoading").hidden = true;
    $("#rankLoadingText").textContent = "מדרג…";
  }
});

/* ---------------- rag ---------------- */

$("#ragBtn").addEventListener("click", async () => {
  const btn = $("#ragBtn");
  clearError($("#screen-rag"));
  const query = $("#ragQuery").value.trim();
  if (!query) {
    showError(btn, "צריך שאילתת חיפוש.");
    return;
  }
  const docsRaw = $("#ragDocs").value.split("\n").map((t) => t.trim()).filter(Boolean);
  const topKVal = $("#ragTopK").value;
  const body = { query };
  if (docsRaw.length > 0) body.documents = docsRaw;
  if (topKVal) body.top_k = Number(topKVal);

  btn.disabled = true;
  $("#ragLoading").hidden = false;
  $("#ragList").hidden = true;
  try {
    let res = await api("/rag", { method: "POST", body: JSON.stringify(body) });
    if (res.status === 503) {
      await waitForModel("mpnet");
      res = await api("/rag", { method: "POST", body: JSON.stringify(body) });
    }
    if (!res.ok) throw new Error(res.data.error || "השרת החזיר שגיאה");
    renderRankList($("#ragList"), res.data.results || []);
  } catch (e) {
    showError(btn, "משהו השתבש: " + e.message);
  } finally {
    btn.disabled = false;
    $("#ragLoading").hidden = true;
  }
});

/* ---------------- benchmark ---------------- */

const CAT_HE = { high: "גבוה", medium: "בינוני", low: "נמוך" };

function renderBenchmark(data) {
  const stats = data.stats || {};
  const fmt = (v) => (v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(3));
  $("#benchStats").innerHTML = `
    <div class="stat-card c-high"><div class="v">${fmt(stats.high_avg)}</div><div class="l">ממוצע זוגות "גבוה"</div></div>
    <div class="stat-card c-medium"><div class="v">${fmt(stats.medium_avg)}</div><div class="l">ממוצע זוגות "בינוני"</div></div>
    <div class="stat-card c-low"><div class="v">${fmt(stats.low_avg)}</div><div class="l">ממוצע זוגות "נמוך"</div></div>
    <div class="stat-card c-sep"><div class="v">${fmt(stats.separation_score)}</div><div class="l">ציון הפרדה (גבוה − נמוך)</div></div>`;

  const wrap = $("#benchPairs");
  wrap.innerHTML = "";
  (data.pairs || []).forEach((p) => {
    const row = document.createElement("div");
    row.className = "bpair " + (p.category || "");
    const clamped = Math.max(0, Math.min(1, p.score));
    row.innerHTML = `
      <div class="cat" title="${esc(CAT_HE[p.category] || p.category)}"></div>
      <div class="sents">
        <div>${esc(p.sent1)}</div>
        <div>${esc(p.sent2)}</div>
      </div>
      <div class="track"><div class="fill"></div></div>
      <div class="score">${Number(p.score).toFixed(3)}</div>`;
    wrap.appendChild(row);
    const fill = row.querySelector(".fill");
    requestAnimationFrame(() =>
      requestAnimationFrame(() => { fill.style.transform = `scaleX(${clamped})`; })
    );
  });

  $("#benchResults").hidden = false;
  $("#benchRefreshBtn").hidden = false;
  $("#benchMeta").textContent = data.cached
    ? `תוצאה שמורה מ־${data.ts || ""}`
    : `הריצה הסתיימה ב־${((data.ran_ms || 0) / 1000).toFixed(1)} שניות`;
}

async function runBenchmark(refresh = false) {
  const btn = $("#benchBtn");
  clearError($("#screen-benchmark"));
  btn.disabled = true;
  $("#benchRefreshBtn").disabled = true;
  $("#benchLoading").hidden = false;
  try {
    let res = await api("/benchmark" + (refresh ? "?refresh=1" : ""));
    if (res.status === 503) {
      await waitForModel("mpnet");
      res = await api("/benchmark" + (refresh ? "?refresh=1" : ""));
    }
    if (!res.ok) throw new Error(res.data.error || "השרת החזיר שגיאה");
    renderBenchmark(res.data);
  } catch (e) {
    showError(btn, "משהו השתבש: " + e.message);
  } finally {
    btn.disabled = false;
    $("#benchRefreshBtn").disabled = false;
    $("#benchLoading").hidden = true;
  }
}

$("#benchBtn").addEventListener("click", () => runBenchmark(false));
$("#benchRefreshBtn").addEventListener("click", () => runBenchmark(true));

/* ---------------- init ---------------- */

refreshStatus();
