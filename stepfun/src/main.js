import { createEmbedder, readUrlOptions, MODEL_ID } from "./embedder.js";

const MODEL_NAME = MODEL_ID.split("/").pop();

const $ = (id) => document.getElementById(id);
const api = async (path, body) => {
  const res = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: { "content-type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
};

let embedder = null;
let docSeq = 0;

function setBadges({ device, dtype, count }) {
  if (device) $("badge-backend").textContent = `推理后端：${device.toUpperCase()}`;
  if (dtype) $("badge-model").textContent = `模型：${MODEL_NAME} (${dtype})`;
  if (typeof count === "number") $("badge-docs").textContent = `文档数：${count}`;
}

function chunkText(text) {
  return text
    .split(/\n\s*\n/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

async function boot() {
  try {
    const health = await api("/api/health");
    setBadges({ count: health.count });
  } catch (e) {
    $("badge-docs").textContent = "seekdb 未连接";
    $("import-status").textContent = `seekdb 连接失败：${e.message}`;
  }

  $("import-status").textContent = "模型加载中…";
  const t0 = performance.now();
  const opts = readUrlOptions();
  embedder = await createEmbedder({
    dtype: opts.dtype,
    device: opts.backend ?? undefined,
    onProgress: ({ file, pct }) => {
      $("import-status").textContent = `模型下载 ${file} ${pct.toFixed(0)}%`;
    },
  });
  const ms = Math.round(performance.now() - t0);
  setBadges({ device: embedder.device, dtype: embedder.dtype });
  $("import-status").textContent = `模型就绪（${ms} ms，后端 ${embedder.device}）`;
  $("btn-import").disabled = false;
  $("btn-search").disabled = false;
  if (embedder.device !== "webgpu") {
    $("badge-backend").title = "当前环境不支持 WebGPU，已自动降级到 WASM";
  }
}

async function doImport() {
  const btn = $("btn-import");
  btn.disabled = true;
  const t0 = performance.now();
  try {
    let texts = chunkText($("import-text").value);
    const files = $("import-file").files;
    for (const f of files) {
      const content = await f.text();
      texts = texts.concat(chunkText(content).map((t) => t));
    }
    if (!texts.length) throw new Error("没有可导入的文本");
    $("import-status").textContent = `向量化 ${texts.length} 段…`;
    const vectors = await embedder.embed(texts);
    const docs = texts.map((text, i) => ({
      id: `d${Date.now()}_${docSeq++}_${i}`,
      text,
      meta: { source: files.length ? "file" : "paste" },
    }));
    const r = await api("/api/ingest", { docs, vectors });
    const ms = Math.round(performance.now() - t0);
    $("import-status").textContent = `已写入 ${r.inserted} 段，库内共 ${r.count} 段（${ms} ms）`;
    setBadges({ count: r.count });
  } catch (e) {
    $("import-status").textContent = `导入失败：${e.message}`;
  } finally {
    btn.disabled = false;
  }
}

async function doSearch() {
  const q = $("query").value.trim();
  if (!q) return;
  $("btn-search").disabled = true;
  const t0 = performance.now();
  try {
    const [vec] = await embedder.embed([q]);
    const r = await api("/api/search", { queryText: q, queryVector: vec, topK: 5 });
    const tbody = $("results").querySelector("tbody");
    tbody.innerHTML = "";
    r.results.forEach((row, i) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${i + 1}</td><td></td><td>${row.distance?.toFixed?.(4) ?? "-"}</td><td></td>`;
      tr.children[1].textContent = row.text ?? "";
      tr.children[3].textContent = row.meta?.source ?? "-";
      tbody.appendChild(tr);
    });
    const ms = Math.round(performance.now() - t0);
    $("search-status").textContent = `${r.mode === "hybrid" ? "向量 + 全文混合" : "纯向量"}检索，${r.results.length} 条结果（${ms} ms）`;
    $("foot-timing").textContent = `本地推理 + seekdb 检索端到端 ${ms} ms`;
  } catch (e) {
    $("search-status").textContent = `检索失败：${e.message}`;
  } finally {
    $("btn-search").disabled = false;
  }
}

async function doReset() {
  try {
    const r = await api("/api/reset", {});
    $("import-status").textContent = `已清空，当前 ${r.count} 段`;
    setBadges({ count: r.count });
  } catch (e) {
    $("import-status").textContent = `清空失败：${e.message}`;
  }
}

$("btn-import").addEventListener("click", doImport);
$("btn-search").addEventListener("click", doSearch);
$("btn-reset").addEventListener("click", doReset);
$("query").addEventListener("keydown", (e) => e.key === "Enter" && doSearch());

boot();
