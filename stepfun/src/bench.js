import { createEmbedder, measureModelBytes } from "./embedder.js";

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

const DOCS = [
  { id: "d1", text: "seekdb 是面向 AI 的向量数据库，支持向量、全文与标量过滤的混合检索。" },
  { id: "d2", text: "OceanBase 是分布式关系数据库，兼容 MySQL 协议，支持多租户。" },
  { id: "d3", text: "all-MiniLM-L6-v2 是一个 384 维的轻量句向量模型，适合端侧运行。" },
  { id: "d4", text: "WebGPU 可以在浏览器中调用显卡做矩阵运算，比 WASM 快数倍。" },
  { id: "d5", text: "杭州西湖的断桥残雪是冬季经典景点，秋天适合环湖步行。" },
  { id: "d6", text: "HNSW 是一种近似最近邻图索引，用少量召回损失换取查询速度。" },
  { id: "d7", text: "INT8 量化把权重压缩到 8 位整数，体积约为 fp32 的四分之一。" },
  { id: "d8", text: "Vite 是前端构建工具，开发时基于原生 ESM 提供热更新。" },
  { id: "d9", text: "全文索引使用 IK 分词器，中文按词典切分后建立倒排。" },
  { id: "d10", text: "向量检索与全文检索通过 RRF 融合，可以兼顾语义与关键词命中。" },
  { id: "d11", text: "Docker Compose 用一个 YAML 描述多容器应用，便于本地复现。" },
  { id: "d12", text: "浏览器端 embedding 不会把原文发送到云端，隐私更好。" },
];

const QUERIES = [
  { q: "哪个模型适合在浏览器里做句向量", expect: ["d3"] },
  { q: "怎么把检索结果用关键词和语义一起排", expect: ["d10", "d1"] },
  { q: "显卡加速比 CPU 快吗", expect: ["d4"] },
  { q: "模型压到 8 位体积变多少", expect: ["d7"] },
  { q: "中文分词用什么做全文索引", expect: ["d9"] },
  { q: "杭州秋天去哪里逛", expect: ["d5"] },
];

const CONFIGS = [
  { dtype: "fp32", device: "webgpu" },
  { dtype: "fp16", device: "webgpu" },
  { dtype: "q8", device: "webgpu" },
  { dtype: "fp32", device: "wasm" },
  { dtype: "fp16", device: "wasm" },
  { dtype: "q8", device: "wasm" },
];

const median = (xs) => {
  const a = [...xs].sort((x, y) => x - y);
  if (!a.length) return 0;
  const m = Math.floor(a.length / 2);
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
};

function score(ranked, expect) {
  const set = new Set(expect);
  const top3 = ranked.slice(0, 3);
  const hits = top3.filter((id) => set.has(id));
  const recall = hits.length / set.size;
  let dcg = 0;
  top3.forEach((id, i) => {
    if (set.has(id)) dcg += 1 / Math.log2(i + 2);
  });
  const ideal = Array.from({ length: Math.min(3, set.size) }).reduce((s, _, i) => s + 1 / Math.log2(i + 2), 0);
  return { recall, ndcg: ideal ? dcg / ideal : 0 };
}

const rows = [];
let bytes = {};

async function runConfig({ dtype, device }) {
  const t0 = performance.now();
  const emb = await createEmbedder({ dtype, device });
  const loadMs = performance.now() - t0;

  await api("/api/reset", {});

  const tEmbed = performance.now();
  const docVectors = await emb.embed(DOCS.map((d) => d.text));
  const embedMs = performance.now() - tEmbed;

  const tIngest = performance.now();
  await api("/api/ingest", {
    docs: DOCS.map((d) => ({ id: d.id, text: d.text, meta: { src: "bench" } })),
    vectors: docVectors,
  });
  const ingestMs = performance.now() - tIngest;

  const queryEmbed = [];
  const searchMs = [];
  const e2e = [];
  const recalls = [];
  const ndcgs = [];
  const vecRecalls = [];
  const vecNdcgs = [];
  const rankings = [];

  for (const item of QUERIES) {
    const q0 = performance.now();
    const [vec] = await emb.embed([item.q]);
    const q1 = performance.now();
    const r = await api("/api/search", { queryText: item.q, queryVector: vec, topK: 3 });
    const q2 = performance.now();
    queryEmbed.push(q1 - q0);
    searchMs.push(q2 - q1);
    e2e.push(q2 - q0);
    const ranked = r.results.map((x) => x.id);
    const s = score(ranked, item.expect);
    recalls.push(s.recall);
    ndcgs.push(s.ndcg);

    // 纯向量检索：不受关键词匹配影响，能体现量化带来的语义质量变化
    const rv = await api("/api/search", { queryText: "", queryVector: vec, topK: 3 });
    const rankedVec = rv.results.map((x) => x.id);
    rankings.push({ q: item.q, hybrid: ranked, vector: rankedVec });
    const sv = score(rankedVec, item.expect);
    vecRecalls.push(sv.recall);
    vecNdcgs.push(sv.ndcg);
  }

  await emb.dispose?.();

  return {
    config: `${dtype} + ${device}`,
    backend: emb.device,
    dtype: emb.dtype,
    bytes: bytes[dtype] ?? null,
    loadMs: Math.round(loadMs),
    embedMsPerDoc: +(embedMs / DOCS.length).toFixed(1),
    ingestMs: Math.round(ingestMs),
    queryEmbedMs: +median(queryEmbed).toFixed(1),
    searchMs: +median(searchMs).toFixed(1),
    e2eMs: +median(e2e).toFixed(1),
    recall3: +(recalls.reduce((a, b) => a + b, 0) / recalls.length).toFixed(3),
    ndcg3: +(ndcgs.reduce((a, b) => a + b, 0) / ndcgs.length).toFixed(3),
    vecRecall3: +(vecRecalls.reduce((a, b) => a + b, 0) / vecRecalls.length).toFixed(3),
    vecNdcg3: +(vecNdcgs.reduce((a, b) => a + b, 0) / vecNdcgs.length).toFixed(3),
    rankings,
  };
}

function fmtBytes(n) {
  if (!n) return "未知";
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function render() {
  const tbody = document.querySelector("#bench-table tbody");
  tbody.innerHTML = "";
  const base = rows.find((r) => r.config === "fp32 + webgpu")?.bytes || rows.find((r) => r.dtype === "fp32")?.bytes;
  for (const r of rows) {
    const tr = document.createElement("tr");
    const ratio = r.bytes && base ? `${(base / r.bytes).toFixed(2)}×` : "-";
    const cells = [
      r.config,
      r.backend,
      `${fmtBytes(r.bytes)}${r.bytes && base ? ` (${ratio})` : ""}`,
      r.loadMs,
      r.embedMsPerDoc,
      r.ingestMs,
      r.queryEmbedMs,
      r.searchMs,
      r.e2eMs,
      r.recall3,
      r.ndcg3,
      r.vecRecall3,
      r.vecNdcg3,
    ];
    for (const c of cells) {
      const td = document.createElement("td");
      td.textContent = String(c);
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
}

async function runAll() {
  const status = document.getElementById("bench-status");
  document.getElementById("btn-run").disabled = true;
  rows.length = 0;
  for (const dtype of ["fp32", "fp16", "q8"]) {
    bytes[dtype] = await measureModelBytes(dtype);
  }
  for (const cfg of CONFIGS) {
    status.textContent = `运行中：${cfg.dtype} + ${cfg.device}…`;
    try {
      const r = await runConfig(cfg);
      rows.push(r);
      render();
    } catch (e) {
      status.textContent = `${cfg.dtype} + ${cfg.device} 失败：${e.message}`;
    }
  }
  status.textContent = "全部完成";
  document.getElementById("btn-run").disabled = false;
  window.__benchRows = rows;
}

document.getElementById("btn-run").addEventListener("click", runAll);
document.getElementById("btn-json").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify({ model: "Xenova/all-MiniLM-L6-v2", docs: DOCS.length, queries: QUERIES.length, bytes, rows }, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "bench.json";
  a.click();
});

if (new URLSearchParams(location.search).get("auto") === "1") runAll();
