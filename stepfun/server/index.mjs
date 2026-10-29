import http from "node:http";
import { SeekdbClient, Schema, FulltextIndexConfig, VectorIndexConfig } from "seekdb";

const PORT = Number(process.env.PORT || 8787);
const DIM = Number(process.env.EMBED_DIM || 384);
const COLLECTION = process.env.COLLECTION || "kb";
const SEEKDB = {
  host: process.env.SEEKDB_HOST || "127.0.0.1",
  port: Number(process.env.SEEKDB_PORT || 2881),
  user: "root",
  password: "",
  database: "test",
};

// 向量全部由浏览器端计算并随请求提交；SDK 仍要求注册一个 embedding function，
// 这里注册一个只声明维度、不参与计算的占位实现。
const CLIENT_EMBED = {
  name: "client-embed",
  dimension: DIM,
  async generate() {
    throw new Error("向量必须由客户端提供");
  },
  getConfig() {
    return { name: "client-embed", dimension: DIM };
  },
};

let client = null;
let collection = null;

async function connect() {
  if (collection) return collection;
  client = new SeekdbClient(SEEKDB);
  const cols = await client.listCollections();
  const exists = cols.some((c) => (c?.name ?? c) === COLLECTION);
  if (exists) {
    collection = await client.getCollection({ name: COLLECTION, embeddingFunction: CLIENT_EMBED });
  } else {
    const schema = new Schema({
      fulltextIndex: new FulltextIndexConfig("ik", { ik_mode: "smart" }),
      vectorIndex: new VectorIndexConfig({ hnsw: { dimension: DIM, distance: "cosine" } }),
    });
    try {
      collection = await client.createCollection({ name: COLLECTION, schema, embeddingFunction: CLIENT_EMBED });
    } catch (e) {
      // listCollections 可能漏报已存在的集合，重复创建时回退到已有集合
      if (!/duplicate/i.test(e.message)) throw e;
      collection = await client.getCollection({ name: COLLECTION, embeddingFunction: CLIENT_EMBED });
    }
  }
  return collection;
}

function json(res, code, body) {
  const buf = Buffer.from(JSON.stringify(body));
  res.writeHead(code, {
    "content-type": "application/json; charset=utf-8",
    "content-length": buf.length,
    "access-control-allow-origin": "*",
    "access-control-allow-headers": "content-type",
    "access-control-allow-methods": "GET,POST,OPTIONS",
  });
  res.end(buf);
}

async function readBody(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  if (!chunks.length) return {};
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

const routes = {
  "GET /api/health": async () => {
    const col = await connect();
    return { ok: true, seekdb: SEEKDB, collection: COLLECTION, dim: DIM, count: await col.count() };
  },

  "POST /api/ingest": async ({ docs, vectors }) => {
    if (!Array.isArray(docs) || !Array.isArray(vectors) || docs.length !== vectors.length) {
      throw new Error("docs 与 vectors 数量必须一致");
    }
    if (!docs.length) return { inserted: 0 };
    if (vectors[0].length !== DIM) throw new Error(`向量维度应为 ${DIM}，实际 ${vectors[0].length}`);
    const col = await connect();
    await col.add({
      ids: docs.map((d) => String(d.id)),
      documents: docs.map((d) => d.text),
      metadatas: docs.map((d) => d.meta || {}),
      embeddings: vectors,
    });
    await col.refresh_index();
    return { inserted: docs.length, count: await col.count() };
  },

  "POST /api/search": async ({ queryText, queryVector, topK = 5 }) => {
    if (!Array.isArray(queryVector)) throw new Error("queryVector 必须是数组");
    if (queryVector.length !== DIM) throw new Error(`查询向量维度应为 ${DIM}，实际 ${queryVector.length}`);
    const col = await connect();
    const hasText = typeof queryText === "string" && queryText.trim().length > 0;

    const options = {
      query: hasText ? { whereDocument: { $contains: queryText.trim() } } : {},
      knn: { queryEmbeddings: [queryVector] },
      nResults: Number(topK),
      include: ["documents", "metadatas", "distances"],
    };
    const r = await col.hybridSearch(options);
    const ids = r.ids?.[0] ?? [];
    const docs = r.documents?.[0] ?? [];
    const metas = r.metadatas?.[0] ?? [];
    const dists = r.distances?.[0] ?? [];
    return {
      mode: hasText ? "hybrid" : "vector",
      results: ids.map((id, i) => ({ id, text: docs[i], meta: metas[i], distance: dists[i] })),
    };
  },

  "GET /api/stats": async () => {
    const col = await connect();
    return { count: await col.count(), collection: COLLECTION, dim: DIM };
  },

  "POST /api/reset": async () => {
    await connect();
    try {
      await client.deleteCollection(COLLECTION);
    } catch (e) {
      if (!/not found|does not exist/i.test(e.message)) throw e;
    }
    collection = null;
    const col = await connect();
    return { count: await col.count() };
  },
};

const server = http.createServer(async (req, res) => {
  if (req.method === "OPTIONS") {
    res.writeHead(204, {
      "access-control-allow-origin": "*",
      "access-control-allow-headers": "content-type",
      "access-control-allow-methods": "GET,POST,OPTIONS",
    });
    return res.end();
  }
  const key = `${req.method} ${req.url.split("?")[0]}`;
  const handler = routes[key];
  if (!handler) return json(res, 404, { error: `no route ${key}` });
  try {
    const body = req.method === "POST" ? await readBody(req) : {};
    json(res, 200, await handler(body));
  } catch (err) {
    console.error(key, err);
    json(res, 500, { error: err.message });
  }
});

server.listen(PORT, "127.0.0.1", async () => {
  console.log(`api listening on http://127.0.0.1:${PORT}`);
  try {
    const col = await connect();
    console.log(`seekdb ready, collection=${COLLECTION}, count=${await col.count()}`);
  } catch (e) {
    console.error("seekdb connection failed:", e.message);
  }
});

for (const sig of ["SIGINT", "SIGTERM"]) {
  process.on(sig, async () => {
    try {
      await client?.close?.();
    } catch {}
    process.exit(0);
  });
}
