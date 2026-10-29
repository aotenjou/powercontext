# 一页技术说明：端侧知识库

## 1. 模块结构

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| 端侧推理 | `src/embedder.js` | 加载 ONNX embedding 模型；WebGPU 优先、失败自动降级 WASM；支持 fp32 / fp16 / q8 |
| 交互层 | `src/main.js`、`index.html` | 文档导入（粘贴/文件）、查询、结果表、后端与精度徽标 |
| 对比实验 | `src/bench.js`、`bench.html` | 6 个配置的体积/延迟/质量矩阵，可导出 JSON |
| 本地服务 | `server/index.mjs` | HTTP API；写入后 `refresh_index()`；封装 seekdb SDK |
| 数据库 | `docker-compose.yml` | `oceanbase/seekdb:latest`，2881 端口，持久卷 |

数据流：浏览器向量化 → `POST /api/ingest` → seekdb（原文 + 向量 + 元数据同表）→ `refresh_index()` → 查询时浏览器再向量化 → `POST /api/search` → `collection.hybridSearch()`。

## 2. 模型与量化

- 模型：`Xenova/paraphrase-multilingual-MiniLM-L12-v2`，384 维，多语言（中英均可），Apache-2.0。
- 选型依据：先用 `all-MiniLM-L6-v2`（英文）实测，中文纯向量 Recall@3 仅 0.5；换多语言模型后升到 0.833。
- 量化方式：ONNX Runtime Web 的模型精度选项，`fp16` 为半精度权重，`q8` 为 INT8 量化权重（均由 Xenova 提供的 ONNX 权重文件）。
- 体积（实测 Content-Length）：fp32 448.5 MB / fp16 224.4 MB（2.00×）/ q8 112.8 MB（3.97×）。
- 质量（12 篇文档、6 条中文查询，Recall@3 / nDCG@3）：

| 精度 | 纯向量 R@3 | nDCG@3 | 结论 |
| --- | ---: | ---: | --- |
| fp32 | 0.833 | 0.820 | 基线 |
| fp16 | 0.833 | 0.820 | 无损失 |
| q8（WASM） | 0.750 | 0.769 | 可测量的下降 |

结论：默认 `fp16`——体积减半、质量与 fp32 一致。

## 3. 端侧运行与降级

- 检测 `navigator.gpu`，存在则在 `webgpu` 上初始化；初始化异常则自动改在 `wasm` 上重试，并把实际后端显示在页头徽标。
- `?backend=wasm` 可强制降级用于演示与自测；`?dtype=` 可切换精度。
- 全部向量计算发生在页面内，网络侧只有「向量 + 原文」到本地服务的请求，不调用任何云端推理 API。

## 4. seekdb 使用

- 集合 `kb`：`FulltextIndexConfig("ik", {ik_mode:"smart"})` + `VectorIndexConfig({hnsw:{dimension:384, distance:"cosine"}})`。
- 写入：`add({ids, documents, metadatas, embeddings})` 后立即 `refresh_index()`；未刷新时近似检索与全文检索都会返回空，这是本项目实测到的关键点。
- 检索：`hybridSearch({query:{whereDocument:{$contains:q}}, knn:{queryEmbeddings:[v]}, nResults})`，融合由数据库完成，客户端只渲染返回顺序。
- 用量：容器内存约 150 MB（空库），写入 12 篇后仍在 200 MB 以内。

## 5. 状态与增量

- 增量导入：多次 `ingest` 累加，每次刷新索引；新文档立即可检索（样例 2）。
- 会话状态：集合名、维度、条数由 `/api/health` 与 `/api/stats` 返回，页面刷新后保持；「清空库」走 `deleteCollection` 后重建。
- 失败处理：seekdb 未就绪时 API 返回 500 与具体错误，页面在状态栏显示失败原因，不静默降级。

## 6. 实测优化点

| 优化 | 效果（本机） |
| --- | --- |
| fp16 替代 fp32 | 体积 448→224 MB，质量不变，加载 59s→27s（冷启动） |
| 复用浏览器缓存 | 二次加载 100–900 ms |
| 批量向量化（一次 embed 12 段） | 44.5 ms/段 → 4.1 ms/段（fp16+WebGPU） |
| 写入合并 + 单次 refresh_index | 12 段写入+刷新 600–800 ms |

## 7. 局限

- 绝对延迟取决于设备与浏览器；WASM 下 q8 表现更好但质量略降，需按场景取舍。
- 未实现多知识库切换、图片内容检索、跨会话持久化记忆。
