# 端侧知识库 · 浏览器向量化 + seekdb 混合检索

阶跃星辰赛道参赛作品的实现：文档在**浏览器本地**完成 Embedding，向量与原文写入 seekdb，检索由 seekdb 的**向量 + 全文混合索引**完成，客户端不做二次拼合。

## 架构

```
浏览器（Transformers.js / ONNX Runtime Web）
  ├─ 文档向量化 ──┐
  ├─ 查询向量化 ──┤ HTTP
  └─ 结果展示     │
                  ▼
        本地 Node API (server/index.mjs, :8787)
                  │  seekdb JS SDK (MySQL 协议)
                  ▼
        seekdb 容器 (:2881)  fulltext(ik) + hnsw(384, cosine)
```

- 推理位置：`transformers.js` 在页面内运行，无任何云端推理 API 调用。
- 检索位置：`collection.hybridSearch()` 在 seekdb 内完成 RRF 融合。
- 数据库：向量、原文、元数据同表存储，写入后调用 `refresh_index()` 保证可检索。

## 环境要求

| 组件 | 版本 |
| --- | --- |
| Node.js | ≥ 20（本项目验证于 22.23.4） |
| Docker / Docker Compose | 用于运行 seekdb |
| 浏览器 | Chrome / Edge 最新版（支持 WebGPU 时优先使用） |

## 启动

```bash
# 1. 起数据库
docker compose up -d            # oceanbase/seekdb:latest，端口 2881

# 2. 装依赖
npm install --legacy-peer-deps --ignore-scripts
#   --legacy-peer-deps：seekdb@1.4.0 的 peerDependencies 里误写成 workspace:*
#   --ignore-scripts：跳过 onnxruntime-node 的 postinstall（浏览器场景用不到，且其下载脚本不跟随 302）

# 3. 起 API 与前端（两个终端）
npm run dev:server              # http://127.0.0.1:8787
npm run dev:web                 # http://localhost:5173
```

首次打开页面会下载模型文件（约 224 MB，fp16），之后走浏览器缓存。

### 可调参数（URL query）

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `dtype` | `fp32` / `fp16` / `q8` | 模型精度，默认 `fp16` |
| `backend` | `webgpu` / `wasm` | 强制推理后端；缺省自动检测（WebGPU 优先，失败降级 WASM） |

示例：`http://localhost:5173/?dtype=q8&backend=wasm`（用于演示降级路径）

## 验证样例

### 样例 1：导入与混合检索

1. 打开 `http://localhost:5173/`，等待右上角显示「推理后端 / 模型 / 文档数」。
2. 在「导入文档」粘贴（空行分段）：

   ```
   seekdb 是面向 AI 的向量数据库，支持向量与全文混合检索。

   OceanBase 是分布式关系数据库，兼容 MySQL 协议。

   向量化在浏览器本地完成，不会把原文发给云端。
   ```

3. 点「向量化并写入 seekdb」，状态栏应显示「已写入 3 段，库内共 N 段」。
4. 在「检索」输入 `哪个数据库支持混合检索`，结果表第一条应为 seekdb 那条。

**预期**：表格包含名称/片段、距离、来源；状态栏标注「向量 + 全文混合检索」。

### 样例 2：索引刷新后可检索（增量导入）

1. 已有数据的情况下，再导入一段新文本，例如 `断桥残雪是西湖的冬季景点`。
2. 立即搜索 `断桥残雪`。

**预期**：新文档出现在结果中——因为写入后调用了 `refresh_index()`。

### 样例 3：WASM 降级

打开 `http://localhost:5173/?backend=wasm`，徽标应显示「推理后端：WASM」，上述两个样例的全部流程仍可完成。

### 样例 4：量化对比（bench 页）

打开 `http://localhost:5173/bench.html`，点「开始跑全部配置」，会依次跑 6 个配置（fp32/fp16/q8 × webgpu/wasm），输出体积、加载、向量化、写入、查询、端到端延迟与 Recall@3 / nDCG@3。

**重置方法**：任一页面点「清空库」，或 `curl -X POST http://127.0.0.1:8787/api/reset -H 'content-type: application/json' -d '{}'`。

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 数据库连接、集合名、维度、当前条数 |
| POST | `/api/ingest` | `{docs:[{id,text,meta}], vectors:[[...]]}`，写入后刷新索引 |
| POST | `/api/search` | `{queryText, queryVector, topK}`，有文本走混合检索，空文本走纯向量 |
| POST | `/api/reset` | 删除并重建集合 |

## 实测结果（本机 WSL2，Chrome headless，12 篇文档 / 6 条查询）

模型：`Xenova/paraphrase-multilingual-MiniLM-L12-v2`（384 维，多语言）

| 配置 | 模型体积 | 压缩比 | 加载 ms | 向量化 ms/段 | 写入+刷新 ms | 查询向量化 ms | seekdb 检索 ms | 端到端 ms | 混合 R@3 | 向量 R@3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32 + WebGPU | 448.5 MB | 1.00× | 58972 | 44.5 | 669 | 19.1 | 16.2 | 33.6 | 0.917 | 0.833 |
| fp16 + WebGPU | 224.4 MB | 2.00× | 27499 | 4.1 | 802 | 13.1 | 9.2 | 22.5 | 0.917 | 0.833 |
| q8 + WebGPU | 112.8 MB | 3.97× | 16092 | 40.9 | 518 | 206.5 | 9.5 | 216.3 | 0.917 | 0.833 |
| fp32 + WASM | 448.5 MB | 1.00× | 1269 | 54.3 | 513 | 26.4 | 9.5 | 37.6 | 0.917 | 0.833 |
| fp16 + WASM | 224.4 MB | 2.00× | 1106 | 53.0 | 611 | 27.4 | 9.3 | 36.7 | 0.917 | 0.833 |
| q8 + WASM | 112.8 MB | 3.97× | 644 | 25.1 | 509 | 13.0 | 9.6 | 23.1 | 0.917 | 0.750 |

要点：fp16 体积减半且检索质量与 fp32 完全一致，故设为默认；q8 压到 ~1/4，WASM 下向量召回从 0.833 降到 0.750；q8 在 WebGPU 上单条查询向量化反而更慢（int8 算子回退），需要实测选择。

## 已知限制

- 端侧推理速度依赖设备；本机 WSL2 + headless Chrome 的绝对数值不代表用户机器。
- 首屏需下载模型（fp16 约 224 MB），弱网下等待明显；已支持浏览器缓存。
- 集合只在 `database=test` 下建一个固定集合 `kb`，多知识库切换未实现。
- 图片内容支持（题目中的可选扩展）未实现。
- seekdb 容器内存预算由 compose 环境变量控制，小内存机器建议同时限制 Docker 资源。

## 数据与许可

- Embedding 模型：`Xenova/paraphrase-multilingual-MiniLM-L12-v2`（Apache-2.0，ONNX 权重由 Xenova 转换）。
- 数据库：seekdb（Apache-2.0）。
- 页面内所有示例文本均为本项目自撰，无第三方语料。
