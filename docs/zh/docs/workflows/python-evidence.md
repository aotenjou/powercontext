---
title: 在 Python 中采集与准备证据
description: 通过内置 Python Runtime 采集结构化文档、控制处理时机，并输出可精确回查的证据。
---

# 在 Python 中采集与准备证据

这些能力使用内置 Python Runtime。保留返回的 Source 引用和精确 Memory citation，重启后仍可回查原始证据。

## 结构化文档来源

`powercontext.sources.csv_rows` 支持 UTF-8 CSV，包括 BOM、引号包围的多行单元格。
`parse_csv_document` 完整解析并验证文档后才返回行。列名、字段值、主键中的空格和列序保持原样。
注册 `CSV_ROW_SOURCE_DEFINITION` 后，可解析、读取和投影行级 `powercontext.text-evidence/1`。

```python
from powercontext.sources import SourceDefinitionRegistry
from powercontext.sources.csv_rows import CSV_ROW_SOURCE_DEFINITION, CSVDocumentInput, parse_csv_document

async def capture_rows(content: bytes):
    rows = parse_csv_document(CSVDocumentInput(document_id="airports", primary_key="id", content=content))
    registry = SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,))
    return tuple([await registry.resolve(row) for row in rows])
```

`powercontext.sources.pi_session` 提供 `parse_pi_session` 和 `PiSessionConnector(path, leaf_id=None)`，
处理 Pi v3 JSONL。选定叶节点及其祖先决定采集分支。可见消息、工具调用、完整工具结果及摘要作为原始证据保留。
通过 `ConnectorLifecycle` 运行 Connector：持久化 checkpoint 检测历史改写、避免重复导入共享祖先，
并且只在入库成功后推进。运行失败保留此前已提交的 checkpoint。

安装 `powercontext[pdf]` 后可使用 `powercontext.sources.pdf_pages`。
`parse_pdf_document(PDFDocumentInput(...))` 在返回前解析全部物理页，包括空白页。
每个 `PDFPageInput` 保留文档 ID、完整文件的 SHA-256、从 1 开始的物理页码及页面正文。
注册 `PDF_PAGE_SOURCE_DEFINITION`，再通过 Runtime 导入 observation。PDF 修订产生新的页身份，
历史引用仍指向原修订。页面证据处理为 Memory 后，
`await search_pdf_pages(memory, question, limit=20)` 使用离线词项检索和财报术语扩展，返回不同页面的可读 citation。
排序匹配公司身份与报告年度，并优先选择与财务指标相关的报表标题。

可以在本地 Source registry 注册 Definition，也可以通过 `manifest_for_definition` 和
`project_source_for_transport` 使用既有 ingestion API 传输。采集负责保存证据；将符合条件的证据转为 Memory，
还需配置生成模型或应用自己的 pipeline。

## 数量门槛与共享 Embedding

`RuntimeConfig.memory_source_count_threshold` 和 `topic_memory_source_count_threshold` 默认为 `None`。
设置为严格的正整数后，某个 Scope 的未消费 Source journal 条目数达到门槛，就会自动启动处理。
既有时间条件仍然生效，显式 flush 绕过数量门槛。重复 capture 和 receipt 信任升级不会增加 journal 身份。
只用于 lineage 的来源计入启动数量，但仍从生成证据中排除。

窗口上限保持原有意义：Memory 消费一窗后重新判断剩余数量；Topic Memory 已接受的 wave 按既有规则完成后续窗口。
每个 Scope 和 Family 保留独立的成功游标，失败及重启后仍有效。启用自动工作前，先配置通常的生成 provider 和 worker。

用 `powercontext.builtin.inference.coalescing.CoalescingEmbeddingModel` 包装 Embedding 模型，再作为
`embedding_model` 传给 `open_builtin_contexts`。并发调用共享相同文本的在途计算及有界的成功向量 LRU 缓存。
`max_batch_size` 限制每次 provider 批次，`coalesce_ms` 设置合并窗口，`cache_size=0` 关闭跨调用缓存。
取消独立影响当前调用；usage 对当前调用等待的每个不同底层批次只汇总一次。纯缓存命中报告零次请求。
每个包装器绑定一个模型 profile 和事件循环。

`powercontext.builtin.artifacts.memory.http_reranker.HTTPMemoryReranker` 接收 endpoint 和调用方拥有的
`httpx.AsyncClient`。组合 contexts 或 Runtime 时通过 `memory_reranker` 传入。
它分批发送全局编号的 `rank`/`text` 候选，验证完整的有限分数，并在所有批次间统一选择。
同分保持原始 rank 顺序；返回的决策保留 citation，并报告实际请求用量。

seekdb 的完整 Embedding 路径使用一条候选 SQL 完成混合 Memory 检索，原生全文及向量分支分别设置候选上限。
Scope、Artifact 和 tag 条件在候选上限之前应用。返回 citation 使用已验证的当前 Memory head，
包括未修改、投影仍记录较早 head 的活跃条目。FTS-only、vector-only 和 OceanBase 的组合路径保持既有行为。

## 原文片段与增量传输

设置 `PrepareContextRequest(evidence_selection=True, ...)`，按查询选择 Memory 原文片段。该模式默认关闭。
`PreparedContextBuilder.build_scopes_result` 返回的 `fragments` 包含带 Scope 的 `MemoryEntryAddress`、
半开 Python 字符坐标及精确原文。Origins 与片段逐项对应。
完整 trust policy、envelope、JSON、citation 和正文共同计入 `max_bytes`。选择模式处理 Memory 候选。
选择时优先匹配查询的研究对象与标识，并归一化常见词形和相关词根。
保留中间原文比重复 citation 更省空间时，将邻近证据合并为同一原文区间。
预算选择比较完整渲染后的组合，同时保留原始空白和字符坐标。

`ScopedContextApplication.prepare_incremental(request, base_pack=None)` 先准备普通目标上下文。
首次返回 `ContextPack`；传入接收端上一包后，返回 `ContextPackDelta`。发送前先序列化数据包。
接收端通过 `ContextPack.from_bytes` 解码首包，通过 `ContextPackDelta.from_bytes` 解码增量，
再用 `apply_context_pack_delta(base, delta)` 重建。

v1 wire 格式使用规范 UTF-8 JSON、Base64 分块和 SHA-256 摘要。分块保持 UTF-8 边界并遵守容量上限。
非法字段、编码、摘要和前缀抛出 `ContextPackError`。普通空结果重建为空字节。
评估节省量时，比较 `len(packet.to_bytes())` 与普通全文的字节数；前缀变化可能使增量大于全文。

## 撤回与恢复依赖证据

通过 `runtime.lifecycle.for_scope(scope_id)` 将 Source、精确 Memory citation 或 Topic Memory 修订
注册到独立的节点 ID。父节点须先在同一 Scope 注册。相同注册幂等；冲突绑定、缺失父节点、重复父节点及环被拒绝。
每份完整正文对应一个节点：`policy="all"` 要求所有父节点可用，`policy="any"` 表示任一父节点
都能独立支持完整正文。

```python
from powercontext.builtin.runtime.source_lifecycle import SourceLifecycleEvent

async def bind_and_withdraw(runtime, scope_id, source_ref, citation, topic_ref):
    service = runtime.lifecycle.for_scope(scope_id)
    await service.register_source("source", source_ref, revision=3)
    await service.register_memory("fact", citation, parents=("source",))
    await service.register_topic("topic", topic_ref, parents=("fact",))
    return await service.apply(SourceLifecycleEvent(
        node_id="source", revision=3, sequence=1, available=False, reason="Source withdrawn"
    ))
```

事件须匹配已注册的生命周期 revision，并使用单调递增的 sequence。重放相同事件不会新增 Memory 修订。
撤回与恢复传播到全部依赖节点：失去支持的 Memory 退出活跃召回，Topic 退出 Topic 搜索及准备的 Context。
原始正文和审计引用仍可读。恢复时发送更高 sequence、`available=True` 的事件。

`resolve(node_id, purpose="generation")` 拒绝不可用证据；`purpose="audit"` 返回原始引用。
注册、接受的事件及原生 Memory 状态变更共同提交。状态写入失败会回滚事件，使相同事件在重启后仍可重试。
绑定期间不支持外部改写已注册的正文版本。
