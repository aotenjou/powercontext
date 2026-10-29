---
title: Capture and prepare evidence in Python
description: Capture structured documents, control processing, and prepare exact evidence through the built-in Python runtime.
---

# Capture and prepare evidence in Python

These capabilities use the built-in Python runtime. Retain the returned Source references and exact Memory citations
to read the original evidence after a restart.

## Structured document sources

`powercontext.sources.csv_rows` parses UTF-8 CSV, including BOM and quoted multiline cells. `parse_csv_document`
validates the whole document before returning rows. Columns, values, primary-key spaces, and column order remain
unchanged. Register `CSV_ROW_SOURCE_DEFINITION` to resolve, read, and project rows as `powercontext.text-evidence/1`.

```python
from powercontext.sources import SourceDefinitionRegistry
from powercontext.sources.csv_rows import CSV_ROW_SOURCE_DEFINITION, CSVDocumentInput, parse_csv_document

async def capture_rows(content: bytes):
    rows = parse_csv_document(CSVDocumentInput(document_id="airports", primary_key="id", content=content))
    registry = SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,))
    return tuple([await registry.resolve(row) for row in rows])
```

`powercontext.sources.pi_session` provides `parse_pi_session` and `PiSessionConnector(path, leaf_id=None)` for Pi v3
JSONL. The selected leaf and its ancestors determine the captured branch. Visible messages, tool calls, complete tool
results, and summaries remain original evidence. Run the connector through `ConnectorLifecycle`; its persisted
checkpoint detects rewritten history, avoids importing shared ancestors twice, and advances only after successful
ingestion. A failed run retains the committed checkpoint.

Install `powercontext[pdf]` to use `powercontext.sources.pdf_pages`. `parse_pdf_document(PDFDocumentInput(...))`
extracts every physical page, including blank pages, before returning. Each `PDFPageInput` retains the document ID,
the complete file's SHA-256, the one-based physical page number, and its text. Register `PDF_PAGE_SOURCE_DEFINITION`
and ingest the observations through the runtime. A revised PDF gets new page identities; historical references keep
their original revision. With page evidence processed into Memory, `await search_pdf_pages(memory, question, limit=20)`
returns distinct, readable page citations using offline lexical retrieval and financial vocabulary expansion.

Definitions can be registered in a local Source registry, or transported with `manifest_for_definition` and
`project_source_for_transport` through the existing ingestion API. Capture stores evidence; configure generation or
an application pipeline to turn eligible evidence into Memory.

## Processing thresholds and shared embeddings

`RuntimeConfig.memory_source_count_threshold` and `topic_memory_source_count_threshold` default to `None`.
Set either to a strict positive integer to automatically process a scope when its unconsumed Source journal reaches
that count. The existing time condition also remains effective, and explicit flush bypasses the threshold. Repeated
capture and receipt trust upgrades add no journal identity. Lineage-only sources count toward admission while remaining
excluded from generation evidence.

Window limits retain their meaning. Memory consumes one window and reevaluates the remaining count; an accepted Topic
Memory wave completes its existing windows. Each scope and family retains its own successful cursor, including after
failure and restart. Configure the ordinary generation provider and worker settings before enabling automatic work.

Wrap an embedding model with `CoalescingEmbeddingModel` from `powercontext.builtin.inference.coalescing`, then pass it
as `embedding_model` to `open_builtin_contexts`. Concurrent calls share identical in-flight text and a bounded LRU cache
of successful vectors. `max_batch_size` bounds each provider batch, `coalesce_ms` sets the collection window, and
`cache_size=0` disables retained caching. Cancellation affects the caller independently; usage sums each distinct batch
the caller waited for once. Pure cache hits report zero requests. Keep a wrapper bound to one model profile and event loop.

`HTTPMemoryReranker` from `powercontext.builtin.artifacts.memory.http_reranker` accepts an endpoint and a caller-owned
`httpx.AsyncClient`. Supply it as `memory_reranker` when composing contexts or the runtime. It sends globally numbered
`rank`/`text` candidates in batches, validates complete finite scores, and selects across all batches. Equal scores keep
the original rank order; the returned decision retains citations and reports actual request usage.

For seekdb with complete embeddings, hybrid Memory retrieval executes one candidate SQL statement with independently
limited native full-text and vector branches. Scope, artifact, and tag restrictions apply before candidate limits.
Returned citations use the validated current Memory head, including entries whose unchanged projections were written
under an older head. FTS-only, vector-only, and the OceanBase composition retain their existing behavior.

## Literal excerpts and incremental delivery

Set `PrepareContextRequest(evidence_selection=True, ...)` to select query-relevant original Memory excerpts. The default
is disabled. `PreparedContextBuilder.build_scopes_result` exposes `fragments`: each holds a scoped `MemoryEntryAddress`,
half-open Python character offsets, and the exact original text. Origins correspond to the fragments. The complete
trust policy, envelope, JSON, citations, and text count toward `max_bytes`. Selection mode processes Memory candidates.

`ScopedContextApplication.prepare_incremental(request, base_pack=None)` prepares the ordinary target context. The first
call returns a `ContextPack`; subsequent calls return `ContextPackDelta` when the receiver's previous pack is supplied.
Serialize packets before sending them. The receiver decodes the first pack with `ContextPack.from_bytes`, decodes each
delta with `ContextPackDelta.from_bytes`, and reconstructs with `apply_context_pack_delta(base, delta)`.

The v1 wire format uses canonical UTF-8 JSON, Base64 chunks, and SHA-256 digests. Chunks preserve UTF-8 boundaries and
their capacity limit. Invalid fields, encodings, digests, and prefixes raise `ContextPackError`. An ordinary empty result
reconstructs empty bytes. Measure `len(packet.to_bytes())` against the ordinary full text when assessing savings; a
changed prefix can make a delta larger than the full text.

## Withdraw and restore dependent evidence

Use `runtime.lifecycle.for_scope(scope_id)` to register a Source, exact Memory citation, or Topic Memory revision under
an independent node ID. Parents must already exist in the scope. Repeated identical registration is idempotent;
conflicting bindings, missing parents, duplicates, and cycles are rejected. Register each evidence body as one node:
`policy="all"` needs every parent, while `policy="any"` means any one parent independently supports the complete body.

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

Events require the registered lifecycle revision and a monotonically increasing sequence. Replaying an identical event
adds no Memory revision. Withdrawal and restoration propagate through every dependency: unsupported Memory entries
leave active recall, and unsupported topics leave Topic search and prepared context. Their original bodies and audit
references remain readable. Restore by sending a higher-sequence event with `available=True`.

`resolve(node_id, purpose="generation")` rejects unavailable evidence; `purpose="audit"` returns its original reference.
Registrations, accepted events, and native Memory state transitions commit together. A failed state write rolls back
the event so the same event can retry after restart. Binding does not support external body revisions while registered.
