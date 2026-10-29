# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Scoped PDF retrieval selects financial evidence across company documents."""

from pathlib import Path

import pytest

from powercontext.builtin.artifacts.memory import MemoryCandidateRequest, MemoryEntryInput
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    BuiltinConfig,
    GetMemoryEntryRequest,
    SubmitSourceObservation,
    open_builtin_runtime,
)
from powercontext.builtin.scope import ScopeDraft
from powercontext.sources import (
    TEXT_EVIDENCE_PROJECTION_KEY,
    SourceDefinitionRegistry,
    SourceObservation,
    TextEvidence,
    manifest_for_definition,
    project_source_for_transport,
)
from powercontext.sources.pdf_pages import PDF_PAGE_SOURCE_DEFINITION, PDFPageInput, search_pdf_pages


class _PagePipeline:
    async def extract(self, request: MemoryCandidateRequest, /) -> tuple[MemoryEntryInput, ...]:
        return tuple(
            MemoryEntryInput(
                kind="document",
                text=TextEvidence.model_validate(source.projection(TEXT_EVIDENCE_PROJECTION_KEY)).content,
                sources=(source,),
            )
            for source in request.sources
            if isinstance(source, SourceObservation)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("question", "expected_page"),
    (
        ("Does Alpha Tech have an improving gross margin in FY2024?", 1),
        ("Does Alpha Tech have a healthy quick ratio in FY2024?", 2),
    ),
)
async def test_financial_metrics_retrieve_the_named_company_after_restart(
    tmp_path: Path, question: str, expected_page: int
) -> None:
    registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'pages.db'}"))
    observations = []
    documents = {
        "ALPHATECH_2024_10K": (
            "CONSOLIDATED STATEMENTS OF INCOME\nRevenues 500 400. Cost of sales 200 190. Operating income 300 210.",
            "CONSOLIDATED BALANCE SHEETS\nCash 100 90. Receivables 50 40. Current assets 150 130. Current liabilities 60 70.",
        ),
        "BETATECH_2024_10K": (
            "Beta Tech gross margins, operating margin, quick ratio and liquidity in 2024. "
            "Revenue sales income cash receivables current assets liabilities. " * 4,
        )
        * 6,
    }
    for document_id, texts in documents.items():
        for number, text in enumerate(texts, 1):
            page = PDFPageInput(document_id=document_id, document_sha256="a" * 64, page_number=number, text=text)
            observations.append(project_source_for_transport(registry, await registry.resolve(page)))
    async with open_builtin_runtime(config, candidate_pipeline=_PagePipeline()) as runtime:
        assert runtime.scopes is not None
        scope = await runtime.scopes.create(
            ScopeDraft(title="PDF pages", summary="Financial evidence", idempotency_key="pdf-pages")
        )
        await runtime.ingestion.register(manifest_for_definition(PDF_PAGE_SOURCE_DEFINITION))
        for observation in observations:
            await runtime.ingestion.submit(SubmitSourceObservation(scope_id=scope.scope_id, observation=observation))
        while (await runtime.memory.for_scope(scope.scope_id).flush()).processed:
            pass
    async with open_builtin_runtime(config) as runtime:
        memory = runtime.memory.for_scope(scope.scope_id)
        citations = await search_pdf_pages(memory, question, limit=1)
        assert len(citations) == 1
        entry = (await memory.get(GetMemoryEntryRequest(citation=citations[0]))).entry
        expected = observations[expected_page - 1]
        assert entry.text == TextEvidence.model_validate(expected.projection(TEXT_EVIDENCE_PROJECTION_KEY)).content
        assert len(entry.sources) == 1 and entry.sources[0].source_id == expected.name
