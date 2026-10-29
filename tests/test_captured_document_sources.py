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

"""Captured document identities and original values survive projection."""

from __future__ import annotations

import asyncio
import hashlib
import json

from powercontext.sources import SourceDefinitionRegistry, TextEvidence
from powercontext.sources.csv_rows import CSV_ROW_SOURCE_DEFINITION, CSVDocumentInput, parse_csv_document
from powercontext.sources.pdf_pages import PDF_PAGE_SOURCE_DEFINITION, PDFPageInput
from powercontext.sources.pi_session import PI_EVENT_SOURCE_DEFINITION, PiEventInput


def test_csv_keeps_multiline_values_and_significant_primary_key_spaces() -> None:
    rows = parse_csv_document(
        CSVDocumentInput(
            document_id="table", primary_key=" id ", content=b' id ,description\n 01 ,"first\nsecond"\n01,plain\n'
        )
    )
    assert [row.primary_key_value for row in rows] == [" 01 ", "01"]

    async def run() -> None:
        registry = SourceDefinitionRegistry((CSV_ROW_SOURCE_DEFINITION,))
        for row in rows:
            source = await registry.resolve(row)
            assert await registry.read(source) == row
            evidence = TextEvidence.model_validate(registry.project(source, registry.projection_keys(source)[0]))
            assert json.loads(evidence.content) == {" id ": row.primary_key_value, "description": row.values[1]}

    asyncio.run(run())


def test_hashed_document_and_event_identities_do_not_alias_literal_ids() -> None:
    async def run() -> None:
        document = "long-document" * 30
        digest = hashlib.sha256(document.encode()).hexdigest()
        page = PDFPageInput(document_id=document, document_sha256="a" * 64, page_number=1, text="Evidence")
        literal = page.model_copy(update={"document_id": "sha256-" + digest})
        first = await PDF_PAGE_SOURCE_DEFINITION.resolve(page)
        second = await PDF_PAGE_SOURCE_DEFINITION.resolve(literal)
        assert first.name != second.name
        assert len(first.name) <= 256
        event = PiEventInput(
            session_id="session" * 80,
            event_id="event",
            parent_id=None,
            event_type="message",
            role="user",
            content="user: Evidence",
            record={},
        )
        hashed = await PI_EVENT_SOURCE_DEFINITION.resolve(event)
        literal_event = event.model_copy(update={"session_id": "sha256", "event_id": hashed.name.rsplit(":", 1)[1]})
        direct = await PI_EVENT_SOURCE_DEFINITION.resolve(literal_event)
        assert hashed.name != direct.name
        assert len(hashed.name) <= 256

    asyncio.run(run())
