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

"""Full physical-page PDF capture and scoped, exact page retrieval."""

# ruff: noqa: TRY003 -- Input validation reports errors at the owning boundary.

from __future__ import annotations

import hashlib
import io
import math
import re
from collections import Counter
from typing import TYPE_CHECKING
from urllib.parse import quote

from pydantic import BaseModel, Field, field_validator

from powercontext.artifacts import MemoryCitation
from powercontext.sources import Source, SourceDefinition, SourceMaterialization, SourceProjection, TextEvidence

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import ScopedMemoryApplication


class PDFDocumentInput(BaseModel):
    document_id: str
    content: bytes

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("PDF document identity must be nonempty")
        return value


class PDFPageInput(BaseModel):
    document_id: str
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_number: int = Field(strict=True, ge=1)
    text: str


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    from pypdf import PdfReader

    if not value.content.startswith(b"%PDF-"):
        raise ValueError("Expected a complete PDF document")
    try:
        reader = PdfReader(io.BytesIO(value.content), strict=True)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF requires a password")  # noqa: TRY301 - normalize all PDF reader failures
        digest = hashlib.sha256(value.content).hexdigest()
        return tuple(
            PDFPageInput(
                document_id=value.document_id,
                document_sha256=digest,
                page_number=number,
                text=page.extract_text() or "",
            )
            for number, page in enumerate(reader.pages, 1)
        )
    except Exception as error:
        raise ValueError("PDF could not be fully parsed") from error


class PDFPageSource(Source):
    page: PDFPageInput


class _PDFProjection:
    name: str = "powercontext.text-evidence"
    version: str = "1"
    source_class: type[PDFPageSource] = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        page = source.page
        return TextEvidence(
            source_type="pdf-page",
            source_id=source.name,
            content=f"{page.document_id}\n{page.text}",
            metadata={
                "document_id": page.document_id,
                "document_sha256": page.document_sha256,
                "page_number": page.page_number,
            },
        )


class _PDFDefinition:
    name: str = "pdf-page"
    version: str = "1"
    input_class: type[PDFPageInput] = PDFPageInput
    source_class: type[PDFPageSource] = PDFPageSource
    projections: tuple[SourceProjection[PDFPageSource], ...] = (_PDFProjection(),)

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        document = "id-" + quote(value.document_id, safe="")
        if len(document) > 170:
            document = "sha256-" + hashlib.sha256(value.document_id.encode()).hexdigest()
        identity = f"pdf:{document}:{value.document_sha256}:{value.page_number}"
        return PDFPageSource(name=identity, materialization=SourceMaterialization.CAPTURED, page=value)

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return source.page


PDF_PAGE_SOURCE_DEFINITION: SourceDefinition[PDFPageInput, PDFPageSource, PDFPageInput] = _PDFDefinition()


def _stem_pdf_term(term: str) -> str:
    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    for suffix in ("ations", "ation", "ions", "ion", "ing", "ed", "s"):
        if term.endswith(suffix) and len(term) - len(suffix) >= 4:
            return term[: -len(suffix)]
    return term


def _pdf_search_plan(question: str) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, float]]:
    stop = {
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "on",
        "for",
        "from",
        "by",
        "with",
        "and",
        "or",
        "is",
        "was",
        "were",
        "are",
        "what",
        "how",
        "does",
        "did",
        "do",
        "would",
        "should",
        "could",
        "company",
        "companies",
        "year",
        "fiscal",
        "report",
        "annual",
        "reported",
        "calculate",
        "based",
        "according",
        "much",
        "many",
        "have",
        "has",
        "had",
        "its",
        "their",
        "it",
        "as",
        "at",
        "that",
        "this",
        "which",
        "when",
        "during",
        "please",
        "provide",
        "answer",
        "question",
        "fy",
        "give",
        "response",
        "respond",
        "define",
        "assuming",
        "assume",
        "perspective",
        "investment",
        "analyst",
        "only",
        "use",
        "details",
        "shown",
        "within",
        "relying",
        "unadjusted",
        "then",
        "state",
        "explicitly",
        "outlined",
        "directly",
        "important",
        "primarily",
    }
    terms = tuple(dict.fromkeys(term for term in re.findall(r"[a-z]+|\d+", question.lower()) if term not in stop))
    # Expand reporting vocabulary, independent of document and question identities.
    synonyms = {
        "geographies": ("geographic", "countries", "international", "segments"),
        "geography": ("geographic", "countries", "international", "segments"),
        "operates": ("operations",),
        "battles": ("litigation", "proceedings", "lawsuits"),
        "legal": ("litigation", "proceedings", "lawsuits"),
        "ebitda": ("operating", "income", "depreciation", "amortization"),
        "capex": ("capital", "expenditures"),
    }
    expansions = tuple(dict.fromkeys(alternative for term in terms for alternative in synonyms.get(term, ())))
    # Short probes avoid long-question lexical admission discarding relevant pages.
    probes = [" ".join(terms[start : start + 24]) for start in range(0, len(terms), 24)] + [
        " ".join(terms[start : start + 2]) for start in range(0, len(terms), 2)
    ]
    probes.extend(expansions)
    weights = {_stem_pdf_term(term): 1.0 for term in terms}
    for term in expansions:
        weights.setdefault(_stem_pdf_term(term), 0.7)
    return terms, tuple(dict.fromkeys(probes)), weights


def _rank_pdf_hits(hits, terms: tuple[str, ...], weights: dict[str, float]):
    frequencies = {
        key: Counter(_stem_pdf_term(term) for term in re.findall(r"[a-z]+|\d+", hit.text.lower()))
        for key, hit in hits.items()
    }
    document_frequency = Counter(term for counts in frequencies.values() for term in weights if counts[term])

    def score(key):
        counts = frequencies[key]
        length = sum(counts.values())
        total = 0.0
        for term, query_weight in weights.items():
            tf = counts[term]
            weight = math.log(1 + (len(hits) + 1) / (1 + document_frequency[term]))
            total += query_weight * weight * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / 500))
        # Questions about statement metrics also benefit from matching statement headings.
        text = " ".join(re.findall(r"[a-z]+|\d+", hits[key].text.lower()))
        if {"income", "ebitda", "earnings"} & set(terms) and re.search(
            r"consolidated statements? (?:of )?(?:income|operations|earnings)", text
        ):
            total *= 2.0
        return total

    return sorted(hits, key=lambda key: (-score(key), key))


async def search_pdf_pages(
    memory: ScopedMemoryApplication, question: str, limit: int = 10
) -> tuple[MemoryCitation, ...]:
    """Retrieve through Memory, then rank literal pages without changing citations."""
    from powercontext.builtin.runtime import GetMemoryEntryRequest, SearchMemoryRequest

    if type(limit) is not int or limit < 1:
        raise ValueError("Page search limit must be a positive integer")
    terms, probes, weights = _pdf_search_plan(question)
    if not terms:
        return ()
    hits = {}
    for probe in dict.fromkeys(probes):
        page = await memory.search(SearchMemoryRequest(query=probe, mode="fts", limit=100))
        for hit in page.hits:
            hits[(hit.memory_ref.artifact_id, hit.entry_id, hit.entry_version_id)] = hit
    citations, locations = [], set()
    for key in _rank_pdf_hits(hits, terms, weights):
        hit = hits[key]
        citation = MemoryCitation(
            memory_ref=hit.memory_ref, entry_id=hit.entry_id, entry_version_id=hit.entry_version_id
        )
        entry = (await memory.get(GetMemoryEntryRequest(citation=citation))).entry
        pages = tuple(source.source_id for source in entry.sources if source.source_type == "pdf-page")
        if len(pages) != 1 or pages[0] in locations:
            continue
        locations.add(pages[0])
        citations.append(citation)
        if len(citations) == limit:
            break
    return tuple(citations)
