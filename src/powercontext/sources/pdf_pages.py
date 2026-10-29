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
        "geographies": ("geographic", "countries", "international", "segments", "america", "europe", "asia", "africa"),
        "geography": ("geographic", "countries", "international", "segments", "america", "europe", "asia", "africa"),
        "operates": ("operations",),
        "battles": ("litigation", "proceedings", "lawsuits"),
        "legal": ("litigation", "proceedings", "lawsuits"),
        "ebitda": ("operating", "income", "depreciation", "amortization"),
        "capex": ("capital", "expenditures", "cash", "flows"),
        "margin": ("revenue", "revenues", "sales", "cost", "income"),
        "margins": ("revenue", "revenues", "sales", "cost", "income"),
        "quick": ("cash", "receivables", "current", "assets", "liabilities"),
        "liquidity": ("cash", "receivables", "current", "assets", "liabilities"),
        "intensive": ("property", "plant", "equipment", "assets", "cash", "flows"),
        "cyclicality": ("cyclical", "economic", "demand", "commercial"),
        "concentration": ("major", "significant", "customers", "sales"),
        "growth": ("revenues", "sales", "segment"),
        "liability": ("liabilities",),
        "categories": ("segment", "segments", "revenues", "sales"),
        "retain": ("retention", "attrition", "acquire"),
        "members": ("cardmembers",),
        "primary": ("major", "principal", "significant", "commercial", "government"),
    }
    expansions = tuple(dict.fromkeys(alternative for term in terms for alternative in synonyms.get(term, ())))
    # Short probes avoid long-question lexical admission discarding relevant pages.
    probes = [" ".join(terms[start : start + 24]) for start in range(0, len(terms), 24)] + [
        " ".join(terms[start : start + 2]) for start in range(0, len(terms), 2)
    ]
    probes.extend(expansions)
    probes.extend(("consolidated statements income", "consolidated balance sheets", "consolidated cash flows"))
    weights = {_stem_pdf_term(term): 1.0 for term in terms}
    for term in expansions:
        weights.setdefault(_stem_pdf_term(term), 0.7)
    return terms, tuple(dict.fromkeys(probes)), weights


def _pdf_company_matches(documents, frequencies, terms, weights):
    # The projection's first line preserves the caller's document identity. Match
    # multiword company names without relying on a particular spelling or corpus.
    query_names = {
        "".join(terms[start:end])
        for start in range(len(terms))
        for end in range(start + 1, min(len(terms), start + 4) + 1)
    }

    companies = {
        key: "".join(re.findall(r"[a-z]+|\d+", re.split(r"_(?=\d{4})", name.lower())[0]))
        for key, name in documents.items()
    }
    matched_companies = {company for company in companies.values() if company in query_names}
    if not matched_companies:
        # A shortened company name may occur only in the body, rather than the ID.
        term_documents = {}
        for key, counts in frequencies.items():
            for term in weights:
                if counts[term] and len(term) > 3:
                    term_documents.setdefault(term, Counter())[documents[key]] += 1
        distinctive = {
            name
            for occurrences in term_documents.values()
            if len(occurrences) == 1
            for name, count in occurrences.items()
            if count >= 3
        }
        matched_companies = {companies[key] for key in documents if documents[key] in distinctive}
    return companies, matched_companies


def _pdf_intent_weight(content: str, terms: tuple[str, ...]) -> float:
    weight = 1.0
    # Distinguish statement tables from prose mentioning the same metrics.
    text = " ".join(re.findall(r"[a-z]+|\d+", content.lower()))
    query_terms = set(terms)
    income = bool(query_terms & {"income", "ebitda", "earnings", "margin", "margins", "tax"})
    balance = bool(query_terms & {"balance", "quick", "liquidity", "liability", "intensive"})
    cash = bool(query_terms & {"ebitda", "capex", "cash", "intensive"})
    heading = text[:600]
    if income and re.search(r"consolidated statements? (?:of )?(?:income|operations|earnings)", heading):
        weight *= 4.0
    if balance and re.search(
        r"consolidated (?:balance sheets?|statements? of financial (?:position|condition))", heading
    ):
        weight *= 4.0
    if cash and re.search(r"consolidated statements? (?:of )?cash flows", heading):
        weight *= 4.0
    if "tax" in query_terms and re.search(r"effective (?:income )?tax rates?", text):
        weight *= 3.0
    if query_terms & {"growth", "categories", "intensive"} and re.search(
        r"(?:sales|revenues|data|income|results).{0,100}(?:by business segment|segment data)|summary of business segment",
        heading,
    ):
        weight *= 3.0
    if "concentration" in query_terms and re.search(r"customer.{0,60}accounted for|sales to.{0,40}customer", text):
        weight *= 2.0
    if "retain" in query_terms and re.search(r"retain.{0,60}card members?|card ?members?.{0,60}retention", text):
        weight *= 2.0
    return weight


def _rank_pdf_hits(hits, terms: tuple[str, ...], weights: dict[str, float]):
    frequencies = {
        key: Counter(_stem_pdf_term(term) for term in re.findall(r"[a-z]+|\d+", hit.text.lower()))
        for key, hit in hits.items()
    }
    document_frequency = Counter(term for counts in frequencies.values() for term in weights if counts[term])

    documents = {key: hit.text.partition("\n")[0] for key, hit in hits.items()}
    companies, matched_companies = _pdf_company_matches(documents, frequencies, terms, weights)
    identity_terms = {
        _stem_pdf_term(term)
        for key, name in documents.items()
        if companies[key] in matched_companies
        for term in re.findall(r"[a-z]+|\d+", re.split(r"_(?=\d{4})", name.lower())[0])
    }
    years = [int(term) for term in terms if re.fullmatch(r"(?:19|20)\d{2}", term)]

    def score(key):
        counts = frequencies[key]
        length = sum(counts.values())
        total = 0.0
        for term, query_weight in weights.items():
            tf = counts[term]
            weight = math.log(1 + (len(hits) + 1) / (1 + document_frequency[term]))
            if term in identity_terms or (term.isdigit() and len(term) == 4):
                query_weight *= 0.1
            if term.isalpha() and len(term) == 1:
                continue
            total += query_weight * weight * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / 500))
        total *= _pdf_intent_weight(hits[key].text, terms)
        if matched_companies:
            total *= 1.0 if companies[key] in matched_companies else 0.05
        document_years = re.findall(r"(?:19|20)\d{2}", documents[key])
        if years and document_years and max(years) != int(document_years[0]):
            total *= 0.4
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
