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

"""Query-specific literal excerpts retain scoped identity and complete byte accounting."""

import json

import pytest

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import MemoryHit
from powercontext.builtin.runtime import PrepareContextRequest
from powercontext.builtin.runtime.prepared_context import PreparedContextBuilder, PreparedMemoryCandidates


def test_chinese_sentences_without_spaces_keep_scoped_literal_ranges() -> None:
    text = "无关内容。数据库服务恢复成功。之后继续工作。"
    reference = ArtifactRef(family="memory", artifact_id="memory", revision=1)
    hit = MemoryHit(
        memory_ref=reference, entry_id="entry", entry_version_id="version", text=text, score=1.0, matched_by=("fts",)
    )
    build = PreparedContextBuilder().build_scopes_result(
        request=PrepareContextRequest(query="数据库", evidence_selection=True, max_bytes=2000),
        current_scope_id="left",
        memory_candidates=(
            PreparedMemoryCandidates(scope_id="left", hits=(hit,)),
            PreparedMemoryCandidates(scope_id="right", hits=(hit,)),
        ),
    )
    assert {fragment.address.memory.scope_id for fragment in build.fragments} == {"left", "right"}
    assert len(build.origins) == len(build.fragments) == 2
    for fragment, origin in zip(build.fragments, build.origins, strict=True):
        assert fragment.text == text[fragment.start : fragment.end] == "数据库服务恢复成功。"
        assert fragment.address == origin
    assert build.context.content is not None
    assert build.context.content_bytes == len(build.context.content.encode("utf-8")) <= 2000


def test_adjacent_evidence_shares_a_citation_under_a_tight_budget() -> None:
    text = "Drug inhibited bacterial growth. Drug inhibits bacterial infection."
    reference = ArtifactRef(family="memory", artifact_id="memory", revision=1)
    hit = MemoryHit(
        memory_ref=reference, entry_id="entry", entry_version_id="version", text=text, score=1.0, matched_by=("fts",)
    )
    build = PreparedContextBuilder().build_scopes_result(
        request=PrepareContextRequest(query="drug inhibiting growth infection", evidence_selection=True, max_bytes=650),
        current_scope_id="scope",
        memory_candidates=(PreparedMemoryCandidates(scope_id="scope", hits=(hit, hit)),),
    )
    assert len(build.fragments) == len(build.origins) == 1
    fragment = build.fragments[0]
    assert (fragment.start, fragment.end, fragment.text) == (0, len(text), text)
    assert fragment.address == build.origins[0]
    assert build.context.content is not None
    assert build.context.content_bytes == len(build.context.content.encode("utf-8")) <= 650
    items = json.loads(build.context.content.splitlines()[-2])["items"]
    assert len(items) == 1 and items[0]["content"] == text
    assert items[0]["citation"]["memory_ref"] == reference.model_dump(mode="json")


def test_abbreviations_and_original_whitespace_are_retained_in_evidence() -> None:
    text = "Background: samples were collected.\n  Results: Treatment inhibited C. difficile growth.  \n"
    reference = ArtifactRef(family="memory", artifact_id="memory", revision=1)
    hit = MemoryHit(
        memory_ref=reference, entry_id="entry", entry_version_id="version", text=text, score=1.0, matched_by=("fts",)
    )
    build = PreparedContextBuilder().build_scopes_result(
        request=PrepareContextRequest(
            query="treatment inhibiting difficile growth", evidence_selection=True, max_bytes=700
        ),
        current_scope_id="scope",
        memory_candidates=(PreparedMemoryCandidates(scope_id="scope", hits=(hit,)),),
    )
    assert len(build.fragments) == 1
    fragment = build.fragments[0]
    assert (
        fragment.text
        == text[fragment.start : fragment.end]
        == "\n  Results: Treatment inhibited C. difficile growth.  \n"
    )


def test_short_intervening_text_is_cheaper_than_repeating_a_citation() -> None:
    text = "Drug inhibited bacterial growth. Calibration finished. Drug inhibits bacterial infection."
    reference = ArtifactRef(family="memory", artifact_id="memory", revision=1)
    hit = MemoryHit(
        memory_ref=reference, entry_id="entry", entry_version_id="version", text=text, score=1.0, matched_by=("fts",)
    )
    build = PreparedContextBuilder().build_scopes_result(
        request=PrepareContextRequest(query="drug inhibiting growth infection", evidence_selection=True, max_bytes=650),
        current_scope_id="scope",
        memory_candidates=(PreparedMemoryCandidates(scope_id="scope", hits=(hit,)),),
    )
    assert len(build.fragments) == 1
    fragment = build.fragments[0]
    assert (fragment.start, fragment.end, fragment.text) == (0, len(text), text)
    assert build.context.content_bytes <= 650


def test_specific_research_subject_beats_generic_matching_outcomes() -> None:
    texts = (
        "We found that KIN7 mutations alter pancreatic insulin secretion and cause neonatal hypoglycaemia.",
        "We found an increased risk of metabolic disease developing by age 14 years in a large cohort.",
    )
    reference = ArtifactRef(family="memory", artifact_id="memory", revision=1)
    hits = tuple(
        MemoryHit(
            memory_ref=reference,
            entry_id=str(index),
            entry_version_id="version",
            text=text,
            score=1.0,
            matched_by=("fts",),
        )
        for index, text in enumerate(texts)
    )
    build = PreparedContextBuilder().build_scopes_result(
        request=PrepareContextRequest(
            query="A mutation in KIN7 leads to an increased risk of metabolic disease by age 14 years.",
            evidence_selection=True,
            max_bytes=700,
        ),
        current_scope_id="scope",
        memory_candidates=(PreparedMemoryCandidates(scope_id="scope", hits=hits),),
    )
    assert len(build.fragments) == 1
    assert build.fragments[0].address.entry_id == "0"
    assert build.fragments[0].text == texts[0]


def test_late_evidence_in_an_oversized_unpunctuated_paragraph_remains_usable() -> None:
    text = "General administrative background; " * 60 + "the cobalt sensor is calibrated at 37 degrees"
    reference = ArtifactRef(family="memory", artifact_id="memory", revision=1)
    hit = MemoryHit(
        memory_ref=reference, entry_id="entry", entry_version_id="version", text=text, score=1.0, matched_by=("fts",)
    )
    build = PreparedContextBuilder().build_scopes_result(
        request=PrepareContextRequest(query="cobalt sensor calibrated", evidence_selection=True, max_bytes=700),
        current_scope_id="scope",
        memory_candidates=(PreparedMemoryCandidates(scope_id="scope", hits=(hit,)),),
    )
    assert build.context.status == "ready"
    assert any("cobalt sensor is calibrated at 37 degrees" in fragment.text for fragment in build.fragments)
    for fragment in build.fragments:
        assert fragment.text == text[fragment.start : fragment.end]
        assert fragment.address.memory.scope_id == "scope"
    assert build.context.content is not None
    assert build.context.content_bytes == len(build.context.content.encode("utf-8")) <= 700


def test_word_windows_do_not_skip_fitting_evidence_after_an_oversized_word() -> None:
    text = "background" * 65 + " cobalt sensor calibrated ignored"
    reference = ArtifactRef(family="memory", artifact_id="memory", revision=1)
    hit = MemoryHit(
        memory_ref=reference, entry_id="entry", entry_version_id="version", text=text, score=1.0, matched_by=("fts",)
    )
    build = PreparedContextBuilder().build_scopes_result(
        request=PrepareContextRequest(query="cobalt sensor calibrated", evidence_selection=True, max_bytes=650),
        current_scope_id="scope",
        memory_candidates=(PreparedMemoryCandidates(scope_id="scope", hits=(hit,)),),
    )
    assert build.context.status == "ready"
    assert any("cobalt sensor calibrated" in fragment.text for fragment in build.fragments)
    for fragment in build.fragments:
        assert fragment.text == text[fragment.start : fragment.end]
    assert build.context.content_bytes <= 650


def test_retrieval_order_contributes_without_overriding_query_relevance() -> None:
    texts = (
        "The cobalt sensor is calibrated at 37 degrees and records temperature.",
        "The amber sensor is calibrated at 19 degrees and records temperature.",
    )
    reference = ArtifactRef(family="memory", artifact_id="memory", revision=1)
    hits = tuple(
        MemoryHit(
            memory_ref=reference,
            entry_id=str(index),
            entry_version_id="version",
            text=text,
            score=1.0 + index,
            matched_by=("fts",),
        )
        for index, text in enumerate(texts)
    )
    builder = PreparedContextBuilder()
    groups = (PreparedMemoryCandidates(scope_id="scope", hits=hits),)
    for query, expected in (("sensor calibrated temperature", "1"), ("cobalt", "0")):
        build = builder.build_scopes_result(
            request=PrepareContextRequest(query=query, evidence_selection=True, max_bytes=700),
            current_scope_id="scope",
            memory_candidates=groups,
        )
        assert len(build.fragments) == 1
        assert build.fragments[0].address.entry_id == expected


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_unicode_line_separators_preserve_a_readable_json_envelope(separator: str) -> None:
    text = f"Ocean{separator}station records temperature."
    reference = ArtifactRef(family="memory", artifact_id="memory", revision=1)
    hit = MemoryHit(
        memory_ref=reference, entry_id="entry", entry_version_id="version", text=text, score=1.0, matched_by=("fts",)
    )
    build = PreparedContextBuilder().build_scopes_result(
        request=PrepareContextRequest(query="station temperature", evidence_selection=True, max_bytes=700),
        current_scope_id="scope",
        memory_candidates=(PreparedMemoryCandidates(scope_id="scope", hits=(hit,)),),
    )
    assert build.context.content is not None
    item = json.loads(build.context.content.splitlines()[-2])["items"][0]
    assert item["content"] == build.fragments[0].text == text
    assert build.context.content_bytes == len(build.context.content.encode("utf-8")) <= 700
