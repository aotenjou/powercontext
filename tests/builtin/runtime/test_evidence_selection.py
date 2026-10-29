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
