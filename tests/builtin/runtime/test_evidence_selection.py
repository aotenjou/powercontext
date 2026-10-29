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
