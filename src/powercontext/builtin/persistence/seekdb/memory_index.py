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

"""Native seekdb hybrid candidates with independent indexed channel limits."""

from __future__ import annotations

from pyobvector import VECTOR
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncConnection
from typing_extensions import override

from powercontext.builtin.artifacts.memory import (
    CapabilityNotSupportedError,
    EmbeddingProfile,
    MemorySearchChannels,
    MemorySearchRequest,
)
from powercontext.builtin.artifacts.memory.canonical import validate_embedding
from powercontext.builtin.persistence.memory_index import CompositeMemoryIndex, memory_channel_hits
from powercontext.builtin.persistence.oceanbase.memory_index import OceanBaseMemoryFTSIndex, OceanBaseMemoryVectorIndex
from powercontext.builtin.persistence.tags import memory_tag_parameters, memory_tag_sql

_FTS_SQL = """
SELECT f.memory_artifact_id, f.head_revision, f.entry_id, f.entry_version_id, v.text,
       NULL AS distance, MATCH(f.searchable_text) AGAINST (:query) AS channel_score,
       'fts' AS channel
FROM pc_memory_entry_heads AS f
JOIN pc_memory_entry_versions AS v
  ON v.scope_id = f.scope_id
 AND v.memory_artifact_id = f.memory_artifact_id
 AND v.entry_version_id = f.entry_version_id
WHERE f.scope_id = :scope_id AND f.memory_artifact_id IN :memory_artifact_ids
  AND MATCH(f.searchable_text) AGAINST (:query)
{tags}
ORDER BY channel_score DESC, f.memory_artifact_id, f.entry_id, f.entry_version_id
LIMIT :candidate_limit
"""
_VECTOR_SQL = """
SELECT m.memory_artifact_id, m.head_revision, m.entry_id, m.entry_version_id, v.text,
       l2_distance(m.embedding, :query_vector) AS distance,
       -l2_distance(m.embedding, :query_vector) AS channel_score, 'vector' AS channel
FROM pc_memory_vector_entries AS m
JOIN pc_memory_entry_versions AS v
  ON v.scope_id = m.scope_id
 AND v.memory_artifact_id = m.memory_artifact_id
 AND v.entry_version_id = m.entry_version_id
WHERE m.scope_id = :scope_id AND m.memory_artifact_id IN :memory_artifact_ids
{tags}
ORDER BY l2_distance(m.embedding, :query_vector){approximate}
LIMIT :candidate_limit
"""


class SeekDBMemoryHybridIndex(CompositeMemoryIndex):
    """Reuse projection maintenance and single-channel behavior; combine hybrid SQL."""

    def __init__(self, profile: EmbeddingProfile) -> None:
        self._vector = OceanBaseMemoryVectorIndex(profile)
        super().__init__(OceanBaseMemoryFTSIndex(), self._vector)
        self._search = self._statement(tagged=False)
        self._tagged_search = self._statement(tagged=True)

    def _statement(self, *, tagged: bool):
        fts = _FTS_SQL.format(tags=memory_tag_sql("f") if tagged else "")
        vector = _VECTOR_SQL.format(
            tags=memory_tag_sql("m") if tagged else "", approximate="" if tagged else " APPROXIMATE"
        )
        statement = text(
            f"SELECT * FROM ({fts}) AS lexical UNION ALL SELECT * FROM ({vector}) AS semantic "  # noqa: S608 - only static SQL fragments
            "ORDER BY channel, channel_score DESC, memory_artifact_id, entry_id, entry_version_id"
        ).bindparams(
            bindparam("memory_artifact_ids", expanding=True),
            bindparam("query_vector", type_=VECTOR(self._vector.profile.dimension)),
        )
        if tagged:
            statement = statement.bindparams(
                bindparam("tag_keys", expanding=True), bindparam("tag_hashes", expanding=True)
            )
        return statement

    @override
    async def search(
        self, connection: AsyncConnection, scope_id: str, request: MemorySearchRequest, /
    ) -> MemorySearchChannels:
        if request.mode != "hybrid":
            return await super().search(connection, scope_id, request)
        profile = self._vector.profile
        if request.embedding_profile != profile or request.query_vector is None:
            raise CapabilityNotSupportedError("embedding-profile")
        if not await self._vector.vector_complete(connection, scope_id, request.memories, profile):
            raise CapabilityNotSupportedError("vector")
        rows = (
            (
                await connection.execute(
                    self._search if request.tag_filter is None else self._tagged_search,
                    {
                        "scope_id": scope_id,
                        "memory_artifact_ids": tuple(memory.artifact_id for memory in request.memories),
                        "query": request.analyzed_query,
                        "query_vector": validate_embedding(request.query_vector, dimension=profile.dimension),
                        "candidate_limit": request.candidate_limit,
                        **memory_tag_parameters(request.tag_filter),
                    },
                )
            )
            .mappings()
            .all()
        )
        return MemorySearchChannels(
            fts=memory_channel_hits((row for row in rows if row["channel"] == "fts"), request.memories),
            vector=memory_channel_hits((row for row in rows if row["channel"] == "vector"), request.memories),
        )
