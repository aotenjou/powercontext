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

"""Dependency changes commit with native Memory state and survive restart."""

from __future__ import annotations

import asyncio
import sqlite3

import pytest
from sqlalchemy.exc import IntegrityError

from powercontext.builtin.artifacts.memory import MemoryEntryInput
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    BuiltinConfig,
    CaptureSource,
    GetMemoryEntryRequest,
    RememberMemoryRequest,
    SearchMemoryRequest,
    open_builtin_runtime,
)
from powercontext.builtin.runtime.source_lifecycle import SourceLifecycleError, SourceLifecycleEvent
from powercontext.builtin.scope import ScopeDraft


def test_failed_propagation_rolls_back_and_same_event_retries_after_restart(tmp_path) -> None:
    path = tmp_path / "lifecycle.db"
    config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{path}"))

    async def run() -> None:
        async with open_builtin_runtime(config) as runtime:
            assert runtime.scopes is not None
            scope = (
                await runtime.scopes.create(
                    ScopeDraft(title="Evidence", summary="Withdrawal", idempotency_key="lifecycle")
                )
            ).scope_id
            service = runtime.lifecycle.for_scope(scope)
            receipt = await runtime.sources.for_scope(scope).capture(
                CaptureSource(source_id="root", content="Root", metadata={})
            )
            await service.register_source("root", receipt.source_ref)
            citations = []
            for index in range(2):
                result = await runtime.memory.for_scope(scope).remember(
                    RememberMemoryRequest(entries=(MemoryEntryInput(kind="fact", text=f"Recoverable fact {index}"),))
                )
                assert result.entry is not None
                citations.append(result.entry.citation)
                await service.register_memory(f"fact{index}", result.entry.citation, parents=("root",))
            head = (await runtime.memory.for_scope(scope).list()).memory_ref
            assert head is not None
            # Reject the second native revision after allowing the first state transition.
            with sqlite3.connect(path) as database:
                database.execute(
                    "CREATE TRIGGER reject_state_write BEFORE INSERT ON pc_artifacts "
                    f"WHEN NEW.family = 'memory' AND NEW.revision = {head.revision + 2} "
                    "BEGIN SELECT RAISE(ABORT, 'controlled state failure'); END"
                )
            event = SourceLifecycleEvent(node_id="root", revision=1, sequence=1, available=False, reason="Withdraw")
            with pytest.raises(IntegrityError):
                await service.apply(event)
            assert (await runtime.memory.for_scope(scope).list()).memory_ref == head
            assert await service.resolve("root") == receipt.source_ref
            assert len((await runtime.memory.for_scope(scope).list()).entries) == 2
            with sqlite3.connect(path) as database:
                database.execute("DROP TRIGGER reject_state_write")

        async with open_builtin_runtime(config) as runtime:
            service = runtime.lifecycle.for_scope(scope)
            assert await service.apply(event) == []
            page = await runtime.memory.for_scope(scope).search(SearchMemoryRequest(query="Recoverable", mode="fts"))
            assert not page.hits
            for index, citation in enumerate(citations):
                with pytest.raises(SourceLifecycleError):
                    await service.resolve(f"fact{index}")
                assert await service.resolve(f"fact{index}", purpose="audit") == citation
                assert (await runtime.memory.for_scope(scope).get(GetMemoryEntryRequest(citation=citation))).entry.text
            inactive = (await runtime.memory.for_scope(scope).list(include_inactive=True)).memory_ref
            assert await service.apply(event) == []
            assert (await runtime.memory.for_scope(scope).list(include_inactive=True)).memory_ref == inactive
            restored = event.model_copy(update={"sequence": 2, "available": True})
            assert await service.apply(restored) == ["fact0", "fact1", "root"]
            assert len((await runtime.memory.for_scope(scope).list()).entries) == 2

    asyncio.run(run())
