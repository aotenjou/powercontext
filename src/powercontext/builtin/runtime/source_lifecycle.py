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

"""Persist scoped evidence dependencies and atomically propagate availability."""

# ruff: noqa: TRY003 -- Input validation reports errors at the owning boundary.
# ruff: noqa: TRY301 -- Normalize repository lookup failures without hiding domain validation failures.

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator
from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.artifacts.topic_memory import TopicMemory
from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_NODES_TABLE as NODES
from powercontext.sources import SourceMaterialization, SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import BuiltinRuntime
    from powercontext.builtin.runtime.relational import RelationalContexts


class SourceLifecycleError(ValueError):
    """An unknown, conflicting, unavailable, or invalid evidence dependency."""


class SourceLifecycleEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    node_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    revision: int = Field(gt=0)
    sequence: int = Field(gt=0)
    available: bool

    @field_validator("node_id", "reason")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Lifecycle identity and reason must be nonblank")
        return value


class _Node(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    node_id: str
    kind: Literal["source", "memory", "topic"]
    reference: dict[str, JsonValue]
    parents: tuple[str, ...] = ()
    policy: Literal["all", "any"] = "all"
    revision: int = 1
    available: bool = True
    event: SourceLifecycleEvent | None = None

    def ref(self) -> SourceRef | MemoryCitation | ArtifactRef:
        cls = {"source": SourceRef, "memory": MemoryCitation, "topic": ArtifactRef}[self.kind]
        return cls.model_validate(self.reference)


class SourceLifecycleApplication:
    def __init__(self, runtime: BuiltinRuntime, contexts: RelationalContexts) -> None:
        self._runtime, self._contexts = runtime, contexts

    def for_scope(self, scope_id: str) -> ScopedSourceLifecycle:
        from powercontext.builtin.sources import validate_scope_id

        return ScopedSourceLifecycle(self._runtime, self._contexts, validate_scope_id(scope_id))


class ScopedSourceLifecycle:
    def __init__(self, runtime: BuiltinRuntime, contexts: RelationalContexts, scope_id: str) -> None:
        self._runtime, self._contexts, self.scope_id = runtime, contexts, scope_id

    async def _load(self, connection: AsyncConnection) -> dict[str, _Node]:
        rows = await connection.scalars(select(NODES.c.payload).where(NODES.c.scope_id == self.scope_id))
        return {node.node_id: node for node in (_Node.model_validate_json(bytes(row)) for row in rows)}

    async def _save(self, connection: AsyncConnection, node: _Node, *, create: bool = False) -> None:
        ref = node.ref()
        artifact_id = (
            None
            if isinstance(ref, SourceRef)
            else (ref.memory_ref.artifact_id if isinstance(ref, MemoryCitation) else ref.artifact_id)
        )
        values = {
            "kind": node.kind,
            "artifact_id": artifact_id,
            "available": node.available,
            "payload": node.model_dump_json().encode("utf-8"),
        }
        statement = (
            insert(NODES).values(scope_id=self.scope_id, node_id=node.node_id, **values)
            if create
            else update(NODES)
            .where(NODES.c.scope_id == self.scope_id, NODES.c.node_id == node.node_id)
            .values(**values)
        )
        await connection.execute(statement)

    async def register_source(self, node_id: str, source_ref: SourceRef, *, revision: int = 1) -> None:
        if type(revision) is not int or revision <= 0:
            raise SourceLifecycleError("Source lifecycle revision must be a positive integer")
        await self._register(
            _Node(node_id=node_id, kind="source", reference=source_ref.model_dump(mode="json"), revision=revision)
        )

    async def register_memory(
        self, node_id: str, citation: MemoryCitation, *, parents: tuple[str, ...], policy: Literal["all", "any"] = "all"
    ) -> None:
        await self._register(
            _Node(
                node_id=node_id,
                kind="memory",
                reference=citation.model_dump(mode="json"),
                parents=parents,
                policy=policy,
            )
        )

    async def register_topic(
        self,
        node_id: str,
        artifact_ref: ArtifactRef,
        *,
        parents: tuple[str, ...],
        policy: Literal["all", "any"] = "all",
    ) -> None:
        if artifact_ref.family != "topic-memory":
            raise SourceLifecycleError("Expected a Topic Memory reference")
        await self._register(
            _Node(
                node_id=node_id,
                kind="topic",
                reference=artifact_ref.model_dump(mode="json"),
                parents=parents,
                policy=policy,
            )
        )

    async def _register(self, node: _Node) -> None:
        if not isinstance(node.node_id, str) or not node.node_id.strip() or len(node.node_id) > 256:
            raise SourceLifecycleError("Invalid lifecycle node identity")
        async with (
            self._runtime._scoped_operation(self.scope_id),
            self._runtime._locked(self.scope_id),
            self._contexts.database.transaction() as connection,
        ):
            nodes = await self._load(connection)
            existing = nodes.get(node.node_id)
            if existing is not None:
                binding = lambda value: (value.kind, value.reference, value.parents, value.policy, value.revision)
                if binding(existing) != binding(node):
                    raise SourceLifecycleError("Lifecycle node is already bound differently")
                return
            if node.kind != "source":
                if not node.parents or len(set(node.parents)) != len(node.parents) or node.node_id in node.parents:
                    raise SourceLifecycleError("Invalid lifecycle parents")
                if any(parent not in nodes for parent in node.parents):
                    raise SourceLifecycleError("Unknown lifecycle parent")
                node = node.model_copy(update={"available": self._supported(node, nodes)})
            await self._validate_reference(connection, node)
            await self._save(connection, node, create=True)
            await self._sync_memory(connection, node)

    async def _validate_reference(self, connection: AsyncConnection, node: _Node) -> None:
        ref = node.ref()
        try:
            if isinstance(ref, SourceRef):
                source = await self._contexts.repositories.sources.get(connection, self.scope_id, ref)
                if source.value.materialization is not SourceMaterialization.CAPTURED:
                    raise SourceLifecycleError("Lifecycle requires a captured Source")
            elif isinstance(ref, MemoryCitation):
                services = self._contexts._services_for(self.scope_id)
                _, catalog = services.sources(connection)
                service = services.memory(catalog, connection)
                memory = await service.revision(ref.memory_ref)
                entries = await service.entries(memory)
                if not any(
                    entry.entry_id == ref.entry_id and entry.entry_version_id == ref.entry_version_id
                    for entry in entries
                ):
                    raise SourceLifecycleError("Unknown exact Memory citation")
            else:
                artifact = await self._contexts.repositories.artifacts.get(connection, self.scope_id, ref)
                if not isinstance(artifact, TopicMemory):
                    raise SourceLifecycleError("Expected a Topic Memory artifact")
        except SourceLifecycleError:
            raise
        except (ValueError, LookupError) as error:
            raise SourceLifecycleError("Lifecycle reference is not readable in this scope") from error

    @staticmethod
    def _supported(node: _Node, nodes: dict[str, _Node]) -> bool:
        states = [nodes[parent].available for parent in node.parents]
        return all(states) if node.policy == "all" else any(states)

    async def _sync_memory(self, connection: AsyncConnection, node: _Node) -> None:
        if node.kind != "memory":
            return
        ref = MemoryCitation.model_validate(node.reference)
        services = self._contexts._services_for(self.scope_id)
        _, catalog = services.sources(connection)
        service = services.memory(catalog, connection)
        current = await service.latest(await service.revision(ref.memory_ref))
        entry = next((entry for entry in await service.entries(current) if entry.entry_id == ref.entry_id), None)
        if entry is None or entry.entry_version_id != ref.entry_version_id:
            raise SourceLifecycleError("Lifecycle Memory body version changed")
        if node.available:
            await service.reactivate(current, entries=(entry,), reason="Source lifecycle restored")
        else:
            await service.forget(current, entries=(entry,), reason="Source lifecycle withdrawn")

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        async with (
            self._runtime._scoped_operation(self.scope_id),
            self._runtime._locked(self.scope_id),
            self._contexts.database.transaction() as connection,
        ):
            nodes = await self._load(connection)
            source = nodes.get(event.node_id)
            if source is None or source.kind != "source" or source.revision != event.revision:
                raise SourceLifecycleError("Unknown Source or lifecycle revision")
            previous = source.event
            if previous is not None:
                if event.sequence < previous.sequence or (event.sequence == previous.sequence and event != previous):
                    raise SourceLifecycleError("Stale or conflicting lifecycle event")
                if event == previous:
                    return sorted(node_id for node_id, node in nodes.items() if node.available)
            changed = source.model_copy(update={"available": event.available, "event": event})
            nodes[event.node_id] = changed
            await self._save(connection, changed)
            resolved = {node_id for node_id, node in nodes.items() if node.kind == "source"}
            pending = set(nodes) - resolved
            while pending:
                ready = sorted(node_id for node_id in pending if set(nodes[node_id].parents) <= resolved)
                if not ready:
                    raise SourceLifecycleError("Invalid persisted lifecycle graph")
                for node_id in ready:
                    node = nodes[node_id]
                    available = self._supported(node, nodes)
                    if node.available != available:
                        node = node.model_copy(update={"available": available})
                        nodes[node_id] = node
                        await self._save(connection, node)
                        await self._sync_memory(connection, node)
                    resolved.add(node_id)
                    pending.remove(node_id)
            return sorted(node_id for node_id, node in nodes.items() if node.available)

    async def resolve(
        self, node_id: str, *, purpose: Literal["generation", "audit"] = "generation"
    ) -> SourceRef | MemoryCitation | ArtifactRef:
        if purpose not in {"generation", "audit"}:
            raise SourceLifecycleError("Unknown lifecycle resolution purpose")
        async with self._runtime._scoped_operation(self.scope_id), self._contexts.database.transaction() as connection:
            row = await connection.scalar(
                select(NODES.c.payload).where(NODES.c.scope_id == self.scope_id, NODES.c.node_id == node_id)
            )
            if row is None:
                raise SourceLifecycleError("Unknown lifecycle node")
            node = _Node.model_validate_json(bytes(row))
            if purpose == "generation" and not node.available:
                raise SourceLifecycleError("Lifecycle evidence is unavailable for generation")
            return node.ref()
