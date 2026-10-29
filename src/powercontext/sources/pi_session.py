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

"""Capture immutable evidence from a Pi v3 session's selected history branch."""

# ruff: noqa: TRY003 -- Input validation reports errors at the owning boundary.

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.sources import Source, SourceDefinition, SourceMaterialization, SourceProjection, TextEvidence
from powercontext.sources.connectors import ConnectorRunCompletion, ConnectorRunSession, ConnectorRunStatus


class PiSessionInput(BaseModel):
    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, Any]


def _records(value: PiSessionInput) -> tuple[str, dict[str, dict[str, Any]]]:
    try:
        records = [json.loads(line) for line in value.content.decode("utf-8-sig").splitlines() if line.strip()]
    except (UnicodeError, ValueError) as error:
        raise ValueError("Invalid Pi session JSONL") from error
    if not records or not isinstance(records[0], dict):
        raise ValueError("Missing Pi session header")
    header = records[0]
    if header.get("type") != "session" or type(header.get("version")) is not int or header["version"] != 3:
        raise ValueError("Expected a Pi v3 session")
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("Missing Pi session identity")
    nodes: dict[str, dict[str, Any]] = {}
    for record in records[1:]:
        if not isinstance(record, dict):
            raise ValueError("Invalid Pi node")  # noqa: TRY004 - malformed JSONL has a ValueError contract
        identity = record.get("id")
        parent = record.get("parentId")
        if (
            not isinstance(identity, str)
            or not identity.strip()
            or identity in nodes
            or "parentId" not in record
            or (parent is not None and (not isinstance(parent, str) or parent not in nodes))
        ):
            raise ValueError("Invalid Pi node identity or parent")
        nodes[identity] = record
    if value.leaf_id is not None and value.leaf_id not in nodes:
        raise ValueError("Unknown Pi leaf")
    return session_id, nodes


def _visible_blocks(content: Any, *, include_tools: bool = False) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    blocks = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            blocks.append(block["text"])
        elif include_tools and block.get("type") == "toolCall":
            arguments = json.dumps(
                block.get("arguments", {}), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            blocks.append(f"[tool_call {block.get('name', '')} {block.get('id', '')}] {arguments}")
    return "\n".join(blocks)


def _event_body(record: dict[str, Any]) -> tuple[str, str]:
    kind = record.get("type", "")
    if kind == "message":
        message = record.get("message", {})
        if not isinstance(message, dict):
            raise ValueError("Invalid Pi message")
        role = message.get("role", "")
        body = _visible_blocks(message.get("content"), include_tools=True)
        if role == "toolResult":
            body = f"[tool_result {message.get('toolName', '')} {message.get('toolCallId', '')}] {body}"
        elif role == "bashExecution":
            body = f"$ {message.get('command', '')}\n{message.get('output', '')}"
    elif kind in {"compaction", "branch_summary"}:
        role, body = kind, record.get("summary", "")
    elif kind == "custom_message":
        role, body = "custom", _visible_blocks(record.get("content"))
    else:
        return "", ""
    if not isinstance(body, str) or not isinstance(role, str):
        raise ValueError("Invalid visible Pi event")  # noqa: TRY004 - malformed sessions raise ValueError
    return role, body


def _events(session_id: str, nodes: dict[str, dict[str, Any]], leaf_id: str | None) -> tuple[PiEventInput, ...]:
    chain = []
    leaf = leaf_id if leaf_id is not None else next(reversed(nodes), None)
    while leaf is not None:
        record = nodes[leaf]
        chain.append(record)
        leaf = record["parentId"]
    events = []
    for record in reversed(chain):
        role, body = _event_body(record)
        if body:
            events.append(
                PiEventInput(
                    session_id=session_id,
                    event_id=record["id"],
                    parent_id=record["parentId"],
                    event_type=record["type"],
                    role=role,
                    content=f"{role}: {body}",
                    record=record,
                )
            )
    return tuple(events)


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    session_id, nodes = _records(value)
    return _events(session_id, nodes, value.leaf_id)


class PiEventSource(Source):
    event: PiEventInput


class _PiProjection:
    name: str = "powercontext.text-evidence"
    version: str = "1"
    source_class: type[PiEventSource] = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        event = source.event
        return TextEvidence(
            source_type="pi-event",
            source_id=source.name,
            content=event.content,
            metadata={
                key: getattr(event, key) for key in ("session_id", "event_id", "parent_id", "event_type", "role")
            },
        )


class _PiDefinition:
    name: str = "pi-event"
    version: str = "1"
    input_class: type[PiEventInput] = PiEventInput
    source_class: type[PiEventSource] = PiEventSource
    projections: tuple[SourceProjection[PiEventSource], ...] = (_PiProjection(),)

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        identity = f"pi:event:{quote(value.session_id, safe='')}:{quote(value.event_id, safe='')}"
        if len(identity) > 256:
            # A canonical tuple hash retains identity for arbitrarily long provider IDs.
            identity = (
                "pi:sha256:"
                + hashlib.sha256(
                    json.dumps([value.session_id, value.event_id], ensure_ascii=False).encode()
                ).hexdigest()
            )
        return PiEventSource(name=identity, materialization=SourceMaterialization.CAPTURED, event=value)

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return source.event


PI_EVENT_SOURCE_DEFINITION: SourceDefinition[PiEventInput, PiEventSource, PiEventInput] = _PiDefinition()


class PiSessionConnector:
    name: str = "pi-session-history"
    version: str = "1"
    source_definitions = frozenset({"pi-event"})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self.path, self.leaf_id = Path(path), leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        value = PiSessionInput(content=self.path.read_bytes(), leaf_id=self.leaf_id)
        session_id, nodes = _records(value)
        events = _events(session_id, nodes, self.leaf_id)
        previous = session.checkpoint
        if previous is not None and (not isinstance(previous, dict) or previous.get("session_id") != session_id):
            raise ValueError("Pi checkpoint belongs to another session")
        imported = dict(previous.get("imported", {})) if isinstance(previous, dict) else {}
        fingerprints = {
            identity: hashlib.sha256(
                json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            for identity, record in nodes.items()
        }
        for identity, digest in imported.items():
            if fingerprints.get(identity) != digest:
                raise ValueError("Previously captured Pi event changed or disappeared")
        for event in events:
            if event.event_id not in imported:
                source = await PI_EVENT_SOURCE_DEFINITION.resolve(event)
                await session.submit(source.name, "pi-event", event)
                imported[event.event_id] = fingerprints[event.event_id]
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE, checkpoint={"session_id": session_id, "imported": imported}
        )
