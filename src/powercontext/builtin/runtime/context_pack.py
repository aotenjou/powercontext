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

"""Deterministic UTF-8 context packets and independently verifiable prefix deltas."""

# ruff: noqa: TRY003 -- Input validation reports errors at the owning boundary.

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, cast

_PACK_SCHEMA = "powercontext.context-pack.v1"
_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"


class ContextPackError(ValueError):
    """Invalid context packet, encoding, digest, or chunk boundary."""


def _json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _encoded(items: tuple[bytes, ...]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _validate_items(items: tuple[bytes, ...]) -> None:
    if not isinstance(items, tuple) or any(not isinstance(item, bytes) for item in items):
        raise ContextPackError("Chunks must be a tuple of bytes")
    try:
        for item in items:
            item.decode("utf-8")
    except UnicodeError as error:
        raise ContextPackError("Chunks must contain valid UTF-8") from error


def _decode_items(items: object) -> tuple[bytes, ...]:
    if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
        raise ContextPackError("Wire chunks must be Base64 strings")
    try:
        result = tuple(base64.b64decode(item, validate=True) for item in cast(list[str], items))
    except (ValueError, binascii.Error) as error:
        raise ContextPackError("Invalid Base64 chunk") from error
    if _encoded(result) != items:
        raise ContextPackError("Noncanonical Base64 chunk")
    _validate_items(result)
    return result


def _read(raw: bytes, fields: set[str], schema: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ContextPackError("Duplicate wire field")
            value[key] = item
        return value

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique)
    except (UnicodeError, ValueError, AttributeError) as error:
        raise ContextPackError("Invalid context packet JSON") from error
    if not isinstance(value, dict) or set(value) != fields or value.get("schema") != schema:
        raise ContextPackError("Invalid context packet fields or schema")
    return value


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_json(value)).hexdigest()


def _validate_digest(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ContextPackError("Invalid context digest")


@dataclass(frozen=True)
class ContextPack:
    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    def __post_init__(self) -> None:
        _validate_items(self.items)
        expected = _digest({"schema": _PACK_SCHEMA, "items": _encoded(self.items)})
        if self.digest != expected or type(self.byte_count) is not int or self.byte_count != len(self.to_bytes()):
            raise ContextPackError("Context pack digest or byte count mismatch")

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        _validate_items(items)
        payload = {"schema": _PACK_SCHEMA, "items": _encoded(items)}
        payload["digest"] = _digest(payload)
        count = 0
        while True:
            payload["byte_count"] = count
            actual = len(_json(payload))
            if actual == count:
                return cls(items=items, digest=payload["digest"], byte_count=count)
            count = actual

    @classmethod
    def from_content(cls, content: str | bytes | None, *, chunk_bytes: int = 512) -> ContextPack:
        if type(chunk_bytes) is not int or chunk_bytes <= 0:
            raise ContextPackError("Chunk capacity must be a positive integer")
        raw = b"" if content is None else content.encode("utf-8") if isinstance(content, str) else content
        _validate_items((raw,))
        chunks = []
        start = 0
        while start < len(raw):
            end = min(start + chunk_bytes, len(raw))
            while end > start:
                try:
                    raw[start:end].decode("utf-8")
                    break
                except UnicodeDecodeError:
                    end -= 1
            if end == start:
                raise ContextPackError("Chunk capacity cannot hold a UTF-8 character")
            chunks.append(raw[start:end])
            start = end
        return cls.build(tuple(chunks))

    def to_bytes(self) -> bytes:
        return _json({
            "schema": _PACK_SCHEMA,
            "items": _encoded(self.items),
            "digest": self.digest,
            "byte_count": self.byte_count,
        })

    @classmethod
    def from_bytes(cls, raw: bytes) -> ContextPack:
        value = _read(raw, {"schema", "items", "digest", "byte_count"}, _PACK_SCHEMA)
        pack = cls(_decode_items(value["items"]), value["digest"], value["byte_count"])
        if len(raw) != pack.byte_count or raw != pack.to_bytes():
            raise ContextPackError("Noncanonical context pack encoding")
        return pack


@dataclass(frozen=True)
class ContextPackDelta:
    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    def __post_init__(self) -> None:
        _validate_digest(self.base_digest)
        _validate_digest(self.target_digest)
        _validate_items(self.appended)
        if type(self.prefix_items) is not int or self.prefix_items < 0:
            raise ContextPackError("Prefix length must be a nonnegative integer")

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        prefix = 0
        for left, right in zip(base.items, target.items, strict=False):
            if left != right:
                break
            prefix += 1
        return cls(base.digest, target.digest, prefix, target.items[prefix:])

    def to_bytes(self) -> bytes:
        return _json({
            "schema": _DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": _encoded(self.appended),
        })

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    @classmethod
    def from_bytes(cls, raw: bytes) -> ContextPackDelta:
        value = _read(raw, {"schema", "base_digest", "target_digest", "prefix_items", "appended"}, _DELTA_SCHEMA)
        delta = cls(
            value["base_digest"], value["target_digest"], value["prefix_items"], _decode_items(value["appended"])
        )
        return delta


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    if base.digest != delta.base_digest or delta.prefix_items > len(base.items):
        raise ContextPackError("Context delta base or prefix mismatch")
    target = ContextPack.build(base.items[: delta.prefix_items] + delta.appended)
    if target.digest != delta.target_digest:
        raise ContextPackError("Context delta target digest mismatch")
    return target
