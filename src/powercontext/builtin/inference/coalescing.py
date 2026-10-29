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

"""Share in-flight embedding work and cache successful vectors with bounded LRU storage."""

# ruff: noqa: TRY003 -- Input validation reports errors at the owning boundary.

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel

_Completion = tuple[EmbeddingVector, int, InferenceUsage]


class CoalescingEmbeddingModel:
    def __init__(
        self, model: EmbeddingModel, *, max_batch_size: int = 64, coalesce_ms: float = 5.0, cache_size: int = 1024
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size < 1 or type(cache_size) is not int or cache_size < 0:
            raise ValueError("Invalid embedding batch or cache capacity")
        if type(coalesce_ms) not in {int, float} or not math.isfinite(coalesce_ms) or coalesce_ms < 0:
            raise ValueError("Invalid embedding coalescing window")
        self.profile = model.profile
        self._model = model
        self._max_batch_size, self._delay, self._cache_size = max_batch_size, coalesce_ms / 1000, cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._inflight: dict[str, asyncio.Future[_Completion]] = {}
        self._pending: dict[str, asyncio.Future[_Completion]] = {}
        self._scheduled = False
        self._tasks: set[asyncio.Task[None]] = set()
        self._batch_id = 0

    def _start(self, coroutine) -> None:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=())
        cached: dict[str, EmbeddingVector] = {}
        waiting: dict[str, asyncio.Future[_Completion]] = {}
        for text in texts:
            if text in self._cache:
                self._cache.move_to_end(text)
                cached[text] = self._cache[text]
            elif text not in waiting:
                future = self._inflight.get(text)
                if future is None:
                    future = asyncio.get_running_loop().create_future()
                    # A cancelled sole waiter can leave a failed computation without an observer.
                    future.add_done_callback(lambda done: None if done.cancelled() else done.exception())
                    self._inflight[text] = self._pending[text] = future
                waiting[text] = future
        if self._pending and not self._scheduled:
            self._scheduled = True
            self._start(self._dispatch())
        completions = await asyncio.gather(*(asyncio.shield(future) for future in waiting.values()))
        usage_by_batch: dict[int, InferenceUsage] = {}
        vectors = dict(cached)
        for text, (vector, batch_id, usage) in zip(waiting, completions, strict=True):
            vectors[text] = vector
            usage_by_batch[batch_id] = usage
        usages = tuple(usage_by_batch.values())
        tokens = lambda field: (
            sum(getattr(usage, field) or 0 for usage in usages)
            if any(getattr(usage, field) is not None for usage in usages)
            else None
        )
        return EmbeddingResult(
            vectors=tuple(vectors[text] for text in texts),
            usage=InferenceUsage(
                requests=sum(usage.requests for usage in usages),
                input_tokens=tokens("input_tokens"),
                output_tokens=tokens("output_tokens"),
            ),
        )

    async def _dispatch(self) -> None:
        await asyncio.sleep(self._delay)
        pending, self._pending = self._pending, {}
        self._scheduled = False
        items = tuple(pending.items())
        for start in range(0, len(items), self._max_batch_size):
            self._batch_id += 1
            self._start(self._compute(items[start : start + self._max_batch_size], self._batch_id))

    async def _compute(self, items: tuple[tuple[str, asyncio.Future[_Completion]], ...], batch_id: int) -> None:
        try:
            result = await self._model.embed(tuple(text for text, _ in items))
            if len(result.vectors) != len(items):
                raise ValueError("Embedding provider returned the wrong vector count")  # noqa: TRY301 - propagate batch validation to its waiters
            for (text, future), vector in zip(items, result.vectors, strict=True):
                self._cache_vector(text, vector)
                future.set_result((vector, batch_id, result.usage))
        except asyncio.CancelledError:
            for _, future in items:
                future.cancel()
            raise
        except Exception as error:
            for _, future in items:
                if not future.done():
                    future.set_exception(error)
        finally:
            for text, future in items:
                if self._inflight.get(text) is future:
                    del self._inflight[text]

    def _cache_vector(self, text: str, vector: EmbeddingVector) -> None:
        if self._cache_size:
            self._cache[text] = vector
            self._cache.move_to_end(text)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
