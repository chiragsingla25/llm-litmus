"""Small helper for running async calls with a concurrency limit."""

import asyncio
from collections.abc import Awaitable, Sequence
from typing import TypeVar

T = TypeVar("T")


async def gather_with_concurrency(limit: int, coros: Sequence[Awaitable[T]]) -> list[T]:
    """Runs the given coroutines concurrently, at most `limit` in flight at once.

    Results keep the same order as `coros`. An item that raised is returned as the exception
    object itself, not re-raised -- callers must check `isinstance(result, Exception)`.
    """
    semaphore = asyncio.Semaphore(limit)

    async def _run(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    return await asyncio.gather(*(_run(c) for c in coros), return_exceptions=True)
