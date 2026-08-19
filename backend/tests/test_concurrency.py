import asyncio

from app.core.concurrency import gather_with_concurrency


async def _delay_return(value, delay):
    await asyncio.sleep(delay)
    return value


def test_results_preserve_input_order_regardless_of_completion_order():
    # First coro is slowest, last is fastest -- if order weren't preserved this would fail.
    coros = [
        _delay_return("a", 0.03),
        _delay_return("b", 0.02),
        _delay_return("c", 0.01),
    ]
    results = asyncio.run(gather_with_concurrency(3, coros))
    assert results == ["a", "b", "c"]


def test_never_exceeds_concurrency_limit():
    current = 0
    peak = 0

    async def _tracked():
        nonlocal current, peak
        current += 1
        peak = max(peak, current)
        await asyncio.sleep(0.02)
        current -= 1
        return "done"

    coros = [_tracked() for _ in range(10)]
    results = asyncio.run(gather_with_concurrency(3, coros))

    assert peak <= 3
    assert results == ["done"] * 10


def test_exception_from_one_coro_is_returned_not_raised():
    async def _raise():
        raise ValueError("boom")

    async def _succeed():
        return "ok"

    coros = [_succeed(), _raise(), _succeed()]
    results = asyncio.run(gather_with_concurrency(3, coros))

    assert results[0] == "ok"
    assert isinstance(results[1], ValueError)
    assert str(results[1]) == "boom"
    assert results[2] == "ok"
