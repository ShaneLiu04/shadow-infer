"""Tests for AsyncAgentExecutor."""

from __future__ import annotations

import time
import pytest

from shadowinfer.core.async_executor import (
    AsyncAgentExecutor,
    AsyncProfilerWorker,
    _example_aggregator,
)


def _slow_add(a: int, b: int, delay: float = 0.0) -> int:
    if delay:
        time.sleep(delay)
    return a + b


def _failing_task() -> None:
    raise ValueError("expected failure")


class TestAsyncAgentExecutor:
    def test_submit_and_result(self):
        with AsyncAgentExecutor(max_workers=2) as exe:
            future = exe.submit(_slow_add, 2, 3)
            assert future.result(timeout=5) == 5

    def test_map(self):
        with AsyncAgentExecutor(max_workers=2) as exe:
            results = list(exe.map(_slow_add, [1, 2, 3], [10, 20, 30]))
            assert results == [11, 22, 33]

    def test_stats_track_completed(self):
        with AsyncAgentExecutor(max_workers=2) as exe:
            f1 = exe.submit(_slow_add, 1, 1)
            f2 = exe.submit(_slow_add, 2, 2)
            f1.result(timeout=5)
            f2.result(timeout=5)
            stats = exe.stats()
            assert stats["submitted"] == 2
            assert stats["completed"] == 2
            assert stats["failed"] == 0

    def test_stats_track_failed(self):
        with AsyncAgentExecutor(max_workers=2) as exe:
            future = exe.submit(_failing_task)
            with pytest.raises(ValueError):
                future.result(timeout=5)
            stats = exe.stats()
            assert stats["submitted"] == 1
            assert stats["completed"] == 0
            assert stats["failed"] == 1

    def test_submit_after_shutdown_raises(self):
        exe = AsyncAgentExecutor(max_workers=1)
        exe.shutdown()
        with pytest.raises(RuntimeError):
            exe.submit(_slow_add, 1, 2)

    def test_context_manager(self):
        with AsyncAgentExecutor(max_workers=1) as exe:
            future = exe.submit(_slow_add, 5, 5)
            assert future.result(timeout=5) == 10


class TestAsyncProfilerWorker:
    def test_schedule_and_get_result(self):
        with AsyncAgentExecutor(max_workers=2) as exe:
            worker = AsyncProfilerWorker(exe)
            worker.schedule_aggregation(
                _example_aggregator,
                [("a", 1.0), ("b", 3.0)],
            )
            result = worker.get_result(timeout=5)
            assert result is not None
            assert result["mean"] == pytest.approx(2.0)
            assert result["max"] == pytest.approx(3.0)

    def test_get_result_non_blocking(self):
        with AsyncAgentExecutor(max_workers=2) as exe:
            worker = AsyncProfilerWorker(exe)
            worker.schedule_aggregation(_slow_add, 1, 1, delay=0.2)
            assert worker.get_result(block=False) is None
            time.sleep(0.4)
            assert worker.get_result(block=False) == 2

    def test_cancel_all(self):
        with AsyncAgentExecutor(max_workers=2) as exe:
            worker = AsyncProfilerWorker(exe)
            worker.schedule_aggregation(_slow_add, 1, 1, delay=1.0)
            worker.cancel_all()
            assert worker.get_result(timeout=0.1) is None
            assert worker._pending == []
