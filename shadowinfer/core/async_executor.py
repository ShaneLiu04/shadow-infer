"""Asynchronous task executor for ShadowInfer agents.

This module provides a thin, backend-agnostic wrapper around a thread pool.
It is used to offload work that does not need to block the main denoising
loop, such as:

- Profiler report aggregation
- A/B test statistics
- Dashboard data rendering
- Batch metric export

Version: 3.2.2
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

logger = logging.getLogger(__name__)


class AsyncAgentExecutor:
    """Thread-pool executor tailored for Agent background work.

    Args:
        max_workers: Maximum number of background threads. ``None`` lets
            Python pick a default based on the machine.
        thread_name_prefix: Prefix used for worker thread names.
        initializer: Optional callable run once in each worker thread.
        initargs: Arguments passed to ``initializer``.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        thread_name_prefix: str = "shadowinfer-async",
        initializer: Optional[Callable[..., None]] = None,
        initargs: tuple = (),
    ) -> None:
        self._lock = threading.RLock()
        self._shutdown = False
        self._submitted: int = 0
        self._completed: int = 0
        self._failed: int = 0
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
            initializer=initializer,
            initargs=initargs,
        )

    def submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Future:
        """Submit a callable to run in the background.

        Returns a ``concurrent.futures.Future`` that can be used to retrieve
        the result or attach callbacks.
        """
        with self._lock:
            if self._shutdown:
                raise RuntimeError("Cannot submit after shutdown")
            self._submitted += 1

        future: Future = self._executor.submit(fn, *args, **kwargs)

        def _on_done(f: Future) -> None:
            with self._lock:
                if f.exception() is not None:
                    self._failed += 1
                    logger.exception("Background task failed: %s", f.exception())
                else:
                    self._completed += 1

        future.add_done_callback(_on_done)
        return future

    def map(
        self,
        fn: Callable[..., Any],
        *iterables: Iterable[Any],
        timeout: Optional[float] = None,
        chunksize: int = 1,
    ) -> Iterator[Any]:
        """Apply ``fn`` to the iterables in parallel, yielding results.

        This is a thin wrapper around ``ThreadPoolExecutor.map`` that respects
        the executor's shutdown state.
        """
        with self._lock:
            if self._shutdown:
                raise RuntimeError("Cannot map after shutdown")
        return self._executor.map(fn, *iterables, timeout=timeout, chunksize=chunksize)

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        """Shut down the executor and release worker threads."""
        with self._lock:
            self._shutdown = True
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def stats(self) -> Dict[str, int]:
        """Return runtime statistics for observability."""
        with self._lock:
            return {
                "submitted": self._submitted,
                "completed": self._completed,
                "failed": self._failed,
                "pending": self._submitted - self._completed - self._failed,
            }

    def __enter__(self) -> "AsyncAgentExecutor":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.shutdown(wait=True)


class AsyncProfilerWorker:
    """Example worker that runs Profiler aggregation in the background.

    This is intentionally lightweight: it receives a raw metrics payload,
    runs an aggregation function, and stores the result.  The Orchestrator
    can poll ``get_result`` without blocking the denoising loop.
    """

    def __init__(self, executor: AsyncAgentExecutor) -> None:
        self._executor = executor
        self._pending: List[Future] = []

    def schedule_aggregation(
        self,
        aggregator: Callable[..., Dict[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Future:
        """Schedule an aggregation task and track it for cancellation."""
        future = self._executor.submit(aggregator, *args, **kwargs)
        self._pending.append(future)
        return future

    def get_result(
        self,
        timeout: Optional[float] = None,
        block: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Return the most recent completed aggregation result.

        If ``block`` is True, waits up to ``timeout`` seconds.  Non-blocking
        calls return ``None`` when no result is ready.
        """
        # Drop completed futures and keep only the latest result.
        latest: Optional[Dict[str, Any]] = None
        still_pending: List[Future] = []
        for future in self._pending:
            if future.done():
                try:
                    latest = future.result(timeout=0 if not block else timeout)
                except Exception:  # noqa: BLE001
                    logger.exception("Profiler aggregation task failed")
            else:
                still_pending.append(future)
        self._pending = still_pending
        return latest

    def cancel_all(self) -> None:
        """Cancel all pending aggregation tasks."""
        for future in self._pending:
            future.cancel()
        self._pending = []


def _example_aggregator(pairs: List[tuple]) -> Dict[str, float]:
    """Simple aggregator used in tests/examples."""
    if not pairs:
        return {"mean": 0.0, "max": 0.0}
    values = [v for _, v in pairs]
    return {"mean": sum(values) / len(values), "max": max(values)}


if __name__ == "__main__":  # pragma: no cover
    # Quick sanity run.
    with AsyncAgentExecutor(max_workers=2) as exe:
        f1 = exe.submit(_example_aggregator, [("a", 1.0), ("b", 2.0)])
        print(f1.result())
        print(exe.stats())
