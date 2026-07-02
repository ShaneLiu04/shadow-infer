"""Thread-safe rate and concurrency limiters for production serving.

对应文档：ROADMAP.md §3.6 Production & Serving
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional


class TokenBucketRateLimiter:
    """Token-bucket rate limiter.

    Allows *burst* requests immediately, then throttles to *rate* requests per
    second.  Acquire returns ``True`` if a token is available, otherwise
    ``False`` (no blocking).
    """

    def __init__(self, rate: float, burst: int) -> None:
        if rate <= 0:
            raise ValueError("rate must be > 0")
        if burst < 1:
            raise ValueError("burst must be >= 1")
        self.rate = rate
        self.burst = burst
        self._tokens = float(burst)
        self._last_update = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        """Try to acquire one token.  Returns True on success."""
        now = time.monotonic()
        with self._lock:
            elapsed = now - self._last_update
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
            self._last_update = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    def get_state(self) -> Dict[str, float]:
        """Return current bucket state for observability."""
        with self._lock:
            return {"tokens": self._tokens, "rate": self.rate, "burst": self.burst}


class ConcurrencyLimiter:
    """Semaphore-based concurrency limiter with optional timeout."""

    def __init__(self, max_concurrent: int) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self.max_concurrent = max_concurrent
        self._semaphore = threading.Semaphore(max_concurrent)
        self._in_flight = 0
        self._lock = threading.Lock()

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """Try to enter the critical section.  Returns True if allowed."""
        acquired = self._semaphore.acquire(timeout=timeout if timeout is not None else 0)
        if acquired:
            with self._lock:
                self._in_flight += 1
        return acquired

    def release(self) -> None:
        """Leave the critical section."""
        with self._lock:
            if self._in_flight > 0:
                self._in_flight -= 1
        self._semaphore.release()

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    def get_state(self) -> Dict[str, int]:
        """Return current concurrency state."""
        with self._lock:
            return {
                "max": self.max_concurrent,
                "in_flight": self._in_flight,
                "available": self.max_concurrent - self._in_flight,
            }
