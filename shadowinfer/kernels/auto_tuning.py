"""Persistent kernel auto-tuning cache.

This module provides a small, CPU-testable auto-tuner that records the best
launch configuration per (kernel, shape signature, dtype, device) tuple. On
machines without CUDA the tuner still works and simply stores/retrieves
cached ``TuningConfig`` values.

Version: 3.1.1
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class TuningConfig:
    """A single kernel launch configuration candidate.

    Attributes:
        block_size: CUDA block size / tile dimension.
        num_warps: Number of warps per block (or SIMT equivalent).
    """

    block_size: int = 128
    num_warps: int = 4

    def __repr__(self) -> str:
        return f"TuningConfig(block_size={self.block_size}, num_warps={self.num_warps})"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _make_key(
    kernel_name: str,
    tensor_shapes: List[Tuple[int, ...]],
    dtype: str,
    device_name: str,
) -> str:
    """Build a deterministic, shape-aware cache key."""
    shapes_str = "_".join("x".join(str(dim) for dim in shape) for shape in tensor_shapes)
    return f"{kernel_name}|{shapes_str}|{dtype}|{device_name}"


def _config_to_dict(cfg: TuningConfig) -> Dict[str, Any]:
    return asdict(cfg)


def _config_from_dict(data: Dict[str, Any]) -> TuningConfig:
    return TuningConfig(block_size=int(data["block_size"]), num_warps=int(data["num_warps"]))


class KernelCache:
    """In-memory shape-aware cache with optional size bounding.

    The cache maps a string key built from kernel metadata to the best known
    ``TuningConfig`` and its measured latency. When ``max_entries`` is
    exceeded, the oldest entries are evicted.
    """

    def __init__(self, max_entries: int = 1024) -> None:
        self.max_entries = max_entries
        self._cache: Dict[str, Tuple[TuningConfig, float]] = {}
        self._order: List[str] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def put(
        self,
        kernel_name: str,
        tensor_shapes: List[Tuple[int, ...]],
        dtype: str,
        device_name: str,
        config: TuningConfig,
        latency_ms: float = float("inf"),
    ) -> None:
        """Store or update the best config for a signature.

        If a previous entry exists with a lower latency it is kept; otherwise
        the new config replaces it. The entry is touched for LRU ordering.
        """
        key = _make_key(kernel_name, tensor_shapes, dtype, device_name)
        existing = self._cache.get(key)
        if existing is not None and existing[1] <= latency_ms:
            self._cache[key] = (existing[0], existing[1])
        else:
            self._cache[key] = (config, float(latency_ms))
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)
        self._evict_if_needed()

    def get(
        self,
        kernel_name: str,
        tensor_shapes: List[Tuple[int, ...]],
        dtype: str,
        device_name: str,
    ) -> Optional[TuningConfig]:
        """Return the best cached config, or None if no entry exists."""
        key = _make_key(kernel_name, tensor_shapes, dtype, device_name)
        entry = self._cache.get(key)
        if entry is None:
            return None
        return entry[0]

    def get_with_latency(
        self,
        kernel_name: str,
        tensor_shapes: List[Tuple[int, ...]],
        dtype: str,
        device_name: str,
    ) -> Optional[Tuple[TuningConfig, float]]:
        """Return ``(config, latency_ms)`` or None."""
        key = _make_key(kernel_name, tensor_shapes, dtype, device_name)
        return self._cache.get(key)

    def cache_size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
        self._order.clear()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Export cache as a JSON-safe dictionary."""
        return {
            key: {"config": _config_to_dict(cfg), "latency_ms": latency}
            for key, (cfg, latency) in self._cache.items()
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Import cache from a dictionary."""
        self.clear()
        for key, entry in data.items():
            if not isinstance(entry, dict):
                continue
            cfg = _config_from_dict(entry["config"])
            latency = float(entry.get("latency_ms", float("inf")))
            self._cache[key] = (cfg, latency)
            self._order.append(key)
        self._evict_if_needed()

    def from_legacy_list(self, data: List[List[Any]]) -> None:
        """Load legacy list-of-tuples cache format for backward compatibility."""
        self.clear()
        for row in data:
            if not isinstance(row, (list, tuple)) or len(row) < 3:
                continue
            key_parts, cfg_data, latency = row
            if not isinstance(key_parts, (list, tuple)) or len(key_parts) != 4:
                continue
            shape_parts = key_parts[1]
            if isinstance(shape_parts, (list, tuple)):
                if shape_parts and isinstance(shape_parts[0], int):
                    shapes = [tuple(shape_parts)]
                else:
                    shapes = [tuple(s) for s in shape_parts]
            else:
                shapes = [tuple(shape_parts)]
            key = _make_key(
                str(key_parts[0]),
                shapes,
                str(key_parts[2]),
                str(key_parts[3]),
            )
            cfg = _config_from_dict(cfg_data)
            self._cache[key] = (cfg, float(latency))
            self._order.append(key)
        self._evict_if_needed()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        while len(self._order) > self.max_entries:
            oldest = self._order.pop(0)
            self._cache.pop(oldest, None)


# ---------------------------------------------------------------------------
# Auto-tuner
# ---------------------------------------------------------------------------


class KernelAutoTuner:
    """Persistent auto-tuner for kernel launch configurations.

    The tuner benchmarks candidate ``TuningConfig`` values for a specific
    kernel invocation signature and remembers the fastest one on disk. If a
    matching cache entry already exists, ``benchmark`` returns it directly.

    Args:
        cache_path: Path to the JSON cache file. Defaults to
            ``.shadowinfer/cache/kernel_autotune_cache.json`` under the current
            working directory.
        save_interval: Number of ``benchmark`` calls between disk writes.
        max_entries: Maximum number of cache entries to keep in memory.
    """

    _DEFAULT_CANDIDATES: Tuple[TuningConfig, ...] = (
        TuningConfig(block_size=128, num_warps=4),
        TuningConfig(block_size=256, num_warps=4),
        TuningConfig(block_size=256, num_warps=8),
        TuningConfig(block_size=512, num_warps=8),
    )

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        save_interval: int = 50,
        max_entries: int = 1024,
    ) -> None:
        self.cache_path = cache_path or Path(".shadowinfer/cache/kernel_autotune_cache.json")
        self.save_interval = max(1, save_interval)
        self.max_entries = max_entries
        self._cache = KernelCache(max_entries=max_entries)
        self._lock = threading.RLock()
        self._writes_since_save = 0
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def cache(self) -> KernelCache:
        """Return the underlying ``KernelCache``."""
        return self._cache

    def cache_size(self) -> int:
        """Return the number of cached signatures."""
        with self._lock:
            return self._cache.cache_size()

    def get_best_config(
        self,
        kernel_name: str,
        tensor_shapes: List[Tuple[int, ...]],
        dtype: str,
        device_name: str,
        default: Optional[TuningConfig] = None,
    ) -> Optional[TuningConfig]:
        """Return the cached best config or ``default``."""
        with self._lock:
            cached = self._cache.get(kernel_name, tensor_shapes, dtype, device_name)
            return cached if cached is not None else default

    def benchmark(
        self,
        kernel_name: str,
        tensor_shapes: List[Tuple[int, ...]],
        dtype: str,
        device_name: str,
        benchmark_fn: Callable[[TuningConfig], float],
        candidates: Optional[Tuple[TuningConfig, ...]] = None,
        warmup: int = 2,
        repeats: int = 5,
    ) -> TuningConfig:
        """Benchmark candidates and return the fastest, using the cache.

        Args:
            kernel_name: Logical kernel identifier, e.g. ``sparse_gemm_ffn``.
            tensor_shapes: Shape tuples of the input tensors.
            dtype: String representation of the tensor dtype.
            device_name: Device name used to scope tuning results.
            benchmark_fn: Callable that measures a config and returns elapsed
                milliseconds. It receives a single ``TuningConfig`` argument.
            candidates: Tuple of configs to try. Uses defaults if None.
            warmup: Number of warmup iterations before timing.
            repeats: Number of timed iterations averaged for the result.

        Returns:
            The ``TuningConfig`` with the lowest measured latency.
        """
        with self._lock:
            cached = self._cache.get_with_latency(kernel_name, tensor_shapes, dtype, device_name)
            if cached is not None:
                return cached[0]

        configs = candidates if candidates is not None else self._DEFAULT_CANDIDATES
        if not configs:
            raise ValueError("At least one tuning candidate is required")

        best_config: Optional[TuningConfig] = None
        best_latency: float = float("inf")
        for cfg in configs:
            latency = self._time_config(benchmark_fn, cfg, warmup, repeats)
            if latency < best_latency:
                best_latency = latency
                best_config = cfg

        if best_config is None:  # pragma: no cover - guarded above
            raise ValueError("No candidate produced a valid latency")

        with self._lock:
            self._cache.put(
                kernel_name,
                tensor_shapes,
                dtype,
                device_name,
                best_config,
                best_latency,
            )
            self._writes_since_save += 1
            self._maybe_save()

        return best_config

    def save(self) -> None:
        """Persist the cache to disk immediately."""
        with self._lock:
            self._save()

    def clear(self) -> None:
        """Clear in-memory cache and delete the on-disk cache file."""
        with self._lock:
            self._cache.clear()
            self._writes_since_save = 0
            try:
                if self.cache_path.exists():
                    self.cache_path.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _time_config(
        benchmark_fn: Callable[[TuningConfig], float],
        config: TuningConfig,
        warmup: int,
        repeats: int,
    ) -> float:
        """Run warmup + repeated benchmark for a single config."""
        for _ in range(max(0, warmup)):
            benchmark_fn(config)

        total = 0.0
        for _ in range(max(1, repeats)):
            total += float(benchmark_fn(config))
        return total / max(1, repeats)

    def _maybe_save(self) -> None:
        if self._writes_since_save >= self.save_interval:
            self._save()
            self._writes_since_save = 0

    def _save(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.cache_path.with_suffix(".tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(self._cache.to_dict(), f, indent=2, sort_keys=True)
            tmp_path.replace(self.cache_path)
        except (OSError, TypeError):
            # On read-only or otherwise broken filesystems, keep running with
            # the in-memory cache rather than failing the whole pipeline.
            pass

    def _load(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            with self.cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError, TypeError):
            return

        if isinstance(data, dict):
            self._cache.from_dict(data)
        elif isinstance(data, list):
            self._cache.from_legacy_list(data)
