"""Tests for the kernel auto-tuning cache and dispatcher.

Covers shape-aware cache keys, persistence, thread safety, and CPU fallback
preservation. Designed to run on CPU-only Windows CI.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, wait

import torch

from shadowinfer import kernels as K
from shadowinfer.kernels.auto_tuning import KernelAutoTuner, KernelCache, TuningConfig
from shadowinfer.kernels.dispatcher import KernelDispatcher

# ---------------------------------------------------------------------------
# TuningConfig / KernelCache basics
# ---------------------------------------------------------------------------


def test_tuning_config_as_key():
    """TuningConfig instances must be hashable and comparable."""
    c1 = TuningConfig(block_size=128, num_warps=4)
    c2 = TuningConfig(block_size=128, num_warps=4)
    c3 = TuningConfig(block_size=256, num_warps=4)
    assert c1 == c2
    assert hash(c1) == hash(c2)
    assert c1 != c3
    assert repr(c1) == "TuningConfig(block_size=128, num_warps=4)"


def test_kernel_cache_shape_aware_key():
    """Cache keys must distinguish kernel, shape signature, dtype and device."""
    cache = KernelCache()
    cfg = TuningConfig(128, 4)
    cache.put("k1", [(2, 3), (3, 4)], "float32", "cpu", cfg)
    assert cache.get("k1", [(2, 3), (3, 4)], "float32", "cpu") == cfg
    assert cache.get("k1", [(2, 3), (3, 4)], "float16", "cpu") is None
    assert cache.get("k1", [(2, 3), (3, 5)], "float32", "cpu") is None
    assert cache.get("k1", [(2, 3), (3, 4)], "float32", "cuda:0") is None
    assert cache.get("k2", [(2, 3), (3, 4)], "float32", "cpu") is None


def test_kernel_cache_get_best_returns_fastest():
    """get_best must return the config with the lowest stored latency."""
    cache = KernelCache()
    fast = TuningConfig(64, 2)
    slow = TuningConfig(128, 4)
    cache.put("k", [(4, 4)], "float32", "cpu", fast, latency_ms=1.0)
    cache.put("k", [(4, 4)], "float32", "cpu", slow, latency_ms=10.0)
    best = cache.get("k", [(4, 4)], "float32", "cpu")
    assert best == fast


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_tuner_persists_cache(tmp_path):
    """The tuner must write and reload its cache as JSON."""
    cache_path = tmp_path / "cache.json"
    tuner = KernelAutoTuner(cache_path=cache_path, save_interval=1)

    def bench_fn(cfg):
        return float(cfg.block_size)

    best = tuner.benchmark(
        "gemm",
        tensor_shapes=[(32, 32), (32, 32)],
        dtype="float32",
        device_name="cpu",
        benchmark_fn=bench_fn,
        candidates=(TuningConfig(64, 2), TuningConfig(128, 2)),
        warmup=0,
        repeats=1,
    )
    assert best == TuningConfig(64, 2)
    assert cache_path.exists()

    # Reload into a fresh tuner instance.
    tuner2 = KernelAutoTuner(cache_path=cache_path)
    assert tuner2.cache_size() == 1
    cached = tuner2.get_best_config("gemm", [(32, 32), (32, 32)], "float32", "cpu")
    assert cached == best


def test_tuner_loads_malformed_cache_gracefully(tmp_path):
    """A corrupted cache file must not break tuner construction."""
    cache_path = tmp_path / "cache.json"
    cache_path.write_text("not json", encoding="utf-8")
    tuner = KernelAutoTuner(cache_path=cache_path)
    assert tuner.cache_size() == 0


def test_tuner_loads_legacy_list_format(tmp_path):
    """Backward compatibility: legacy list-of-tuples cache must load."""
    cache_path = tmp_path / "cache.json"
    # Legacy format was a flat list of [key_tuple, config_dict, latency].
    legacy = [[["kernel", [1, 2], "float32", "cpu"], {"block_size": 64, "num_warps": 2}, 1.5]]
    cache_path.write_text(json.dumps(legacy), encoding="utf-8")
    tuner = KernelAutoTuner(cache_path=cache_path)
    assert tuner.cache_size() == 1
    best = tuner.get_best_config("kernel", [(1, 2)], "float32", "cpu")
    assert best == TuningConfig(64, 2)


# ---------------------------------------------------------------------------
# Benchmarking and candidate selection
# ---------------------------------------------------------------------------


def test_benchmark_selects_fastest_candidate(tmp_path):
    """benchmark must run all candidates and return the one with lowest latency."""
    tuner = KernelAutoTuner(cache_path=tmp_path / "cache.json", save_interval=1000)

    latencies = {
        TuningConfig(64, 2): 5.0,
        TuningConfig(128, 2): 2.0,
        TuningConfig(256, 4): 8.0,
    }

    def bench_fn(cfg):
        return latencies[cfg]

    best = tuner.benchmark(
        "add",
        tensor_shapes=[(16, 16)],
        dtype="float32",
        device_name="cpu",
        benchmark_fn=bench_fn,
        candidates=tuple(latencies.keys()),
        warmup=0,
        repeats=1,
    )
    assert best == TuningConfig(128, 2)


def test_benchmark_returns_cached_result_without_rerunning(tmp_path):
    """A cached result must be returned without invoking benchmark_fn."""
    tuner = KernelAutoTuner(cache_path=tmp_path / "cache.json", save_interval=1000)
    tuner.cache.put("add", [(8,)], "float32", "cpu", TuningConfig(64, 2), 0.1)

    def bench_fn(_):
        raise AssertionError("should not be called")

    best = tuner.benchmark(
        "add",
        tensor_shapes=[(8,)],
        dtype="float32",
        device_name="cpu",
        benchmark_fn=bench_fn,
        warmup=0,
        repeats=1,
    )
    assert best == TuningConfig(64, 2)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_concurrent_benchmark_for_same_key_is_consistent(tmp_path):
    """Concurrent calls for the same signature must not corrupt the cache."""
    tuner = KernelAutoTuner(cache_path=tmp_path / "cache.json", save_interval=1000)
    candidates = (TuningConfig(64, 2), TuningConfig(128, 2))

    def bench_fn(cfg):
        time.sleep(0.005)
        return float(cfg.block_size)

    def task():
        return tuner.benchmark(
            "conv",
            tensor_shapes=[(4, 4)],
            dtype="float32",
            device_name="cpu",
            benchmark_fn=bench_fn,
            candidates=candidates,
            warmup=0,
            repeats=1,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(task) for _ in range(16)]
        wait(futures)
        results = [f.result() for f in futures]

    assert all(r == TuningConfig(64, 2) for r in results)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_dispatcher_force_cpu(monkeypatch):
    """SHADOWINFER_FORCE_CPU_KERNELS must disable CUDA dispatch."""
    monkeypatch.setenv("SHADOWINFER_FORCE_CPU_KERNELS", "1")
    dispatcher = KernelDispatcher()
    assert dispatcher.force_cpu() is True

    tensor = torch.randn(4, 4)
    assert dispatcher.use_cuda(tensor, object()) is False

    monkeypatch.delenv("SHADOWINFER_FORCE_CPU_KERNELS", raising=False)


def test_dispatcher_use_cuda_requires_cuda_tensor_and_ops():
    """use_cuda must be false for CPU tensors or missing CUDA ops."""
    dispatcher = KernelDispatcher()
    cpu_tensor = torch.randn(4, 4)
    assert dispatcher.use_cuda(cpu_tensor, object()) is False
    assert dispatcher.use_cuda(cpu_tensor, None) is False


# ---------------------------------------------------------------------------
# Integration with shadowinfer.kernels
# ---------------------------------------------------------------------------


def test_kernel_status_reports_autotune_metadata():
    """get_kernel_status must expose auto-tuning cache size and force_cpu flag."""
    status = K.get_kernel_status()
    assert "autotune_cache_entries" in status
    assert "force_cpu" in status
    assert isinstance(status["autotune_cache_entries"], int)
    assert isinstance(status["force_cpu"], bool)


def test_cpu_fallback_unchanged_by_autotune_integration():
    """Quantization/dequantization must remain accurate on CPU."""
    x = torch.randn(16, 32)
    q, scale, zp = K.quantize_per_channel_int8(x)
    x_hat = K.dequantize_per_channel_int8(q, scale, zp)
    assert torch.allclose(x_hat, x, atol=0.05)

    q4, s4, zp4 = K.quantize_per_channel_int4(x, axis=0)
    x4_hat = K.dequantize_per_channel_int4(q4, s4, zp4, axis=0, original_shape=x.shape)
    # INT4 is much coarser than INT8.
    assert torch.allclose(x4_hat, x, atol=0.5)


# ---------------------------------------------------------------------------
# Cache pruning
# ---------------------------------------------------------------------------


def test_cache_evicts_oldest_entries():
    """The cache must evict oldest entries when max_entries is exceeded."""
    cache = KernelCache(max_entries=3)
    for i in range(5):
        cache.put(f"k{i}", [(1,)], "float32", "cpu", TuningConfig(i, 1), float(i))
    assert cache.cache_size() == 3
    assert cache.get("k0", [(1,)], "float32", "cpu") is None
    assert cache.get("k1", [(1,)], "float32", "cpu") is None
    assert cache.get("k2", [(1,)], "float32", "cpu") is not None
