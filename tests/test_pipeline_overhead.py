"""AR005 pipeline overhead tests (loader memoization + batched INT4 packing)."""

from __future__ import annotations

import threading
import time
import warnings

import pytest
import torch

import shadowinfer.kernels as kernels
from shadowinfer.shadowkv.packed_kv_cache import PackedKVCache


@pytest.fixture(autouse=True)
def _restore_kernel_state():
    saved = (
        kernels._cuda_ops,
        kernels._load_attempted,
        kernels._warned_no_cuda,
        kernels._CUDA_AVAILABLE,
    )
    yield
    (
        kernels._cuda_ops,
        kernels._load_attempted,
        kernels._warned_no_cuda,
        kernels._CUDA_AVAILABLE,
    ) = saved


def _reset_loader(available=False):
    kernels._cuda_ops = None
    kernels._load_attempted = False
    kernels._warned_no_cuda = False
    kernels._CUDA_AVAILABLE = available


def test_ut1_loader_attempted_once(monkeypatch):
    _reset_loader()
    calls = {"n": 0}

    def fake_once():
        calls["n"] += 1
        return None

    monkeypatch.setattr(kernels, "_load_cuda_extensions_once", fake_once)
    for _ in range(200):
        assert kernels._load_cuda_extensions() is None
    assert calls["n"] == 1


def test_ut2_concurrent_first_call(monkeypatch):
    _reset_loader()
    calls = {"n": 0}
    started = threading.Event()

    def slow_once():
        calls["n"] += 1
        started.set()
        time.sleep(0.05)
        return None

    monkeypatch.setattr(kernels, "_load_cuda_extensions_once", slow_once)
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(kernels._load_cuda_extensions()))
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls["n"] == 1
    assert results == [None] * 8


def test_ut3_unavailable_warns_once(monkeypatch):
    _reset_loader(available=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            assert kernels._load_cuda_extensions() is None
    loader_warnings = [w for w in caught if "CUDA not available" in str(w.message)]
    assert len(loader_warnings) == 1


def _precision_map(seq, heads, pattern):
    return {t: {h: pattern(t, h) for h in range(heads)} for t in range(seq)}


def _assert_pack_golden(batch, heads, seq, dim, group_size):
    torch.manual_seed(0)
    k = torch.randn(batch, heads, seq, dim)
    v = torch.randn(batch, heads, seq, dim)
    pmap = _precision_map(
        seq,
        heads,
        lambda t, h: "fp16" if (t + h) % 5 == 0 else ("int8" if (t + h) % 5 == 1 else "int4"),
    )

    vec = PackedKVCache.pack(k, v, pmap, group_size=group_size)["packed_kv"]
    ref = PackedKVCache(batch, heads, seq, dim)
    ref._do_pack_reference(k, v, pmap, group_size=group_size)

    assert set(vec.packed_data) == set(ref.packed_data)
    for precision, data in ref.packed_data.items():
        assert torch.equal(vec.packed_data[precision]["k"], data["k"]), precision
        assert torch.equal(vec.packed_data[precision]["v"], data["v"]), precision
        assert vec.packed_data[precision]["indices"] == data["indices"]
    assert vec.packed_data["int4"]["byte_offsets"] == ref.packed_data["int4"]["byte_offsets"]
    assert torch.equal(vec.packed_data["int4"]["scale_k"], ref.packed_data["int4"]["scale_k"])
    assert vec.memory_bytes() == ref.memory_bytes()


def test_ut4_pack_golden_bytes_identical():
    for args in (
        (1, 3, 5, 16, 16),
        (1, 2, 7, 24, 16),
        (2, 2, 9, 32, 16),
        (1, 4, 16, 128, 16),
    ):
        _assert_pack_golden(*args)


def test_ut5_pack_vectorized_speedup():
    batch, heads, seq, dim = 1, 4, 64, 64
    torch.manual_seed(0)
    k = torch.randn(batch, heads, seq, dim)
    v = torch.randn(batch, heads, seq, dim)
    pmap = _precision_map(seq, heads, lambda t, h: "int4" if (t + h) % 2 == 0 else "int8")

    def timed(fn, n=3):
        samples = []
        for _ in range(n):
            start = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - start)
        return sorted(samples)[len(samples) // 2]

    vec = timed(lambda: PackedKVCache.pack(k, v, pmap, group_size=16))
    ref = timed(
        lambda: PackedKVCache(batch, heads, seq, dim)._do_pack_reference(k, v, pmap, group_size=16)
    )
    assert vec <= ref * 0.5


def test_ut6_pipeline_smoke_after_optimization(tmp_path):
    from shadowinfer.orchestrator import Orchestrator

    model_config = {
        "num_layers": 2,
        "num_heads": 2,
        "head_dim": 16,
        "hidden_dim": 32,
        "intermediate_dim": 64,
        "vocab_size": 64,
        "max_seq_len": 32,
        "batch_size": 1,
        "seq_len": 8,
    }
    orch = Orchestrator(config={"output_dir": str(tmp_path)})
    orch.initialize(dict(model_config))
    try:
        result = orch.run_optimized(prompt="ar005", num_steps=2)
    finally:
        orch.shutdown()

    assert isinstance(result, dict)
    assert len(orch.step_results) == 2
    assert all("step_id" in sr for sr in orch.step_results)
