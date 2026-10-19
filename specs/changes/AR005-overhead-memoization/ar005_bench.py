"""AR005 baseline / A-B microbenchmark (CPU, fast)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import shadowinfer.kernels as kernels  # noqa: E402
from shadowinfer.shadowkv.packed_kv_cache import PackedKVCache  # noqa: E402

torch.manual_seed(42)


def median(values):
    return sorted(values)[len(values) // 2]


def loader_ab():
    kernels._load_cuda_extensions()
    x = torch.arange(-8, 8, dtype=torch.int8)

    start = time.perf_counter()
    for _ in range(200):
        kernels._pack_int4(x)
    memoized = (time.perf_counter() - start) / 200

    retries = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(20):
            kernels._load_attempted = False
            kernels._cuda_ops = None
            kernels._load_cuda_extensions()
        retries.append((time.perf_counter() - start) / 20)
    kernels._load_attempted = True
    kernels._cuda_ops = None
    kernels._load_cuda_extensions()

    legacy = median(retries)
    print(
        f"loader: memoized_us={memoized * 1e6:.1f} "
        f"legacy_retry_us={legacy * 1e6:.1f} ratio={memoized / legacy:.4f}"
    )


def pack_ab():
    batch, heads, seq, dim = 1, 8, 128, 128
    k = torch.randn(batch, heads, seq, dim)
    v = torch.randn(batch, heads, seq, dim)
    pmap = {
        t: {
            h: ("fp16" if (t + h) % 4 == 0 else ("int8" if (t + h) % 4 == 1 else "int4"))
            for h in range(heads)
        }
        for t in range(seq)
    }

    vec, ref = [], []
    for _ in range(3):
        start = time.perf_counter()
        PackedKVCache.pack(k, v, pmap, group_size=16)
        vec.append(time.perf_counter() - start)
        start = time.perf_counter()
        cache = PackedKVCache(batch, heads, seq, dim)
        cache._do_pack_reference(k, v, pmap, group_size=16)
        ref.append(time.perf_counter() - start)

    print(
        f"pack: vec_ms={median(vec) * 1e3:.2f} "
        f"ref_ms={median(ref) * 1e3:.2f} ratio={median(vec) / median(ref):.3f}"
    )


def golden_check():
    batch, heads, seq, dim = 1, 4, 7, 24
    k = torch.randn(batch, heads, seq, dim)
    v = torch.randn(batch, heads, seq, dim)
    pmap = {t: {h: ("int4" if h % 2 else "int8") for h in range(heads)} for t in range(seq)}
    vec = PackedKVCache.pack(k, v, pmap, group_size=16)["packed_kv"]
    ref = PackedKVCache(batch, heads, seq, dim)
    ref._do_pack_reference(k, v, pmap, group_size=16)
    for precision in ("int8", "int4"):
        assert torch.equal(vec.packed_data[precision]["k"], ref.packed_data[precision]["k"])
        assert torch.equal(vec.packed_data[precision]["v"], ref.packed_data[precision]["v"])
    assert vec.packed_data["int4"]["byte_offsets"] == ref.packed_data["int4"]["byte_offsets"]
    print("golden: int8/int4 bytes + byte_offsets identical")


loader_ab()
pack_ab()
golden_check()
