"""AR012 canonical packed KV store benchmark (real tensors, CPU).

Run: python specs/changes/AR012-canonical-kv-store/ar012_bench.py
"""

import statistics
import time

import torch

from shadowinfer.shadowkv.packed_kv_cache import PackedKVCache

PRECISIONS = ("fp32", "fp16", "int8", "int4")


def mixed_map(seq_len, heads):
    return {
        t: {h: PRECISIONS[(t * heads + h) % len(PRECISIONS)] for h in range(heads)}
        for t in range(seq_len)
    }


def timed(fn, repeats=3):
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        values.append(time.perf_counter() - start)
    return statistics.median(values)


def reference_reuse(packed, previous, mask):
    for precision, data in packed.packed_data.items():
        prev_data = previous.packed_data.get(precision)
        if prev_data is None:
            continue
        prev_map = {entry: i for i, entry in enumerate(prev_data["indices"])}
        for position, entry in enumerate(data["indices"]):
            if not mask.get(entry, False) or entry not in prev_map:
                continue
            source = prev_map[entry]
            if precision in ("fp32", "fp16", "int8"):
                data["k"][position] = prev_data["k"][source]
                data["v"][position] = prev_data["v"][source]
                if precision == "int8":
                    data["scale_k"][position] = prev_data["scale_k"][source]
                    data["scale_v"][position] = prev_data["scale_v"][source]
            else:
                k0, k1, v0, v1 = data["byte_offsets"][position]
                p0, p1, q0, q1 = prev_data["byte_offsets"][source]
                data["k"][k0:k1] = prev_data["k"][p0:p1]
                data["v"][v0:v1] = prev_data["v"][q0:q1]
                data["scale_k"][position] = prev_data["scale_k"][source]
                data["scale_v"][position] = prev_data["scale_v"][source]


def reference_evict(k, v, precision_map, evicted, group_size):
    unpacked = PackedKVCache.pack(k, v, precision_map, group_size=group_size)["packed_kv"]
    k_full, v_full = unpacked.unpack_reference()
    keep_map = {
        t: {
            h: ("evicted" if (t, h) in evicted else precision_map[t][h])
            for h in range(k.shape[1])
        }
        for t in range(k.shape[2])
    }
    return PackedKVCache.pack(k_full, v_full, keep_map, group_size=group_size)


def run(seq_len, batch=1, heads=8, head_dim=64):
    torch.manual_seed(12012)
    k = torch.randn(batch, heads, seq_len, head_dim)
    v = torch.randn(batch, heads, seq_len, head_dim)
    precision_map = mixed_map(seq_len, heads)
    packed = PackedKVCache.pack(k, v, precision_map, group_size=16)["packed_kv"]

    k_ref, _ = packed.unpack_reference()
    k_vec, _ = packed.unpack()
    assert torch.equal(k_ref, k_vec)

    unpack_ref = timed(packed.unpack_reference)
    unpack_vec = timed(packed.unpack)

    previous = PackedKVCache.pack(k + 0.1, v + 0.1, precision_map, group_size=16)["packed_kv"]
    mask = {(t, h): (t + h) % 2 == 0 for t in range(seq_len) for h in range(heads)}
    reuse_vec = timed(lambda: PackedKVCache.pack(k, v, precision_map, group_size=16)[
        "packed_kv"
    ].apply_reuse_mask(previous, mask))
    reuse_ref = timed(lambda: reference_reuse(
        PackedKVCache.pack(k, v, precision_map, group_size=16)["packed_kv"], previous, mask
    ))

    evicted = set()
    for data in packed.packed_data.values():
        evicted.update(data["indices"][::4])
    evict_vec = timed(lambda: PackedKVCache.pack(k, v, precision_map, group_size=16)[
        "packed_kv"
    ].evict(evicted))
    evict_ref = timed(lambda: reference_evict(k, v, precision_map, evicted, 16))

    before_bytes = packed.memory_bytes()
    target = PackedKVCache.pack(k, v, precision_map, group_size=16)["packed_kv"]
    target.evict(evicted)
    dropped = 1.0 - target.memory_bytes() / before_bytes

    print(
        f"S={seq_len:<5} unpack={unpack_vec * 1e3:8.3f}ms ref={unpack_ref * 1e3:9.3f}ms "
        f"x{unpack_ref / unpack_vec:6.1f} | reuse={reuse_vec * 1e3:7.3f}ms "
        f"ref={reuse_ref * 1e3:8.3f}ms x{reuse_ref / reuse_vec:5.1f} | "
        f"evict={evict_vec * 1e3:6.3f}ms ref={evict_ref * 1e3:8.3f}ms "
        f"x{evict_ref / evict_vec:6.1f} | bytes -{dropped * 100:.1f}%"
    )
    return unpack_ref / unpack_vec, reuse_ref / reuse_vec, evict_ref / evict_vec


def int4_batch_error(seq_len=8, batch=2, heads=2, head_dim=64):
    torch.manual_seed(12013)
    k = torch.randn(batch, heads, seq_len, head_dim)
    v = torch.randn(batch, heads, seq_len, head_dim)
    precision_map = {t: {h: "int4" for h in range(heads)} for t in range(seq_len)}
    packed = PackedKVCache.pack(k, v, precision_map, group_size=16)["packed_kv"]
    k_back, v_back = packed.unpack()
    errors = []
    for idx in range(batch):
        rel = float(((k[idx] - k_back[idx]) ** 2).mean() / (k[idx] ** 2).mean())
        errors.append(rel)
        assert rel <= 0.02, f"batch {idx} relative MSE {rel:.4f}"
    print(f"INT4 B={batch} per-batch K relative MSE: {['%.4f' % e for e in errors]}")


if __name__ == "__main__":
    print("AR012 canonical KV store benchmark (CPU, real tensors, median of 3)")
    results = [run(size) for size in (512, 1024, 2048)]
    for unpack_ratio, reuse_ratio, evict_ratio in results:
        assert unpack_ratio >= 5.0
        assert reuse_ratio >= 3.0
        assert evict_ratio >= 10.0
    int4_batch_error()
    print("all AR012 benchmark assertions passed")
