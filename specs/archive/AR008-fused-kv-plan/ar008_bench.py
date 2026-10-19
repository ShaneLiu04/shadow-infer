"""AR008 benchmark: legacy compress+pack vs fused precision plan (CPU).

Real tensors only. Reports median/min/max over 5 repeats and verifies the
fused outputs are bitwise identical to the legacy paths.

Usage:
    python specs/archive/AR008-fused-kv-plan/ar008_bench.py
"""

import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from shadowinfer.core.config import Config  # noqa: E402
from shadowinfer.shadowkv.packed_kv_cache import PackedKVCache  # noqa: E402
from shadowinfer.shadowkv.precision_plan import PrecisionPlan  # noqa: E402
from shadowinfer.shadowkv.shadowkv_agent import ShadowKVAgent  # noqa: E402

PRECISIONS = ("fp32", "fp16", "int8", "int4")


def make_precision_map(seq_len: int, num_heads: int) -> dict:
    return {
        token_idx: {
            head_idx: PRECISIONS[(token_idx + head_idx) % len(PRECISIONS)]
            for head_idx in range(num_heads)
        }
        for token_idx in range(seq_len)
    }


def make_reuse_mask(seq_len: int, num_heads: int) -> dict:
    return {
        (token_idx, head_idx): ((token_idx + head_idx) % 7 == 0)
        for token_idx in range(seq_len)
        for head_idx in range(num_heads)
    }


def bench(fn, repeats: int = 5):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), min(samples), max(samples)


def main() -> None:
    torch.manual_seed(42)
    agent = ShadowKVAgent(Config({}))

    shapes = [(256, 8, 64), (512, 8, 64), (1024, 8, 64), (512, 16, 128)]
    print("shape | compress legacy->plan | pack legacy->plan | both legacy->fused")
    for seq_len, num_heads, head_dim in shapes:
        agent.on_init(
            {
                "num_layers": 4,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "model_name": "ar008-bench",
                "run_id": "bench",
            }
        )
        k = torch.randn(1, num_heads, seq_len, head_dim)
        v = torch.randn(1, num_heads, seq_len, head_dim)
        k_prev = torch.randn(1, num_heads, seq_len, head_dim)
        v_prev = torch.randn(1, num_heads, seq_len, head_dim)
        precision_map = make_precision_map(seq_len, num_heads)
        reuse_mask = make_reuse_mask(seq_len, num_heads)
        plan = PrecisionPlan.build(precision_map, seq_len, num_heads, reuse_mask=reuse_mask)

        def compress_legacy():
            return agent._compress_kv_vectorized(k, v, k_prev, v_prev, precision_map, reuse_mask)

        def compress_plan():
            return agent._compress_kv_vectorized(
                k,
                v,
                k_prev,
                v_prev,
                precision_map,
                reuse_mask,
                plan=plan,
                build_scale_map=False,
            )

        legacy_kv = compress_legacy()
        plan_kv = compress_plan()
        assert torch.equal(legacy_kv["k"], plan_kv["k"])
        assert torch.equal(legacy_kv["v"], plan_kv["v"])

        pack_plan = PrecisionPlan.build(precision_map, seq_len, num_heads)
        payload = agent._compress_kv_vectorized(
            k, v, None, None, precision_map, None, plan=pack_plan, build_scale_map=False
        )["group_payload"]

        def pack_legacy():
            return PackedKVCache.pack(k, v, precision_map)

        def pack_plan_path():
            return PackedKVCache.pack(k, v, precision_map, plan=pack_plan, precomputed=payload)

        legacy_packed = pack_legacy()["packed_kv"]
        fused_packed = pack_plan_path()["packed_kv"]
        assert legacy_packed.memory_bytes() == fused_packed.memory_bytes()
        for precision, data in legacy_packed.packed_data.items():
            other = fused_packed.packed_data[precision]
            for key, value in data.items():
                if torch.is_tensor(value):
                    assert torch.equal(value, other[key]), (precision, key)
                else:
                    assert value == other[key], (precision, key)

        def both_legacy():
            compress_legacy()
            pack_legacy()

        def both_fused():
            plan_local = PrecisionPlan.build(
                precision_map, seq_len, num_heads, reuse_mask=reuse_mask
            )
            core = agent._compress_kv_vectorized(
                k,
                v,
                k_prev,
                v_prev,
                precision_map,
                reuse_mask,
                plan=plan_local,
                build_scale_map=False,
            )
            PackedKVCache.pack(
                k, v, precision_map, plan=plan_local, precomputed=core["group_payload"]
            )

        c_legacy = bench(compress_legacy)
        c_plan = bench(compress_plan)
        p_legacy = bench(pack_legacy)
        p_plan = bench(pack_plan_path)
        b_legacy = bench(both_legacy)
        b_fused = bench(both_fused)
        print(
            f"{seq_len}x{num_heads}x{head_dim} | "
            f"{c_legacy[0]:.2f}->{c_plan[0]:.2f} ms "
            f"({c_legacy[0] / c_plan[0]:.2f}x) | "
            f"{p_legacy[0]:.2f}->{p_plan[0]:.2f} ms "
            f"({p_legacy[0] / p_plan[0]:.2f}x) | "
            f"{b_legacy[0]:.2f}->{b_fused[0]:.2f} ms "
            f"({b_legacy[0] / b_fused[0]:.2f}x) | "
            f"ranges c[{c_legacy[1]:.2f},{c_legacy[2]:.2f}] "
            f"b[{b_legacy[1]:.2f},{b_legacy[2]:.2f}]"
        )


if __name__ == "__main__":
    main()
