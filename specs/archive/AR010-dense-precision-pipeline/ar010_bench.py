"""AR010 benchmark: dict precision pipeline vs dense int8 codes (CPU).

Compares the v3.11 path (``allocate_precision`` dict + ``admit_budget`` +
``PrecisionPlan.build``) with the AR010 path (``allocate_precision_codes`` +
``admit_codes`` + ``PrecisionPlan.from_codes``) on real tensors. Verifies that
both paths produce the same map, predicted bytes, eviction count and plan
fingerprint, then reports the median of 3 repeats.

Usage:
    python specs/changes/AR010-dense-precision-pipeline/ar010_bench.py
"""

import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from shadowinfer.shadowkv.precision_plan import (  # noqa: E402
    PrecisionPlan,
    codes_to_map,
)
from shadowinfer.shadowkv.rd_allocator import (  # noqa: E402
    entry_bytes,
    predict_codes_bytes,
)
from shadowinfer.shadowkv.shadowkv_agent import ShadowKVAgent  # noqa: E402

CFG = {
    "num_layers": 1,
    "num_heads": 32,
    "head_dim": 128,
    "sink_tokens": 4,
    "use_packed_cache": False,
    "rd_allocation_enabled": True,
    "hard_eviction_enabled": False,
}
SCALES = (512, 2048, 4096, 8192)
REPEATS = 3


def _measure(fn, repeats=REPEATS):
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), min(samples), max(samples), result


def main() -> None:
    torch.manual_seed(0)
    for seq_len in SCALES:
        heads, head_dim, batch = 32, 128, 1
        importance = torch.rand(seq_len, heads)
        key = torch.randn(1, heads, seq_len, head_dim)
        value = torch.randn(1, heads, seq_len, head_dim)
        full_bytes = int(seq_len * heads * entry_bytes("fp32", batch, head_dim))
        budget = int(full_bytes * 0.18)

        agent = ShadowKVAgent(config=CFG)
        agent.on_init(dict(CFG, model_name="ar010-bench"))
        agent.cache_manager.memory_budget_bytes = budget

        def baseline():
            precision_map = agent.allocate_precision(importance, "balanced")
            admitted = agent._admit_precision_budget(
                importance, precision_map, {"k": key, "v": value}
            )
            plan = PrecisionPlan.build(admitted, seq_len, heads)
            return admitted, plan

        def dense():
            codes = agent.allocate_precision_codes(importance, "balanced")
            admitted = agent._admit_precision_codes(codes, importance, {"k": key, "v": value})
            plan = PrecisionPlan.from_codes(admitted)
            return admitted, plan

        base_ms, base_min, base_max, base = _measure(baseline)
        dense_ms, dense_min, dense_max, fused = _measure(dense)
        bmap, bplan = base
        dcodes, dplan = fused
        assert codes_to_map(dcodes) == bmap
        assert dplan.fingerprint == bplan.fingerprint
        assert predict_codes_bytes(dcodes, batch, head_dim) == agent._estimate_precision_bytes(
            bmap, batch, seq_len, head_dim
        )
        predicted = predict_codes_bytes(dcodes, batch, head_dim)
        print(
            f"seq={seq_len:5d} baseline={base_ms:8.1f}ms [{base_min:.1f}-{base_max:.1f}] "
            f"dense={dense_ms:7.1f}ms [{dense_min:.1f}-{dense_max:.1f}] "
            f"speedup={base_ms / dense_ms:5.2f}x budget={budget} predicted={predicted}"
        )


if __name__ == "__main__":
    main()
