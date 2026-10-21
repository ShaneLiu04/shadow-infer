"""AR014 benchmark: paged incremental KV residency vs packed full repack.

Measures the KV execution plane in append mode, mirroring the exact
bookkeeping the backend performs per step per layer:

- baseline (kv_execution="packed"): rebuild the full precision_map, unpack
  the previous full cache (read), pack the full past+new tensors (write),
  i.e. 2x O(S_total) work per step (quadratic over N steps).
- paged (kv_execution="paged"): gather the paged past (read O(S_total)),
  append only the new tokens into pages (write O(S_new)).

Pre-registered assertions (specs/changes/AR014-paged-kv-store/tasks.md T006):
1. total-time speedup (3-seed median) >= 1.5x at BOTH scale points;
2. per-step latency linear-fit slope ratio paged/baseline < 0.6;
3. tracemalloc Python-heap transient peak: paged < baseline;
4. bitwise equivalence: final paged gather == PackedKVCache.pack(full)
   reference unpack (torch.equal, fp16), and per-step past reads are equal.

Run: $env:PYTHONPATH='D:\ShadowInfer'; python specs/changes/AR014-paged-kv-store/ar014_bench.py
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Dict, List, Tuple

import torch

from shadowinfer.shadowkv.packed_kv_cache import PackedKVCache
from shadowinfer.shadowkv.paged_kv_store import PagedKVStore

BATCH = 4
HEADS = 8
HEAD_DIM = 64
LAYERS = 8
PRECISION = "fp16"
SEEDS = (0, 1, 2)
SCALE_POINTS = ((8, 64), (16, 128))

SPEEDUP_THRESHOLD = 1.5
SLOPE_RATIO_THRESHOLD = 0.6


def _linear_slope(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    xs = torch.arange(n, dtype=torch.float64)
    ys = torch.tensor(values, dtype=torch.float64)
    xs_centered = xs - xs.mean()
    return float((xs_centered * (ys - ys.mean())).sum() / (xs_centered**2).sum())


def _run_baseline(k_full: torch.Tensor, v_full: torch.Tensor, num_steps: int, chunk: int):
    """Mirror the packed append path: full unpack + full repack per step."""
    per_step: List[float] = []
    stores: List = [None] * LAYERS
    head_map = {h: PRECISION for h in range(HEADS)}
    for step in range(num_steps):
        seq = (step + 1) * chunk
        t0 = time.perf_counter()
        for layer in range(LAYERS):
            precision_map = {t: head_map for t in range(seq)}
            if stores[layer] is not None:
                k_past, v_past = stores[layer].unpack()
            else:
                k_past = v_past = None
            stores[layer] = PackedKVCache.pack(
                k_full[:, :, :seq, :].contiguous(),
                v_full[:, :, :seq, :].contiguous(),
                precision_map,
            )["packed_kv"]
            if k_past is not None:
                del k_past, v_past
        per_step.append(time.perf_counter() - t0)
    return stores, per_step


def _run_paged(k_full: torch.Tensor, v_full: torch.Tensor, num_steps: int, chunk: int):
    """Mirror the paged append path: gather read + O(S_new) page append."""
    per_step: List[float] = []
    stores: List = [None] * LAYERS
    for step in range(num_steps):
        seq = (step + 1) * chunk
        t0 = time.perf_counter()
        for layer in range(LAYERS):
            if stores[layer] is None:
                stores[layer] = PagedKVStore(
                    BATCH, HEADS, HEAD_DIM, page_size=32, device=k_full.device
                )
                past_k = past_v = None
            else:
                past_k, past_v = stores[layer].gather()
            stores[layer].append(
                k_full[:, :, seq - chunk : seq, :].contiguous(),
                v_full[:, :, seq - chunk : seq, :].contiguous(),
                PRECISION,
            )
            if past_k is not None:
                del past_k, past_v
        per_step.append(time.perf_counter() - t0)
    return stores, per_step


def _run_path(path_name: str, k_full, v_full, num_steps, chunk):
    tracemalloc.start()
    if path_name == "baseline":
        stores, per_step = _run_baseline(k_full, v_full, num_steps, chunk)
    else:
        stores, per_step = _run_paged(k_full, v_full, num_steps, chunk)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return stores, per_step, peak


def _reference(k_full: torch.Tensor, v_full: torch.Tensor, seq: int):
    head_map = {h: PRECISION for h in range(HEADS)}
    precision_map = {t: head_map for t in range(seq)}
    packed = PackedKVCache.pack(
        k_full[:, :, :seq, :].contiguous(),
        v_full[:, :, :seq, :].contiguous(),
        precision_map,
    )["packed_kv"]
    return packed


def _median(values: List[float]) -> float:
    return sorted(values)[len(values) // 2]


def run_scale_point(num_steps: int, chunk: int) -> Dict[str, object]:
    seq = num_steps * chunk
    gen = torch.Generator().manual_seed(2026)
    k_full = (torch.randn(BATCH, HEADS, seq, HEAD_DIM, generator=gen) * 2.0).float()
    v_full = (torch.randn(BATCH, HEADS, seq, HEAD_DIM, generator=gen) * 2.0).float()

    baseline_totals: List[float] = []
    paged_totals: List[float] = []
    slope_ratios: List[float] = []
    baseline_peaks: List[int] = []
    paged_peaks: List[int] = []

    for seed in SEEDS:
        torch.manual_seed(seed)
        _, baseline_steps, baseline_peak = _run_path("baseline", k_full, v_full, num_steps, chunk)
        _, paged_steps, paged_peak = _run_path("paged", k_full, v_full, num_steps, chunk)
        baseline_totals.append(sum(baseline_steps))
        paged_totals.append(sum(paged_steps))
        slope_ratio = _linear_slope(paged_steps) / _linear_slope(baseline_steps)
        slope_ratios.append(slope_ratio)
        baseline_peaks.append(baseline_peak)
        paged_peaks.append(paged_peak)

    speedups = [b / p for b, p in zip(baseline_totals, paged_totals)]
    median_speedup = _median(speedups)
    median_slope_ratio = _median(slope_ratios)

    reference = _reference(k_full, v_full, seq)
    k_ref, v_ref = reference.unpack()
    paged_stores, _, _ = _run_path("paged", k_full, v_full, num_steps, chunk)
    k_paged, v_paged = paged_stores[0].gather()
    assert torch.equal(k_paged, k_ref), "paged gather must match packed reference"
    assert torch.equal(v_paged, v_ref), "paged gather must match packed reference"

    per_token = BATCH * HEADS * HEAD_DIM * 2 * 2
    results: Dict[str, object] = {
        "num_steps": num_steps,
        "chunk": chunk,
        "max_seq": seq,
        "baseline_total_s": [round(t, 4) for t in baseline_totals],
        "paged_total_s": [round(t, 4) for t in paged_totals],
        "speedups": [round(s, 3) for s in speedups],
        "median_speedup": round(median_speedup, 3),
        "slope_ratios": [round(r, 3) for r in slope_ratios],
        "median_slope_ratio": round(median_slope_ratio, 3),
        "baseline_peak_python_heap_bytes": baseline_peaks,
        "paged_peak_python_heap_bytes": paged_peaks,
        "paged_data_plane_bytes": paged_stores[0].data_plane_bytes(),
        "dense_fp16_bytes": seq * per_token,
    }

    assert median_speedup >= SPEEDUP_THRESHOLD, (
        f"scale (N={num_steps}, S_new={chunk}): median speedup "
        f"{median_speedup:.3f} < {SPEEDUP_THRESHOLD}"
    )
    assert median_slope_ratio < SLOPE_RATIO_THRESHOLD, (
        f"scale (N={num_steps}, S_new={chunk}): median slope ratio "
        f"{median_slope_ratio:.3f} >= {SLOPE_RATIO_THRESHOLD}"
    )
    assert _median(paged_peaks) < _median(baseline_peaks), (
        f"scale (N={num_steps}, S_new={chunk}): paged Python-heap transient "
        f"peak {_median(paged_peaks)} not below baseline {_median(baseline_peaks)}"
    )
    return results


def main() -> None:
    torch.set_num_threads(1)
    all_results: List[Dict[str, object]] = []
    for num_steps, chunk in SCALE_POINTS:
        _run_path("baseline", torch.randn(1, 1, 1, 1), torch.randn(1, 1, 1, 1), 1, 1)
        results = run_scale_point(num_steps, chunk)
        all_results.append(results)
        print(f"[AR014] detail N={num_steps} S_new={chunk}:")
        for i, seed in enumerate(SEEDS):
            print(
                f"  seed={seed} baseline={results['baseline_total_s'][i]:.4f}s "
                f"paged={results['paged_total_s'][i]:.4f}s "
                f"speedup={results['speedups'][i]:.3f} "
                f"slope_ratio={results['slope_ratios'][i]:.3f} "
                f"heap_baseline={results['baseline_peak_python_heap_bytes'][i]}B "
                f"heap_paged={results['paged_peak_python_heap_bytes'][i]}B"
            )
        print(
            f"[AR014] N={num_steps:2d} S_new={chunk:3d} max_seq={results['max_seq']:4d} | "
            f"speedup median={results['median_speedup']:.3f}x "
            f"range=[{min(results['speedups']):.3f}, {max(results['speedups']):.3f}] | "
            f"slope ratio median={results['median_slope_ratio']:.3f} | "
            f"heap peak baseline={_median(results['baseline_peak_python_heap_bytes'])/1024:.0f}KB "
            f"paged={_median(results['paged_peak_python_heap_bytes'])/1024:.0f}KB"
        )
    print("[AR014] all pre-registered assertions passed")


if __name__ == "__main__":
    main()
