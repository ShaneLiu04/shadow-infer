"""AR009 benchmark: importance-only vs rate-distortion KV allocation (CPU).

Uses a real ``SimpleDiffusionLLM`` forward (randomly initialized weights, real
attention/KV tensors) and the production ``admit_budget`` allocator. Reports
attention-weighted KV reconstruction error at equal byte budgets, budget
compliance, and allocator overhead.

Usage:
    python specs/archive/AR009-rd-kv-allocation/ar009_bench.py
"""

import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from shadowinfer.models import SimpleDiffusionLLM  # noqa: E402
from shadowinfer.shadowkv.rd_allocator import EVICTED, admit_budget, kv_sensitivity  # noqa: E402

LEVELS = {"fp16": 1 << 15, "int8": 127, "int4": 7}


def quant_error(x: torch.Tensor, precision: str) -> float:
    if precision == EVICTED:
        return float(x.pow(2).mean())
    if precision == "fp32":
        return 0.0
    if precision == "fp16":
        return float((x.half().float() - x).pow(2).mean())
    levels = LEVELS[precision]
    scale = float(x.abs().max()) / levels
    if scale == 0.0:
        return 0.0
    quantized = torch.round(x / scale).clamp(-levels - 1, levels).to(torch.int8)
    return float((quantized.float() * scale - x).pow(2).mean())


def collect_layers(seed: int, num_layers=8, heads=8, head_dim=64, seq_len=128):
    torch.manual_seed(seed)
    model = SimpleDiffusionLLM(
        num_layers=num_layers,
        num_heads=heads,
        head_dim=head_dim,
        hidden_dim=heads * head_dim,
        vocab_size=500,
        max_seq_len=seq_len,
        device=torch.device("cpu"),
    )
    ids = torch.randint(0, 500, (1, seq_len))
    hidden = model.embed(ids) + model.pos_embed(torch.arange(seq_len)).unsqueeze(0)
    layers = []
    for block in model.blocks:
        hidden, (k, v), attention = block(hidden, None)
        importance = attention[0].sum(dim=1).transpose(0, 1).contiguous()  # [seq, heads]
        layers.append({"k": k[0], "v": v[0], "importance": importance})
    return layers


def evaluate(layers, batch, head_dim):
    numerator = denominator = 0.0
    for layer in layers:
        for token_idx in range(layer["importance"].shape[0]):
            for head_idx in range(layer["importance"].shape[1]):
                precision = layer["precision_map"][token_idx][head_idx]
                importance = float(layer["importance"][token_idx, head_idx]) + 1e-6
                err = quant_error(layer["k"][head_idx, token_idx], precision) + quant_error(
                    layer["v"][head_idx, token_idx], precision
                )
                energy = float(
                    layer["k"][head_idx, token_idx].pow(2).mean()
                    + layer["v"][head_idx, token_idx].pow(2).mean()
                )
                numerator += importance * err
                denominator += importance * energy
    return numerator / denominator


def run(seed, total_budget, rd, evict, head_dim=64):
    layers = collect_layers(seed)
    layer_budget = int(total_budget) // len(layers)
    predicted_total = 0
    allocator_ms = 0.0
    for layer in layers:
        seq_len, heads = layer["importance"].shape
        initial = {
            token_idx: {head_idx: "fp32" for head_idx in range(heads)}
            for token_idx in range(seq_len)
        }
        sensitivity = kv_sensitivity(layer["k"].unsqueeze(0), layer["v"].unsqueeze(0))
        started = time.perf_counter()
        precision_map, predicted, _ = admit_budget(
            layer["importance"],
            sensitivity,
            initial,
            layer_budget,
            batch=1,
            head_dim=head_dim,
            sink_tokens=4,
            rd_allocation=rd,
            allow_eviction=evict,
        )
        allocator_ms += (time.perf_counter() - started) * 1000.0
        layer["precision_map"] = precision_map
        predicted_total += predicted
    return evaluate(layers, 1, head_dim), predicted_total, allocator_ms


def main() -> None:
    print("equal-budget comparison (8 layers, 8 heads, dim64, seq128, 5 seeds)")
    for total_budget in (900_000, 1_200_000, 1_500_000):
        base, rd = [], []
        base_bytes, rd_bytes = [], []
        base_ms, rd_ms = [], []
        for seed in range(5):
            err, used, ms = run(seed, total_budget, rd=False, evict=False)
            base.append(err)
            base_bytes.append(used)
            base_ms.append(ms)
            err, used, ms = run(seed, total_budget, rd=True, evict=False)
            rd.append(err)
            rd_bytes.append(used)
            rd_ms.append(ms)
        gain = (1 - statistics.median(rd) / statistics.median(base)) * 100
        print(
            f" budget={total_budget} baseline_err={statistics.median(base):.6f} "
            f"rd_err={statistics.median(rd):.6f} gain={gain:+.2f}% "
            f"bytes(base/rd)={max(base_bytes)}/{max(rd_bytes)} "
            f"alloc_ms(base/rd)={statistics.median(base_ms):.1f}/"
            f"{statistics.median(rd_ms):.1f}"
        )

    print("sub-int4-floor budgets (hard eviction, 5 seeds)")
    for total_budget in (300_000, 500_000, 700_000):
        rows = []
        for seed in range(5):
            base_err, base_used, _ = run(seed, total_budget, rd=False, evict=False)
            rd_err, rd_used, overhead = run(seed, total_budget, rd=True, evict=True)
            rows.append((base_err, base_used, rd_err, rd_used, overhead))
        print(
            f" budget={total_budget} baseline_bytes={max(r[1] for r in rows)} "
            f"(over={max(r[1] for r in rows) / total_budget:.2f}x) "
            f"rd_bytes={max(r[3] for r in rows)} "
            f"(fit={max(r[3] for r in rows) <= total_budget}) "
            f"rd_err_median={statistics.median(r[2] for r in rows):.6f} "
            f"allocator_ms_median={statistics.median(r[4] for r in rows):.2f}"
        )


if __name__ == "__main__":
    main()
