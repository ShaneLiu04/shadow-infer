"""AR011 benchmark: executable INT8 FFN (CPU).

Measures FP32 vs INT8 linear at 7B FFN shapes and small-model crossover, checks
weight memory, and validates logits against the FP32 golden on a real
``SimpleDiffusionLLM``. No mocks; all numbers come from real tensors.

Usage:
    python specs/changes/AR011-int8-ffn-execution/ar011_bench.py
"""

import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from shadowinfer.ffn_optimizer.int8_linear import INT8Linear  # noqa: E402
from shadowinfer.models import SimpleDiffusionLLM  # noqa: E402

FFN_SHAPES = (
    (1, 4096, 11008),
    (16, 4096, 11008),
    (64, 4096, 11008),
    (128, 4096, 11008),
    (64, 11008, 4096),
)
SMALL_SHAPES = ((128, 128, 512), (512, 256, 1024))
REPEATS = 3


def _measure(fn, repeats=REPEATS):
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), min(samples), max(samples), result


def _relative_l2(actual, reference):
    return (
        torch.linalg.vector_norm(actual.float() - reference.float())
        / torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
    ).item()


def _bench_linear(shape):
    m, k, n = shape
    torch.manual_seed(m + k + n)
    linear = nn.Linear(k, n, bias=True)
    int8_linear = INT8Linear.from_linear(linear)
    x = torch.randn(m, k)
    fp_ms, fp_min, fp_max, reference = _measure(lambda: F.linear(x, linear.weight, linear.bias))
    i8_ms, i8_min, i8_max, output = _measure(lambda: int8_linear(x))
    error = _relative_l2(output, reference)
    ratio = int8_linear.dense_equivalent_bytes() / int8_linear.memory_bytes()
    print(
        f"linear M={m:4d} K={k:5d} N={n:5d} fp32={fp_ms:7.3f}ms [{fp_min:.3f}-{fp_max:.3f}] "
        f"int8={i8_ms:7.3f}ms [{i8_min:.3f}-{i8_max:.3f}] speedup={fp_ms / i8_ms:5.2f}x "
        f"mem={ratio:4.2f}x err={error:.4%}"
    )
    assert int8_linear.memory_bytes() == (
        int8_linear.qweight.numel()
        + 4 * int8_linear.weight_scale.numel()
        + (4 * int8_linear.bias.numel() if int8_linear.bias is not None else 0)
    )
    return fp_ms / i8_ms, fp_ms, i8_ms, error


def _bench_model(seed):
    torch.manual_seed(seed)
    model = SimpleDiffusionLLM(
        num_layers=4,
        num_heads=8,
        head_dim=64,
        hidden_dim=512,
        intermediate_dim=2048,
        vocab_size=256,
        max_seq_len=128,
    )
    model.eval()
    x = torch.randn(1, 128, 512)
    with torch.no_grad():
        reference = model.forward_step(x, step_t=2, total_steps=8)[0].clone()
    model.enable_int8_ffn(keep_fp32_reference=True)
    with torch.no_grad():
        output = model.forward_step(x, step_t=2, total_steps=8)[0]
    error = _relative_l2(output, reference)
    top1 = (output.argmax(-1) == reference.argmax(-1)).float().mean().item()
    print(f"model seed={seed} logits_rel_l2={error:.4%} top1={top1:.2%}")
    return error, top1


def _bench_medium_model():
    torch.manual_seed(7)
    model = SimpleDiffusionLLM(
        num_layers=4,
        num_heads=16,
        head_dim=64,
        hidden_dim=1024,
        intermediate_dim=4096,
        vocab_size=1024,
        max_seq_len=256,
    )
    model.eval()
    x = torch.randn(1, 256, 1024)

    def _forward():
        with torch.no_grad():
            return model.forward_step(x, step_t=2, total_steps=8)[0]

    fp_ms, fp_min, fp_max, reference = _measure(_forward)
    model.enable_int8_ffn()
    i8_ms, i8_min, i8_max, output = _measure(_forward)
    error = _relative_l2(output, reference)
    print(
        f"model dim=1024 seq=256 fp32={fp_ms:7.1f}ms [{fp_min:.1f}-{fp_max:.1f}] "
        f"int8={i8_ms:7.1f}ms [{i8_min:.1f}-{i8_max:.1f}] speedup={fp_ms / i8_ms:5.2f}x "
        f"err={error:.4%}"
    )
    return fp_ms / i8_ms, error


def main() -> None:
    print("== 7B / large FFN shapes ==")
    large_speedups = []
    for shape in FFN_SHAPES:
        speedup, _, _, _ = _bench_linear(shape)
        large_speedups.append(speedup)
        m, k, n = shape
        if m >= 16 and k == 4096 and n == 11008:
            assert speedup >= 3.0, f"expected >=3x for 7B up-projection shape {shape}"

    print("== small shapes (crossover) ==")
    for shape in SMALL_SHAPES:
        _bench_linear(shape)

    print("== medium model end-to-end ==")
    model_speedup, model_error = _bench_medium_model()

    print("== real 4-layer model logits (3 seeds) ==")
    model_errors = []
    top1_scores = []
    for seed in (0, 1, 2):
        error, top1 = _bench_model(seed)
        model_errors.append(error)
        top1_scores.append(top1)
    print(
        f"model median rel_l2={statistics.median(model_errors):.4%} "
        f"median top1={statistics.median(top1_scores):.2%}"
    )

    assert max(large_speedups[:4]) >= 3.0, "expected >=3x on at least one 7B shape"
    assert model_speedup >= 1.0, "medium-model INT8 should not be slower"
    assert model_error <= 0.03, "medium-model logits error exceeds 3%"
    assert statistics.median(model_errors) <= 0.03, "median logits error exceeds 3%"
    assert statistics.median(top1_scores) >= 0.95, "median top-1 agreement below 95%"


if __name__ == "__main__":
    main()
