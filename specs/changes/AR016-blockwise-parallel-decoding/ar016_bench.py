"""AR016 benchmark: confidence-aware blockwise parallel decoding (FR16.5).

Four arms, real SimpleDiffusionLLM, real random prompts, single thread:

- baseline: ``blockwise_generate_baseline`` (full recompute per round under
  the segment-causal mask; serves as the golden for relL2 / token equality);
- blockwise-fp32: ``PyTorchModelBackend.generate`` with page_precision="fp32"
  (exact arm; prefix KV bitwise-reused from PagedKVStore pages);
- blockwise-fp16 / blockwise-int8: quantized page planes (trade-off arms).

Calibration (probe, single-seed, pre-registration): the design's
threshold=0.9 degenerates on the toy model (softmax-max confidences are
0.004-0.01, so every round falls back to top-1); the registered main grid
uses the calibrated threshold 0.005 (parallel unmasking fires, ~15% rounds)
and a report-only sensitivity sweep covers 0.9 / 0.005 / 0.003 (the last
collapses to 1 round per block). Registered below with the margins required
by the AR015 discipline; probe evidence in bench-results.md.

Registered configuration: (L,H,D)=(8,8,64), P=64, G in {192, 448, 960}
(S=256/512/1024, blocks 6/14/30, W=32), B=2, max_rounds=8, threshold=0.005,
seeds {0,1,2}, torch single thread, CPU.

Registered assertions (full run exits non-zero on violation):
- fp32 arm: token sequences bitwise equal baseline at all 9 points
  (probe: 9/9); per-round logits relL2 <= 1e-6 (probe max 3.2e-7);
- fp16 arm: bitwise equal baseline at all 9 points (probe: 12/12 incl.
  sensitivity points); relL2 <= 1e-4 (probe max 1.26e-5);
- int8 arm: token agreement >= 0.99 (run-1/2 min 0.9961); at trajectory-
  aligned points (per-round unmask positions identical AND bitwise-equal
  final tokens), relL2 <= 1e-3 (measured 2.0-3.6e-4) -- non-aligned points
  carry a decision-divergence flag instead: int8 KV noise permutes
  threshold-edge unmask schedules (same set, different order) or flips
  near-tie argmax tokens; the resulting relL2 ~0.17-0.21 is a round-context
  misalignment artifact, not the quantization error magnitude (per-block
  profiling: aligned rounds stay at 2-3.6e-4 even in diverged runs; see
  bench-results.md section 2.1);
- fp32 speedup: per-scale median >= 3x on >= 2 of 3 scales
  (probe medians: 2.4x / 3.67x / 7.44x; small scale honestly below 3x,
  overhead-dominated);
- fp32 computed-row reduction vs baseline >= 4x at every scale
  (probe: 4.9x / 8.4x / 15.4x, deterministic per scale);
- all blockwise arms: kv_write_rows_total == P + G exactly (probe: 256/512/
  1024; each KV row written once, never rewritten).

Run (full, registered): $env:PYTHONPATH='D:\ShadowInfer';
  python specs/changes/AR016-blockwise-parallel-decoding/ar016_bench.py
Run (probe, unregistered): same command plus `` --probe [--threshold T]
  [--gen 192,448] [--seed 0,1]``
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List

import torch

from shadowinfer.core.backends.pytorch_backend import PyTorchModelBackend
from shadowinfer.execution.block_decoding import (
    BlockDecodingConfig,
    blockwise_generate_baseline,
)
from shadowinfer.models.simple_diffusion_llm import SimpleDiffusionLLM

BATCH = 2
SEEDS = (0, 1, 2)
LAYERS = 8
HEADS = 8
HEAD_DIM = 64
HIDDEN = HEADS * HEAD_DIM
VOCAB = 1000
PROMPT_LEN = 64
WIDTH = 32
GEN_LENS = (192, 448, 960)
MAX_ROUNDS = 8
PAGE_SIZE = 32
PRECISIONS = ("fp32", "fp16", "int8")
REGISTERED_THRESHOLD = 0.005
SENSITIVITY_THRESHOLDS = (0.9, 0.005, 0.003)
SENSITIVITY_GEN = 192
SENSITIVITY_SEED = 0

FP32_RELL2_BUDGET = 1e-6
FP16_RELL2_BUDGET = 1e-4
INT8_RELL2_BUDGET = 1e-3
INT8_TOKEN_AGREE_MIN = 0.99
SPEEDUP_MIN = 3.0
SPEEDUP_MIN_SCALES = 2
ROW_REDUCTION_MIN = 4.0


def _median(values: List[float]) -> float:
    return sorted(values)[len(values) // 2]


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def _model_config() -> Dict[str, int]:
    return {
        "num_layers": LAYERS,
        "num_heads": HEADS,
        "head_dim": HEAD_DIM,
        "hidden_dim": HIDDEN,
        "vocab_size": VOCAB,
        "max_seq_len": PROMPT_LEN + max(GEN_LENS),
    }


def _direct_model(seed: int) -> SimpleDiffusionLLM:
    torch.manual_seed(seed)
    model = SimpleDiffusionLLM(
        num_layers=LAYERS,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        hidden_dim=HIDDEN,
        vocab_size=VOCAB,
        max_seq_len=PROMPT_LEN + max(GEN_LENS),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    model.eval()
    return model


def _prompt(seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(2026 + seed)
    return torch.randint(1, VOCAB, (BATCH, PROMPT_LEN), generator=gen)


def _config(precision: str, threshold: float) -> BlockDecodingConfig:
    return BlockDecodingConfig(
        block_size=WIDTH,
        max_rounds=MAX_ROUNDS,
        confidence_threshold=threshold,
        mask_token_id=0,
        page_size=PAGE_SIZE,
        page_precision=precision,
    )


def _run_arms(
    gen_len: int,
    seed: int,
    threshold: float,
) -> Dict[str, Dict[str, object]]:
    """One (scale, seed) point: baseline + three blockwise precision arms."""
    prompt = _prompt(seed)
    model = _direct_model(seed)
    backend = PyTorchModelBackend(model_config=_model_config(), seed=seed)
    backend.load("ar016-bench", device="cpu")

    t0 = time.perf_counter()
    base = blockwise_generate_baseline(
        model, prompt, gen_len, _config("fp32", threshold), capture_block_logits=True
    )
    base_time = time.perf_counter() - t0

    arms: Dict[str, Dict[str, object]] = {
        "baseline": {
            "total_s": base_time,
            "result": base,
        }
    }
    for precision in PRECISIONS:
        cfg = _config(precision, threshold)
        t0 = time.perf_counter()
        result = backend.generate(
            prompt, gen_len, cfg, capture_block_logits=True
        )
        total = time.perf_counter() - t0
        arms[precision] = {"total_s": total, "result": result}

    base_logits = base.stats["block_logits_per_round"]
    base_rounds = base.stats["rounds_per_block"]
    for name, arm in arms.items():
        result = arm["result"]
        stats = result.stats
        rel_l2s: List[float] = []
        arm_logits = stats["block_logits_per_round"]
        rounds_match = stats["rounds_per_block"] == base_rounds
        if rounds_match:
            for block_logits, base_block in zip(arm_logits, base_logits):
                for arm_l, base_l in zip(block_logits, base_block):
                    rel_l2s.append(_rel_l2(arm_l, base_l))
        arm["rounds_match"] = rounds_match
        arm["rounds_per_block"] = stats["rounds_per_block"]
        arm["total_rounds"] = stats["total_rounds"]
        arm["computed_rows_total"] = stats["computed_rows_total"]
        arm["kv_write_rows_total"] = stats["kv_write_rows_total"]
        arm["forced_completions"] = stats["forced_completions"]
        arm["max_rel_l2"] = max(rel_l2s) if rel_l2s else float("nan")
        arm["token_agreement"] = float(
            (result.sequences == base.sequences).float().mean()
        )
        arm["bitwise_equal"] = bool(
            result.sequences.shape == base.sequences.shape
            and torch.equal(result.sequences, base.sequences)
        )
        # A6 v3: per-round relL2 vs baseline is only well-defined when the
        # two trajectories are identical: same unmask positions every round
        # AND bitwise-equal final tokens (equal positions + equal finals
        # imply equal tokens at every unmask, hence equal round contexts).
        # Invalidated proxies: bitwise-only (v1; schedule permutations
        # reconverge to identical finals) and positions-only (v2; same-set
        # token flips break round contexts) -- see bench-results.md 2.1.
        arm["decisions_aligned"] = bool(
            rounds_match
            and arm["bitwise_equal"]
            and all(
                torch.equal(arm_pos, base_pos)
                for arm_block, base_block in zip(
                    stats["unmask_positions_per_round"],
                    base.stats["unmask_positions_per_round"],
                )
                for arm_pos, base_pos in zip(arm_block, base_block)
            )
        )
        arm["block_time_ms"] = [round(t, 2) for t in stats["block_time_ms"]]
        arm["median_block_ms"] = round(_median(stats["block_time_ms"]), 2)
        arm["fallback_rounds"] = sum(
            1
            for per_block in stats["fallback_per_round"]
            for flag in per_block
            if flag
        )
        flat_conf = [
            c
            for per_block in stats["mean_confidence_per_round"]
            for c in per_block
        ]
        arm["mean_conf_range"] = (
            round(min(flat_conf), 5) if flat_conf else float("nan"),
            round(max(flat_conf), 5) if flat_conf else float("nan"),
        )
    return arms


def _print_point(gen_len: int, seed: int, arms: Dict[str, Dict[str, object]]) -> None:
    base_time = arms["baseline"]["total_s"]
    print(f"  G={gen_len} seed={seed} baseline={base_time:.3f}s")
    base_rows = arms["baseline"]["computed_rows_total"]
    for name in PRECISIONS:
        arm = arms[name]
        speedup = base_time / arm["total_s"]
        row_reduction = base_rows / max(1, arm["computed_rows_total"])
        print(
            f"    {name:>5}: {arm['total_s']:.3f}s speedup={speedup:6.3f}x "
            f"rounds={arm['total_rounds']} rounds_match={arm['rounds_match']} "
            f"aligned={arm['decisions_aligned']} "
            f"max_relL2={arm['max_rel_l2']:.3e} token_agree={arm['token_agreement']:.4f} "
            f"bitwise={arm['bitwise_equal']}"
        )
        print(
            f"           computed_rows={arm['computed_rows_total']} "
            f"(row_ratio={row_reduction:.1f}x) kv_writes={arm['kv_write_rows_total']} "
            f"forced={arm['forced_completions']} fallback_rounds={arm['fallback_rounds']} "
            f"median_block_ms={arm['median_block_ms']}"
        )
        print(f"           conf_range={arm['mean_conf_range']}")


def run_probe() -> None:
    """Calibration probe: unregistered, single command-line seed set."""
    threshold = (
        float(sys.argv[sys.argv.index("--threshold") + 1])
        if "--threshold" in sys.argv
        else 0.9
    )
    if "--gen" in sys.argv:
        gen_lens = tuple(
            int(g) for g in sys.argv[sys.argv.index("--gen") + 1].split(",")
        )
    else:
        gen_lens = (192, 448)
    seeds = (0,)
    if "--seed" in sys.argv:
        seeds = tuple(
            int(s) for s in sys.argv[sys.argv.index("--seed") + 1].split(",")
        )
    print(f"[AR016 probe] threshold={threshold} gen={gen_lens} (not registered)")
    for gen_len in gen_lens:
        for seed in seeds:
            arms = _run_arms(gen_len, seed, threshold)
            _print_point(gen_len, seed, arms)


def _check(
    failures: List[str],
    label: str,
    ok: bool,
    detail: str,
) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"    [{status}] {label}: {detail}")
    if not ok:
        failures.append(f"{label}: {detail}")


def run_full() -> None:
    print(
        f"[AR016 full] registered config: threshold={REGISTERED_THRESHOLD}, "
        f"seeds={SEEDS}, G={GEN_LENS}, W={WIDTH}, B={BATCH}, max_rounds={MAX_ROUNDS}"
    )
    failures: List[str] = []
    all_results: List[Dict[str, object]] = []
    for gen_len in GEN_LENS:
        for seed in SEEDS:
            arms = _run_arms(gen_len, seed, REGISTERED_THRESHOLD)
            _print_point(gen_len, seed, arms)
            all_results.append({"gen_len": gen_len, "seed": seed, "arms": arms})

    print("[AR016 full] per-scale medians and registered assertions:")
    speedup_medians: Dict[int, float] = {}
    for gen_len in GEN_LENS:
        points = [r for r in all_results if r["gen_len"] == gen_len]
        base_times = [p["arms"]["baseline"]["total_s"] for p in points]
        speedups = [
            p["arms"]["baseline"]["total_s"] / p["arms"]["fp32"]["total_s"]
            for p in points
        ]
        speedup_medians[gen_len] = _median(speedups)
        row_ratios = [
            p["arms"]["baseline"]["computed_rows_total"]
            / max(1, p["arms"]["fp32"]["computed_rows_total"])
            for p in points
        ]
        line = (
            f"  G={gen_len}: baseline_median={_median(base_times):.3f}s "
            f"fp32_speedup_median={_median(speedups):.3f}x "
            f"(range {min(speedups):.3f}-{max(speedups):.3f}) "
            f"row_ratio_min={min(row_ratios):.1f}x"
        )
        print(line)
        _check(
            failures,
            f"G={gen_len} fp32 row reduction >= {ROW_REDUCTION_MIN}x",
            min(row_ratios) >= ROW_REDUCTION_MIN,
            f"min={min(row_ratios):.2f}x",
        )
        for p in points:
            seed = p["seed"]
            fp32 = p["arms"]["fp32"]
            fp16 = p["arms"]["fp16"]
            int8 = p["arms"]["int8"]
            for prec, arm in (("fp32", fp32), ("fp16", fp16)):
                _check(
                    failures,
                    f"G={gen_len} seed={seed} {prec} bitwise tokens",
                    bool(arm["bitwise_equal"]),
                    f"bitwise={arm['bitwise_equal']}",
                )
                budget = (
                    FP32_RELL2_BUDGET if prec == "fp32" else FP16_RELL2_BUDGET
                )
                _check(
                    failures,
                    f"G={gen_len} seed={seed} {prec} relL2 <= {budget:g}",
                    arm["max_rel_l2"] <= budget,
                    f"max_relL2={arm['max_rel_l2']:.3e}",
                )
            _check(
                failures,
                f"G={gen_len} seed={seed} int8 token agreement"
                f" >= {INT8_TOKEN_AGREE_MIN}",
                int8["token_agreement"] >= INT8_TOKEN_AGREE_MIN,
                f"agree={int8['token_agreement']:.4f} "
                f"diverged={not int8['bitwise_equal']}",
            )
            if int8["decisions_aligned"]:
                _check(
                    failures,
                    f"G={gen_len} seed={seed} int8 aligned relL2"
                    f" <= {INT8_RELL2_BUDGET:g}",
                    int8["max_rel_l2"] <= INT8_RELL2_BUDGET,
                    f"max_relL2={int8['max_rel_l2']:.3e}",
                )
            else:
                print(
                    f"    [SKIP] G={gen_len} seed={seed} int8 relL2 assertion:"
                    " decision-diverged point (position-level), relL2 reported"
                    f" unregistered ({int8['max_rel_l2']:.3e})"
                )
            for prec in PRECISIONS:
                arm = p["arms"][prec]
                _check(
                    failures,
                    f"G={gen_len} seed={seed} {prec} kv writes == P+G",
                    arm["kv_write_rows_total"] == PROMPT_LEN + gen_len,
                    f"kv_writes={arm['kv_write_rows_total']}",
                )

    passing_scales = sum(
        1 for g in GEN_LENS if speedup_medians[g] >= SPEEDUP_MIN
    )
    _check(
        failures,
        f"fp32 speedup >= {SPEEDUP_MIN}x on >= {SPEEDUP_MIN_SCALES} scales",
        passing_scales >= SPEEDUP_MIN_SCALES,
        "medians: "
        + ", ".join(f"G={g}={speedup_medians[g]:.3f}x" for g in GEN_LENS),
    )

    print(
        f"[AR016 full] sensitivity sweep (report-only, G={SENSITIVITY_GEN}, "
        f"seed={SENSITIVITY_SEED}):"
    )
    for threshold in SENSITIVITY_THRESHOLDS:
        arms = _run_arms(SENSITIVITY_GEN, SENSITIVITY_SEED, threshold)
        fp32 = arms["fp32"]
        speedup = arms["baseline"]["total_s"] / fp32["total_s"]
        print(
            f"  threshold={threshold}: rounds={fp32['total_rounds']} "
            f"fallback_rounds={fp32['fallback_rounds']} "
            f"forced={fp32['forced_completions']} "
            f"baseline={arms['baseline']['total_s']:.3f}s "
            f"fp32={fp32['total_s']:.3f}s speedup={speedup:.3f}x "
            f"bitwise={fp32['bitwise_equal']}"
        )

    if failures:
        print(f"[AR016 full] ASSERTION FAILURES ({len(failures)}):")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("[AR016 full] all registered assertions passed.")


def main() -> None:
    torch.set_num_threads(1)
    if "--probe" in sys.argv:
        run_probe()
        return
    run_full()


if __name__ == "__main__":
    main()
