"""AR017 benchmark: blockwise MTP draft-verify decoding (FR17.4).

Four arms, real SimpleDiffusionLLM, real random prompts, single thread:

- baseline: ``blockwise_generate_baseline`` with mask-init scheduling (full
  recompute per round; AR016 reference arm);
- baseline-dv: ``blockwise_generate_baseline`` with draft_mode=True (full
  recompute; decision-equivalent reference arm for the draft-verify paged
  arm, FR17.2);
- mask-init: ``PyTorchModelBackend.generate`` with fp32 pages (AR016
  pipeline);
- draft-verify: ``PyTorchModelBackend.generate`` with fp32 pages and
  draft_mode=True (this AR).

Calibration (probe, pre-registration; bench-results.md section 1): the v1
preregistered gates (>= 1.5x end-to-end, >= 2x round reduction, >= 90%
token agreement vs mask-init) are mechanistically unreachable on the
random-weight toy model: probe1 shows both arms burn 8 rounds + 1 sealing
forward per block (parity, 9 forwards/block), and probe2 shows the
pure-Jacobi changed fraction plateaus at 62-97% without decay for all
widths W in {8, 16, 32} -- no stable argmax fixpoint exists under random
weights, so MTP round-reduction gains are contingent on real pretrained
draft quality (Fast-dLLM / Jacobi literature evidence all comes from
pretrained models). With user confirmation the registration was revised
(v2, honest): exactness anchors (two-arm bitwise + schedule-table
equality), a zero-regression bound (<= 1.2x), bad-draft rejection
(acceptance <= 0.25), and closed-form accounting. Performance claims
(1.5x / 2x / 90%) move to the s12 GPU protocol as server-venue assertions
(requires AR018 real dLLM weights). Revision history: bench-results.md
section 1.3; design.md section 6.5 erratum.

Registered configuration: threshold=0.9 (multi-round regime per srs),
(L,H,D)=(8,8,64), P=64, G in {192, 448, 960}, B=2, W=32, max_rounds=8,
page_size=32, seeds {0, 1, 2}, torch single thread, CPU.

Registered assertions (full run exits non-zero on violation):
- A1: draft-verify paged tokens bitwise equal baseline-dv at all 9 points
  (UT T004 anchor: 2/2 scales);
- A2: draft-verify paged schedule stats equal baseline-dv (rounds_per_block,
  total_rounds, rounds_to_fixpoint, changed_per_round, forced_completions,
  draft_acceptance_rate) at all 9 points (UT T004 anchor: 2/2 scales);
- A3: zero-regression bound: draft-verify paged end-to-end / mask-init
  paged end-to-end <= 1.2 per point (probe: structural parity, both arms
  9 forwards/block at threshold=0.9);
- A4: bad-draft rejection: draft_acceptance_rate <= 0.25 at all 9 points
  (probe1: 0.038-0.086; the self-consistency verifier rejects low-quality
  random-weight drafts instead of locking them in);
- A5: closed-form accounting per point: computed_rows_total ==
  sum(W * (rounds + seal)) with seal = 0 iff rounds_to_fixpoint != -1;
  forced_completions == #{rounds_to_fixpoint == -1}; kv_write_rows_total
  == P + G (deterministic structure).

Report-only (not registered; v1 gates withdrawn, actuals recorded):
draft-verify vs mask-init token agreement (random-level on the toy),
end-to-end ratio vs mask-init (parity expected), oscillation
characterization (A6: changed fraction plateau per seed), threshold
sensitivity sweep at 0.005 (G=192, seed 0).

Run (full, registered): $env:PYTHONPATH='D:\ShadowInfer';
  python specs/changes/AR017-mtp-draft-verify-decoding/ar017_bench.py
Run (probe, unregistered): same command plus `` --probe [--threshold T]
  [--gen 192,448] [--seed 0] [--max-rounds R]``
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
REGISTERED_THRESHOLD = 0.9
SENSITIVITY_THRESHOLD = 0.005
SENSITIVITY_GEN = 192
SENSITIVITY_SEED = 0

NO_REGRESSION_MAX = 1.2
ACCEPTANCE_MAX = 0.25

SCHEDULE_KEYS = (
    "rounds_per_block",
    "total_rounds",
    "rounds_to_fixpoint",
    "changed_per_round",
    "forced_completions",
)


def _median(values: List[float]) -> float:
    return sorted(values)[len(values) // 2]


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


def _config(
    draft: bool,
    threshold: float,
    max_rounds: int = MAX_ROUNDS,
) -> BlockDecodingConfig:
    return BlockDecodingConfig(
        block_size=WIDTH,
        max_rounds=max_rounds,
        confidence_threshold=threshold,
        mask_token_id=0,
        page_size=PAGE_SIZE,
        page_precision="fp32",
        draft_mode=draft,
    )


def _run_arms(
    gen_len: int,
    seed: int,
    threshold: float,
    max_rounds: int = MAX_ROUNDS,
) -> Dict[str, Dict[str, object]]:
    """One (scale, seed) point: two recompute arms + two paged arms."""
    prompt = _prompt(seed)
    model = _direct_model(seed)
    backend = PyTorchModelBackend(model_config=_model_config(), seed=seed)
    backend.load("ar017-bench", device="cpu")

    cfg_mask = _config(False, threshold, max_rounds)
    cfg_draft = _config(True, threshold, max_rounds)

    t0 = time.perf_counter()
    base = blockwise_generate_baseline(model, prompt, gen_len, cfg_mask)
    base_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    base_dv = blockwise_generate_baseline(model, prompt, gen_len, cfg_draft)
    base_dv_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    mask_res = backend.generate(prompt, gen_len, cfg_mask)
    mask_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    dv_res = backend.generate(prompt, gen_len, cfg_draft)
    dv_time = time.perf_counter() - t0

    arms: Dict[str, Dict[str, object]] = {
        "baseline": {"total_s": base_time, "result": base},
        "baseline-dv": {"total_s": base_dv_time, "result": base_dv},
        "mask-init": {"total_s": mask_time, "result": mask_res},
        "draft-verify": {"total_s": dv_time, "result": dv_res},
    }

    mask_stats = mask_res.stats
    mask_arm = arms["mask-init"]
    mask_arm["rounds_per_block"] = mask_stats["rounds_per_block"]
    mask_arm["forced"] = mask_stats["forced_completions"]
    mask_arm["computed_rows"] = mask_stats["computed_rows_total"]
    mask_arm["kv_writes"] = mask_stats["kv_write_rows_total"]
    mask_arm["bitwise_vs_baseline"] = bool(
        mask_res.sequences.shape == base.sequences.shape
        and torch.equal(mask_res.sequences, base.sequences)
    )

    base_dv_arm = arms["baseline-dv"]
    base_dv_arm["rounds_per_block"] = base_dv.stats["rounds_per_block"]
    base_dv_arm["forced"] = base_dv.stats["forced_completions"]

    dv_arm = arms["draft-verify"]
    dvs = dv_res.stats
    dv_arm["bitwise_vs_baseline_dv"] = bool(
        dv_res.sequences.shape == base_dv.sequences.shape
        and torch.equal(dv_res.sequences, base_dv.sequences)
    )
    mismatched = [
        key for key in SCHEDULE_KEYS if dvs.get(key) != base_dv.stats.get(key)
    ]
    acc_diff = abs(
        float(dvs["draft_acceptance_rate"])
        - float(base_dv.stats["draft_acceptance_rate"])
    )
    dv_arm["schedule_match"] = not mismatched and acc_diff < 1e-12
    dv_arm["schedule_mismatch_keys"] = mismatched

    rounds_pb = list(dvs["rounds_per_block"])
    rtf = list(dvs["rounds_to_fixpoint"])
    changed = list(dvs["changed_per_round"])
    dv_arm["rounds_per_block"] = rounds_pb
    dv_arm["rtf"] = rtf
    dv_arm["forced"] = dvs["forced_completions"]
    dv_arm["blocks"] = dvs["blocks"]
    dv_arm["acceptance"] = float(dvs["draft_acceptance_rate"])
    dv_arm["computed_rows"] = dvs["computed_rows_total"]
    dv_arm["kv_writes"] = dvs["kv_write_rows_total"]

    expected_rows = sum(
        WIDTH * (rounds + (0 if rt != -1 else 1))
        for rounds, rt in zip(rounds_pb, rtf)
    )
    dv_arm["rows_closed_form_ok"] = dvs["computed_rows_total"] == expected_rows
    dv_arm["forced_match_ok"] = dvs["forced_completions"] == sum(
        1 for rt in rtf if rt == -1
    )

    verify_fracs = [
        c / float(BATCH * WIDTH) for block in changed for c in block[1:]
    ]
    final_fracs = [block[-1] / float(BATCH * WIDTH) for block in changed]
    dv_arm["plateau"] = (
        round(min(verify_fracs), 3),
        round(max(verify_fracs), 3),
    )
    dv_arm["final_changed_frac"] = (
        round(min(final_fracs), 3),
        round(max(final_fracs), 3),
    )
    dv_arm["agree_vs_maskinit"] = float(
        (dv_res.sequences == mask_res.sequences).float().mean()
    )
    dv_arm["agree_vs_baseline"] = float(
        (dv_res.sequences == base.sequences).float().mean()
    )
    return arms


def _print_point(gen_len: int, seed: int, arms: Dict[str, Dict[str, object]]) -> None:
    base_time = arms["baseline"]["total_s"]
    base_dv_time = arms["baseline-dv"]["total_s"]
    print(
        f"  G={gen_len} seed={seed} baseline={base_time:.3f}s "
        f"baseline-dv={base_dv_time:.3f}s"
    )
    mask_arm = arms["mask-init"]
    print(
        f"    mask-init  : {mask_arm['total_s']:.3f}s "
        f"rounds={mask_arm['rounds_per_block']} forced={mask_arm['forced']} "
        f"rows={mask_arm['computed_rows']} kv={mask_arm['kv_writes']} "
        f"bitwise_vs_base={mask_arm['bitwise_vs_baseline']}"
    )
    dv_arm = arms["draft-verify"]
    mask_time = mask_arm["total_s"]
    dv_time = dv_arm["total_s"]
    ratio = float(dv_time) / float(mask_time)
    print(
        f"    draft-verif: {dv_time:.3f}s ratio_vs_mask={ratio:.3f}x "
        f"rounds={dv_arm['rounds_per_block']} forced={dv_arm['forced']} "
        f"accept={dv_arm['acceptance']:.4f}"
    )
    print(
        f"                bitwise_vs_base-dv={dv_arm['bitwise_vs_baseline_dv']} "
        f"schedule_match={dv_arm['schedule_match']} "
        f"rows={dv_arm['computed_rows']} "
        f"(closed_form={dv_arm['rows_closed_form_ok']} "
        f"forced_match={dv_arm['forced_match_ok']}) kv={dv_arm['kv_writes']}"
    )
    print(
        f"                plateau={dv_arm['plateau']} "
        f"final_changed={dv_arm['final_changed_frac']} "
        f"agree_vs_mask={dv_arm['agree_vs_maskinit']:.4f} "
        f"agree_vs_base={dv_arm['agree_vs_baseline']:.4f}"
    )


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


def run_probe() -> None:
    """Calibration probe: unregistered, single command-line grid."""
    threshold = (
        float(sys.argv[sys.argv.index("--threshold") + 1])
        if "--threshold" in sys.argv
        else 0.9
    )
    max_rounds = (
        int(sys.argv[sys.argv.index("--max-rounds") + 1])
        if "--max-rounds" in sys.argv
        else MAX_ROUNDS
    )
    if "--gen" in sys.argv:
        gen_lens = tuple(
            int(g) for g in sys.argv[sys.argv.index("--gen") + 1].split(",")
        )
    else:
        gen_lens = (192,)
    if "--seed" in sys.argv:
        seeds = tuple(
            int(s) for s in sys.argv[sys.argv.index("--seed") + 1].split(",")
        )
    else:
        seeds = (0,)
    print(
        f"[AR017 probe] threshold={threshold} max_rounds={max_rounds} "
        f"gen={gen_lens} seeds={seeds} (not registered)"
    )
    for gen_len in gen_lens:
        for seed in seeds:
            arms = _run_arms(gen_len, seed, threshold, max_rounds)
            _print_point(gen_len, seed, arms)


def run_full() -> None:
    print(
        f"[AR017 full] registered config: threshold={REGISTERED_THRESHOLD}, "
        f"seeds={SEEDS}, G={GEN_LENS}, W={WIDTH}, B={BATCH}, "
        f"max_rounds={MAX_ROUNDS} (assertions v2, honest; see "
        f"bench-results.md section 1.3)"
    )
    failures: List[str] = []
    all_results: List[Dict[str, object]] = []
    for gen_len in GEN_LENS:
        for seed in SEEDS:
            arms = _run_arms(gen_len, seed, REGISTERED_THRESHOLD)
            _print_point(gen_len, seed, arms)
            all_results.append({"gen_len": gen_len, "seed": seed, "arms": arms})
            dv_arm = arms["draft-verify"]
            mask_arm = arms["mask-init"]
            ratio = float(dv_arm["total_s"]) / float(mask_arm["total_s"])
            _check(
                failures,
                f"G={gen_len} seed={seed} A1 draft-verify bitwise vs baseline-dv",
                bool(dv_arm["bitwise_vs_baseline_dv"]),
                f"bitwise={dv_arm['bitwise_vs_baseline_dv']}",
            )
            _check(
                failures,
                f"G={gen_len} seed={seed} A2 schedule table equality",
                bool(dv_arm["schedule_match"]),
                f"mismatched_keys={dv_arm['schedule_mismatch_keys']}",
            )
            _check(
                failures,
                f"G={gen_len} seed={seed} A3 zero-regression"
                f" ratio <= {NO_REGRESSION_MAX}",
                ratio <= NO_REGRESSION_MAX,
                f"ratio={ratio:.3f}x",
            )
            _check(
                failures,
                f"G={gen_len} seed={seed} A4 acceptance"
                f" <= {ACCEPTANCE_MAX}",
                float(dv_arm["acceptance"]) <= ACCEPTANCE_MAX,
                f"accept={dv_arm['acceptance']:.4f}",
            )
            _check(
                failures,
                f"G={gen_len} seed={seed} A5 closed-form accounting",
                bool(dv_arm["rows_closed_form_ok"])
                and bool(dv_arm["forced_match_ok"])
                and dv_arm["kv_writes"] == PROMPT_LEN + gen_len,
                f"rows_ok={dv_arm['rows_closed_form_ok']} "
                f"forced_ok={dv_arm['forced_match_ok']} "
                f"kv={dv_arm['kv_writes']} (expect {PROMPT_LEN + gen_len})",
            )

    print("[AR017 full] per-scale medians (report-only):")
    for gen_len in GEN_LENS:
        points = [r for r in all_results if r["gen_len"] == gen_len]
        base_times = [
            float(p["arms"]["baseline"]["total_s"]) for p in points
        ]
        base_dv_times = [
            float(p["arms"]["baseline-dv"]["total_s"]) for p in points
        ]
        mask_times = [
            float(p["arms"]["mask-init"]["total_s"]) for p in points
        ]
        dv_times = [
            float(p["arms"]["draft-verify"]["total_s"]) for p in points
        ]
        ratios = [d / m for d, m in zip(dv_times, mask_times)]
        accepts = [
            float(p["arms"]["draft-verify"]["acceptance"]) for p in points
        ]
        print(
            f"  G={gen_len}: baseline_median={_median(base_times):.3f}s "
            f"baseline_dv_median={_median(base_dv_times):.3f}s "
            f"mask_median={_median(mask_times):.3f}s "
            f"dv_median={_median(dv_times):.3f}s "
            f"dv/mask_ratio_median={_median(ratios):.3f}x "
            f"(range {min(ratios):.3f}-{max(ratios):.3f}) "
            f"accept_range=[{min(accepts):.4f}, {max(accepts):.4f}]"
        )

    print(
        f"[AR017 full] sensitivity sweep (report-only, G={SENSITIVITY_GEN}, "
        f"seed={SENSITIVITY_SEED}, threshold={SENSITIVITY_THRESHOLD}):"
    )
    arms = _run_arms(SENSITIVITY_GEN, SENSITIVITY_SEED, SENSITIVITY_THRESHOLD)
    _print_point(SENSITIVITY_GEN, SENSITIVITY_SEED, arms)

    if failures:
        print(f"[AR017 full] ASSERTION FAILURES ({len(failures)}):")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("[AR017 full] all registered assertions (v2) passed.")


def main() -> None:
    torch.set_num_threads(1)
    if "--probe" in sys.argv:
        run_probe()
        return
    run_full()


if __name__ == "__main__":
    main()
