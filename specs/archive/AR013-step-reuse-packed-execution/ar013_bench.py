"""AR013 benchmark: cross-step reuse and packed KV execution.

Real tensors, real SimpleDiffusionLLM forwards, controlled denoising
trajectories. No mocks.

Run: python specs/changes/AR013-step-reuse-packed-execution/ar013_bench.py
"""

from __future__ import annotations

import statistics
import time
from typing import Dict, List, Tuple

import torch

from shadowinfer.core.backends.pytorch_backend import PyTorchModelBackend
from shadowinfer.core.structs import StepConfig

CONFIGS = (
    (8, 8, 64, 512, 256),
    (8, 8, 64, 512, 512),
    (8, 16, 64, 1024, 256),
)
SIGMAS = torch.linspace(0.9, 0.02, 10)


def _model_config(layers: int, heads: int, head_dim: int, hidden: int, seq: int) -> Dict:
    return {
        "name": "ar013-bench",
        "num_layers": layers,
        "num_heads": heads,
        "head_dim": head_dim,
        "hidden_dim": hidden,
        "vocab_size": 1000,
        "max_seq_len": seq,
        "intermediate_dim": 4 * hidden,
    }


def _make_backend(
    cfg: Dict,
    seed: int,
    *,
    execution: str,
    reuse: bool = False,
    threshold: float = 0.2,
) -> PyTorchModelBackend:
    backend = PyTorchModelBackend(model_config=_model_config(*cfg), seed=seed)
    backend.load(model_name="ar013-bench", device="cpu")
    backend.configure_execution(
        kv_lifecycle="replace",
        kv_execution=execution,
        packed_precision="int8" if execution == "packed" else None,
        step_reuse_enabled=reuse,
        step_reuse_threshold=threshold,
        step_reuse_warmup_steps=1,
    )
    return backend


def _trajectory(cfg: Dict, seed: int) -> List[torch.Tensor]:
    layers, heads, head_dim, hidden, seq = cfg
    gen = torch.Generator().manual_seed(seed)
    x0 = torch.randn(1, seq, hidden, generator=gen)
    noise = torch.randn(1, seq, hidden, generator=gen)
    return [x0 + float(sigma) * noise for sigma in SIGMAS]


def _run(backend: PyTorchModelBackend, xs: List[torch.Tensor]) -> Tuple[List, float]:
    outputs = []
    start = time.perf_counter()
    for step_id, x in enumerate(xs):
        cfg = StepConfig(
            step_id=step_id,
            total_steps=len(xs),
            noise_level=float(SIGMAS[step_id]),
            shadowkv_mode="aggressive" if backend._kv_execution == "packed" else "balanced",
            ffn_mode="full",
        )
        result = backend.forward_step(x, cfg)
        outputs.append((result["output"].detach(), result["execution_stats"]))
    return outputs, time.perf_counter() - start


def _timed(cfg: Dict, seed: int, repeats: int, **kwargs) -> Tuple[List, List[float]]:
    xs = _trajectory(cfg, seed)
    times = []
    last_outputs = None
    for _ in range(repeats):
        backend = _make_backend(cfg, seed, **kwargs)
        outputs, elapsed = _run(backend, xs)
        times.append(elapsed)
        last_outputs = outputs
    return last_outputs, times


def _rel_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return float((actual - reference).norm() / reference.norm().clamp_min(1e-12))


def _quality(outputs, reference) -> float:
    errors = [_rel_l2(o, r) for (o, _), (r, _) in zip(outputs, reference)]
    return statistics.median(errors)


def _memory(outputs) -> Tuple[int, int, int]:
    _, stats = outputs[-1]
    return stats["resident_bytes"], stats["data_plane_bytes"], stats["dense_bytes"]


def run_config(cfg: Dict, seed: int = 0, repeats: int = 3) -> Dict:
    base_outputs, base_times = _timed(cfg, seed, repeats, execution="dense")
    reuse_outputs, reuse_times = _timed(
        cfg, seed, repeats, execution="dense", reuse=True, threshold=0.1
    )
    packed_outputs, packed_times = _timed(cfg, seed, repeats, execution="packed")
    combined_outputs, combined_times = _timed(
        cfg, seed, repeats, execution="packed", reuse=True, threshold=0.1
    )

    base = statistics.median(base_times)
    result = {
        "config": cfg,
        "base": base,
        "reuse": statistics.median(reuse_times),
        "packed": statistics.median(packed_times),
        "combined": statistics.median(combined_times),
        "reuse_err": _quality(reuse_outputs, base_outputs),
        "packed_err": _quality(packed_outputs, base_outputs),
        "combined_err": _quality(combined_outputs, base_outputs),
        "packed_mem": _memory(packed_outputs),
        "combined_mem": _memory(combined_outputs),
        "reuse_ratio": statistics.median([stats["reused_ratio"] for _, stats in reuse_outputs]),
        "combined_ratio": statistics.median(
            [stats["reused_ratio"] for _, stats in combined_outputs]
        ),
        "reuse_skips": sum(len(s["reused_layers"]) for _, s in reuse_outputs),
        "combined_skips": sum(len(s["reused_layers"]) for _, s in combined_outputs),
    }
    return result


def main() -> None:
    print("AR013 benchmark (CPU, real SimpleDiffusionLLM, median of 3)")
    failures = []
    for cfg in CONFIGS:
        for seed in (0, 1, 2):
            r = run_config(cfg, seed=seed)
            resident, data_plane, dense = r["packed_mem"]
            # dense_bytes is the FP16 dense baseline (SRS D4). resident is the
            # full accounting (data + per-entry Python metadata); data_plane is
            # tensor storage only, the deployment-transferable portion.
            mem_ratio = resident / max(dense, 1)
            data_plane_ratio = data_plane / max(dense, 1)
            c_resident, c_data_plane, c_dense = r["combined_mem"]
            combined_mem_ratio = c_resident / max(c_dense, 1)
            combined_data_plane_ratio = c_data_plane / max(c_dense, 1)
            print(
                f"L{cfg[0]} H{cfg[1]} D{cfg[2]} HN{cfg[3]} S{cfg[4]} seed{seed} | "
                f"reuse {r['base'] / r['reuse']:.2f}x err {r['reuse_err'] * 100:.2f}% "
                f"skip {r['reuse_ratio'] * 100:.0f}% | "
                f"packed mem {mem_ratio * 100:.0f}% "
                f"(data-plane {data_plane_ratio * 100:.0f}%) "
                f"err {r['packed_err'] * 100:.2f}% | "
                f"combined {r['base'] / r['combined']:.2f}x "
                f"err {r['combined_err'] * 100:.2f}% mem {combined_mem_ratio * 100:.0f}% "
                f"(data-plane {combined_data_plane_ratio * 100:.0f}%)"
            )
            # Quality / memory assertions hold for every seed; speed assertions
            # use seed 0 (median of 3 repeats) to stay robust to CPU jitter.
            if r["reuse_err"] > 0.02:
                failures.append(f"reuse error {cfg} seed{seed}: {r['reuse_err']:.4f}")
            if r["packed_err"] > 0.02:
                failures.append(f"packed error {cfg} seed{seed}: {r['packed_err']:.4f}")
            if r["combined_err"] > 0.02:
                failures.append(f"combined error {cfg} seed{seed}: {r['combined_err']:.4f}")
            if data_plane_ratio > 0.6:
                failures.append(
                    f"packed data-plane memory {cfg} seed{seed}: {data_plane_ratio:.3f}"
                )
            if seed != 0:
                continue
            if r["base"] / r["reuse"] < 1.15:
                failures.append(f"reuse speed {cfg}")
            if r["base"] / r["combined"] < 1.10:
                failures.append(f"combined speed {cfg}")

    if failures:
        raise SystemExit("AR013 benchmark failures: " + "; ".join(failures))
    print("all AR013 benchmark assertions passed")


if __name__ == "__main__":
    main()
