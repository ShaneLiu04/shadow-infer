"""Reproducible experiment harness (AR007).

Runs the real optimization pipeline (baseline + optimized) with a real model
backend, fixed seeds and explicit config capture, then writes machine-readable
results plus a Markdown summary. No mock backends are used.

Usage:
    python scripts/run_experiment.py --profile s1 --steps 6 --repeats 3
    python scripts/run_experiment.py --config configs/experiments/s2_7b_gpu.yaml \
        --backend hf --model-name <hf-id> --steps 32 --repeats 3 --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shadowinfer.core.backends.registry import get_backend_registry  # noqa: E402
from shadowinfer.core.model_backend import ModelBackend  # noqa: E402
from shadowinfer.orchestrator import Orchestrator  # noqa: E402


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def _git_sha() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _config_digest(config: Dict[str, Any], model_config: Dict[str, Any]) -> str:
    payload = json.dumps(
        {"config": config, "model_config": model_config}, sort_keys=True, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_backend(
    backend_name: str,
    model_name: str,
    model_config: Dict[str, Any],
    device: str,
    seed: int,
) -> ModelBackend:
    """Create and load a real backend (``pytorch`` or ``hf``)."""
    registry = get_backend_registry()
    if backend_name == "hf":
        backend_name = "huggingface"
    if backend_name in ("mock", "auto"):
        raise ValueError(
            "Reproducible experiments require a real backend "
            "('pytorch' or 'huggingface'); "
            f"got {backend_name!r}."
        )
    torch.manual_seed(seed)
    backend = registry.create(
        backend_name,
        model_config=model_config,
        seed=seed,
    )
    backend.load(model_name, device=device)
    return backend


def _flatten_model_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Accept either flat model configs or the repository template layout."""
    flat = dict(raw)
    architecture = flat.pop("architecture", None)
    if isinstance(architecture, dict):
        flat = {**architecture, **flat}
    model_meta = flat.pop("model", None)
    if isinstance(model_meta, dict) and "name" not in flat:
        flat["name"] = model_meta.get("name", "unknown")
    return flat


def _run_once(
    backend: ModelBackend,
    config: Dict[str, Any],
    prompt: str,
    steps: int,
    seed: int,
    workdir: Path,
) -> Dict[str, Any]:
    """Run baseline then optimized with a fresh orchestrator and fixed seed."""
    torch.manual_seed(seed)
    backend.set_kv_cache({})
    profiler_cfg = dict(config.get("profiler", {}))
    profiler_cfg["output_dir"] = str(workdir)
    run_config = {
        **config,
        "output_dir": str(workdir),
        "log_dir": str(workdir / "logs"),
        "profiler": profiler_cfg,
    }
    orch = Orchestrator(config=run_config)
    orch.set_model_backend(backend)
    orch.initialize(model_config=backend.get_model_config())
    try:
        torch.manual_seed(seed)
        baseline = orch.run_baseline(prompt=prompt, num_steps=steps)
        torch.manual_seed(seed)
        optimized = orch.run_optimized(prompt=prompt, num_steps=steps)
    finally:
        orch.shutdown()
    return {"baseline": baseline, "optimized": optimized}


def _summarize(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    baseline_e2e = [r["baseline"]["latency"]["e2e_ms"] for r in runs]
    optimized_e2e = [r["optimized"]["latency"]["e2e_ms"] for r in runs]
    speedups = [b / o for b, o in zip(baseline_e2e, optimized_e2e) if o > 0]

    def stats(values: List[float]) -> Dict[str, float]:
        return {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }

    last = runs[-1]["optimized"]
    baseline_kv = runs[-1]["baseline"].get("kv_cache", {}).get("memory_mb", {})
    optimized_kv = last.get("kv_cache", {}).get("memory_mb", {})

    def _total(memory: Any) -> float:
        if not isinstance(memory, dict):
            return 0.0
        return float(sum(v for v in memory.values() if isinstance(v, (int, float))))

    base_kv_mb = _total(baseline_kv)
    opt_kv_mb = _total(optimized_kv)
    compression = base_kv_mb / opt_kv_mb if opt_kv_mb > 0 else None
    return {
        "baseline_e2e_ms": stats(baseline_e2e),
        "optimized_e2e_ms": stats(optimized_e2e),
        "speedup": stats(speedups) if speedups else {"median": 0.0, "min": 0.0, "max": 0.0},
        "optimized_steps": last.get("actual_steps", last.get("num_steps")),
        "kv_memory_mb": {"baseline": base_kv_mb, "optimized": opt_kv_mb},
        "kv_compression_ratio": compression,
    }


def run_experiment(
    config: Dict[str, Any],
    model_config: Dict[str, Any],
    *,
    backend: str,
    model_name: str,
    prompt: str = "ShadowInfer reproducible experiment",
    steps: int = 6,
    repeats: int = 1,
    seed: int = 42,
    device: str = "cpu",
    out_dir: Path,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute a reproducible baseline/optimized experiment and persist results."""
    out_dir = Path(out_dir)
    model_config = _flatten_model_config(model_config)
    digest = _config_digest(config, model_config)
    plan = {
        "backend": backend,
        "model_name": model_name,
        "device": device,
        "steps": steps,
        "repeats": repeats,
        "seed": seed,
        "config_digest": digest,
    }
    if dry_run:
        return {"status": "dry_run", "plan": plan}

    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    out_dir.mkdir(parents=True, exist_ok=True)

    backend_obj = _build_backend(backend, model_name, model_config, device, seed)
    loaded_model_config = backend_obj.get_model_config()
    runs = []
    for repeat in range(repeats):
        workdir = out_dir / f"run_{repeat}"
        workdir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        runs.append(_run_once(backend_obj, config, prompt, steps, seed, workdir))
        runs[-1]["wall_seconds"] = time.perf_counter() - started

    result = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": device,
            "git_sha": _git_sha(),
        },
        "plan": plan,
        "config": config,
        "model_config": model_config,
        "loaded_model_config": loaded_model_config,
        "summary": _summarize(runs),
        "runs": runs,
        "command": " ".join([sys.executable] + sys.argv),
    }

    result_path = out_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "summary.md").write_text(_render_summary(result), encoding="utf-8")
    return result


def _render_summary(result: Dict[str, Any]) -> str:
    plan = result["plan"]
    summary = result["summary"]
    env = result["environment"]
    base = summary["baseline_e2e_ms"]
    opt = summary["optimized_e2e_ms"]
    sp = summary["speedup"]
    return f"""# ShadowInfer experiment summary

- Backend: `{plan['backend']}` / model `{plan['model_name']}` on `{plan['device']}`
- Steps: {plan['steps']} x {plan['repeats']} repeat(s), seed {plan['seed']}
- Config digest: `{plan['config_digest']}`
- Git SHA: `{env.get('git_sha')}`, torch `{env['torch']}`, python `{env['python']}`
- Timestamp: {result['timestamp']}

| Metric | Median | Min | Max |
|--------|--------|-----|-----|
| Baseline e2e (ms) | {base['median']:.2f} | {base['min']:.2f} | {base['max']:.2f} |
| Optimized e2e (ms) | {opt['median']:.2f} | {opt['min']:.2f} | {opt['max']:.2f} |
| Speedup | {sp['median']:.3f}x | {sp['min']:.3f}x | {sp['max']:.3f}x |

- Optimized steps executed: {summary['optimized_steps']}
- KV compression ratio: {summary['kv_compression_ratio']}

Machine-readable data: `result.json`. Per-repeat artifacts: `run_*/profile_*.json`.
"""


def _profiles() -> Dict[str, Dict[str, Any]]:
    return {
        "s1": {
            "config": {
                "log_level": "ERROR",
                "warmup_steps": 1,
                "latency_budget_ms": 500.0,
                "memory_budget_mb": 2048.0,
            },
            "model_config": {
                "name": "s1-cpu-smoke",
                "num_layers": 4,
                "num_heads": 4,
                "head_dim": 32,
                "hidden_dim": 128,
                "intermediate_dim": 512,
                "vocab_size": 256,
                "max_seq_len": 128,
                "batch_size": 1,
                "seq_len": 64,
            },
        }
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ShadowInfer reproducible experiment runner")
    parser.add_argument("--config", help="Orchestrator config YAML (optimization knobs)")
    parser.add_argument("--model-config", help="Model config YAML (dims/layers/seq_len)")
    parser.add_argument("--profile", choices=sorted(_profiles()), help="Built-in scale profile")
    parser.add_argument("--backend", default="pytorch", help="Real backend name (pytorch|hf)")
    parser.add_argument("--model-name", default="simple-diffusion-llm", help="Model id/name")
    parser.add_argument("--prompt", default="ShadowInfer reproducible experiment")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="results/latest")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    config = _load_yaml(args.config)
    model_config = _flatten_model_config(_load_yaml(args.model_config))
    if args.profile:
        preset = _profiles()[args.profile]
        config = {**preset["config"], **config}
        model_config = {**preset["model_config"], **model_config}
    if not model_config:
        raise SystemExit("Provide --profile, --model-config, or both.")

    result = run_experiment(
        config,
        model_config,
        backend=args.backend,
        model_name=args.model_name,
        prompt=args.prompt,
        steps=args.steps,
        repeats=args.repeats,
        seed=args.seed,
        device=args.device,
        out_dir=Path(args.out),
        dry_run=args.dry_run,
    )
    if result["status"] == "dry_run":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Results written to {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
