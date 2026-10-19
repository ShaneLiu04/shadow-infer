"""UT for the AR007 reproducible experiment harness (real backend only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_experiment as harness  # noqa: E402

MODEL_CONFIG = {
    "name": "ar007-ut",
    "num_layers": 2,
    "num_heads": 4,
    "head_dim": 16,
    "hidden_dim": 64,
    "intermediate_dim": 128,
    "vocab_size": 128,
    "max_seq_len": 64,
    "batch_size": 1,
    "seq_len": 32,
}
CONFIG = {
    "log_level": "ERROR",
    "warmup_steps": 1,
    "latency_budget_ms": 500.0,
    "memory_budget_mb": 2048.0,
}


def _run(tmp_path: Path, **overrides) -> dict:
    kwargs = {
        "backend": "pytorch",
        "model_name": "ar007-ut",
        "steps": 2,
        "repeats": 1,
        "seed": 7,
        "device": "cpu",
        "out_dir": tmp_path / "exp",
    }
    kwargs.update(overrides)
    return harness.run_experiment(CONFIG, MODEL_CONFIG, **kwargs)


def test_real_pytorch_backend_and_artifacts(tmp_path: Path) -> None:
    result = _run(tmp_path)

    assert result["status"] == "ok"
    assert result["plan"]["backend"] == "pytorch"
    assert result["environment"]["torch"]
    assert result["environment"]["python"]
    assert result["loaded_model_config"]["num_layers"] == MODEL_CONFIG["num_layers"]
    assert len(result["plan"]["config_digest"]) == 64

    out = tmp_path / "exp"
    assert (out / "result.json").exists()
    assert (out / "summary.md").exists()
    assert (out / "run_0" / "profile_baseline.json").exists()
    assert (out / "run_0" / "profile_optimized.json").exists()

    summary = result["summary"]
    assert summary["baseline_e2e_ms"]["median"] > 0.0
    assert summary["optimized_e2e_ms"]["median"] > 0.0
    assert summary["optimized_steps"] == 2
    assert summary["kv_memory_mb"]["baseline"] > 0.0
    assert summary["kv_compression_ratio"] > 1.0

    run = result["runs"][0]
    assert run["baseline"]["num_steps"] == 2
    assert len(run["baseline"]["latency"]["per_step_ms"]) == 2
    assert run["optimized"]["requested_steps"] == 2
    assert run["optimized"]["actual_steps"] == 2


def test_repeats_report_median_and_structural_reproducibility(tmp_path: Path) -> None:
    result = _run(tmp_path, repeats=2)
    summary = result["summary"]

    assert summary["baseline_e2e_ms"]["min"] <= summary["baseline_e2e_ms"]["max"]
    assert summary["speedup"]["min"] <= summary["speedup"]["max"]
    for key in ("baseline_e2e_ms", "optimized_e2e_ms", "speedup"):
        assert summary[key]["min"] <= summary[key]["median"] <= summary[key]["max"]

    baseline_kv = [run["baseline"]["kv_cache"]["memory_mb"] for run in result["runs"]]
    assert baseline_kv[0] == baseline_kv[1]
    assert all(run["optimized"]["actual_steps"] == 2 for run in result["runs"])


def test_dry_run_does_not_execute(tmp_path: Path) -> None:
    result = _run(tmp_path, dry_run=True)
    assert result["status"] == "dry_run"
    assert result["plan"]["steps"] == 2
    assert not (tmp_path / "exp").exists()


def test_mock_backend_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="real backend"):
        _run(tmp_path, backend="mock")


def test_config_digest_is_stable_and_sensitive() -> None:
    digest = harness._config_digest(CONFIG, MODEL_CONFIG)
    assert digest == harness._config_digest(dict(CONFIG), dict(MODEL_CONFIG))
    changed = {**CONFIG, "latency_budget_ms": 42.0}
    assert harness._config_digest(changed, MODEL_CONFIG) != digest


def test_flatten_template_model_config() -> None:
    raw = {
        "model": {"name": "tpl"},
        "architecture": {"num_layers": 8, "hidden_dim": 256},
        "diffusion": {"num_steps": 10},
    }
    flat = harness._flatten_model_config(raw)
    assert flat["name"] == "tpl"
    assert flat["num_layers"] == 8
    assert flat["hidden_dim"] == 256
    assert "architecture" not in flat


def test_cli_writes_summary(tmp_path: Path) -> None:
    code = harness.main(
        [
            "--profile",
            "s1",
            "--steps",
            "2",
            "--device",
            "cpu",
            "--out",
            str(tmp_path / "cli"),
        ]
    )
    assert code == 0
    result = json.loads((tmp_path / "cli" / "result.json").read_text(encoding="utf-8"))
    assert result["plan"]["backend"] == "pytorch"
    assert result["summary"]["optimized_steps"] == 2
    summary_md = (tmp_path / "cli" / "summary.md").read_text(encoding="utf-8")
    assert "ShadowInfer experiment summary" in summary_md
