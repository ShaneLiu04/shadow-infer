"""Real PyTorchModelBackend integration with Orchestrator and CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from shadowinfer.core.backends.pytorch_backend import PyTorchModelBackend
from shadowinfer.orchestrator import Orchestrator


@pytest.fixture
def tiny_backend():
    """A tiny deterministic PyTorchModelBackend for integration tests."""
    cfg = {
        "name": "tiny-test",
        "num_layers": 2,
        "num_heads": 2,
        "head_dim": 16,
        "hidden_dim": 32,
        "vocab_size": 64,
        "max_seq_len": 32,
        "intermediate_dim": 64,
        "batch_size": 1,
        "seq_len": 4,
    }
    backend = PyTorchModelBackend(model_config=cfg, seed=42)
    backend.load("tiny-test", device="cpu")
    return backend


class TestOrchestratorRealBackend:
    def test_run_baseline_with_pytorch_backend(self, tiny_backend):
        """Orchestrator can run baseline using PyTorchModelBackend."""
        orch = Orchestrator(config="configs/optimize_full.yaml", model_backend=tiny_backend)
        orch.initialize(model_config=tiny_backend.get_model_config())
        try:
            result = orch.run_baseline(prompt="test", num_steps=2)
            assert isinstance(result, dict)
            assert "latency" in result
            assert "kv_cache" in result
            assert result.get("num_steps") == 2
        finally:
            orch.shutdown()

    def test_run_optimized_with_pytorch_backend(self, tiny_backend):
        """Orchestrator can run optimized pipeline using PyTorchModelBackend."""
        orch = Orchestrator(config="configs/optimize_full.yaml", model_backend=tiny_backend)
        orch.initialize(model_config=tiny_backend.get_model_config())
        try:
            result = orch.run_optimized(prompt="test", num_steps=2)
            assert isinstance(result, dict)
            assert "latency" in result
            assert "kv_cache" in result
        finally:
            orch.shutdown()


class TestCLIRealBackend:
    def test_profiler_backend_argument_parsing(self):
        """``--backend pytorch`` is accepted by the CLI parser."""
        from shadowinfer.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            [
                "profiler",
                "--model",
                "tiny-test",
                "--backend",
                "pytorch",
                "--device",
                "cpu",
                "--seed",
                "123",
                "--num-steps",
                "1",
            ]
        )
        assert args.backend == "pytorch"
        assert args.device == "cpu"
        assert args.seed == 123
        assert args.num_steps == 1

    def test_optimize_backend_argument_parsing(self):
        """``--backend mock`` is accepted by the optimize subcommand."""
        from shadowinfer.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            [
                "optimize",
                "--model",
                "tiny-test",
                "--backend",
                "mock",
                "--device",
                "cpu",
                "--num-steps",
                "1",
            ]
        )
        assert args.backend == "mock"
        assert args.device == "cpu"

    def test_serve_backend_argument_parsing(self):
        """``serve`` subcommand accepts backend/device/seed/port."""
        from shadowinfer.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            [
                "serve",
                "--model",
                "tiny-test",
                "--backend",
                "pytorch",
                "--device",
                "cpu",
                "--seed",
                "42",
                "--port",
                "8123",
            ]
        )
        assert args.command == "serve"
        assert args.backend == "pytorch"
        assert args.port == 8123

    @pytest.mark.slow
    def test_profiler_cli_runs_with_pytorch_backend(self, tmp_path):
        """End-to-end: the profiler CLI can run with --backend pytorch."""
        output_dir = tmp_path / "profiling"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "shadowinfer",
                "profiler",
                "--model",
                "tiny-test",
                "--config",
                "configs/optimize_full.yaml",
                "--backend",
                "pytorch",
                "--device",
                "cpu",
                "--seed",
                "42",
                "--num-steps",
                "1",
                "--output",
                str(output_dir),
            ],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "Profiling completed" in result.stdout

    def test_serve_cli_one_shot_runs(self):
        """End-to-end: the serve CLI one-shot mode loads the backend and generates."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "shadowinfer",
                "serve",
                "--model",
                "tiny-test",
                "--backend",
                "pytorch",
                "--device",
                "cpu",
                "--seed",
                "42",
                "--one-shot",
                "--num-steps",
                "2",
            ],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "[serve]" in result.stdout
        assert "ShadowInfer serving mode" in result.stdout
        assert "One-shot generation completed" in result.stdout
