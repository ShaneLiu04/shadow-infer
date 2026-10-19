"""AR010 pipeline check: non-contiguous KV fallback count on a real 4-layer backend.

Runs the real ``PyTorchModelBackend`` through the orchestrator and counts calls
to ``_compress_kv_reference`` (fallback) vs ``_compress_kv_with_plan`` (plan
path). The wrapper only instruments call counts; model math is untouched.

Usage:
    python specs/changes/AR010-dense-precision-pipeline/pipeline-check.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from shadowinfer.core.backends.pytorch_backend import PyTorchModelBackend  # noqa: E402
from shadowinfer.orchestrator import Orchestrator  # noqa: E402
from shadowinfer.shadowkv.shadowkv_agent import ShadowKVAgent  # noqa: E402


def main() -> None:
    calls = {"reference": 0, "plan": 0}
    original_reference = ShadowKVAgent._compress_kv_reference
    original_plan = ShadowKVAgent._compress_kv_with_plan

    def counting_reference(self, *args, **kwargs):
        calls["reference"] += 1
        return original_reference(self, *args, **kwargs)

    def counting_plan(self, *args, **kwargs):
        calls["plan"] += 1
        return original_plan(self, *args, **kwargs)

    ShadowKVAgent._compress_kv_reference = counting_reference
    ShadowKVAgent._compress_kv_with_plan = counting_plan

    model_config = {
        "name": "ar010-pipeline",
        "num_layers": 4,
        "num_heads": 4,
        "head_dim": 32,
        "hidden_dim": 128,
        "intermediate_dim": 512,
        "vocab_size": 256,
        "max_seq_len": 128,
        "batch_size": 1,
        "seq_len": 64,
    }
    config = {
        "log_level": "ERROR",
        "warmup_steps": 1,
        "latency_budget_ms": 500.0,
        "memory_budget_mb": 2048.0,
        "shadowkv": {
            "use_packed_cache": True,
            "rd_allocation_enabled": True,
            "dense_precision_enabled": True,
        },
    }
    backend = PyTorchModelBackend(model_config=model_config, seed=1)
    backend.load("ar010-pipeline", device="cpu")
    torch.manual_seed(1)
    orchestrator = Orchestrator(config=config)
    orchestrator.set_model_backend(backend)
    orchestrator.initialize(model_config=backend.get_model_config())
    orchestrator.run_optimized(prompt="pipeline-check", num_steps=4)
    orchestrator.shutdown()
    print(f"plan_calls={calls['plan']} reference_calls={calls['reference']}")
    if calls["reference"] != 0:
        raise SystemExit("non-contiguous KV still falls back to the reference path")


if __name__ == "__main__":
    main()
