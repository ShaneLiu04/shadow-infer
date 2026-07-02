"""Tests for Orchestrator uncertainty-aware early stopping."""

from __future__ import annotations

import pytest
import torch

from shadowinfer.orchestrator import Orchestrator


class TestOrchestratorEarlyStop:
    def test_early_stop_triggered(self):
        orch = Orchestrator(
            config={
                "early_stop": {
                    "enabled": True,
                    "min_steps": 2,
                    "max_steps": 20,
                    "stability_window": 1,
                    "similarity_threshold": -10.0,
                    "metric": "relative_l2",
                }
            }
        )
        orch.initialize(
            model_config={
                "name": "test",
                "num_layers": 2,
                "num_heads": 2,
                "head_dim": 8,
                "hidden_dim": 16,
                "intermediate_dim": 32,
                "batch_size": 1,
                "seq_len": 4,
            }
        )
        result = orch.run_optimized(prompt="test", num_steps=20)
        # The output should stabilize quickly with the tiny mock model.
        assert len(orch.step_results) < 20
        assert any("early_stop" in step.get("outputs", {}) for step in orch.step_results)
        assert result["early_stopped"] is True
        assert result["stopped_step"] is not None
        assert result["actual_steps"] == len(orch.step_results)
        assert result["requested_steps"] == 20

    def test_early_stop_disabled_by_default(self):
        orch = Orchestrator(config={})
        orch.initialize(
            model_config={
                "name": "test",
                "num_layers": 2,
                "num_heads": 2,
                "head_dim": 8,
                "hidden_dim": 16,
                "intermediate_dim": 32,
                "batch_size": 1,
                "seq_len": 4,
            }
        )
        orch.run_optimized(prompt="test", num_steps=5)
        assert len(orch.step_results) == 5
        assert not any("early_stop" in step.get("outputs", {}) for step in orch.step_results)

    def test_stream_early_stop(self):
        orch = Orchestrator(
            config={
                "early_stop": {
                    "enabled": True,
                    "min_steps": 2,
                    "max_steps": 10,
                    "stability_window": 1,
                    "similarity_threshold": -10.0,
                    "metric": "relative_l2",
                }
            }
        )
        orch.initialize(
            model_config={
                "name": "test",
                "num_layers": 2,
                "num_heads": 2,
                "head_dim": 8,
                "hidden_dim": 16,
                "intermediate_dim": 32,
                "batch_size": 1,
                "seq_len": 4,
            }
        )
        states = list(orch.run_stream(prompt="test", num_steps=10))
        assert len(states) < 10
