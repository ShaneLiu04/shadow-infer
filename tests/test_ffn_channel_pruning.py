"""Tests for FFN dynamic channel pruning."""

from __future__ import annotations

import pytest
import torch

from shadowinfer.core.structs import StepConfig
from shadowinfer.ffn_optimizer import FFNOptimizerAgent


class TestFFNChannelPruning:
    @pytest.fixture
    def small_weights(self):
        torch.manual_seed(0)
        return {
            "up": torch.randn(64, 32),
            "down": torch.randn(32, 64),
        }

    def test_static_channel_mask_shape(self, small_weights):
        agent = FFNOptimizerAgent(config={})
        agent.channel_importance = torch.rand(64)
        agent.channel_pruning_ratio = 0.5
        mask = agent.compute_channel_mask(
            torch.randn(1, 4, 32), small_weights["up"], method="static_importance"
        )
        assert mask.shape == (64,)
        assert mask.dtype == torch.bool
        assert mask.sum().item() == 32

    def test_dynamic_channel_mask_shape(self, small_weights):
        agent = FFNOptimizerAgent(config={})
        agent.channel_pruning_ratio = 0.5
        mask = agent.compute_channel_mask(
            torch.randn(1, 4, 32), small_weights["up"], method="dynamic_activation"
        )
        assert mask.shape == (64,)
        assert mask.sum().item() == 32

    def test_pruned_full_compute_output_shape(self, small_weights):
        agent = FFNOptimizerAgent(config={})
        agent.channel_importance = torch.rand(64)
        agent.channel_pruning_ratio = 0.5
        mask = agent.compute_channel_mask(
            torch.randn(1, 4, 32), small_weights["up"], method="static_importance"
        )
        x = torch.randn(1, 4, 32)
        output = agent._full_compute(x, small_weights["up"], small_weights["down"], mask)
        assert output.shape == (1, 4, 32)

    def test_pruned_full_matches_full_when_ratio_zero(self, small_weights):
        agent = FFNOptimizerAgent(config={})
        agent.channel_importance = torch.rand(64)
        agent.channel_pruning_ratio = 0.0
        mask = agent.compute_channel_mask(
            torch.randn(1, 4, 32), small_weights["up"], method="static_importance"
        )
        x = torch.randn(1, 4, 32)
        pruned = agent._full_compute(x, small_weights["up"], small_weights["down"], mask)
        full = agent._full_compute(x, small_weights["up"], small_weights["down"])
        assert torch.allclose(pruned, full, atol=1e-5)

    def test_on_step_channel_pruning_reduces_flops(
        self, mock_config, mock_ffn_weights, mock_ffn_inputs
    ):
        cfg = dict(mock_config)
        cfg.update(
            {
                "channel_pruning_enabled": True,
                "channel_pruning_ratio": 0.5,
                "channel_pruning_method": "static_importance",
            }
        )
        agent = FFNOptimizerAgent(config=cfg)
        agent.on_init({"model_name": "test", "weights": mock_ffn_weights})

        step_config = StepConfig(
            step_id=0,
            total_steps=10,
            noise_level=0.0,
            shadowkv_mode="balanced",
            ffn_mode="full",
            sensitivity_score=0.0,
            drift_score=0.0,
        )
        inputs = {
            "ffn_input_current": mock_ffn_inputs,
            "ffn_input_previous": None,
            "ffn_output_previous": None,
            "weights": mock_ffn_weights,
            "qdrift_signal": {"sensitivity_score": 0.2, "ffn_mode": "full"},
            "layer_id": 0,
        }
        result = agent.on_step(step_config, inputs)
        assert result["compute_path"] == "full"
        assert result["channel_pruning"]["enabled"] is True
        assert result["channel_pruning"]["active_ratio"] < 1.0
        assert result["compute_stats"]["flops_total"] > result["compute_stats"]["flops_saved"]

    def test_high_sensitivity_disables_channel_pruning(self, small_weights):
        cfg = {
            "channel_pruning_enabled": True,
            "channel_pruning_ratio": 0.5,
            "channel_pruning_method": "static_importance",
        }
        agent = FFNOptimizerAgent(config=cfg)
        agent.on_init({"model_name": "test", "weights": small_weights})
        agent.channel_importance = torch.rand(64)

        step_config = StepConfig(
            step_id=0,
            total_steps=10,
            noise_level=0.0,
            shadowkv_mode="balanced",
            ffn_mode="full",
            sensitivity_score=0.8,
            drift_score=0.0,
        )
        x = torch.randn(1, 4, 32)
        inputs = {
            "ffn_input_current": x,
            "ffn_input_previous": None,
            "ffn_output_previous": None,
            "weights": small_weights,
            "qdrift_signal": {"sensitivity_score": 0.8, "ffn_mode": "full"},
            "layer_id": 0,
        }
        result = agent.on_step(step_config, inputs)
        assert result["channel_pruning"]["active_ratio"] == 1.0
