"""Tests for UncertaintyEarlyStopper."""

from __future__ import annotations

import pytest
import torch

from shadowinfer.qdrift.early_stopper import (
    EarlyStopConfig,
    UncertaintyEarlyStopper,
)


class TestUncertaintyEarlyStopper:
    def test_min_steps_enforced(self):
        stopper = UncertaintyEarlyStopper(
            min_steps=5, max_steps=20, stability_window=1, similarity_threshold=1.0
        )
        output = torch.randn(2, 8)
        for step in range(4):
            state = stopper.observe(step, output)
            assert not state.should_stop
        state = stopper.observe(4, output)
        assert state.should_stop

    def test_max_steps_hard_limit(self):
        stopper = UncertaintyEarlyStopper(
            min_steps=100, max_steps=5, stability_window=1, similarity_threshold=1.0
        )
        output = torch.randn(2, 8)
        for step in range(4):
            state = stopper.observe(step, output)
            assert not state.should_stop
        state = stopper.observe(4, output)
        assert state.should_stop

    def test_stability_window(self):
        stopper = UncertaintyEarlyStopper(
            min_steps=2, max_steps=20, stability_window=3, similarity_threshold=0.99
        )
        base = torch.randn(2, 8)
        for step in range(5):
            # Add tiny noise to keep similarity high.
            output = base + torch.randn_like(base) * 1e-4
            state = stopper.observe(step, output)
        assert state.stable_steps >= 3
        assert state.should_stop

    def test_reset(self):
        stopper = UncertaintyEarlyStopper(
            min_steps=1, max_steps=10, stability_window=1, similarity_threshold=1.0
        )
        output = torch.randn(2, 8)
        stopper.observe(0, output)
        stopper.reset()
        assert stopper._previous is None
        assert stopper._stable_count == 0
        assert not stopper.should_stop()

    def test_invalid_metric(self):
        with pytest.raises(ValueError):
            UncertaintyEarlyStopper(metric="unknown")

    def test_invalid_min_steps(self):
        with pytest.raises(ValueError):
            UncertaintyEarlyStopper(min_steps=0)

    def test_history(self):
        stopper = UncertaintyEarlyStopper()
        out1 = torch.randn(2, 8)
        out2 = torch.randn(2, 8)
        stopper.observe(0, out1)
        stopper.observe(1, out2)
        history = stopper.history()
        assert len(history) == 2
        assert history[0]["step_id"] == 0
        assert history[1]["step_id"] == 1

    def test_config_build_disabled(self):
        cfg = EarlyStopConfig(enabled=False)
        with pytest.raises(RuntimeError):
            cfg.build()

    def test_config_build_enabled(self):
        cfg = EarlyStopConfig(enabled=True, min_steps=3, max_steps=30)
        stopper = cfg.build()
        assert stopper.min_steps == 3
        assert stopper.max_steps == 30
