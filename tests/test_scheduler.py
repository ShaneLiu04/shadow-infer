"""Tests for the learned scheduler and its Orchestrator integration."""

import os
import tempfile

import torch

from shadowinfer.core.scheduler import LearnedScheduler, StepExperience
from shadowinfer.orchestrator import Orchestrator


class TestLearnedScheduler:
    def test_predict_returns_valid_action(self):
        """预测应返回合法的动作元组。"""
        scheduler = LearnedScheduler()
        action = scheduler.predict(
            step_id=5,
            total_steps=50,
            noise_level=0.1,
            sensitivity_score=0.2,
            drift_score=0.3,
        )
        assert isinstance(action, tuple)
        assert len(action) == 2
        shadowkv_mode, ffn_mode = action
        assert shadowkv_mode in ("conservative", "balanced", "aggressive")
        assert ffn_mode in ("full", "mixed", "sparse")

    def test_training_reduces_loss(self):
        """训练应降低经验上的平均损失。"""
        scheduler = LearnedScheduler()
        for i in range(20):
            exp = StepExperience(
                step_id=i,
                total_steps=20,
                noise_level=i / 20.0,
                sensitivity_score=0.1 * (i % 5),
                drift_score=0.05 * (i % 3),
                prev_latency_ms=10.0 + i,
                prev_memory_mb=100.0 + i * 2,
                prev_accuracy_drop=0.0,
                shadowkv_mode="balanced",
                ffn_mode="mixed",
                latency_ms=10.0 + i * 0.5,
                memory_mb=100.0 + i,
                accuracy_drop=0.0,
            )
            scheduler.add_experience(exp)

        with torch.no_grad():
            initial_values = scheduler.model(
                torch.stack([e.features() for e in scheduler.experiences])
            )
        initial_loss = float(
            (
                initial_values
                - torch.tensor([e.reward() for e in scheduler.experiences], dtype=torch.float32)
                .unsqueeze(1)
                .expand_as(initial_values)
            )
            .pow(2)
            .mean()
            .item()
        )

        losses = scheduler.train(epochs=50)
        assert len(losses) == 50
        final_loss = losses[-1]
        assert final_loss < initial_loss

    def test_save_load_roundtrip(self):
        """保存并加载后，预测结果应保持一致。"""
        scheduler = LearnedScheduler()
        for i in range(10):
            exp = StepExperience(
                step_id=i,
                total_steps=10,
                noise_level=0.5,
                sensitivity_score=0.5,
                drift_score=0.5,
                prev_latency_ms=1.0,
                prev_memory_mb=1.0,
                prev_accuracy_drop=0.0,
                shadowkv_mode="balanced",
                ffn_mode="mixed",
                latency_ms=1.0,
                memory_mb=1.0,
                accuracy_drop=0.0,
            )
            scheduler.add_experience(exp)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "scheduler_model.pt")
            scheduler.save(path)
            assert os.path.exists(path)

            loaded = LearnedScheduler(model_path=path)
            assert len(loaded.experiences) == len(scheduler.experiences)

            action_before = scheduler.predict(
                step_id=3,
                total_steps=10,
                noise_level=0.5,
                sensitivity_score=0.5,
                drift_score=0.5,
            )
            action_after = loaded.predict(
                step_id=3,
                total_steps=10,
                noise_level=0.5,
                sensitivity_score=0.5,
                drift_score=0.5,
            )
            assert action_before == action_after


class TestOrchestratorSchedulerIntegration:
    def test_orchestrator_with_scheduler_runs_step(self):
        """启用学习调度器后，Orchestrator 应能正常执行单步。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        scheduler = LearnedScheduler()
        orch.enable_learned_scheduler(scheduler=scheduler)
        orch.initialize()

        inputs = orch._make_dummy_inputs(0, 10)
        result = orch.run_step(0, 10, inputs)
        assert result["step_id"] == 0
        assert "qdrift" in result
        assert "shadowkv" in result
        assert "ffn" in result
        assert "profiler" in result
        assert len(orch.scheduler.experiences) == 1

    def test_orchestrator_scheduler_trains_on_shutdown(self):
        """关闭 Orchestrator 时应触发调度器训练。"""
        orch = Orchestrator(config="configs/optimize_full.yaml")
        scheduler = LearnedScheduler()
        orch.enable_learned_scheduler(scheduler=scheduler)
        orch.initialize()

        for step_id in range(3):
            inputs = orch._make_dummy_inputs(step_id, 3)
            orch.run_step(step_id, 3, inputs)

        assert len(orch.scheduler.experiences) == 3
        orch.shutdown()
        # Training is called internally; no exception means success.
