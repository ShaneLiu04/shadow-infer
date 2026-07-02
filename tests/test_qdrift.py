"""测试 QDrift Agent 核心算法。"""

import pytest
import torch

from shadowinfer.core.structs import StepConfig
from shadowinfer.qdrift import QDriftAgent


class TestQDrift:
    def test_sensitivity_linear(self, mock_config):
        """验证 linear schedule 的敏感度单调递增。"""
        agent = QDriftAgent(config=mock_config)
        agent.on_init({"num_layers": 4})
        scores = []
        for step_id in range(0, 11):
            score = agent.estimate_sensitivity(
                step_id=step_id,
                total_steps=10,
                noise_level=0.0,
                noise_schedule="linear",
                temperature=1.0,
            )
            scores.append(score)
        # 单调递增（由于 noise_correction 在 step 0 不为零，起点非零）
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1] + 1e-9
        # step 0 有 noise_correction=1.0 贡献 0.3，故最小值约为 0.3
        assert scores[0] >= 0.25
        assert scores[-1] <= 1.0

    def test_sensitivity_cosine(self, mock_config):
        """验证 cosine schedule 的敏感度曲线。"""
        agent = QDriftAgent(config=mock_config)
        agent.on_init({"num_layers": 4})
        score_early = agent.estimate_sensitivity(
            step_id=0,
            total_steps=10,
            noise_level=0.0,
            noise_schedule="cosine",
            temperature=1.0,
        )
        score_mid = agent.estimate_sensitivity(
            step_id=5,
            total_steps=10,
            noise_level=0.0,
            noise_schedule="cosine",
            temperature=1.0,
        )
        score_late = agent.estimate_sensitivity(
            step_id=10,
            total_steps=10,
            noise_level=0.0,
            noise_schedule="cosine",
            temperature=1.0,
        )
        assert 0.0 <= score_early <= 1.0
        assert 0.0 <= score_mid <= 1.0
        assert 0.0 <= score_late <= 1.0
        assert score_early < score_mid < score_late

    def test_sensitivity_sigmoid(self, mock_config):
        """验证 sigmoid schedule 的敏感度曲线。"""
        agent = QDriftAgent(config=mock_config)
        agent.on_init({"num_layers": 4})
        score_early = agent.estimate_sensitivity(
            step_id=0,
            total_steps=10,
            noise_level=0.0,
            noise_schedule="sigmoid",
            temperature=1.0,
        )
        score_mid = agent.estimate_sensitivity(
            step_id=5,
            total_steps=10,
            noise_level=0.0,
            noise_schedule="sigmoid",
            temperature=1.0,
        )
        score_late = agent.estimate_sensitivity(
            step_id=10,
            total_steps=10,
            noise_level=0.0,
            noise_schedule="sigmoid",
            temperature=1.0,
        )
        assert 0.0 <= score_early <= 1.0
        assert 0.0 <= score_mid <= 1.0
        assert 0.0 <= score_late <= 1.0
        assert score_early < score_mid < score_late
        # 由于 noise_correction=0.3 的固定贡献，mid 点略高于 0.5
        assert score_mid >= 0.45

    def test_drift_relative_l2(self, mock_config):
        """验证 relative L2 漂移检测。"""
        agent = QDriftAgent(config=mock_config)
        q_current = torch.randn(1, 128, 32, 128)
        q_previous = q_current + 0.01 * torch.randn_like(q_current)
        a_current = torch.randn(1, 128, 4096)
        a_previous = a_current + 0.01 * torch.randn_like(a_current)

        drift = agent.detect_drift(
            q_current, q_previous, a_current, a_previous, method="relative_l2"
        )
        assert 0.0 <= drift <= 1.0
        # 几乎相同的张量，drift 应该很小
        drift_same = agent.detect_drift(
            q_current, q_current, a_current, a_current, method="relative_l2"
        )
        assert drift_same == pytest.approx(0.0, abs=0.01)

    def test_drift_cosine_similarity(self, mock_config):
        """验证 cosine similarity 漂移检测。"""
        agent = QDriftAgent(config=mock_config)
        q_current = torch.randn(1, 128, 32, 128)
        q_previous = q_current + 0.1 * torch.randn_like(q_current)
        a_current = torch.randn(1, 128, 4096)
        a_previous = a_current + 0.1 * torch.randn_like(a_current)

        drift = agent.detect_drift(
            q_current, q_previous, a_current, a_previous, method="cosine_similarity"
        )
        assert 0.0 <= drift <= 1.0
        # 相同张量 -> drift = 0
        drift_same = agent.detect_drift(
            q_current, q_current, a_current, a_current, method="cosine_similarity"
        )
        assert drift_same == pytest.approx(0.0, abs=0.01)

    def test_drift_kl_divergence(self, mock_config):
        """验证 KL divergence 漂移检测。"""
        agent = QDriftAgent(config=mock_config)
        q_current = torch.randn(1, 128, 32, 128)
        q_previous = torch.randn(1, 128, 32, 128)
        a_current = torch.randn(1, 128, 4096)
        a_previous = torch.randn(1, 128, 4096)

        drift = agent.detect_drift(
            q_current, q_previous, a_current, a_previous, method="kl_divergence"
        )
        assert 0.0 <= drift <= 1.0
        # 相同张量 -> drift ≈ 0
        drift_same = agent.detect_drift(
            q_current, q_current, a_current, a_current, method="kl_divergence"
        )
        assert drift_same == pytest.approx(0.0, abs=0.01)

    def test_dispatch_matrix(self, mock_config):
        """验证调度矩阵 5×3 映射正确。"""
        agent = QDriftAgent(config=mock_config)
        dispatch_cases = [
            # (sensitivity, drift) -> expected shadowkv_mode, ffn_mode
            ((0.1, 0.1), "aggressive", "sparse"),
            ((0.1, 0.3), "balanced", "sparse"),
            ((0.5, 0.1), "balanced", "mixed"),
            ((0.5, 0.3), "conservative", "mixed"),
            ((0.8, 0.0), "conservative", "full"),
            ((0.8, 0.5), "conservative", "full"),
        ]
        for (sens, drift), expected_kv, expected_ffn in dispatch_cases:
            dispatch = agent.generate_dispatch(sens, drift)
            assert dispatch["shadowkv_mode"] == expected_kv
            assert dispatch["ffn_mode"] == expected_ffn

    def test_adaptive_adjustment(self, mock_config):
        """验证自适应学习调整敏感度。"""
        agent = QDriftAgent(config=mock_config)
        base = 0.5
        # accuracy_delta > 0.005 -> 提高敏感度
        adjusted_high = agent.adaptive_adjustment(base, {"accuracy_delta": 0.01})
        assert adjusted_high > base
        assert adjusted_high <= 1.0

        # accuracy_delta < 0.001 -> 降低敏感度
        adjusted_low = agent.adaptive_adjustment(base, {"accuracy_delta": 0.0005})
        assert adjusted_low < base
        assert adjusted_low >= 0.0

        # 0.001 <= accuracy_delta <= 0.005 -> 不变
        adjusted_same = agent.adaptive_adjustment(base, {"accuracy_delta": 0.003})
        assert adjusted_same == pytest.approx(base, abs=0.01)

    def test_phase_detection(self, mock_config):
        """验证 early/mid/late 阶段检测。"""
        agent = QDriftAgent(config=mock_config)
        assert agent.get_phase(0, 10) == "early"
        assert agent.get_phase(3, 10) == "early"
        assert agent.get_phase(4, 10) == "mid"
        assert agent.get_phase(6, 10) == "mid"
        assert agent.get_phase(7, 10) == "late"
        assert agent.get_phase(10, 10) == "late"

    def test_sensitivity_monotonicity(self, mock_config):
        """验证后期敏感度不低于前期。"""
        agent = QDriftAgent(config=mock_config)
        for schedule in ["linear", "cosine", "sigmoid"]:
            scores = [
                agent.estimate_sensitivity(
                    step_id=i,
                    total_steps=10,
                    noise_level=0.0,
                    noise_schedule=schedule,
                    temperature=1.0,
                )
                for i in range(11)
            ]
            for i in range(len(scores) - 1):
                assert scores[i] <= scores[i + 1] + 1e-6, f"Non-monotonic in {schedule} at step {i}"

    def test_on_step_and_shutdown(self, mock_config):
        """验证 QDriftAgent on_step 和 on_shutdown 完整流程。"""
        agent = QDriftAgent(config=mock_config)
        agent.on_init({"num_layers": 4})

        step_config = StepConfig(
            step_id=0,
            total_steps=10,
            noise_level=0.0,
            shadowkv_mode="balanced",
            reuse_layers=[0],
            compression_target=0.5,
            ffn_mode="full",
            weight_precision_map={},
            compute_path="full",
            sensitivity_score=0.0,
            drift_score=0.0,
        )
        inputs = {
            "step_id": 0,
            "total_steps": 10,
            "noise_level": 0.0,
            "query_current": torch.randn(1, 128, 32, 128),
            "query_previous": torch.randn(1, 128, 32, 128),
            "activation_current": torch.randn(1, 128, 4096),
            "activation_previous": torch.randn(1, 128, 4096),
            "profiler_feedback": {"accuracy_delta": 0.001, "latency_ms": 10.0},
        }
        result = agent.on_step(step_config, inputs)
        assert "sensitivity_score" in result
        assert "drift_score" in result
        assert "dispatch" in result
        assert "step_phase" in result
        assert "learning_state" in result

        profile = agent.on_shutdown()
        assert profile is not None
        assert profile.model_name != ""
