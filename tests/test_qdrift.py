"""测试 QDrift Agent 核心算法。"""

import math

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


def _spearman(x, y):
    """手写 Spearman 秩相关（平均秩处理并列，无 scipy 依赖）。

    任一序列零方差（全部并列）时返回 0.0——恒定信号与任何扰动
    均无秩相关，这正是归一化塌缩缺陷（drift 恒 1.0）应被判定的情形。
    """

    def _ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = _ranks(x), _ranks(y)
    n = len(x)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mean_x) ** 2 for a in rx) * sum((b - mean_y) ** 2 for b in ry))
    if den == 0.0:
        return 0.0
    return num / den


class TestComputeSensitivity:
    """AR002 M3-lite：位置先验 + 实测扰动响应混合（FR3.4）。"""

    def _make_pair(self, seed, magnitude, scale=1.0):
        """构造已知扰动幅度的 current/previous 张量对。"""
        gen = torch.Generator().manual_seed(seed)
        q = torch.randn(2, 16, 8, 32, generator=gen)
        a = torch.randn(2, 16, 256, generator=gen)
        noise_q = torch.randn(q.shape, generator=gen)
        noise_a = torch.randn(a.shape, generator=gen)
        return (
            q,
            q + magnitude * scale * noise_q,
            a,
            a + magnitude * scale * noise_a,
        )

    def test_default_none_matches_positional(self, mock_config):
        """张量参数全部缺省时行为与 v3.3.0 位置先验完全一致（design 4.2.2）。"""
        agent = QDriftAgent(config=mock_config)
        blended = agent.compute_sensitivity(step_id=4, total_steps=10, noise_level=0.1)
        positional = agent.estimate_sensitivity(step_id=4, total_steps=10, noise_level=0.1)
        assert blended == positional

    def test_previous_missing_falls_back_and_logs(self, mock_config, monkeypatch):
        """previous 缺失回退位置先验并记录 info 日志（FR3.4 异常回退）。"""
        agent = QDriftAgent(config=mock_config)
        alerts = []
        monkeypatch.setattr(
            agent.logger, "log_alert", lambda level, msg, **kw: alerts.append((level, msg))
        )
        q, _, a, _ = self._make_pair(seed=0, magnitude=1.0)
        blended = agent.compute_sensitivity(
            step_id=4,
            total_steps=10,
            noise_level=0.1,
            query_current=q,
            query_previous=None,
            activation_current=a,
            activation_previous=None,
        )
        positional = agent.estimate_sensitivity(step_id=4, total_steps=10, noise_level=0.1)
        assert blended == positional
        assert len(alerts) == 1
        assert alerts[0][0] == "info"

    def test_blend_formula_matches_design(self, mock_config):
        """混合公式 = alpha*位置先验 + (1-alpha)*tanh(raw)，alpha 默认 0.3。"""
        agent = QDriftAgent(config=mock_config)
        q_cur, q_prev, a_cur, a_prev = self._make_pair(seed=1, magnitude=0.8)
        positional = agent.estimate_sensitivity(step_id=4, total_steps=10, noise_level=0.1)
        raw = agent._compute_raw_drift(q_cur, q_prev, a_cur, a_prev, method=agent.drift_method)
        expected = 0.3 * positional + 0.7 * math.tanh(raw)
        blended = agent.compute_sensitivity(
            step_id=4,
            total_steps=10,
            noise_level=0.1,
            query_current=q_cur,
            query_previous=q_prev,
            activation_current=a_cur,
            activation_previous=a_prev,
        )
        assert blended == pytest.approx(expected, abs=1e-9)

    def test_blend_responds_to_perturbation_magnitude(self, mock_config):
        """同位置先验下，扰动越大混合敏感度越高（实测信号生效）。"""
        agent = QDriftAgent(config=mock_config)
        small = self._make_pair(seed=2, magnitude=0.05)
        large = self._make_pair(seed=2, magnitude=3.0)
        kwargs = {"step_id": 4, "total_steps": 10, "noise_level": 0.1}
        sens_small = agent.compute_sensitivity(
            query_current=small[0],
            query_previous=small[1],
            activation_current=small[2],
            activation_previous=small[3],
            **kwargs,
        )
        sens_large = agent.compute_sensitivity(
            query_current=large[0],
            query_previous=large[1],
            activation_current=large[2],
            activation_previous=large[3],
            **kwargs,
        )
        assert sens_small < sens_large

    def test_drift_dynamic_range_restored(self, mock_config):
        """drift_score 修复恒 1.0 塌缩（FR3.4）：大幅扰动可分辨、不再触 1.0。

        历史缺陷：clamp(raw, 0, 1) 使 raw >= 1 的扰动全部塌缩为 1.0
        （T001 baseline-profile.md 实录：6 step 中 5 个 drift_score = 1.0）。
        """
        agent = QDriftAgent(config=mock_config)
        # 幅度选取需使 raw 保持在 float64 可分辨区间（tanh(x) 在 x>19 后
        # 双精度下饱和为 1.0，属浮点极限而非归一化缺陷）
        q2 = self._make_pair(seed=3, magnitude=2.0)
        q8 = self._make_pair(seed=3, magnitude=8.0)
        drift2 = agent.detect_drift(q2[0], q2[1], q2[2], q2[3])
        drift8 = agent.detect_drift(q8[0], q8[1], q8[2], q8[3])
        # 不再恒 1.0：raw >> 1 的扰动仍严格小于 1
        assert drift2 < 1.0
        assert drift8 < 1.0
        # 动态范围恢复：不同幅度大幅扰动给出可分辨的不同值
        assert drift2 != pytest.approx(drift8, abs=1e-3)
        assert drift8 > drift2

    def test_spearman_measured_response(self, mock_config):
        """FR3.4 验收：50 组与位置进度无关的不同幅度扰动，Spearman >= 0.9。

        位置进度固定（同一 step_id），扰动幅度 0.1-4.9 均匀铺开并
        跨越原 clamp 塌缩区（raw >= 1）；恒定位置先验下混合敏感度的
        秩序完全由实测扰动响应决定。若归一化再次塌缩（原 clamp 行为），
        大幅扰动段全部并列 → Spearman 显著低于 0.9。
        """
        agent = QDriftAgent(config=mock_config)
        magnitudes = [0.1 * (i + 1) for i in range(50)]
        sensitivities = []
        for idx, mag in enumerate(magnitudes):
            q_cur, q_prev, a_cur, a_prev = self._make_pair(seed=100 + idx, magnitude=mag)
            sens = agent.compute_sensitivity(
                step_id=4,
                total_steps=10,
                noise_level=0.1,
                query_current=q_cur,
                query_previous=q_prev,
                activation_current=a_cur,
                activation_previous=a_prev,
            )
            sensitivities.append(sens)
        rho = _spearman(magnitudes, sensitivities)
        assert rho >= 0.9, f"Spearman {rho:.4f} < 0.9：实测扰动响应失效"

    def test_on_step_uses_measured_signal(self, mock_config):
        """on_step 接线：提供张量时 sensitivity 偏离纯位置先验。"""
        config = dict(mock_config)
        agent_positional = QDriftAgent(config=config)
        agent_blended = QDriftAgent(config=config)
        q_cur, q_prev, a_cur, a_prev = self._make_pair(seed=4, magnitude=1.5)

        def _inputs(q_prev_val, a_prev_val):
            return {
                "step_id": 4,
                "total_steps": 10,
                "noise_level": 0.1,
                "query_current": q_cur,
                "query_previous": q_prev_val,
                "activation_current": a_cur,
                "activation_previous": a_prev_val,
            }

        step_config = StepConfig(
            step_id=4,
            total_steps=10,
            noise_level=0.1,
            shadowkv_mode="balanced",
            reuse_layers=[0],
            compression_target=0.5,
            ffn_mode="full",
            weight_precision_map={},
            compute_path="full",
            sensitivity_score=0.0,
            drift_score=0.0,
        )
        positional = agent_positional.on_step(step_config, _inputs(None, None))
        blended = agent_blended.on_step(step_config, _inputs(q_prev, a_prev))
        assert blended["sensitivity_score"] != pytest.approx(
            positional["sensitivity_score"], abs=1e-6
        )
        assert 0.0 < blended["drift_score"] < 1.0
