"""AR003 T006 / FR3.5 ΔA 跨步差分信号导出与 QDrift 可选消费测试。

TDD Red 阶段：被测行为实现尚未存在——ShadowKVAgent 返回
``importance_drift`` / ``irap_stats``、step custom_metrics 字段、QDrift
``use_importance_drift`` 配置与受控融合、orchestrator 一步滞后接线均为
design §4.1「ΔA 信号」/ §4.3.4 / §4.3.5 / §6.1 UT-16~UT-18 与
§6.2「orchestrator ΔA 一步滞后」规定的目标行为，本文件只写测试。

用例：
- test_ut16_delta_a_first_step_none_and_monotonic
- test_ut16_step_stat_contains_importance_drift
- test_ut17_default_off_bitwise_identical
- test_ut18_fusion_enabled_formula
- test_ut18_invalid_value_ignored_with_warning
- test_qdrift_weight_validation
- test_orchestrator_one_step_lag
"""

import math

import pytest
import torch

from shadowinfer.core.structs import StepConfig
from shadowinfer.orchestrator import Orchestrator
from shadowinfer.qdrift import QDriftAgent
from shadowinfer.shadowkv import ShadowKVAgent
from shadowinfer.shadowkv.kv_cache_manager import KVCacheManager

SEED = 20261018
BATCH = 1
HEADS = 4
SEQ = 32
HEAD_DIM = 64
NUM_LAYERS = 1
LAYER_ID = 0

MODEL_CONFIG = {
    "num_layers": NUM_LAYERS,
    "num_heads": HEADS,
    "head_dim": HEAD_DIM,
    "model_name": "t",
}


# =============================================================================
# ShadowKVAgent：UT-16 ΔA 导出（首步 None + 突变幅度单调）
# =============================================================================


def _make_shadowkv(mock_config):
    """轻量 ShadowKVAgent（seq=32 / heads=4 / head_dim=64，关闭 packed cache）。"""
    config = dict(mock_config)
    config.update(
        {
            "num_layers": NUM_LAYERS,
            "num_heads": HEADS,
            "head_dim": HEAD_DIM,
            "use_packed_cache": False,
            "attn_is_probabilities": True,
        }
    )
    agent = ShadowKVAgent(config=config)
    agent.on_init(dict(MODEL_CONFIG))
    return agent


def _attention_probs(perturb_magnitude):
    """概率契约 attention：基础均匀分布 + 后半段列向注入 ``magnitude`` 质量。

    行已归一化（attn_is_probabilities=True），避免 logits 路径的 softmax
    饱和干扰突变幅度单调性构造。
    """
    probs = torch.ones(BATCH, HEADS, SEQ, SEQ, dtype=torch.float64)
    probs[..., SEQ // 2 :] += float(perturb_magnitude)
    return probs / probs.sum(dim=-1, keepdim=True)


def _kv_tensors(seed):
    gen = torch.Generator().manual_seed(seed)
    return {
        "k": torch.randn(BATCH, HEADS, SEQ, HEAD_DIM, generator=gen),
        "v": torch.randn(BATCH, HEADS, SEQ, HEAD_DIM, generator=gen),
    }


def _step_config(step_id, total_steps=2):
    return StepConfig(
        step_id=step_id,
        total_steps=total_steps,
        noise_level=0.5,
        shadowkv_mode="balanced",
    )


def _run_two_steps(mock_config, perturb_magnitude, seed=SEED):
    """同一 layer 连续两步：首步无 kv_previous，第二步带 kv_previous。"""
    agent = _make_shadowkv(mock_config)
    kv = _kv_tensors(seed)

    first = agent.on_step(
        _step_config(1),
        {
            "attention_scores": _attention_probs(0.0),
            "kv_current": kv,
            "kv_previous": None,
            "layer_id": LAYER_ID,
            "attn_is_probabilities": True,
        },
    )
    second = agent.on_step(
        _step_config(2),
        {
            "attention_scores": _attention_probs(perturb_magnitude),
            "kv_current": kv,
            "kv_previous": kv,
            "layer_id": LAYER_ID,
            "attn_is_probabilities": True,
        },
    )
    return agent, first, second


def test_ut16_delta_a_first_step_none_and_monotonic(mock_config):
    """UT-16：首步 importance_drift=None；突变 0.1/0.5/2.0 下 delta_a 严格递增。"""
    deltas = []
    for magnitude in (0.1, 0.5, 2.0):
        _agent, first, second = _run_two_steps(mock_config, magnitude)

        assert first["importance_drift"] is None, "首步无 A_prev，importance_drift 必须为 None"
        assert first["irap_stats"]["delta_a"] is None
        assert second["importance_drift"] is not None, "第二步应导出 ΔA 信号"
        assert second["importance_drift"] > 0.0

        # 返回值与 irap_stats 一致
        assert second["importance_drift"] == second["irap_stats"]["delta_a"]
        assert second["irap_stats"]["seq_k"] == SEQ
        assert second["irap_stats"]["gamma"] == pytest.approx(0.9)

        deltas.append(second["importance_drift"])

    assert deltas[0] < deltas[1] < deltas[2], f"ΔA 未严格单调递增: {deltas}"


def test_ut16_step_stat_contains_importance_drift(mock_config):
    """UT-16：step custom_metrics 含 importance_drift（首步 None、次步数值）。"""
    agent, first, second = _run_two_steps(mock_config, 0.5)

    first_metrics = agent.step_stats[1].custom_metrics
    second_metrics = agent.step_stats[2].custom_metrics

    assert "importance_drift" in first_metrics
    assert first_metrics["importance_drift"] is None
    assert "importance_drift" in second_metrics
    assert (
        second_metrics["importance_drift"]
        == second["importance_drift"]
        == (second["irap_stats"]["delta_a"])
    )


# =============================================================================
# QDriftAgent：UT-17 默认关逐位一致 / UT-18 融合公式与非法值
# =============================================================================


def _make_qdrift(mock_config, **overrides):
    config = dict(mock_config)
    config.update(overrides)
    agent = QDriftAgent(config=config)
    agent.on_init({"num_layers": 4})
    return agent


def _make_drift_pair(seed=0, magnitude=0.5):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(2, 16, 8, 32, generator=gen)
    a = torch.randn(2, 16, 256, generator=gen)
    noise_q = torch.randn(q.shape, generator=gen)
    noise_a = torch.randn(a.shape, generator=gen)
    return q, q + magnitude * noise_q, a, a + magnitude * noise_a


def _qdrift_step_config(step_id=0, total_steps=10):
    return StepConfig(
        step_id=step_id,
        total_steps=total_steps,
        noise_level=0.1,
        shadowkv_mode="balanced",
    )


def test_ut17_default_off_bitwise_identical(mock_config):
    """UT-17：use_importance_drift=False 时传/不传 importance_drift 逐位一致。

    drift_score 用 ``==``（非 approx）；on_step 的 sensitivity_score 亦不变。
    """
    agent = _make_qdrift(mock_config)
    q_cur, q_prev, a_cur, a_prev = _make_drift_pair(seed=7, magnitude=0.5)

    baseline = agent.detect_drift(q_cur, q_prev, a_cur, a_prev)
    with_delta = agent.detect_drift(q_cur, q_prev, a_cur, a_prev, importance_drift=0.9)
    assert with_delta == baseline, "默认关闭时 drift_score 必须逐位一致"

    inputs = {
        "step_id": 0,
        "total_steps": 10,
        "noise_level": 0.1,
        "query_current": q_cur,
        "query_previous": q_prev,
        "activation_current": a_cur,
        "activation_previous": a_prev,
    }
    result_plain = agent.on_step(_qdrift_step_config(), dict(inputs))
    result_with = agent.on_step(_qdrift_step_config(), dict(inputs, importance_drift=0.9))
    assert result_with["drift_score"] == result_plain["drift_score"]
    assert result_with["sensitivity_score"] == result_plain["sensitivity_score"]


def test_ut18_fusion_enabled_formula(mock_config):
    """UT-18：开启后 raw=(1-w)*raw_qa+w*delta，drift=tanh(raw)，方向正确。"""
    weight = 0.3
    agent = _make_qdrift(
        mock_config,
        use_importance_drift=True,
        importance_drift_weight=weight,
    )
    q_cur, q_prev, a_cur, a_prev = _make_drift_pair(seed=11, magnitude=0.5)

    raw_qa = agent._compute_raw_drift(q_cur, q_prev, a_cur, a_prev)
    assert raw_qa > 0.0

    delta = 0.6
    expected = math.tanh((1.0 - weight) * raw_qa + weight * delta)
    fused = agent.detect_drift(q_cur, q_prev, a_cur, a_prev, importance_drift=delta)
    assert fused == pytest.approx(expected, abs=1e-12)

    # 方向正确：delta 增大 → drift 增大
    smaller = agent.detect_drift(q_cur, q_prev, a_cur, a_prev, importance_drift=0.2)
    larger = agent.detect_drift(q_cur, q_prev, a_cur, a_prev, importance_drift=0.8)
    assert smaller < fused < larger


def test_ut18_invalid_value_ignored_with_warning(mock_config, monkeypatch):
    """UT-18：NaN / -1 / None 忽略（与不传一致），warning 至多一次。"""
    agent = _make_qdrift(
        mock_config,
        use_importance_drift=True,
        importance_drift_weight=0.3,
    )
    q_cur, q_prev, a_cur, a_prev = _make_drift_pair(seed=13, magnitude=0.5)
    baseline = agent.detect_drift(q_cur, q_prev, a_cur, a_prev)

    warnings = []
    monkeypatch.setattr(
        agent.logger,
        "log_alert",
        lambda level, message, **kwargs: warnings.append((level, message)),
    )

    for invalid in (float("nan"), -1.0, None):
        result = agent.detect_drift(q_cur, q_prev, a_cur, a_prev, importance_drift=invalid)
        assert result == baseline, f"非法值 {invalid!r} 应被忽略且结果逐位一致"

    assert len(warnings) == 1, f"非法值应告警且至多一次，实际 {len(warnings)} 次"
    assert warnings[0][0] == "warning"


def test_qdrift_weight_validation(mock_config):
    """importance_drift_weight ∉ [0, 1] → on_init ValueError。"""
    for invalid_weight in (-0.1, 1.1):
        agent = QDriftAgent(
            config=dict(
                mock_config,
                use_importance_drift=True,
                importance_drift_weight=invalid_weight,
            )
        )
        with pytest.raises(ValueError):
            agent.on_init({"num_layers": 4})


# =============================================================================
# Orchestrator：ΔA 一步滞后接线
# =============================================================================


class _QDriftSpy:
    """记录 on_step 收到的 qdrift_inputs 的最小替身。"""

    def __init__(self):
        self.name = "qdrift"
        self.calls = []

    def on_step(self, step_config, inputs):
        self.calls.append(dict(inputs))
        return {
            "sensitivity_score": 0.3,
            "drift_score": 0.1,
            "dispatch": {"shadowkv_mode": "balanced", "ffn_mode": "mixed"},
            "step_phase": "early",
            "learning_state": {},
        }


class _ShadowKVSpy:
    """每层返回确定性 importance_drift 的最小替身（step 0 -> 0.42，step 1 -> 0.84）。"""

    def __init__(self, num_layers=2):
        self.name = "shadowkv"
        self.calls = []
        self.prefetch_enabled = False
        self.cache_manager = KVCacheManager(num_layers)

    def on_step(self, step_config, kv_inputs):
        self.calls.append((step_config.step_id, dict(kv_inputs)))
        value = 0.42 if step_config.step_id == 0 else 0.84
        return {
            "precision_map": {},
            "reuse_decision": {},
            "compressed_kv": {"memory_mb": 0.0},
            "importance_drift": value,
            "irap_stats": {"delta_a": value, "gamma": 0.9, "seq_k": 8},
        }


def test_orchestrator_one_step_lag():
    """UT §6.2：step t 的 ΔA 在 step t+1 的 qdrift inputs 中；step 0 为 None。"""
    model_config = {
        "name": "tiny-test",
        "num_layers": 2,
        "num_heads": 2,
        "head_dim": 16,
        "hidden_dim": 32,
        "intermediate_dim": 64,
        "vocab_size": 64,
        "max_seq_len": 32,
        "batch_size": 1,
        "seq_len": 8,
    }
    orch = Orchestrator(config={})
    orch.initialize(dict(model_config))

    qdrift_spy = _QDriftSpy()
    shadowkv_spy = _ShadowKVSpy(num_layers=2)
    orch.qdrift = qdrift_spy
    orch.shadowkv = shadowkv_spy

    orch.run_optimized(prompt="test", num_steps=2)

    assert len(qdrift_spy.calls) == 2
    assert len(shadowkv_spy.calls) == 4, "应逐层调用 shadowkv（2 层 × 2 步）"

    assert qdrift_spy.calls[0]["importance_drift"] is None
    assert qdrift_spy.calls[1]["importance_drift"] == pytest.approx(0.42)
