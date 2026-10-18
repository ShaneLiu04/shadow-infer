"""ShadowKV attention 输入契约显式化与单次归一化测试（AR003 T002 / FR3.1，KV-P2）。

TDD Red 阶段：被测行为实现尚未存在——config/inputs 契约键
``attn_is_probabilities``、单次归一化语义、两路径签名扩展、决策缓存指纹
扩展与 orchestrator kv_inputs 接线均为 design 4.1 / 4.2.2 / 4.3.1 / 4.3.5
规定的目标行为，本文件只写测试，不写实现。

覆盖 design 6.1 归属 FR3.1 的用例：
- UT-1  契约-概率输入：异质构造（3/4 query 行 x6 锐化 + softmax，
        seed=20261017，参照 baseline-profile 1）下非 fp32 档占比 >= 50%
        且档位种类 >= 3；与双重 softmax（现状）结果不等
- UT-1b 行和安全校正：行和非 1 的概率输入校正后与严格归一化输入 <= 1e-6
- UT-2  契约-logits 输入：单次 softmax，与现行公式复刻 <= 1e-7
- UT-3  契约-默认告警：配置缺失 -> log_alert("warning") 恰好一次且按 logits 处理
- 指纹用例：同 (step, layer) 同张量但契约标志不同 -> 缓存不命中（重算）
- 优先级用例：inputs 键 > config 键
- golden 用例：probs 契约下向量化 vs 标量参考 <= 1e-6；probs 直喂 ==
        现行公式复刻；probs 契约 == mock logits 路径（baseline 1 第三行）
- 接线用例：_run_shadowkv 构造的 kv_inputs 含
        attn_is_probabilities = self._is_real_backend()
"""

import math

import pytest
import torch
import torch.nn.functional as F

from shadowinfer.core.backends.pytorch_backend import PyTorchModelBackend
from shadowinfer.core.structs import StepConfig
from shadowinfer.orchestrator import Orchestrator
from shadowinfer.shadowkv import ShadowKVAgent
from shadowinfer.shadowkv.kv_cache_manager import KVCacheManager

SEED = 20261017
BATCH = 1
HEADS = 8
SEQ = 128
HEAD_DIM = 64  # int4 量化要求 head_dim 可被 group_size(64) 整除（on_step 全链路）
LAYER_ID = 0
NUM_LAYERS = 4

MODEL_CONFIG = {
    "num_layers": NUM_LAYERS,
    "num_heads": HEADS,
    "head_dim": HEAD_DIM,
    "model_name": "t",
}


def _step_config(step_id=0, total_steps=6, mode="balanced"):
    return StepConfig(
        step_id=step_id,
        total_steps=total_steps,
        noise_level=0.5,
        shadowkv_mode=mode,
    )


def _make_agent(mock_config, **overrides):
    """构造已完成 on_init 的 ShadowKVAgent（轻量维度，关闭 packed cache）。"""
    config = dict(mock_config)
    config.update(
        {
            "num_layers": NUM_LAYERS,
            "num_heads": HEADS,
            "head_dim": HEAD_DIM,
            "use_packed_cache": False,
        }
    )
    config.update(overrides)
    agent = ShadowKVAgent(config=config)
    agent.on_init(dict(MODEL_CONFIG))
    return agent


def _counting_agent(mock_config, **overrides):
    """构造带 _compute_importance_matrix 调用计数的 agent（缓存命中观测）。"""
    agent = _make_agent(mock_config, **overrides)
    calls = {"count": 0}
    original = agent._compute_importance_matrix

    def _counting(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    agent._compute_importance_matrix = _counting
    return agent, calls


def _make_hetero_probs(seed=SEED, sharpen_rows=3 * SEQ // 4):
    """异质注意力构造（baseline-profile 1 口径，锐化行占比提升到 3/4）。

    返回 (logits, probs)：probs = softmax(logits)，行已归一化，
    模拟真实后端 SimpleDiffusionLLM 的 post-softmax 返回。
    """
    torch.manual_seed(seed)
    logits = torch.randn(BATCH, HEADS, SEQ, SEQ)
    logits[:, :, :sharpen_rows, :] = logits[:, :, :sharpen_rows, :] * 6.0
    probs = torch.softmax(logits, dim=-1)
    return logits, probs


def _make_logits(seed=SEED + 1):
    """mock 路径语义的 logits 输入（orchestrator randn 口径）。"""
    torch.manual_seed(seed)
    return torch.randn(BATCH, HEADS, SEQ, SEQ)


def _make_inputs(attention, seed=SEED + 2, **extra):
    """构造 on_step 输入；extra 追加/覆盖输入键。"""
    torch.manual_seed(seed)
    kv = {
        "k": torch.randn(BATCH, HEADS, SEQ, HEAD_DIM),
        "v": torch.randn(BATCH, HEADS, SEQ, HEAD_DIM),
    }
    inputs = {
        "attention_scores": attention,
        "kv_current": kv,
        "layer_id": LAYER_ID,
        "qdrift_signal": {"sensitivity_score": 0.3, "drift_score": 0.1},
    }
    inputs.update(extra)
    return inputs


def _correct_rows(weights):
    """契约 True 路径的行和数值安全校正（design 4.1 规格复刻）。"""
    return weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-10)


def _replica_importance(weights, layer_index=LAYER_ID, num_layers=NUM_LAYERS):
    """现行公式复刻：把 weights 直接当作归一化权重喂入（golden 基准）。

    与 shadowkv_agent._compute_importance_matrix 现行实现逐算子一致，
    仅以入参 weights 替代其中的 softmax 输出。
    """
    num_query = weights.shape[2]
    seq_len = weights.shape[-1]
    log_seq = math.log(seq_len)
    clamped = weights.clamp(min=1e-10)
    denom = clamped.sum(dim=(0, 3))
    probs = clamped / denom.unsqueeze(0).unsqueeze(-1)
    entropy = -(probs * torch.log(probs)).sum(dim=(0, 3))
    normalized_entropy = entropy / log_seq

    layer_factor = 1.0 + 0.1 * (layer_index / max(num_layers, 1))

    pos_factor = torch.ones(num_query, dtype=weights.dtype)
    pos_factor[:5] = 1.2
    pos_start = max(seq_len - 4, 0)
    if pos_start < num_query:
        pos_factor[pos_start:] = 1.2

    scores = normalized_entropy.t() * pos_factor.unsqueeze(1) * layer_factor
    return scores.clamp(max=1.0).float()


def _precision_summary(precision_map):
    """返回 (非 fp32 占比, 档位种类数)。"""
    counts = {}
    total = 0
    for token_map in precision_map.values():
        for prec in token_map.values():
            counts[prec] = counts.get(prec, 0) + 1
            total += 1
    non_fp32 = 1.0 - counts.get("fp32", 0) / total
    return non_fp32, len(counts)


class TestUT1ProbabilitiesContract:
    """UT-1 契约-概率输入（FR3.1 验收：档位散开 + 与双重 softmax 不等）。"""

    def test_ut1_probs_contract_spreads_precision_tiers(self, mock_config):
        """probs 契约（跳过 softmax）下异质构造：非 fp32 >= 50% 且档位 >= 3。

        修复前（probs 再 softmax）同构造全 fp32 塌缩（baseline-profile 1）。
        """
        agent = _make_agent(mock_config, attn_is_probabilities=True)
        _, probs = _make_hetero_probs()

        result = agent.on_step(_step_config(), _make_inputs(probs))

        non_fp32, kinds = _precision_summary(result["precision_map"])
        assert non_fp32 >= 0.5, f"非 fp32 档占比 {non_fp32:.3f} 应 >= 50%"
        assert kinds >= 3, f"档位种类 {kinds} 应 >= 3，实际分布应散开"

    def test_ut1_double_softmax_contrast_collapses(self, mock_config):
        """历史缺陷锚点（AR003 T003 修订）与 IRAP 鲁棒性。

        - 旧公式（`_replica_importance`）+ 双重 softmax：全 fp32 塌缩
          （baseline-profile 1 的历史缺陷复现，作为回归锚点保留）；
        - 新 IRAP 公式（rank 主信号）对双重 softmax 鲁棒：生产路径不再塌缩，
          但两契约的重要性均值仍可区分。
        """
        agent = _make_agent(mock_config, attn_is_probabilities=True)
        _, probs = _make_hetero_probs()
        inputs = _make_inputs(probs)

        result_true = agent.on_step(_step_config(), dict(inputs, attn_is_probabilities=True))
        result_false = agent.on_step(_step_config(), dict(inputs, attn_is_probabilities=False))

        # 1) 历史缺陷复现（旧公式 + 双重 softmax，仅测试内复刻，不参与生产路径）
        old_imp = _replica_importance(F.softmax(probs, dim=-1))
        old_precision = agent.allocate_precision(old_imp, "balanced")
        non_fp32_old, _ = _precision_summary(old_precision)
        assert non_fp32_old < 0.5, "旧公式双重 softmax 应全 fp32 塌缩（历史锚点）"

        # 2) 新公式鲁棒 + 契约可区分（秩主信号不受熵饱和影响）
        non_fp32_false, _ = _precision_summary(result_false["precision_map"])
        assert non_fp32_false >= 0.5, "IRAP 秩主信号应使双重 softmax 下档位仍散开"
        mean_true = result_true["importance_stats"]["mean"]
        mean_false = result_false["importance_stats"]["mean"]
        assert abs(mean_true - mean_false) > 1e-3, "两契约结果应不等"


class TestUT1bRowSumCorrection:
    """UT-1b 行和非 1 的概率输入 -> 校正后与严格归一化结果 <= 1e-6。"""

    def test_ut1b_unnormalized_rows_corrected_to_strict(self, mock_config):
        agent = _make_agent(mock_config, attn_is_probabilities=True)
        _, probs = _make_hetero_probs()

        torch.manual_seed(SEED + 3)
        perturbed = probs * (1.0 + 0.05 * torch.randn(probs.shape))
        assert not torch.allclose(
            perturbed.sum(dim=-1), torch.ones_like(perturbed.sum(dim=-1))
        ), "构造应使行和偏离 1"
        strict = perturbed / perturbed.sum(dim=-1, keepdim=True)

        imp_perturbed = agent._compute_importance_matrix(
            perturbed, LAYER_ID, NUM_LAYERS, attn_is_probabilities=True
        )
        imp_strict = agent._compute_importance_matrix(
            strict, LAYER_ID, NUM_LAYERS, attn_is_probabilities=True
        )

        diff = (imp_perturbed - imp_strict).abs().max().item()
        assert diff <= 1e-6, f"校正后与严格归一化差异 {diff:.3e} 应 <= 1e-6"

    def test_ut1b_zero_row_stays_finite(self, mock_config):
        """全零行（行和为 0）经 clamp(min=1e-10) 校正后输出有限，不产生 NaN/Inf。"""
        agent = _make_agent(mock_config, attn_is_probabilities=True)
        _, probs = _make_hetero_probs()
        probs[:, :, 3, :] = 0.0

        imp = agent._compute_importance_matrix(
            probs, LAYER_ID, NUM_LAYERS, attn_is_probabilities=True
        )
        assert torch.isfinite(imp).all(), "数值安全校正应保证零行输出有限"


class TestUT2LogitsContract:
    """UT-2 契约-logits 输入：单次 softmax，与现行公式复刻一致。"""

    def test_ut2_logits_single_softmax_matches_replica(self, mock_config):
        """logits + False 契约 = 单次 softmax 后喂入 IRAP 公式复刻（<= 1e-7）。

        AR003 T003 修订：复刻目标由旧熵公式切换为新统一公式
        （_replica_new_formula，根因：FR3.2 公式变更）。
        """
        agent = _make_agent(mock_config)
        logits, _ = _make_hetero_probs()

        imp = agent._compute_importance_matrix(
            logits, LAYER_ID, NUM_LAYERS, attn_is_probabilities=False
        )
        expected = _replica_new_formula(F.softmax(logits, dim=-1))

        diff = (imp - expected).abs().max().item()
        assert diff <= 1e-7, f"logits 单次 softmax 与统一公式复刻差异 {diff:.3e}"

    def test_softmax_once_for_logits_never_for_probs(self, mock_config, monkeypatch):
        """False 契约恰好调用一次 softmax；True 契约完全跳过 softmax。"""
        agent = _make_agent(mock_config)
        logits, probs = _make_hetero_probs()

        calls = {"n": 0}
        real_softmax = torch.nn.functional.softmax

        def _spy_softmax(*args, **kwargs):
            calls["n"] += 1
            return real_softmax(*args, **kwargs)

        monkeypatch.setattr("shadowinfer.shadowkv.shadowkv_agent.F.softmax", _spy_softmax)

        agent._compute_importance_matrix(logits, LAYER_ID, NUM_LAYERS, attn_is_probabilities=False)
        assert calls["n"] == 1, "logits 契约应恰好执行一次 softmax"

        calls["n"] = 0
        agent._compute_importance_matrix(probs, LAYER_ID, NUM_LAYERS, attn_is_probabilities=True)
        assert calls["n"] == 0, "probs 契约应跳过 softmax（仅行和校正）"


class TestContractSemanticsGolden:
    """golden：True 路径 == 把输入当作权重直接喂入现行公式。"""

    def test_probs_contract_equals_direct_weight_feed_replica(self, mock_config):
        """True 路径重要性 == 校正后权重直喂统一公式的测试内复刻（<= 1e-6）。

        AR003 T003 修订：复刻目标为新统一公式（根因：FR3.2 公式变更）。
        """
        agent = _make_agent(mock_config)
        _, probs = _make_hetero_probs()

        imp = agent._compute_importance_matrix(
            probs, LAYER_ID, NUM_LAYERS, attn_is_probabilities=True
        )
        expected = _replica_new_formula(_correct_rows(probs))

        diff = (imp - expected).abs().max().item()
        assert diff <= 1e-6, f"probs 直喂复刻差异 {diff:.3e} 应 <= 1e-6"

    def test_probs_contract_matches_mock_logits_path(self, mock_config):
        """probs 契约结果与 mock logits 单次 softmax 路径逐位一致（baseline 1 第三行）。"""
        agent = _make_agent(mock_config)
        logits, probs = _make_hetero_probs()

        imp_true = agent._compute_importance_matrix(
            probs, LAYER_ID, NUM_LAYERS, attn_is_probabilities=True
        )
        imp_mock = agent._compute_importance_matrix(
            logits, LAYER_ID, NUM_LAYERS, attn_is_probabilities=False
        )

        diff = (imp_true - imp_mock).abs().max().item()
        assert diff <= 1e-6, f"probs 契约与 mock 路径差异 {diff:.3e} 应 <= 1e-6"


class TestUT3MissingConfigWarning:
    """UT-3 契约-默认告警：配置缺失按 logits 处理并 warning 恰好一次。"""

    def test_ut3_missing_config_warns_once_and_uses_logits_semantics(
        self, mock_config, monkeypatch
    ):
        agent = _make_agent(mock_config)
        assert "attn_is_probabilities" not in agent.config

        alerts = []
        monkeypatch.setattr(
            agent.logger, "log_alert", lambda level, msg, **kw: alerts.append((level, msg))
        )

        logits = _make_logits()
        inputs = _make_inputs(logits)
        result0 = agent.on_step(_step_config(step_id=0), inputs)
        agent.on_step(_step_config(step_id=1), inputs)
        agent.on_step(_step_config(step_id=2), inputs)

        warnings = [a for a in alerts if a[0] == "warning"]
        assert len(warnings) == 1, f"配置缺失应 warning 恰好一次，实际 {len(warnings)}"

        expected = agent._compute_importance_matrix(
            logits, LAYER_ID, NUM_LAYERS, attn_is_probabilities=False
        )
        assert result0["importance_stats"]["mean"] == pytest.approx(
            float(expected.mean()), abs=1e-6
        ), "配置缺失应按 logits（False）语义处理"

    def test_ut3_explicit_config_respected_and_no_warning(self, mock_config, monkeypatch):
        """config 显式设置契约时不告警，且该契约实际生效。"""
        agent = _make_agent(mock_config, attn_is_probabilities=True)

        alerts = []
        monkeypatch.setattr(
            agent.logger, "log_alert", lambda level, msg, **kw: alerts.append((level, msg))
        )

        _, probs = _make_hetero_probs()
        result = agent.on_step(_step_config(), _make_inputs(probs))

        assert not [a for a in alerts if a[0] == "warning"], "显式配置不应告警"

        expected = agent._compute_importance_matrix(
            probs, LAYER_ID, NUM_LAYERS, attn_is_probabilities=True
        )
        assert result["importance_stats"]["mean"] == pytest.approx(
            float(expected.mean()), abs=1e-6
        ), "config 契约键应生效（probs 跳过 softmax）"


class TestContractKeyPriority:
    """inputs 键优先级高于 config 键（design 4.3.1）。"""

    def test_inputs_true_overrides_config_false(self, mock_config):
        agent = _make_agent(mock_config, attn_is_probabilities=False)
        _, probs = _make_hetero_probs()
        inputs = _make_inputs(probs)

        result_true = agent.on_step(_step_config(), dict(inputs, attn_is_probabilities=True))
        result_false = agent.on_step(_step_config(), dict(inputs, attn_is_probabilities=False))

        expected_true = agent._compute_importance_matrix(
            probs, LAYER_ID, NUM_LAYERS, attn_is_probabilities=True
        )
        assert result_true["importance_stats"]["mean"] == pytest.approx(
            float(expected_true.mean()), abs=1e-6
        ), "inputs 键 True 应覆盖 config False"

        mean_true = result_true["importance_stats"]["mean"]
        mean_false = result_false["importance_stats"]["mean"]
        assert abs(mean_true - mean_false) > 1e-3, "两契约结果应可区分"

    def test_inputs_false_overrides_config_true(self, mock_config):
        agent = _make_agent(mock_config, attn_is_probabilities=True)
        _, probs = _make_hetero_probs()
        inputs = _make_inputs(probs)

        result_true = agent.on_step(_step_config(), dict(inputs, attn_is_probabilities=True))
        result_false = agent.on_step(_step_config(), dict(inputs, attn_is_probabilities=False))

        expected_false = agent._compute_importance_matrix(
            probs, LAYER_ID, NUM_LAYERS, attn_is_probabilities=False
        )
        assert result_false["importance_stats"]["mean"] == pytest.approx(
            float(expected_false.mean()), abs=1e-6
        ), "inputs 键 False 应覆盖 config True"

        mean_true = result_true["importance_stats"]["mean"]
        mean_false = result_false["importance_stats"]["mean"]
        assert abs(mean_true - mean_false) > 1e-3, "两契约结果应可区分"


class TestDecisionCacheFingerprint:
    """契约标志计入决策缓存指纹（design 4.2.2 点 1）。"""

    def test_fingerprint_includes_contract_flag(self, mock_config):
        """同 (step, layer) 同张量但契约不同 -> 不命中缓存（重算）；同契约重复 -> 命中。"""
        agent, calls = _counting_agent(mock_config)
        _, probs = _make_hetero_probs()
        inputs = _make_inputs(probs)

        agent.on_step(_step_config(), dict(inputs, attn_is_probabilities=True))
        assert calls["count"] == 1

        agent.on_step(_step_config(), dict(inputs, attn_is_probabilities=False))
        assert calls["count"] == 2, "契约标志不同应使指纹不同（缓存不命中）"

        agent.on_step(_step_config(), dict(inputs, attn_is_probabilities=False))
        assert calls["count"] == 2, "同契约重复调用应命中缓存"


class TestVecRefGolden:
    """golden：向量化与标量参考在两契约下逐元素一致（<= 1e-6）。"""

    def test_golden_vec_vs_scalar_probabilities_contract(self, mock_config):
        agent = _make_agent(mock_config)
        _, probs = _make_hetero_probs()

        vec = agent._compute_importance_matrix(
            probs, LAYER_ID, NUM_LAYERS, attn_is_probabilities=True
        )
        seq_q = probs.shape[2]
        ref = torch.zeros(seq_q, HEADS)
        for token_idx in range(seq_q):
            for head_idx in range(HEADS):
                ref[token_idx, head_idx] = agent.compute_importance_score(
                    probs,
                    token_idx,
                    head_idx,
                    LAYER_ID,
                    NUM_LAYERS,
                    attn_is_probabilities=True,
                )

        diff = (vec - ref).abs().max().item()
        assert diff <= 1e-6, f"probs 契约 vec/ref 差异 {diff:.3e} 应 <= 1e-6"

    def test_golden_vec_vs_scalar_logits_contract(self, mock_config):
        agent = _make_agent(mock_config)
        logits, _ = _make_hetero_probs()

        vec = agent._compute_importance_matrix(
            logits, LAYER_ID, NUM_LAYERS, attn_is_probabilities=False
        )
        seq_q = logits.shape[2]
        ref = torch.zeros(seq_q, HEADS)
        for token_idx in range(seq_q):
            for head_idx in range(HEADS):
                ref[token_idx, head_idx] = agent.compute_importance_score(
                    logits,
                    token_idx,
                    head_idx,
                    LAYER_ID,
                    NUM_LAYERS,
                    attn_is_probabilities=False,
                )

        diff = (vec - ref).abs().max().item()
        assert diff <= 1e-6, f"logits 契约 vec/ref 差异 {diff:.3e} 应 <= 1e-6"


class _ShadowKVSpy:
    """记录 on_step 收到的 kv_inputs 的最小 shadowkv 替身。"""

    def __init__(self, num_layers=2):
        self.calls = []
        self.prefetch_enabled = False
        self.cache_manager = KVCacheManager(num_layers)

    def on_step(self, step_config, kv_inputs):
        self.calls.append(kv_inputs)
        return {
            "precision_map": {},
            "reuse_decision": {},
            "compressed_kv": {"memory_mb": 0.0},
        }


class TestOrchestratorWiring:
    """orchestrator 接线：kv_inputs 含 attn_is_probabilities = _is_real_backend()。"""

    @staticmethod
    def _tiny_real_backend():
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

    @staticmethod
    def _wire(backend=None):
        orch = Orchestrator(config={})
        if backend is not None:
            orch.model_backend = backend
        orch._model_config = {"num_layers": 2}
        spy = _ShadowKVSpy()
        orch.shadowkv = spy
        return orch, spy

    @staticmethod
    def _run_shadowkv(orch):
        torch.manual_seed(0)
        attention = torch.softmax(torch.randn(1, 2, 8, 8), dim=-1)
        kv = {
            "k": torch.randn(1, 2, 8, 16),
            "v": torch.randn(1, 2, 8, 16),
        }
        inputs = {"attention_scores": attention, "kv_current": kv}
        step = StepConfig(
            step_id=0,
            total_steps=2,
            noise_level=0.5,
            shadowkv_mode="balanced",
        )
        return orch._run_shadowkv(step, inputs, 0.3, 0.1)

    def test_real_backend_wires_true(self):
        """真实后端（非 Mock）路径下 agent 收到 attn_is_probabilities=True。"""
        orch, spy = self._wire(self._tiny_real_backend())
        assert orch._is_real_backend() is True

        self._run_shadowkv(orch)

        assert len(spy.calls) == 2, "应逐层调用 on_step"
        for kv_inputs in spy.calls:
            assert kv_inputs.get("attn_is_probabilities") is True, "真实后端（post-softmax）应接线 True"

    def test_mock_backend_wires_false(self):
        """mock 后端（randn logits）路径下 agent 收到 attn_is_probabilities=False。"""
        orch, spy = self._wire()
        assert orch._is_real_backend() is False

        self._run_shadowkv(orch)

        assert len(spy.calls) == 2, "应逐层调用 on_step"
        for kv_inputs in spy.calls:
            assert kv_inputs.get("attn_is_probabilities") is False, "mock 后端（logits）应接线 False"


# =============================================================================
# AR003 T003 / FR3.2：接收注意力重要性轴 + sink 前缀硬保护 + FP6 公式单源化
#
# TDD Red 阶段：以下目标行为实现尚未存在——rank 归一化、entropy_weight、
# sink_tokens 硬保护、旧 pos_factor 移除、共享核心 compute_irap_importance
# 与 ImportanceModel 委托/删除 _history 均为 design 4.1 / 4.3.1 / 4.3.2 与
# 任务书规定的目标行为，本文件只写测试，不写实现。
# 覆盖 design 6.1 的 UT-4 / UT-5 / UT-10 / UT-19 及 sink/β 合法边界。
# =============================================================================

_PRECISION_TIER = {"int4": 0, "int8": 1, "fp16": 2, "fp32": 3}


def _received_attention(weights):
    """列向接收注意力 [heads, seq_k]：对 batch 与 query 维取 mean（design 4.1）。"""
    return weights.mean(dim=(0, 2))


def _rank_ascending(received):
    """逐 head 在 token 维升序 rank（1-based）/ seq_k ∈ (0, 1]。"""
    order = received.argsort(dim=-1)
    ranks = order.argsort(dim=-1).to(torch.float64)
    return (ranks + 1.0) / received.shape[-1]


def _replica_new_formula(
    weights,
    layer_index=LAYER_ID,
    num_layers=NUM_LAYERS,
    entropy_weight=0.3,
    sink_tokens=4,
):
    """统一公式的测试内复刻（design 4.1）：weights 已完成契约归一化（概率）。

    ``imp = clamp(((1-β)·r + β·norm_ent) · layer_factor, max=1)``，
    随后 ``imp[:sink_tokens] = 1.0``（硬保护，优先于一切）。
    """
    seq_k = weights.shape[-1]
    received = _received_attention(weights)
    rank = _rank_ascending(received)

    clamped = weights.clamp(min=1e-10)
    denom = clamped.sum(dim=(0, 3))
    probs = clamped / denom.unsqueeze(0).unsqueeze(-1)
    entropy = -(probs * torch.log(probs)).sum(dim=(0, 3))  # [heads, seq_q]
    normalized_entropy = (entropy / math.log(seq_k)).transpose(0, 1)  # [seq_q, heads]

    layer_factor = 1.0 + 0.1 * (layer_index / max(num_layers, 1))
    blended = (1.0 - entropy_weight) * rank.transpose(0, 1) + entropy_weight * normalized_entropy
    importance = (blended * layer_factor).clamp(max=1.0).float()

    if sink_tokens > 0:
        importance[:sink_tokens] = 1.0
    return importance


def _average_ranks(values):
    """手写平均秩（scipy 不可用时的 Spearman 兜底）。"""
    order = values.argsort(kind="mergesort")
    ranked = values[order]
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and ranked[j] == ranked[i]:
            j += 1
        avg = 0.5 * (i + j - 1) + 1.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return torch.tensor(ranks, dtype=torch.float64)


def _spearman_corr(a, b):
    """Spearman 相关（优先 scipy.stats.spearmanr，缺失时手写秩相关）。"""
    try:
        from scipy.stats import spearmanr

        return float(spearmanr(a, b).statistic)
    except Exception:
        ra = _average_ranks(a)
        rb = _average_ranks(b)
        ra = ra - ra.mean()
        rb = rb - rb.mean()
        denom = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
        return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def _make_sink_probs(seed=SEED):
    """sink 构造（baseline-profile 2 口径）：全部 query 行集中于 token 0。"""
    torch.manual_seed(seed)
    logits = torch.full((BATCH, HEADS, SEQ, SEQ), -4.0)
    logits[:, :, :, 0] = 4.0
    return logits, torch.softmax(logits, dim=-1)


class TestUT4SinkProtectionReversal:
    """UT-4 sink 反转（srs 3.2 验收 1）：sink token 档位最高，非 sink 更低。

    对照：修复前（行熵语义）同构造下 sink token 为最低档（baseline-profile 2）。
    """

    def test_ut4_sink_token_gets_top_tier_on_step(self, mock_config):
        agent = _make_agent(mock_config, attn_is_probabilities=True)
        _, probs = _make_sink_probs()

        result = agent.on_step(_step_config(), _make_inputs(probs))

        precision_map = result["precision_map"]
        sink_heads = [precision_map[0][h] for h in range(HEADS)]
        assert all(p == "fp32" for p in sink_heads), f"sink token 应全 head 最高档，实际 {sink_heads}"

        non_sink = [precision_map[t][h] for t in range(4, SEQ) for h in range(HEADS)]
        assert all(
            _PRECISION_TIER[p] < _PRECISION_TIER["fp32"] for p in non_sink
        ), "非 sink token 档位应低于 sink（修复前 sink 反而最低——对照组反转）"

    def test_ut4_sink_importance_forced_to_one(self, mock_config):
        agent = _make_agent(mock_config)
        _, probs = _make_sink_probs()

        imp = agent._compute_importance_matrix(
            probs, LAYER_ID, NUM_LAYERS, attn_is_probabilities=True
        )

        assert torch.allclose(imp[:4], torch.ones_like(imp[:4])), "sink 前缀（默认 4）应被硬保护为 1.0"
        assert imp[4:].max().item() < 1.0, "非 sink token 不应被强制 1.0"


class TestUT5ReceivedAttentionAxis:
    """UT-5 轴向正确性（srs 3.2 验收 2）：接收注意力高的 token 档位更高。"""

    @staticmethod
    def _make_axis_probs():
        """异质构造：所有行对 token 10 加权（+4），token 20 减权（-4）。"""
        torch.manual_seed(SEED + 7)
        logits = torch.randn(BATCH, HEADS, SEQ, SEQ)
        logits[:, :, :, 10] = logits[:, :, :, 10] + 4.0
        logits[:, :, :, 20] = logits[:, :, :, 20] - 4.0
        return torch.softmax(logits, dim=-1)

    def test_ut5_high_received_token_outranks_control(self, mock_config):
        agent = _make_agent(mock_config)
        probs = self._make_axis_probs()

        imp = agent._compute_importance_matrix(
            probs, LAYER_ID, NUM_LAYERS, attn_is_probabilities=True
        )
        received = _received_attention(probs)

        assert (received[:, 10] > received[:, 20]).all(), "构造应使 token 10 接收注意力更高"
        assert (imp[10] > imp[20]).all(), "所有 head 上 token 10 重要性应严格高于 token 20"

    def test_ut5_importance_spearman_with_received(self, mock_config):
        agent = _make_agent(mock_config)
        probs = self._make_axis_probs()

        # 关闭 sink 保护以隔离轴向信号（sink 硬保护本身由 UT-4 覆盖）
        imp = agent._compute_importance_matrix(
            probs,
            LAYER_ID,
            NUM_LAYERS,
            attn_is_probabilities=True,
            sink_tokens=0,
        )
        received = _received_attention(probs)

        corr = _spearman_corr(
            imp.flatten().detach().cpu().double().numpy(),
            received.transpose(0, 1).flatten().detach().cpu().double().numpy(),
        )
        assert corr >= 0.9, f"全序列 imp 与列向接收注意力 Spearman {corr:.3f} 应 >= 0.9"


class TestGoldenNewFormula:
    """UT-10 新公式 golden：向量化 vs 逐元素参考（两契约，<= 1e-6）。"""

    @staticmethod
    def _assert_vec_ref_golden(agent, tensor, attn_is_probabilities):
        vec = agent._compute_importance_matrix(
            tensor,
            LAYER_ID,
            NUM_LAYERS,
            attn_is_probabilities=attn_is_probabilities,
            entropy_weight=0.3,
            sink_tokens=4,
        )
        seq = vec.shape[0]
        ref = torch.zeros(seq, HEADS)
        for token_idx in range(seq):
            for head_idx in range(HEADS):
                ref[token_idx, head_idx] = agent.compute_importance_score(
                    tensor,
                    token_idx,
                    head_idx,
                    LAYER_ID,
                    NUM_LAYERS,
                    attn_is_probabilities=attn_is_probabilities,
                    entropy_weight=0.3,
                    sink_tokens=4,
                )

        diff = (vec - ref).abs().max().item()
        assert diff <= 1e-6, f"probs={attn_is_probabilities} 契约新公式 vec/ref 差异 {diff:.3e} 应 <= 1e-6"

    def test_golden_new_formula_probabilities_contract(self, mock_config):
        agent = _make_agent(mock_config)
        _, probs = _make_hetero_probs()
        self._assert_vec_ref_golden(agent, probs, True)

    def test_golden_new_formula_logits_contract(self, mock_config):
        agent = _make_agent(mock_config)
        logits, _ = _make_hetero_probs()
        self._assert_vec_ref_golden(agent, logits, False)

    def test_golden_new_formula_matches_replica(self, mock_config):
        """统一公式复刻对照：probs 契约结果与 design 4.1 公式 <= 1e-6。"""
        agent = _make_agent(mock_config)
        _, probs = _make_hetero_probs()

        imp = agent._compute_importance_matrix(
            probs,
            LAYER_ID,
            NUM_LAYERS,
            attn_is_probabilities=True,
            entropy_weight=0.3,
            sink_tokens=4,
        )
        expected = _replica_new_formula(_correct_rows(probs))

        diff = (imp - expected).abs().max().item()
        assert diff <= 1e-6, f"统一公式复刻差异 {diff:.3e} 应 <= 1e-6"


class TestUT19SingleSource:
    """UT-19 单源化（FP6）：ImportanceModel 委托共享核心，删除 _history。"""

    def test_ut19_importance_model_delegates_to_vec_path(self, mock_config):
        from shadowinfer.shadowkv.importance_model import ImportanceModel

        agent = _make_agent(mock_config)
        logits, _ = _make_hetero_probs()
        model = ImportanceModel(num_layers=NUM_LAYERS, num_heads=HEADS, head_dim=HEAD_DIM)

        assert not hasattr(model, "_history"), "ImportanceModel 不应再持有 _history 容器"

        scored = model.score(logits, layer_id=LAYER_ID)
        vec = agent._compute_importance_matrix(
            logits, LAYER_ID, NUM_LAYERS, attn_is_probabilities=False
        )

        diff = (scored - vec).abs().max().item()
        assert diff <= 1e-6, f"ImportanceModel 与 agent 向量化结果差异 {diff:.3e} 应 <= 1e-6"
        assert not hasattr(model, "_history"), "score 调用后不应累积 _history"

    def test_ut19_importance_model_matches_new_formula(self, mock_config):
        from shadowinfer.shadowkv.importance_model import ImportanceModel

        logits, _ = _make_hetero_probs()
        model = ImportanceModel(num_layers=NUM_LAYERS, num_heads=HEADS, head_dim=HEAD_DIM)

        scored = model.score(logits, layer_id=LAYER_ID)
        expected = _replica_new_formula(torch.softmax(logits, dim=-1))

        diff = (scored - expected).abs().max().item()
        assert diff <= 1e-6, f"ImportanceModel 与新公式差异 {diff:.3e} 应 <= 1e-6"

    def test_ut19_compute_irap_importance_module_core(self, mock_config):
        from shadowinfer.shadowkv.importance_model import compute_irap_importance

        _, probs = _make_hetero_probs()

        imp, new_ema, delta_a = compute_irap_importance(
            probs,
            received_ema_prev=None,
            irap_decay=0.9,
            entropy_weight=0.3,
            sink_tokens=4,
            layer_index=LAYER_ID,
            num_layers=NUM_LAYERS,
        )

        expected = _replica_new_formula(probs)
        diff = (imp - expected).abs().max().item()
        assert diff <= 1e-6, f"compute_irap_importance 与公式复刻差异 {diff:.3e} 应 <= 1e-6"

        received = _received_attention(probs)
        assert new_ema.shape == imp.shape, "new_ema 形状应为 [seq_k, heads]"
        ema_diff = (new_ema - received.transpose(0, 1)).abs().max().item()
        assert ema_diff <= 1e-6, f"冷启动 new_ema 应等于 received，差异 {ema_diff:.3e}"
        assert delta_a is None, "received_ema_prev=None 时 delta_a 应为 None"


class TestSinkBoundary:
    """sink_tokens 合法边界（UT-9 合法侧补充）。"""

    def test_sink_boundary_zero_disables_protection(self, mock_config):
        agent = _make_agent(mock_config)
        _, probs = _make_sink_probs()

        imp = agent._compute_importance_matrix(
            probs,
            LAYER_ID,
            NUM_LAYERS,
            attn_is_probabilities=True,
            sink_tokens=0,
        )

        assert not torch.allclose(imp[0], torch.ones(HEADS)), "sink_tokens=0 时 imp[0] 不应再被强制为 1.0"
        assert imp[0].max().item() < 1.0

    def test_sink_boundary_covers_full_sequence(self, mock_config):
        agent = _make_agent(mock_config, attn_is_probabilities=True, sink_tokens=SEQ)
        _, probs = _make_hetero_probs()

        result = agent.on_step(_step_config(), _make_inputs(probs))

        precisions = {
            prec for token_map in result["precision_map"].values() for prec in token_map.values()
        }
        assert precisions == {"fp32"}, f"sink_tokens >= seq_len 应全保护，实际档位 {precisions}"


class TestEntropyWeightBoundary:
    """entropy_weight 合法边界：1.0 纯熵 / 0.0 纯 rank。"""

    def test_entropy_weight_boundary_pure_entropy(self, mock_config):
        """β=1：imp == 纯熵（去 pos_factor、含 layer_factor、含 sink 保护）<= 1e-6。"""
        agent = _make_agent(mock_config)
        _, probs = _make_hetero_probs()

        imp = agent._compute_importance_matrix(
            probs,
            LAYER_ID,
            NUM_LAYERS,
            attn_is_probabilities=True,
            entropy_weight=1.0,
            sink_tokens=4,
        )
        expected = _replica_new_formula(probs, entropy_weight=1.0, sink_tokens=4)

        diff = (imp - expected).abs().max().item()
        assert diff <= 1e-6, f"纯熵公式差异 {diff:.3e} 应 <= 1e-6"
        assert torch.allclose(imp[:4], torch.ones_like(imp[:4])), "纯熵路径仍须 sink 保护"

    def test_entropy_weight_boundary_pure_rank(self, mock_config):
        """β=0：imp 仅由 rank × layer_factor 决定（无 sink 保护时）。"""
        agent = _make_agent(mock_config)
        _, probs = _make_hetero_probs()

        imp = agent._compute_importance_matrix(
            probs,
            LAYER_ID,
            NUM_LAYERS,
            attn_is_probabilities=True,
            entropy_weight=0.0,
            sink_tokens=0,
        )

        received = _received_attention(probs)
        layer_factor = 1.0 + 0.1 * (LAYER_ID / max(NUM_LAYERS, 1))
        expected = (_rank_ascending(received) * layer_factor).clamp(max=1.0).t().float()

        diff = (imp - expected).abs().max().item()
        assert diff <= 1e-6, f"纯 rank 公式差异 {diff:.3e} 应 <= 1e-6"
