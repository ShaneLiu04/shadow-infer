"""ShadowKV/FFN M5 决策记忆化与记账修复测试（AR002 T006）。

覆盖 design 6.1（test_decision_cache / test_on_init_reset）与 design §7.2
（FFN 反量化结果缓存）及 srs FR3.3 的 memory_mb 真实节省口径。
"""

import time

import pytest
import torch

from shadowinfer.core.structs import StepConfig
from shadowinfer.ffn_optimizer.packed_weight import PackedFFNWeight
from shadowinfer.shadowkv import ShadowKVAgent
from shadowinfer.shadowkv.eviction_policy import ImportanceAgeEvictionPolicy


def _step_config(step_id=0, total_steps=6, mode="balanced"):
    return StepConfig(
        step_id=step_id,
        total_steps=total_steps,
        noise_level=0.5,
        shadowkv_mode=mode,
    )


def _make_inputs(batch=1, heads=8, seq=64, head_dim=16, seed=0, peaked=False):
    """构造 on_step 输入；peaked=True 时 attention 近 one-hot（低熵）。"""
    torch.manual_seed(seed)
    logits = torch.randn(batch, heads, seq, seq)
    if peaked:
        logits[..., 0] = 50.0
    attention = torch.softmax(logits, dim=-1)
    kv = {
        "k": torch.randn(batch, heads, seq, head_dim),
        "v": torch.randn(batch, heads, seq, head_dim),
    }
    return {
        "attention_scores": attention,
        "kv_current": kv,
        "layer_id": 0,
        "qdrift_signal": {"sensitivity_score": 0.3, "drift_score": 0.1},
    }


class TestDecisionCache:
    """M5 决策缓存（design 4.2.2 点 5 / 6.1 test_decision_cache）。"""

    def _counting_agent(self, mock_config, **overrides):
        config = dict(mock_config)
        config["use_packed_cache"] = False
        config.update(overrides)
        agent = ShadowKVAgent(config=config)
        agent.on_init({"num_layers": 4, "num_heads": 8, "head_dim": 16, "model_name": "t"})
        calls = {"count": 0}
        original = agent._compute_importance_matrix

        def _counting(*args, **kwargs):
            calls["count"] += 1
            return original(*args, **kwargs)

        agent._compute_importance_matrix = _counting
        return agent, calls

    def test_same_input_second_call_hits_cache(self, mock_config):
        """同 (step_id, layer_id) 同输入二次调用：结果一致且不重算决策。"""
        agent, calls = self._counting_agent(mock_config)
        inputs = _make_inputs(seed=1)
        step = _step_config(step_id=0)

        first = agent.on_step(step, inputs)
        assert calls["count"] == 1
        second = agent.on_step(step, inputs)
        assert calls["count"] == 1, "同输入二次调用应命中决策缓存"

        assert second["precision_map"] == first["precision_map"]
        assert second["importance_stats"] == pytest.approx(first["importance_stats"])
        assert second["reuse_decision"]["reuse_stats"] == first["reuse_decision"]["reuse_stats"]

    def test_decision_compute_latency_drop_90pct(self, mock_config):
        """srs FR3.3 验收：同输入第二次调用的决策计算耗时下降 >= 90%。

        计时口径为决策面三方法（重要性/精度分配/复用决策）的独占耗时
        之和——即缓存覆盖面；二次调用命中缓存时该值为精确 0，
        断言对机器负载不敏感（端到端 on_step 计时受压缩/统计等
        未缓存固定开销影响，不适合作为 90% 断言的分母）。
        """
        agent, _ = self._counting_agent(mock_config)
        timers = {"total": 0.0}
        for name in (
            "_compute_importance_matrix",
            "allocate_precision",
            "_build_reuse_decision",
        ):
            original = getattr(agent, name)

            def _timed(*args, _orig=original, **kwargs):
                start = time.perf_counter()
                result = _orig(*args, **kwargs)
                timers["total"] += time.perf_counter() - start
                return result

            setattr(agent, name, _timed)

        inputs = _make_inputs(batch=4, heads=8, seq=256, head_dim=8, seed=2)
        step = _step_config(step_id=0)

        timers["total"] = 0.0
        first = agent.on_step(step, inputs)
        first_decision_time = timers["total"]
        assert first_decision_time > 0.0

        timers["total"] = 0.0
        second = agent.on_step(step, inputs)
        second_decision_time = timers["total"]

        assert second["precision_map"] == first["precision_map"]
        assert second_decision_time * 10 <= first_decision_time, (
            f"决策耗时 {second_decision_time:.6f}s 应较首次 " f"{first_decision_time:.6f}s 下降 >= 90%"
        )

    def test_different_input_same_key_no_dirty_hit(self, mock_config):
        """同键不同输入（新张量对象）不命中脏缓存而重算。"""
        agent, calls = self._counting_agent(mock_config)
        step = _step_config(step_id=0)

        inputs_a = _make_inputs(seed=3)
        result_a = agent.on_step(step, inputs_a)
        assert calls["count"] == 1

        inputs_b = _make_inputs(seed=4, peaked=True)
        result_b = agent.on_step(step, inputs_b)
        assert calls["count"] == 2, "同键不同输入应重算（不脏命中）"
        assert result_b["importance_stats"]["mean"] != pytest.approx(
            result_a["importance_stats"]["mean"]
        )

    def test_step_rollover_invalidates_cache(self, mock_config):
        """step_id 滚动后缓存失效，需重算。"""
        agent, calls = self._counting_agent(mock_config)
        inputs = _make_inputs(seed=5)

        agent.on_step(_step_config(step_id=0), inputs)
        agent.on_step(_step_config(step_id=0), inputs)
        assert calls["count"] == 1

        agent.on_step(_step_config(step_id=1), inputs)
        assert calls["count"] == 2, "step 滚动后应失效重算"

        agent.on_step(_step_config(step_id=0), inputs)
        assert calls["count"] == 3, "回到旧 step 亦应重算（缓存已被滚动清空）"

    def test_fingerprint_failure_bypasses_cache(self, mock_config, monkeypatch):
        """指纹计算异常（返回 None）时旁路缓存：不命中、不写入、结果正确。"""
        agent, calls = self._counting_agent(mock_config)
        monkeypatch.setattr(agent, "_decision_input_fingerprint", lambda *args, **kwargs: None)
        inputs = _make_inputs(seed=6)
        step = _step_config(step_id=0)

        first = agent.on_step(step, inputs)
        second = agent.on_step(step, inputs)
        assert calls["count"] == 2, "指纹异常应旁路缓存逐次重算"
        assert second["precision_map"] == first["precision_map"]
        assert agent._decision_cache == {}

    def test_fingerprint_method_swallows_exceptions(self, mock_config):
        """_decision_input_fingerprint 内部异常被吞掉并返回 None。"""
        agent = ShadowKVAgent(config=mock_config)
        assert (
            agent._decision_input_fingerprint("not-a-tensor", {}, None, 6, 0.3, 0.1, "balanced")
            is None
        )


class TestOnInitReset:
    """M5 on_init 全量重置（design 6.1 test_on_init_reset）。"""

    def test_reinit_resets_all_accumulated_state(self, mock_config):
        agent = ShadowKVAgent(config=dict(mock_config))
        model_config = {
            "num_layers": 4,
            "num_heads": 8,
            "head_dim": 16,
            "model_name": "t",
        }
        agent.on_init(model_config)
        agent.on_step(_step_config(step_id=0), _make_inputs(seed=7))

        assert agent._compression_stats["step_count"] > 0
        assert sum(agent._precision_distribution.values()) > 0
        assert sum(agent._reuse_stats.values()) > 0
        assert len(agent._decision_cache) > 0

        agent.on_init(model_config)
        assert agent._compression_stats == {
            "total_original_bytes": 0,
            "total_compressed_bytes": 0,
            "step_count": 0,
        }
        assert agent._precision_distribution == {"fp32": 0, "fp16": 0, "int8": 0, "int4": 0}
        assert agent._reuse_stats == {
            "full_reuse_count": 0,
            "partial_reuse_count": 0,
            "no_reuse_count": 0,
        }
        assert agent._decision_cache == {}
        assert agent._decision_cache_step is None

    def test_reinit_preserves_cache_manager_config(self, mock_config):
        """on_init 重建 cache_manager 不再丢失 memory_budget / eviction 配置。"""
        config = dict(mock_config)
        config["memory_budget_mb"] = 256
        config["eviction_policy"] = "age"
        agent = ShadowKVAgent(config=config)
        agent.on_init({"num_layers": 4, "num_heads": 8, "head_dim": 16, "model_name": "t"})

        assert agent.cache_manager.memory_budget_bytes == 256 * 1024 * 1024
        assert isinstance(agent.cache_manager.eviction_policy, ImportanceAgeEvictionPolicy)


class TestFFNDequantCache:
    """design §7.2：PackedFFNWeight 反量化结果缓存。"""

    def test_dequantize_for_matmul_cache_hit(self):
        """同精度计划二次调用不触发逐通道 unpack，结果逐元素一致。"""
        torch.manual_seed(8)
        weight = torch.randn(64, 128)
        precision_map = {
            c: ("fp16" if c % 3 == 0 else "int8" if c % 3 == 1 else "int4") for c in range(64)
        }
        packed = PackedFFNWeight.pack(weight, precision_map)

        calls = {"count": 0}
        original_unpack = packed.unpack

        def _counting_unpack():
            calls["count"] += 1
            return original_unpack()

        packed.unpack = _counting_unpack

        first = packed.dequantize_for_matmul()
        assert calls["count"] == 1
        second = packed.dequantize_for_matmul()
        assert calls["count"] == 1, "缓存命中不应触发逐通道 unpack"
        assert torch.equal(first, second)
        assert second is first

    def test_repack_invalidates_cache(self):
        """pack 重建精度计划后缓存失效重算，且数值反映新计划。"""
        torch.manual_seed(9)
        weight = torch.randn(64, 128)
        mixed_map = {c: ("int8" if c % 2 == 0 else "fp16") for c in range(64)}
        packed = PackedFFNWeight.pack(weight, mixed_map)
        first = packed.dequantize_for_matmul()

        weight2 = torch.randn(64, 128)
        fp32_map = {c: "fp32" for c in range(64)}
        packed._do_pack(weight2, fp32_map)
        assert packed._dequant_cache is None

        calls = {"count": 0}
        original_unpack = packed.unpack

        def _counting_unpack():
            calls["count"] += 1
            return original_unpack()

        packed.unpack = _counting_unpack
        second = packed.dequantize_for_matmul()
        assert calls["count"] == 1, "重建精度计划后应重算"
        assert torch.allclose(second, weight2, atol=1e-6), "fp32 计划应精确还原 weight2"
        assert not torch.allclose(second, first)


class TestMemoryMbSavingsSemantics:
    """srs FR3.3：memory_mb 口径 = 真实节省字节数（未压缩 - 压缩后）。"""

    def test_memory_mb_is_real_savings_packed(self, mock_config):
        config = dict(mock_config)
        config["use_packed_cache"] = True
        agent = ShadowKVAgent(config=config)
        torch.manual_seed(10)
        batch, heads, seq, head_dim = 1, 8, 32, 16
        kv = {
            "k": torch.randn(batch, heads, seq, head_dim),
            "v": torch.randn(batch, heads, seq, head_dim),
        }
        precision_map = {t: {h: "fp16" for h in range(heads)} for t in range(seq)}

        result = agent.compress_kv(kv, precision_map)

        uncompressed = (
            kv["k"].numel() * kv["k"].element_size() + kv["v"].numel() * kv["v"].element_size()
        )
        assert result["uncompressed_bytes"] == uncompressed
        assert result["compressed_bytes"] < uncompressed
        expected_mb = (uncompressed - result["compressed_bytes"]) / (1024.0 * 1024.0)
        assert result["memory_mb"] == pytest.approx(expected_mb)
        # 全 fp16 计划：节省应为原始字节的一半
        assert result["compressed_bytes"] == pytest.approx(uncompressed / 2, rel=1e-6)
        assert result["memory_mb"] == pytest.approx(uncompressed / 2 / (1024.0 * 1024.0))

    def test_memory_mb_zero_savings_for_fp32(self, mock_config):
        """全 fp32 计划：压缩后尺寸等于未压缩，节省恰为 0（诚实口径）。"""
        config = dict(mock_config)
        config["use_packed_cache"] = True
        agent = ShadowKVAgent(config=config)
        torch.manual_seed(11)
        kv = {
            "k": torch.randn(1, 4, 16, 16),
            "v": torch.randn(1, 4, 16, 16),
        }
        precision_map = {t: {h: "fp32" for h in range(4)} for t in range(16)}

        result = agent.compress_kv(kv, precision_map)
        assert result["compressed_bytes"] == result["uncompressed_bytes"]
        assert result["memory_mb"] == 0.0

    def test_memory_mb_is_real_savings_estimate_path(self, mock_config):
        config = dict(mock_config)
        config["use_packed_cache"] = False
        agent = ShadowKVAgent(config=config)
        torch.manual_seed(12)
        kv = {
            "k": torch.randn(1, 4, 16, 8),
            "v": torch.randn(1, 4, 16, 8),
        }
        precision_map = {t: {h: "fp16" for h in range(4)} for t in range(16)}

        result = agent.compress_kv(kv, precision_map)
        uncompressed = result["uncompressed_bytes"]
        expected = (uncompressed - result["compressed_bytes"]) / (1024.0 * 1024.0)
        assert result["memory_mb"] == pytest.approx(expected)
        assert result["compressed_bytes"] == pytest.approx(uncompressed / 2, rel=1e-6)
