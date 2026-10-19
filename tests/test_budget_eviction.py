"""预算准入与驱逐接线测试（AR003 T005 / FR3.4，KV-P4/P8/P10）。

TDD Red 阶段：本文件只写测试，不写实现。被测目标行为（design 4.1 / 4.3.3）：

- ``eviction_pressure_threshold`` 配置键（默认 0.9，(0,1] 校验，on_init ValueError）；
- pre-pack 降档准入：``layer_budget = memory_budget_bytes // num_layers``，
  ``predicted_bytes = Σ bits(π[t,h])·head_dim·2·batch/8``，非 sink 条目按
  importance 升序逐级降档（fp32→fp16→int8→int4），sink 永不降档；
- 压缩后 packed 实际字节（含 scale）≤ layer_budget；
- ``KVCacheManager._enforce_budget`` 的 ``estimated_entry_bytes`` 改为
  per-precision bits 口径（修复 batch*head_dim 的 8x 低估）；
- ``memory_pressure`` 超阈时的跨层全局驱逐（``enforce_global_budget``）；
- 无预算 + 超阈 → 零动作且 warning 恰好一次；未超阈 → 零动作；
- ``get_importance_matrix`` 未存储层返回 None；
- 降档在 compress 之前完成；正常路径每层每步 ``PackedKVCache.pack`` 仅一次。

覆盖 design 6.1 归属 FR3.4 的用例：
UT-11 / UT-12 / UT-13 / UT-14 / UT-15 / UT-15b + pack-once 回归锚点。
"""

import pytest
import torch

from shadowinfer.core.structs import KVCacheEntry, StepConfig
from shadowinfer.shadowkv import ShadowKVAgent
from shadowinfer.shadowkv.importance_model import TokenHeadScore
from shadowinfer.shadowkv.kv_cache_manager import KVCacheManager
from shadowinfer.shadowkv.packed_kv_cache import PackedKVCache

SEED = 20261018
BATCH = 1
SEQ = 64
HEADS = 4
HEAD_DIM = 64
NUM_LAYERS = 2
SINK = 2

BITS = {"fp32": 32, "fp16": 16, "int8": 8, "int4": 4}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _step_config(step_id=0, total_steps=8, mode="balanced"):
    return StepConfig(
        step_id=step_id,
        total_steps=total_steps,
        noise_level=0.5,
        shadowkv_mode=mode,
    )


def _make_kv(seed=SEED + 2):
    torch.manual_seed(seed)
    return {
        "k": torch.randn(BATCH, HEADS, SEQ, HEAD_DIM),
        "v": torch.randn(BATCH, HEADS, SEQ, HEAD_DIM),
    }


def _make_inputs(layer_id=0, memory_pressure=0.0, seed=SEED + 3):
    torch.manual_seed(seed + layer_id)
    attention = torch.randn(BATCH, HEADS, SEQ, SEQ)
    return {
        "attention_scores": attention,
        "kv_current": _make_kv(seed + layer_id),
        "layer_id": layer_id,
        "qdrift_signal": {"sensitivity_score": 0.3, "drift_score": 0.1},
        "memory_pressure": memory_pressure,
        "attn_is_probabilities": False,
    }


def _fixed_importance(layer_id=0, seed=SEED + 11):
    """确定性重要性矩阵：sink 前缀 = 1.0，其余为散布在 [0,1) 的值。"""
    torch.manual_seed(seed + layer_id)
    imp = torch.rand(SEQ, HEADS)
    imp[:SINK, :] = 1.0
    return imp


def _patch_importance(agent, monkeypatch, matrices):
    """固定 on_step 的重要性矩阵（隔离 IRAP 计算，聚焦预算接线）。

    AR003 review 修复：内部 API 已改为 `_compute_importance_with_state`
    （返回 (importance, new_ema, delta_a)），此处返回固定矩阵且不推进 EMA。
    """

    def fake(attention_scores, layer_index, num_layers, **kwargs):
        return matrices[layer_index].clone(), None, None

    monkeypatch.setattr(agent, "_compute_importance_with_state", fake)


def _make_agent(mock_config, monkeypatch, budget_mb=None, matrices=None, **overrides):
    config = dict(mock_config)
    config.update(
        {
            "num_layers": NUM_LAYERS,
            "num_heads": HEADS,
            "head_dim": HEAD_DIM,
            "use_packed_cache": True,
            "attn_is_probabilities": False,
            "sink_tokens": SINK,
        }
    )
    if budget_mb is not None:
        config["memory_budget_mb"] = budget_mb
    config.update(overrides)
    agent = ShadowKVAgent(config=config)
    agent.on_init(
        {"num_layers": NUM_LAYERS, "num_heads": HEADS, "head_dim": HEAD_DIM, "model_name": "t"}
    )
    if matrices is None:
        matrices = {layer: _fixed_importance(layer) for layer in range(NUM_LAYERS)}
    _patch_importance(agent, monkeypatch, matrices)
    return agent


def _predicted_bytes(precision_map, head_dim=HEAD_DIM, batch=BATCH):
    total = 0
    for token_map in precision_map.values():
        for precision in token_map.values():
            total += BITS[precision] * head_dim * 2 * batch // 8
    return total


def _layer_budget(budget_mb, num_layers=NUM_LAYERS):
    return int(budget_mb * 1024 * 1024) // num_layers


def _capture_compress_precision_map(agent, monkeypatch):
    """在 compress_kv 入口记录 precision_map（验证降档先于压缩）。"""
    captured = {}
    original = agent.compress_kv

    def spy(kv_current, precision_map, *args, **kwargs):
        captured["precision_map"] = {
            token: dict(head_map) for token, head_map in precision_map.items()
        }
        return original(kv_current, precision_map, *args, **kwargs)

    monkeypatch.setattr(agent, "compress_kv", spy)
    return captured


def _count_pack_calls(monkeypatch):
    calls = []
    original = PackedKVCache.pack

    def counting_pack(k_tensor, v_tensor, precision_map, group_size=16, **kwargs):
        calls.append((tuple(k_tensor.shape), tuple(sorted(precision_map))))
        return original(k_tensor, v_tensor, precision_map, group_size=group_size, **kwargs)

    monkeypatch.setattr(PackedKVCache, "pack", staticmethod(counting_pack))
    return calls


def _all_precision_map(precision, seq_len, num_heads):
    return {t: {h: precision for h in range(num_heads)} for t in range(seq_len)}


def _assert_bits_monotonic(precision_map, importance):
    entries = sorted(
        (float(importance[t, h]), t, h) for t in range(SINK, SEQ) for h in range(HEADS)
    )
    bits = [BITS[precision_map[t][h]] for _, t, h in entries]
    assert bits == sorted(bits), "importance 升序降档后，档位应随 importance 单调不减"


# ---------------------------------------------------------------------------
# UT-11 准入阶梯
# ---------------------------------------------------------------------------


class TestUT11PrePackAdmission:
    def test_ut11_downgrades_before_pack_and_fits_layer_budget(self, mock_config, monkeypatch):
        budget_mb = 0.08
        agent = _make_agent(mock_config, monkeypatch, budget_mb=budget_mb)
        importance = _fixed_importance(0)
        _patch_importance(agent, monkeypatch, {0: importance, 1: _fixed_importance(1)})
        captured = _capture_compress_precision_map(agent, monkeypatch)

        agent.on_step(_step_config(), _make_inputs(layer_id=0))

        precision_map = captured["precision_map"]
        layer_budget = _layer_budget(budget_mb)
        initial = agent.allocate_precision(importance, "balanced")
        assert _predicted_bytes(initial) > layer_budget, "构造应触发准入降档"

        # 3) 压缩前 precision_map 已满足层预算（pre-pack）
        assert _predicted_bytes(precision_map) <= layer_budget
        # 2) 压缩后实际 packed 字节（含 scale）≤ 层预算
        packed = agent.cache_manager.retrieve_packed(0)
        assert packed is not None
        assert packed.memory_bytes() <= layer_budget

        # 2) sink 永不降档（保持最高档 fp32）
        for t in range(SINK):
            for h in range(HEADS):
                assert precision_map[t][h] == "fp32"
                assert precision_map[t][h] == initial[t][h]

        # 只降不升，且确有降档发生
        downgraded = []
        for t in range(SEQ):
            for h in range(HEADS):
                assert BITS[precision_map[t][h]] <= BITS[initial[t][h]]
                if BITS[precision_map[t][h]] < BITS[initial[t][h]]:
                    downgraded.append((t, h))
        assert downgraded, "小预算下应有非 sink 条目被降档"

        # 2) 非 sink 按 importance 升序逐级降档 → 档位随 importance 单调不减
        _assert_bits_monotonic(precision_map, importance)

        # 非 sink 高重要性档位 ≥ 低重要性档位（显式抽样断言）
        low_t, low_h = min(
            ((t, h) for t in range(SINK, SEQ) for h in range(HEADS)),
            key=lambda th: float(importance[th[0], th[1]]),
        )
        high_t, high_h = max(
            ((t, h) for t in range(SINK, SEQ) for h in range(HEADS)),
            key=lambda th: float(importance[th[0], th[1]]),
        )
        assert BITS[precision_map[high_t][high_h]] >= BITS[precision_map[low_t][low_h]]


# ---------------------------------------------------------------------------
# UT-12 字节估算修正
# ---------------------------------------------------------------------------


class TestUT12ByteEstimate:
    def test_ut12_fp32_only_estimate_matches_packed_bytes(self, monkeypatch):
        """fp32-only 层：_enforce_budget 的估算 ≈ 实测 packed 字节（2%）。

        修复前估算为 ``batch * head_dim``（64 B/条），实测 fp32 为 512 B/条，8x 低估。
        """
        seq_len, num_heads, head_dim = 8, 2, 64
        k = torch.ones(BATCH, num_heads, seq_len, head_dim)
        v = torch.ones(BATCH, num_heads, seq_len, head_dim)
        precision_map = _all_precision_map("fp32", seq_len, num_heads)
        packed = PackedKVCache.pack(k, v, precision_map)["packed_kv"]

        manager = KVCacheManager(num_layers=1)
        manager.store(
            0,
            KVCacheEntry(
                k_tensor=k,
                v_tensor=v,
                precision="fp32",
                packed_kv=packed,
            ),
        )
        meta = {}
        for t in range(seq_len):
            for h in range(num_heads):
                meta[(t, h)] = TokenHeadScore(layer_id=0, token_idx=t, head_idx=h, importance=0.5)
        manager.set_token_head_meta(0, meta)
        manager.memory_budget_bytes = packed.memory_bytes() - 1

        captured = {}

        def spy(entries, budget_bytes, current_usage_bytes, estimated_entry_bytes):
            captured["estimated"] = estimated_entry_bytes
            return set()

        monkeypatch.setattr(manager.eviction_policy, "select_for_eviction", spy)
        manager._enforce_budget(0)

        assert "estimated" in captured, "超预算时应调用驱逐策略计算估算字节"
        actual_per_entry = packed.memory_bytes() / (seq_len * num_heads)
        assert captured["estimated"] == pytest.approx(
            actual_per_entry, rel=0.02
        ), "per-precision 估算应接近实测 packed 字节（修复 8x 低估）"

    def test_ut12_mixed_precision_stored_bytes_within_layer_budget(self, mock_config, monkeypatch):
        """混合精度小预算下，准入后 packed 实际字节 ≤ 层预算（含 scale 口径）。"""
        budget_mb = 0.06  # 层预算 31457B > 底线（sink fp32 + 全 int4 = 27904B）
        agent = _make_agent(mock_config, monkeypatch, budget_mb=budget_mb)
        captured = _capture_compress_precision_map(agent, monkeypatch)

        agent.on_step(_step_config(), _make_inputs(layer_id=0))

        precision_map = captured["precision_map"]
        precisions = {p for head_map in precision_map.values() for p in head_map.values()}
        assert len(precisions) >= 2, "构造应产生混合精度档位"
        packed = agent.cache_manager.retrieve_packed(0)
        assert packed.memory_bytes() <= _layer_budget(budget_mb)


# ---------------------------------------------------------------------------
# UT-13 触底告警
# ---------------------------------------------------------------------------


class TestUT13BottomTierWarning:
    def test_ut13_all_int4_still_over_budget_warns_and_keeps_bottom_tier(
        self, mock_config, monkeypatch
    ):
        budget_mb = 0.004
        agent = _make_agent(mock_config, monkeypatch, budget_mb=budget_mb)
        alerts = []
        monkeypatch.setattr(
            agent.logger, "log_alert", lambda level, message, **kw: alerts.append((level, message))
        )
        captured = _capture_compress_precision_map(agent, monkeypatch)

        result = agent.on_step(_step_config(), _make_inputs(layer_id=0))
        precision_map = captured["precision_map"]
        layer_budget = _layer_budget(budget_mb)

        # 全部非 sink 触底 int4，sink 保持 fp32
        for t in range(SINK, SEQ):
            for h in range(HEADS):
                assert precision_map[t][h] == "int4"
        for t in range(SINK):
            for h in range(HEADS):
                assert precision_map[t][h] == "fp32"

        # 触底仍超预算 → warning；保留底线档，不静默丢数据
        assert [a for a in alerts if a[0] == "warning"], "触底仍超预算应告警"
        packed = agent.cache_manager.retrieve_packed(0)
        assert packed.memory_bytes() > layer_budget
        assert result["precision_map"] == precision_map


# ---------------------------------------------------------------------------
# UT-14 跨层全局驱逐
# ---------------------------------------------------------------------------


class TestUT14GlobalEviction:
    def test_ut14_enforce_global_budget_evicts_lowest_importance_across_layers(self):
        seq_len, num_heads, head_dim = 16, 2, 64
        sink_tokens = 2
        k = torch.ones(BATCH, num_heads, seq_len, head_dim)
        v = torch.ones(BATCH, num_heads, seq_len, head_dim)
        precision_map = _all_precision_map("fp32", seq_len, num_heads)

        manager = KVCacheManager(num_layers=2)
        importances = {}
        for layer_id in range(2):
            packed = PackedKVCache.pack(k, v, precision_map)["packed_kv"]
            manager.store(
                layer_id,
                KVCacheEntry(
                    k_tensor=k,
                    v_tensor=v,
                    precision="fp32",
                    packed_kv=packed,
                ),
            )
            torch.manual_seed(SEED + 30 + layer_id)
            imp = torch.rand(seq_len, num_heads)
            imp[:sink_tokens, :] = 1.0
            importances[layer_id] = imp
            manager.set_importance_matrix(layer_id, imp)

        assert manager.get_importance_matrix(99) is None
        assert manager.get_importance_matrix(2) is None

        budget_bytes = 20000
        evicted_count = manager.enforce_global_budget(budget_bytes, sink_tokens)
        assert evicted_count > 0, "超阈应触发真实跨层驱逐（修复前为空操作）"

        usage = manager.get_memory_usage()
        lower_bound = 0
        for layer_id in range(2):
            packed = manager.retrieve_packed(layer_id)
            sink_entries = sink_tokens * num_heads
            int4_entries = (seq_len - sink_tokens) * num_heads
            lower_bound += sink_entries * head_dim * 2 * 4
            lower_bound += int4_entries * (head_dim * 2 * 4 // 8 + 8)
        assert (
            usage <= int(budget_bytes * 0.9) or usage <= lower_bound
        ), "驱逐后用量应 ≤ 预算·0.9（或非 sink 全触底）"

        for layer_id in range(2):
            packed = manager.retrieve_packed(layer_id)
            fp32_indices = set(packed.packed_data.get("fp32", {}).get("indices", []))
            int4_indices = set(packed.packed_data.get("int4", {}).get("indices", []))
            imp = importances[layer_id]

            # sink 存活：仍为 fp32 且未被清零
            for t in range(sink_tokens):
                for h in range(num_heads):
                    assert (t, h) in fp32_indices, "sink token-head 不可被驱逐"
                    assert (t, h) not in int4_indices
            k_unpacked, _ = packed.unpack()
            for t in range(sink_tokens):
                assert torch.all(k_unpacked[:, :, t, :].abs() > 0.5), "sink 数据不应被清零"

            # 最低重要性先驱逐：被驱逐条目重要性 ≤ 所有保留 fp32 条目
            if fp32_indices:
                max_retained_min = min(float(imp[t, h]) for t, h in fp32_indices)
                for t, h in int4_indices:
                    assert float(imp[t, h]) <= max_retained_min

    def test_ut14_agent_consumes_memory_pressure_and_stores_importance_matrix(
        self, mock_config, monkeypatch
    ):
        agent = _make_agent(mock_config, monkeypatch, budget_mb=0.02)
        calls = []

        def spy_enforce(budget_bytes, sink_tokens):
            calls.append((budget_bytes, sink_tokens))
            return 0

        monkeypatch.setattr(agent.cache_manager, "enforce_global_budget", spy_enforce)

        agent.on_step(_step_config(), _make_inputs(layer_id=0, memory_pressure=0.95))

        assert calls == [
            (agent.cache_manager.memory_budget_bytes, SINK)
        ], "pressure ≥ 阈值且配置预算时应调用跨层全局驱逐"
        matrix = agent.cache_manager.get_importance_matrix(0)
        assert matrix is not None, "store 后应写入张量级重要性矩阵"
        assert tuple(matrix.shape) == (SEQ, HEADS)


# ---------------------------------------------------------------------------
# UT-15 未超阈稳定 / 阈值校验
# ---------------------------------------------------------------------------


class TestUT15BelowThreshold:
    def test_ut15_below_threshold_makes_no_budget_action(self, mock_config, monkeypatch):
        agent = _make_agent(mock_config, monkeypatch, budget_mb=0.5)
        importance = _fixed_importance(0)
        _patch_importance(agent, monkeypatch, {0: importance, 1: _fixed_importance(1)})
        captured = _capture_compress_precision_map(agent, monkeypatch)
        calls = []
        monkeypatch.setattr(
            agent.cache_manager,
            "enforce_global_budget",
            lambda budget_bytes, sink_tokens: calls.append((budget_bytes, sink_tokens)) or 0,
        )

        result = agent.on_step(_step_config(), _make_inputs(layer_id=0, memory_pressure=0.5))

        assert calls == [], "pressure < 阈值时不应触发全局驱逐"
        expected = agent.allocate_precision(importance, "balanced")
        assert captured["precision_map"] == expected, "大预算下不应发生准入降档"
        assert result["precision_map"] == expected

    def test_ut15_pressure_threshold_validation(self, mock_config):
        base = dict(mock_config)
        base.update(
            {
                "num_layers": NUM_LAYERS,
                "num_heads": HEADS,
                "head_dim": HEAD_DIM,
                "use_packed_cache": True,
            }
        )
        for bad in (0.0, -0.1, 1.01, 2.0):
            cfg = dict(base)
            cfg["eviction_pressure_threshold"] = bad
            with pytest.raises(ValueError):
                agent = ShadowKVAgent(config=cfg)
                agent.on_init(
                    {
                        "num_layers": NUM_LAYERS,
                        "num_heads": HEADS,
                        "head_dim": HEAD_DIM,
                    }
                )


# ---------------------------------------------------------------------------
# UT-15b 无预算默认路径
# ---------------------------------------------------------------------------


class TestUT15bNoBudgetDefault:
    def test_ut15b_no_budget_pressure_warns_once_and_makes_no_action(
        self, mock_config, monkeypatch
    ):
        agent = _make_agent(mock_config, monkeypatch)
        assert agent.cache_manager.memory_budget_bytes is None

        alerts = []
        monkeypatch.setattr(
            agent.logger, "log_alert", lambda level, message, **kw: alerts.append((level, message))
        )
        calls = []
        monkeypatch.setattr(
            agent.cache_manager,
            "enforce_global_budget",
            lambda budget_bytes, sink_tokens: calls.append((budget_bytes, sink_tokens)) or 0,
        )

        importance = _fixed_importance(0)
        expected = agent.allocate_precision(importance, "balanced")
        for step_id in range(3):
            result = agent.on_step(
                _step_config(step_id=step_id),
                _make_inputs(layer_id=0, memory_pressure=0.95, seed=SEED + 50 + step_id),
            )
            assert result["precision_map"] == expected, "无预算时不得降档"
            assert result["precision_map"] is not None

        warnings = [a for a in alerts if a[0] == "warning"]
        assert len(warnings) == 1, f"无预算默认路径应告警恰好一次，实际 {len(warnings)}"
        assert calls == [], "无预算时不得触发全局驱逐"


# ---------------------------------------------------------------------------
# pack-once 回归锚点
# ---------------------------------------------------------------------------


class TestPackOnce:
    def test_pack_once_per_layer_per_step(self, mock_config, monkeypatch):
        agent = _make_agent(mock_config, monkeypatch, budget_mb=0.5)
        calls = _count_pack_calls(monkeypatch)

        for layer_id in range(NUM_LAYERS):
            agent.on_step(
                _step_config(step_id=0),
                _make_inputs(layer_id=layer_id, memory_pressure=0.1, seed=SEED + 70 + layer_id),
            )

        assert len(calls) == NUM_LAYERS, (
            f"正常路径每层每步应 pack 一次（预期 {NUM_LAYERS}，实际 {len(calls)}）；" "准入降档不得引入额外重打包"
        )
