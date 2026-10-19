"""AR009 rate-distortion KV allocation and optional hard eviction tests.

TDD Red contract for ``shadowinfer.shadowkv.rd_allocator`` and its integration
with the agent, ``PrecisionPlan`` and ``PackedKVCache``. All tests use real
torch tensors and real agents; no math is mocked.
"""

import itertools

import torch

from shadowinfer.core.structs import StepConfig
from shadowinfer.shadowkv import ShadowKVAgent
from shadowinfer.shadowkv.packed_kv_cache import PackedKVCache
from shadowinfer.shadowkv.precision_plan import PrecisionPlan
from shadowinfer.shadowkv.rd_allocator import EVICTED, admit_budget, entry_bytes, kv_sensitivity

BITS = {"fp32": 32, "fp16": 16, "int8": 8, "int4": 4, EVICTED: 0}
PRECISION_CODES = {"fp32": 0, "fp16": 1, "int8": 2, "int4": 3, EVICTED: 4}


def _uniform_map(seq_len, num_heads, precision="fp32"):
    return {
        token_idx: {head_idx: precision for head_idx in range(num_heads)}
        for token_idx in range(seq_len)
    }


def _clone_map(precision_map):
    return {token_idx: dict(head_map) for token_idx, head_map in precision_map.items()}


def _predicted_bytes(precision_map, batch, head_dim):
    return int(
        sum(
            entry_bytes(precision, batch, head_dim)
            for head_map in precision_map.values()
            for precision in head_map.values()
        )
    )


def _all_indices(packed):
    return set(
        itertools.chain.from_iterable(
            data.get("indices", []) for data in packed.packed_data.values()
        )
    )


def _agent(config):
    agent = ShadowKVAgent(config=config)
    agent.on_init(
        {
            "num_layers": config.get("num_layers", 1),
            "num_heads": config.get("num_heads", 1),
            "head_dim": config.get("head_dim", 16),
            "model_name": "rd-test",
        }
    )
    return agent


def test_ut9_1_entry_bytes_values():
    assert entry_bytes("fp32", 2, 16) == 2 * 2 * 16 * 4
    assert entry_bytes("fp16", 2, 16) == 2 * 2 * 16 * 2
    assert entry_bytes("int8", 2, 16) == 2 * (2 * 16 + 2 * 4)
    assert entry_bytes("int4", 2, 16) == 2 * (2 * 16 / 2 + 2 * (16 // 16) * 4)
    assert entry_bytes("int4", 2, 48) == 2 * (2 * 48 / 2 + 2 * (48 // 16) * 4)
    assert entry_bytes("int4", 2, 20) == 2 * (2 * 20 / 2 + 2 * (20 // 20) * 4)
    assert entry_bytes(EVICTED, 2, 16) == 0
    assert entry_bytes("evicted", 2, 16) == 0


def test_ut9_2_kv_sensitivity_matches_manual_batch_mean():
    torch.manual_seed(901)
    batch, heads, seq_len, head_dim = 2, 3, 5, 16
    k = torch.randn(batch, heads, seq_len, head_dim)
    v = torch.randn(batch, heads, seq_len, head_dim)

    sensitivity = kv_sensitivity(k, v)
    manual = torch.zeros(seq_len, heads)
    for token_idx in range(seq_len):
        for head_idx in range(heads):
            k_absmax = k[:, head_idx, token_idx, :].abs().max()
            v_absmax = v[:, head_idx, token_idx, :].abs().max()
            manual[token_idx, head_idx] = (k_absmax**2 + v_absmax**2) / 2.0

    assert sensitivity.shape == (seq_len, heads)
    assert torch.allclose(sensitivity, manual)


def test_ut9_3_rd_ordering_downgrades_lowest_sensitivity_first():
    seq_len, heads = 4, 1
    sink_tokens = 1
    importance = torch.ones(seq_len, heads)
    sensitivity = torch.tensor([[9.0], [0.1], [0.5], [1.0]])
    initial = _uniform_map(seq_len, heads, "fp32")
    budget = 512 - (int(entry_bytes("fp32", 1, 16)) - int(entry_bytes("fp16", 1, 16)))

    result, predicted, evicted_count = admit_budget(
        importance,
        sensitivity,
        _clone_map(initial),
        budget,
        batch=1,
        head_dim=16,
        sink_tokens=sink_tokens,
        rd_allocation=True,
        allow_eviction=False,
    )

    assert evicted_count == 0
    assert predicted <= budget
    assert result[0][0] == "fp32"
    assert result[1][0] == "fp16"
    assert result[2][0] == "fp32"
    assert result[3][0] == "fp32"
    for token_idx in range(seq_len):
        for head_idx in range(heads):
            assert BITS[result[token_idx][head_idx]] <= BITS[initial[token_idx][head_idx]]


def test_ut9_4_legacy_importance_ordering_when_rd_disabled():
    seq_len, heads = 4, 1
    importance = torch.tensor([[0.1], [0.2], [0.3], [0.4]])
    sensitivity = torch.ones(seq_len, heads)
    initial = _uniform_map(seq_len, heads, "fp32")
    budget = 512 - (int(entry_bytes("fp32", 1, 16)) - int(entry_bytes("fp16", 1, 16)))

    result, predicted, evicted_count = admit_budget(
        importance,
        sensitivity,
        _clone_map(initial),
        budget,
        batch=1,
        head_dim=16,
        sink_tokens=0,
        rd_allocation=False,
        allow_eviction=False,
    )

    assert evicted_count == 0
    assert predicted <= budget
    assert result[0][0] == "fp16"
    assert result[1][0] == "fp32"
    bits_by_importance = [BITS[result[token_idx][0]] for token_idx in range(seq_len)]
    assert bits_by_importance == sorted(bits_by_importance)


def test_ut9_5_protected_entries_not_downgraded_or_evicted():
    seq_len, heads = 4, 1
    importance = torch.full((seq_len, heads), 0.5)
    sensitivity = torch.ones(seq_len, heads)
    initial = _uniform_map(seq_len, heads, "fp32")
    protected = frozenset({(2, 0)})

    result, predicted, evicted_count = admit_budget(
        importance,
        sensitivity,
        _clone_map(initial),
        budget_bytes=0,
        batch=1,
        head_dim=16,
        sink_tokens=1,
        rd_allocation=True,
        allow_eviction=True,
        protected=protected,
    )

    assert result[0][0] == "fp32"
    assert result[2][0] == "fp32"
    assert result[1][0] == EVICTED
    assert result[3][0] == EVICTED
    assert evicted_count == 2
    assert predicted == int(entry_bytes("fp32", 1, 16)) * 2


def test_ut9_6_hard_eviction_meets_tiny_budget():
    seq_len, heads = 4, 1
    importance = torch.full((seq_len, heads), 0.5)
    sensitivity = torch.ones(seq_len, heads)
    initial = _uniform_map(seq_len, heads, "fp32")

    result, predicted, evicted_count = admit_budget(
        importance,
        sensitivity,
        _clone_map(initial),
        budget_bytes=1,
        batch=1,
        head_dim=16,
        sink_tokens=0,
        rd_allocation=True,
        allow_eviction=True,
    )

    assert evicted_count > 0
    assert predicted <= 1
    assert predicted == 0
    assert all(result[token_idx][0] == EVICTED for token_idx in range(seq_len))


def test_ut9_7_without_hard_eviction_keeps_int4_floor_over_budget():
    seq_len, heads = 3, 1
    importance = torch.full((seq_len, heads), 0.5)
    sensitivity = torch.ones(seq_len, heads)
    initial = _uniform_map(seq_len, heads, "fp32")

    result, predicted, evicted_count = admit_budget(
        importance,
        sensitivity,
        _clone_map(initial),
        budget_bytes=1,
        batch=1,
        head_dim=16,
        sink_tokens=0,
        rd_allocation=True,
        allow_eviction=False,
    )

    assert evicted_count == 0
    assert predicted > 1
    assert predicted == _predicted_bytes(result, 1, 16)
    assert all(result[token_idx][0] == "int4" for token_idx in range(seq_len))


def test_ut9_8_agent_uses_rd_by_default_and_legacy_when_disabled():
    base = {
        "num_layers": 1,
        "num_heads": 1,
        "head_dim": 16,
        "sink_tokens": 0,
        "use_packed_cache": True,
        "attn_is_probabilities": False,
    }
    agent = _agent(dict(base))
    assert agent.rd_allocation_enabled is True
    assert agent.hard_eviction_enabled is False
    agent.cache_manager.memory_budget_bytes = 192

    k = torch.full((1, 1, 2, 16), 1.0)
    v = torch.full((1, 1, 2, 16), 1.0)
    k[0, 0, 0, :] = 10.0
    v[0, 0, 0, :] = 10.0
    kv_current = {"k": k, "v": v}
    importance = torch.tensor([[0.1], [1.0]])
    initial = _uniform_map(2, 1, "fp32")

    rd_result = agent._admit_precision_budget(importance, _clone_map(initial), kv_current)
    assert rd_result[0][0] == "fp32"
    assert rd_result[1][0] == "fp16"

    legacy_agent = _agent(dict(base, rd_allocation_enabled=False))
    legacy_agent.cache_manager.memory_budget_bytes = 192
    legacy_result = legacy_agent._admit_precision_budget(
        importance, _clone_map(initial), kv_current
    )
    assert legacy_result[0][0] == "fp16"
    assert legacy_result[1][0] == "fp32"


def test_ut9_9_precision_plan_supports_evicted():
    precision_map = {
        0: {0: "fp32", 1: EVICTED},
        1: {0: "int8", 1: "int4"},
        2: {0: EVICTED, 1: "fp16"},
    }
    plan = PrecisionPlan.build(precision_map, seq_len=3, num_heads=2)

    assert plan.codes.dtype == torch.int8
    for token_idx in range(3):
        for head_idx in range(2):
            expected = PRECISION_CODES[precision_map[token_idx][head_idx]]
            assert int(plan.codes[token_idx, head_idx].item()) == expected

    expected_evicted = [(0, 1), (2, 0)]
    assert sorted(plan.evicted_entries) == expected_evicted
    assert "evicted" not in plan.groups
    grouped = [entry for entries in plan.groups.values() for entry in entries]
    assert len(grouped) == 4
    assert set(grouped).isdisjoint(expected_evicted)

    changed = _clone_map(precision_map)
    changed[0][1] = "int4"
    changed_plan = PrecisionPlan.build(changed, seq_len=3, num_heads=2)
    assert changed_plan.fingerprint != plan.fingerprint
    assert sorted(changed_plan.evicted_entries) == [(2, 0)]


def test_ut9_10_compress_and_pack_omit_evicted_entries():
    config = {
        "num_layers": 1,
        "num_heads": 2,
        "head_dim": 16,
        "sink_tokens": 0,
        "use_packed_cache": True,
        "attn_is_probabilities": False,
    }
    agent = _agent(config)
    torch.manual_seed(910)
    k = torch.randn(1, 2, 4, 16)
    v = torch.randn(1, 2, 4, 16)
    precision_map = {
        0: {0: "fp32", 1: "int8"},
        1: {0: EVICTED, 1: "int4"},
        2: {0: "int4", 1: EVICTED},
        3: {0: "fp16", 1: "fp32"},
    }
    evicted = [(1, 0), (2, 1)]

    result = agent.compress_kv({"k": k, "v": v}, precision_map)
    for token_idx, head_idx in evicted:
        assert torch.count_nonzero(result["k"][:, head_idx, token_idx, :]) == 0
        assert torch.count_nonzero(result["v"][:, head_idx, token_idx, :]) == 0
    assert torch.count_nonzero(result["k"][:, 0, 0, :]) > 0

    packed = result["packed_kv"]
    packed_indices = _all_indices(packed)
    for entry in evicted:
        assert entry not in packed_indices

    unpacked_k, unpacked_v = packed.unpack()
    for token_idx, head_idx in evicted:
        assert torch.count_nonzero(unpacked_k[:, head_idx, token_idx, :]) == 0
        assert torch.count_nonzero(unpacked_v[:, head_idx, token_idx, :]) == 0

    single_config = dict(config, num_heads=1)
    single_agent = _agent(single_config)
    single_map = {
        0: {0: "fp32"},
        1: {0: EVICTED},
        2: {0: EVICTED},
    }
    single = single_agent.compress_kv({"k": k[:, :1, :3, :], "v": v[:, :1, :3, :]}, single_map)
    assert single["packed_kv"].memory_bytes() == int(entry_bytes("fp32", 1, 16))


def test_ut9_11_evict_does_not_resurrect_evicted_entries():
    torch.manual_seed(911)
    k = torch.randn(1, 1, 3, 16)
    v = torch.randn(1, 1, 3, 16)
    precision_map = {
        0: {0: "fp32"},
        1: {0: "fp32"},
        2: {0: EVICTED},
    }
    packed = PackedKVCache.pack(k, v, precision_map)["packed_kv"]
    assert (2, 0) not in _all_indices(packed)
    assert packed.memory_bytes() == int(entry_bytes("fp32", 1, 16)) * 2

    packed.evict({(0, 0)})

    assert (2, 0) not in _all_indices(packed)
    assert (1, 0) in _all_indices(packed)
    unpacked_k, unpacked_v = packed.unpack()
    assert torch.count_nonzero(unpacked_k[:, 0, 2, :]) == 0
    assert torch.count_nonzero(unpacked_v[:, 0, 2, :]) == 0


def test_ut9_12_stats_count_evicted_as_zero_bits():
    agent = _agent(
        {
            "num_layers": 1,
            "num_heads": 1,
            "head_dim": 16,
            "use_packed_cache": True,
        }
    )
    agent._precision_distribution = {EVICTED: 4}
    assert agent.get_compression_stats()["avg_precision_bits"] == 0.0

    agent._precision_distribution = {EVICTED: 4, "fp32": 4}
    assert agent.get_compression_stats()["avg_precision_bits"] == 16.0


def test_ut9_13_hard_eviction_on_step_and_protected_entries():
    config = {
        "num_layers": 1,
        "num_heads": 2,
        "head_dim": 16,
        "sink_tokens": 2,
        "use_packed_cache": True,
        "attn_is_probabilities": False,
        "memory_budget_mb": 0.0001,
        "hard_eviction_enabled": True,
    }
    agent = _agent(config)
    assert agent.hard_eviction_enabled is True

    torch.manual_seed(913)
    attention = torch.randn(1, 2, 8, 8)
    kv_current = {"k": torch.randn(1, 2, 8, 16), "v": torch.randn(1, 2, 8, 16)}
    result = agent.on_step(
        StepConfig(
            step_id=0,
            total_steps=4,
            noise_level=0.5,
            shadowkv_mode="balanced",
        ),
        {
            "attention_scores": attention,
            "kv_current": kv_current,
            "layer_id": 0,
            "qdrift_signal": {"sensitivity_score": 0.3, "drift_score": 0.1},
            "memory_pressure": 0.0,
            "attn_is_probabilities": False,
        },
    )
    precision_map = result["precision_map"]
    for token_idx in range(2):
        for head_idx in range(2):
            assert precision_map[token_idx][head_idx] == "fp32"
    assert (
        sum(
            1
            for head_map in precision_map.values()
            for precision in head_map.values()
            if precision == EVICTED
        )
        > 0
    )

    protected_agent = _agent(dict(config, sink_tokens=1, memory_budget_mb=None))
    protected_agent.cache_manager.memory_budget_bytes = 1
    importance = torch.rand(4, 2)
    initial = _uniform_map(4, 2, "fp32")
    protected_kv = {
        "k": torch.randn(1, 2, 4, 16),
        "v": torch.randn(1, 2, 4, 16),
    }
    protected = {(2, 1)}
    protected_result = protected_agent._admit_precision_budget(
        importance,
        _clone_map(initial),
        protected_kv,
        protected=protected,
    )
    assert protected_result[0][0] == "fp32"
    assert protected_result[0][1] == "fp32"
    assert protected_result[2][1] == "fp32"
    assert any(
        precision == EVICTED
        for head_map in protected_result.values()
        for precision in head_map.values()
    )
