"""AR010 dense precision-code path tests (UT-10.1 ~ UT-10.9).

TDD Red contract for ``map_to_codes`` / ``codes_to_map`` /
``PrecisionPlan.from_codes`` / ``admit_codes`` and the agent dense path.
All tests use real torch tensors and real agents; no math is mocked.
"""

import random

import pytest
import torch

from shadowinfer.core.structs import StepConfig
from shadowinfer.shadowkv import ShadowKVAgent
from shadowinfer.shadowkv.precision_plan import (
    EVICTED,
    PRECISION_CODES,
    PrecisionPlan,
    codes_to_map,
    map_to_codes,
)
from shadowinfer.shadowkv.rd_allocator import admit_budget, admit_codes

PRECISIONS = ("fp32", "fp16", "int8", "int4")


def _mixed_map(seq_len, num_heads, seed=0):
    rng = random.Random(seed)
    return {
        token_idx: {head_idx: rng.choice(PRECISIONS) for head_idx in range(num_heads)}
        for token_idx in range(seq_len)
    }


def _uniform_map(seq_len, num_heads, precision="fp32"):
    return {
        token_idx: {head_idx: precision for head_idx in range(num_heads)}
        for token_idx in range(seq_len)
    }


def _clone_map(precision_map):
    return {token_idx: dict(head_map) for token_idx, head_map in precision_map.items()}


def _normalized_map(precision_map, seq_len, num_heads):
    normalized = {}
    for token_idx in range(seq_len):
        token_map = precision_map.get(token_idx, {})
        normalized[token_idx] = {}
        for head_idx in range(num_heads):
            raw = token_map.get(head_idx, "fp16")
            normalized[token_idx][head_idx] = raw.lower() if isinstance(raw, str) else raw
    return normalized


def _packed_dict(cache):
    data = getattr(cache, "_data", None)
    if data is None:
        data = getattr(cache, "packed_data", None)
    assert isinstance(data, dict)
    return data


def _packed_indices(cache):
    return {
        precision: set(payload.get("indices", []))
        for precision, payload in _packed_dict(cache).items()
    }


def _assert_nested_equal(actual, expected, path):
    if torch.is_tensor(actual):
        assert torch.is_tensor(expected), path
        assert actual.dtype == expected.dtype, path
        assert actual.shape == expected.shape, path
        assert torch.equal(actual, expected), path
    elif isinstance(actual, dict):
        assert isinstance(expected, dict), path
        assert set(actual.keys()) == set(expected.keys()), path
        for key in actual:
            _assert_nested_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, (list, tuple)):
        assert isinstance(expected, (list, tuple)), path
        assert len(actual) == len(expected), path
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_nested_equal(actual_item, expected_item, f"{path}[{index}]")
    elif actual is None:
        assert expected is None, path
    else:
        assert actual == expected, path


def _assert_packed_equal(actual, expected):
    actual_data = _packed_dict(actual)
    expected_data = _packed_dict(expected)
    assert set(actual_data.keys()) == set(expected_data.keys())
    for precision in actual_data:
        _assert_nested_equal(actual_data[precision], expected_data[precision], precision)
    assert actual.memory_bytes() == expected.memory_bytes()


def _assert_scale_map_equal(actual, expected):
    assert set(actual.keys()) == set(expected.keys())
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if expected_value is None:
            assert actual_value is None
        else:
            assert actual_value is not None
            assert actual_value.shape == expected_value.shape
            assert actual_value.dtype == expected_value.dtype
            assert torch.equal(actual_value, expected_value)


def _make_agent(mock_config, **overrides):
    config = dict(mock_config)
    config.update(
        {
            "num_layers": 1,
            "num_heads": 4,
            "head_dim": 16,
            "sink_tokens": 0,
            "use_packed_cache": True,
            "attn_is_probabilities": False,
        }
    )
    config.update(overrides)
    agent = ShadowKVAgent(config=config)
    agent.on_init(
        {
            "num_layers": config["num_layers"],
            "num_heads": config["num_heads"],
            "head_dim": config["head_dim"],
            "model_name": "dense-precision-test",
        }
    )
    return agent


def _step_config(step_id=0, total_steps=4, mode="balanced"):
    return StepConfig(
        step_id=step_id,
        total_steps=total_steps,
        noise_level=0.5,
        shadowkv_mode=mode,
    )


def _make_inputs(seq_len=8, num_heads=4, head_dim=16, seed=1010):
    torch.manual_seed(seed)
    attention = torch.randn(1, num_heads, seq_len, seq_len)
    kv_current = {
        "k": torch.randn(1, num_heads, seq_len, head_dim),
        "v": torch.randn(1, num_heads, seq_len, head_dim),
    }
    return {
        "attention_scores": attention,
        "kv_current": kv_current,
        "layer_id": 0,
        "qdrift_signal": {"sensitivity_score": 0.3, "drift_score": 0.1},
        "memory_pressure": 0.0,
        "attn_is_probabilities": False,
    }


def _clone_inputs(inputs):
    return {
        "attention_scores": inputs["attention_scores"].clone(),
        "kv_current": {
            "k": inputs["kv_current"]["k"].clone(),
            "v": inputs["kv_current"]["v"].clone(),
        },
        "layer_id": inputs["layer_id"],
        "qdrift_signal": dict(inputs["qdrift_signal"]),
        "memory_pressure": inputs["memory_pressure"],
        "attn_is_probabilities": inputs["attn_is_probabilities"],
    }


def test_ut10_1_map_codes_map_roundtrip_missing_and_evicted():
    seq_len, num_heads = 5, 3
    precision_map = {
        0: {0: "fp32", 2: EVICTED},
        1: {1: "int8"},
        2: {0: "int4", 1: "fp16", 2: "fp32"},
        4: {2: EVICTED},
    }

    codes = map_to_codes(precision_map, seq_len, num_heads)
    assert torch.is_tensor(codes)
    assert codes.dtype == torch.int8
    assert codes.shape == (seq_len, num_heads)

    expected = _normalized_map(precision_map, seq_len, num_heads)
    for token_idx in range(seq_len):
        for head_idx in range(num_heads):
            assert (
                int(codes[token_idx, head_idx].item())
                == PRECISION_CODES[expected[token_idx][head_idx]]
            )
    assert int(codes[0, 1].item()) == PRECISION_CODES["fp16"]
    assert int(codes[3, 0].item()) == PRECISION_CODES["fp16"]
    assert int(codes[0, 2].item()) == PRECISION_CODES[EVICTED]
    assert int(codes[4, 2].item()) == PRECISION_CODES[EVICTED]

    roundtrip = codes_to_map(codes)
    assert roundtrip == expected

    unknown = {0: {0: "bf16"}}
    with pytest.raises(ValueError):
        map_to_codes(unknown, 1, 1, strict=True)

    lenient = map_to_codes(unknown, 1, 1, strict=False)
    assert int(lenient[0, 0].item()) == PRECISION_CODES["fp16"]


@pytest.mark.parametrize("seq_len,num_heads", [(64, 4), (512, 8), (2048, 8)])
def test_ut10_2_from_codes_matches_build(seq_len, num_heads):
    precision_map = _mixed_map(seq_len, num_heads, seed=seq_len)
    for token_idx in range(0, seq_len, 17):
        precision_map[token_idx][token_idx % num_heads] = EVICTED

    rng = random.Random(seq_len + 1)
    reuse_mask = {
        (token_idx, head_idx): rng.random() < 0.25
        for token_idx in range(seq_len)
        for head_idx in range(num_heads)
    }

    built = PrecisionPlan.build(precision_map, seq_len, num_heads, reuse_mask=reuse_mask)
    codes = map_to_codes(precision_map, seq_len, num_heads)
    from_codes = PrecisionPlan.from_codes(codes, reuse_mask=reuse_mask)

    assert from_codes.seq_len == built.seq_len
    assert from_codes.num_heads == built.num_heads
    assert from_codes.codes.dtype == torch.int8
    assert torch.equal(from_codes.codes, built.codes)
    assert from_codes.groups == built.groups
    assert set(from_codes.flat_indices.keys()) == set(built.flat_indices.keys())
    for precision in built.flat_indices:
        assert torch.equal(from_codes.flat_indices[precision], built.flat_indices[precision])
    assert from_codes.reuse_entries == built.reuse_entries
    assert torch.equal(from_codes.reuse_flat, built.reuse_flat)
    assert from_codes.evicted_entries == built.evicted_entries
    assert from_codes.fingerprint == built.fingerprint

    for precision, entries in from_codes.groups.items():
        assert entries == sorted(entries, key=lambda item: (item[0], item[1]))
        expected_flat = [head_idx * seq_len + token_idx for token_idx, head_idx in entries]
        assert from_codes.flat_indices[precision].tolist() == expected_flat
    assert EVICTED not in from_codes.groups


def test_ut10_3_admit_codes_matches_admit_budget():
    seq_len, num_heads = 6, 2
    batch, head_dim, sink_tokens = 2, 16, 1
    torch.manual_seed(1030)
    importance = torch.rand(seq_len, num_heads)
    sensitivity = torch.rand(seq_len, num_heads) * 5.0

    codes = map_to_codes(_uniform_map(seq_len, num_heads, "fp32"), seq_len, num_heads)
    precision_map = codes_to_map(codes)

    protected_mask = torch.zeros(seq_len, num_heads, dtype=torch.bool)
    protected_mask[3, 0] = True
    protected_mask[4, 1] = True
    protected = {
        (token_idx, head_idx)
        for token_idx in range(seq_len)
        for head_idx in range(num_heads)
        if bool(protected_mask[token_idx, head_idx])
    }

    scenarios = [
        ("large", 1_000_000, False),
        ("tight", 1600, False),
        ("floor", 1, False),
        ("evict", 1, True),
    ]

    for rd_allocation in (True, False):
        for name, budget, allow_eviction in scenarios:
            expected_map, expected_predicted, expected_evicted = admit_budget(
                importance,
                sensitivity,
                _clone_map(precision_map),
                budget,
                batch=batch,
                head_dim=head_dim,
                sink_tokens=sink_tokens,
                rd_allocation=rd_allocation,
                allow_eviction=allow_eviction,
                protected=protected,
            )
            actual_codes, predicted, evicted = admit_codes(
                codes,
                importance,
                sensitivity,
                budget,
                batch=batch,
                head_dim=head_dim,
                sink_tokens=sink_tokens,
                rd_allocation=rd_allocation,
                allow_eviction=allow_eviction,
                protected_mask=protected_mask,
            )

            path = f"{name}/rd={rd_allocation}"
            assert torch.is_tensor(actual_codes), path
            assert actual_codes.dtype == torch.int8, path
            assert actual_codes.shape == codes.shape, path
            assert codes_to_map(actual_codes) == expected_map, path
            assert predicted == expected_predicted, path
            assert evicted == expected_evicted, path

            for token_idx in range(seq_len):
                if token_idx < sink_tokens:
                    for head_idx in range(num_heads):
                        assert (
                            int(actual_codes[token_idx, head_idx].item()) == PRECISION_CODES["fp32"]
                        ), path
            for token_idx, head_idx in protected:
                assert (
                    int(actual_codes[token_idx, head_idx].item()) == PRECISION_CODES["fp32"]
                ), path


def test_ut10_3b_admit_codes_tie_order_matches_admit_budget():
    seq_len, num_heads = 8, 3
    importance = torch.full((seq_len, num_heads), 0.5)
    sensitivity = torch.ones(seq_len, num_heads)
    codes = map_to_codes(_uniform_map(seq_len, num_heads, "fp32"), seq_len, num_heads)
    precision_map = codes_to_map(codes)

    expected_map, expected_predicted, expected_evicted = admit_budget(
        importance,
        sensitivity,
        _clone_map(precision_map),
        budget_bytes=1,
        batch=1,
        head_dim=16,
        allow_eviction=True,
    )
    actual_codes, predicted, evicted = admit_codes(
        codes,
        importance,
        sensitivity,
        budget_bytes=1,
        batch=1,
        head_dim=16,
        allow_eviction=True,
    )

    assert codes_to_map(actual_codes) == expected_map
    assert (predicted, evicted) == (expected_predicted, expected_evicted)


def test_ut10_3c_code_shape_guards():
    with pytest.raises(ValueError):
        PrecisionPlan.from_codes(torch.zeros(3, dtype=torch.int8))
    with pytest.raises(ValueError):
        admit_codes(
            torch.zeros(4, dtype=torch.int8),
            torch.ones(1, 1),
            torch.ones(1, 1),
            budget_bytes=1,
            batch=1,
            head_dim=16,
        )


def test_ut10_4_admit_codes_does_not_mutate_input():
    seq_len, num_heads = 5, 2
    codes = map_to_codes(_uniform_map(seq_len, num_heads, "fp32"), seq_len, num_heads)
    before = codes.clone()
    torch.manual_seed(1040)
    importance = torch.rand(seq_len, num_heads)
    sensitivity = torch.rand(seq_len, num_heads)

    out, _, evicted = admit_codes(
        codes,
        importance,
        sensitivity,
        budget_bytes=1,
        batch=1,
        head_dim=16,
        sink_tokens=0,
        rd_allocation=True,
        allow_eviction=True,
    )

    assert torch.equal(codes, before)
    assert evicted > 0
    assert not torch.equal(out, before)


def test_ut10_5_non_contiguous_kv_plan_path_matches_legacy(mock_config):
    agent = _make_agent(mock_config)
    torch.manual_seed(1050)
    batch_size, num_heads, seq_len, head_dim = 1, 4, 6, 16
    base_k = torch.randn(batch_size, num_heads, seq_len, head_dim * 2)
    base_v = torch.randn(batch_size, num_heads, seq_len, head_dim * 2)
    k = base_k[..., ::2]
    v = base_v[..., ::2]
    assert not k.is_contiguous()
    assert not v.is_contiguous()

    precision_map = _mixed_map(seq_len, num_heads, seed=1051)
    plan = PrecisionPlan.build(precision_map, seq_len, num_heads)

    legacy = agent._compress_kv_vectorized(k, v, None, None, precision_map, None)
    fused = agent._compress_kv_vectorized(k, v, None, None, precision_map, None, plan=plan)

    assert torch.equal(fused["k"], legacy["k"])
    assert torch.equal(fused["v"], legacy["v"])
    _assert_scale_map_equal(fused["scale_k_map"], legacy["scale_k_map"])
    _assert_scale_map_equal(fused["scale_v_map"], legacy["scale_v_map"])

    result = agent.compress_kv({"k": k, "v": v}, precision_map)
    assert torch.count_nonzero(result["k"]) > 0
    assert torch.count_nonzero(result["v"]) > 0
    assert torch.equal(result["k"], legacy["k"])
    assert torch.equal(result["v"], legacy["v"])

    fallback_calls = {"count": 0}
    original_reference = agent._compress_kv_reference

    def _counting_reference(*args, **kwargs):
        fallback_calls["count"] += 1
        return original_reference(*args, **kwargs)

    agent._compress_kv_reference = _counting_reference
    try:
        agent.compress_kv({"k": k, "v": v}, precision_map)
    finally:
        agent._compress_kv_reference = original_reference
    assert fallback_calls["count"] == 0


def test_ut10_6_compress_kv_with_codes_matches_dict(mock_config):
    agent = _make_agent(mock_config)
    torch.manual_seed(1060)
    batch_size, num_heads, seq_len, head_dim = 1, 4, 7, 16
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    precision_map = _mixed_map(seq_len, num_heads, seed=1061)
    precision_map[2][1] = EVICTED
    codes = map_to_codes(precision_map, seq_len, num_heads)

    dict_result = agent.compress_kv({"k": k, "v": v}, precision_map)
    code_result = agent.compress_kv({"k": k, "v": v}, precision_map, precision_codes=codes)

    assert torch.equal(code_result["k"], dict_result["k"])
    assert torch.equal(code_result["v"], dict_result["v"])
    _assert_scale_map_equal(code_result["scale_k_map"], dict_result["scale_k_map"])
    _assert_scale_map_equal(code_result["scale_v_map"], dict_result["scale_v_map"])
    _assert_packed_equal(code_result["packed_kv"], dict_result["packed_kv"])
    assert code_result["memory_bytes"] == dict_result["memory_bytes"]
    assert code_result["compressed_bytes"] == dict_result["compressed_bytes"]


def test_ut10_6b_agent_admit_codes_matches_admit_budget(mock_config):
    agent = _make_agent(
        mock_config,
        num_heads=2,
        head_dim=16,
        sink_tokens=1,
        rd_allocation_enabled=True,
        hard_eviction_enabled=True,
    )
    torch.manual_seed(1062)
    seq_len, num_heads = 6, 2
    importance = torch.rand(seq_len, num_heads)
    kv_current = {
        "k": torch.randn(1, num_heads, seq_len, 16),
        "v": torch.randn(1, num_heads, seq_len, 16),
    }
    codes = map_to_codes(_uniform_map(seq_len, num_heads, "fp32"), seq_len, num_heads)
    protected = {(2, 0)}

    for budget, allow_eviction in ((100000, False), (1, True)):
        agent.cache_manager.memory_budget_bytes = budget
        agent.hard_eviction_enabled = allow_eviction
        expected_map = agent._admit_precision_budget(
            importance,
            codes_to_map(codes),
            kv_current,
            protected=protected,
        )
        actual_codes = agent._admit_precision_codes(
            codes,
            importance,
            kv_current,
            protected=protected,
        )
        assert codes_to_map(actual_codes) == expected_map
        assert int(actual_codes[0, 0].item()) == PRECISION_CODES["fp32"]
        assert int(actual_codes[2, 0].item()) == PRECISION_CODES["fp32"]


@pytest.mark.parametrize("mode", ["aggressive", "balanced", "conservative"])
def test_ut10_7_allocate_precision_codes_matches_allocate_precision(mock_config, mode):
    agent = _make_agent(mock_config)
    assert getattr(agent, "dense_precision_enabled", None) is True

    adjustment = {"aggressive": -0.1, "balanced": 0.0, "conservative": 0.1}[mode]
    thresholds = [max(0.0, 0.8 + adjustment), max(0.0, 0.5 + adjustment)]
    thresholds.append(max(0.0, 0.2 + adjustment))

    torch.manual_seed(1070)
    seq_len, num_heads = 9, 4
    importance = torch.rand(seq_len, num_heads)
    importance[0, 0] = thresholds[0]
    importance[1, 1] = thresholds[1]
    importance[2, 2] = thresholds[2]
    importance[3, 0] = thresholds[0] - 1e-6
    importance[4, 1] = thresholds[1] - 1e-6
    importance[5, 2] = thresholds[2] - 1e-6

    expected = agent.allocate_precision(importance, mode)
    codes = agent.allocate_precision_codes(importance, mode)

    assert torch.is_tensor(codes)
    assert codes.dtype == torch.int8
    assert codes.shape == (seq_len, num_heads)
    assert codes_to_map(codes) == expected


@pytest.mark.parametrize("dense_enabled", [True, False])
def test_ut10_8_on_step_dense_and_dict_paths_match(mock_config, dense_enabled):
    config = dict(mock_config)
    config.update(
        {
            "num_layers": 1,
            "num_heads": 4,
            "head_dim": 16,
            "sink_tokens": 2,
            "use_packed_cache": True,
            "attn_is_probabilities": False,
            "rd_allocation_enabled": True,
            "hard_eviction_enabled": False,
            "dense_precision_enabled": dense_enabled,
        }
    )
    agent = _make_agent(config)
    assert agent.dense_precision_enabled is dense_enabled
    agent.cache_manager.memory_budget_bytes = 2097

    inputs = _make_inputs(seed=1080)
    result = agent.on_step(_step_config(), inputs)

    reference = _make_agent(dict(config, dense_precision_enabled=not dense_enabled))
    reference.cache_manager.memory_budget_bytes = 2097
    reference_result = reference.on_step(_step_config(), _clone_inputs(inputs))

    assert result["precision_map"] == reference_result["precision_map"]
    assert torch.equal(result["compressed_kv"]["k"], reference_result["compressed_kv"]["k"])
    assert torch.equal(result["compressed_kv"]["v"], reference_result["compressed_kv"]["v"])
    _assert_packed_equal(
        result["compressed_kv"]["packed_kv"],
        reference_result["compressed_kv"]["packed_kv"],
    )


@pytest.mark.parametrize(
    "signal",
    [
        {"sensitivity_score": 0.1, "drift_score": 0.8},
        {"sensitivity_score": 0.9, "drift_score": 0.1},
    ],
)
def test_ut10_8b_dense_dict_parity_drift_and_forced_mode(mock_config, signal):
    config = dict(mock_config)
    config.update(
        {
            "num_layers": 1,
            "num_heads": 4,
            "head_dim": 16,
            "sink_tokens": 1,
            "use_packed_cache": True,
            "attn_is_probabilities": False,
            "rd_allocation_enabled": True,
            "hard_eviction_enabled": False,
        }
    )

    dense_agent = _make_agent(dict(config, dense_precision_enabled=True))
    dict_agent = _make_agent(dict(config, dense_precision_enabled=False))

    inputs = _make_inputs(seed=1085)
    inputs["qdrift_signal"] = dict(signal)
    dense_result = dense_agent.on_step(_step_config(mode="balanced"), inputs)

    reference_inputs = _clone_inputs(inputs)
    reference_inputs["qdrift_signal"] = dict(signal)
    dict_result = dict_agent.on_step(_step_config(mode="balanced"), reference_inputs)

    assert dense_result["precision_map"] == dict_result["precision_map"]
    assert torch.equal(dense_result["compressed_kv"]["k"], dict_result["compressed_kv"]["k"])
    assert torch.equal(dense_result["compressed_kv"]["v"], dict_result["compressed_kv"]["v"])
    _assert_packed_equal(
        dense_result["compressed_kv"]["packed_kv"],
        dict_result["compressed_kv"]["packed_kv"],
    )


def test_ut10_9_decision_cache_stores_precision_codes(mock_config):
    config = dict(mock_config)
    config.update(
        {
            "num_layers": 1,
            "num_heads": 4,
            "head_dim": 16,
            "sink_tokens": 1,
            "use_packed_cache": True,
            "attn_is_probabilities": False,
            "dense_precision_enabled": True,
        }
    )
    agent = _make_agent(config)
    agent.cache_manager.memory_budget_bytes = 2097

    inputs = _make_inputs(seed=1090)
    first = agent.on_step(_step_config(step_id=3), inputs)
    cache_key = (3, 0)
    assert cache_key in agent._decision_cache
    entry = agent._decision_cache[cache_key]
    assert entry.get("precision_codes") is not None
    assert torch.is_tensor(entry["precision_codes"])
    assert entry["precision_codes"].dtype == torch.int8
    expected = map_to_codes(
        first["precision_map"],
        first["compressed_kv"]["k"].shape[2],
        first["compressed_kv"]["k"].shape[1],
    )
    assert torch.equal(entry["precision_codes"], expected)

    second = agent.on_step(_step_config(step_id=3), inputs)
    assert second["precision_map"] == first["precision_map"]
    assert torch.equal(second["compressed_kv"]["k"], first["compressed_kv"]["k"])
    assert torch.equal(second["compressed_kv"]["v"], first["compressed_kv"]["v"])
    assert agent._decision_cache[cache_key]["precision_codes"] is not None


def test_ut10_9b_cache_hit_reuses_codes_without_recompute(mock_config):
    config = dict(mock_config)
    config.update(
        {
            "num_layers": 1,
            "num_heads": 4,
            "head_dim": 16,
            "sink_tokens": 1,
            "use_packed_cache": True,
            "attn_is_probabilities": False,
            "dense_precision_enabled": True,
        }
    )
    agent = _make_agent(config)
    agent.cache_manager.memory_budget_bytes = 2097

    calls = {"importance": 0}
    original = agent._compute_importance_with_state

    def _counting_importance(*args, **kwargs):
        calls["importance"] += 1
        return original(*args, **kwargs)

    agent._compute_importance_with_state = _counting_importance
    inputs = _make_inputs(seed=1095)
    first = agent.on_step(_step_config(step_id=4), inputs)
    assert calls["importance"] == 1

    first_codes = agent._decision_cache[(4, 0)]["precision_codes"].clone()
    second = agent.on_step(_step_config(step_id=4), inputs)
    assert calls["importance"] == 1
    assert torch.equal(agent._decision_cache[(4, 0)]["precision_codes"], first_codes)
    assert second["precision_map"] == first["precision_map"]
    assert torch.equal(second["compressed_kv"]["k"], first["compressed_kv"]["k"])
    assert torch.equal(second["compressed_kv"]["v"], first["compressed_kv"]["v"])
