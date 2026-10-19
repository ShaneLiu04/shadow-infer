"""AR008 fused KV plan tests (UT-8.1 ~ UT-8.8).

These tests are the TDD Red contract for ``PrecisionPlan`` and the fused
compress/pack paths. They use real tensors and real agents only.
"""

import pytest
import torch

from shadowinfer.shadowkv import ShadowKVAgent
from shadowinfer.shadowkv.packed_kv_cache import PackedKVCache
from shadowinfer.shadowkv.precision_plan import PrecisionPlan

PRECISIONS = ("fp32", "fp16", "int8", "int4")
PRECISION_CODES = {"fp32": 0, "fp16": 1, "int8": 2, "int4": 3}


def _mixed_precision_map(seq_len: int, num_heads: int) -> dict:
    return {
        token_idx: {
            head_idx: PRECISIONS[(token_idx + head_idx) % len(PRECISIONS)]
            for head_idx in range(num_heads)
        }
        for token_idx in range(seq_len)
    }


def _uniform_precision_map(seq_len: int, num_heads: int, precision: str) -> dict:
    return {
        token_idx: {head_idx: precision for head_idx in range(num_heads)}
        for token_idx in range(seq_len)
    }


def _assert_scale_map_equal(actual: dict, expected: dict) -> None:
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


def _assert_nested_equal(actual, expected, path: str) -> None:
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


def _packed_dict(cache) -> dict:
    data = getattr(cache, "_data", None)
    if data is None:
        data = getattr(cache, "packed_data", None)
    assert isinstance(data, dict), "packed cache exposes neither _data nor packed_data"
    return data


def _assert_packed_equal(actual, expected) -> None:
    actual_data = _packed_dict(actual)
    expected_data = _packed_dict(expected)
    assert set(actual_data.keys()) == set(expected_data.keys())
    for precision in actual_data:
        _assert_nested_equal(actual_data[precision], expected_data[precision], precision)
    assert actual.memory_bytes() == expected.memory_bytes()


def test_ut8_1_plan_grouping_codes_flat_indices_fingerprint():
    seq_len, num_heads = 5, 3
    precision_map = _mixed_precision_map(seq_len, num_heads)
    reuse_mask = {
        (0, 0): True,
        (1, 2): True,
        (2, 0): True,
        (3, 1): False,
        (4, 1): True,
    }

    plan = PrecisionPlan.build(precision_map, seq_len, num_heads, reuse_mask=reuse_mask)
    assert plan.seq_len == seq_len
    assert plan.num_heads == num_heads
    assert torch.is_tensor(plan.codes)
    assert plan.codes.dtype == torch.int8
    assert plan.codes.shape == (seq_len, num_heads)
    for token_idx in range(seq_len):
        for head_idx in range(num_heads):
            precision = precision_map[token_idx][head_idx]
            assert int(plan.codes[token_idx, head_idx].item()) == PRECISION_CODES[precision]

    expected_groups = {precision: [] for precision in PRECISIONS}
    for token_idx in range(seq_len):
        for head_idx in range(num_heads):
            expected_groups[precision_map[token_idx][head_idx]].append((token_idx, head_idx))

    for precision, expected_entries in expected_groups.items():
        assert plan.groups.get(precision, []) == expected_entries
        expected_flat = [head_idx * seq_len + token_idx for token_idx, head_idx in expected_entries]
        flat = plan.flat_indices.get(precision)
        if expected_flat:
            assert flat is not None
            assert flat.dtype == torch.long
            assert flat.tolist() == expected_flat
        else:
            assert flat is None or flat.numel() == 0

    expected_reuse = sorted(
        (token_idx, head_idx) for (token_idx, head_idx), flag in reuse_mask.items() if flag
    )
    assert sorted(tuple(entry) for entry in plan.reuse_entries) == expected_reuse
    assert plan.reuse_flat.dtype == torch.long
    assert sorted(int(value) for value in plan.reuse_flat.tolist()) == sorted(
        head_idx * seq_len + token_idx for token_idx, head_idx in expected_reuse
    )

    assert isinstance(plan.fingerprint, str)
    assert len(plan.fingerprint) == 64
    int(plan.fingerprint, 16)

    same_plan = PrecisionPlan.build(precision_map, seq_len, num_heads, reuse_mask=reuse_mask)
    assert same_plan.fingerprint == plan.fingerprint

    changed_precision = {t: dict(heads) for t, heads in precision_map.items()}
    changed_precision[0][0] = "int8" if precision_map[0][0] != "int8" else "fp32"
    changed_precision_plan = PrecisionPlan.build(
        changed_precision, seq_len, num_heads, reuse_mask=reuse_mask
    )
    assert changed_precision_plan.fingerprint != plan.fingerprint

    changed_reuse_mask = dict(reuse_mask)
    changed_reuse_mask[(3, 1)] = True
    changed_reuse_plan = PrecisionPlan.build(
        precision_map, seq_len, num_heads, reuse_mask=changed_reuse_mask
    )
    assert changed_reuse_plan.fingerprint != plan.fingerprint

    no_reuse_plan = PrecisionPlan.build(precision_map, seq_len, num_heads, reuse_mask=None)
    assert list(no_reuse_plan.reuse_entries) == []
    assert no_reuse_plan.reuse_flat.numel() == 0


def test_ut8_2_plan_compress_matches_legacy_with_reuse(mock_config):
    agent = ShadowKVAgent(config=mock_config)
    torch.manual_seed(81)
    batch_size, num_heads, seq_len, head_dim = 1, 4, 7, 64
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k_prev = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v_prev = torch.randn(batch_size, num_heads, seq_len, head_dim)
    precision_map = _mixed_precision_map(seq_len, num_heads)
    reuse_mask = {(0, 1): True, (2, 3): True, (5, 0): True}

    plan = PrecisionPlan.build(precision_map, seq_len, num_heads, reuse_mask=reuse_mask)
    legacy = agent._compress_kv_vectorized(k, v, k_prev, v_prev, precision_map, reuse_mask)
    fused = agent._compress_kv_vectorized(
        k, v, k_prev, v_prev, precision_map, reuse_mask, plan=plan
    )

    assert "group_payload" in fused
    assert torch.equal(fused["k"], legacy["k"])
    assert torch.equal(fused["v"], legacy["v"])
    _assert_scale_map_equal(fused["scale_k_map"], legacy["scale_k_map"])
    _assert_scale_map_equal(fused["scale_v_map"], legacy["scale_v_map"])


def test_ut8_3_build_scale_map_false_matches_legacy(mock_config):
    agent = ShadowKVAgent(config=mock_config)
    torch.manual_seed(82)
    batch_size, num_heads, seq_len, head_dim = 1, 4, 6, 64
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    precision_map = _mixed_precision_map(seq_len, num_heads)
    plan = PrecisionPlan.build(precision_map, seq_len, num_heads)

    legacy = agent._compress_kv_vectorized(k, v, None, None, precision_map, None)
    with_scales = agent._compress_kv_vectorized(
        k, v, None, None, precision_map, None, plan=plan, build_scale_map=True
    )
    without_scales = agent._compress_kv_vectorized(
        k, v, None, None, precision_map, None, plan=plan, build_scale_map=False
    )

    assert torch.equal(without_scales["k"], legacy["k"])
    assert torch.equal(without_scales["v"], legacy["v"])
    assert torch.equal(without_scales["k"], with_scales["k"])
    assert torch.equal(without_scales["v"], with_scales["v"])
    for key in ("scale_k_map", "scale_v_map"):
        value = without_scales.get(key)
        assert value is None or not (isinstance(value, dict) and len(value) > 0)


def test_ut8_4_fused_pack_matches_legacy(mock_config):
    agent = ShadowKVAgent(config=mock_config)
    torch.manual_seed(84)
    batch_size, num_heads, seq_len, head_dim = 1, 4, 8, 64
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    precision_map = _mixed_precision_map(seq_len, num_heads)
    plan = PrecisionPlan.build(precision_map, seq_len, num_heads)

    precomputed = agent._compress_kv_vectorized(
        k, v, None, None, precision_map, None, plan=plan, build_scale_map=False
    )["group_payload"]

    legacy = PackedKVCache.pack(k, v, precision_map, group_size=16)
    fused = PackedKVCache.pack(
        k, v, precision_map, group_size=16, plan=plan, precomputed=precomputed
    )
    _assert_packed_equal(fused["packed_kv"], legacy["packed_kv"])
    assert fused["memory_bytes"] == legacy["memory_bytes"]

    fallback = PackedKVCache.pack(k, v, precision_map, group_size=16, plan=plan, precomputed=None)
    _assert_packed_equal(fallback["packed_kv"], legacy["packed_kv"])
    assert fallback["memory_bytes"] == legacy["memory_bytes"]


def test_ut8_5_batch2_odd_head_dim_int4_and_reuse(mock_config):
    agent = ShadowKVAgent(config=mock_config)
    torch.manual_seed(85)
    batch_size, num_heads, seq_len, head_dim = 2, 3, 6, 48
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k_prev = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v_prev = torch.randn(batch_size, num_heads, seq_len, head_dim)
    precision_map = _mixed_precision_map(seq_len, num_heads)
    reuse_mask = {(0, 0): True, (2, 1): True, (4, 2): True, (5, 0): False}
    assert any(
        precision_map[token_idx][head_idx] == "int4"
        for token_idx in range(seq_len)
        for head_idx in range(num_heads)
    )

    plan = PrecisionPlan.build(precision_map, seq_len, num_heads, reuse_mask=reuse_mask)
    legacy = agent._compress_kv_vectorized(k, v, k_prev, v_prev, precision_map, reuse_mask)
    fused = agent._compress_kv_vectorized(
        k, v, k_prev, v_prev, precision_map, reuse_mask, plan=plan
    )

    assert torch.equal(fused["k"], legacy["k"])
    assert torch.equal(fused["v"], legacy["v"])
    _assert_scale_map_equal(fused["scale_k_map"], legacy["scale_k_map"])
    _assert_scale_map_equal(fused["scale_v_map"], legacy["scale_v_map"])
    for token_idx, head_idx in ((0, 0), (2, 1), (4, 2)):
        assert torch.equal(fused["k"][:, head_idx, token_idx, :], k_prev[:, head_idx, token_idx, :])
        assert torch.equal(fused["v"][:, head_idx, token_idx, :], v_prev[:, head_idx, token_idx, :])
        assert torch.equal(
            fused["k"][:, head_idx, token_idx, :], legacy["k"][:, head_idx, token_idx, :]
        )
        assert torch.equal(
            fused["v"][:, head_idx, token_idx, :], legacy["v"][:, head_idx, token_idx, :]
        )

    pack_plan = PrecisionPlan.build(precision_map, seq_len, num_heads)
    precomputed = agent._compress_kv_vectorized(
        k, v, None, None, precision_map, None, plan=pack_plan, build_scale_map=False
    )["group_payload"]
    legacy_packed = PackedKVCache.pack(k, v, precision_map, group_size=16)
    fused_packed = PackedKVCache.pack(
        k, v, precision_map, group_size=16, plan=pack_plan, precomputed=precomputed
    )
    _assert_packed_equal(fused_packed["packed_kv"], legacy_packed["packed_kv"])
    assert fused_packed["memory_bytes"] == legacy_packed["memory_bytes"]


def test_ut8_6_strict_unsupported_precision_raises():
    seq_len, num_heads = 4, 2
    precision_map = _mixed_precision_map(seq_len, num_heads)
    precision_map[1][1] = "bf16"
    with pytest.raises(ValueError):
        PrecisionPlan.build(precision_map, seq_len, num_heads, strict=True)


def test_ut8_7_compress_kv_no_scale_map_and_packed_cache(mock_config):
    agent = ShadowKVAgent(config=mock_config)
    assert agent.use_packed_cache is True
    torch.manual_seed(87)
    batch_size, num_heads, seq_len, head_dim = 1, 4, 8, 64
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    precision_map = _mixed_precision_map(seq_len, num_heads)
    kv_current = {"k": k, "v": v}

    legacy = agent.compress_kv(kv_current, precision_map)
    fused = agent.compress_kv(kv_current, precision_map, build_scale_map=False)

    assert fused.get("packed_kv") is not None
    assert torch.equal(fused["k"], legacy["k"])
    assert torch.equal(fused["v"], legacy["v"])
    for key in ("scale_k_map", "scale_v_map"):
        value = fused.get(key)
        assert value is None or not (isinstance(value, dict) and len(value) > 0)

    direct_packed = PackedKVCache.pack(k, v, precision_map)["packed_kv"]
    _assert_packed_equal(fused["packed_kv"], legacy["packed_kv"])
    _assert_packed_equal(fused["packed_kv"], direct_packed)


def test_ut8_8_plan_shape_mismatch_raises(mock_config):
    agent = ShadowKVAgent(config=mock_config)
    torch.manual_seed(88)
    batch_size, num_heads, seq_len, head_dim = 1, 4, 6, 64
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    precision_map = _mixed_precision_map(seq_len, num_heads)

    longer_map = _mixed_precision_map(seq_len + 1, num_heads)
    bad_seq_plan = PrecisionPlan.build(longer_map, seq_len + 1, num_heads)
    with pytest.raises(ValueError):
        agent._compress_kv_vectorized(k, v, None, None, precision_map, None, plan=bad_seq_plan)

    bad_heads_plan = PrecisionPlan.build(precision_map, seq_len, num_heads + 1)
    with pytest.raises(ValueError):
        PackedKVCache.pack(k, v, precision_map, plan=bad_heads_plan)
