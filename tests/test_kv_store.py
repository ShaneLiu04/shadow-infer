"""AR012 canonical KV store acceptance tests (red-first).

Covers the vectorized ``PackedKVCache`` read paths (unpack / reuse / evict /
resident accounting), the single-source packed storage contract of
``ShadowKVAgent`` and the unified INT4 group size.

All tensors are real and all quantization goes through ``Quantizer``; no
math is mocked.
"""

import time
from statistics import median

import pytest
import torch

from shadowinfer.core.structs import KVCacheEntry, StepConfig
from shadowinfer.shadowkv import KVCacheManager, ShadowKVAgent
from shadowinfer.shadowkv.importance_model import TokenHeadScore
from shadowinfer.shadowkv.packed_kv_cache import PackedKVCache
from shadowinfer.shadowkv.rd_allocator import EVICTED, entry_bytes, predict_codes_bytes
from shadowinfer.utils.quantization import Quantizer

PRECISIONS = ("fp32", "fp16", "int8", "int4")


def _uniform_map(seq_len, num_heads, precision):
    return {t: {h: precision for h in range(num_heads)} for t in range(seq_len)}


def _mixed_map(seq_len, num_heads, span=1):
    """Cycle precisions per token-head; ``span`` groups consecutive entries."""
    precision_map = {}
    for token_idx in range(seq_len):
        precision_map[token_idx] = {}
        for head_idx in range(num_heads):
            flat = token_idx * num_heads + head_idx
            precision_map[token_idx][head_idx] = PRECISIONS[(flat // span) % 4]
    return precision_map


def _pack(k, v, precision_map, group_size=16):
    return PackedKVCache.pack(k, v, precision_map, group_size=group_size)["packed_kv"]


def _entries_of(packed):
    entries = []
    for data in packed.packed_data.values():
        entries.extend(data["indices"])
    return sorted(entries, key=lambda e: (e[0], e[1]))


def _assert_equal(actual, expected, precision=None):
    assert actual.shape == expected.shape
    if precision == "int4":
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    else:
        assert torch.equal(actual, expected)


def _reference_apply_reuse_mask(packed, previous, reuse_mask):
    """Per-entry reuse helper used as the timing baseline."""
    if not reuse_mask or not previous.packed_data:
        return
    for precision, data in packed.packed_data.items():
        prev_data = previous.packed_data.get(precision)
        if prev_data is None:
            continue
        prev_map = {entry: i for i, entry in enumerate(prev_data["indices"])}
        for slice_idx, (token_idx, head_idx) in enumerate(data["indices"]):
            if not reuse_mask.get((token_idx, head_idx), False):
                continue
            prev_slice_idx = prev_map.get((token_idx, head_idx))
            if prev_slice_idx is None:
                continue
            if precision in ("fp32", "fp16", "int8"):
                data["k"][slice_idx] = prev_data["k"][prev_slice_idx]
                data["v"][slice_idx] = prev_data["v"][prev_slice_idx]
                if precision == "int8":
                    data["scale_k"][slice_idx] = prev_data["scale_k"][prev_slice_idx]
                    data["scale_v"][slice_idx] = prev_data["scale_v"][prev_slice_idx]
            else:
                k_start, k_end, v_start, v_end = data["byte_offsets"][slice_idx]
                pk_start, pk_end, pv_start, pv_end = prev_data["byte_offsets"][prev_slice_idx]
                data["k"][k_start:k_end] = prev_data["k"][pk_start:pk_end]
                data["v"][v_start:v_end] = prev_data["v"][pv_start:pv_end]
                data["scale_k"][slice_idx] = prev_data["scale_k"][prev_slice_idx]
                data["scale_v"][slice_idx] = prev_data["scale_v"][prev_slice_idx]


class TestUnpackParity:
    @pytest.mark.parametrize("batch", [1, 2])
    @pytest.mark.parametrize("evict_half", [False, True])
    @pytest.mark.parametrize("precision", PRECISIONS)
    def test_unpack_matches_reference(self, precision, batch, evict_half):
        """UT-12.1: vectorized unpack equals the golden per-entry loop."""
        torch.manual_seed(1201)
        seq_len, num_heads, head_dim = 8, 2, 64
        k = torch.randn(batch, num_heads, seq_len, head_dim)
        v = torch.randn(batch, num_heads, seq_len, head_dim)

        precision_map = _uniform_map(seq_len, num_heads, precision)
        if evict_half:
            for token_idx in range(seq_len // 2, seq_len):
                for head_idx in range(num_heads):
                    precision_map[token_idx][head_idx] = EVICTED

        packed = _pack(k, v, precision_map)
        k_ref, v_ref = packed.unpack_reference()
        k_vec, v_vec = packed.unpack()

        assert k_ref.shape == (batch, num_heads, seq_len, head_dim)
        assert v_ref.shape == (batch, num_heads, seq_len, head_dim)
        _assert_equal(k_vec, k_ref, precision)
        _assert_equal(v_vec, v_ref, precision)

        if evict_half:
            assert torch.count_nonzero(k_vec[:, :, seq_len // 2 :, :]) == 0
            assert torch.count_nonzero(v_vec[:, :, seq_len // 2 :, :]) == 0

    def test_int4_batch_scale_layout(self):
        """UT-12.2: corrected INT4 scale layout keeps per-batch error low (B=2)."""
        torch.manual_seed(1202)
        batch, seq_len, num_heads, head_dim = 2, 8, 2, 64
        k = torch.randn(batch, num_heads, seq_len, head_dim)
        v = torch.randn(batch, num_heads, seq_len, head_dim)
        precision_map = _uniform_map(seq_len, num_heads, "int4")

        packed = _pack(k, v, precision_map, group_size=16)
        k_back, v_back = packed.unpack()

        for original, restored, name in ((k, k_back, "k"), (v, v_back, "v")):
            for batch_idx in range(batch):
                rel_mse = float(
                    ((original[batch_idx] - restored[batch_idx]) ** 2).mean()
                    / (original[batch_idx] ** 2).mean()
                )
                assert rel_mse <= 0.02, f"{name} batch {batch_idx} relative MSE {rel_mse:.5f}"


class TestApplyReuseMask:
    def _build(self, seq_len=8, num_heads=2, head_dim=64, batch=1):
        torch.manual_seed(1203)
        k_prev = torch.randn(batch, num_heads, seq_len, head_dim)
        v_prev = torch.randn(batch, num_heads, seq_len, head_dim)
        k_cur = torch.randn(batch, num_heads, seq_len, head_dim)
        v_cur = torch.randn(batch, num_heads, seq_len, head_dim)
        precision_map = _mixed_map(seq_len, num_heads)
        prev = _pack(k_prev, v_prev, precision_map)
        cur = _pack(k_cur, v_cur, precision_map)
        return prev, cur, precision_map

    def test_partial_reuse_50_percent(self):
        """UT-12.3: reused positions take previous values, others stay put."""
        prev, cur, precision_map = self._build()
        prev_k, prev_v = prev.unpack()
        before_k, before_v = cur.unpack()

        reuse_mask = {
            (t, h): (t * precision_map[t].__len__() + h) % 2 == 0
            for t in precision_map
            for h in precision_map[t]
        }
        stats = cur.apply_reuse_mask(prev, reuse_mask, return_stats=True)

        after_k, after_v = cur.unpack()
        applied = 0
        for token_idx, head_map in precision_map.items():
            for head_idx, precision in head_map.items():
                key = (token_idx, head_idx)
                if reuse_mask[key]:
                    applied += 1
                    _assert_equal(
                        after_k[:, head_idx, token_idx, :],
                        prev_k[:, head_idx, token_idx, :],
                        precision,
                    )
                    _assert_equal(
                        after_v[:, head_idx, token_idx, :],
                        prev_v[:, head_idx, token_idx, :],
                        precision,
                    )
                else:
                    assert torch.equal(
                        after_k[:, head_idx, token_idx, :],
                        before_k[:, head_idx, token_idx, :],
                    )
                    assert torch.equal(
                        after_v[:, head_idx, token_idx, :],
                        before_v[:, head_idx, token_idx, :],
                    )
        assert stats["applied"] == applied
        assert stats["skipped_precision"] == 0

    def test_full_reuse_100_percent(self):
        """UT-12.3: 100% mask makes the whole current cache equal to previous."""
        prev, cur, precision_map = self._build()
        prev_k, prev_v = prev.unpack()
        reuse_mask = {(t, h): True for t in precision_map for h in precision_map[t]}
        stats = cur.apply_reuse_mask(prev, reuse_mask, return_stats=True)
        after_k, after_v = cur.unpack()

        assert stats["applied"] == len(reuse_mask)
        for precision in PRECISIONS:
            entries = [
                (t, h)
                for t, head_map in precision_map.items()
                for h, p in head_map.items()
                if p == precision
            ]
            if not entries:
                continue
            for token_idx, head_idx in entries:
                _assert_equal(
                    after_k[:, head_idx, token_idx, :],
                    prev_k[:, head_idx, token_idx, :],
                    precision,
                )
                _assert_equal(
                    after_v[:, head_idx, token_idx, :],
                    prev_v[:, head_idx, token_idx, :],
                    precision,
                )

    def test_returns_none_by_default(self):
        """UT-12.3: backward-compatible return value is None."""
        prev, cur, _ = self._build(seq_len=4, num_heads=2, head_dim=64)
        assert cur.apply_reuse_mask(prev, {(0, 0): True}) is None

    def test_cross_precision_is_skipped(self):
        """UT-12.3: cross-precision entries keep current values and are counted."""
        torch.manual_seed(1204)
        seq_len, num_heads, head_dim = 4, 2, 64
        k_prev = torch.randn(1, num_heads, seq_len, head_dim)
        v_prev = torch.randn(1, num_heads, seq_len, head_dim)
        k_cur = torch.randn(1, num_heads, seq_len, head_dim)
        v_cur = torch.randn(1, num_heads, seq_len, head_dim)

        prev = _pack(k_prev, v_prev, _uniform_map(seq_len, num_heads, "fp16"))
        current_map = _uniform_map(seq_len, num_heads, "fp16")
        current_map[0][0] = "int4"
        cur = _pack(k_cur, v_cur, current_map)

        before_k, before_v = cur.unpack()
        stats = cur.apply_reuse_mask(prev, {(0, 0): True}, return_stats=True)
        after_k, after_v = cur.unpack()

        assert stats["applied"] == 0
        assert stats["skipped_precision"] >= 1
        assert torch.equal(after_k, before_k)
        assert torch.equal(after_v, before_v)

    def test_head_dim_mismatch_raises(self):
        """UT-12.3: head_dim mismatch raises ValueError."""
        seq_len, num_heads = 4, 2
        prev = _pack(
            torch.randn(1, num_heads, seq_len, 32),
            torch.randn(1, num_heads, seq_len, 32),
            _uniform_map(seq_len, num_heads, "fp16"),
        )
        cur = _pack(
            torch.randn(1, num_heads, seq_len, 64),
            torch.randn(1, num_heads, seq_len, 64),
            _uniform_map(seq_len, num_heads, "fp16"),
        )
        with pytest.raises(ValueError):
            cur.apply_reuse_mask(prev, {(0, 0): True})

    def test_group_size_mismatch_raises(self):
        """UT-12.3: INT4 group_size mismatch raises ValueError."""
        seq_len, num_heads, head_dim = 4, 2, 64
        precision_map = _uniform_map(seq_len, num_heads, "int4")
        prev = _pack(
            torch.randn(1, num_heads, seq_len, head_dim),
            torch.randn(1, num_heads, seq_len, head_dim),
            precision_map,
            group_size=16,
        )
        cur = _pack(
            torch.randn(1, num_heads, seq_len, head_dim),
            torch.randn(1, num_heads, seq_len, head_dim),
            precision_map,
            group_size=32,
        )
        with pytest.raises(ValueError):
            cur.apply_reuse_mask(prev, {(0, 0): True})


class TestEvict:
    def test_evict_drops_bytes_without_resurrection(self):
        """UT-12.4: bucket-level eviction drops bytes and never resurrects."""
        torch.manual_seed(1205)
        batch, seq_len, num_heads, head_dim = 2, 8, 2, 64
        k = torch.randn(batch, num_heads, seq_len, head_dim)
        v = torch.randn(batch, num_heads, seq_len, head_dim)
        precision_map = _mixed_map(seq_len, num_heads, span=4)
        packed = _pack(k, v, precision_map, group_size=16)

        before_k, before_v = packed.unpack()
        before_bytes = packed.memory_bytes()
        before_counts = {
            precision: len(data["indices"]) for precision, data in packed.packed_data.items()
        }
        int4_group = packed.packed_data["int4"]["group_size"]
        int4_groups = packed.packed_data["int4"]["num_groups"]

        ordered = _entries_of(packed)
        evict_set = set(ordered[::4])
        removed = packed.evict(evict_set)

        assert isinstance(removed, int)
        assert removed == len(evict_set)
        assert packed.memory_bytes() < before_bytes

        after_counts = {
            precision: len(data["indices"]) for precision, data in packed.packed_data.items()
        }
        assert sum(after_counts.values()) == sum(before_counts.values()) - len(evict_set)
        for precision, count in after_counts.items():
            assert count <= before_counts[precision]
        assert packed.packed_data["int4"]["group_size"] == int4_group
        assert packed.packed_data["int4"]["num_groups"] == int4_groups

        after_k, after_v = packed.unpack()
        for token_idx, head_idx in evict_set:
            assert torch.count_nonzero(after_k[:, head_idx, token_idx, :]) == 0
            assert torch.count_nonzero(after_v[:, head_idx, token_idx, :]) == 0
        for token_idx, head_idx in ordered:
            if (token_idx, head_idx) in evict_set:
                continue
            assert torch.equal(
                after_k[:, head_idx, token_idx, :],
                before_k[:, head_idx, token_idx, :],
            )
            assert torch.equal(
                after_v[:, head_idx, token_idx, :],
                before_v[:, head_idx, token_idx, :],
            )

        removed_again = packed.evict(evict_set)
        assert removed_again == 0
        again_k, again_v = packed.unpack()
        assert torch.equal(again_k, after_k)
        assert torch.equal(again_v, after_v)


class TestResidentBytes:
    def test_empty_cache_accounting(self):
        """UT-12.5: an empty cache has zero resident bytes."""
        packed = PackedKVCache(1, 2, 8, 64)
        assert packed.memory_bytes() == 0
        assert packed.resident_bytes() == 0

    def test_resident_bounds_and_manager_usage(self):
        """UT-12.5: resident_bytes bounds include metadata and feed the manager."""
        torch.manual_seed(1206)
        batch, seq_len, num_heads, head_dim = 1, 8, 2, 64
        k = torch.randn(batch, num_heads, seq_len, head_dim)
        v = torch.randn(batch, num_heads, seq_len, head_dim)
        packed = _pack(k, v, _mixed_map(seq_len, num_heads, span=1), group_size=16)

        num_entries = sum(len(data["indices"]) for data in packed.packed_data.values())
        resident = packed.resident_bytes()
        assert resident >= packed.memory_bytes()
        assert resident <= packed.memory_bytes() + 64 * num_entries

        manager = KVCacheManager(num_layers=4)
        entry = KVCacheEntry(
            k_tensor=None,
            v_tensor=None,
            precision="mixed",
            packed_kv=packed,
        )
        manager.store(0, entry)
        assert manager.get_memory_usage() == packed.resident_bytes()


class TestSingleSourceStorage:
    def _make_agent(self, mock_config, store_dense=None):
        config = dict(mock_config)
        config["use_packed_cache"] = True
        if store_dense is not None:
            config["store_dense_kv"] = store_dense
        agent = ShadowKVAgent(config=config)
        agent.on_init(
            {
                "num_layers": 4,
                "num_heads": 4,
                "head_dim": 64,
                "model_name": "test",
            }
        )
        return agent

    def _run_step(self, agent, seed=1207):
        torch.manual_seed(seed)
        inputs = {
            "attention_scores": torch.randn(1, 4, 8, 8),
            "kv_current": {
                "k": torch.randn(1, 4, 8, 64),
                "v": torch.randn(1, 4, 8, 64),
            },
            "kv_previous": None,
            "layer_id": 0,
            "qdrift_signal": {
                "sensitivity_score": 0.3,
                "drift_score": 0.1,
                "shadowkv_mode": "balanced",
            },
        }
        step_config = StepConfig(
            step_id=0,
            total_steps=10,
            noise_level=0.1,
            shadowkv_mode="balanced",
            reuse_layers=[0],
            compression_target=0.5,
            ffn_mode="full",
            weight_precision_map={},
            compute_path="full",
            sensitivity_score=0.3,
            drift_score=0.1,
        )
        result = agent.on_step(step_config, inputs)
        return result["compressed_kv"]["packed_kv"]

    def test_packed_only_store_and_load(self, mock_config):
        """UT-12.6: packed-only entries load through dequantization."""
        agent = self._make_agent(mock_config)
        assert agent.store_dense_kv is False
        packed = self._run_step(agent)

        entry = agent.cache_manager.retrieve(0)
        assert entry is not None
        assert entry.packed_kv is not None
        assert entry.precision == "mixed"
        assert entry.k_tensor is None
        assert entry.v_tensor is None
        assert entry.scale_k is None
        assert entry.scale_v is None

        loaded = agent._load_kv(0)
        k_ref, v_ref = packed.unpack()
        assert torch.allclose(loaded["k"], k_ref, atol=1e-6)
        assert torch.allclose(loaded["v"], v_ref, atol=1e-6)

    def test_store_dense_kv_enabled(self, mock_config):
        """UT-12.6: explicit dense storage preserves legacy behavior."""
        agent = self._make_agent(mock_config, store_dense=True)
        assert agent.store_dense_kv is True
        packed = self._run_step(agent, seed=1208)

        entry = agent.cache_manager.retrieve(0)
        assert entry.k_tensor is not None
        assert entry.v_tensor is not None

        loaded = agent._load_kv(0)
        assert packed is not None
        assert torch.equal(loaded["k"], entry.k_tensor)
        assert torch.equal(loaded["v"], entry.v_tensor)
        assert loaded["k"].shape == entry.k_tensor.shape
        assert loaded["v"].shape == entry.v_tensor.shape

    def test_store_dense_default_without_packed(self, mock_config):
        """UT-12.6: store_dense_kv defaults to True without packed cache."""
        config = dict(mock_config)
        config["use_packed_cache"] = False
        agent = ShadowKVAgent(config=config)
        assert agent.store_dense_kv is True


class TestInt4SingleQuantization:
    def test_int4_quantized_once_and_group_size_respected(self, mock_config, monkeypatch):
        """UT-12.7: compress quantizes INT4 once; pack reuses payload and group size."""
        config = dict(mock_config)
        config["use_packed_cache"] = True
        config["int4_group_size"] = 32
        agent = ShadowKVAgent(config=config)
        assert agent.int4_group_size == 32

        torch.manual_seed(1209)
        batch, seq_len, num_heads, head_dim = 1, 4, 2, 128
        k = torch.randn(batch, num_heads, seq_len, head_dim)
        v = torch.randn(batch, num_heads, seq_len, head_dim)
        precision_map = _mixed_map(seq_len, num_heads, span=1)
        assert any(p == "int4" for head_map in precision_map.values() for p in head_map.values())

        original = Quantizer.quantize_int4
        calls = []

        def spy(tensor, *args, **kwargs):
            calls.append(tuple(tensor.shape))
            return original(tensor, *args, **kwargs)

        monkeypatch.setattr(Quantizer, "quantize_int4", spy)
        compressed = agent.compress_kv({"k": k, "v": v}, precision_map)
        packed = compressed["packed_kv"]

        assert len(calls) == 2, f"expected one K/V group each, got {len(calls)} calls"
        int4_data = packed.packed_data["int4"]
        assert int4_data["group_size"] == agent.int4_group_size
        assert int4_data["num_groups"] == head_dim // agent.int4_group_size

        num_entries = len(int4_data["indices"])
        packed_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                int4_data["k"],
                int4_data["v"],
                int4_data["scale_k"],
                int4_data["scale_v"],
            )
        )
        per_entry = packed_bytes / num_entries
        expected = entry_bytes(
            "int4",
            batch,
            head_dim,
            group_size=agent.int4_group_size,
        )
        assert abs(per_entry - expected) / expected <= 0.01

    def test_int4_group_size_default(self, mock_config):
        """UT-12.7: default INT4 group size is 16."""
        assert ShadowKVAgent(config=dict(mock_config)).int4_group_size == 16


class TestBoundariesAndAccounting:
    def test_empty_cache_unpack_and_evict(self):
        """UT-12.10: empty cache is safe and reports zero."""
        packed = PackedKVCache(1, 2, 8, 64)
        k, v = packed.unpack()
        assert torch.count_nonzero(k) == 0
        assert torch.count_nonzero(v) == 0
        assert packed.evict({(0, 0)}) == 0
        assert packed.memory_bytes() == 0
        assert packed.resident_bytes() == 0

    @pytest.mark.parametrize("precision", ["int8", "int4"])
    def test_evict_entire_bucket(self, precision):
        """UT-12.10: evicting every entry leaves zeros and zero bytes."""
        torch.manual_seed(1212)
        k = torch.randn(1, 2, 4, 64)
        v = torch.randn(1, 2, 4, 64)
        packed = _pack(k, v, _uniform_map(4, 2, precision))
        entries = set(_entries_of(packed))
        assert packed.evict(entries) == len(entries)
        assert packed.memory_bytes() == 0
        assert packed.resident_bytes() == 0
        k_back, v_back = packed.unpack()
        assert torch.count_nonzero(k_back) == 0
        assert torch.count_nonzero(v_back) == 0

    def test_reuse_batch_mismatch_raises(self):
        """UT-12.10: reuse across different batch sizes is rejected."""
        map_ = _uniform_map(4, 2, "fp16")
        prev = _pack(torch.randn(1, 2, 4, 64), torch.randn(1, 2, 4, 64), map_)
        cur = _pack(torch.randn(2, 2, 4, 64), torch.randn(2, 2, 4, 64), map_)
        with pytest.raises(ValueError):
            cur.apply_reuse_mask(prev, {(0, 0): True})

    def test_combined_dense_and_packed_accounting(self):
        """UT-12.10: retained dense shadow is added to packed resident bytes."""
        torch.manual_seed(1213)
        k = torch.randn(1, 2, 8, 64)
        v = torch.randn(1, 2, 8, 64)
        packed = _pack(k, v, _mixed_map(8, 2), group_size=16)
        manager = KVCacheManager(num_layers=1)
        entry = KVCacheEntry(k_tensor=k, v_tensor=v, precision="mixed", packed_kv=packed)
        manager.store(0, entry)
        dense_bytes = k.numel() * k.element_size() + v.numel() * v.element_size()
        assert manager.get_memory_usage() == packed.resident_bytes() + dense_bytes

    def test_release_dense_shadow_clears_fields(self):
        """UT-12.10: pressure releases the optional dense shadow physically."""
        torch.manual_seed(1216)
        k = torch.randn(1, 2, 8, 64)
        v = torch.randn(1, 2, 8, 64)
        packed = _pack(k, v, _mixed_map(8, 2), group_size=16)
        manager = KVCacheManager(num_layers=1)
        entry = KVCacheEntry(k_tensor=k, v_tensor=v, precision="mixed", packed_kv=packed)
        manager.store(0, entry)
        assert manager._release_dense_shadows() is True
        assert entry.k_tensor is None and entry.v_tensor is None
        assert manager.get_memory_usage() == packed.resident_bytes()
        assert manager._release_dense_shadows() is False

    def test_manager_eviction_frees_real_bytes(self):
        """UT-12.10: manager policy eviction drops bytes (tuple shape normalized)."""
        torch.manual_seed(1214)
        k = torch.randn(1, 2, 8, 64)
        v = torch.randn(1, 2, 8, 64)
        packed = _pack(k, v, _uniform_map(8, 2, "fp16"))
        manager = KVCacheManager(num_layers=1)
        manager.store(
            0, KVCacheEntry(k_tensor=None, v_tensor=None, precision="mixed", packed_kv=packed)
        )
        meta = {
            (t, h): TokenHeadScore(layer_id=0, token_idx=t, head_idx=h, importance=0.5)
            for t in range(8)
            for h in range(2)
        }
        manager.set_token_head_meta(0, meta)
        before = packed.memory_bytes()
        manager.memory_budget_bytes = before - 1
        manager._enforce_budget(0)
        assert packed.memory_bytes() < before


class TestInt4BudgetModel:
    def test_budget_model_uses_configured_group_size(self, mock_config):
        """UT-12.11: cost model matches packed bytes for non-default group size."""
        config = dict(mock_config)
        config["use_packed_cache"] = True
        config["int4_group_size"] = 32
        agent = ShadowKVAgent(config=config)
        torch.manual_seed(1215)
        k = torch.randn(1, 2, 4, 128)
        v = torch.randn(1, 2, 4, 128)
        precision_map = _uniform_map(4, 2, "int4")
        packed = PackedKVCache.pack(k, v, precision_map, group_size=agent.int4_group_size)[
            "packed_kv"
        ]
        predicted = agent._estimate_precision_bytes(precision_map, 1, 4, 128)
        assert predicted == pytest.approx(packed.memory_bytes(), rel=0.01)

        codes = torch.full((4, 2), 3, dtype=torch.int8)
        assert predict_codes_bytes(
            codes, 1, 128, group_size=agent.int4_group_size
        ) == pytest.approx(packed.memory_bytes(), rel=0.01)


@pytest.mark.slow
class TestSpeedSmoke:
    def test_vectorized_paths_beat_references(self):
        """UT-12.8: median-of-3 ratios for unpack / reuse / evict are >= 2x."""
        torch.manual_seed(1210)
        batch, seq_len, num_heads, head_dim = 1, 512, 8, 64
        k = torch.randn(batch, num_heads, seq_len, head_dim)
        v = torch.randn(batch, num_heads, seq_len, head_dim)
        precision_map = _mixed_map(seq_len, num_heads, span=1)
        packed = _pack(k, v, precision_map, group_size=16)

        ref_times, vec_times = [], []
        for _ in range(3):
            start = time.perf_counter()
            packed.unpack_reference()
            ref_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            packed.unpack()
            vec_times.append(time.perf_counter() - start)
        unpack_ref, unpack_vec = median(ref_times), median(vec_times)
        print(
            f"unpack reference={unpack_ref:.6f}s vectorized={unpack_vec:.6f}s "
            f"ratio={unpack_ref / unpack_vec:.2f}x"
        )
        assert unpack_vec * 5 <= unpack_ref

        prev = _pack(k, v, precision_map, group_size=16)
        cur_vec = _pack(k + 0.1, v + 0.1, precision_map, group_size=16)
        cur_ref = _pack(k + 0.1, v + 0.1, precision_map, group_size=16)
        reuse_mask = {(t, h): (t + h) % 2 == 0 for t in range(seq_len) for h in range(num_heads)}

        ref_times, vec_times = [], []
        for _ in range(3):
            start = time.perf_counter()
            cur_vec.apply_reuse_mask(prev, reuse_mask)
            vec_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            _reference_apply_reuse_mask(cur_ref, prev, reuse_mask)
            ref_times.append(time.perf_counter() - start)
        reuse_ref, reuse_vec = median(ref_times), median(vec_times)
        print(
            f"reuse reference={reuse_ref:.6f}s vectorized={reuse_vec:.6f}s "
            f"ratio={reuse_ref / reuse_vec:.2f}x"
        )
        assert reuse_vec * 3 <= reuse_ref

        ordered = _entries_of(packed)
        evict_set = set(ordered[::4])
        keep_map = {
            (t, h): precision
            for t, head_map in precision_map.items()
            for h, precision in head_map.items()
            if (t, h) not in evict_set
        }
        repack_full_map = {
            t: {h: keep_map.get((t, h), EVICTED) for h in range(num_heads)} for t in range(seq_len)
        }

        def _reference_evict(source):
            k_full, v_full = source.unpack_reference()
            return _pack(k_full, v_full, repack_full_map, group_size=16)

        ref_times, vec_times = [], []
        for _ in range(3):
            vec_cache = _pack(k, v, precision_map, group_size=16)
            start = time.perf_counter()
            vec_cache.evict(evict_set)
            vec_times.append(time.perf_counter() - start)

            ref_cache = _pack(k, v, precision_map, group_size=16)
            start = time.perf_counter()
            _reference_evict(ref_cache)
            ref_times.append(time.perf_counter() - start)
        evict_ref, evict_vec = median(ref_times), median(vec_times)
        print(
            f"evict reference={evict_ref:.6f}s vectorized={evict_vec:.6f}s "
            f"ratio={evict_ref / evict_vec:.2f}x"
        )
        assert evict_vec * 10 <= evict_ref


class TestNoneDenseManager:
    def test_none_dense_entry_with_packed(self):
        """UT-12.9: None dense tensors are safe in the manager when packed exists."""
        torch.manual_seed(1211)
        seq_len, num_heads, head_dim = 8, 2, 64
        k = torch.randn(1, num_heads, seq_len, head_dim)
        v = torch.randn(1, num_heads, seq_len, head_dim)
        packed = _pack(k, v, _mixed_map(seq_len, num_heads, span=1), group_size=16)

        manager = KVCacheManager(num_layers=2)
        entry = KVCacheEntry(
            k_tensor=None,
            v_tensor=None,
            precision="mixed",
            packed_kv=packed,
        )
        manager.store(0, entry)

        retrieved = manager.retrieve(0)
        assert retrieved is entry
        assert retrieved.k_tensor is None
        assert retrieved.v_tensor is None
        assert retrieved.packed_kv is packed
        assert manager.get_memory_usage() == packed.resident_bytes()
        assert manager.get_memory_usage() > 0
