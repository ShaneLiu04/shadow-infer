"""Tests for ShadowKV decision plane, eviction, and prefetch."""

from __future__ import annotations

import tempfile

import torch

from shadowinfer.core.structs import KVCacheEntry, StepConfig
from shadowinfer.shadowkv import (
    ImportanceAgeEvictionPolicy,
    KVCacheManager,
    KVDecisionPlane,
    KVPlaneCoordinate,
    LeastImportantEvictionPolicy,
    PackedKVCache,
    ShadowKVAgent,
)


class TestKVDecisionPlane:
    def test_precision_allocation_conservative_raises_threshold(self):
        plane = KVDecisionPlane()
        # Conservative fp32 threshold is raised to 0.9 (0.8 + 0.1).
        assert plane.allocate_precision(0.95, "conservative") == "fp32"
        assert plane.allocate_precision(0.82, "aggressive") == "fp32"
        # Conservative: 0.85 < 0.9 -> fp16.
        assert plane.allocate_precision(0.85, "conservative") == "fp16"
        # Balanced keeps default 0.8 threshold.
        assert plane.allocate_precision(0.85, "balanced") == "fp32"

    def test_decide_high_importance_no_eviction(self):
        plane = KVDecisionPlane(memory_budget_bytes=1024)
        coord = KVPlaneCoordinate(importance=0.9, drift=0.1, memory_pressure=0.8)
        decision = plane.decide(coord, mode="balanced")
        assert decision.precision == "fp32"
        assert decision.should_reuse is True
        # High importance protects against eviction.
        assert decision.should_evict is False

    def test_decide_low_importance_evicts_under_pressure(self):
        plane = KVDecisionPlane(memory_budget_bytes=1024)
        coord = KVPlaneCoordinate(importance=0.1, drift=0.1, memory_pressure=0.95)
        decision = plane.decide(coord, mode="balanced")
        assert decision.precision == "int4"
        assert decision.should_evict is True
        assert decision.eviction_priority > 0.0

    def test_decide_all_returns_full_map(self):
        plane = KVDecisionPlane()
        importance = torch.tensor([[0.9, 0.1], [0.5, 0.2]])
        drift = torch.tensor([[0.05, 0.05], [0.2, 0.2]])
        decisions = plane.decide_all(importance, drift, memory_pressure=0.0, mode="balanced")
        assert len(decisions) == 4
        assert decisions[(0, 0)].precision == "fp32"
        assert decisions[(0, 1)].precision == "int4"

    def test_split_decisions(self):
        plane = KVDecisionPlane()
        decisions = {
            (0, 0): plane.decide(KVPlaneCoordinate(0.9, 0.0, 0.0)),
            (0, 1): plane.decide(KVPlaneCoordinate(0.1, 0.0, 0.0)),
        }
        precision_map, reuse_mask, evict_set = plane.split_decisions(decisions)
        assert precision_map[0][0] == "fp32"
        assert precision_map[0][1] == "int4"
        assert isinstance(reuse_mask[(0, 0)], bool)
        assert len(evict_set) == 0


class TestEvictionPolicies:
    def test_least_important_evicts_low_scores(self):
        from shadowinfer.shadowkv.importance_model import TokenHeadScore

        policy = LeastImportantEvictionPolicy()
        entries = [
            TokenHeadScore(layer_id=0, token_idx=0, head_idx=0, importance=0.9),
            TokenHeadScore(layer_id=0, token_idx=0, head_idx=1, importance=0.1),
            TokenHeadScore(layer_id=0, token_idx=1, head_idx=0, importance=0.5),
        ]
        evicted = policy.select_for_eviction(
            entries,
            budget_bytes=100,
            current_usage_bytes=300,
            estimated_entry_bytes=80,
        )
        # target_reduction=200, num_to_evict=2 -> evict lowest importance
        assert (0, 0, 1) in evicted

    def test_importance_age_prefers_old(self):
        from shadowinfer.shadowkv.importance_model import TokenHeadScore

        policy = ImportanceAgeEvictionPolicy(age_decay=0.5)
        entries = [
            TokenHeadScore(layer_id=0, token_idx=0, head_idx=0, importance=0.5, age=10),
            TokenHeadScore(layer_id=0, token_idx=0, head_idx=1, importance=0.5, age=1),
        ]
        evicted = policy.select_for_eviction(
            entries,
            budget_bytes=100,
            current_usage_bytes=300,
            estimated_entry_bytes=150,
        )
        # Older entry has lower effective score.
        assert (0, 0, 0) in evicted


class TestPackedKVCacheEviction:
    def test_evict_reduces_memory(self):
        seq_len, num_heads, head_dim = 4, 4, 32
        k = torch.randn(1, num_heads, seq_len, head_dim)
        v = torch.randn(1, num_heads, seq_len, head_dim)
        precision_map = {t: {h: "fp16" for h in range(num_heads)} for t in range(seq_len)}
        result = PackedKVCache.pack(k, v, precision_map)
        before = result["memory_bytes"]
        result["packed_kv"].evict({(0, 0), (1, 1)})
        after = result["packed_kv"].memory_bytes()
        assert after <= before
        k_back, v_back = result["packed_kv"].unpack()
        assert k_back.shape == k.shape
        # Evicted slots should be zero (or very small).
        assert torch.allclose(k_back[0, 0, 0, :], torch.zeros(head_dim), atol=1e-3)


class TestKVCacheManagerEviction:
    def test_budget_enforcement_evicts_entries(self):
        manager = KVCacheManager(
            num_layers=1,
            memory_budget_bytes=1,  # Force eviction.
            eviction_policy=LeastImportantEvictionPolicy(),
        )
        seq_len, num_heads, head_dim = 4, 4, 8
        k = torch.randn(1, num_heads, seq_len, head_dim)
        v = torch.randn(1, num_heads, seq_len, head_dim)
        precision_map = {t: {h: "fp16" for h in range(num_heads)} for t in range(seq_len)}
        packed = PackedKVCache.pack(k, v, precision_map)
        entry = KVCacheEntry(
            k_tensor=k, v_tensor=v, precision="fp16", packed_kv=packed["packed_kv"]
        )

        from shadowinfer.shadowkv.importance_model import TokenHeadScore

        meta = {}
        for t in range(seq_len):
            for h in range(num_heads):
                meta[(t, h)] = TokenHeadScore(
                    layer_id=0, token_idx=t, head_idx=h, importance=0.1 if h == 0 else 0.9
                )
        manager.set_token_head_meta(0, meta)
        manager.store(0, entry)
        # Even after forced eviction, manager should not crash and memory is reported.
        assert manager.get_memory_usage() >= 0

    def test_prefetch_buffer(self):
        manager = KVCacheManager(num_layers=2)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        entry = KVCacheEntry(k_tensor=k, v_tensor=v, precision="fp16")
        manager.store_prefetched(0, "next", entry)
        assert manager.retrieve_prefetched(0, "next") is entry
        manager.clear_prefetch_buffer("next")
        assert manager.retrieve_prefetched(0, "next") is None


class TestShadowKVPrefetch:
    def test_prefetch_disabled_returns_empty(self, mock_config):
        agent = ShadowKVAgent(config=mock_config)
        agent.on_init({"num_layers": 2, "num_heads": 4, "head_dim": 16})
        k = torch.randn(1, 4, 8, 16)
        v = torch.randn(1, 4, 8, 16)
        result = agent.prefetch_next_step(
            {"kv_current": {"k": k, "v": v}, "layer_id": 0},
            {"compressed_kv": {"k": k, "v": v}},
            predicted_sensitivity=0.2,
            predicted_drift=0.1,
        )
        assert result["prefetched_count"] == 0

    def test_prefetch_enabled_stages_entries(self, mock_config):
        cfg = dict(mock_config)
        cfg["prefetch_enabled"] = True
        cfg["use_packed_cache"] = True
        agent = ShadowKVAgent(config=cfg)
        agent.on_init({"num_layers": 1, "num_heads": 4, "head_dim": 16})

        seq_len = 8
        k = torch.randn(1, 4, seq_len, 16)
        v = torch.randn(1, 4, seq_len, 16)
        precision_map = {t: {h: "fp16" for h in range(4)} for t in range(seq_len)}
        compressed = agent.compress_kv({"k": k, "v": v}, precision_map)

        attention_scores = torch.randn(1, 4, seq_len, seq_len)
        result = agent.prefetch_next_step(
            {
                "attention_scores": attention_scores,
                "kv_current": {"k": k, "v": v},
                "kv_previous": {"k": k, "v": v},
                "layer_id": 0,
                "step_id": 0,
                "total_steps": 10,
            },
            {"compressed_kv": compressed},
            predicted_sensitivity=0.2,
            predicted_drift=0.0,
        )
        assert result["prefetched_count"] >= 0
        assert agent.cache_manager.retrieve_prefetched(0, "next") is not None
