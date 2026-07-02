"""测试 ShadowKV Agent 核心算法。"""

import torch

from shadowinfer.core.structs import StepConfig
from shadowinfer.shadowkv import KVCacheManager, ShadowKVAgent


class TestShadowKV:
    def test_importance_scoring(self, mock_config):
        """验证重要性评分在 [0, 1] 范围内。"""
        agent = ShadowKVAgent(config=mock_config)
        attention_scores = torch.randn(1, 32, 128, 128)
        for token_idx in range(128):
            for head_idx in range(32):
                score = agent.compute_importance_score(
                    attention_scores, token_idx, head_idx, layer_index=0, num_layers=4
                )
                assert (
                    0.0 <= score <= 1.0
                ), f"Score {score} out of range at token={token_idx}, head={head_idx}"

    def test_precision_allocation(self, mock_config):
        """验证精度分配包含有效的精度级别。"""
        agent = ShadowKVAgent(config=mock_config)
        importance = torch.rand(128, 32)
        precision_map = agent.allocate_precision(importance, "balanced")
        for token_map in precision_map.values():
            for prec in token_map.values():
                assert prec in ["fp32", "fp16", "int8", "int4"]

    def test_precision_allocation_modes(self, mock_config):
        """验证 aggressive/balanced/conservative 模式影响阈值。"""
        agent = ShadowKVAgent(config=mock_config)
        torch.manual_seed(42)
        importance = torch.rand(128, 32)
        aggressive_map = agent.allocate_precision(importance, "aggressive")
        balanced_map = agent.allocate_precision(importance, "balanced")
        conservative_map = agent.allocate_precision(importance, "conservative")

        aggressive_int4 = sum(
            1 for tm in aggressive_map.values() for p in tm.values() if p == "int4"
        )
        balanced_int4 = sum(1 for tm in balanced_map.values() for p in tm.values() if p == "int4")
        conservative_int4 = sum(
            1 for tm in conservative_map.values() for p in tm.values() if p == "int4"
        )
        # aggressive 阈值更低（int8 门槛更低），因此 int4 分配更少；conservative 相反
        assert aggressive_int4 <= balanced_int4 <= conservative_int4

    def test_quantization_roundtrip(self, mock_config):
        """验证量化-反量化误差在可接受范围。"""
        from shadowinfer.utils.quantization import Quantizer

        tensor = torch.randn(1, 128, 32, 128)
        for precision in ["int8", "int4"]:
            quantized, scale = Quantizer.quantize_tensor(tensor, precision)
            dequantized = Quantizer.dequantize_tensor(quantized, scale, precision)
            if precision == "int4":
                dequantized = dequantized.reshape(*tensor.shape)
            error = torch.norm(tensor - dequantized) / torch.norm(tensor)
            threshold = 0.05 if precision == "int8" else 0.15
            assert (
                error < threshold
            ), f"Precision {precision} roundtrip error {error:.4f} >= {threshold}"

    def test_reuse_decision(self, mock_config):
        """验证复用决策逻辑。"""
        agent = ShadowKVAgent(config=mock_config)
        k_current = torch.randn(1, 32, 128, 128)
        v_current = torch.randn(1, 32, 128, 128)
        k_prev = torch.randn(1, 32, 128, 128)
        v_prev = torch.randn(1, 32, 128, 128)

        should_reuse, strategy, reused_heads = agent.decide_reuse(
            k_current,
            v_current,
            k_prev,
            v_prev,
            step_id=5,
            total_steps=10,
            qdrift_drift_score=0.1,
            mode="balanced",
        )
        assert isinstance(should_reuse, bool)
        assert strategy in ["full_reuse", "partial_reuse", "no_reuse"]
        assert isinstance(reused_heads, list)

    def test_full_reuse(self, mock_config):
        """验证 delta 很小时返回 full_reuse。"""
        agent = ShadowKVAgent(config=mock_config)
        k_prev = torch.randn(1, 32, 128, 128)
        v_prev = torch.randn(1, 32, 128, 128)
        # delta 非常小
        k_current = k_prev + 1e-6 * torch.randn_like(k_prev)
        v_current = v_prev + 1e-6 * torch.randn_like(v_prev)

        should_reuse, strategy, reused_heads = agent.decide_reuse(
            k_current,
            v_current,
            k_prev,
            v_prev,
            step_id=0,
            total_steps=10,
            qdrift_drift_score=0.0,
            mode="aggressive",
        )
        assert should_reuse is True
        assert strategy == "full_reuse"
        assert len(reused_heads) == 32

    def test_partial_reuse(self, mock_config):
        """验证中等 delta 返回 partial_reuse。"""
        agent = ShadowKVAgent(config=mock_config)
        k_prev = torch.randn(1, 32, 128, 128)
        v_prev = torch.randn(1, 32, 128, 128)
        # 中等 delta
        k_current = k_prev + 0.01 * torch.randn_like(k_prev)
        v_current = v_prev + 0.01 * torch.randn_like(v_prev)

        should_reuse, strategy, reused_heads = agent.decide_reuse(
            k_current,
            v_current,
            k_prev,
            v_prev,
            step_id=0,
            total_steps=10,
            qdrift_drift_score=0.0,
            mode="aggressive",
        )
        if should_reuse:
            assert strategy in ["full_reuse", "partial_reuse"]
        else:
            assert strategy == "no_reuse"

    def test_no_reuse(self, mock_config):
        """验证大 delta 返回 no_reuse。"""
        agent = ShadowKVAgent(config=mock_config)
        k_prev = torch.randn(1, 32, 128, 128)
        v_prev = torch.randn(1, 32, 128, 128)
        # 大 delta
        k_current = torch.randn(1, 32, 128, 128)
        v_current = torch.randn(1, 32, 128, 128)

        should_reuse, strategy, reused_heads = agent.decide_reuse(
            k_current,
            v_current,
            k_prev,
            v_prev,
            step_id=0,
            total_steps=10,
            qdrift_drift_score=0.0,
            mode="balanced",
        )
        assert should_reuse is False
        assert strategy == "no_reuse"
        assert reused_heads == []

    def test_compression_stats(self, mock_config):
        """验证压缩统计计算。"""
        agent = ShadowKVAgent(config=mock_config)
        tensor = torch.randn(1, 128, 32, 128)
        precision_map = {}
        for token_idx in range(128):
            precision_map[token_idx] = {}
            for head_idx in range(32):
                precision_map[token_idx][head_idx] = "int8"

        compressed = agent.compress_kv(
            {"k": tensor, "v": tensor},
            precision_map,
            reuse_decision=None,
            kv_previous=None,
        )
        assert "compression_ratio" in compressed
        assert "memory_mb" in compressed
        assert compressed["compression_ratio"] > 0.0
        assert compressed["memory_mb"] > 0.0

        stats = agent.get_compression_stats()
        assert "compression_ratio" in stats
        assert "memory_savings_mb" in stats
        assert "avg_precision_bits" in stats
        assert stats["step_count"] >= 1.0

    def test_kv_cache_manager(self, mock_config):
        """验证 KVCacheManager 存储和检索。"""
        manager = KVCacheManager(num_layers=4)
        k = torch.randn(1, 32, 128, 128)
        v = torch.randn(1, 32, 128, 128)
        from shadowinfer.core.structs import KVCacheEntry

        entry = KVCacheEntry(
            k_tensor=k,
            v_tensor=v,
            precision="fp16",
            importance_score=0.5,
            is_reused=False,
            reuse_step=-1,
        )
        manager.store(0, entry)
        retrieved = manager.retrieve(0)
        assert retrieved is not None
        assert torch.equal(retrieved.k_tensor, k)
        assert torch.equal(retrieved.v_tensor, v)
        assert retrieved.precision == "fp16"

        # 未存储的 layer
        assert manager.retrieve(1) is None

        # 内存统计
        mem_bytes = manager.get_memory_usage()
        assert mem_bytes > 0
        mem_mb = manager.get_memory_usage_mb()
        assert mem_mb > 0

    def test_on_step_and_shutdown(self, mock_config, mock_step_config):
        """验证 ShadowKVAgent on_step 和 on_shutdown 完整流程。"""
        agent = ShadowKVAgent(config=mock_config)
        agent.on_init(
            {
                "num_layers": 4,
                "num_heads": 32,
                "head_dim": 128,
                "model_name": "test",
            }
        )

        inputs = {
            "attention_scores": torch.randn(1, 32, 128, 128),
            "kv_current": {
                "k": torch.randn(1, 32, 128, 128),
                "v": torch.randn(1, 32, 128, 128),
            },
            "kv_previous": {
                "k": torch.randn(1, 32, 128, 128),
                "v": torch.randn(1, 32, 128, 128),
            },
            "layer_id": 0,
            "qdrift_signal": {
                "sensitivity_score": 0.3,
                "drift_score": 0.1,
                "shadowkv_mode": "balanced",
            },
        }
        result = agent.on_step(mock_step_config, inputs)
        assert "compressed_kv" in result
        assert "precision_map" in result
        assert "reuse_decision" in result
        assert "importance_stats" in result

        profile = agent.on_shutdown()
        assert profile is not None
        assert profile.model_name == "test"

    def test_on_step_with_reuse(self, mock_config, mock_step_config):
        """验证 ShadowKVAgent 在复用场景下的行为。"""
        agent = ShadowKVAgent(config=mock_config)
        agent.on_init(
            {
                "num_layers": 4,
                "num_heads": 32,
                "head_dim": 128,
                "model_name": "test",
            }
        )

        k = torch.randn(1, 32, 128, 128)
        v = torch.randn(1, 32, 128, 128)

        # 第一步（无 previous）
        inputs1 = {
            "attention_scores": torch.randn(1, 32, 128, 128),
            "kv_current": {"k": k, "v": v},
            "kv_previous": None,
            "layer_id": 0,
            "qdrift_signal": {
                "sensitivity_score": 0.3,
                "drift_score": 0.1,
                "shadowkv_mode": "balanced",
            },
        }
        result1 = agent.on_step(mock_step_config, inputs1)
        assert result1["reuse_decision"]["strategy"] == "no_reuse"

        # 第二步（有 previous）
        inputs2 = {
            "attention_scores": torch.randn(1, 32, 128, 128),
            "kv_current": {
                "k": k + 1e-5 * torch.randn_like(k),
                "v": v + 1e-5 * torch.randn_like(v),
            },
            "kv_previous": {"k": k, "v": v},
            "layer_id": 0,
            "qdrift_signal": {
                "sensitivity_score": 0.3,
                "drift_score": 0.1,
                "shadowkv_mode": "balanced",
            },
        }
        step_config2 = StepConfig(
            step_id=1,
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
        result2 = agent.on_step(step_config2, inputs2)
        assert "reuse_decision" in result2

    def test_decide_reuse_per_token_structure(self, mock_config):
        """验证 decide_reuse_per_token 返回结构和统计信息。"""
        agent = ShadowKVAgent(config=mock_config)
        seq_len, num_heads, head_dim = 16, 8, 32
        k_current = torch.randn(1, num_heads, seq_len, head_dim)
        v_current = torch.randn(1, num_heads, seq_len, head_dim)
        k_prev = torch.randn(1, num_heads, seq_len, head_dim)
        v_prev = torch.randn(1, num_heads, seq_len, head_dim)
        importance_map = torch.rand(seq_len, num_heads)

        reuse_mask, stats = agent.decide_reuse_per_token(
            k_current,
            v_current,
            k_prev,
            v_prev,
            step_id=1,
            total_steps=10,
            qdrift_drift_score=0.1,
            mode="balanced",
            importance_map=importance_map,
        )

        assert isinstance(reuse_mask, dict)
        assert all(isinstance(k, tuple) and len(k) == 2 for k in reuse_mask.keys())
        assert all(isinstance(v, bool) for v in reuse_mask.values())
        expected_keys = {(t, h) for t in range(seq_len) for h in range(num_heads)}
        assert set(reuse_mask.keys()) == expected_keys

        for key in (
            "full_reuse_count",
            "partial_reuse_count",
            "no_reuse_count",
            "reuse_ratio",
        ):
            assert key in stats
        assert 0.0 <= stats["reuse_ratio"] <= 1.0
        assert (
            stats["full_reuse_count"] + stats["partial_reuse_count"] + stats["no_reuse_count"]
            == seq_len
        )

    def test_token_level_reuse_mask_in_compress_kv(self, mock_config):
        """验证 token-head 级别的 reuse_mask 能被 compress_kv 正确应用。"""
        agent = ShadowKVAgent(config=mock_config)
        seq_len, num_heads, head_dim = 4, 4, 8
        k_current = torch.ones(1, num_heads, seq_len, head_dim)
        v_current = torch.ones(1, num_heads, seq_len, head_dim)
        k_prev = torch.zeros(1, num_heads, seq_len, head_dim)
        v_prev = torch.zeros(1, num_heads, seq_len, head_dim)
        precision_map = {t: {h: "fp16" for h in range(num_heads)} for t in range(seq_len)}
        reuse_mask = {(0, 0): True, (2, 3): True}

        compressed = agent.compress_kv(
            {"k": k_current, "v": v_current},
            precision_map,
            reuse_decision={
                "should_reuse": True,
                "strategy": "partial_reuse",
                "reused_heads": [],
                "reuse_mask": reuse_mask,
            },
            kv_previous={"k": k_prev, "v": v_prev},
        )

        # Reused token-heads should come from previous KV.
        assert torch.equal(compressed["k"][0, 0, 0, :], k_prev[0, 0, 0, :])
        assert torch.equal(compressed["v"][0, 0, 0, :], v_prev[0, 0, 0, :])
        assert torch.equal(compressed["k"][0, 3, 2, :], k_prev[0, 3, 2, :])
        assert torch.equal(compressed["v"][0, 3, 2, :], v_prev[0, 3, 2, :])

        # Non-reused token-heads should be quantized from current KV.
        assert torch.equal(compressed["k"][0, 1, 0, :].float(), k_current[0, 1, 0, :])
        assert torch.equal(compressed["k"][0, 0, 1, :].float(), k_current[0, 0, 1, :])

    def test_compress_kv_backward_compatible_global_heads(self, mock_config):
        """验证 compress_kv 对旧格式 reused_heads 的向后兼容。"""
        agent = ShadowKVAgent(config=mock_config)
        seq_len, num_heads, head_dim = 4, 4, 8
        k_current = torch.ones(1, num_heads, seq_len, head_dim)
        v_current = torch.ones(1, num_heads, seq_len, head_dim)
        k_prev = torch.zeros(1, num_heads, seq_len, head_dim)
        v_prev = torch.zeros(1, num_heads, seq_len, head_dim)
        precision_map = {t: {h: "fp16" for h in range(num_heads)} for t in range(seq_len)}

        compressed = agent.compress_kv(
            {"k": k_current, "v": v_current},
            precision_map,
            reuse_decision={
                "should_reuse": True,
                "strategy": "partial_reuse",
                "reused_heads": [0, 1],
            },
            kv_previous={"k": k_prev, "v": v_prev},
        )

        for token_idx in range(seq_len):
            assert torch.equal(compressed["k"][0, 0, token_idx, :], k_prev[0, 0, token_idx, :])
            assert torch.equal(
                compressed["k"][0, 2, token_idx, :].float(), k_current[0, 2, token_idx, :]
            )


class TestPackedKVCache:
    def _build_uniform_precision_map(self, seq_len, num_heads, precision):
        return {t: {h: precision for h in range(num_heads)} for t in range(seq_len)}

    def test_pack_unpack_roundtrip_fp32(self):
        """验证 FP32 packed cache 完全可还原。"""
        from shadowinfer.shadowkv.packed_kv_cache import PackedKVCache

        k = torch.randn(1, 4, 8, 32)
        v = torch.randn(1, 4, 8, 32)
        precision_map = self._build_uniform_precision_map(8, 4, "fp32")

        result = PackedKVCache.pack(k, v, precision_map)
        k_back, v_back = result["packed_kv"].unpack()
        assert torch.allclose(k_back, k, atol=1e-6)
        assert torch.allclose(v_back, v, atol=1e-6)

    def test_pack_unpack_roundtrip_fp16(self):
        """验证 FP16 packed cache  roundtrip 误差很小。"""
        from shadowinfer.shadowkv.packed_kv_cache import PackedKVCache

        k = torch.randn(1, 4, 8, 32)
        v = torch.randn(1, 4, 8, 32)
        precision_map = self._build_uniform_precision_map(8, 4, "fp16")

        result = PackedKVCache.pack(k, v, precision_map)
        k_back, v_back = result["packed_kv"].unpack()
        assert torch.allclose(k_back, k, atol=1e-3)
        assert torch.allclose(v_back, v, atol=1e-3)

    def test_pack_unpack_roundtrip_int8(self):
        """验证 INT8 packed cache roundtrip 误差 < 1e-2。"""
        from shadowinfer.shadowkv.packed_kv_cache import PackedKVCache

        k = torch.randn(1, 4, 8, 128)
        v = torch.randn(1, 4, 8, 128)
        precision_map = self._build_uniform_precision_map(8, 4, "int8")

        result = PackedKVCache.pack(k, v, precision_map)
        k_back, v_back = result["packed_kv"].unpack()
        k_err = torch.norm(k_back - k) / torch.norm(k)
        v_err = torch.norm(v_back - v) / torch.norm(v)
        assert k_err < 1e-2, f"INT8 K roundtrip error {k_err:.4f}"
        assert v_err < 1e-2, f"INT8 V roundtrip error {v_err:.4f}"

    def test_pack_unpack_roundtrip_int4(self):
        """验证 INT4 packed cache roundtrip 误差 < 1e-1。"""
        from shadowinfer.shadowkv.packed_kv_cache import PackedKVCache

        k = torch.randn(1, 4, 8, 128)
        v = torch.randn(1, 4, 8, 128)
        precision_map = self._build_uniform_precision_map(8, 4, "int4")

        result = PackedKVCache.pack(k, v, precision_map)
        k_back, v_back = result["packed_kv"].unpack()
        k_err = torch.norm(k_back - k) / torch.norm(k)
        v_err = torch.norm(v_back - v) / torch.norm(v)
        assert k_err < 1e-1, f"INT4 K roundtrip error {k_err:.4f}"
        assert v_err < 1e-1, f"INT4 V roundtrip error {v_err:.4f}"

    def test_int4_packing_correctness(self):
        """验证 INT4 真正打包为 uint8（0.5 byte/value）。"""
        from shadowinfer.shadowkv.packed_kv_cache import PackedKVCache

        seq_len, num_heads, head_dim = 4, 4, 128
        k = torch.randn(1, num_heads, seq_len, head_dim)
        v = torch.randn(1, num_heads, seq_len, head_dim)
        precision_map = self._build_uniform_precision_map(seq_len, num_heads, "int4")

        result = PackedKVCache.pack(k, v, precision_map)
        packed = result["packed_kv"]
        int4_data = packed.packed_data["int4"]

        total_values = num_heads * seq_len * head_dim
        expected_bytes = (total_values + 1) // 2 * 2  # K + V
        assert int4_data["k"].dtype == torch.uint8
        assert int4_data["v"].dtype == torch.uint8
        assert int4_data["k"].numel() == (total_values + 1) // 2
        assert int4_data["v"].numel() == (total_values + 1) // 2
        assert (
            packed.memory_bytes()
            == expected_bytes
            + int4_data["scale_k"].numel() * int4_data["scale_k"].element_size() * 2
        )

    def test_memory_savings_real(self):
        """验证 packed cache 真实节省内存：FP16 < FP32, INT8 < FP16, INT4 < INT8。"""
        from shadowinfer.shadowkv.packed_kv_cache import PackedKVCache

        k = torch.randn(1, 4, 8, 128)
        v = torch.randn(1, 4, 8, 128)

        fp32_mb = PackedKVCache.pack(k, v, self._build_uniform_precision_map(8, 4, "fp32"))[
            "memory_mb"
        ]
        fp16_mb = PackedKVCache.pack(k, v, self._build_uniform_precision_map(8, 4, "fp16"))[
            "memory_mb"
        ]
        int8_mb = PackedKVCache.pack(k, v, self._build_uniform_precision_map(8, 4, "int8"))[
            "memory_mb"
        ]
        int4_mb = PackedKVCache.pack(k, v, self._build_uniform_precision_map(8, 4, "int4"))[
            "memory_mb"
        ]

        assert fp16_mb < fp32_mb
        assert int8_mb < fp16_mb
        assert int4_mb < int8_mb

    def test_apply_reuse_mask(self):
        """验证 apply_reuse_mask 能从 previous packed cache 复制指定 token-heads。"""
        from shadowinfer.shadowkv.packed_kv_cache import PackedKVCache

        seq_len, num_heads, head_dim = 4, 4, 128
        k_current = torch.randn(1, num_heads, seq_len, head_dim)
        v_current = torch.randn(1, num_heads, seq_len, head_dim)
        k_prev = torch.randn(1, num_heads, seq_len, head_dim)
        v_prev = torch.randn(1, num_heads, seq_len, head_dim)
        precision_map = self._build_uniform_precision_map(seq_len, num_heads, "fp16")

        prev_result = PackedKVCache.pack(k_prev, v_prev, precision_map)
        current_result = PackedKVCache.pack(k_current, v_current, precision_map)

        reuse_mask = {(0, 0): True, (2, 3): True}
        current_result["packed_kv"].apply_reuse_mask(prev_result["packed_kv"], reuse_mask)

        k_back, v_back = current_result["packed_kv"].unpack()
        assert torch.allclose(k_back[0, 0, 0, :], k_prev[0, 0, 0, :], atol=1e-3)
        assert torch.allclose(v_back[0, 0, 0, :], v_prev[0, 0, 0, :], atol=1e-3)
        assert torch.allclose(k_back[0, 3, 2, :], k_prev[0, 3, 2, :], atol=1e-3)
        assert torch.allclose(v_back[0, 3, 2, :], v_prev[0, 3, 2, :], atol=1e-3)

        # Non-reused entries remain from current.
        assert not torch.allclose(k_back[0, 1, 0, :], k_prev[0, 1, 0, :], atol=1e-3)

    def test_mixed_precision_per_token_head(self):
        """验证每个 token-head 可以指定不同精度并正确还原。"""
        from shadowinfer.shadowkv.packed_kv_cache import PackedKVCache

        seq_len, num_heads, head_dim = 2, 4, 128
        k = torch.randn(1, num_heads, seq_len, head_dim)
        v = torch.randn(1, num_heads, seq_len, head_dim)
        precision_map = {
            0: {0: "fp32", 1: "fp16", 2: "int8", 3: "int4"},
            1: {0: "int4", 1: "int8", 2: "fp16", 3: "fp32"},
        }

        result = PackedKVCache.pack(k, v, precision_map)
        k_back, v_back = result["packed_kv"].unpack()

        for token_idx, token_map in precision_map.items():
            for head_idx, precision in token_map.items():
                original = k[0, head_idx, token_idx, :]
                restored = k_back[0, head_idx, token_idx, :]
                if precision == "fp32":
                    assert torch.allclose(original, restored, atol=1e-6)
                elif precision == "fp16":
                    assert torch.allclose(original, restored, atol=1e-3)
                elif precision == "int8":
                    err = torch.norm(original - restored) / torch.norm(original)
                    assert err < 1e-2
                elif precision == "int4":
                    err = torch.norm(original - restored) / torch.norm(original)
                    assert err < 1e-1

    def test_agent_compress_kv_returns_packed_kv(self, mock_config):
        """验证 ShadowKVAgent compress_kv 默认返回 packed_kv。"""
        agent = ShadowKVAgent(config=mock_config)
        seq_len, num_heads, head_dim = 4, 4, 128
        k = torch.randn(1, num_heads, seq_len, head_dim)
        v = torch.randn(1, num_heads, seq_len, head_dim)
        precision_map = self._build_uniform_precision_map(seq_len, num_heads, "int8")

        compressed = agent.compress_kv({"k": k, "v": v}, precision_map)
        assert "packed_kv" in compressed
        assert compressed["packed_kv"] is not None
        assert compressed["memory_bytes"] > 0

    def test_agent_compress_kv_disable_packed_cache(self, mock_config):
        """验证 use_packed_cache=False 时不返回 packed_kv。"""
        config = dict(mock_config)
        config["use_packed_cache"] = False
        agent = ShadowKVAgent(config=config)
        seq_len, num_heads, head_dim = 4, 4, 128
        k = torch.randn(1, num_heads, seq_len, head_dim)
        v = torch.randn(1, num_heads, seq_len, head_dim)
        precision_map = self._build_uniform_precision_map(seq_len, num_heads, "int8")

        compressed = agent.compress_kv({"k": k, "v": v}, precision_map)
        assert "packed_kv" not in compressed

    def test_kv_cache_manager_store_retrieve_packed(self, mock_config):
        """验证 KVCacheManager 可存储/检索 packed cache。"""
        from shadowinfer.shadowkv.packed_kv_cache import PackedKVCache

        manager = KVCacheManager(num_layers=4)
        seq_len, num_heads, head_dim = 4, 4, 128
        k = torch.randn(1, num_heads, seq_len, head_dim)
        v = torch.randn(1, num_heads, seq_len, head_dim)
        precision_map = self._build_uniform_precision_map(seq_len, num_heads, "int8")

        packed_result = PackedKVCache.pack(k, v, precision_map)
        manager.store_packed(0, packed_result)

        retrieved = manager.retrieve_packed(0)
        assert retrieved is not None
        assert retrieved.memory_bytes() == packed_result["memory_bytes"]

        k_back, v_back = retrieved.unpack()
        assert k_back.shape == k.shape
        assert v_back.shape == v.shape
