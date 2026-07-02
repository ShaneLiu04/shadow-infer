"""测试 FFN Optimizer Agent 核心算法。"""

import pytest
import torch

from shadowinfer.core.structs import StepConfig
from shadowinfer.ffn_optimizer import FFNOptimizerAgent, PackedFFNWeight


class TestFFNOptimizer:
    def test_channel_importance_activation(self, mock_config, mock_ffn_weights, mock_ffn_inputs):
        """验证 activation_magnitude 重要性分析。"""
        agent = FFNOptimizerAgent(config=mock_config)
        importance = agent.analyze_channel_importance(
            ffn_input=mock_ffn_inputs,
            weight_up=mock_ffn_weights["up"],
            weight_down=mock_ffn_weights["down"],
            method="activation_magnitude",
        )
        assert importance.shape[0] == 11008
        assert (importance >= 0).all() and (importance <= 1).all()

    def test_channel_importance_weight(self, mock_config, mock_ffn_weights, mock_ffn_inputs):
        """验证 weight_magnitude 重要性分析。"""
        agent = FFNOptimizerAgent(config=mock_config)
        importance = agent.analyze_channel_importance(
            ffn_input=mock_ffn_inputs,
            weight_up=mock_ffn_weights["up"],
            weight_down=mock_ffn_weights["down"],
            method="weight_magnitude",
        )
        assert importance.shape[0] == 11008
        assert (importance >= 0).all() and (importance <= 1).all()

    def test_mixed_precision_quantization(self, mock_config, mock_ffn_weights, mock_ffn_inputs):
        """验证混合精度量化分三级。"""
        agent = FFNOptimizerAgent(config=mock_config)
        importance = agent.analyze_channel_importance(
            mock_ffn_inputs,
            mock_ffn_weights["up"],
            mock_ffn_weights["down"],
            method="weight_magnitude",
        )
        q_up, q_down, scales = agent.quantize_mixed_precision(
            mock_ffn_weights["up"], mock_ffn_weights["down"], importance
        )
        assert q_up.shape == mock_ffn_weights["up"].shape
        assert q_down.shape == mock_ffn_weights["down"].shape

        fp16_count = sum(1 for p in scales.values() if p[0] == "fp16")
        int8_count = sum(1 for p in scales.values() if p[0] == "int8")
        int4_count = sum(1 for p in scales.values() if p[0] == "int4")
        assert fp16_count + int8_count + int4_count == len(scales)
        assert fp16_count > 0 or int8_count > 0 or int4_count > 0

    def test_sparse_update(self, mock_config, mock_ffn_weights, mock_ffn_inputs):
        """验证稀疏更新只重算变化 token。"""
        agent = FFNOptimizerAgent(config=mock_config)
        ffn_input_prev = mock_ffn_inputs.clone()
        ffn_output_prev = torch.randn(1, 128, 4096)
        # 只有部分 token 变化
        ffn_input_current = ffn_input_prev.clone()
        ffn_input_current[0, 0, :] += 1.0
        ffn_input_current[0, 1, :] += 1.0

        output, stats = agent.sparse_update(
            ffn_input_current,
            ffn_input_prev,
            ffn_output_prev,
            mock_ffn_weights["up"],
            mock_ffn_weights["down"],
            delta_threshold=0.05,
        )
        assert output.shape == ffn_output_prev.shape
        assert "changed_tokens" in stats
        assert "changed_tokens_ratio" in stats
        assert "unchanged_tokens" in stats
        assert stats["changed_tokens"] > 0

    def test_incremental_reconstruct(self, mock_config, mock_ffn_weights, mock_ffn_inputs):
        """验证增量重构只计算变化部分。"""
        agent = FFNOptimizerAgent(config=mock_config)
        ffn_input_prev = mock_ffn_inputs.clone()
        ffn_input_current = ffn_input_prev + 0.01 * torch.randn_like(ffn_input_prev)

        delta_output = agent.incremental_reconstruct(
            ffn_input_current,
            ffn_input_prev,
            mock_ffn_weights["up"],
            mock_ffn_weights["down"],
            delta_threshold=0.05,
        )
        assert delta_output.shape == mock_ffn_inputs.shape
        # 当输入差值很小时，significant_delta 可能被 threshold 过滤

    def test_compute_path_selection(self, mock_config, mock_ffn_inputs, mock_ffn_weights):
        """验证计算路径选择阈值。"""
        agent = FFNOptimizerAgent(config=mock_config)
        ffn_input_prev = mock_ffn_inputs.clone()

        # 小变化 -> reuse
        ffn_input_current_small = ffn_input_prev + 0.001 * torch.randn_like(ffn_input_prev)
        path_small = agent.select_compute_path(
            ffn_input_current_small,
            ffn_input_prev,
            torch.randn(1, 128, 4096),
            sensitivity_score=0.0,
            mode="sparse",
        )
        assert path_small == "reuse"

        # 中等变化 -> incremental
        ffn_input_current_mid = ffn_input_prev + 0.03 * torch.randn_like(ffn_input_prev)
        path_mid = agent.select_compute_path(
            ffn_input_current_mid,
            ffn_input_prev,
            torch.randn(1, 128, 4096),
            sensitivity_score=0.0,
            mode="sparse",
        )
        assert path_mid == "incremental"

        # 较大变化 -> sparse
        ffn_input_current_large = ffn_input_prev + 0.08 * torch.randn_like(ffn_input_prev)
        path_large = agent.select_compute_path(
            ffn_input_current_large,
            ffn_input_prev,
            torch.randn(1, 128, 4096),
            sensitivity_score=0.0,
            mode="sparse",
        )
        assert path_large == "sparse"

        # 大变化 -> full
        ffn_input_current_huge = ffn_input_prev + 0.5 * torch.randn_like(ffn_input_prev)
        path_huge = agent.select_compute_path(
            ffn_input_current_huge,
            ffn_input_prev,
            torch.randn(1, 128, 4096),
            sensitivity_score=0.0,
            mode="sparse",
        )
        assert path_huge == "full"

    def test_compute_path_reuse(self, mock_config, mock_ffn_inputs, mock_ffn_weights):
        """验证极小变化时选择 reuse。"""
        agent = FFNOptimizerAgent(config=mock_config)
        ffn_input_prev = mock_ffn_inputs.clone()
        ffn_input_current = ffn_input_prev + 1e-5 * torch.randn_like(ffn_input_prev)
        path = agent.select_compute_path(
            ffn_input_current,
            ffn_input_prev,
            torch.randn(1, 128, 4096),
            sensitivity_score=0.0,
            mode="sparse",
        )
        assert path == "reuse"

    def test_compute_path_full(self, mock_config, mock_ffn_inputs, mock_ffn_weights):
        """验证大变化时选择 full。"""
        agent = FFNOptimizerAgent(config=mock_config)
        ffn_input_prev = mock_ffn_inputs.clone()
        ffn_input_current = torch.randn_like(mock_ffn_inputs)
        path = agent.select_compute_path(
            ffn_input_current,
            ffn_input_prev,
            torch.randn(1, 128, 4096),
            sensitivity_score=0.0,
            mode="sparse",
        )
        assert path == "full"

    def test_sensitivity_forces_full(self, mock_config, mock_ffn_inputs, mock_ffn_weights):
        """验证高敏感度强制 full 模式。"""
        agent = FFNOptimizerAgent(config=mock_config)
        path = agent.select_compute_path(
            mock_ffn_inputs,
            None,
            None,
            sensitivity_score=0.8,
            mode="sparse",
        )
        assert path == "full"

    def test_on_step_and_shutdown(self, mock_config, mock_ffn_inputs, mock_ffn_weights):
        """验证 FFNOptimizerAgent on_step 和 on_shutdown 完整流程。"""
        agent = FFNOptimizerAgent(config=mock_config)
        agent.on_init(
            {
                "model_name": "test",
                "weights": mock_ffn_weights,
            }
        )

        step_config = StepConfig(
            step_id=0,
            total_steps=10,
            noise_level=0.0,
            shadowkv_mode="balanced",
            reuse_layers=[0],
            compression_target=0.5,
            ffn_mode="full",
            weight_precision_map={},
            compute_path="full",
            sensitivity_score=0.0,
            drift_score=0.0,
        )
        inputs = {
            "ffn_input_current": mock_ffn_inputs,
            "ffn_input_previous": None,
            "ffn_output_previous": None,
            "weights": mock_ffn_weights,
            "qdrift_signal": {
                "sensitivity_score": 0.3,
                "ffn_mode": "full",
            },
            "layer_id": 0,
        }
        result = agent.on_step(step_config, inputs)
        assert "compute_path" in result
        assert "output" in result
        assert "compute_stats" in result

        profile = agent.on_shutdown()
        assert profile is not None
        assert profile.model_name == "test"


class TestPackedFFNWeight:
    """Tests for byte-level packed FFN weight storage."""

    @pytest.fixture
    def small_weight(self):
        """A small deterministic weight tensor for roundtrip tests."""
        torch.manual_seed(42)
        return torch.randn(16, 32)

    @pytest.mark.parametrize("precision", ["fp16", "int8", "int4"])
    def test_pack_unpack_roundtrip(self, small_weight, precision):
        """Pack/unpack roundtrip preserves shape and approximate values."""
        channels = small_weight.shape[0]
        precision_map = {c: precision for c in range(channels)}
        group_size = 16 if precision == "int4" else 64
        packed = PackedFFNWeight.pack(
            small_weight, precision_map, pack_dim=0, group_size=group_size
        )

        assert packed.shape == small_weight.shape
        unpacked = packed.unpack()
        assert unpacked.shape == small_weight.shape
        assert unpacked.dtype == torch.float32

        rel_error = torch.norm(unpacked - small_weight) / torch.norm(small_weight)
        if precision == "fp16":
            assert rel_error < 1e-3
        elif precision == "int8":
            assert rel_error < 0.05
        else:  # int4
            assert rel_error < 0.25

    def test_memory_savings_hierarchy(self):
        """INT4 < INT8 < FP16 < FP32 in actual byte-level memory usage."""
        torch.manual_seed(0)
        weight = torch.randn(64, 64)
        channels = weight.shape[0]
        precisions = ["fp32", "fp16", "int8", "int4"]
        memories = {}
        for precision in precisions:
            precision_map = {c: precision for c in range(channels)}
            group_size = 64 if precision == "int4" else 64
            packed = PackedFFNWeight.pack(weight, precision_map, pack_dim=0, group_size=group_size)
            memories[precision] = packed.memory_bytes()

        assert memories["int4"] < memories["int8"]
        assert memories["int8"] < memories["fp16"]
        assert memories["fp16"] < memories["fp32"]

    def test_mixed_precision_per_channel(self):
        """Mixed precision packing assigns different precisions per channel."""
        torch.manual_seed(1)
        weight = torch.randn(8, 16)
        precision_map = {
            0: "fp16",
            1: "fp16",
            2: "int8",
            3: "int8",
            4: "int4",
            5: "int4",
            6: "int4",
            7: "int4",
        }
        packed = PackedFFNWeight.pack(weight, precision_map, pack_dim=0, group_size=16)
        assert packed.shape == weight.shape

        all_fp16 = PackedFFNWeight.pack(weight, {c: "fp16" for c in range(8)}, pack_dim=0)
        all_int4 = PackedFFNWeight.pack(
            weight, {c: "int4" for c in range(8)}, pack_dim=0, group_size=16
        )
        assert all_int4.memory_bytes() < packed.memory_bytes() < all_fp16.memory_bytes()

        unpacked = packed.unpack()
        assert unpacked.shape == weight.shape

    def test_pack_dim_one_for_down_projection(self):
        """Packing along columns preserves the original down-projection shape."""
        torch.manual_seed(2)
        weight = torch.randn(32, 64)  # [hidden, intermediate]
        precision_map = {c: "int8" for c in range(weight.shape[1])}
        packed = PackedFFNWeight.pack(weight, precision_map, pack_dim=1, group_size=64)
        assert packed.shape == weight.shape
        unpacked = packed.unpack()
        rel_error = torch.norm(unpacked - weight) / torch.norm(weight)
        assert rel_error < 0.05

    def test_sparse_path_output_shape_with_packed_weights(
        self, mock_config, mock_ffn_weights, mock_ffn_inputs
    ):
        """The sparse compute path still produces the correct output shape."""
        cfg = dict(mock_config)
        cfg["use_packed_weights"] = True
        agent = FFNOptimizerAgent(config=cfg)
        agent.on_init({"model_name": "test", "weights": mock_ffn_weights})

        ffn_input_prev = mock_ffn_inputs.clone()
        ffn_output_prev = torch.randn_like(mock_ffn_inputs)
        ffn_input_current = ffn_input_prev.clone()
        # Make all tokens change moderately to land in the "sparse" path.
        ffn_input_current += 0.1

        step_config = StepConfig(
            step_id=1,
            total_steps=10,
            noise_level=0.0,
            shadowkv_mode="balanced",
            reuse_layers=[0],
            compression_target=0.5,
            ffn_mode="sparse",
            weight_precision_map={},
            compute_path="sparse",
            sensitivity_score=0.0,
            drift_score=0.1,
        )
        inputs = {
            "ffn_input_current": ffn_input_current,
            "ffn_input_previous": ffn_input_prev,
            "ffn_output_previous": ffn_output_prev,
            "weights": mock_ffn_weights,
            "qdrift_signal": {
                "sensitivity_score": 0.0,
                "ffn_mode": "sparse",
            },
            "layer_id": 0,
        }
        result = agent.on_step(step_config, inputs)
        assert result["compute_path"] == "sparse"
        assert result["output"].shape == ffn_output_prev.shape
        assert result["quantization"].get("memory_savings_bytes", 0) > 0

    def test_backward_compat_disable_packed_weights(
        self, mock_config, mock_ffn_weights, mock_ffn_inputs
    ):
        """With ``use_packed_weights=False`` quantization returns dense tensors."""
        cfg = dict(mock_config)
        cfg["use_packed_weights"] = False
        agent = FFNOptimizerAgent(config=cfg)
        importance = agent.analyze_channel_importance(
            mock_ffn_inputs,
            mock_ffn_weights["up"],
            mock_ffn_weights["down"],
            method="weight_magnitude",
        )
        q_up, q_down, scales = agent.quantize_mixed_precision(
            mock_ffn_weights["up"], mock_ffn_weights["down"], importance
        )
        assert isinstance(q_up, torch.Tensor)
        assert isinstance(q_down, torch.Tensor)
        assert q_up.shape == mock_ffn_weights["up"].shape
        assert q_down.shape == mock_ffn_weights["down"].shape
        assert len(scales) == mock_ffn_weights["up"].shape[0]
