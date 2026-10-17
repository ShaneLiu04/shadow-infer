"""ShadowKV M1 热路径向量化 golden 回归测试（AR002 T004/T005）。

以现有循环实现为基准，验证向量化接口逐元素等价：
Quantizer.quantize_batch、ShadowKVAgent._compute_importance_matrix、
_compress_kv_vectorized、allocate_precision、decide_reuse_per_token、
PackedKVCache._do_pack；向量化主路径异常时 compress_kv 回退 reference。
"""

import time

import pytest
import torch

from shadowinfer.shadowkv import ShadowKVAgent
from shadowinfer.shadowkv.packed_kv_cache import PackedKVCache
from shadowinfer.utils.quantization import Quantizer

PRECISIONS = ("fp32", "fp16", "int8", "int4")


class TestM1Vectorization:
    @staticmethod
    def _allocate_precision_baseline(importance, mode):
        mode_adjustment = {
            "aggressive": -0.1,
            "balanced": 0.0,
            "conservative": 0.1,
        }[mode]
        thresholds = {
            "fp32": max(0.0, 0.8 + mode_adjustment),
            "fp16": max(0.0, 0.5 + mode_adjustment),
            "int8": max(0.0, 0.2 + mode_adjustment),
            "int4": 0.0,
        }
        seq_len, num_heads = importance.shape
        precision_map = {}
        for token_idx in range(seq_len):
            token_map = {}
            for head_idx in range(num_heads):
                score = float(importance[token_idx, head_idx].item())
                if score >= thresholds["fp32"]:
                    token_map[head_idx] = "fp32"
                elif score >= thresholds["fp16"]:
                    token_map[head_idx] = "fp16"
                elif score >= thresholds["int8"]:
                    token_map[head_idx] = "int8"
                else:
                    token_map[head_idx] = "int4"
            precision_map[token_idx] = token_map
        return precision_map

    @staticmethod
    def _extract_core(result):
        if isinstance(result, dict):
            k_out = result.get("k", result.get("k_out"))
            v_out = result.get("v", result.get("v_out"))
            return k_out, v_out, result["scale_k_map"], result["scale_v_map"]
        k_out, v_out, scale_k_map, scale_v_map = result
        return k_out, v_out, scale_k_map, scale_v_map

    @staticmethod
    def _assert_scale_equal(actual, expected):
        if expected is None:
            assert actual is None
            return
        assert actual is not None
        assert actual.shape == expected.shape
        assert torch.allclose(actual.float(), expected.float(), atol=1e-7)

    def _assert_compress_results_equal(self, actual, expected):
        k_vec, v_vec, scale_k_vec, scale_v_vec = self._extract_core(actual)
        k_ref, v_ref, scale_k_ref, scale_v_ref = self._extract_core(expected)
        assert torch.equal(k_vec, k_ref)
        assert torch.equal(v_vec, v_ref)
        assert set(scale_k_vec.keys()) == set(scale_k_ref.keys())
        assert set(scale_v_vec.keys()) == set(scale_v_ref.keys())
        for key, ref_scale in scale_k_ref.items():
            self._assert_scale_equal(scale_k_vec[key], ref_scale)
        for key, ref_scale in scale_v_ref.items():
            self._assert_scale_equal(scale_v_vec[key], ref_scale)
        if isinstance(actual, dict) and isinstance(expected, dict):
            assert set(actual.keys()) == set(expected.keys())
            for key, ref_value in expected.items():
                if key in ("k", "v", "k_out", "v_out", "scale_k_map", "scale_v_map"):
                    continue
                if isinstance(ref_value, (int, float)):
                    assert isinstance(actual[key], (int, float))
                    assert actual[key] == ref_value

    def test_quantize_batch_golden(self):
        """quantize_batch 与逐行 quantize_tensor 的整数码 bitwise 一致。"""
        torch.manual_seed(0)
        rows = 37

        tensor_int8 = torch.randn(rows, 8)
        tensor_int8[3] = 0.0
        batched_q, batched_scale = Quantizer.quantize_batch(tensor_int8, "int8")
        assert batched_q.dtype == torch.int8
        assert batched_scale.shape == (rows, 1)
        for i in range(rows):
            ref_q, ref_scale = Quantizer.quantize_tensor(tensor_int8[i : i + 1], "int8")
            assert torch.equal(batched_q[i], ref_q[0])
            assert torch.allclose(batched_scale[i], ref_scale[0], atol=1e-7)

        tensor_int4 = torch.randn(rows, 128)
        tensor_int4[5] = 0.0
        batched_q, batched_scale = Quantizer.quantize_batch(tensor_int4, "int4", group_size=64)
        assert batched_q.dtype == torch.int8
        assert batched_scale.shape == (rows, 2, 1)
        for i in range(rows):
            ref_q, ref_scale = Quantizer.quantize_tensor(
                tensor_int4[i : i + 1], "int4", group_size=64
            )
            assert torch.equal(batched_q[i].reshape(-1), ref_q.reshape(-1))
            assert torch.allclose(batched_scale[i], ref_scale[0], atol=1e-7)

        tensor_fp = torch.randn(rows, 8)
        quantized, scale = Quantizer.quantize_batch(tensor_fp, "fp32")
        assert torch.equal(quantized, tensor_fp)
        assert scale is None
        quantized, scale = Quantizer.quantize_batch(tensor_fp, "fp16")
        assert quantized.dtype == torch.float16
        assert torch.equal(quantized, tensor_fp.half())
        assert scale is None

    def test_importance_matrix_golden(self, mock_config):
        """_compute_importance_matrix 与逐 (token, head) 基准一致（含 batch=2）。"""
        agent = ShadowKVAgent(config=mock_config)
        torch.manual_seed(1)
        heads, seq_len = 8, 32
        for batch_size in (1, 2):
            scores = torch.softmax(torch.randn(batch_size, heads, seq_len, seq_len), dim=-1)
            golden = torch.zeros(seq_len, heads)
            for token_idx in range(seq_len):
                for head_idx in range(heads):
                    golden[token_idx, head_idx] = agent.compute_importance_score(
                        scores, token_idx, head_idx, 0, 4
                    )
            matrix = agent._compute_importance_matrix(scores, 0, 4)
            assert matrix.shape == (seq_len, heads)
            assert torch.allclose(matrix, golden, atol=1e-6)

        # 非方形 attention_scores（seq_q != seq_k）：token 维度为 shape[2]，
        # 归一化与位置阈值以 shape[-1] 为基准（与 reference 口径一致）
        torch.manual_seed(11)
        scores = torch.softmax(torch.randn(1, 4, 4, 8), dim=-1)
        golden = torch.zeros(4, 4)
        for token_idx in range(4):
            for head_idx in range(4):
                golden[token_idx, head_idx] = agent.compute_importance_score(
                    scores, token_idx, head_idx, 0, 4
                )
        matrix = agent._compute_importance_matrix(scores, 0, 4)
        assert matrix.shape == (4, 4)
        assert torch.allclose(matrix, golden, atol=1e-6)

    def test_allocate_precision_golden(self, mock_config):
        """allocate_precision 三种 mode 与独立阈值循环基准逐条相等。"""
        agent = ShadowKVAgent(config=mock_config)
        torch.manual_seed(2)
        importance = torch.rand(32, 8)
        importance[0, 0] = 0.95
        importance[0, 1] = 0.65
        importance[0, 2] = 0.35
        importance[0, 3] = 0.05
        reference = getattr(agent, "_allocate_precision_reference", None)
        for mode in ("aggressive", "balanced", "conservative"):
            baseline = self._allocate_precision_baseline(importance, mode)
            result = agent.allocate_precision(importance, mode)
            assert result == baseline
            if reference is not None:
                assert reference(importance, mode) == baseline

    def test_decide_reuse_golden(self, mock_config):
        """decide_reuse_per_token 与循环基准的 reuse_mask/stats 逐字段相等。"""
        agent = ShadowKVAgent(config=mock_config)
        torch.manual_seed(3)
        batch_size, heads, seq_len, head_dim = 2, 8, 32, 16
        k_current = torch.randn(batch_size, heads, seq_len, head_dim)
        v_current = torch.randn(batch_size, heads, seq_len, head_dim)
        k_previous = k_current + 0.1 * torch.randn_like(k_current)
        v_previous = v_current + 0.1 * torch.randn_like(v_current)
        importance = torch.rand(seq_len, heads)
        reference = getattr(agent, "_decide_reuse_per_token_reference", None)
        baseline_fn = reference if reference is not None else agent.decide_reuse_per_token
        for mode in ("balanced", "conservative"):
            mask_base, stats_base = baseline_fn(
                k_current,
                v_current,
                k_previous,
                v_previous,
                step_id=2,
                total_steps=10,
                qdrift_drift_score=0.2,
                mode=mode,
                importance_map=importance,
            )
            reuse_mask, stats = agent.decide_reuse_per_token(
                k_current,
                v_current,
                k_previous,
                v_previous,
                step_id=2,
                total_steps=10,
                qdrift_drift_score=0.2,
                mode=mode,
                importance_map=importance,
            )
            assert reuse_mask == mask_base
            assert stats == stats_base

    def test_compress_kv_golden(self, mock_config):
        """_compress_kv_vectorized 与 _compress_kv_reference 逐元素一致。"""
        agent = ShadowKVAgent(config=mock_config)
        vectorized = getattr(agent, "_compress_kv_vectorized", None)
        reference = getattr(agent, "_compress_kv_reference", None)
        assert vectorized is not None, "_compress_kv_vectorized 尚未实现"
        assert reference is not None, "_compress_kv_reference 尚未实现"

        torch.manual_seed(4)
        batch_size, heads, seq_len, head_dim = 2, 8, 32, 64
        k_current = torch.randn(batch_size, heads, seq_len, head_dim)
        v_current = torch.randn(batch_size, heads, seq_len, head_dim)
        k_previous = torch.randn(batch_size, heads, seq_len, head_dim)
        v_previous = torch.randn(batch_size, heads, seq_len, head_dim)

        precision_map = agent.allocate_precision(torch.rand(seq_len, heads), "balanced")
        for token_idx in range(0, seq_len, 8):
            precision_map[token_idx][0] = "fp32"
            precision_map[token_idx][1] = "fp16"
            precision_map[token_idx][2] = "int8"
        # reference 的 int4 写回在 batch>1 时 RuntimeError（预存缺陷），
        # batch=2 场景将自然分配的 int4 重映射为 fp16
        for t_map in precision_map.values():
            for h_idx, prec in t_map.items():
                if prec == "int4":
                    t_map[h_idx] = "fp16"

        partial_reuse = {
            (token_idx, head_idx): (token_idx + head_idx) % 3 == 0
            for token_idx in range(seq_len)
            for head_idx in range(heads)
        }

        for reuse_mask in (None, partial_reuse):
            result_vec = vectorized(
                k_current, v_current, k_previous, v_previous, precision_map, reuse_mask
            )
            result_ref = reference(
                k_current, v_current, k_previous, v_previous, precision_map, reuse_mask
            )
            self._assert_compress_results_equal(result_vec, result_ref)

        # int4 比对限制在 reference 可用包络：batch=1 且 head_dim==group_size=64
        # （reference 的 int4 写回在 batch>1 或多分组时 RuntimeError，预存缺陷）
        k1 = torch.randn(1, heads, seq_len, head_dim)
        v1 = torch.randn(1, heads, seq_len, head_dim)
        precision_int4 = {
            t: {h: ("int4" if (t + h) % 2 == 0 else "fp16") for h in range(heads)}
            for t in range(seq_len)
        }
        result_vec = vectorized(k1, v1, None, None, precision_int4, None)
        result_ref = reference(k1, v1, None, None, precision_int4, None)
        self._assert_compress_results_equal(result_vec, result_ref)

    def test_compress_kv_fallback(self, mock_config, monkeypatch):
        """向量化路径抛异常时 compress_kv 回退 reference 且结果正确。

        异常回退须记录 error 级告警（design 6.4 异常场景：张量路径
        失败 → log_alert error + reference 兜底）。
        """
        agent = ShadowKVAgent(config=mock_config)
        alerts = []
        monkeypatch.setattr(
            agent.logger,
            "log_alert",
            lambda level, msg, **kw: alerts.append((level, msg)),
        )
        torch.manual_seed(5)
        heads, seq_len, head_dim = 4, 8, 64
        k_current = torch.randn(1, heads, seq_len, head_dim)
        v_current = torch.randn(1, heads, seq_len, head_dim)
        precision_map = {t: {h: "fp16" for h in range(heads)} for t in range(seq_len)}
        kv_current = {"k": k_current, "v": v_current}
        expected = agent.compress_kv(kv_current, precision_map)

        def _raise_runtime_error(*args, **kwargs):
            raise RuntimeError("vectorized path failure injected")

        monkeypatch.setattr(agent, "_compress_kv_vectorized", _raise_runtime_error)
        result = agent.compress_kv(kv_current, precision_map)
        assert torch.equal(result["k"], expected["k"])
        assert torch.equal(result["v"], expected["v"])
        assert set(result["scale_k_map"].keys()) == set(expected["scale_k_map"].keys())
        assert result["compression_ratio"] == expected["compression_ratio"]
        assert result["memory_mb"] == expected["memory_mb"]
        # 异常回退记录 error 级告警
        assert len(alerts) == 1
        assert alerts[0][0] == "error"
        assert "vectorized" in alerts[0][1] or "fallback" in alerts[0][1].lower()

    def test_packed_pack_golden(self):
        """_do_pack 打包结构、内存字节数与 unpack 还原一致性。

        注：int4 往返误差校验使用 batch=1（现有 unpack 的 scale 索引在
        batch>1 的 int4 路径存在预存缺陷，vec/ref golden 不受影响）。
        """
        torch.manual_seed(6)
        heads, seq_len, head_dim = 8, 32, 16

        def _build_inputs(batch_size):
            k_tensor = torch.randn(batch_size, heads, seq_len, head_dim)
            v_tensor = torch.randn(batch_size, heads, seq_len, head_dim)
            mixed_map = {
                t: {h: PRECISIONS[(t + h) % 4] for h in range(heads)} for t in range(seq_len)
            }
            return k_tensor, v_tensor, mixed_map

        batch_size = 2
        k_current, v_current, precision_map = _build_inputs(batch_size)
        cache = PackedKVCache(batch_size, heads, seq_len, head_dim)
        cache._do_pack(k_current, v_current, precision_map, group_size=16)

        expected_bytes = 0
        for data in cache.packed_data.values():
            for field in ("k", "v", "scale_k", "scale_v"):
                tensor = data.get(field)
                if tensor is not None:
                    expected_bytes += tensor.numel() * tensor.element_size()
        assert cache.memory_bytes() == expected_bytes

        for precision, data in cache.packed_data.items():
            expected_indices = [
                (t, h)
                for t in range(seq_len)
                for h in range(heads)
                if precision_map[t][h] == precision
            ]
            assert data["indices"] == expected_indices
            if precision == "int4":
                assert len(data["byte_offsets"]) == len(expected_indices)

        reference = getattr(cache, "_do_pack_reference", None)
        if reference is not None:
            cache_ref = PackedKVCache(batch_size, heads, seq_len, head_dim)
            cache_ref._do_pack_reference(k_current, v_current, precision_map, group_size=16)
            assert cache.memory_bytes() == cache_ref.memory_bytes()
            for precision, data_ref in cache_ref.packed_data.items():
                assert precision in cache.packed_data
                data_vec = cache.packed_data[precision]
                assert data_vec["indices"] == data_ref["indices"]
                if precision == "int4":
                    assert data_vec["byte_offsets"] == data_ref["byte_offsets"]
            k_back, v_back = cache.unpack()
            k_ref_back, v_ref_back = cache_ref.unpack()
            assert torch.allclose(k_back, k_ref_back, atol=1e-6)
            assert torch.allclose(v_back, v_ref_back, atol=1e-6)

        batch_size = 1
        k_current, v_current, precision_map = _build_inputs(batch_size)
        cache = PackedKVCache(batch_size, heads, seq_len, head_dim)
        cache._do_pack(k_current, v_current, precision_map, group_size=16)
        k_back, v_back = cache.unpack()
        assert k_back.shape == k_current.shape
        assert v_back.shape == v_current.shape
        error_thresholds = {"fp32": 1e-6, "fp16": 1e-2, "int8": 1e-2, "int4": 0.15}
        for precision in PRECISIONS:
            entries = [
                (t, h)
                for t in range(seq_len)
                for h in range(heads)
                if precision_map[t][h] == precision
            ]
            for original, restored in ((k_current, k_back), (v_current, v_back)):
                orig = torch.cat([original[:, h, t, :].reshape(-1) for t, h in entries])
                rest = torch.cat([restored[:, h, t, :].reshape(-1) for t, h in entries])
                error = float(torch.norm(orig - rest) / torch.norm(orig))
                assert error < error_thresholds[precision]

    @pytest.mark.slow
    def test_compress_kv_perf_5x(self, mock_config):
        """向量化压缩较 reference 循环耗时下降 >= 5 倍（T005 证据）。"""
        agent = ShadowKVAgent(config=mock_config)
        vectorized = getattr(agent, "_compress_kv_vectorized", None)
        reference = getattr(agent, "_compress_kv_reference", None)
        assert vectorized is not None, "_compress_kv_vectorized 尚未实现"
        assert reference is not None, "_compress_kv_reference 尚未实现"

        torch.manual_seed(7)
        batch_size, heads, seq_len, head_dim = 1, 32, 128, 128
        k_current = torch.randn(batch_size, heads, seq_len, head_dim)
        v_current = torch.randn(batch_size, heads, seq_len, head_dim)
        precision_map = agent.allocate_precision(torch.rand(seq_len, heads), "balanced")
        # int4 写回在 reference 中为预存缺陷路径（head_dim=128 多分组时
        # RuntimeError），比对场景将 int4 重映射为 fp16，int8 保留
        for t_map in precision_map.values():
            for h_idx, prec in t_map.items():
                if prec == "int4":
                    t_map[h_idx] = "fp16"
        args = (k_current, v_current, None, None, precision_map, None)

        start = time.perf_counter()
        reference(*args)
        reference_time = time.perf_counter() - start

        start = time.perf_counter()
        vectorized(*args)
        vectorized_time = time.perf_counter() - start

        assert (
            vectorized_time * 5 <= reference_time
        ), f"vectorized {vectorized_time:.4f}s vs reference {reference_time:.4f}s"
