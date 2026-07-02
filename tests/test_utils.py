"""测试工具模块。"""

import math

import pytest
import torch

from shadowinfer.utils.memory_utils import MemoryTracker
from shadowinfer.utils.metrics import Metrics
from shadowinfer.utils.quantization import Quantizer


class TestQuantizer:
    def test_quantize_fp32(self):
        """验证 FP32 量化不修改数据。"""
        tensor = torch.randn(10, 20)
        q, scale = Quantizer.quantize_tensor(tensor, "fp32")
        assert torch.equal(q, tensor)
        assert scale is None

    def test_quantize_fp16(self):
        """验证 FP16 量化转换为 half 精度。"""
        tensor = torch.randn(10, 20)
        q, scale = Quantizer.quantize_tensor(tensor, "fp16")
        assert q.dtype == torch.float16
        assert scale is None

    def test_quantize_int8(self):
        """验证 INT8 量化为 int8 类型。"""
        tensor = torch.randn(10, 20)
        q, scale = Quantizer.quantize_tensor(tensor, "int8")
        assert q.dtype == torch.int8
        assert scale is not None

    def test_quantize_int4(self):
        """验证 INT4 量化为 int8 存储（每组量化）。"""
        tensor = torch.randn(10, 64)
        q, scale = Quantizer.quantize_tensor(tensor, "int4")
        assert q.dtype == torch.int8
        assert scale is not None

    def test_roundtrip_error(self):
        """验证量化-反量化误差在可接受范围。"""
        tensor = torch.randn(10, 128)
        for precision in ["int8", "int4"]:
            q, scale = Quantizer.quantize_tensor(tensor, precision)
            dq = Quantizer.dequantize_tensor(q, scale, precision)
            if precision == "int4":
                dq = dq.reshape(*tensor.shape)
            error = Quantizer.compute_quantization_error(tensor, dq)
            # INT4 误差较大，放宽阈值到 15%
            threshold = 0.15 if precision == "int4" else 0.05
            assert (
                error < threshold
            ), f"Precision {precision} roundtrip error {error:.4f} >= {threshold}"

    def test_unsupported_precision(self):
        """验证不支持的精度抛出 ValueError。"""
        with pytest.raises(ValueError):
            Quantizer.quantize_tensor(torch.randn(10), "int2")

    def test_pack_unpack_int4(self):
        """验证 INT4 打包/解包一致性。"""
        tensor = torch.randint(-8, 8, (128,), dtype=torch.int8)
        packed = Quantizer.pack_int4(tensor)
        unpacked = Quantizer.unpack_int4(packed, tensor.numel())
        assert torch.equal(tensor, unpacked)

    def test_dequantize_requires_scale(self):
        """验证 INT8/INT4 反量化需要 scale。"""
        with pytest.raises(ValueError):
            Quantizer.dequantize_tensor(torch.randint(0, 10, (10,), dtype=torch.int8), None, "int8")
        with pytest.raises(ValueError):
            Quantizer.dequantize_tensor(
                torch.randint(0, 10, (10, 2), dtype=torch.int8), None, "int4"
            )


class TestMetrics:
    def test_perplexity(self):
        """验证 perplexity 计算。"""
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))
        ppl = Metrics.compute_perplexity(logits, labels)
        assert ppl > 0.0
        assert isinstance(ppl, float)

    def test_bleu_score(self):
        """验证 BLEU 分数计算。"""
        ref = "the quick brown fox jumps over the lazy dog"
        cand = "the quick brown fox jumps over the lazy dog"
        bleu = Metrics.compute_bleu_score(ref, cand)
        assert 0.0 <= bleu <= 1.0
        assert bleu > 0.9  # 完全匹配应该接近 1

        cand2 = "completely different text"
        bleu2 = Metrics.compute_bleu_score(ref, cand2)
        assert bleu2 < 0.5

    def test_relative_error(self):
        """验证相对误差计算。"""
        a = torch.randn(10, 10)
        b = a + 0.01 * torch.randn_like(a)
        err = Metrics.compute_relative_error(a, b)
        assert 0.0 <= err < 1.0

        same = Metrics.compute_relative_error(a, a)
        assert same == pytest.approx(0.0, abs=1e-6)

    def test_cosine_similarity(self):
        """验证余弦相似度计算。"""
        a = torch.randn(10)
        b = a.clone()
        sim = Metrics.compute_cosine_similarity(a, b)
        assert sim == pytest.approx(1.0, abs=1e-5)

        orth = torch.randn(10)
        orth = orth - (orth.dot(a) / a.dot(a)) * a
        sim_orth = Metrics.compute_cosine_similarity(a, orth)
        assert abs(sim_orth) < 0.1

    def test_kl_divergence(self):
        """验证 KL 散度计算。"""
        p = torch.softmax(torch.randn(10), dim=-1)
        q = p.clone()
        kl = Metrics.compute_kl_divergence(p, q)
        assert kl == pytest.approx(0.0, abs=1e-5)

        q2 = torch.softmax(torch.randn(10), dim=-1)
        kl2 = Metrics.compute_kl_divergence(p, q2)
        assert kl2 >= 0.0

    def test_entropy(self):
        """验证 entropy 计算。"""
        uniform = torch.ones(10) / 10.0
        entropy = Metrics.compute_entropy(uniform)
        assert entropy == pytest.approx(math.log(10), abs=0.01)

        deterministic = torch.zeros(10)
        deterministic[0] = 1.0
        entropy_det = Metrics.compute_entropy(deterministic)
        assert entropy_det == pytest.approx(0.0, abs=0.01)

    def test_latency_stats(self):
        """验证延迟统计计算。"""
        latencies = [10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 100.0]
        stats = Metrics.compute_latency_stats(latencies)
        assert stats["mean"] == pytest.approx(33.7, abs=0.1)
        assert stats["median"] == 27.5
        assert stats["min"] == 10.0
        assert stats["max"] == 100.0
        assert stats["p95"] >= stats["p90"] if "p90" in stats else True
        assert stats["p99"] >= stats["p95"]
        assert stats["std"] > 0.0

    def test_compression_ratio(self):
        """验证压缩比计算。"""
        ratio = Metrics.compute_compression_ratio(1000, 500)
        assert ratio == 0.5
        ratio0 = Metrics.compute_compression_ratio(1000, 1000)
        assert ratio0 == 0.0

    def test_flops_macs(self):
        """验证 FLOPs/MACs 计算。"""
        macs = Metrics.compute_flops_macs(4096, 11008, 1, 128)
        assert macs == 4096 * 11008 * 1 * 128
        assert Metrics.compute_flops_macs(0, 11008, 1, 128) == 0

    def test_accuracy_drop(self):
        """验证 accuracy drop 计算。"""
        drop = Metrics.compute_accuracy_drop(0.95, 0.93)
        assert drop == pytest.approx(0.02, abs=1e-6)
        drop_neg = Metrics.compute_accuracy_drop(0.93, 0.95)
        assert drop_neg == pytest.approx(-0.02, abs=1e-6)


class TestMemoryTracker:
    def test_tensor_memory(self):
        """验证 tensor 内存计算。"""
        tensor = torch.randn(10, 20, dtype=torch.float32)
        mem = MemoryTracker.get_tensor_memory(tensor)
        assert mem == 10 * 20 * 4  # float32 = 4 bytes

        tensor_fp16 = torch.randn(10, 20, dtype=torch.float16)
        mem_fp16 = MemoryTracker.get_tensor_memory(tensor_fp16)
        assert mem_fp16 == 10 * 20 * 2

    def test_kv_cache_memory(self):
        """验证 KV cache 内存计算。"""
        kv_cache = {
            "k": torch.randn(1, 32, 128, 128, dtype=torch.float32),
            "v": torch.randn(1, 32, 128, 128, dtype=torch.float32),
        }
        total = MemoryTracker.get_kv_cache_memory(kv_cache)
        expected = 2 * (1 * 32 * 128 * 128 * 4)
        assert total == expected

    def test_empty_tensor(self):
        """验证空张量内存为 0。"""
        empty = torch.tensor([])
        assert MemoryTracker.get_tensor_memory(empty) == 0
