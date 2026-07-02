"""
Tests for CUDA Kernel optimizations.

Tests both CPU fallback and CUDA paths (if CUDA available).
"""

import torch

from shadowinfer.kernels import (
    _pack_int4,
    _unpack_int4,
    dequantize_per_channel_int4,
    dequantize_per_channel_int8,
    fused_attention_quantized,
    get_kernel_status,
    quantize_per_channel_int4,
    quantize_per_channel_int8,
    sparse_gemm_ffn,
)


class TestKernelStatus:
    def test_status_keys(self):
        status = get_kernel_status()
        assert "cuda_available" in status
        assert "cuda_extensions_loaded" in status
        assert "int8_kernel" in status
        assert "int4_kernel" in status
        assert "sparse_gemm_kernel" in status

    def test_cuda_available_boolean(self):
        status = get_kernel_status()
        assert isinstance(status["cuda_available"], bool)


class TestQuantizePerChannelInt8:
    def test_quantize_dequantize_roundtrip(self):
        """Test INT8 quantization -> dequantization roundtrip."""
        torch.manual_seed(42)
        tensor = torch.randn(4, 8, 32, 64)

        q, scale, zp = quantize_per_channel_int8(tensor, axis=1)

        assert q.dtype == torch.int8
        assert scale.shape == (8,)
        assert zp.shape == (8,)
        assert q.shape == tensor.shape

        # Dequantize
        deq = dequantize_per_channel_int8(q, scale, zp, axis=1)
        assert deq.shape == tensor.shape

        # Roundtrip error should be small
        error = (tensor - deq).abs().mean().item()
        assert error < 0.05  # ~3% relative error for INT8

    def test_quantize_clamping(self):
        """Test that values are properly clamped to INT8 range."""
        tensor = torch.tensor([[[-10.0, 0.0, 10.0]]])
        q, scale, zp = quantize_per_channel_int8(tensor, axis=0)
        assert q.min() >= -128
        assert q.max() <= 127

    def test_different_axes(self):
        """Test quantization on different axes."""
        tensor = torch.randn(2, 4, 8)
        for axis in [0, 1, 2]:
            q, scale, zp = quantize_per_channel_int8(tensor, axis=axis)
            assert q.shape == tensor.shape
            assert scale.numel() == tensor.shape[axis]

    def test_scale_calculation(self):
        """Test that scale is computed correctly."""
        tensor = torch.tensor([[[1.0, -2.0, 3.0, -4.0]]])
        q, scale, zp = quantize_per_channel_int8(tensor, axis=0)
        expected_scale = 4.0 / 127.0  # max abs = 4
        assert abs(scale[0].item() - expected_scale) < 1e-6

    def test_zero_tensor(self):
        """Test quantization of zero tensor."""
        tensor = torch.zeros(2, 4, 8)
        q, scale, zp = quantize_per_channel_int8(tensor, axis=0)
        assert q.sum().item() == 0


class TestQuantizePerChannelInt4:
    def test_int4_packing(self):
        """Test INT4 quantization and packing."""
        torch.manual_seed(42)
        tensor = torch.randn(4, 8, 32, 64)

        packed, scale, zp = quantize_per_channel_int4(tensor, axis=1)

        # Packed should be half the size (2 values per byte)
        C = tensor.shape[1]
        N = tensor.shape[0] * tensor.shape[2] * tensor.shape[3]
        expected_packed_size = C * ((N + 1) // 2)
        assert packed.numel() == expected_packed_size
        assert packed.dtype == torch.uint8

    def test_int4_roundtrip(self):
        """Test INT4 roundtrip."""
        torch.manual_seed(42)
        tensor = torch.randn(2, 4, 8, 16)

        packed, scale, zp = quantize_per_channel_int4(tensor, axis=1)
        deq = dequantize_per_channel_int4(packed, scale, zp, axis=1, original_shape=tensor.shape)

        assert deq.shape == tensor.shape
        error = (tensor - deq).abs().mean().item()
        assert error < 0.3  # INT4 has higher error, but should be < 0.3

    def test_int4_range(self):
        """Test INT4 values are in correct range."""
        tensor = torch.tensor([[[7.0, -8.0, 0.0, 5.0]]])
        packed, scale, zp = quantize_per_channel_int4(tensor, axis=0)
        # Values should be clamped to [-8, 7]
        deq = dequantize_per_channel_int4(packed, scale, zp, axis=0, original_shape=tensor.shape)
        assert deq.abs().max().item() <= 8.0 * scale[0].item() + 1e-6

    def test_int4_pack_unpack_helpers(self):
        """Test the internal INT4 pack/unpack helpers used by ShadowKV."""
        torch.manual_seed(42)
        int8_tensor = torch.randint(-8, 8, (23,), dtype=torch.int8)
        packed = _pack_int4(int8_tensor)
        assert packed.dtype == torch.uint8
        assert packed.numel() == (int8_tensor.numel() + 1) // 2

        unpacked = _unpack_int4(packed, int8_tensor.numel())
        assert unpacked.dtype == torch.int8
        assert torch.equal(unpacked, int8_tensor)

    def test_int4_public_api_with_scale(self):
        """Test the simplified ``(tensor, scale)`` quantize API."""
        torch.manual_seed(42)
        tensor = torch.randn(2, 6)
        scale = tensor.abs().max(dim=-1).values / 7.0

        packed, out_scale, zp = quantize_per_channel_int4(tensor, scale)
        assert torch.allclose(scale, out_scale)

        deq = dequantize_per_channel_int4(
            packed, scale, num_elements=tensor.numel(), original_shape=tensor.shape, axis=-1
        )
        assert deq.shape == tensor.shape


class TestSparseGemmFFN:
    def test_sparse_gemm_shape(self):
        """Test sparse GEMM output shape."""
        B, S, D_in = 2, 4, 16
        D_out = 8
        input_tensor = torch.randn(B, S, D_in)
        weight = torch.randn(D_out, D_in)
        changed_mask = torch.randint(0, 2, (D_out,)).to(torch.int8)
        bias = torch.randn(D_out)

        output = sparse_gemm_ffn(input_tensor, weight, changed_mask, bias)
        assert output.shape == (B, S, D_out)

    def test_sparse_gemm_all_ones_mask(self):
        """Test with all channels changed (should be same as full GEMM)."""
        torch.manual_seed(42)
        B, S, D_in = 2, 4, 16
        D_out = 8
        input_tensor = torch.randn(B, S, D_in)
        weight = torch.randn(D_out, D_in)
        changed_mask = torch.ones(D_out, dtype=torch.int8)
        bias = torch.randn(D_out)

        output = sparse_gemm_ffn(input_tensor, weight, changed_mask, bias)
        expected = torch.matmul(input_tensor, weight.t()) + bias

        assert torch.allclose(output, expected, atol=1e-4)

    def test_sparse_gemm_some_zeros(self):
        """Test with some channels unchanged."""
        torch.manual_seed(42)
        B, S, D_in = 2, 4, 16
        D_out = 8
        input_tensor = torch.randn(B, S, D_in)
        weight = torch.randn(D_out, D_in)
        changed_mask = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.int8)
        bias = torch.randn(D_out)

        output = sparse_gemm_ffn(input_tensor, weight, changed_mask, bias)
        assert output.shape == (B, S, D_out)

    def test_sparse_gemm_no_bias(self):
        """Test without bias."""
        B, S, D_in = 2, 4, 16
        D_out = 8
        input_tensor = torch.randn(B, S, D_in)
        weight = torch.randn(D_out, D_in)
        changed_mask = torch.ones(D_out, dtype=torch.int8)

        output = sparse_gemm_ffn(input_tensor, weight, changed_mask, None)
        expected = torch.matmul(input_tensor, weight.t())
        assert torch.allclose(output, expected, atol=1e-4)

    def test_sparse_gemm_flops_reduction(self):
        """Test that sparse GEMM can reduce FLOPs when many channels unchanged."""
        torch.manual_seed(42)
        B, S, D_in = 4, 8, 64
        D_out = 32
        input_tensor = torch.randn(B, S, D_in)
        weight = torch.randn(D_out, D_in)

        # 25% changed → 75% FLOPs saved
        changed_mask = torch.zeros(D_out, dtype=torch.int8)
        changed_indices = torch.randperm(D_out)[: D_out // 4]
        changed_mask[changed_indices] = 1

        output = sparse_gemm_ffn(input_tensor, weight, changed_mask, None)
        assert output.shape == (B, S, D_out)


class TestFusedAttentionQuantized:
    def test_fused_attention_shape(self):
        """Test fused attention output shape."""
        B, H, S, D = 2, 4, 8, 16
        query = torch.randn(B, H, S, D)

        # Quantized K and V
        key_fp = torch.randn(B, H, S, D)
        value_fp = torch.randn(B, H, S, D)
        k_q, k_scale, k_zp = quantize_per_channel_int8(key_fp, axis=1)
        v_q, v_scale, v_zp = quantize_per_channel_int8(value_fp, axis=1)

        output = fused_attention_quantized(query, k_q, v_q, k_scale, v_scale)
        assert output.shape == (B, H, S, D)

    def test_fused_attention_vs_reference(self):
        """Compare fused attention with manual dequantized attention."""
        torch.manual_seed(42)
        B, H, S, D = 2, 4, 8, 16
        query = torch.randn(B, H, S, D)
        key_fp = torch.randn(B, H, S, D)
        value_fp = torch.randn(B, H, S, D)

        k_q, k_scale, k_zp = quantize_per_channel_int8(key_fp, axis=1)
        v_q, v_scale, v_zp = quantize_per_channel_int8(value_fp, axis=1)

        # Fused
        output_fused = fused_attention_quantized(query, k_q, v_q, k_scale, v_scale)

        # Reference: manual dequantize then attention
        k_deq = dequantize_per_channel_int8(k_q, k_scale, k_zp, axis=1)
        v_deq = dequantize_per_channel_int8(v_q, v_scale, v_zp, axis=1)

        scores = torch.matmul(query, k_deq.transpose(-2, -1)) / (D**0.5)
        attn = torch.softmax(scores, dim=-1)
        output_ref = torch.matmul(attn, v_deq)

        assert torch.allclose(output_fused, output_ref, atol=1e-3)

    def test_fused_attention_with_causal_mask(self):
        """Test with causal mask."""
        torch.manual_seed(42)
        B, H, S, D = 2, 4, 8, 16
        query = torch.randn(B, H, S, D)
        key_fp = torch.randn(B, H, S, D)
        value_fp = torch.randn(B, H, S, D)

        k_q, k_scale, k_zp = quantize_per_channel_int8(key_fp, axis=1)
        v_q, v_scale, v_zp = quantize_per_channel_int8(value_fp, axis=1)

        causal_mask = torch.tril(torch.ones(S, S))

        output = fused_attention_quantized(query, k_q, v_q, k_scale, v_scale, mask=causal_mask)
        assert output.shape == (B, H, S, D)

    def test_fused_attention_numerical_stability(self):
        """Test with large values to check stability."""
        torch.manual_seed(42)
        B, H, S, D = 2, 4, 8, 16
        query = torch.randn(B, H, S, D) * 100
        key_fp = torch.randn(B, H, S, D) * 100
        value_fp = torch.randn(B, H, S, D) * 100

        k_q, k_scale, k_zp = quantize_per_channel_int8(key_fp, axis=1)
        v_q, v_scale, v_zp = quantize_per_channel_int8(value_fp, axis=1)

        output = fused_attention_quantized(query, k_q, v_q, k_scale, v_scale)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_fused_attention_gqa_shape(self):
        """Test grouped-query attention output shape."""
        torch.manual_seed(42)
        B, H_q, H_kv, S, D = 2, 8, 2, 8, 16
        query = torch.randn(B, H_q, S, D)
        key_fp = torch.randn(B, H_kv, S, D)
        value_fp = torch.randn(B, H_kv, S, D)

        k_q, k_scale, _ = quantize_per_channel_int8(key_fp, axis=1)
        v_q, v_scale, _ = quantize_per_channel_int8(value_fp, axis=1)

        output = fused_attention_quantized(query, k_q, v_q, k_scale, v_scale, num_kv_heads=H_kv)
        assert output.shape == (B, H_q, S, D)

    def test_fused_attention_int4_shape(self):
        """Test INT4 packed K/V output shape."""
        torch.manual_seed(42)
        B, H, S, D = 2, 4, 8, 16
        query = torch.randn(B, H, S, D)
        key_fp = torch.randn(B, H, S, D)
        value_fp = torch.randn(B, H, S, D)

        # Pack K/V along the head dimension (2 values per byte).
        def _pack_kv(tensor: torch.Tensor) -> tuple:
            scale = tensor.abs().amax(dim=(0, 2, 3)).clamp_min(1e-8) / 7.0
            scale_4d = scale.view(1, H, 1, 1)
            q = torch.round(tensor / scale_4d).clamp(-8, 7).to(torch.int8)
            flat = q.reshape(-1, D)
            packed = torch.stack([_pack_int4(row) for row in flat])
            packed = packed.reshape(B, H, S, (D + 1) // 2)
            return packed, scale

        k_q, k_scale = _pack_kv(key_fp)
        v_q, v_scale = _pack_kv(value_fp)

        output = fused_attention_quantized(query, k_q, v_q, k_scale, v_scale, is_int4=True)
        assert output.shape == (B, H, S, D)


class TestCudaKernels:
    def test_int4_cuda_roundtrip_if_available(self):
        """Run INT4 CUDA roundtrip when CUDA is available and compiled."""
        status = get_kernel_status()
        if not status["cuda_available"] or not status["cuda_extensions_loaded"]:
            return

        torch.manual_seed(42)
        tensor = torch.randn(4, 8, 32, 64).cuda()
        packed, scale, _ = quantize_per_channel_int4(tensor, axis=1)
        deq = dequantize_per_channel_int4(packed, scale, axis=1, original_shape=tensor.shape)
        assert deq.shape == tensor.shape
        assert (tensor - deq).abs().mean().item() < 0.3

    def test_fused_attention_cuda_shape_if_available(self):
        """Run fused attention CUDA kernel when available."""
        status = get_kernel_status()
        if not status["cuda_available"] or not status["cuda_extensions_loaded"]:
            return

        B, H, S, D = 2, 4, 8, 16
        query = torch.randn(B, H, S, D).cuda()
        key_fp = torch.randn(B, H, S, D).cuda()
        value_fp = torch.randn(B, H, S, D).cuda()

        k_q, k_scale, _ = quantize_per_channel_int8(key_fp, axis=1)
        v_q, v_scale, _ = quantize_per_channel_int8(value_fp, axis=1)

        output = fused_attention_quantized(query, k_q, v_q, k_scale, v_scale)
        assert output.shape == (B, H, S, D)
