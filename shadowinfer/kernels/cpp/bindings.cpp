// ShadowInfer CUDA Kernels: Python Bindings
// ==========================================

#include <torch/extension.h>
#include <tuple>

// Forward declarations from .cu files
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> quantize_per_channel_int8_cuda(
    torch::Tensor input, int axis, int qmax, int qmin);
torch::Tensor dequantize_per_channel_int8_cuda(
    torch::Tensor input, torch::Tensor scale, torch::Tensor zero_point, int axis);
torch::Tensor sparse_gemm_ffn_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor changed_mask,
    c10::optional<torch::Tensor> bias);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> quantize_per_channel_int4_cuda(
    torch::Tensor input, int axis, int qmax, int qmin);
torch::Tensor dequantize_per_channel_int4_cuda(
    torch::Tensor input, torch::Tensor scale, torch::Tensor zero_point, int axis,
    c10::optional<std::vector<int64_t>> output_shape, int64_t num_elements);
torch::Tensor pack_int4_cuda(torch::Tensor input);
torch::Tensor unpack_int4_cuda(torch::Tensor input, int64_t num_elements);

torch::Tensor fused_attention_quantized_cuda(
    torch::Tensor query, torch::Tensor key, torch::Tensor value,
    torch::Tensor k_scale, torch::Tensor v_scale,
    c10::optional<torch::Tensor> mask, bool is_int4, int num_kv_heads);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_per_channel_int8_cuda", &quantize_per_channel_int8_cuda,
          "Per-channel INT8 quantization (CUDA)");
    m.def("dequantize_per_channel_int8_cuda", &dequantize_per_channel_int8_cuda,
          "Per-channel INT8 dequantization (CUDA)");
    m.def("sparse_gemm_ffn_cuda", &sparse_gemm_ffn_cuda,
          "Sparse GEMM for FFN (CUDA)");

    m.def("quantize_per_channel_int4_cuda", &quantize_per_channel_int4_cuda,
          "Per-channel INT4 quantization with 2x packing (CUDA)");
    m.def("dequantize_per_channel_int4_cuda", &dequantize_per_channel_int4_cuda,
          "Per-channel INT4 dequantization with unpacking (CUDA)");
    m.def("pack_int4_cuda", &pack_int4_cuda,
          "Pack int8-encoded INT4 values into uint8 (CUDA)");
    m.def("unpack_int4_cuda", &unpack_int4_cuda,
          "Unpack uint8 bytes back to int8-encoded INT4 values (CUDA)");

    m.def("fused_attention_quantized_cuda", &fused_attention_quantized_cuda,
          "Fused attention with INT8/INT4 quantized K/V cache (CUDA)");
}
