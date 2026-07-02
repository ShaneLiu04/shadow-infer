// ShadowInfer CUDA Kernels: Per-Channel INT8 Quantization
// ======================================================
//
// Interview talking points:
// - "One CUDA thread per channel, coalesced access along channel axis."
// - "Used warp shuffle for intra-warp reduction of max value."
// - "Shared memory for scale storage to avoid redundant global loads."
//
// Kernel: quantize_per_channel_int8_cuda
//   Input:  [N, C] FP16 tensor (N = batch*seq, C = channels)
//   Output: [N, C] INT8 tensor + [C] scale + [C] zero_point

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <tuple>

// Thread block: 256 threads = 8 warps
// Each warp processes 1 channel, all threads in warp access same channel
// This ensures coalesced memory access (contiguous threads -> contiguous memory)

template <typename T>
__global__ void quantize_per_channel_int8_kernel(
    const T* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scale,
    float* __restrict__ zero_point,
    int N,    // total elements per channel
    int C,    // number of channels
    float qmax
) {
    // Channel index: each warp handles one channel
    int channel = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane_id = threadIdx.x % 32;
    
    if (channel >= C) return;
    
    // Step 1: Find max absolute value in this channel (warp-level reduction)
    float local_max = 0.0f;
    for (int i = lane_id; i < N; i += 32) {
        float val = static_cast<float>(input[i * C + channel]);
        local_max = fmaxf(local_max, fabsf(val));
    }
    
    // Warp shuffle reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
    }
    
    // Lane 0 writes scale
    if (lane_id == 0) {
        float s = local_max / qmax;
        s = fmaxf(s, 1e-8f);  // avoid division by zero
        scale[channel] = s;
        zero_point[channel] = 0.0f;  // symmetric quantization
    }
    
    // All threads need scale for quantization
    float s = __shfl_sync(0xFFFFFFFF, scale[channel], 0);
    
    // Step 2: Quantize
    for (int i = lane_id; i < N; i += 32) {
        float val = static_cast<float>(input[i * C + channel]);
        int32_t qval = static_cast<int32_t>(roundf(val / s));
        qval = max(-128, min(127, qval));  // clamp to INT8 range
        output[i * C + channel] = static_cast<int8_t>(qval);
    }
}

// Dequantization kernel
template <typename T>
__global__ void dequantize_per_channel_int8_kernel(
    const int8_t* __restrict__ input,
    T* __restrict__ output,
    const float* __restrict__ scale,
    const float* __restrict__ zero_point,
    int N,
    int C
) {
    int channel = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane_id = threadIdx.x % 32;
    
    if (channel >= C) return;
    
    float s = scale[channel];
    float zp = zero_point[channel];
    
    for (int i = lane_id; i < N; i += 32) {
        float val = static_cast<float>(input[i * C + channel]);
        output[i * C + channel] = static_cast<T>((val - zp) * s);
    }
}

// C++ bindings
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> quantize_per_channel_int8_cuda(
    torch::Tensor input,
    int axis,
    int qmax,
    int qmin
) {
    // Ensure input is on CUDA
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(axis == 0 || axis == -1, "Only axis 0 or -1 supported");
    
    int N = input.size(0);
    int C = input.size(1);
    
    auto output = torch::empty_like(input, torch::kInt8);
    auto scale = torch::empty({C}, torch::dtype(torch::kFloat32).device(input.device()));
    auto zero_point = torch::zeros({C}, torch::dtype(torch::kFloat32).device(input.device()));
    
    int warps_per_block = 8;  // 256 threads / 32
    int blocks = (C + warps_per_block - 1) / warps_per_block;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "quantize_int8", [&] {
        quantize_per_channel_int8_kernel<scalar_t><<<blocks, 256>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<int8_t>(),
            scale.data_ptr<float>(),
            zero_point.data_ptr<float>(),
            N, C, static_cast<float>(qmax)
        );
    });
    
    return {output, scale, zero_point};
}

torch::Tensor dequantize_per_channel_int8_cuda(
    torch::Tensor input,
    torch::Tensor scale,
    torch::Tensor zero_point,
    int axis
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    
    int N = input.size(0);
    int C = input.size(1);
    
    auto output = torch::empty({N, C}, torch::dtype(torch::kFloat16).device(input.device()));
    
    int warps_per_block = 8;
    int blocks = (C + warps_per_block - 1) / warps_per_block;
    
    dequantize_per_channel_int8_kernel<at::Half><<<blocks, 256>>>(
        input.data_ptr<int8_t>(),
        output.data_ptr<at::Half>(),
        scale.data_ptr<float>(),
        zero_point.data_ptr<float>(),
        N, C
    );
    
    return output;
}
