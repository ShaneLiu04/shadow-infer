// ShadowInfer CUDA Kernels: Per-Channel INT4 Quantization
// ========================================================
//
// Packs two signed INT4 values into one uint8 byte.
// High 4 bits = first value, low 4 bits = second value.
// Odd element counts are handled by zero-padding the low nibble of the
// final byte.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

__forceinline__ __device__ int8_t int4_nibble_to_signed(uint8_t nibble) {
    int8_t v = static_cast<int8_t>(nibble & 0x0F);
    if (v > 7) v -= 16;
    return v;
}

__forceinline__ __device__ int8_t uint4_nibble_to_signed(uint8_t nibble) {
    // Offset encoding used by shadowinfer.utils.Quantizer: 0..15 maps to -8..7.
    return static_cast<int8_t>(static_cast<int16_t>(nibble & 0x0F) - 8);
}

// ---------------------------------------------------------------------------
// Per-channel INT4 quantization + packing
// Input:  [N, C] FP16/FP32 tensor (C = channel dim)
// Output: [(N+1)/2, C] uint8 packed tensor + [C] scale + [C] zero_point
// ---------------------------------------------------------------------------

template <typename T>
__global__ void quantize_per_channel_int4_kernel(
    const T* __restrict__ input,
    uint8_t* __restrict__ output,
    float* __restrict__ scale,
    float* __restrict__ zero_point,
    int N,
    int C,
    float qmax
) {
    int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= C) return;

    // Per-channel max absolute value.
    float local_max = 0.0f;
    for (int i = 0; i < N; ++i) {
        float val = static_cast<float>(input[i * C + channel]);
        local_max = fmaxf(local_max, fabsf(val));
    }

    float s = fmaxf(local_max / qmax, 1e-8f);
    scale[channel] = s;
    zero_point[channel] = 0.0f;

    int packed_N = (N + 1) / 2;
    for (int i = 0; i < packed_N; ++i) {
        int idx0 = (2 * i) * C + channel;
        int idx1 = (2 * i + 1) * C + channel;

        int8_t v0 = 0;
        int8_t v1 = 0;
        if (2 * i < N) {
            int q = static_cast<int>(roundf(static_cast<float>(input[idx0]) / s));
            q = max(-8, min(7, q));
            v0 = static_cast<int8_t>(q);
        }
        if (2 * i + 1 < N) {
            int q = static_cast<int>(roundf(static_cast<float>(input[idx1]) / s));
            q = max(-8, min(7, q));
            v1 = static_cast<int8_t>(q);
        }

        // High nibble stores the first value, low nibble the second.
        uint8_t byte = (static_cast<uint8_t>(v0 & 0x0F) << 4) |
                       (static_cast<uint8_t>(v1 & 0x0F) & 0x0F);
        output[i * C + channel] = byte;
    }
}

// ---------------------------------------------------------------------------
// Per-channel INT4 dequantization + unpacking
// Input:  [(N+1)/2, C] uint8 packed tensor + [C] scale + [C] zero_point
// Output: [N, C] FP16/FP32 tensor
// ---------------------------------------------------------------------------

template <typename T>
__global__ void dequantize_per_channel_int4_kernel(
    const uint8_t* __restrict__ input,
    T* __restrict__ output,
    const float* __restrict__ scale,
    const float* __restrict__ zero_point,
    int N,
    int C
) {
    int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= C) return;

    float s = scale[channel];
    float zp = zero_point[channel];
    int total = N * C;
    int packed_N = (N + 1) / 2;

    for (int i = 0; i < packed_N; ++i) {
        uint8_t byte = input[i * C + channel];
        int8_t v0 = int4_nibble_to_signed(byte >> 4);
        int8_t v1 = int4_nibble_to_signed(byte);

        int idx0 = (2 * i) * C + channel;
        if (idx0 < total) {
            output[idx0] = static_cast<T>((static_cast<float>(v0) - zp) * s);
        }
        int idx1 = idx0 + C;
        if (idx1 < total) {
            output[idx1] = static_cast<T>((static_cast<float>(v1) - zp) * s);
        }
    }
}

// ---------------------------------------------------------------------------
// Low-level INT4 pack/unpack helpers (offset encoding, compatible with
// shadowinfer.utils.Quantizer.pack_int4 / unpack_int4)
// ---------------------------------------------------------------------------

__global__ void pack_int4_kernel(
    const int8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int packed_N = (N + 1) / 2;
    if (i >= packed_N) return;

    int idx0 = 2 * i;
    int idx1 = 2 * i + 1;

    uint8_t v0 = (idx0 < N) ? static_cast<uint8_t>(static_cast<int16_t>(input[idx0]) + 8) : 0;
    uint8_t v1 = (idx1 < N) ? static_cast<uint8_t>(static_cast<int16_t>(input[idx1]) + 8) : 0;

    // High nibble = first value, low nibble = second value.
    output[i] = ((v0 & 0x0F) << 4) | (v1 & 0x0F);
}

__global__ void unpack_int4_kernel(
    const uint8_t* __restrict__ input,
    int8_t* __restrict__ output,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int byte_idx = i / 2;
    bool is_high = (i % 2 == 0);
    uint8_t byte = input[byte_idx];
    uint8_t nibble = is_high ? (byte >> 4) : (byte & 0x0F);

    output[i] = static_cast<int8_t>(static_cast<int16_t>(nibble) - 8);
}

// ---------------------------------------------------------------------------
// C++ bindings
// ---------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> quantize_per_channel_int4_cuda(
    torch::Tensor input,
    int axis,
    int qmax,
    int qmin
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "quantize_per_channel_int4_cuda expects a 2D [N, C] input");

    int N = input.size(0);
    int C = input.size(1);

    auto packed = torch::empty(
        {(N + 1) / 2, C},
        torch::dtype(torch::kUInt8).device(input.device())
    );
    auto scale = torch::empty(
        {C},
        torch::dtype(torch::kFloat32).device(input.device())
    );
    auto zero_point = torch::zeros(
        {C},
        torch::dtype(torch::kFloat32).device(input.device())
    );

    int threads = 256;
    int blocks = (C + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "quantize_int4", [&] {
        quantize_per_channel_int4_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            packed.data_ptr<uint8_t>(),
            scale.data_ptr<float>(),
            zero_point.data_ptr<float>(),
            N, C, static_cast<float>(qmax)
        );
    });

    return {packed, scale, zero_point};
}

torch::Tensor dequantize_per_channel_int4_cuda(
    torch::Tensor input,
    torch::Tensor scale,
    torch::Tensor zero_point,
    int axis,
    c10::optional<std::vector<int64_t>> output_shape,
    int64_t num_elements
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "dequantize_per_channel_int4_cuda expects a 2D [N_packed, C] input");

    int packed_N = input.size(0);
    int C = input.size(1);
    int N = num_elements > 0 ? num_elements : packed_N * 2;

    std::vector<int64_t> out_shape = output_shape.value_or(std::vector<int64_t>{N, C});
    auto output = torch::empty(
        out_shape,
        torch::dtype(torch::kFloat16).device(input.device())
    );

    int threads = 256;
    int blocks = (C + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output.scalar_type(), "dequantize_int4", [&] {
        dequantize_per_channel_int4_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<uint8_t>(),
            output.data_ptr<scalar_t>(),
            scale.data_ptr<float>(),
            zero_point.data_ptr<float>(),
            N, C
        );
    });

    return output;
}

torch::Tensor pack_int4_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kInt8, "Input must be int8");

    int N = input.numel();
    auto output = torch::empty(
        {(N + 1) / 2},
        torch::dtype(torch::kUInt8).device(input.device())
    );

    int threads = 256;
    int blocks = ((N + 1) / 2 + threads - 1) / threads;

    pack_int4_kernel<<<blocks, threads>>>(
        input.data_ptr<int8_t>(),
        output.data_ptr<uint8_t>(),
        N
    );

    return output;
}

torch::Tensor unpack_int4_cuda(torch::Tensor input, int64_t num_elements) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "Input must be uint8");

    int N = num_elements > 0 ? num_elements : input.numel() * 2;
    auto output = torch::empty(
        {N},
        torch::dtype(torch::kInt8).device(input.device())
    );

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    unpack_int4_kernel<<<blocks, threads>>>(
        input.data_ptr<uint8_t>(),
        output.data_ptr<int8_t>(),
        N
    );

    return output;
}
