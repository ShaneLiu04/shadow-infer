// ShadowInfer CUDA Kernels: Fused Quantized Attention
// ====================================================
//
// Reference CUDA skeleton that computes single-head attention with INT8 or
// INT4 K/V cache. The kernel dequantizes K/V on-the-fly while computing
// Q @ K^T and Softmax @ V.
//
// Supports grouped-query attention (GQA): num_kv_heads <= num_q_heads.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <float.h>
#include <limits>

// Dequantize a single K/V element.
// For INT8: layout is [B, H_kv, S, D] of int8 values.
// For INT4: layout is [B, H_kv, S, D_packed] of uint8 values where two
// adjacent head-dim values are packed into one byte (high nibble first).
template <bool IsInt4>
__forceinline__ __device__ float dequantize_kv_element(
    const void* kv_ptr,
    int d,
    int D,
    int D_packed,
    float scale
) {
    if (IsInt4) {
        const uint8_t* ptr = static_cast<const uint8_t*>(kv_ptr);
        int byte_idx = d / 2;
        bool is_high = (d % 2 == 0);
        uint8_t byte = ptr[byte_idx];
        uint8_t nibble = is_high ? (byte >> 4) : (byte & 0x0F);
        int8_t v = static_cast<int8_t>(nibble & 0x0F);
        if (v > 7) v -= 16;
        return static_cast<float>(v) * scale;
    } else {
        const int8_t* ptr = static_cast<const int8_t*>(kv_ptr);
        return static_cast<float>(ptr[d]) * scale;
    }
}

template <typename T, bool IsInt4>
__global__ void fused_attention_quantized_kernel(
    const T* __restrict__ query,
    const void* __restrict__ key,
    const void* __restrict__ value,
    T* __restrict__ output,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    const float* __restrict__ mask,
    int B,
    int H_q,
    int H_kv,
    int S_q,
    int S_kv,
    int D,
    int D_packed
) {
    // One block per (query_seq, query_head, batch).
    int sq = blockIdx.x;
    int qh = blockIdx.y;
    int b = blockIdx.z;
    int d = threadIdx.x;

    if (d >= D) return;

    // Map query head to KV head for GQA.
    int kv_head = qh * H_kv / H_q;

    // Strides.
    int q_stride = H_q * S_q * D;
    int qh_stride = S_q * D;
    int kv_stride = H_kv * S_kv * (IsInt4 ? D_packed : D);
    int kv_head_stride = S_kv * (IsInt4 ? D_packed : D);
    int kv_token_stride = IsInt4 ? D_packed : D;

    const T* q_ptr = query + b * q_stride + qh * qh_stride + sq * D;
    float qval = static_cast<float>(q_ptr[d]);

    float s_k = k_scale[kv_head];
    float s_v = v_scale[kv_head];

    // Shared memory layout: product buffer (D floats) + logits (S_kv floats)
    // + softmax max + softmax sum.
    extern __shared__ float smem[];
    float* prod = smem;                  // [D]
    float* logits = smem + D;            // [S_kv]
    float* smem_max = logits + S_kv;     // [1]
    float* smem_sum = smem_max + 1;      // [1]

    // -----------------------------------------------------------------------
    // 1. Compute Q @ K^T / sqrt(D) for all KV tokens, dequantizing K on-the-fly.
    // -----------------------------------------------------------------------
    const uint8_t* k_base = static_cast<const uint8_t*>(key) +
                            b * kv_stride + kv_head * kv_head_stride;

    for (int j = 0; j < S_kv; ++j) {
        const void* k_ptr = k_base + j * kv_token_stride;

        // Each thread computes the product for its head dimension.
        prod[d] = qval * dequantize_kv_element<IsInt4>(k_ptr, d, D, D_packed, s_k);
        __syncthreads();

        // Reduce products to a scalar dot product (thread 0 does it for simplicity).
        if (d == 0) {
            float dot = 0.0f;
            for (int dd = 0; dd < D; ++dd) {
                dot += prod[dd];
            }
            logits[j] = dot / sqrtf(static_cast<float>(D));
            if (mask != nullptr) {
                if (mask[sq * S_kv + j] == 0.0f) {
                    logits[j] = -std::numeric_limits<float>::infinity();
                }
            }
        }
        __syncthreads();
    }

    // -----------------------------------------------------------------------
    // 2. Softmax over logits (single-pass stable softmax).
    // -----------------------------------------------------------------------
    if (d == 0) {
        float max_val = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < S_kv; ++j) {
            max_val = fmaxf(max_val, logits[j]);
        }
        *smem_max = max_val;
    }
    __syncthreads();

    if (d == 0) {
        float sum = 0.0f;
        for (int j = 0; j < S_kv; ++j) {
            logits[j] = expf(logits[j] - *smem_max);
            sum += logits[j];
        }
        *smem_sum = fmaxf(sum, 1e-8f);
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // 3. Compute Softmax @ V, dequantizing V on-the-fly.
    // -----------------------------------------------------------------------
    const uint8_t* v_base = static_cast<const uint8_t*>(value) +
                            b * kv_stride + kv_head * kv_head_stride;

    float accum = 0.0f;
    for (int j = 0; j < S_kv; ++j) {
        const void* v_ptr = v_base + j * kv_token_stride;
        float v_deq = dequantize_kv_element<IsInt4>(v_ptr, d, D, D_packed, s_v);
        accum += logits[j] * v_deq;
    }
    accum /= *smem_sum;

    int out_idx = b * q_stride + qh * qh_stride + sq * D + d;
    output[out_idx] = static_cast<T>(accum);
}

// ---------------------------------------------------------------------------
// C++ binding
// ---------------------------------------------------------------------------

torch::Tensor fused_attention_quantized_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    c10::optional<torch::Tensor> mask,
    bool is_int4,
    int num_kv_heads
) {
    TORCH_CHECK(query.is_cuda(), "Query must be a CUDA tensor");
    TORCH_CHECK(key.is_cuda(), "Key must be a CUDA tensor");
    TORCH_CHECK(value.is_cuda(), "Value must be a CUDA tensor");

    int B = query.size(0);
    int H_q = query.size(1);
    int S_q = query.size(2);
    int D = query.size(3);

    int H_kv = num_kv_heads > 0 ? num_kv_heads : key.size(1);
    int S_kv = key.size(2);
    int D_packed = is_int4 ? (D + 1) / 2 : D;

    TORCH_CHECK(H_q % H_kv == 0,
                "num_q_heads (", H_q, ") must be divisible by num_kv_heads (", H_kv, ")");
    TORCH_CHECK(k_scale.numel() >= H_kv, "k_scale must have at least num_kv_heads elements");
    TORCH_CHECK(v_scale.numel() >= H_kv, "v_scale must have at least num_kv_heads elements");

    auto output = torch::empty_like(query);

    c10::optional<torch::Tensor> mask_f;
    if (mask.has_value()) {
        mask_f = mask.value().to(torch::dtype(torch::kFloat32).device(query.device()));
        TORCH_CHECK(mask_f.value().size(0) == S_q && mask_f.value().size(1) == S_kv,
                    "Mask must have shape [S_q, S_kv]");
    }

    dim3 blocks(S_q, H_q, B);
    int threads = D;  // one thread per head dimension
    size_t smem_size = (D + S_kv + 2) * sizeof(float);

    if (is_int4) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(query.scalar_type(), "fused_attention_quantized", [&] {
            fused_attention_quantized_kernel<scalar_t, true><<<blocks, threads, smem_size>>>(
                query.data_ptr<scalar_t>(),
                key.data_ptr<uint8_t>(),
                value.data_ptr<uint8_t>(),
                output.data_ptr<scalar_t>(),
                k_scale.data_ptr<float>(),
                v_scale.data_ptr<float>(),
                mask_f.has_value() ? mask_f.value().data_ptr<float>() : nullptr,
                B, H_q, H_kv, S_q, S_kv, D, D_packed
            );
        });
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(query.scalar_type(), "fused_attention_quantized", [&] {
            fused_attention_quantized_kernel<scalar_t, false><<<blocks, threads, smem_size>>>(
                query.data_ptr<scalar_t>(),
                key.data_ptr<int8_t>(),
                value.data_ptr<int8_t>(),
                output.data_ptr<scalar_t>(),
                k_scale.data_ptr<float>(),
                v_scale.data_ptr<float>(),
                mask_f.has_value() ? mask_f.value().data_ptr<float>() : nullptr,
                B, H_q, H_kv, S_q, S_kv, D, D_packed
            );
        });
    }

    return output;
}
