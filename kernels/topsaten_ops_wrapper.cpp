/**
 * Thin C-linkage wrapper around TopsAten optimized kernels for core ops:
 *   - RMS Normalization (topsvllmRmsNorm)
 *   - Fused Add + RMS Normalization (topsvllmFusedAddRmsNorm)
 *   - Matrix Multiplication (topsatenMatmul)
 *   - Flash Attention (topsfa*)
 *
 * Registers the same graph-aware allocator as topsaten_moe_wrapper.cpp so
 * dense / FA paths (which never touch MoE) still get correct capture/replay
 * workspace allocs.
 */
#include <cstdint>
#include <cstdio>
#include <atomic>
#include <tops/tops_runtime.h>
#include <topsaten/topsaten_define.h>
#include <topsaten/topsaten_vllm.h>
#include <topsaten/topsaten_ops.h>
#include <topsaten/topsaten_fa.h>

// ---------------------------------------------------------------------------
// Graph-aware allocator for topsaten kernel workspaces (same as MoE wrapper).
// ---------------------------------------------------------------------------

static topsError_t graph_aware_malloc_async(void **ptr, size_t size,
                                            topsStream_t stream, uint64_t) {
    return topsMallocAsync(ptr, size, stream, 0);
}

static topsError_t graph_aware_free_async(void *ptr, topsStream_t stream) {
    return topsFreeAsync(ptr, stream);
}

// ---------------------------------------------------------------------------
// topsaten initialization (shared init state with topsaten_moe_wrapper.cpp
// via the atomic guard).
// ---------------------------------------------------------------------------

static std::atomic<bool> g_ops_topsaten_initialized{false};

static void ensure_ops_init() {
    bool expected = false;
    if (g_ops_topsaten_initialized.compare_exchange_strong(expected, true)) {
        topsatenStatus_t st = topsatenInit();
        if (st != TOPSATEN_STATUS_SUCCESS) {
            fprintf(stderr, "[topsaten_ops_wrapper] topsatenInit failed: %d\n",
                    static_cast<int>(st));
            g_ops_topsaten_initialized.store(false);
            return;
        }
        topsatenMallocAsyncFuncRegister(graph_aware_malloc_async);
        topsatenFreeAsyncFuncRegister(graph_aware_free_async);
    }
}

// ---------------------------------------------------------------------------
// Tensor helpers
// ---------------------------------------------------------------------------

static topsatenTensor make_tensor_1d(void *data, int64_t d0,
                                     topsatenDataType_t dtype) {
    int64_t dims[1] = {d0};
    int64_t strides[1] = {1};
    topsatenSize_t dim_size(dims, 1);
    topsatenSize_t stride_size(strides, 1);
    return topsatenTensor(dim_size, stride_size, dtype,
                          static_cast<topsatenDeviceMemHandle_t>(data));
}

static topsatenTensor make_tensor_2d(void *data, int64_t d0, int64_t d1,
                                     topsatenDataType_t dtype) {
    int64_t dims[2] = {d0, d1};
    int64_t strides[2] = {d1, 1};
    topsatenSize_t dim_size(dims, 2);
    topsatenSize_t stride_size(strides, 2);
    return topsatenTensor(dim_size, stride_size, dtype,
                          static_cast<topsatenDeviceMemHandle_t>(data));
}

static topsatenTensor make_tensor_3d(void *data, int64_t d0, int64_t d1,
                                     int64_t d2, topsatenDataType_t dtype) {
    int64_t dims[3] = {d0, d1, d2};
    int64_t strides[3] = {d1 * d2, d2, 1};
    topsatenSize_t dim_size(dims, 3);
    topsatenSize_t stride_size(strides, 3);
    return topsatenTensor(dim_size, stride_size, dtype,
                          static_cast<topsatenDeviceMemHandle_t>(data));
}

static topsatenTensor make_tensor_4d(void *data, int64_t d0, int64_t d1,
                                     int64_t d2, int64_t d3,
                                     topsatenDataType_t dtype) {
    int64_t dims[4] = {d0, d1, d2, d3};
    int64_t strides[4] = {d1 * d2 * d3, d2 * d3, d3, 1};
    topsatenSize_t dim_size(dims, 4);
    topsatenSize_t stride_size(strides, 4);
    return topsatenTensor(dim_size, stride_size, dtype,
                          static_cast<topsatenDeviceMemHandle_t>(data));
}

static topsatenTensor make_tensor_nd(void *data, int32_t ndim,
                                     const int64_t *shape,
                                     topsatenDataType_t dtype) {
    int64_t strides[8];
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * shape[i + 1];
    topsatenSize_t dim_size(shape, ndim);
    topsatenSize_t stride_size(strides, ndim);
    return topsatenTensor(dim_size, stride_size, dtype,
                          static_cast<topsatenDeviceMemHandle_t>(data));
}

static topsatenTensor make_tensor_strided(void *data, int32_t ndim,
                                          const int64_t *shape,
                                          const int64_t *strides,
                                          topsatenDataType_t dtype) {
    topsatenSize_t dim_size(shape, ndim);
    topsatenSize_t stride_size(strides, ndim);
    return topsatenTensor(dim_size, stride_size, dtype,
                          static_cast<topsatenDeviceMemHandle_t>(data));
}

static topsatenTensor make_tensor_broadcast_2d(void *data, int64_t d0,
                                               int64_t d1,
                                               int64_t stride1,
                                               topsatenDataType_t dtype) {
    const int64_t shape[2] = {d0, d1};
    const int64_t strides[2] = {0, stride1};
    return make_tensor_strided(data, 2, shape, strides, dtype);
}

static topsatenDataType_t dtype_from_code(int32_t code) {
    return static_cast<topsatenDataType_t>(code);
}

extern "C" {

// Contiguous dtype conversion. TopsAten's Copy kernel supports differing
// input/output dtypes and is substantially better suited to large tensors
// than launching the generic Ubridge cast_* kernels.
int topsaten_cast(void *out_ptr, const void *in_ptr, int64_t numel,
                  int32_t input_dtype_code, int32_t output_dtype_code,
                  void *stream_) {
    ensure_ops_init();
    topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
    auto input = make_tensor_1d(const_cast<void *>(in_ptr), numel,
                                dtype_from_code(input_dtype_code));
    auto output = make_tensor_1d(out_ptr, numel,
                                 dtype_from_code(output_dtype_code));
    return static_cast<int>(topsaten::topsatenCopy(output, input, false, stream));
}

// Int8 KV-cache operators used by the GCU vLLM backend.  Keep these wrappers
// in the shared Ubridge topsaten library so Candle can call the same kernels
// without depending on libtorch or registering PyTorch custom operators.
static topsatenScalar_t make_scalar_i64(int64_t v);
static topsatenScalar_t make_scalar_f64(double v);

int tops_reshape_and_cache_flash_int8kv(
    void *key_cache_ptr, void *value_cache_ptr,
    const void *key_ptr, const void *value_ptr, const void *slot_mapping_ptr,
    const void *k_scale_ptr, const void *v_scale_ptr,
    const void *k_zp_ptr, const void *v_zp_ptr,
    int32_t num_tokens, int32_t num_heads, int32_t head_size,
    int32_t num_blocks, int32_t block_size,
    int32_t key_stride, int32_t value_stride, int32_t scale_len,
    int32_t input_dtype_code, void *stream_) {
    ensure_ops_init();
    topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
    const int64_t input_shape[3] = {num_tokens, num_heads, head_size};
    const int64_t input_strides[3] = {
        key_stride, head_size, 1,
    };
    const int64_t value_strides[3] = {
        value_stride, head_size, 1,
    };
    auto t_key = make_tensor_strided(
        const_cast<void *>(key_ptr), 3, input_shape, input_strides,
        dtype_from_code(input_dtype_code));
    auto t_value = make_tensor_strided(
        const_cast<void *>(value_ptr), 3, input_shape, value_strides,
        dtype_from_code(input_dtype_code));
    auto t_key_cache = make_tensor_4d(
        key_cache_ptr, num_blocks, block_size, num_heads, head_size,
        TOPSATEN_DATA_I8);
    auto t_value_cache = make_tensor_4d(
        value_cache_ptr, num_blocks, block_size, num_heads, head_size,
        TOPSATEN_DATA_I8);
    auto t_slot_mapping = make_tensor_1d(
        const_cast<void *>(slot_mapping_ptr), num_tokens, TOPSATEN_DATA_I32);

    topsatenTensor t_k_scale;
    topsatenTensor t_v_scale;
    if (k_scale_ptr && v_scale_ptr) {
        t_k_scale = make_tensor_1d(
            const_cast<void *>(k_scale_ptr), scale_len, TOPSATEN_DATA_FP32);
        t_v_scale = make_tensor_1d(
            const_cast<void *>(v_scale_ptr), scale_len, TOPSATEN_DATA_FP32);
    }
    topsatenTensor t_k_zp;
    topsatenTensor t_v_zp;
    if (k_zp_ptr && v_zp_ptr) {
        t_k_zp = make_tensor_1d(
            const_cast<void *>(k_zp_ptr), scale_len, TOPSATEN_DATA_FP32);
        t_v_zp = make_tensor_1d(
            const_cast<void *>(v_zp_ptr), scale_len, TOPSATEN_DATA_FP32);
    }

    return static_cast<int>(topsvllm::topsvllmReshapeAndCacheFlashInt8KV(
        t_key_cache, t_value_cache, t_key, t_value, t_slot_mapping, "int8",
        t_k_scale, t_v_scale, t_k_zp, t_v_zp, stream));
}

int tops_flash_attn_varlen_int8kv(
    void *out_ptr, void *softmax_lse_ptr, const void *q_ptr,
    const void *key_cache_ptr, const void *value_cache_ptr,
    const void *cu_seqlens_q_ptr, const void *seqused_k_ptr,
    const void *block_table_ptr, const void *k_descale_ptr,
    const void *v_descale_ptr, int32_t total_q, int32_t batch,
    int32_t num_heads, int32_t num_heads_k, int32_t head_size,
    int32_t max_seqlen_q, int32_t max_seqlen_k, int32_t num_blocks,
    int32_t page_block_size, int32_t max_num_blocks_per_seq,
    int32_t scale_len, int32_t dtype_code, float softmax_scale,
    float softcap, int32_t is_causal, int32_t window_size_left,
    int32_t window_size_right, int32_t num_splits, void *stream_) {
    ensure_ops_init();
    topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
    const auto dtype = dtype_from_code(dtype_code);

    auto t_q = make_tensor_3d(
        const_cast<void *>(q_ptr), total_q, num_heads, head_size, dtype);
    auto t_out = make_tensor_3d(out_ptr, total_q, num_heads, head_size, dtype);
    auto t_lse = make_tensor_2d(
        softmax_lse_ptr, num_heads, total_q, TOPSATEN_DATA_FP32);
    auto t_k_cache = make_tensor_4d(
        const_cast<void *>(key_cache_ptr), num_blocks, page_block_size,
        num_heads_k, head_size, TOPSATEN_DATA_I8);
    auto t_v_cache = make_tensor_4d(
        const_cast<void *>(value_cache_ptr), num_blocks, page_block_size,
        num_heads_k, head_size, TOPSATEN_DATA_I8);
    auto t_cu_q = make_tensor_1d(
        const_cast<void *>(cu_seqlens_q_ptr), batch + 1, TOPSATEN_DATA_I32);
    auto t_seqused_k = make_tensor_1d(
        const_cast<void *>(seqused_k_ptr), batch, TOPSATEN_DATA_I32);
    auto t_block_table = make_tensor_2d(
        const_cast<void *>(block_table_ptr), batch, max_num_blocks_per_seq,
        TOPSATEN_DATA_I32);

    // Candle keeps one reciprocal scale per KV head.  A zero stride in the
    // batch dimension is intentional: the same per-head scale applies to
    // every request in the batch without allocating a [batch, heads] view.
    auto t_k_descale = make_tensor_broadcast_2d(
        const_cast<void *>(k_descale_ptr), batch, num_heads_k,
        scale_len == 1 ? 0 : 1, TOPSATEN_DATA_FP32);
    auto t_v_descale = make_tensor_broadcast_2d(
        const_cast<void *>(v_descale_ptr), batch, num_heads_k,
        scale_len == 1 ? 0 : 1, TOPSATEN_DATA_FP32);

    topsatenTensor empty;
    std::vector<topsatenTensor> output = {t_out, t_lse, empty, empty};
    auto max_q = make_scalar_i64(max_seqlen_q);
    auto max_k = make_scalar_i64(max_seqlen_k);
    auto scale = make_scalar_f64(static_cast<double>(softmax_scale));
    auto window_left = make_scalar_i64(window_size_left);
    auto window_right = make_scalar_i64(window_size_right);
    auto softcap_scalar = make_scalar_f64(static_cast<double>(softcap));
    auto splits = make_scalar_i64(num_splits);
    auto sm_margin = make_scalar_i64(0);

    return static_cast<int>(topsvllm::topsvllmFlashAttnFwdInt8KV(
        output, t_q, t_k_cache, t_v_cache, empty, empty, empty, t_out, t_cu_q, empty,
        empty, empty, t_seqused_k, max_q, max_k, t_block_table, empty, empty,
        empty, empty, empty, empty, t_k_descale, t_v_descale, empty,
        empty, scale, is_causal != 0, window_left, window_right,
        softcap_scalar, true, empty, splits, false, sm_margin, empty, stream));
}

// =========================================================================
// RMS Normalization
// =========================================================================

/**
 * out = RMSNorm(input, gamma, epsilon)
 *
 * input:  [num_tokens, hidden_size]
 * gamma:  [hidden_size]
 * output: [num_tokens, hidden_size]
 */
int topsaten_rms_norm(
    void *output_ptr, void *input_ptr, void *gamma_ptr,
    int32_t num_tokens, int32_t hidden_size,
    float epsilon, int32_t dtype_code, void *stream_) {
    ensure_ops_init();

    topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
    topsatenDataType_t dtype = dtype_from_code(dtype_code);

    auto t_output = make_tensor_2d(output_ptr, num_tokens, hidden_size, dtype);
    auto t_input = make_tensor_2d(input_ptr, num_tokens, hidden_size, dtype);
    auto t_gamma = make_tensor_1d(gamma_ptr, hidden_size, dtype);

    topsatenScalar_t eps;
    eps.dtype = TOPSATEN_DATA_FP32;
    eps.fval = static_cast<double>(epsilon);

    topsatenStatus_t st = topsvllm::topsvllmRmsNorm(
        t_output, t_input, t_gamma, eps, stream);

    return static_cast<int>(st);
}

// Rank-3 variant used by Qwen Q/K normalization. vLLM passes
// [tokens, heads, head_size] directly to topsvllmRmsNorm; preserve that
// shape instead of flattening it to [tokens * heads, head_size].
int topsaten_rms_norm_3d(
    void *output_ptr, void *input_ptr, void *gamma_ptr,
    int32_t tokens, int32_t heads, int32_t hidden_size,
    float epsilon, int32_t dtype_code, void *stream_) {
    ensure_ops_init();

    topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
    topsatenDataType_t dtype = dtype_from_code(dtype_code);

    auto t_output = make_tensor_3d(output_ptr, tokens, heads, hidden_size, dtype);
    auto t_input = make_tensor_3d(input_ptr, tokens, heads, hidden_size, dtype);
    auto t_gamma = make_tensor_1d(gamma_ptr, hidden_size, dtype);

    topsatenScalar_t eps;
    eps.dtype = TOPSATEN_DATA_FP32;
    eps.fval = static_cast<double>(epsilon);

    topsatenStatus_t st = topsvllm::topsvllmRmsNorm(
        t_output, t_input, t_gamma, eps, stream);

    return static_cast<int>(st);
}

/**
 * Fused add + RMS Normalization (in-place on input and residual).
 *
 * input:    [num_tokens, hidden_size]  — overwritten with normalized output
 * residual: [num_tokens, hidden_size]  — overwritten with input + residual
 * weight:   [hidden_size]
 */
int topsaten_fused_add_rms_norm(
    void *input_ptr, void *residual_ptr, void *weight_ptr,
    int32_t num_tokens, int32_t hidden_size,
    float epsilon, int32_t dtype_code, void *stream_) {
    ensure_ops_init();

    topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
    topsatenDataType_t dtype = dtype_from_code(dtype_code);

    auto t_input = make_tensor_2d(input_ptr, num_tokens, hidden_size, dtype);
    auto t_residual = make_tensor_2d(residual_ptr, num_tokens, hidden_size, dtype);
    auto t_weight = make_tensor_1d(weight_ptr, hidden_size, dtype);

    topsatenStatus_t st = topsvllm::topsvllmFusedAddRmsNorm(
        t_input, t_residual, t_weight, epsilon, stream);

    return static_cast<int>(st);
}

// =========================================================================
// Activation: SiLU-and-Mul
// =========================================================================

/**
 * out = silu(input[..., :D]) * input[..., D:]
 *
 * input:  [num_tokens, 2 * D]
 * output: [num_tokens, D]
 */
int topsaten_silu_and_mul(
    void *output_ptr, void *input_ptr,
    int32_t num_tokens, int32_t d,
    int32_t dtype_code, void *stream_) {
    ensure_ops_init();

    topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
    topsatenDataType_t dtype = dtype_from_code(dtype_code);

    auto t_input = make_tensor_2d(input_ptr, num_tokens, 2 * d, dtype);
    auto t_output = make_tensor_2d(output_ptr, num_tokens, d, dtype);

    topsatenStatus_t st = topsvllm::topsvllmSiluAndMul(
        t_output, t_input, stream);

    return static_cast<int>(st);
}

/**
 * out = gelu(input[..., :D]) * input[..., D:]
 *
 * input:  [num_tokens, 2 * D]
 * output: [num_tokens, D]
 */
int topsaten_gelu_and_mul(
    void *output_ptr, void *input_ptr,
    int32_t num_tokens, int32_t d,
    int32_t dtype_code, void *stream_) {
    ensure_ops_init();

    topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
    topsatenDataType_t dtype = dtype_from_code(dtype_code);

    auto t_input = make_tensor_2d(input_ptr, num_tokens, 2 * d, dtype);
    auto t_output = make_tensor_2d(output_ptr, num_tokens, d, dtype);

    topsatenStatus_t st = topsvllm::topsvllmGeluAndMul(
        t_output, t_input, stream);

    return static_cast<int>(st);
}

/**
 * out = gelu_tanh(input[..., :D]) * input[..., D:]
 *
 * input:  [num_tokens, 2 * D]
 * output: [num_tokens, D]
 */
int topsaten_gelu_tanh_and_mul(
    void *output_ptr, void *input_ptr,
    int32_t num_tokens, int32_t d,
    int32_t dtype_code, void *stream_) {
    ensure_ops_init();

    topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
    topsatenDataType_t dtype = dtype_from_code(dtype_code);

    auto t_input = make_tensor_2d(input_ptr, num_tokens, 2 * d, dtype);
    auto t_output = make_tensor_2d(output_ptr, num_tokens, d, dtype);

    topsatenStatus_t st = topsvllm::topsvllmGeluTanhAndMul(
        t_output, t_input, stream);

    return static_cast<int>(st);
}

// =========================================================================
// Softmax
// =========================================================================

/**
 * out = softmax(input, dim)
 *
 * input/output shapes are arbitrary; 'dim' selects the reduction axis.
 * We expose this as a 2D operation [n_rows, n_cols] where softmax runs
 * along the last dimension (dim = -1).
 */
int topsaten_softmax(
    void *output_ptr, void *input_ptr,
    int32_t n_rows, int32_t n_cols,
    int32_t dtype_code, void *stream_) {
    ensure_ops_init();

    topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
    topsatenDataType_t dtype = dtype_from_code(dtype_code);

    auto t_input = make_tensor_2d(input_ptr, n_rows, n_cols, dtype);
    auto t_output = make_tensor_2d(output_ptr, n_rows, n_cols, dtype);

    int64_t dim = -1;

    topsatenStatus_t st = topsaten::topsatenSoftmaxForward(
        t_output, t_input, dim, stream);

    return static_cast<int>(st);
}

// =========================================================================
// Matrix Multiplication
// =========================================================================

/**
 * out = lhs @ rhs
 *
 * Supports 2D and batched 3D matmul.
 * For 2D: lhs [M, K], rhs [K, N] -> out [M, N]
 * For 3D: lhs [B, M, K], rhs [B, K, N] -> out [B, M, N]
 *
 * lhs_transpose / rhs_transpose: flags from candle's layout.transform_ops
 *   that describe the *logical* transpose applied to the data.
 *
 * When transpose==1, the physical memory layout is swapped:
 *   - lhs is stored as [K, M] contiguous (row-major) and needs to be
 *     interpreted as [M, K] by setting strides accordingly.
 *   - rhs is stored as [N, K] contiguous and needs [K, N] interpretation.
 */
int topsaten_matmul(
    void *out_ptr, void *lhs_ptr, void *rhs_ptr,
    int32_t batch, int32_t m, int32_t k, int32_t n,
    int32_t lhs_transpose, int32_t rhs_transpose,
    int32_t dtype_code, void *stream_) {
    ensure_ops_init();

    topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
    topsatenDataType_t dtype = dtype_from_code(dtype_code);

    topsatenTensor t_out, t_lhs, t_rhs;

    if (batch <= 1) {
        if (lhs_transpose) {
            // Physical [K, M], logical [M, K]
            int64_t dims[2] = {m, k};
            int64_t strides[2] = {1, m};
            topsatenSize_t ds(dims, 2), ss(strides, 2);
            t_lhs = topsatenTensor(ds, ss, dtype,
                                   static_cast<topsatenDeviceMemHandle_t>(lhs_ptr));
        } else {
            t_lhs = make_tensor_2d(lhs_ptr, m, k, dtype);
        }

        if (rhs_transpose) {
            // Physical [N, K], logical [K, N]
            int64_t dims[2] = {k, n};
            int64_t strides[2] = {1, k};
            topsatenSize_t ds(dims, 2), ss(strides, 2);
            t_rhs = topsatenTensor(ds, ss, dtype,
                                   static_cast<topsatenDeviceMemHandle_t>(rhs_ptr));
        } else {
            t_rhs = make_tensor_2d(rhs_ptr, k, n, dtype);
        }

        t_out = make_tensor_2d(out_ptr, m, n, dtype);
    } else {
        if (lhs_transpose) {
            int64_t dims[3] = {batch, m, k};
            int64_t strides[3] = {m * k, 1, m};
            topsatenSize_t ds(dims, 3), ss(strides, 3);
            t_lhs = topsatenTensor(ds, ss, dtype,
                                   static_cast<topsatenDeviceMemHandle_t>(lhs_ptr));
        } else {
            t_lhs = make_tensor_3d(lhs_ptr, batch, m, k, dtype);
        }

        if (rhs_transpose) {
            int64_t dims[3] = {batch, k, n};
            int64_t strides[3] = {k * n, 1, k};
            topsatenSize_t ds(dims, 3), ss(strides, 3);
            t_rhs = topsatenTensor(ds, ss, dtype,
                                   static_cast<topsatenDeviceMemHandle_t>(rhs_ptr));
        } else {
            t_rhs = make_tensor_3d(rhs_ptr, batch, k, n, dtype);
        }

        t_out = make_tensor_3d(out_ptr, batch, m, n, dtype);
    }

    topsatenStatus_t st = topsaten::topsatenMatmul(t_out, t_lhs, t_rhs, stream);
    return static_cast<int>(st);
}

/**
 * Batched matmul with broadcast support:
 * lhs [B, M, K] @ rhs [1, K, N] -> out [B, M, N]
 * (rhs is broadcast across batch — we flatten lhs to [B*M, K] × [K, N])
 */
int topsaten_matmul_broadcast(
    void *out_ptr, void *lhs_ptr, void *rhs_ptr,
    int32_t batch, int32_t m, int32_t k, int32_t n,
    int32_t lhs_transpose, int32_t rhs_transpose,
    int32_t dtype_code, void *stream_) {
    ensure_ops_init();

    topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
    topsatenDataType_t dtype = dtype_from_code(dtype_code);

    int64_t total_m = static_cast<int64_t>(batch) * m;
    topsatenTensor t_lhs, t_rhs;

    if (lhs_transpose) {
        int64_t dims[2] = {total_m, k};
        int64_t strides[2] = {1, total_m};
        topsatenSize_t ds(dims, 2), ss(strides, 2);
        t_lhs = topsatenTensor(ds, ss, dtype,
                               static_cast<topsatenDeviceMemHandle_t>(lhs_ptr));
    } else {
        t_lhs = make_tensor_2d(lhs_ptr, total_m, k, dtype);
    }

    if (rhs_transpose) {
        int64_t dims[2] = {k, n};
        int64_t strides[2] = {1, k};
        topsatenSize_t ds(dims, 2), ss(strides, 2);
        t_rhs = topsatenTensor(ds, ss, dtype,
                               static_cast<topsatenDeviceMemHandle_t>(rhs_ptr));
    } else {
        t_rhs = make_tensor_2d(rhs_ptr, k, n, dtype);
    }

    auto t_out = make_tensor_2d(out_ptr, total_m, n, dtype);

    topsatenStatus_t st = topsaten::topsatenMatmul(t_out, t_lhs, t_rhs, stream);
    return static_cast<int>(st);
}

// =========================================================================
// Top-K
// =========================================================================

/**
 * topsatenTopk along the last dimension.
 *
 * input:        [n_rows, n_cols]
 * output_value: [n_rows, k]
 * output_index: [n_rows, k]  (u32/i32 indices — topsaten rejects i64)
 */
int topsaten_topk(
    void *output_value_ptr, void *output_index_ptr,
    void *input_ptr,
    int32_t n_rows, int32_t n_cols, int32_t k,
    int32_t is_largest, int32_t is_sorted,
    int32_t dtype_code, void *stream_) {
    ensure_ops_init();

    topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
    topsatenDataType_t dtype = dtype_from_code(dtype_code);

    auto t_input = make_tensor_2d(input_ptr, n_rows, n_cols, dtype);
    auto t_output_value = make_tensor_2d(output_value_ptr, n_rows, k, dtype);

    int64_t out_dims[2] = {n_rows, k};
    topsatenSize_t out_sz(out_dims, 2);
    int64_t out_strides[2] = {static_cast<int64_t>(k), 1};
    topsatenSize_t out_st(out_strides, 2);
    topsatenTensor t_output_index(out_sz, out_st, TOPSATEN_DATA_U32,
        static_cast<topsatenDeviceMemHandle_t>(output_index_ptr));

    int64_t axis = 1;  // last dim of 2D [n_rows, n_cols]

    topsatenStatus_t st = topsaten::topsatenTopk(
        t_output_value, t_output_index, t_input,
        static_cast<int64_t>(k), axis,
        static_cast<bool>(is_largest),
        static_cast<bool>(is_sorted),
        stream);

    return static_cast<int>(st);
}

// =========================================================================
// Sort
// =========================================================================

/**
 * topsatenSort along the last dimension.
 *
 * input:         [n_rows, n_cols]
 * output_sorted: [n_rows, n_cols]  (sorted values)
 * output_index:  [n_rows, n_cols]  (u32/i32 indices — topsaten rejects i64)
 */
int topsaten_sort(
    void *output_sorted_ptr, void *output_index_ptr,
    void *input_ptr,
    int32_t n_rows, int32_t n_cols,
    int32_t descending,
    int32_t dtype_code, void *stream_) {
    ensure_ops_init();

    topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
    topsatenDataType_t dtype = dtype_from_code(dtype_code);

    auto t_input = make_tensor_2d(input_ptr, n_rows, n_cols, dtype);
    auto t_output_sorted = make_tensor_2d(output_sorted_ptr, n_rows, n_cols, dtype);

    int64_t out_dims[2] = {n_rows, n_cols};
    topsatenSize_t out_sz(out_dims, 2);
    int64_t out_strides[2] = {static_cast<int64_t>(n_cols), 1};
    topsatenSize_t out_st(out_strides, 2);
    topsatenTensor t_output_index(out_sz, out_st, TOPSATEN_DATA_U32,
        static_cast<topsatenDeviceMemHandle_t>(output_index_ptr));

    int64_t dim = -1;
    bool stable = false;

    topsatenStatus_t st = topsaten::topsatenSort(
        t_output_sorted, t_output_index, t_input,
        stable, dim,
        static_cast<bool>(descending),
        stream);

    return static_cast<int>(st);
}

// =========================================================================
// Reduce Sum
// =========================================================================

/**
 * Reduce sum over a single dimension.
 *
 * input:  contiguous tensor of rank 1..5
 * output: input with dim `reduce_dim` removed (keepdims=false)
 * reduce_dim: dimension to reduce (0-based)
 * num_dims: rank of input
 * input_dims: array of num_dims int64 values (input shape)
 * dtype_code: 4=f16, 5=bf16, 8=f32
 */
int topsaten_sum(
    void *output_ptr, void *input_ptr,
    int32_t reduce_dim, int32_t num_dims,
    int64_t *input_dims,
    int32_t dtype_code, void *stream_) {
    ensure_ops_init();

    topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
    topsatenDataType_t dtype = dtype_from_code(dtype_code);

    // Build input tensor
    int64_t in_strides[8];
    in_strides[num_dims - 1] = 1;
    for (int i = num_dims - 2; i >= 0; --i)
        in_strides[i] = in_strides[i + 1] * input_dims[i + 1];
    topsatenSize_t in_dim_size(input_dims, num_dims);
    topsatenSize_t in_stride_size(in_strides, num_dims);
    topsatenTensor t_input(in_dim_size, in_stride_size, dtype,
                           static_cast<topsatenDeviceMemHandle_t>(input_ptr));

    // Build output tensor (input shape with reduce_dim removed)
    int64_t out_dims[8];
    int out_rank = 0;
    for (int i = 0; i < num_dims; ++i) {
        if (i != reduce_dim)
            out_dims[out_rank++] = input_dims[i];
    }
    if (out_rank == 0) {
        out_dims[0] = 1;
        out_rank = 1;
    }
    int64_t out_strides[8];
    out_strides[out_rank - 1] = 1;
    for (int i = out_rank - 2; i >= 0; --i)
        out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
    topsatenSize_t out_dim_size(out_dims, out_rank);
    topsatenSize_t out_stride_size(out_strides, out_rank);
    topsatenTensor t_output(out_dim_size, out_stride_size, dtype,
                            static_cast<topsatenDeviceMemHandle_t>(output_ptr));

    int64_t dim_arr[1] = { static_cast<int64_t>(reduce_dim) };
    topsatenSize_t dimensions(dim_arr, 1);

    topsatenStatus_t st = topsaten::topsatenSum(
        t_output, t_input, dimensions, false, dtype, stream);

    return static_cast<int>(st);
}

// =========================================================================
// Device-to-device memcpy helper
// =========================================================================

int ubridge_memcpy_d2d(void *dst, const void *src, size_t bytes, void *stream_) {
    topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
    topsError_t err = topsMemcpyAsync(dst, src, bytes, topsMemcpyDeviceToDevice, stream);
    return (err == topsSuccess) ? 0 : static_cast<int>(err);
}

// =========================================================================
// Rotary Embedding (RoPE)
// =========================================================================

/**
 * In-place rotary positional embedding on query and key.
 *
 * query:         [num_tokens, num_heads * head_size]  (in-place)
 * key:           [num_tokens, num_kv_heads * head_size]  (in-place)
 * positions:     [num_tokens]  (int64)
 * cos_sin_cache: [max_position, rot_dim]  (f32)
 * head_size:     scalar
 * is_neox:       1 = GPT-NeoX style, 0 = GPT-J style
 */
int topsaten_rotary_embedding(
    void *query_ptr, void *key_ptr,
    void *positions_ptr,
    void *cos_sin_cache_ptr,
    int32_t num_tokens,
    int32_t q_num_heads, int32_t k_num_heads,
    int32_t head_size, int32_t rot_dim,
    int32_t max_position,
    int32_t is_neox,
    int32_t qk_dtype_code,
    int32_t cache_dtype_code,
    void *stream_) {
    ensure_ops_init();

    topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
    topsatenDataType_t qk_dtype = dtype_from_code(qk_dtype_code);
    topsatenDataType_t cache_dtype = dtype_from_code(cache_dtype_code);

    auto t_query = make_tensor_2d(query_ptr,
        static_cast<int64_t>(num_tokens),
        static_cast<int64_t>(q_num_heads) * head_size,
        qk_dtype);
    auto t_key = make_tensor_2d(key_ptr,
        static_cast<int64_t>(num_tokens),
        static_cast<int64_t>(k_num_heads) * head_size,
        qk_dtype);

    int64_t pos_dims[1] = { static_cast<int64_t>(num_tokens) };
    int64_t pos_strides[1] = { 1 };
    topsatenSize_t pos_sz(pos_dims, 1);
    topsatenSize_t pos_st(pos_strides, 1);
    topsatenTensor t_positions(pos_sz, pos_st, TOPSATEN_DATA_I64,
        static_cast<topsatenDeviceMemHandle_t>(positions_ptr));

    auto t_cos_sin_cache = make_tensor_2d(cos_sin_cache_ptr,
        static_cast<int64_t>(max_position),
        static_cast<int64_t>(rot_dim),
        cache_dtype);

    topsatenStatus_t st = topsvllm::topsvllmRotaryEmbedding(
        t_query, t_key, t_positions, t_cos_sin_cache,
        static_cast<int>(head_size),
        static_cast<bool>(is_neox),
        stream);

    return static_cast<int>(st);
}

// =========================================================================
// Unary ops — uniform signature: (out, in, num_el, ndim, shape, dtype, stream)
// =========================================================================

#define DEFINE_UNARY_OP(c_name, aten_fn) \
int c_name( \
    void *out_ptr, const void *in_ptr, \
    int64_t num_el, int32_t ndim, const int64_t *shape, \
    int32_t dtype_code, void *stream_) { \
    ensure_ops_init(); \
    topsStream_t stream = reinterpret_cast<topsStream_t>(stream_); \
    topsatenDataType_t dtype = dtype_from_code(dtype_code); \
    auto t_in  = make_tensor_nd(const_cast<void*>(in_ptr),  ndim, shape, dtype); \
    auto t_out = make_tensor_nd(out_ptr, ndim, shape, dtype); \
    topsatenStatus_t st = topsaten::aten_fn(t_out, t_in, stream); \
    return static_cast<int>(st); \
}

DEFINE_UNARY_OP(topsaten_exp,        topsatenExp)
DEFINE_UNARY_OP(topsaten_log,        topsatenLog)
DEFINE_UNARY_OP(topsaten_sin,        topsatenSin)
DEFINE_UNARY_OP(topsaten_cos,        topsatenCos)
DEFINE_UNARY_OP(topsaten_tanh_op,    topsatenTanh)
DEFINE_UNARY_OP(topsaten_silu,       topsatenSilu)
DEFINE_UNARY_OP(topsaten_rsqrt,      topsatenRsqrt)
DEFINE_UNARY_OP(topsaten_sqrt,       topsatenSqrt)
DEFINE_UNARY_OP(topsaten_neg,        topsatenNeg)
DEFINE_UNARY_OP(topsaten_abs,        topsatenAbs)
DEFINE_UNARY_OP(topsaten_sigmoid,    topsatenSigmoid)
DEFINE_UNARY_OP(topsaten_reciprocal, topsatenReciprocal)

int topsaten_gelu(
    void *out_ptr, const void *in_ptr,
    int64_t num_el, int32_t ndim, const int64_t *shape,
    int32_t dtype_code, void *stream_) {
    ensure_ops_init();
    topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
    topsatenDataType_t dtype = dtype_from_code(dtype_code);
    auto t_in  = make_tensor_nd(const_cast<void*>(in_ptr),  ndim, shape, dtype);
    auto t_out = make_tensor_nd(out_ptr, ndim, shape, dtype);
    topsatenStatus_t st = topsaten::topsatenGelu(t_out, t_in, "none", stream);
    return static_cast<int>(st);
}

// =========================================================================
// Binary ops — uniform signature: (out, lhs, rhs, num_el, ndim, shape, dtype, stream)
// Both lhs and rhs must have the same shape (broadcast resolved by caller).
// =========================================================================

static topsatenScalar_t make_scalar_one() {
    topsatenScalar_t s;
    s.dtype = TOPSATEN_DATA_F64;
    s.fval = 1.0;
    return s;
}

// Add and Sub need an alpha parameter (set to 1.0 for identity)
#define DEFINE_BINARY_OP_ALPHA(c_name, aten_fn) \
int c_name( \
    void *out_ptr, const void *lhs_ptr, const void *rhs_ptr, \
    int64_t num_el, int32_t ndim, const int64_t *shape, \
    int32_t dtype_code, void *stream_) { \
    ensure_ops_init(); \
    topsStream_t stream = reinterpret_cast<topsStream_t>(stream_); \
    topsatenDataType_t dtype = dtype_from_code(dtype_code); \
    auto t_lhs = make_tensor_nd(const_cast<void*>(lhs_ptr), ndim, shape, dtype); \
    auto t_rhs = make_tensor_nd(const_cast<void*>(rhs_ptr), ndim, shape, dtype); \
    auto t_out = make_tensor_nd(out_ptr, ndim, shape, dtype); \
    topsatenScalar_t alpha = make_scalar_one(); \
    topsatenStatus_t st = topsaten::aten_fn(t_out, t_lhs, t_rhs, alpha, stream); \
    return static_cast<int>(st); \
}

#define DEFINE_BINARY_OP(c_name, aten_fn) \
int c_name( \
    void *out_ptr, const void *lhs_ptr, const void *rhs_ptr, \
    int64_t num_el, int32_t ndim, const int64_t *shape, \
    int32_t dtype_code, void *stream_) { \
    ensure_ops_init(); \
    topsStream_t stream = reinterpret_cast<topsStream_t>(stream_); \
    topsatenDataType_t dtype = dtype_from_code(dtype_code); \
    auto t_lhs = make_tensor_nd(const_cast<void*>(lhs_ptr), ndim, shape, dtype); \
    auto t_rhs = make_tensor_nd(const_cast<void*>(rhs_ptr), ndim, shape, dtype); \
    auto t_out = make_tensor_nd(out_ptr, ndim, shape, dtype); \
    topsatenStatus_t st = topsaten::aten_fn(t_out, t_lhs, t_rhs, stream); \
    return static_cast<int>(st); \
}

DEFINE_BINARY_OP_ALPHA(topsaten_add, topsatenAdd)
DEFINE_BINARY_OP_ALPHA(topsaten_sub, topsatenSub)
DEFINE_BINARY_OP(topsaten_mul, topsatenMul)
DEFINE_BINARY_OP(topsaten_div, topsatenDiv)

// =========================================================================
// Memory-Efficient Attention (SDP prefill)
// =========================================================================

/**
 * output:    [batch, q_seq, num_heads, head_size]
 * query:     [batch, q_seq, num_heads, head_size]
 * key:       [batch, kv_seq, num_kv_heads, head_size]
 * value:     [batch, kv_seq, num_kv_heads, head_size]
 * attn_bias: [1, 1, q_seq, kv_seq] or empty
 * scale:     softmax scale (typically 1/sqrt(head_size))
 * is_causal: whether to apply causal mask
 */
int topsaten_efficient_attention(
    void *out_ptr, const void *q_ptr, const void *k_ptr, const void *v_ptr,
    const void *bias_ptr,
    int32_t batch, int32_t q_seq, int32_t kv_seq,
    int32_t num_heads, int32_t num_kv_heads, int32_t head_size,
    float scale, int32_t is_causal, int32_t has_bias,
    int32_t dtype_code, void *stream_) {
    ensure_ops_init();

    topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
    topsatenDataType_t dtype = dtype_from_code(dtype_code);

    auto t_out   = make_tensor_4d(out_ptr,
        batch, q_seq, num_heads, head_size, dtype);
    auto t_query = make_tensor_4d(const_cast<void*>(q_ptr),
        batch, q_seq, num_heads, head_size, dtype);
    auto t_key   = make_tensor_4d(const_cast<void*>(k_ptr),
        batch, kv_seq, num_kv_heads, head_size, dtype);
    auto t_value = make_tensor_4d(const_cast<void*>(v_ptr),
        batch, kv_seq, num_kv_heads, head_size, dtype);

    topsatenTensor t_bias;
    if (has_bias && bias_ptr) {
        t_bias = make_tensor_4d(const_cast<void*>(bias_ptr),
            1, 1, q_seq, kv_seq, dtype);
    }

    topsatenScalar_t dropout_p;
    dropout_p.dtype = TOPSATEN_DATA_F64;
    dropout_p.fval = 0.0;

    topsatenScalar_t scale_s;
    scale_s.dtype = TOPSATEN_DATA_F64;
    scale_s.fval = static_cast<double>(scale);

    topsatenStatus_t st;
    if (is_causal) {
        topsatenScalar_t mask_mode;
        mask_mode.dtype = TOPSATEN_DATA_I64;
        mask_mode.ival = 1;

        topsatenTensor alibi_slopes;
        topsatenScalar_t sliding_window;
        sliding_window.dtype = TOPSATEN_DATA_I64;
        sliding_window.ival = 0;

        st = topsvllm::topsvllmMemoryEfficientAttentionV1(
            t_out, t_query, t_key, t_value, t_bias,
            dropout_p, scale_s, mask_mode, alibi_slopes, sliding_window,
            stream);
    } else {
        st = topsvllm::topsvllmMemoryEfficientAttention(
            t_out, t_query, t_key, t_value, t_bias,
            dropout_p, scale_s, stream);
    }

    return static_cast<int>(st);
}

// =========================================================================
// Flash Attention via topsfa (replaces libflashkernels.so)
// =========================================================================

static topsatenScalar_t make_scalar_i64(int64_t v) {
    topsatenScalar_t s;
    s.dtype = TOPSATEN_DATA_I64;
    s.ival = v;
    return s;
}

static topsatenScalar_t make_scalar_f64(double v) {
    topsatenScalar_t s;
    s.dtype = TOPSATEN_DATA_F64;
    s.fval = v;
    return s;
}

/**
 * topsfaFlashAttnFwd wrapper
 * query:  [batch, seqlen_q, num_heads, head_size]
 * key:    [batch, seqlen_k, num_heads_k, head_size]
 * value:  [batch, seqlen_k, num_heads_k, head_size]
 * output: [batch, seqlen_q, num_heads, head_size]
 * softmax_lse: [batch, num_heads, seqlen_q]
 */
int topsfa_flash_attn_fwd(
    void *out_ptr, void *softmax_lse_ptr,
    const void *q_ptr, const void *k_ptr, const void *v_ptr,
    const void *alibi_slopes_ptr,
    int32_t batch, int32_t seqlen_q, int32_t seqlen_k,
    int32_t num_heads, int32_t num_heads_k, int32_t head_size,
    float softmax_scale, float softcap,
    int32_t is_causal,
    int32_t window_size_left, int32_t window_size_right,
    int32_t dtype_code, void *stream_) {
    ensure_ops_init();
    topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
    topsatenDataType_t dtype = dtype_from_code(dtype_code);

    auto t_q = make_tensor_4d(const_cast<void*>(q_ptr),
        batch, seqlen_q, num_heads, head_size, dtype);
    auto t_k = make_tensor_4d(const_cast<void*>(k_ptr),
        batch, seqlen_k, num_heads_k, head_size, dtype);
    auto t_v = make_tensor_4d(const_cast<void*>(v_ptr),
        batch, seqlen_k, num_heads_k, head_size, dtype);
    auto t_out = make_tensor_4d(out_ptr,
        batch, seqlen_q, num_heads, head_size, dtype);
    auto t_lse = make_tensor_3d(softmax_lse_ptr,
        batch, num_heads, seqlen_q, TOPSATEN_DATA_FP32);

    topsatenTensor t_out_empty;
    topsatenTensor t_alibi;
    if (alibi_slopes_ptr) {
        t_alibi = make_tensor_1d(const_cast<void*>(alibi_slopes_ptr),
            num_heads, dtype);
    }

    topsatenGenerator_t gen;
    gen.seed = 0;
    gen.offset = 0;

    std::tuple<topsatenTensor&, topsatenTensor&,
               topsatenTensor&, topsatenTensor&>
        output(t_out, t_lse, t_out_empty, t_out_empty);

    topsatenStatus_t st = topsfa::topsfaFlashAttnFwd(
        output, t_q, t_k, t_v, t_out_empty, t_alibi,
        make_scalar_f64(0.0),
        make_scalar_f64(static_cast<double>(softmax_scale)),
        is_causal != 0,
        make_scalar_i64(static_cast<int64_t>(window_size_left)),
        make_scalar_i64(static_cast<int64_t>(window_size_right)),
        make_scalar_f64(static_cast<double>(softcap)),
        false, gen, stream);
    return static_cast<int>(st);
}

/**
 * topsfaFlashAttnVarlenFwd wrapper
 *
 *   - Dense (no block_table):
 *       k/v: [total_k, num_heads_k, head_size]
 *       cu_seqlens_k required; seqused_k unused
 *   - Paged (block_table set) — used for chunked prefill + decode:
 *       k/v: [num_blocks, page_block_size, num_heads_k, head_size]
 *       seqused_k: [batch] actual KV lengths (required)
 *       cu_seqlens_k: dummy (may be null; wrapper ignores it)
 *
 * query:  [total_q, num_heads, head_size]
 * output: [total_q, num_heads, head_size]
 * softmax_lse: [num_heads, total_q]  (runtime CheckArgs: [q_num_heads, total_q])
 */
int topsfa_flash_attn_varlen_fwd(
    void *out_ptr, void *softmax_lse_ptr,
    const void *q_ptr, const void *k_ptr, const void *v_ptr,
    const void *cu_seqlens_q_ptr, const void *cu_seqlens_k_ptr,
    const void *seqused_k_ptr,
    const void *block_table_ptr,
    const void *alibi_slopes_ptr,
    int32_t total_q, int32_t total_k,
    int32_t batch, int32_t num_heads, int32_t num_heads_k, int32_t head_size,
    int32_t max_seqlen_q, int32_t max_seqlen_k,
    float softmax_scale, float softcap,
    int32_t is_causal,
    int32_t window_size_left, int32_t window_size_right,
    int32_t has_block_table, int32_t block_table_batch, int32_t block_table_max_blocks,
    int32_t num_blocks, int32_t page_block_size,
    int32_t dtype_code, void *stream_) {
    ensure_ops_init();
    topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
    topsatenDataType_t dtype = dtype_from_code(dtype_code);

    auto t_q = make_tensor_3d(const_cast<void*>(q_ptr),
        total_q, num_heads, head_size, dtype);
    topsatenTensor t_k;
    topsatenTensor t_v;
    if (has_block_table && block_table_ptr) {
        t_k = make_tensor_4d(const_cast<void*>(k_ptr),
            num_blocks, page_block_size, num_heads_k, head_size, dtype);
        t_v = make_tensor_4d(const_cast<void*>(v_ptr),
            num_blocks, page_block_size, num_heads_k, head_size, dtype);
    } else {
        t_k = make_tensor_3d(const_cast<void*>(k_ptr),
            total_k, num_heads_k, head_size, dtype);
        t_v = make_tensor_3d(const_cast<void*>(v_ptr),
            total_k, num_heads_k, head_size, dtype);
    }
    auto t_out = make_tensor_3d(out_ptr,
        total_q, num_heads, head_size, dtype);
    auto t_lse = make_tensor_2d(softmax_lse_ptr,
        num_heads, total_q, TOPSATEN_DATA_FP32);

    auto t_cu_q = make_tensor_1d(const_cast<void*>(cu_seqlens_q_ptr),
        batch + 1, TOPSATEN_DATA_I32);

    // when seqused_k is set, cu_seqlens_k is unused (dummy).
    // Always provide a (batch+1) tensor so the OP sees a valid handle.
    topsatenTensor t_cu_k;
    if (cu_seqlens_k_ptr) {
        t_cu_k = make_tensor_1d(const_cast<void*>(cu_seqlens_k_ptr),
            batch + 1, TOPSATEN_DATA_I32);
    } else {
        t_cu_k = make_tensor_1d(const_cast<void*>(cu_seqlens_q_ptr),
            batch + 1, TOPSATEN_DATA_I32);
    }

    topsatenTensor t_seqused_k;
    if (seqused_k_ptr) {
        t_seqused_k = make_tensor_1d(const_cast<void*>(seqused_k_ptr),
            batch, TOPSATEN_DATA_I32);
    }
    topsatenTensor t_leftpad_k;

    topsatenTensor t_block_table;
    if (has_block_table && block_table_ptr) {
        t_block_table = make_tensor_2d(const_cast<void*>(block_table_ptr),
            block_table_batch, block_table_max_blocks, TOPSATEN_DATA_I32);
    }

    topsatenTensor t_alibi;
    if (alibi_slopes_ptr) {
        t_alibi = make_tensor_1d(const_cast<void*>(alibi_slopes_ptr),
            num_heads, dtype);
    }

    topsatenGenerator_t gen;
    gen.seed = 0;
    gen.offset = 0;

    std::tuple<topsatenTensor, topsatenTensor,
               topsatenTensor, topsatenTensor> output(t_out, t_lse,
               topsatenTensor(), topsatenTensor());

    topsatenStatus_t st = topsfa::topsfaFlashAttnVarlenFwd(
        output, t_q, t_k, t_v, t_cu_q, t_cu_k,
        t_seqused_k, t_leftpad_k, t_block_table, t_alibi,
        make_scalar_i64(max_seqlen_q),
        make_scalar_i64(max_seqlen_k),
        make_scalar_f64(0.0),
        make_scalar_f64(static_cast<double>(softmax_scale)),
        /*zero_tensors=*/true, is_causal != 0,
        make_scalar_i64(static_cast<int64_t>(window_size_left)),
        make_scalar_i64(static_cast<int64_t>(window_size_right)),
        make_scalar_f64(static_cast<double>(softcap)),
        false, gen, stream);

    return static_cast<int>(st);
}

/**
 * topsfaFlashAttnFwdKvcache wrapper
 * query:    [batch, seqlen_q, num_heads, head_size]
 * kcache:   [batch_k, seqlen_k, num_heads_k, head_size] or paged [num_blocks, page_size, num_heads_k, head_size]
 * vcache:   same layout as kcache
 * seqlens_k: [batch] actual kv seq lengths
 * block_table: [batch, max_num_blocks_per_seq] (i32)
 * output:   [batch, seqlen_q, num_heads, head_size]
 */
int topsfa_flash_attn_fwd_kvcache(
    void *out_ptr, void *softmax_lse_ptr,
    const void *q_ptr,
    const void *kcache_ptr, const void *vcache_ptr,
    const void *seqlens_k_ptr,
    const void *block_table_ptr,
    const void *alibi_slopes_ptr,
    int32_t batch, int32_t seqlen_q,
    int32_t num_heads, int32_t num_heads_k, int32_t head_size,
    int32_t seqlen_k, int32_t page_block_size,
    int32_t num_blocks, int32_t max_num_blocks_per_seq,
    float softmax_scale, float softcap,
    int32_t is_causal,
    int32_t window_size_left, int32_t window_size_right,
    int32_t is_rotary_interleaved, int32_t num_splits,
    int32_t has_block_table,
    int32_t dtype_code, void *stream_) {
    ensure_ops_init();
    topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
    topsatenDataType_t dtype = dtype_from_code(dtype_code);

    auto t_q = make_tensor_4d(const_cast<void*>(q_ptr),
        batch, seqlen_q, num_heads, head_size, dtype);
    auto t_out = make_tensor_4d(out_ptr,
        batch, seqlen_q, num_heads, head_size, dtype);
    auto t_lse = make_tensor_3d(softmax_lse_ptr,
        batch, num_heads, seqlen_q, TOPSATEN_DATA_FP32);

    topsatenTensor t_kcache;
    topsatenTensor t_vcache;
    if (has_block_table && block_table_ptr) {
        t_kcache = make_tensor_4d(const_cast<void*>(kcache_ptr),
            num_blocks, page_block_size, num_heads_k, head_size, dtype);
        t_vcache = make_tensor_4d(const_cast<void*>(vcache_ptr),
            num_blocks, page_block_size, num_heads_k, head_size, dtype);
    } else {
        t_kcache = make_tensor_4d(const_cast<void*>(kcache_ptr),
            batch, seqlen_k, num_heads_k, head_size, dtype);
        t_vcache = make_tensor_4d(const_cast<void*>(vcache_ptr),
            batch, seqlen_k, num_heads_k, head_size, dtype);
    }

    topsatenTensor t_key_empty;
    topsatenTensor t_value_empty;
    topsatenTensor t_cos_empty;
    topsatenTensor t_sin_empty;
    topsatenTensor t_cache_batch_idx_empty;
    topsatenTensor t_leftpad_k_empty;

    auto t_seqlens_k = make_tensor_1d(const_cast<void*>(seqlens_k_ptr),
        batch, TOPSATEN_DATA_I32);

    topsatenTensor t_block_table;
    if (has_block_table && block_table_ptr) {
        t_block_table = make_tensor_2d(const_cast<void*>(block_table_ptr),
            batch, max_num_blocks_per_seq, TOPSATEN_DATA_I32);
    }

    topsatenTensor t_alibi;
    if (alibi_slopes_ptr) {
        t_alibi = make_tensor_1d(const_cast<void*>(alibi_slopes_ptr),
            num_heads, dtype);
    }

    auto s_scale = make_scalar_f64(static_cast<double>(softmax_scale));
    auto s_wl = make_scalar_i64(static_cast<int64_t>(window_size_left));
    auto s_wr = make_scalar_i64(static_cast<int64_t>(window_size_right));
    auto s_sc = make_scalar_f64(static_cast<double>(softcap));
    auto s_ns = make_scalar_i64(static_cast<int64_t>(num_splits));

    std::tuple<topsatenTensor, topsatenTensor> output(t_out, t_lse);

    topsatenStatus_t st = topsfa::topsfaFlashAttnFwdKvcache(
        output, t_q, t_kcache, t_vcache,
        t_key_empty, t_value_empty,
        t_seqlens_k,
        t_cos_empty, t_sin_empty,
        t_cache_batch_idx_empty, t_leftpad_k_empty,
        t_block_table, t_alibi,
        s_scale, is_causal != 0,
        s_wl, s_wr, s_sc,
        is_rotary_interleaved != 0, s_ns,
        stream);

    return static_cast<int>(st);
}

/**
 * General (possibly strided) copy via topsatenCopy.
 *
 * Used as a fallback when the custom ucopy_* kernels would request more
 * shared memory than the device can provide (large non-contiguous tensors).
 *
 * input/output may be non-contiguous; shapes are element shapes and
 * strides are in elements (matching candle Layout).
 */
int topsaten_copy(
    void *out_ptr, const void *in_ptr,
    int32_t ndim,
    const int64_t *out_shape, const int64_t *out_strides,
    const int64_t *in_shape, const int64_t *in_strides,
    int32_t dtype_code, void *stream_) {
    ensure_ops_init();
    topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
    topsatenDataType_t dtype = dtype_from_code(dtype_code);

    auto t_in = make_tensor_strided(const_cast<void*>(in_ptr), ndim,
                                    in_shape, in_strides, dtype);
    auto t_out = make_tensor_strided(out_ptr, ndim,
                                     out_shape, out_strides, dtype);

    topsatenStatus_t st = topsaten::topsatenCopy(t_out, t_in, false, stream);
    return static_cast<int>(st);
}

} // extern "C"
