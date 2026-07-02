/**
 * Thin C-linkage wrapper around TopsAten optimized kernels for core ops:
 *   - RMS Normalization (topsvllmRmsNorm)
 *   - Fused Add + RMS Normalization (topsvllmFusedAddRmsNorm)
 *   - Matrix Multiplication (topsatenMatmul)
 *
 * Uses the same graph-aware allocator pattern as topsaten_moe_wrapper.cpp.
 */
#include <cstdint>
#include <cstdio>
#include <atomic>
#include <tops/tops_runtime.h>
#include <topsaten/topsaten_define.h>
#include <topsaten/topsaten_vllm.h>
#include <topsaten/topsaten_ops.h>

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

static topsatenDataType_t dtype_from_code(int32_t code) {
    return static_cast<topsatenDataType_t>(code);
}

extern "C" {

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
 * output_index: [n_rows, k]  (int64_t indices)
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
    topsatenTensor t_output_index(out_sz, out_st, TOPSATEN_DATA_I64,
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
 * output_index:  [n_rows, n_cols]  (int64_t indices)
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
    topsatenTensor t_output_index(out_sz, out_st, TOPSATEN_DATA_I64,
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

} // extern "C"
