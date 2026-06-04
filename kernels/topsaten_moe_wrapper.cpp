/**
 * Thin C-linkage wrapper around topsaten (topsvllmTopkSoftmax,
 * topsvllmMoeAlignBlockSize, topsvllmInvokeFusedMoeNonGatherKernel).
 *
 * Registers graph-aware malloc/free with topsaten so that workspace
 * allocations inside topsaten kernels are handled correctly during
 * GCU graph capture/replay — mirroring what torch_gcu does via
 * topsatenMallocAsyncFuncRegister / topsatenFreeAsyncFuncRegister.
 */
#include <cstdint>
#include <cstdio>
#include <atomic>
#include <tops/tops_runtime.h>
#include <topsaten/topsaten_define.h>
#include <topsaten/topsaten_vllm.h>

// ---------------------------------------------------------------------------
// Graph-aware allocator for topsaten kernel workspaces.
//
// During graph capture, topsMallocAsync / topsFreeAsync are recorded as
// graph alloc/free nodes by the TOPS runtime.  On replay the runtime
// manages those allocations automatically.
//
// Outside of capture the normal async path is used.
// ---------------------------------------------------------------------------

static topsError_t graph_aware_malloc_async(void **ptr, size_t size,
                                            topsStream_t stream, uint64_t) {
    return topsMallocAsync(ptr, size, stream, 0);
}

static topsError_t graph_aware_free_async(void *ptr, topsStream_t stream) {
    return topsFreeAsync(ptr, stream);
}

static std::atomic<bool> g_topsaten_initialized{false};

static void ensure_init() {
    bool expected = false;
    if (g_topsaten_initialized.compare_exchange_strong(expected, true)) {
        topsatenStatus_t st = topsatenInit();
        if (st != TOPSATEN_STATUS_SUCCESS) {
            fprintf(stderr, "[topsaten_moe_wrapper] topsatenInit failed: %d\n",
                    static_cast<int>(st));
            g_topsaten_initialized.store(false);
            return;
        }
        topsatenMallocAsyncFuncRegister(graph_aware_malloc_async);
        topsatenFreeAsyncFuncRegister(graph_aware_free_async);
    }
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

static topsatenTensor make_tensor_1d(void *data, int64_t d0,
                                     topsatenDataType_t dtype) {
    int64_t dims[1] = {d0};
    int64_t strides[1] = {1};
    topsatenSize_t dim_size(dims, 1);
    topsatenSize_t stride_size(strides, 1);
    return topsatenTensor(dim_size, stride_size, dtype,
                          static_cast<topsatenDeviceMemHandle_t>(data));
}

extern "C" {

/**
 * topk_softmax via topsaten — replaces the naive ubridge kernel.
 *
 * All buffers must be pre-allocated by the caller (required for graph safety).
 */
int topsaten_topk_softmax_f32(
    float *gating_output, float *topk_weights, int32_t *topk_indices,
    int32_t *token_expert_indices,
    int32_t num_tokens, int32_t num_experts, int32_t topk,
    int32_t norm_topk_prob, void *stream_) {
    ensure_init();

    topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);

    auto t_gating = make_tensor_2d(gating_output, num_tokens, num_experts,
                                   TOPSATEN_DATA_FP32);
    auto t_weights = make_tensor_2d(topk_weights, num_tokens, topk,
                                    TOPSATEN_DATA_FP32);
    auto t_indices = make_tensor_2d(topk_indices, num_tokens, topk,
                                    TOPSATEN_DATA_I32);
    auto t_tei = make_tensor_2d(token_expert_indices, num_tokens, topk,
                                TOPSATEN_DATA_I32);

    topsatenStatus_t st = topsvllm::topsvllmTopkSoftmax(
        t_weights, t_indices, t_tei, t_gating,
        norm_topk_prob != 0, stream);

    return static_cast<int>(st);
}

/**
 * moe_align_block_size via topsaten — replaces the naive ubridge kernel.
 */
int topsaten_moe_align_block_size(
    int32_t *sorted_token_ids, int32_t *experts_ids,
    int32_t *num_tokens_post_pad, int32_t *topk_ids,
    int32_t num_tokens, int32_t topk, int32_t num_experts,
    int32_t block_size, void *stream_) {
    ensure_init();

    topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);

    int64_t total_tokens = static_cast<int64_t>(num_tokens);
    int64_t total_topk = static_cast<int64_t>(topk);
    int64_t m = total_tokens * total_topk;
    int64_t max_padded = m + static_cast<int64_t>(num_experts) *
                             (static_cast<int64_t>(block_size) - 1);
    int64_t max_blocks = max_padded / static_cast<int64_t>(block_size);

    auto t_topk_ids = make_tensor_2d(topk_ids, total_tokens, total_topk,
                                     TOPSATEN_DATA_I32);
    auto t_sorted = make_tensor_1d(sorted_token_ids, max_padded,
                                   TOPSATEN_DATA_I32);
    auto t_experts = make_tensor_1d(experts_ids, max_blocks,
                                    TOPSATEN_DATA_I32);
    auto t_num_pad = make_tensor_1d(num_tokens_post_pad, 1,
                                    TOPSATEN_DATA_I32);

    topsatenStatus_t st = topsvllm::topsvllmMoeAlignBlockSize(
        t_sorted, t_experts, t_num_pad, t_topk_ids,
        static_cast<int>(num_experts), static_cast<int>(block_size), stream);

    return static_cast<int>(st);
}

/**
 * invoke_fused_moe_kernel via topsaten — replaces the naive ubridge kernel.
 *
 * C:   [max_padded, topk, N] output (bf16/fp16/fp32)
 * A:   [num_tokens, K]       input tokens
 * B:   [num_experts, N, K]   expert weights (transposed)
 * topk_weights:   [num_tokens, topk]  f32
 * topk_ids:       [num_tokens, topk]  i32
 * sorted_token_ids: [max_padded]      i32
 * expert_ids:     [max_blocks]        i32
 * num_tokens_post_padded: [1]         i32
 */
int topsaten_fused_moe_gemm(
    void *c_ptr, void *a_ptr, void *b_ptr,
    float *topk_weights, int32_t *topk_ids,
    int32_t *sorted_token_ids, int32_t *expert_ids,
    int32_t *num_tokens_post_padded,
    int32_t mul_routed_weight,
    int32_t topk_val, int32_t block_size,
    int32_t num_tokens, int32_t hidden_dim, int32_t inter_dim,
    int32_t num_experts,
    int32_t dtype_code, // 4=fp16, 5=bf16, 8=fp32
    void *stream_) {
    ensure_init();

    topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
    topsatenDataType_t dtype = static_cast<topsatenDataType_t>(dtype_code);

    int64_t m = static_cast<int64_t>(num_tokens) * static_cast<int64_t>(topk_val);
    int64_t max_padded = m + static_cast<int64_t>(num_experts) *
                             (static_cast<int64_t>(block_size) - 1);
    int64_t max_blocks = max_padded / static_cast<int64_t>(block_size);

    // C: [num_tokens, topk, inter_dim]
    int64_t c_dims[3] = {num_tokens, topk_val, inter_dim};
    int64_t c_strides[3] = {(int64_t)topk_val * inter_dim, inter_dim, 1};
    topsatenSize_t c_dim_size(c_dims, 3);
    topsatenSize_t c_stride_size(c_strides, 3);
    topsatenTensor t_c(c_dim_size, c_stride_size, dtype,
                       static_cast<topsatenDeviceMemHandle_t>(c_ptr));

    // A: [num_tokens, hidden_dim]
    auto t_a = make_tensor_2d(a_ptr, num_tokens, hidden_dim, dtype);
    // B: [num_experts, inter_dim, hidden_dim]
    int64_t b_dims[3] = {num_experts, inter_dim, hidden_dim};
    int64_t b_strides[3] = {(int64_t)inter_dim * hidden_dim, hidden_dim, 1};
    topsatenSize_t b_dim_size(b_dims, 3);
    topsatenSize_t b_stride_size(b_strides, 3);
    topsatenTensor t_b(b_dim_size, b_stride_size, dtype,
                       static_cast<topsatenDeviceMemHandle_t>(b_ptr));

    auto t_topk_w = make_tensor_2d(topk_weights, num_tokens, topk_val,
                                   TOPSATEN_DATA_FP32);
    auto t_topk_ids = make_tensor_2d(topk_ids, num_tokens, topk_val,
                                     TOPSATEN_DATA_I32);
    auto t_sorted = make_tensor_1d(sorted_token_ids, max_padded,
                                   TOPSATEN_DATA_I32);
    auto t_experts = make_tensor_1d(expert_ids, max_blocks,
                                    TOPSATEN_DATA_I32);
    auto t_num_pad = make_tensor_1d(num_tokens_post_padded, 1,
                                    TOPSATEN_DATA_I32);

    topsatenStatus_t st = topsvllm::topsvllmInvokeFusedMoeNonGatherKernel(
        t_c, t_a, t_b, t_topk_w, t_topk_ids,
        t_sorted, t_experts, t_num_pad,
        mul_routed_weight != 0, static_cast<int>(topk_val),
        static_cast<int>(block_size), stream);

    return static_cast<int>(st);
}

/**
 * No-op — workspace allocations during graph capture are now managed by the
 * TOPS runtime via async alloc/free graph nodes. Kept for FFI compatibility.
 */
void topsaten_release_graph_workspace_pool(void *stream_) {
    (void)stream_;
}

} // extern "C"
