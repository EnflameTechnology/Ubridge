#include <acore_op.h>
#include "utils/utils.h"
#include <tops.h>
#include <tops/tops_runtime.h>
//Example:
// y = [N, M]
// idx=[K]
// top=[K]
// e_out = [K, M]
// w = [N, topk]
#define MAX_M_DIM 1024 * 8
#define MAX_IDX_DIM 1024 * 4
//case 1: batch = 1, seq_len = 7
//prefiling
//y [7, 5120], e_out [3, 5120], topk_weight [7, 6], idx [3], top [3]
//y [7, 5120], e_out [1, 5120], topk_weight [7, 6], idx [1], top [1]
//decoding
//y [1, 5120], e_out [1, 5120], topk_weight [1, 6], idx [1], top [1]

//case 2: batch==8, seq_len=28
//prefiling
//y [224, 5120], e_out [15, 5120], topk_weight [224, 6], idx [15], top [15]
//decoding
//y [8, 5120], e_out [8, 5120], topk_weight [8, 6], idx [8], top [8]

template<typename T, typename ID_TYPE>
__global__ void moe_kernel(T* y, T* e_out, float* w, ID_TYPE* idx, ID_TYPE* top, int N, int K, int M, int topk) {
    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNum();
    int THREAD_STEP = 1;
    int thread_step = 1;
    __local__ __valigned__ T y_buffer[MAX_M_DIM];
    __local__ __valigned__ T e_buffer[MAX_M_DIM];
    __local__ __valigned__ T tmp_buffer[MAX_M_DIM];

    __local__ __valigned__ ID_TYPE idx_buffer[MAX_IDX_DIM];
    __local__ __valigned__ ID_TYPE top_buffer[MAX_IDX_DIM];
    __local__ __valigned__ float w_buffer[MAX_M_DIM*16];

    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx);
    tops::mdspan l1_y(tops::Private, y_buffer, M);
    tops::mdspan l1_e(tops::Private, e_buffer, M);
    tops::mdspan l1_idx(tops::Private, idx_buffer, K);
    tops::mdspan l1_top(tops::Private, top_buffer, K);
    tops::mdspan l1_w(tops::Private, w_buffer, N * topk);
    
    tops::memcpy(ctx, l1_idx, tops::mdspan(tops::Global, idx, K));
    tops::memcpy(ctx, l1_top, tops::mdspan(tops::Global, top, K));
    tops::memcpy(ctx, l1_w, tops::mdspan(tops::Global, w, N * topk));

    GetThreadStep(K, thread_step, THREAD_STEP);
    for (int i = 0; i < thread_step; i++) {
      int k = thread_id * THREAD_STEP + i;
      if (k < K) {
        if (idx_buffer[k] < N) {
          tops::mdspan hbm_y1(tops::Global, y + idx_buffer[k] * M, M);
          tops::memcpy(ctx, l1_y, hbm_y1);
          tops::mdspan hbm_e1(tops::Global, e_out + k * M, M);
          tops::memcpy(ctx, l1_e, hbm_e1);
          uint32_t top_idx = top_buffer[k];
          if (top_idx < topk) {
            float w1 = w_buffer[idx_buffer[k] * topk + top_idx];
            mul<T, T, float>(reinterpret_cast<T*>(tmp_buffer), reinterpret_cast<T*>(e_buffer), w1, M);
            add(reinterpret_cast<T*>(e_buffer), reinterpret_cast<T*>(y_buffer), reinterpret_cast<T*>(tmp_buffer), M);
            tops::memcpy(ctx, hbm_y1, l1_e);
          } 
        } 
      }
    }
}

#define MOE_OP(T, ID_TYPE, RUST_NAME) \
extern "C" void moe_##RUST_NAME(  \
    T* y, T* e_out, float* w, ID_TYPE* idx, ID_TYPE* top, int N, int K, int M, int topk, void* stream_ \
) { \
    topsStream_t stream = (topsStream_t)(stream_);\
    int numBlocks = 2;\
    int dimBlocks = 12;\
    moe_kernel<T, ID_TYPE><<<2, 12, 0, stream>>>(y, e_out, w, idx, top, N, K, M, topk);\
}\

MOE_OP(__fp16, u_int32_t, f16)
MOE_OP(__bf16, u_int32_t, bf16)