#include <stdio.h>
#include <tops.h>
#include <tops/topsrtc.h>
#include <tops/half.h>
#include <tops/bfloat.h>

#include <tops/tops_runtime.h>
#include <tops/topsrtc.h>
#include "utils.h"
using namespace std;

#define SHARE_BUFFER_SIZE 1024 * 1024 * 24 //24MB
__shared__ char raw_cache[SHARE_BUFFER_SIZE];
// Naive implementation of conv1d.
template <typename T, typename A>
__device__ void conv1d(
    const size_t src_numel,
    const size_t l_out,
    const size_t stride,
    const size_t padding,
    const size_t dilation,
    size_t *info,
    T *src,
    T *kernel,
    T *dst
) {
    // src: (b_size, c_in, l_in)
    // k: (c_out, c_in, k_size)

    __local__ __valigned__ size_t infobuf[20];
    __local__ __valigned__ T kernelbuf[1024 * 4];
    __local__ __valigned__ T dstbuf[1024 * 128];

    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx);
    tops::memcpy(ctx, tops::mdspan(tops::Private, infobuf, 12), 
                tops::mdspan(tops::Global, info, 12));

    const size_t *src_dims = infobuf;
    const size_t *src_s = infobuf + 3;
    const size_t *k_dims = infobuf + 6;
    const size_t *k_s = infobuf + 9;
    const size_t k_size = k_dims[2];
    const size_t c_out = k_dims[0];
    const size_t c_in = src_dims[1];
    const size_t l_in = src_dims[2];
    const size_t b_size = src_dims[0];

    tops::memcpy(ctx, tops::mdspan(tops::Private, kernelbuf, c_out * c_in * k_size), 
                tops::mdspan(tops::Global, kernel, c_out * c_in * k_size));


    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNum();
    int N = b_size * c_out * l_out;

    int THREAD_STEP = 1;
    int thread_step = 1;
    if (N > MAX_THREADS) {
      THREAD_STEP = N / MAX_THREADS;
      thread_step = THREAD_STEP;
      if (N % MAX_THREADS != 0) {
        if (thread_id == MAX_THREADS - 1) {
          thread_step += N % MAX_THREADS; //last thread also process remains
        }
      }
    }
    int insize = b_size * c_in * l_in;
    bool cacheable = insize * sizeof(T) < SHARE_BUFFER_SIZE;
    if (!cacheable) {
      if (thread_id == 0) {
        printf("in [b_size (%lu), c_in (%lu), l_in (%lu)], out [c_out (%lu), c_in (%lu), k_size (%lu)]\n", b_size, c_in, l_in, c_out, c_in, k_size);
        printf("Unable to process conv1d because of the input size %d exceed l2 buffer!\n", insize);
      }
      return;
    }
    T* src_cached = reinterpret_cast<T*>(raw_cache);
    if (thread_id == 0) {
        ctx.config_memcpy(
          tops::mdspan(tops::Shared, src_cached, b_size * c_in * l_in),
          tops::mdspan(tops::Global, src, b_size * c_in * l_in));
        ctx.trigger_and_wait();
    }

    for (int i = 0; i < thread_step; i++) {
      int dst_i = thread_id * THREAD_STEP + i;
      if (dst_i < N) {
        size_t b_idx = dst_i / (l_out * c_out);
        size_t dst_c_idx = (dst_i / l_out) % c_out;
        size_t dst_l = dst_i % l_out;

        size_t src_idx0 = b_idx * src_s[0];
        A d = 0;
        for (size_t offset = 0; offset < k_size; ++offset) {
            size_t src_l = (stride * dst_l + offset) * dilation;
            if (src_l < padding || src_l >= padding + l_in) {
                continue;
            }
            src_l -= padding;
            for (size_t src_c_idx = 0; src_c_idx < c_in; ++src_c_idx) {
                size_t src_idx = src_idx0 + src_c_idx * src_s[1] + src_l * src_s[2];
                size_t k_idx = dst_c_idx * k_s[0] + src_c_idx * k_s[1] + offset * k_s[2];
                d += static_cast<A>(src_cached[src_idx]) * static_cast<A>(kernelbuf[k_idx]);
            }
        }
        dstbuf[i] = static_cast<T>(d);
      }
    }
    tops::memcpy(ctx, tops::mdspan(tops::Global, dst + thread_id * THREAD_STEP, thread_step),
                tops::mdspan(tops::Private, dstbuf, thread_step));
}

#define CONV1D_OP(TYPENAME, TYPEACC, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t src_numel, \
    const size_t num_dims, \
    const size_t stride, \
    const size_t padding, \
    const size_t dilation, \
    size_t *info, \
    TYPENAME *src, \
    TYPENAME *kernel, \
    TYPENAME *dst \
) {  \
  conv1d<TYPENAME, TYPEACC>(src_numel, num_dims, stride, padding, dilation, info, src, kernel, dst); \
} \

CONV1D_OP(float, float, conv1d_f32)
CONV1D_OP(double, double, conv1d_f64)
CONV1D_OP(uint8_t, uint8_t, conv1d_u8)
CONV1D_OP(uint32_t, uint32_t, conv1d_u32)
CONV1D_OP(__bf16, float, conv1d_bf16)
CONV1D_OP(__fp16, float, conv1d_f16)

int main() {}