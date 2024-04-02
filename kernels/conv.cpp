/*
 * Copyright 2021-2024 Enflame. All Rights Reserved.

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file    conv.cpp
 * @brief
 *
 * @author  Guoqing Bao
 * @date    2024-01-11
 * @version V0.1
 * @par     Copyright (c) Enflame Tech Company.
 * @par     History: Naive conv1d GCU kernel
 * @par     Comments: gcu kernel for convolution (TODO: 2d Conv, Pooling, etc.)
 */
#include <stdio.h>
#include <tops.h>
#include <tops/topsrtc.h>
#include <tops/half.h>
#include <tops/bfloat.h>

#include <tops/tops_runtime.h>
#include <tops/topsrtc.h>
#include "utils.h"
using namespace std;

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
    __shared__ char raw_cache[SHARE_BUFFER_SIZE];

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
      // if (thread_id == 0) {
      //   printf("in [b_size (%lu), c_in (%lu), l_in (%lu)], out [c_out (%lu), c_in (%lu), k_size (%lu)]\n", b_size, c_in, l_in, c_out, c_in, k_size);
      //   printf("Unable to process conv1d because of the input size %d exceed l2 buffer!\n", insize);
      // }
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


// Naive implementation of conv2d.
template <typename T, typename A>
__device__ void conv2d(
    const size_t src_numel,
    const size_t w_out,
    const size_t h_out,
    const size_t stride,
    const size_t padding,
    const size_t dilation,
    size_t *info,
    T *src,
    T *kernel,
    T *dst
) {
  // const size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;

    __local__ __valigned__ size_t infobuf[20];
    __local__ __valigned__ T kernelbuf[1024 * 16];
    __local__ __valigned__ T dstbuf[1024 * 128];
    __shared__ char raw_cache[SHARE_BUFFER_SIZE];

    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx);
    tops::memcpy(ctx, tops::mdspan(tops::Private, infobuf, 16), 
                tops::mdspan(tops::Global, info, 16));

  // src: (b_size, c_in, h_in, w_in)
  // k: (c_out, c_in, h_k, w_k)
    const size_t *src_dims = infobuf;
    const size_t *src_s = infobuf + 4;
    const size_t *k_dims = infobuf + 8;
    const size_t *k_s = infobuf + 12;
    const size_t h_k = k_dims[2];
    const size_t w_k = k_dims[3];
    const size_t c_out = k_dims[0];
    const size_t c_in = src_dims[1];
    const size_t h_in = src_dims[2];
    const size_t w_in = src_dims[3];
    const size_t b_size = src_dims[0];
    int insize = b_size * c_in * h_in * w_in;
    int ksize = c_out * c_in * h_k * w_k;
    bool cacheable = (insize + ksize) * sizeof(T) < SHARE_BUFFER_SIZE;
    if (!cacheable) {
      return;
    }

    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNum();
    int N = b_size * c_out * w_out * h_out;

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

    T* src_cached = reinterpret_cast<T*>(raw_cache);
    T* kernel_cached = src_cached + insize;

    if (thread_id == 0) {
        ctx.config_memcpy(
          tops::mdspan(tops::Shared, src_cached, insize),
          tops::mdspan(tops::Global, src, insize));
        ctx.trigger_and_wait();
        tops::memcpy(ctx, tops::mdspan(tops::Shared, kernel_cached, ksize), 
        tops::mdspan(tops::Global, kernel, ksize));
    }

    __syncthreads();
    for (int i = 0; i < thread_step; i++) {
      int dst_i = thread_id * THREAD_STEP + i;
      if (dst_i < N) {
          const size_t b_idx = dst_i / (w_out * h_out * c_out);
          const size_t dst_c_idx = (dst_i / (w_out * h_out)) % c_out;
          // NCHW layout.
          const size_t dst_h = (dst_i / w_out) % h_out;
          const size_t dst_w = dst_i % w_out;

          const size_t src_idx0 = b_idx * src_s[0];
          A d = 0;
          for (size_t w_offset = 0; w_offset < w_k; ++w_offset) {
            size_t src_w = stride * dst_w + w_offset * dilation;
            if (src_w < padding || src_w >= w_in + padding) {
              continue;
            }
            src_w -= padding;
            for (size_t h_offset = 0; h_offset < h_k; ++h_offset) {
              size_t src_h = stride * dst_h + h_offset * dilation;
              if (src_h < padding || src_h >= h_in + padding) {
                continue;
              }
              src_h -= padding;
              for (size_t src_c_idx = 0; src_c_idx < c_in; ++src_c_idx) {
                const size_t src_idx = src_idx0 + src_c_idx * src_s[1] + src_h * src_s[2] + src_w * src_s[3];
                const size_t k_idx = dst_c_idx * k_s[0] + src_c_idx * k_s[1] + h_offset * k_s[2] + w_offset * k_s[3];
                d += static_cast<A>(src_cached[src_idx]) * static_cast<A>(kernel_cached[k_idx]);
              }
            }
          }
          dstbuf[i] = static_cast<T>(d);
      }
    }
    tops::memcpy(ctx, tops::mdspan(tops::Global, dst + thread_id * THREAD_STEP, thread_step),
              tops::mdspan(tops::Private, dstbuf, thread_step));
}

#define CONV2D_OP(TYPENAME, TYPEACC, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t src_numel, \
    const size_t w_out, \
    const size_t h_out, \
    const size_t stride, \
    const size_t padding, \
    const size_t dilation, \
    size_t *info, \
    TYPENAME *src, \
    TYPENAME *kernel, \
    TYPENAME *dst \
) {  \
  conv2d<TYPENAME, TYPEACC>(src_numel, w_out, h_out, stride, padding, dilation, info, src, kernel, dst); \
} \

CONV2D_OP(__bf16, float, conv2d_bf16)
CONV2D_OP(__fp16, float, conv2d_f16)
CONV2D_OP(float, float, conv2d_f32)
int main() {}