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
 * @file    affine.cpp
 * @brief
 *
 * @author  Guoqing Bao
 * @date    2023-11-09 - 2024-01-02
 * @version V0.1
 * @par     Copyright (c) Enflame Tech Company.
 * @par     History: Use built-in atomic op
 * @par     Comments: gcu kernel for mul & add (with scalar)
 */
#include <stdio.h>
#include <tops.h>
#include <tops/topsrtc.h>
#include <tops/half.h>
#include <tops/bfloat.h>

#include <tops/tops_runtime.h>
#include <tops/topsrtc.h>
#include <krt/scalar.h>
#include <krt/vector_mask.h>
#include <krt/dispatch.h>
#include <krt/leaptr.h>
#include <krt/vector_infra.h>

#include <acore/atomic_op.h>

#include "utils.h"
using namespace std;
using namespace tops;
#define tile_size 0x8000
#define PING_PONG_SIZE 2

enum AFFINE_DATA_TYPE {
    AFFINE_DATA_F32 = 1,
    AFFINE_DATA_FP16 = 2,
    AFFINE_DATA_BF16 = 3,
};


template <typename T, typename VT>
__device__ __forceinline__ void affine_kernel(T* in, T* out, int len, float mulv, float addv, AFFINE_DATA_TYPE tp) {
  tops_dte_ctx_t ctxs_in[PING_PONG_SIZE];
  tops_dte_ctx_t ctxs_out[PING_PONG_SIZE];
  tops_dte_ctx_t ctxs_cp;
  tops::event evs_in[PING_PONG_SIZE];
  tops::event evs_out[PING_PONG_SIZE];
  __local__ __valigned__ T in_buffer[PING_PONG_SIZE][tile_size];
  __local__ __valigned__ T out_buffer[PING_PONG_SIZE][tile_size];
  __local__ __valigned__ T tmp_buffer[PING_PONG_SIZE][tile_size];

  int N = len;
  tops::mdspan output(tops::Global, out, N);

  int thread_num = GetThreadNum();
  int thread_id = GetThreadIdx();

  // printf("thread_num =%d, thread_id=%d\n", thread_num, thread_id);

  int thread_off_leading = thread_id * tile_size;
  int thread_len_leading =
      N - thread_off_leading >= tile_size ? tile_size : N - thread_off_leading;
  int thread_step = tile_size * thread_num;

  int thread_off_leading_next = thread_off_leading + thread_step;
  int thread_remain_leading = N - thread_off_leading_next;
  int thread_len_leading_next =
      thread_remain_leading >= tile_size ? tile_size : thread_remain_leading;

  int pp_flag = 0;
  tops::dte_scope s_in0(ctxs_in[0]);
  tops::dte_scope s_in1(ctxs_in[1]);
  tops::dte_scope s_out0(ctxs_out[0]);
  tops::dte_scope s_out1(ctxs_out[1]);

  // first config pingpong dma completely: d2s/s2d, linear copy
  if (thread_len_leading > 0) {
    ctxs_in[0].config_memcpy(
        tops::mdspan(tops::Private, in_buffer[pp_flag], thread_len_leading),
        tops::mdspan(tops::Global, in + thread_off_leading,
                     thread_len_leading));

    ctxs_out[0].config_memcpy(
        tops::mdspan(tops::Global, out + thread_off_leading,
                     thread_len_leading),
        tops::mdspan(tops::Private, out_buffer[pp_flag], thread_len_leading));

    evs_in[pp_flag] = ctxs_in[pp_flag].trigger();
  }

  if (thread_len_leading_next > 0) {
    ctxs_in[1].config_memcpy(
        tops::mdspan(tops::Private, in_buffer[1],
                     thread_len_leading_next),
        tops::mdspan(tops::Global, in + thread_off_leading_next,
                     thread_len_leading_next));

    ctxs_out[1].config_memcpy(
        tops::mdspan(tops::Global, out + thread_off_leading_next,
                     thread_len_leading_next),
        tops::mdspan(tops::Private, out_buffer[1],
                     thread_len_leading_next));
  }

  for (int i = thread_off_leading; i < N; i += thread_step) {
    int pp_flag_next = 1 - pp_flag;
    int pp_flag_prev = 1 - pp_flag;
    int thread_off_next = i + thread_step;
    int thread_remain_next = N - thread_off_next;
    int thread_len = N - i >= tile_size ? tile_size : N - i;
    int thread_len_next =
        thread_remain_next >= tile_size ? tile_size : thread_remain_next;
    if (thread_len_next > 0) {
      evs_in[pp_flag_next] = ctxs_in[pp_flag_next].trigger();
    }

    int thread_off_next2 = i + thread_step * 2;
    int thread_remain_next2 = N - thread_off_next2;
    int thread_len_next2 =
        thread_remain_next2 >= tile_size ? tile_size : thread_remain_next2;

    if (thread_len > 0) {
      evs_in[pp_flag].wait();
    }

    if (thread_len_next2 > 0) {
      // ctxs_in[pp_flag].set_dst_addr(in_buffer[pp_flag]);
      // ctxs_in[pp_flag].set_src_addr(in + thread_off_next2);
      // ctxs_in[pp_flag].set_total_size(thread_len_next2);
      ctxs_in[pp_flag].config_memcpy(
        tops::mdspan(tops::Private, in_buffer[pp_flag],
                     thread_len_next2),
        tops::mdspan(tops::Global, in + thread_off_next2,
                     thread_len_next2));
    }

    // call atomic op here
    if (thread_len > 0) {
    //   unary_atomic<T, VT>(in_buffer[pp_flag], out_buffer[pp_flag],
    //                            thread_len, tp);

        if (tp == AFFINE_DATA_FP16) {
            tops::half mulv_a = tops::half(mulv);
            tops::half addv_a = tops::half(addv);
            mul_fp16_scalar(reinterpret_cast<__fp16 *>(tmp_buffer[pp_flag]),
                            reinterpret_cast<__fp16 *>(in_buffer[pp_flag]),
                            *reinterpret_cast<__fp16*>(&mulv_a.value), thread_len);
            add_fp16_scalar(reinterpret_cast<__fp16 *>(out_buffer[pp_flag]),
                            reinterpret_cast<__fp16 *>(tmp_buffer[pp_flag]),
                            *reinterpret_cast<__fp16*>(&addv_a.value), thread_len);
        } else if (tp == AFFINE_DATA_F32) {
            mul_fp32_scalar(reinterpret_cast<float *>(tmp_buffer[pp_flag]),
                            reinterpret_cast<float *>(in_buffer[pp_flag]),
                            *reinterpret_cast<float*>(&mulv), thread_len);
            add_fp32_scalar(reinterpret_cast<float *>(out_buffer[pp_flag]),
                            reinterpret_cast<float *>(tmp_buffer[pp_flag]),
                            *reinterpret_cast<float*>(&addv), thread_len);
        } else if (tp == AFFINE_DATA_BF16) {
            tops::bfloat mulv_a = tops::bfloat(mulv);
            tops::bfloat addv_a = tops::bfloat(addv);

            mul_bf16_scalar(reinterpret_cast<__bf16 *>(tmp_buffer[pp_flag]),
                            reinterpret_cast<__bf16 *>(in_buffer[pp_flag]),
                            *reinterpret_cast<__bf16 *>(&mulv_a.value), thread_len);
            add_bf16_scalar(reinterpret_cast<__bf16 *>(out_buffer[pp_flag]),
                            reinterpret_cast<__bf16 *>(tmp_buffer[pp_flag]),
                            *reinterpret_cast<__bf16 *>(&addv_a.value), thread_len);
        } 

      evs_out[pp_flag] = ctxs_out[pp_flag].trigger();
    }

    if (i != thread_off_leading) {
      int thread_off_prev = i - thread_step;
      int thread_remain_prev = N - thread_off_prev;
      int thread_len_prev =
          thread_remain_prev >= tile_size ? tile_size : thread_remain_prev;
      if (thread_len_prev > 0) {
        evs_out[pp_flag_prev].wait();
      }

      if (thread_len_next > 0) {
        // ctxs_out[pp_flag_prev].set_dst_addr(out + thread_off_next);
        // ctxs_out[pp_flag_prev].set_src_addr(out_buffer[pp_flag_prev]);
        // ctxs_out[pp_flag_prev].set_total_size(thread_len_next);
        ctxs_out[pp_flag_prev].config_memcpy(
        tops::mdspan(tops::Global, out + thread_off_next,
                     thread_len_next),
        tops::mdspan(tops::Private, out_buffer[pp_flag_prev],
                     thread_len_next));
      }
    }
    pp_flag = 1 - pp_flag;
  }

  if (thread_len_leading > 0) {
    evs_out[1 - pp_flag].wait();
  }
}

#define AFFINE_OP(TYPE, VT, FN_NAME, TP) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    TYPE *inp, \
    TYPE *out, \
    float mulv, \
    float addv) \
{ \
    affine_kernel<TYPE, VT>(inp, out, numel, mulv, addv, TP); \
} \


AFFINE_OP(__bf16, vbfloat, affine_bf16, AFFINE_DATA_BF16)
AFFINE_OP(__fp16, vhalf, affine_f16, AFFINE_DATA_FP16)
AFFINE_OP(float, vfloat, affine_f32, AFFINE_DATA_F32)

template<typename T>
int test() {
  T *lhs_d, *out_d;
  int *shape_lhs_d;
  T *lhs_h, *out_h;
  size_t size_lhs = 1024;
  size_t size_out = size_lhs;
  size_t dim = 1;
  topsHostMalloc((T**)&lhs_h, size_lhs * sizeof(T));
  topsHostMalloc((T**)&out_h, size_out * sizeof(T));
    T a = 0.5;
    T zero = 0.0;
    for (size_t i = 0; i < size_lhs; i++) {
        lhs_h[i] = static_cast<T>(a);
    }
    for (size_t i = 0; i < size_out; i++) {
        out_h[i] = static_cast<T>(zero);
    }
  topsMalloc(&lhs_d, size_lhs * sizeof(T));
  topsMalloc(&out_d, size_out * sizeof(T));

  printf("info: copy Host2Device\n");
  topsMemcpy(lhs_d, lhs_h, size_lhs * sizeof(T),
                  topsMemcpyHostToDevice);
  topsMemcpy(out_d, out_h, size_out * sizeof(T),
                  topsMemcpyHostToDevice);
  int grids = size_out/tile_size;
  int threads = 6;
  if (grids < 1) {
    grids = 1;
  } else if (grids / 6 > 0) {
    grids = grids / 6;
  } else {
    threads = 1;
  }
  float mulv = 0.5;
  float addv = 1.256;
  affine_f16<<<dim3(grids, 1, 1), dim3(threads, 1, 1)>>>(size_out, lhs_d, out_d, mulv, addv);

  printf("info: copy Device2Host\n");
  topsMemcpy(out_h, out_d, size_out * sizeof(T), topsMemcpyDeviceToHost);

  for (size_t j = 0; j < size_out; j++) {
      printf("%.2f, ", float(out_h[j]));
  }
  topsHostFree(lhs_h);
  topsHostFree(out_h);
  topsFree(lhs_d);
  topsFree(out_d);
  return 0;
}

int main() {
    return test<__fp16>();
}