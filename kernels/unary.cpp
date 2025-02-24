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
 * @file    unary.cpp
 * @brief
 *
 * @author  Guoqing Bao
 * @date    2023-11-07 to 2024-03-04
 * @version V0.1
 * @par     Copyright (c) Enflame Tech Company.
 * @par     History: remove unary op_type
 * @par     Comments: gcu kernel for unary operations.
 */

#include <stdio.h>
#include <tops.h>
#include <tops/topsrtc.h>
#include <tops/half.h>
#include <tops/bfloat.h>
#include <string>
#include <tops/tops_runtime.h>
#include <tops/topsrtc.h>

#include <acore_op.h>
#include "utils/utils.h"
using namespace std;
using namespace tops;

template <typename T>
__forceinline__ __device__ void square(T* out, T* in,
                                            unsigned int len) {}

template <>
__forceinline__ __device__ void  square(float* out, float* in,
                                            unsigned int len) {
  mul(out, in, in, len);
}

template <>
__forceinline__ __device__ void  square(__fp16* out, __fp16* in,
                                            unsigned int len) {
    mul(out, in, in, len);
}

template <>
__forceinline__ __device__ void  square(__bf16* out, __bf16* in,
                                            unsigned int len) {
    mul(out, in, in, len);
}


#if 0
#define UNARY_OP(T, FN_NAME, ATOMIC_FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    T *in, \
    T *out) {\
    tops_dte_ctx_t ctx; \
    tops::dte_scope s(ctx); \ 
    int thread_id = GetThreadIdx();\
    int MAX_THREADS = GetThreadNum();\
    const int TILESIZE = 512 * 1024 / sizeof(T); \
    __local__ __valigned__ T buffer1[TILESIZE]; \
    __local__ __valigned__ T buffer2[TILESIZE]; \
    __shared__ char raw_cache[SHARE_BUFFER_SIZE]; \
    bool cachable = numel * sizeof(T) < SHARE_BUFFER_SIZE; \
    T* sharedBuffer = reinterpret_cast<T*>(raw_cache); \
    tops::mdspan buffer_l1(tops::Private, buffer1, TILESIZE); \
    tops::mdspan out_hbm(tops::Global, out, numel); \
    int N = numel; \
    int THREAD_STEP = 1; \
    int thread_step = 1; \
    GetThreadStep(N, thread_step, THREAD_STEP);\
    if (GetThreadIdxInBlock() == 0 && cachable) { \
      tops::memcpy(ctx, tops::mdspan(tops::Shared, sharedBuffer, numel), tops::mdspan(tops::Global, in, numel)); \
    } \
    __syncthreads(); \
    for (int i = 0; i < thread_step; i+=TILESIZE) { \
      int bufsize = (i + TILESIZE < thread_step) ? TILESIZE : thread_step - i; \
      int offset = thread_id * THREAD_STEP + i; \
      tops::memcpy(ctx, buffer_l1, cachable ? tops::mdspan(tops::Shared, sharedBuffer + offset, bufsize) : tops::mdspan(tops::Global, in + offset, bufsize)); \
      ATOMIC_FUNC(buffer2, buffer1, bufsize); \
      tops::deslice(ctx, out_hbm, tops::mdspan(tops::Private, buffer2, bufsize), {offset}); \
    } \
} \

#endif


#define PING_PONG_SIZE 2
#define TILE_SIZE AlignDown(((VDMEM_VALID_SIZE) / 32), 256)

#define UNARY_OP(T, FN_NAME, ATOMIC_FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    T *in, \
    T *out) {\
  tops_dte_ctx_t ctx; \
  tops::dte_scope s(ctx); \ 
  tops_dte_ctx_t ctxs_in[PING_PONG_SIZE];\
  tops_dte_ctx_t ctxs_out[PING_PONG_SIZE];\
  tops::event evs_in[PING_PONG_SIZE];\
  tops::event evs_out[PING_PONG_SIZE];\
  __local__ __valigned__ T in_buffer[PING_PONG_SIZE][TILE_SIZE];\
  __local__ __valigned__ T out_buffer[PING_PONG_SIZE][TILE_SIZE];\
  int N = numel;\
  tops::mdspan output(tops::Global, out, N);\
  int thread_num = GetThreadNum();\
  int thread_id = GetThreadIdx();\
  int thread_off_leading = thread_id * TILE_SIZE;\
  int thread_len_leading = N - thread_off_leading >= TILE_SIZE ? TILE_SIZE : N - thread_off_leading;\
  int thread_step = TILE_SIZE * thread_num;\
  int thread_off_leading_next = thread_off_leading + thread_step;\
  int thread_remain_leading = N - thread_off_leading_next;\
  int thread_len_leading_next = thread_remain_leading >= TILE_SIZE ? TILE_SIZE : thread_remain_leading;\
  int pp_flag = 0;\
  tops::dte_scope s_in0(ctxs_in[0]);\
  tops::dte_scope s_in1(ctxs_in[1]);\
  tops::dte_scope s_out0(ctxs_out[0]);\
  tops::dte_scope s_out1(ctxs_out[1]);\
  if (thread_len_leading > 0) {\
    ctxs_in[0].config_memcpy(\
        tops::mdspan(tops::Private, in_buffer[pp_flag], thread_len_leading),\
        tops::mdspan(tops::Global, in + thread_off_leading,\
                     thread_len_leading));\
    ctxs_out[0].config_memcpy(\
        tops::mdspan(tops::Global, out + thread_off_leading,\
                     thread_len_leading),\
        tops::mdspan(tops::Private, out_buffer[pp_flag], thread_len_leading));\
    evs_in[pp_flag] = ctxs_in[pp_flag].trigger();\
  }\
  if (thread_len_leading_next > 0) {\
    ctxs_in[1].config_memcpy(\
        tops::mdspan(tops::Private, in_buffer[1],\
                     thread_len_leading_next),\
        tops::mdspan(tops::Global, in + thread_off_leading_next,\
                     thread_len_leading_next));\
    ctxs_out[1].config_memcpy(\
        tops::mdspan(tops::Global, out + thread_off_leading_next,\
                     thread_len_leading_next),\
        tops::mdspan(tops::Private, out_buffer[1],\
                     thread_len_leading_next));\
  }\
  for (int i = thread_off_leading; i < N; i += thread_step) {\
    int pp_flag_next = 1 - pp_flag;\
    int pp_flag_prev = 1 - pp_flag;\
    int thread_off_next = i + thread_step;\
    int thread_remain_next = N - thread_off_next;\
    int thread_len = N - i >= TILE_SIZE ? TILE_SIZE : N - i;\
    int thread_len_next = thread_remain_next >= TILE_SIZE ? TILE_SIZE : thread_remain_next;\
    if (thread_len_next > 0) {\
      evs_in[pp_flag_next] = ctxs_in[pp_flag_next].trigger();\
    }\
    int thread_off_next2 = i + thread_step * 2;\
    int thread_remain_next2 = N - thread_off_next2;\
    int thread_len_next2 =\
        thread_remain_next2 >= TILE_SIZE ? TILE_SIZE : thread_remain_next2;\
    if (thread_len > 0) {\
      evs_in[pp_flag].wait();\
    }\
    if (thread_len_next2 > 0) {\
      ctxs_in[pp_flag].config_memcpy(\
        tops::mdspan(tops::Private, in_buffer[pp_flag],\
                     thread_len_next2),\
        tops::mdspan(tops::Global, in + thread_off_next2,\
                     thread_len_next2));\
    }\
    if (thread_len > 0) {\
      ATOMIC_FUNC(out_buffer[pp_flag], in_buffer[pp_flag], \
                               thread_len);\
      evs_out[pp_flag] = ctxs_out[pp_flag].trigger();\
    }\
    if (i != thread_off_leading) {\
      int thread_off_prev = i - thread_step;\
      int thread_remain_prev = N - thread_off_prev;\
      int thread_len_prev =\
          thread_remain_prev >= TILE_SIZE ? TILE_SIZE : thread_remain_prev;\
      if (thread_len_prev > 0) {\
        evs_out[pp_flag_prev].wait();\
      }\
      if (thread_len_next > 0) {\
        ctxs_out[pp_flag_prev].config_memcpy(\
        tops::mdspan(tops::Global, out + thread_off_next,\
                     thread_len_next),\
        tops::mdspan(tops::Private, out_buffer[pp_flag_prev],\
                     thread_len_next));\
      }\
    }\
    pp_flag = 1 - pp_flag;\
  }\
  if (thread_len_leading > 0) {\
    evs_out[1 - pp_flag].wait();\
  }\
}\


UNARY_OP(__bf16, uneg_bf16, neg)
UNARY_OP(__bf16, uexp_bf16, exp)
UNARY_OP(__bf16, ulog_bf16, log)
UNARY_OP(__bf16, usin_bf16, sin)
UNARY_OP(__bf16, ucos_bf16, cos)
UNARY_OP(__bf16, uabs_bf16, abs)
UNARY_OP(__bf16, usqr_bf16, square)
UNARY_OP(__bf16, usqrt_bf16, sqrt)
UNARY_OP(__bf16, ursqrt_bf16, rsqrt)
UNARY_OP(__bf16, ugelu_bf16, gelu)
UNARY_OP(__bf16, urelu_bf16, relu) 
UNARY_OP(__bf16, usilu_bf16, swish) 
UNARY_OP(__bf16, utanh_bf16, tanh) 
UNARY_OP(__bf16, usigmoid_bf16, sigmoid) 
UNARY_OP(__bf16, urecip_bf16, reciprocal) 
UNARY_OP(__bf16, ugelu_erf_bf16, gelu) //TODO: gelu erf

UNARY_OP(__fp16, uneg_f16, neg)
UNARY_OP(__fp16, uexp_f16, exp)
UNARY_OP(__fp16, ulog_f16, log)
UNARY_OP(__fp16, usin_f16, sin)
UNARY_OP(__fp16, ucos_f16, cos)
UNARY_OP(__fp16, uabs_f16, abs)
UNARY_OP(__fp16, usqr_f16, square)
UNARY_OP(__fp16, usqrt_f16, sqrt)
UNARY_OP(__fp16, ursqrt_f16, rsqrt)
UNARY_OP(__fp16, ugelu_f16, gelu)
UNARY_OP(__fp16, urelu_f16, relu)
UNARY_OP(__fp16, usilu_f16, swish)
UNARY_OP(__fp16, utanh_f16, tanh)
UNARY_OP(__fp16, usigmoid_f16, sigmoid) 
UNARY_OP(__fp16, urecip_f16, reciprocal)
UNARY_OP(__fp16, ugelu_erf_f16, gelu) //TODO: gelu erf

UNARY_OP(float, uneg_f32, neg)
UNARY_OP(float, uexp_f32, exp)
UNARY_OP(float, ulog_f32, log)
UNARY_OP(float, usin_f32, sin)
UNARY_OP(float, ucos_f32, cos)
UNARY_OP(float, uabs_f32, abs)
UNARY_OP(float, usqr_f32, square)
UNARY_OP(float, usqrt_f32, sqrt)
UNARY_OP(float, ursqrt_f32, rsqrt)
UNARY_OP(float, ugelu_f32, gelu)
UNARY_OP(float, urelu_f32, relu)
UNARY_OP(float, usilu_f32, swish)
UNARY_OP(float, utanh_f32, tanh)
UNARY_OP(float, usigmoid_f32, sigmoid) 
UNARY_OP(float, urecip_f32, reciprocal)
UNARY_OP(float, ugelu_erf_f32, gelu) //TODO: gelu erf

#define UNARY_OP1(T, EXTTYPE, FN_NAME, ATOMIC_FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    T *in, \
    T *out, \
    EXTTYPE extValue) \
{ \
    tops_dte_ctx_t ctx; \
    tops::dte_scope s(ctx); \ 
    int thread_id = GetThreadIdx(); \
    int MAX_THREADS = GetThreadNum(); \
    const int TILESIZE = 512 * 1024 / sizeof(T); \
    __local__ __valigned__ T buffer1[TILESIZE]; \
    __local__ __valigned__ T buffer2[TILESIZE]; \
    extern __shared__ char raw_cache[]; \
    bool cachable = numel * sizeof(T) < SHARE_BUFFER_SIZE; \
    T* sharedBuffer = reinterpret_cast<T*>(raw_cache); \
    tops::mdspan buffer_l1(tops::Private, buffer1, TILESIZE); \
    tops::mdspan out_hbm(tops::Global, out, numel); \
    int N = numel; \
    int THREAD_STEP = 1; \
    int thread_step = 1; \
    GetThreadStep(N, thread_step, THREAD_STEP);\
    if (GetThreadIdxInBlock() == 0 && cachable) { \
      tops::mdspan shared_in(tops::Shared, sharedBuffer, numel);\
      tops::mdspan hbm_in(tops::Global, in, numel);\
      tops::memcpy(ctx, shared_in, hbm_in); \
    } \
    __syncthreads(); \
    for (int i = 0; i < thread_step; i+=TILESIZE) { \
      int bufsize = (i + TILESIZE < thread_step) ? TILESIZE : thread_step - i; \
      int offset = thread_id * THREAD_STEP + i; \
      tops::mdspan shared_in(tops::Shared, sharedBuffer + offset, bufsize);\
      tops::mdspan hbm_in(tops::Global, in + offset, bufsize);\
      tops::memcpy(ctx, buffer_l1, cachable ? shared_in : hbm_in); \
      ATOMIC_FUNC(buffer2, buffer1, bufsize, extValue); \
      tops::deslice(ctx, out_hbm, tops::mdspan(tops::Private, buffer2, bufsize), {offset}); \
    } \
} \

UNARY_OP1(__bf16, float, uelu_bf16, elu) 
UNARY_OP1(__fp16, float, uelu_f16, elu)
UNARY_OP1(float, float, uelu_f32, elu)


int main() { }