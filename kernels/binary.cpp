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
 * @file    binary.cpp
 * @brief
 *
 * @author  Guoqing Bao
 * @date    2023-11-07 - 2024-01-02
 * @version V0.1
 * @par     Copyright (c) Enflame Tech Company.
 * @par     History: Use built-in atomic op
 * @par     Comments: gcu kernel for binary operation, such as add, mul, max, div, eq, etc.
 */

#include <cstdint>
#include <stdio.h>
#include <tops.h>
#include <tops/half.h>
#include <tops/bfloat.h>
#include "tops/tops_runtime.h"
#include <acore_op.h>
#include "utils/utils.h"

using namespace std;

// #define PRINTHELPER(ARRAY, SZ, MSG) \
//   printf(MSG); \
//   for (int i=0; i< SZ; i++) \
//     printf("%d, ", (int)ARRAY[i]); \
//   printf("\n") \

#define TILE_SIZE AlignDown(((VDMEM_VALID_SIZE) / 6), 256)
#define TILE_LEN_BPE4 (TILE_SIZE >> 2)
#define TILE_LEN_BPE2 (TILE_SIZE >> 1)
#define TILE_LEN_BPE1 (TILE_SIZE)
#define PING_PONG_SIZE 2
#define OPERAND_NUM 2

template <typename dst_t, typename lhs_t, typename rhs_t>
__device__ void eqx(dst_t *dst, lhs_t *lhs, rhs_t* rhs, unsigned int num) {
    for(int i=0; i< num; i++) {
      dst[i] = (dst_t)(lhs[i] == rhs[i]? 1 : 0);
    }
}

template <typename dst_t, typename lhs_t, typename rhs_t>
__device__ void eqx_scalar(dst_t *dst, lhs_t *lhs, rhs_t rhs, unsigned int num) {
    for(int i=0; i< num; i++) {
      dst[i] = (dst_t)(lhs[i] == rhs ? 1 : 0);
    }
}

#define BINARY_OP(T, TO, FN_NAME, ATOMIC_FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    size_t *dims_and_strides, \
    T *in_a, \
    T *in_b, \
    TO *out) \
{ \
    int N = (int)numel; \
    tops_dte_ctx_t ctxs_in[OPERAND_NUM][PING_PONG_SIZE];\
    tops_dte_ctx_t ctxs_out[PING_PONG_SIZE];\
    tops::event evs_in[OPERAND_NUM][PING_PONG_SIZE];\
    tops::event evs_out[PING_PONG_SIZE];\
    int thread_num = GetThreadNum();\
    int thread_id = GetThreadIdx();\
    __local__ __valigned__ char in_buffer[OPERAND_NUM][PING_PONG_SIZE][TILE_SIZE];\
    __local__ __valigned__ char out_buffer[PING_PONG_SIZE][TILE_SIZE];\
    int TILE_LEN = sizeof(T) == 4 ? TILE_LEN_BPE4 : \
                      (sizeof(T) == 2 ? TILE_LEN_BPE2 : TILE_LEN_BPE1);\
    int thread_off_leading = thread_id * TILE_LEN;\
    int thread_len_leading = \
        N - thread_off_leading >= TILE_LEN ? TILE_LEN : N - thread_off_leading;\
    int thread_step = TILE_LEN * thread_num;\
    int thread_off_leading_next = thread_off_leading + thread_step;\
    int thread_remain_leading = N - thread_off_leading_next;\
    int thread_len_leading_next = \
        thread_remain_leading >= TILE_LEN ? TILE_LEN : thread_remain_leading;\
    int pp_flag = 0;\
    tops::dte_scope s_a_in0(ctxs_in[0][0]);\
    tops::dte_scope s_a_in1(ctxs_in[0][1]);\
    tops::dte_scope s_b_in0(ctxs_in[1][0]);\
    tops::dte_scope s_b_in1(ctxs_in[1][1]);\
    tops::dte_scope s_out0(ctxs_out[0]);\
    tops::dte_scope s_out1(ctxs_out[1]);\
    if (thread_len_leading > 0) {\
      ctxs_in[0][0].config_memcpy(\
          tops::mdspan(tops::Private,\
                      reinterpret_cast<T*>(in_buffer[0][pp_flag]),\
                      thread_len_leading),\
          tops::mdspan(tops::Global, in_a + thread_off_leading,\
                      thread_len_leading));\
      ctxs_in[1][0].config_memcpy(\
          tops::mdspan(tops::Private,\
                      reinterpret_cast<T*>(in_buffer[1][pp_flag]),\
                      thread_len_leading),\
          tops::mdspan(tops::Global, in_b + thread_off_leading,\
                      thread_len_leading));\
      ctxs_out[0].config_memcpy(\
          tops::mdspan(tops::Global, out + thread_off_leading,\
                      thread_len_leading),\
          tops::mdspan(tops::Private, reinterpret_cast<TO*>(out_buffer[pp_flag]),\
                      thread_len_leading));\
      evs_in[0][pp_flag] = ctxs_in[0][pp_flag].trigger();\
      evs_in[1][pp_flag] = ctxs_in[1][pp_flag].trigger();\
    }\
    if (thread_len_leading_next > 0) {\
      ctxs_in[0][1].config_memcpy(\
          tops::mdspan(tops::Private,\
                      reinterpret_cast<T*>(in_buffer[0][1 - pp_flag]),\
                      thread_len_leading_next),\
          tops::mdspan(tops::Global, in_a + thread_off_leading_next,\
                      thread_len_leading_next));\
      ctxs_in[1][1].config_memcpy(\
          tops::mdspan(tops::Private,\
                      reinterpret_cast<T*>(in_buffer[1][1 - pp_flag]),\
                      thread_len_leading_next),\
          tops::mdspan(tops::Global, in_b + thread_off_leading_next,\
                      thread_len_leading_next));\
      ctxs_out[1].config_memcpy(\
          tops::mdspan(tops::Global, out + thread_off_leading_next,\
                      thread_len_leading_next),\
          tops::mdspan(tops::Private,\
                      reinterpret_cast<TO*>(out_buffer[1 - pp_flag]),\
                      thread_len_leading_next));\
    }\
    for (int i = thread_off_leading; i < N; i += thread_step) {\
      int pp_flag_next = 1 - pp_flag;\
      int pp_flag_prev = 1 - pp_flag;\
      int thread_off_next = i + thread_step;\
      int thread_remain_next = N - thread_off_next;\
      int thread_len = N - i >= TILE_LEN ? TILE_LEN : N - i;\
      int thread_len_next =\
          thread_remain_next >= TILE_LEN ? TILE_LEN : thread_remain_next;\
      if (thread_len_next > 0) {\
        evs_in[0][pp_flag_next] = ctxs_in[0][pp_flag_next].trigger();\
        evs_in[1][pp_flag_next] = ctxs_in[1][pp_flag_next].trigger();\
      }\
      int thread_off_next2 = i + thread_step * 2;\
      int thread_remain_next2 = N - thread_off_next2;\
      int thread_len_next2 =\
          thread_remain_next2 >= TILE_LEN ? TILE_LEN : thread_remain_next2;\
      if (thread_len > 0) {\
        evs_in[0][pp_flag].wait();\
        evs_in[1][pp_flag].wait();\
      }\
      if (thread_len_next2 > 0) {\
        ctxs_in[0][pp_flag].config_memcpy(\
            tops::mdspan(tops::Private,\
                        reinterpret_cast<T*>(in_buffer[0][pp_flag]),\
                        thread_len_next2),\
            tops::mdspan(tops::Global, in_a + thread_off_next2,\
                        thread_len_next2));\
        ctxs_in[1][pp_flag].config_memcpy(\
            tops::mdspan(tops::Private,\
                        reinterpret_cast<T*>(in_buffer[1][pp_flag]),\
                        thread_len_next2),\
            tops::mdspan(tops::Global, in_b + thread_off_next2,\
                        thread_len_next2));\
      }\
      if (thread_len > 0) {\
        ATOMIC_FUNC(reinterpret_cast<TO*>(out_buffer[pp_flag]),\
                  reinterpret_cast<T*>(in_buffer[0][pp_flag]),\
                  reinterpret_cast<T*>(in_buffer[1][pp_flag]),\
                  thread_len);\
        evs_out[pp_flag] = ctxs_out[pp_flag].trigger();\
      }\
      if (i != thread_off_leading) {\
        int thread_off_prev = i - thread_step;\
        int thread_remain_prev = N - thread_off_prev;\
        int thread_len_prev =\
            thread_remain_prev >= TILE_LEN ? TILE_LEN : thread_remain_prev;\
        if (thread_len_prev > 0) {\
          evs_out[pp_flag_prev].wait();\
        }\
        if (thread_len_next > 0) {\
          ctxs_out[pp_flag_prev].config_memcpy(\
              tops::mdspan(tops::Global, out + thread_off_next, thread_len_next),\
              tops::mdspan(tops::Private,\
                          reinterpret_cast<TO*>(out_buffer[pp_flag_prev]),\
                          thread_len_next));\
        }\
      }\
      pp_flag = 1 - pp_flag;\
    }\
    if (thread_len_leading > 0) {\
      evs_out[1 - pp_flag].wait();\
    }\
}\

BINARY_OP(__bf16, __bf16, badd_bf16, add)
BINARY_OP(__bf16, __bf16, bsub_bf16, sub)
BINARY_OP(__bf16, __bf16, bmul_bf16, mul)
BINARY_OP(__bf16, __bf16, bdiv_bf16, div)
BINARY_OP(__bf16, __bf16, bmaximum_bf16, max)
BINARY_OP(__bf16, __bf16, bminimum_bf16, min)
BINARY_OP(__bf16, __bf16, mod_bf16, mod) 

BINARY_OP(__bf16, uint8_t, eq_bf16, eq)
BINARY_OP(__bf16, uint8_t, ne_bf16, ne)
BINARY_OP(__bf16, uint8_t, ge_bf16, ge)
BINARY_OP(__bf16, uint8_t, gt_bf16, gt)
BINARY_OP(__bf16, uint8_t, lt_bf16, lt) 
BINARY_OP(__bf16, uint8_t, le_bf16, le) 


BINARY_OP(__fp16, __fp16, badd_f16, add)
BINARY_OP(__fp16, __fp16, bsub_f16, sub)
BINARY_OP(__fp16, __fp16, bmul_f16, mul)
BINARY_OP(__fp16, __fp16, bdiv_f16, div)
BINARY_OP(__fp16, __fp16, bmaximum_f16, max)
BINARY_OP(__fp16, __fp16, bminimum_f16, min)
BINARY_OP(__fp16, __fp16, mod_f16, mod)

BINARY_OP(__fp16, uint8_t, eq_f16, eq)
BINARY_OP(__fp16, uint8_t, ne_f16, ne)
BINARY_OP(__fp16, uint8_t, ge_f16, ge)
BINARY_OP(__fp16, uint8_t, gt_f16, gt)
BINARY_OP(__fp16, uint8_t, lt_f16, lt)
BINARY_OP(__fp16, uint8_t, le_f16, le)



BINARY_OP(float, float, badd_f32, add)
BINARY_OP(float, float, bsub_f32, sub)
BINARY_OP(float, float, bmul_f32, mul)
BINARY_OP(float, float, bdiv_f32, div)
BINARY_OP(float, float, bmaximum_f32, max)
BINARY_OP(float, float, bminimum_f32, min)
BINARY_OP(float, float, mod_f32, mod)

BINARY_OP(uint32_t, uint32_t, badd_u32, add)
BINARY_OP(uint32_t, uint32_t, bsub_u32, sub)
BINARY_OP(uint32_t, uint32_t, bmul_u32, mul)
BINARY_OP(uint32_t, uint32_t, bdiv_u32, div)
BINARY_OP(uint32_t, uint32_t, bmaximum_u32, max)
BINARY_OP(uint32_t, uint32_t, bminimum_u32, min)
BINARY_OP(uint32_t, uint32_t, mod_u32, mod)

// BINARY_OP(uint32_t, uint8_t, eq_u32, eq)
BINARY_OP(uint32_t, uint8_t, ne_u32, ne)
BINARY_OP(uint32_t, uint8_t, ge_u32, ge)
BINARY_OP(uint32_t, uint8_t, gt_u32, gt)
BINARY_OP(uint32_t, uint8_t, lt_u32, lt)
BINARY_OP(uint32_t, uint8_t, le_u32, le)


BINARY_OP(float, uint8_t, eq_f32, eq)
BINARY_OP(float, uint8_t, ne_f32, ne)
BINARY_OP(float, uint8_t, ge_f32, ge)
BINARY_OP(float, uint8_t, gt_f32, gt)
BINARY_OP(float, uint8_t, lt_f32, lt)
BINARY_OP(float, uint8_t, le_f32, le)


extern "C" __global__ void eq_u32( 
    const size_t numel, 
    const size_t num_dims, 
    size_t *dims_and_strides, 
    uint32_t *in_a, 
    uint32_t *in_b, 
    uint8_t *out) {
    tops_dte_ctx_t ctx; 
    tops::dte_scope s(ctx); 
    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNum();
    const int TILESIZE = 128 * 1024 / sizeof(uint32_t); 
    __local__ __valigned__ uint32_t buffer1[TILESIZE]; 
    __local__ __valigned__ uint32_t buffer2[TILESIZE]; 
    __local__ __valigned__ uint8_t buffer3[TILESIZE]; 

    __local__ __valigned__ size_t dims_strides[128];
    tops::memcpy(ctx, tops::mdspan(tops::Private, dims_strides, num_dims * 3), tops::mdspan(tops::Global, dims_and_strides, num_dims * 3)); 
    size_t r_strides = 1;
    for (int i=0; i<num_dims; i++) {
      r_strides *= dims_strides[2 * num_dims + i];
    }

    tops::mdspan buffera_l1(tops::Private, buffer1, TILESIZE); 
    tops::mdspan bufferb_l1(tops::Private, buffer2, TILESIZE); 
    if (r_strides == 0) {
      tops::memcpy(ctx, bufferb_l1, tops::mdspan(tops::Global, in_b, 1)); 
    }

    int N = numel; 
    int THREAD_STEP = 1; 
    int thread_step = 1; 
    GetThreadStep(N, thread_step, THREAD_STEP);
    for (int i = 0; i < thread_step; i+=TILESIZE) { 
      unsigned int bufsize = (i + TILESIZE < thread_step) ? TILESIZE : thread_step - i; 
      int offset = thread_id * THREAD_STEP; 
      tops::memcpy(ctx, buffera_l1, tops::mdspan(tops::Global, in_a + offset, bufsize)); 
      if (r_strides > 0) {
        tops::memcpy(ctx, bufferb_l1, tops::mdspan(tops::Global, in_b + offset, bufsize)); 
        eqx<uint8_t, uint32_t, uint32_t>(reinterpret_cast<uint8_t*>(buffer3), 
          reinterpret_cast<uint32_t*>(buffer1), reinterpret_cast<uint32_t*>(buffer2), bufsize);
      } else {
          eqx_scalar<uint8_t, uint32_t, uint32_t>(reinterpret_cast<uint8_t*>(buffer3), 
          reinterpret_cast<uint32_t*>(buffer1), buffer2[0], bufsize);
      }
      tops::mdspan out_hbm(tops::Global, out + offset, bufsize); 
      tops::memcpy(ctx, out_hbm, tops::mdspan(tops::Private, buffer3, bufsize));
    } 
} 

int main () {}