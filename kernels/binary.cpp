/**
 * Copyright 2020-2023 Enflame. All Rights Reserved.
 *
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
 */

#include <cstdint>
#include <stdio.h>
#include <tops.h>
#include <tops/half.h>
#include <tops/bfloat.h>

#include "tops/tops_runtime.h"
#include "utils.h"

#if __GCU_ARCH__ >= 300
#include "include/common/atomic_op.h"
#include "include/common/binary.h"
#endif

using namespace std;

#define VDRAM_SIZE 0x180000
#define TILE_SIZE (VDRAM_SIZE / 6)
#define TILE_LEN_BPE4 (TILE_SIZE >> 2)
#define TILE_LEN_BPE2 (TILE_SIZE >> 1)
#define TILE_LEN_BPE1 (TILE_SIZE)
#define TILE_SIZE_SCALAR (VDRAM_SIZE / 4)
#define TILE_LEN_SCALAR_BPE4 (TILE_SIZE_SCALAR >> 2)
#define TILE_LEN_SCALAR_BPE2 (TILE_SIZE_SCALAR >> 1)
#define TILE_LEN_SCALAR_BPE1 (TILE_SIZE_SCALAR)

#define PING_PONG_SIZE 2
#define OPERAND_NUM 2


enum BINARY_TYPE {
    BINARY_TYPE_ADD = 1,
    BINARY_TYPE_SUB = 2,
    BINARY_TYPE_MUL = 3,
    BINARY_TYPE_DIV = 4,
    BINARY_TYPE_MAX = 5,
    BINARY_TYPE_MIN = 6,
    BINARY_TYPE_EQ = 7,
    BINARY_TYPE_NE = 8,
    BINARY_TYPE_GE = 9,
    BINARY_TYPE_GT = 10,
    BINARY_TYPE_LT = 11,
    BINARY_TYPE_LE = 12,
    BINARY_TYPE_MOD = 13,
};


template <typename T>
__device__ __forceinline__ void binary_atomic1(T* out, T* lhs, T* rhs, unsigned int len, BINARY_TYPE tp)
{
  #if __GCU_ARCH__ >= 300 
  switch (tp) {
    case BINARY_TYPE_ADD:
      {
        add(out, lhs, rhs, len);
        break;
      }
    case BINARY_TYPE_SUB:
      {
        sub(out, lhs, rhs, len);
        break;
      }
    case BINARY_TYPE_MUL:
      {
        mul(out, lhs, rhs, len);
        break;
      }
    case BINARY_TYPE_DIV:
      {
        div(out, lhs, rhs, len);
        break;
      }
    case BINARY_TYPE_MAX:
      {
        max(out, lhs, rhs, len);
        break;
      }
    case BINARY_TYPE_MIN:
      {
        min(out, lhs, rhs, len);
        break;
      }
    case BINARY_TYPE_MOD:
      {
        mod(out, lhs, rhs, len);
        break;
      }
    default:
      break;
    }
  #endif
}


template <typename T, typename TO>
__device__ __forceinline__ void binary_atomic2(TO* out, T* lhs, T* rhs, unsigned int len, BINARY_TYPE tp)
{
  #if __GCU_ARCH__ >= 300 
  switch (tp) {
    case BINARY_TYPE_EQ:
      {
        eq(reinterpret_cast<uint8_t*>(out), lhs, rhs, len);
        break;
      }
    case BINARY_TYPE_NE: {
        ne(reinterpret_cast<uint8_t*>(out), lhs, rhs, len);
        break;
    }
    case BINARY_TYPE_GE:
      {
        ge(reinterpret_cast<uint8_t*>(out), lhs, rhs, len);
        break;
      }
    case BINARY_TYPE_GT:
      {
        gt(reinterpret_cast<uint8_t*>(out), lhs, rhs, len);
        break;
      }
    case BINARY_TYPE_LT:
      {
        lt(reinterpret_cast<uint8_t*>(out), lhs, rhs, len);
        
        break;
      }
    case BINARY_TYPE_LE:
      {
        le(reinterpret_cast<uint8_t*>(out), lhs, rhs, len);
        break;
      }
    default:
      break;
    }
  #endif
}

template <typename T, typename TO>
__device__ void binary_kernel(T* in_a, T* in_b, TO* out, int N, BINARY_TYPE tp) {
  tops_dte_ctx_t ctxs_in[OPERAND_NUM][PING_PONG_SIZE];
  tops_dte_ctx_t ctxs_out[PING_PONG_SIZE];
  tops::event evs_in[OPERAND_NUM][PING_PONG_SIZE];
  tops::event evs_out[PING_PONG_SIZE];

  int thread_num = GetThreadNum();
  int thread_id = GetThreadIdx();

  __local__ __valigned__ char in_buffer[OPERAND_NUM][PING_PONG_SIZE][TILE_SIZE];
  __local__ __valigned__ char out_buffer[PING_PONG_SIZE][TILE_SIZE];

  int TILE_LEN = sizeof(T) == 4 ? TILE_LEN_BPE4 :
                    (sizeof(T) == 2 ? TILE_LEN_BPE2 : TILE_LEN_BPE1);
  // printf("thread_num =%d, thread_id=%d\n", thread_num, thread_id);
  int thread_off_leading = thread_id * TILE_LEN;
  int thread_len_leading =
      N - thread_off_leading >= TILE_LEN ? TILE_LEN : N - thread_off_leading;
  int thread_step = TILE_LEN * thread_num;

  int thread_off_leading_next = thread_off_leading + thread_step;
  int thread_remain_leading = N - thread_off_leading_next;
  int thread_len_leading_next =
      thread_remain_leading >= TILE_LEN ? TILE_LEN : thread_remain_leading;

  int pp_flag = 0;
  tops::dte_scope s_a_in0(ctxs_in[0][0]);
  tops::dte_scope s_a_in1(ctxs_in[0][1]);
  tops::dte_scope s_b_in0(ctxs_in[1][0]);
  tops::dte_scope s_b_in1(ctxs_in[1][1]);
  tops::dte_scope s_out0(ctxs_out[0]);
  tops::dte_scope s_out1(ctxs_out[1]);

  // first config pingpong dma completely: d2s/s2d, linear copy
  if (thread_len_leading > 0) {
    ctxs_in[0][0].config_memcpy(
        tops::mdspan(tops::Private,
                     reinterpret_cast<T*>(in_buffer[0][pp_flag]),
                     thread_len_leading),
        tops::mdspan(tops::Global, in_a + thread_off_leading,
                     thread_len_leading));
    ctxs_in[1][0].config_memcpy(
        tops::mdspan(tops::Private,
                     reinterpret_cast<T*>(in_buffer[1][pp_flag]),
                     thread_len_leading),
        tops::mdspan(tops::Global, in_b + thread_off_leading,
                     thread_len_leading));

    ctxs_out[0].config_memcpy(
        tops::mdspan(tops::Global, out + thread_off_leading,
                     thread_len_leading),
        tops::mdspan(tops::Private, reinterpret_cast<TO*>(out_buffer[pp_flag]),
                     thread_len_leading));

    evs_in[0][pp_flag] = ctxs_in[0][pp_flag].trigger();
    evs_in[1][pp_flag] = ctxs_in[1][pp_flag].trigger();
  }

  if (thread_len_leading_next > 0) {
    ctxs_in[0][1].config_memcpy(
        tops::mdspan(tops::Private,
                     reinterpret_cast<T*>(in_buffer[0][1 - pp_flag]),
                     thread_len_leading_next),
        tops::mdspan(tops::Global, in_a + thread_off_leading_next,
                     thread_len_leading_next));
    ctxs_in[1][1].config_memcpy(
        tops::mdspan(tops::Private,
                     reinterpret_cast<T*>(in_buffer[1][1 - pp_flag]),
                     thread_len_leading_next),
        tops::mdspan(tops::Global, in_b + thread_off_leading_next,
                     thread_len_leading_next));

    ctxs_out[1].config_memcpy(
        tops::mdspan(tops::Global, out + thread_off_leading_next,
                     thread_len_leading_next),
        tops::mdspan(tops::Private,
                     reinterpret_cast<TO*>(out_buffer[1 - pp_flag]),
                     thread_len_leading_next));
  }

  for (int i = thread_off_leading; i < N; i += thread_step) {
    int pp_flag_next = 1 - pp_flag;
    int pp_flag_prev = 1 - pp_flag;
    int thread_off_next = i + thread_step;
    int thread_remain_next = N - thread_off_next;
    int thread_len = N - i >= TILE_LEN ? TILE_LEN : N - i;
    int thread_len_next =
        thread_remain_next >= TILE_LEN ? TILE_LEN : thread_remain_next;
    if (thread_len_next > 0) {
      evs_in[0][pp_flag_next] = ctxs_in[0][pp_flag_next].trigger();
      evs_in[1][pp_flag_next] = ctxs_in[1][pp_flag_next].trigger();
    }

    int thread_off_next2 = i + thread_step * 2;
    int thread_remain_next2 = N - thread_off_next2;
    int thread_len_next2 =
        thread_remain_next2 >= TILE_LEN ? TILE_LEN : thread_remain_next2;

    if (thread_len > 0) {
      evs_in[0][pp_flag].wait();
      evs_in[1][pp_flag].wait();
    }

    if (thread_len_next2 > 0) {
      // ctxs_in[0][pp_flag].set_src_addr(in_a + thread_off_next2);
      // ctxs_in[0][pp_flag].set_total_size(thread_len_next2);

      // ctxs_in[1][pp_flag].set_src_addr(in_b + thread_off_next2);
      // ctxs_in[1][pp_flag].set_total_size(thread_len_next2);
      ctxs_in[0][pp_flag].config_memcpy(
          tops::mdspan(tops::Private,
                      reinterpret_cast<T*>(in_buffer[0][pp_flag]),
                      thread_len_next2),
          tops::mdspan(tops::Global, in_a + thread_off_next2,
                      thread_len_next2));
      ctxs_in[1][pp_flag].config_memcpy(
          tops::mdspan(tops::Private,
                      reinterpret_cast<T*>(in_buffer[1][pp_flag]),
                      thread_len_next2),
          tops::mdspan(tops::Global, in_b + thread_off_next2,
                      thread_len_next2));
    }

    // call atomic op here
    if (thread_len > 0) {
        if (tp == BINARY_TYPE_EQ || tp == BINARY_TYPE_NE || tp == BINARY_TYPE_GE || tp == BINARY_TYPE_GT || tp == BINARY_TYPE_LE || tp == BINARY_TYPE_LT)
        {
            binary_atomic2<T, TO>(
                reinterpret_cast<TO*>(out_buffer[pp_flag]),
                reinterpret_cast<T*>(in_buffer[0][pp_flag]),
                reinterpret_cast<T*>(in_buffer[1][pp_flag]),
                thread_len, tp);
        } else {
            binary_atomic1<T>(
                reinterpret_cast<T*>(out_buffer[pp_flag]),
                reinterpret_cast<T*>(in_buffer[0][pp_flag]),
                reinterpret_cast<T*>(in_buffer[1][pp_flag]),
                thread_len, tp);
        }


      evs_out[pp_flag] = ctxs_out[pp_flag].trigger();
    }

    if (i != thread_off_leading) {
      int thread_off_prev = i - thread_step;
      int thread_remain_prev = N - thread_off_prev;
      int thread_len_prev =
          thread_remain_prev >= TILE_LEN ? TILE_LEN : thread_remain_prev;
      if (thread_len_prev > 0) {
        evs_out[pp_flag_prev].wait();
      }

      if (thread_len_next > 0) {
        // ctxs_out[pp_flag_prev].set_dst_addr(out + thread_off_next);
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


#define BINARY_OP(TYPE, TYPE_OUT, FN_NAME, TP) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    TYPE *lhs, \
    TYPE *rhs, \
    TYPE_OUT *out) \
{ \
    binary_kernel<TYPE, TYPE_OUT>(lhs, rhs, out, numel, TP); \
} \


BINARY_OP(__bf16, __bf16, badd_bf16, BINARY_TYPE_ADD)
BINARY_OP(__bf16, __bf16, bsub_bf16, BINARY_TYPE_SUB)
BINARY_OP(__bf16, __bf16, bmul_bf16, BINARY_TYPE_MUL)
BINARY_OP(__bf16, __bf16, bdiv_bf16, BINARY_TYPE_DIV)
BINARY_OP(__bf16, __bf16, bmaximum_bf16, BINARY_TYPE_MAX)
BINARY_OP(__bf16, __bf16, bminimum_bf16, BINARY_TYPE_MIN)
BINARY_OP(__bf16, __bf16, mod_bf16, BINARY_TYPE_MOD) 

BINARY_OP(__bf16, uint8_t, eq_bf16, BINARY_TYPE_EQ)
BINARY_OP(__bf16, uint8_t, ne_bf16, BINARY_TYPE_NE)
BINARY_OP(__bf16, uint8_t, ge_bf16, BINARY_TYPE_GE)
BINARY_OP(__bf16, uint8_t, gt_bf16, BINARY_TYPE_GT)
BINARY_OP(__bf16, uint8_t, lt_bf16, BINARY_TYPE_LT) 
BINARY_OP(__bf16, uint8_t, le_bf16, BINARY_TYPE_LE) 


BINARY_OP(__fp16, __fp16, badd_f16, BINARY_TYPE_ADD)
BINARY_OP(__fp16, __fp16, bsub_f16, BINARY_TYPE_SUB)
BINARY_OP(__fp16, __fp16, bmul_f16, BINARY_TYPE_MUL)
BINARY_OP(__fp16, __fp16, bdiv_f16, BINARY_TYPE_DIV)
BINARY_OP(__fp16, __fp16, bmaximum_f16, BINARY_TYPE_MAX)
BINARY_OP(__fp16, __fp16, bminimum_f16, BINARY_TYPE_MIN)
BINARY_OP(__fp16, __fp16, mod_f16, BINARY_TYPE_MOD)

BINARY_OP(__fp16, uint8_t, eq_f16, BINARY_TYPE_EQ)
BINARY_OP(__fp16, uint8_t, ne_f16, BINARY_TYPE_NE)
BINARY_OP(__fp16, uint8_t, ge_f16, BINARY_TYPE_GE)
BINARY_OP(__fp16, uint8_t, gt_f16, BINARY_TYPE_GT)
BINARY_OP(__fp16, uint8_t, lt_f16, BINARY_TYPE_LT)
BINARY_OP(__fp16, uint8_t, le_f16, BINARY_TYPE_LE)



BINARY_OP(float, float, badd_f32, BINARY_TYPE_ADD)
BINARY_OP(float, float, bsub_f32, BINARY_TYPE_SUB)
BINARY_OP(float, float, bmul_f32, BINARY_TYPE_MUL)
BINARY_OP(float, float, bdiv_f32, BINARY_TYPE_DIV)
BINARY_OP(float, float, bmaximum_f32, BINARY_TYPE_MAX)
BINARY_OP(float, float, bminimum_f32, BINARY_TYPE_MIN)
BINARY_OP(float, float, mod_f32, BINARY_TYPE_MOD)

BINARY_OP(float, uint8_t, eq_f32, BINARY_TYPE_EQ)
BINARY_OP(float, uint8_t, ne_f32, BINARY_TYPE_NE)
BINARY_OP(float, uint8_t, ge_f32, BINARY_TYPE_GE)
BINARY_OP(float, uint8_t, gt_f32, BINARY_TYPE_GT)
BINARY_OP(float, uint8_t, lt_f32, BINARY_TYPE_LT)
BINARY_OP(float, uint8_t, le_f32, BINARY_TYPE_LE)

int main() {
    return 0;
}