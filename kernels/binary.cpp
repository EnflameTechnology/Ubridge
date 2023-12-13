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

#define TILE_SIZE AlignDown(((VDMEM_SIZE) / 6), 256)
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


#define BINARY_SCALAR_OP_DEFINE(NAME) \
template <typename TYPE, typename OUTTYPE> \
__device__ __forceinline__ void binary_scalar_##NAME(OUTTYPE* out, TYPE* lhs, TYPE rhs, unsigned int len) { \
} \

#define BINARY_SCALAR_OP_INIT(NAME, TYPE, OUTTYPE, FUNC) \
template <> \
__device__ __forceinline__ void binary_scalar_##NAME(OUTTYPE* out, TYPE* lhs, TYPE rhs, unsigned int len) { \
  FUNC(out, lhs, rhs, len); \
} \

BINARY_SCALAR_OP_DEFINE(sub)
BINARY_SCALAR_OP_INIT(sub, __fp16, __fp16, sub_fp16_scalar)
BINARY_SCALAR_OP_INIT(sub, __bf16, __bf16, sub_bf16_scalar)
BINARY_SCALAR_OP_INIT(sub, float, float, sub_fp32_scalar)

BINARY_SCALAR_OP_DEFINE(add)
BINARY_SCALAR_OP_INIT(add, __fp16, __fp16, add_fp16_scalar)
BINARY_SCALAR_OP_INIT(add, __bf16, __bf16, add_bf16_scalar)
BINARY_SCALAR_OP_INIT(add, float, float, add_fp32_scalar)

BINARY_SCALAR_OP_DEFINE(mul)
BINARY_SCALAR_OP_INIT(mul, __fp16, __fp16, mul_fp16_scalar)
BINARY_SCALAR_OP_INIT(mul, __bf16, __bf16, mul_bf16_scalar)
BINARY_SCALAR_OP_INIT(mul, float, float, mul_fp32_scalar)

BINARY_SCALAR_OP_DEFINE(div)
BINARY_SCALAR_OP_INIT(div, __fp16, __fp16, div_fp16_scalar)
BINARY_SCALAR_OP_INIT(div, __bf16, __bf16, div_bf16_scalar)
BINARY_SCALAR_OP_INIT(div, float, float, div_fp32_scalar)


BINARY_SCALAR_OP_DEFINE(min)
BINARY_SCALAR_OP_INIT(min, __fp16, __fp16, min_fp16_scalar)
BINARY_SCALAR_OP_INIT(min, __bf16, __bf16, min_bf16_scalar)
BINARY_SCALAR_OP_INIT(min, float, float, min_fp32_scalar)

BINARY_SCALAR_OP_DEFINE(max)
BINARY_SCALAR_OP_INIT(max, __fp16, __fp16, max_fp16_scalar)
BINARY_SCALAR_OP_INIT(max, __bf16, __bf16, max_bf16_scalar)
BINARY_SCALAR_OP_INIT(max, float, float, max_fp32_scalar)

BINARY_SCALAR_OP_DEFINE(mod)
BINARY_SCALAR_OP_INIT(mod, __fp16, __fp16, mod_fp16_scalar)
BINARY_SCALAR_OP_INIT(mod, __bf16, __bf16, mod_bf16_scalar)
BINARY_SCALAR_OP_INIT(mod, float, float, mod_fp32_scalar)


BINARY_SCALAR_OP_DEFINE(eq)
BINARY_SCALAR_OP_INIT(eq, __fp16, uint8_t, eq_fp16_scalar)
BINARY_SCALAR_OP_INIT(eq, __bf16, uint8_t, eq_bf16_scalar)
BINARY_SCALAR_OP_INIT(eq, float, uint8_t, eq_fp32_scalar)


BINARY_SCALAR_OP_DEFINE(ne)
BINARY_SCALAR_OP_INIT(ne, __fp16, uint8_t, ne_fp16_scalar)
BINARY_SCALAR_OP_INIT(ne, __bf16, uint8_t, ne_bf16_scalar)
BINARY_SCALAR_OP_INIT(ne, float, uint8_t, ne_fp32_scalar)


BINARY_SCALAR_OP_DEFINE(ge)
BINARY_SCALAR_OP_INIT(ge, __fp16, uint8_t, ge_fp16_scalar)
BINARY_SCALAR_OP_INIT(ge, __bf16, uint8_t, ge_bf16_scalar)
BINARY_SCALAR_OP_INIT(ge, float, uint8_t, ge_fp32_scalar)

BINARY_SCALAR_OP_DEFINE(gt)
BINARY_SCALAR_OP_INIT(gt, __fp16, uint8_t, gt_fp16_scalar)
BINARY_SCALAR_OP_INIT(gt, __bf16, uint8_t, gt_bf16_scalar)
BINARY_SCALAR_OP_INIT(gt, float, uint8_t, gt_fp32_scalar)

BINARY_SCALAR_OP_DEFINE(lt)
BINARY_SCALAR_OP_INIT(lt, __fp16, uint8_t, lt_fp16_scalar)
BINARY_SCALAR_OP_INIT(lt, __bf16, uint8_t, lt_bf16_scalar)
BINARY_SCALAR_OP_INIT(lt, float, uint8_t, lt_fp32_scalar)


BINARY_SCALAR_OP_DEFINE(le)
BINARY_SCALAR_OP_INIT(le, __fp16, uint8_t, le_fp16_scalar)
BINARY_SCALAR_OP_INIT(le, __bf16, uint8_t, le_bf16_scalar)
BINARY_SCALAR_OP_INIT(le, float, uint8_t, le_fp32_scalar)


template <typename T>
__device__ __forceinline__ void binary_scalar_atomic1(T* out, T* lhs, T rhs, unsigned int len, BINARY_TYPE tp)
{
  #if __GCU_ARCH__ >= 300 
  switch (tp) {
    case BINARY_TYPE_ADD:
      {
        binary_scalar_add(out, lhs, rhs, len);
        break;
      }
    case BINARY_TYPE_SUB:
      {
        binary_scalar_sub(out, lhs, rhs, len);
        break;
      }
    case BINARY_TYPE_MUL:
      {
        binary_scalar_mul(out, lhs, rhs, len);
        break;
      }
    case BINARY_TYPE_DIV:
      {
        binary_scalar_div(out, lhs, rhs, len);
        break;
      }
    case BINARY_TYPE_MAX:
      {
        binary_scalar_max(out, lhs, rhs, len);
        break;
      }
    case BINARY_TYPE_MIN:
      {
        binary_scalar_min(out, lhs, rhs, len);
        break;
      }
    case BINARY_TYPE_MOD:
      {
        binary_scalar_mod(out, lhs, rhs, len);
        break;
      }
    default:
      break;
    }
  #endif
}


template <typename T, typename TO>
__device__ __forceinline__ void binary_scalar_atomic2(TO* out, T* lhs, T rhs, unsigned int len, BINARY_TYPE tp)
{
  #if __GCU_ARCH__ >= 300 
  switch (tp) {
    case BINARY_TYPE_EQ:
      {
        binary_scalar_eq(reinterpret_cast<uint8_t*>(out), lhs, rhs, len);
        break;
      }
    case BINARY_TYPE_NE: {
        binary_scalar_ne(reinterpret_cast<uint8_t*>(out), lhs, rhs, len);
        break;
    }
    case BINARY_TYPE_GE:
      {
        binary_scalar_ge(reinterpret_cast<uint8_t*>(out), lhs, rhs, len);
        break;
      }
    case BINARY_TYPE_GT:
      {
        binary_scalar_gt(reinterpret_cast<uint8_t*>(out), lhs, rhs, len);
        break;
      }
    case BINARY_TYPE_LT:
      {
        binary_scalar_lt(reinterpret_cast<uint8_t*>(out), lhs, rhs, len);
        
        break;
      }
    case BINARY_TYPE_LE:
      {
        binary_scalar_le(reinterpret_cast<uint8_t*>(out), lhs, rhs, len);
        break;
      }
    default:
      break;
    }
  #endif
}


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
__device__ __forceinline__ void binary_kernel_right(T* in_a, T* in_b, TO* out, int element_num,  const size_t num_dims,  size_t *dims_and_strides, BINARY_TYPE tp) 
{
    int dim_size = dims_and_strides[num_dims - 1];
    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNumEachBlock();

    const int N = element_num / dim_size;
    __local__ __valigned__ T in_buffer1[20480];
    __local__ __valigned__ T in_buffer2[20480];
    __local__ __valigned__ TO out_buffer[20480];

    int right_dim_size = N;
    for (int i=0; i<num_dims; i++) {
      int idex = num_dims * 3 - i - 1;
      if (dims_and_strides[idex] == 1) {
        right_dim_size = dims_and_strides[num_dims - i - 1];
        break;
      }
    }

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

          
    tops_dte_ctx_t ctxs_in[2];
    tops_dte_ctx_t ctxs_out;
    ctxs_in[0].init();
    ctxs_in[1].init();
    ctxs_out.init();
    tops::dte_scope s_in0(ctxs_in[0]);
    tops::dte_scope s_in1(ctxs_in[1]);
    tops::dte_scope s_out0(ctxs_out);

    tops::mdspan hbm_in1(tops::Global, in_a, N, dim_size);
    tops::mdspan hbm_in2(tops::Global, in_b, right_dim_size==dim_size?1:N, right_dim_size==dim_size?dim_size:1);


    tops::mdspan l1_in1(tops::Private, in_buffer1, 1, dim_size);
    tops::mdspan l1_in2(tops::Private, in_buffer2, 1, right_dim_size==dim_size?dim_size:1);

    tops::mdspan hbm_out(tops::Global, out, N, dim_size);
    tops::mdspan l1_out(tops::Private, out_buffer, 1, dim_size);

    for (int i = 0; i < thread_step; i++) {
      int idx = thread_id * THREAD_STEP + i;
      if (idx < N) {
          // printf("N %d, dim_size %d, THREAD_STEP %d, thread_step %d", N, dim_size, THREAD_STEP, thread_step);

          ctxs_in[0].config_slice(l1_in1, hbm_in1, {0, 0});
          ctxs_in[0].set_src_offset(0, idx);
          ctxs_in[0].trigger_and_wait();

          ctxs_in[1].config_slice(l1_in2, hbm_in2, {0, 0});
          ctxs_in[1].set_src_offset(0, right_dim_size==dim_size?0:idx);
          ctxs_in[1].trigger_and_wait();

          if (tp == BINARY_TYPE_EQ || tp == BINARY_TYPE_NE || tp == BINARY_TYPE_GE || tp == BINARY_TYPE_GT || tp == BINARY_TYPE_LE || tp == BINARY_TYPE_LT)
          {
            if (right_dim_size==dim_size) {
                binary_atomic2<T, TO>(
                  reinterpret_cast<TO*>(out_buffer),
                  reinterpret_cast<T*>(in_buffer1),
                  reinterpret_cast<T*>(in_buffer2),
                  dim_size, tp);
            } else {
              binary_scalar_atomic2<T, TO>(
                  reinterpret_cast<TO*>(out_buffer),
                  reinterpret_cast<T*>(in_buffer1),
                  in_buffer2[0],
                  dim_size, tp);
            }

          } else {
            if (right_dim_size==dim_size) {
                binary_atomic1<T>(
                  reinterpret_cast<T*>(out_buffer),
                  reinterpret_cast<T*>(in_buffer1),
                  reinterpret_cast<T*>(in_buffer2),
                  dim_size, tp);
            } else {
              binary_scalar_atomic1<T>(
                  reinterpret_cast<T*>(out_buffer),
                  reinterpret_cast<T*>(in_buffer1),
                  in_buffer2[0],
                  dim_size, tp);
            }

          }

          // int printfelements = dim_size;
          // if (printfelements > 20) {
          //   printfelements = 20;
          // }
          // printf("Binary type %d\n", tp);
          // printf("\nBinary Input buffer1 (first 20): ");
          // for (int j=0; j<printfelements; j++) {
          //   printf("%.5f ", in_buffer1[j]);
          // }
          // if (right_dim_size==dim_size) {
          //   printf("\nBinary Input buffer2 (first 20): ");
          //   for (int j=0; j<printfelements; j++) {
          //     printf("%.5f ", in_buffer2[j]);
          //   }
          // } else {
          //   printf("\nBinary Input scalar: %.5f", in_buffer2[0]);
          // }

          // printf("\nBinary Output buffer (first 20): ");

          // for (int j=0; j<printfelements; j++) {
          //   printf("%.5f ", out_buffer[j]);
          // }

          ctxs_out.config_deslice(hbm_out, l1_out, {0, 0});
          ctxs_out.set_dst_offset(0, idx);
          ctxs_out.trigger_and_wait();
      }
    }

}

template <typename T, typename TO>
__device__ __forceinline__ void binary_kernel(T* in_a, T* in_b, TO* out, int N, 
        BINARY_TYPE tp) {
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
            // printf("\nBinary Input buffer: ");
            // for (int j=0; j<thread_len; j++) {
            //   printf("left %.5f, right %.5f   ", reinterpret_cast<T*>(in_buffer[0][pp_flag])[j], 
            //         reinterpret_cast<T*>(in_buffer[1][pp_flag])[j]);
            // }
            // printf("\n\nBinary Output buffer: ");
            // for (int j=0; j<thread_len; j++) {
            //   printf("%.5f ", reinterpret_cast<T*>(out_buffer[pp_flag])[j]);
            // }
            // printf("\n\n");
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
        ctxs_out[pp_flag_prev].config_memcpy(
            tops::mdspan(tops::Global, out + thread_off_next, thread_len_next),
            tops::mdspan(tops::Private,
                         reinterpret_cast<TO*>(out_buffer[pp_flag_prev]),
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
    const size_t num_dims, \
    size_t *dims_and_strides, \
    TYPE *lhs, \
    TYPE *rhs, \
    TYPE_OUT *out) \
{ \
    bool lhs_cont = true; \
    bool rhs_cont = true; \
    __local__ __valigned__ size_t info[128]; \
    tops_dte_ctx_t ctx; \
    tops::dte_scope s(ctx); \
    if (dims_and_strides) { \
      tops::mdspan srcInfo(tops::Global, dims_and_strides, num_dims * 3); \
      tops::mdspan dstInfo(tops::Private, info, num_dims * 3); \
      tops::memcpy(ctx, dstInfo, srcInfo); \
      lhs_cont = is_contiguous(num_dims, info, info + 1 * num_dims); \
      rhs_cont = is_contiguous(num_dims, info, info + 2 * num_dims); \
    } \
    if (lhs_cont && rhs_cont) { \
      binary_kernel<TYPE, TYPE_OUT>(lhs, rhs, out, numel, TP); \
    } else if (lhs_cont) { \
      binary_kernel_right<TYPE, TYPE_OUT>(lhs, rhs, out, numel, num_dims, info, TP); \
    }\
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

template<typename T, typename OUTT>
int test() {
  T *lhs_d; T *rhs_d; OUTT *out_d;
  int *shape_lhs_d;
  T *lhs_h; T *rhs_h; OUTT *out_h;
  size_t* dim_strides_d;
  size_t size_lhs = 2 * 4;
  size_t size_rhs = 2;
  size_t dim_strides[6] = {2, 4, 4, 1, 1, 0};
  size_t size_out = size_lhs;
  size_t dim = 1;
  topsHostMalloc((T**)&lhs_h, size_lhs * sizeof(T));
  topsHostMalloc((T**)&rhs_h, size_rhs * sizeof(T));

  topsHostMalloc((T**)&out_h, size_out * sizeof(OUTT));
  // topsHostMalloc((T**)&dim_strides_h, sizeof(dim_strides));

    // T a = 0.5;
    OUTT zero = 0.0;
    for (size_t i = 0; i < size_lhs; i++) {
        lhs_h[i] = static_cast<T>(1.5);
    }
    for (size_t i = 0; i < size_rhs; i++) {
        rhs_h[i] = static_cast<T>(0.7);
    }
    for (size_t i = 0; i < size_out; i++) {
        out_h[i] = static_cast<OUTT>(zero);
    }
  topsMalloc(&lhs_d, size_lhs * sizeof(T));
  topsMalloc(&rhs_d, size_rhs * sizeof(T));
  topsMalloc(&out_d, size_out * sizeof(OUTT));
  topsMalloc(&dim_strides_d, sizeof(dim_strides));


  printf("info: copy Host2Device\n");
  topsMemcpy(lhs_d, lhs_h, size_lhs * sizeof(T),
                  topsMemcpyHostToDevice);
  topsMemcpy(rhs_d, rhs_h, size_rhs * sizeof(T),
                  topsMemcpyHostToDevice);
  topsMemcpy(out_d, out_h, size_out * sizeof(OUTT),
                  topsMemcpyHostToDevice);
  topsMemcpy(dim_strides_d, dim_strides, sizeof(dim_strides),
                  topsMemcpyHostToDevice);

  bdiv_f32<<<dim3(1, 1, 1), dim3(12, 1, 1)>>>(size_out, 2, dim_strides_d, lhs_d, rhs_d, out_d);

  printf("info: copy Device2Host\n");
  topsMemcpy(out_h, out_d, size_out * sizeof(OUTT), topsMemcpyDeviceToHost);

  for (size_t j = 0; j < size_out; j++) {
      // OUTT dif = static_cast<OUTT>(0.8) - static_cast<OUTT>(out_h[j]);
      // if (dif > static_cast<OUTT>(0.0000001f))
        printf("%.5f, ", static_cast<OUTT>(out_h[j]));
  }
  topsHostFree(lhs_h);
  topsHostFree(rhs_h);
  topsHostFree(out_h);
  topsFree(lhs_d);
  topsFree(out_d);
  return 0;
}

int main() {
    return test<float, float>();
}