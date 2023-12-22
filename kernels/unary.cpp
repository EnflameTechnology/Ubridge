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
#include <stdio.h>
#include <tops.h>
#include <tops/topsrtc.h>
#include <tops/half.h>
#include <tops/bfloat.h>
#include <string>
#include <tops/tops_runtime.h>
#include <tops/topsrtc.h>
#include <krt/scalar.h>
#include <krt/vector_mask.h>
#include <krt/dispatch.h>
#include <krt/leaptr.h>
#include <krt/vector_infra.h>
#include <krt/mmu.h>

#include "include/common/atomic_op.h"
#include "utils.h"
#include "utils/vector_ex.h"
using namespace std;
using namespace tops;
#define PING_PONG_SIZE 2
#define TILE_SIZE AlignDown(((VDMEM_SIZE) / 32), 256)
#define GATHER_COPY

enum UNARY_TYPE {
    UNARY_TYPE_NEG = 1,
    UNARY_TYPE_EXP = 2,
    UNARY_TYPE_LOG = 3,
    UNARY_TYPE_SIN = 4,
    UNARY_TYPE_COS = 5,
    UNARY_TYPE_ABS = 6,
    UNARY_TYPE_SQUARE = 8,
    UNARY_TYPE_SQRT = 9,
    UNARY_TYPE_RSQRT = 10,
    UNARY_TYPE_GELU = 11,
    UNARY_TYPE_RELU = 12,
    UNARY_TYPE_ELU = 13,
    UNARY_TYPE_SILU = 14,
    UNARY_TYPE_TANH = 15,
    UNARY_TYPE_RECIP = 16,
    UNARY_TYPE_COPY = 20,
};

template <typename T>
__device__ __forceinline__ void gelu_kernel(T* output, T* input, int num) {
  using vtype = typename scalar_to_vector<T,
                                          TOPS_VECTOR_LENGTH>::type;
  const int vlength = vector_length<vtype>::value;
  leaptr<vtype> intput_ptr = simple_leaptr<vtype>(input);
  leaptr<vtype> output_ptr = simple_leaptr<vtype>(output);
  int group_num = (num + vlength - 1) / vlength;
  vtype v_input, v_output;
  for (int i = 0; i < group_num; i++) {
    v_input = intput_ptr.load();
    v_output = vgelu(v_input);
    output_ptr.store(v_output);
  }
}

template <typename T, typename VT>
__device__ __forceinline__ void relu_kernel(T* output, T* input, int num) {
  using vtype = typename scalar_to_vector<T,
                                          TOPS_VECTOR_LENGTH>::type;
  const int vlength = vector_length<vtype>::value;
  leaptr<vtype> intput_ptr = simple_leaptr<vtype>(input);
  leaptr<vtype> output_ptr = simple_leaptr<vtype>(output);
  int group_num = (num + vlength - 1) / vlength;
  vtype v_input, v_output;
  auto vzeros = tops::vzero<VT>();
  for (int i = 0; i < group_num; i++) {
    v_input = intput_ptr.load();
    // v_output = vmax(v_input, vzeros); // fix this!
    output_ptr.store(v_output);
  }
}

// template <typename T, typename OP>
// __global__ void gelu_fwd_kernel(T* out, T* in, int num) {
//   int thread_num = GetThreadNum();
//   int thread_id = GetThreadIdx();
//   int elem_per_sip = CeilDiv(num, thread_num);
//   int start = thread_id * elem_per_sip;
//   int elem_this_sip = num - start > elem_per_sip ? elem_per_sip : num - start;
//   int end = start + elem_this_sip;
//   if (start >= num) return;

//   constexpr int sip_size = 0x180000;
//   constexpr int bpe = sizeof(T);
//   constexpr int vector_length = 128;
//   constexpr int block_nums = 8;

// #define PING_PONG_SIZE (1)

//   // int tile_size = AlignDown(sip_size / block_nums / bpe, vector_length);
//   constexpr int tile_size = 64 * 1024 / sizeof(T);
//   __local__ __attribute__((aligned(256))) T in_buf[PING_PONG_SIZE][tile_size];
//   __local__ __attribute__((aligned(256))) T out_buf[PING_PONG_SIZE][tile_size];

//   tops_dte_ctx_t ctx;
//   tops::dte_scope s(ctx);

//   int pp_flag = 0;
//   for (int offset = start; offset < end; offset += tile_size) {
//     int elem_num = end - offset > tile_size ? tile_size : end - offset;
//     // dte in
//     tops::mdspan in_g(tops::Global, in + offset, elem_num);
//     tops::mdspan in_p(tops::Private, in_buf[pp_flag], elem_num);

//     tops::memcpy(ctx, in_p, in_g);

//     // call atomic op here
//     // call_atomic_activation<T, OP>(out_buf[pp_flag], in_buf[pp_flag],
//     // elem_num);
//     gelu(out_buf[pp_flag], in_buf[pp_flag], elem_num);

//     // dte out
//     tops::mdspan out_g(tops::Global, out + offset, elem_num);
//     tops::mdspan out_p(tops::Private, out_buf[pp_flag], elem_num);

//     tops::memcpy(ctx, out_g, out_p);
//   }
// }


template <typename T, typename VT>
__device__ __forceinline__ void unary_atomic(T* in, T* out, int len, UNARY_TYPE tp)
{
  tops_dte_ctx_t ctx;
  ctx.init();
  switch (tp) {
    case UNARY_TYPE_NEG:
      {
        neg(out, in, len);
        break;
      }
    case UNARY_TYPE_EXP:
      {
        exp(out, in, len);
        break;
      }
    case UNARY_TYPE_LOG:
      {
        log(out, in, len);
        break;
      }
    case UNARY_TYPE_SIN:
      {
        sin(out, in, len);
        break;
      }
    case UNARY_TYPE_COS:
      {
        cos(out, in, len);
        break;
      }
    case UNARY_TYPE_ABS:
      {
        abs(out, in, len);
        break;
      }
    case UNARY_TYPE_SQUARE:
      {
        mul(out, in, in, len);
        break;
      }
    case UNARY_TYPE_SQRT:
      {
        sqrt(out, in, len);
        break;
      }
    case UNARY_TYPE_RSQRT:
      {
        rsqrt(out, in, len);
        break;
      }
    case UNARY_TYPE_GELU:
      {
        gelu_kernel(out, in, len);
        break;
      }
    case UNARY_TYPE_RELU:
      {
        relu_kernel<T, VT>(out, in , len);
        break;
      }
    case UNARY_TYPE_ELU:
      {
        // elu(out, in, len);
        break;
      }
    case UNARY_TYPE_SILU:
      {
        //(out, in, len);
        break;
      }
    case UNARY_TYPE_TANH:
      {
        tanh(out, in, len);
        break;
      }
    case UNARY_TYPE_RECIP:
      {
        reciprocal(out, in, len);
        break;
      }
    case UNARY_TYPE_COPY:
      {
        tops::mdspan src_p(tops::Private, in, len);
        tops::mdspan dst_p(tops::Private, out, len);
        tops::memcpy(ctx, dst_p, src_p);
        break;
      }
    default:
      break;
    }
}


template <typename T, typename VT>
__device__ void unary_kernel(T* in, T* out, int len, UNARY_TYPE tp) {
  tops_dte_ctx_t ctxs_in[PING_PONG_SIZE];
  tops_dte_ctx_t ctxs_out[PING_PONG_SIZE];
  tops::event evs_in[PING_PONG_SIZE];
  tops::event evs_out[PING_PONG_SIZE];
  __local__ __valigned__ T in_buffer[PING_PONG_SIZE][TILE_SIZE];
  __local__ __valigned__ T out_buffer[PING_PONG_SIZE][TILE_SIZE];
  int N = len;
  tops::mdspan output(tops::Global, out, N);

  int thread_num = GetThreadNum();
  int thread_id = GetThreadIdx();

  int thread_off_leading = thread_id * TILE_SIZE;
  int thread_len_leading =
      N - thread_off_leading >= TILE_SIZE ? TILE_SIZE : N - thread_off_leading;
  int thread_step = TILE_SIZE * thread_num;

  int thread_off_leading_next = thread_off_leading + thread_step;
  int thread_remain_leading = N - thread_off_leading_next;
  int thread_len_leading_next =
      thread_remain_leading >= TILE_SIZE ? TILE_SIZE : thread_remain_leading;

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
    int thread_len = N - i >= TILE_SIZE ? TILE_SIZE : N - i;
    int thread_len_next =
        thread_remain_next >= TILE_SIZE ? TILE_SIZE : thread_remain_next;
    if (thread_len_next > 0) {
      evs_in[pp_flag_next] = ctxs_in[pp_flag_next].trigger();
    }

    int thread_off_next2 = i + thread_step * 2;
    int thread_remain_next2 = N - thread_off_next2;
    int thread_len_next2 =
        thread_remain_next2 >= TILE_SIZE ? TILE_SIZE : thread_remain_next2;

    if (thread_len > 0) {
      evs_in[pp_flag].wait();
    }

    if (thread_len_next2 > 0) {
      ctxs_in[pp_flag].config_memcpy(
        tops::mdspan(tops::Private, in_buffer[pp_flag],
                     thread_len_next2),
        tops::mdspan(tops::Global, in + thread_off_next2,
                     thread_len_next2));
    }

    // call atomic op here
    if (thread_len > 0) {
      unary_atomic<T, VT>(in_buffer[pp_flag], out_buffer[pp_flag],
                               thread_len, tp);
      evs_out[pp_flag] = ctxs_out[pp_flag].trigger();
    }

    if (i != thread_off_leading) {
      int thread_off_prev = i - thread_step;
      int thread_remain_prev = N - thread_off_prev;
      int thread_len_prev =
          thread_remain_prev >= TILE_SIZE ? TILE_SIZE : thread_remain_prev;
      if (thread_len_prev > 0) {
        evs_out[pp_flag_prev].wait();
      }

      if (thread_len_next > 0) {
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

#define UNARY_OP(TYPE, VT, FN_NAME, TP) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    TYPE *inp, \
    TYPE *out) \
{ \
    unary_kernel<TYPE, VT>(inp, out, numel, TP); \
} \

UNARY_OP(__bf16, vbfloat, uneg_bf16, UNARY_TYPE_NEG)
UNARY_OP(__bf16, vbfloat, uexp_bf16, UNARY_TYPE_EXP)
UNARY_OP(__bf16, vbfloat, ulog_bf16, UNARY_TYPE_LOG)
UNARY_OP(__bf16, vbfloat, usin_bf16, UNARY_TYPE_SIN)
UNARY_OP(__bf16, vbfloat, ucos_bf16, UNARY_TYPE_COS)
UNARY_OP(__bf16, vbfloat, uabs_bf16, UNARY_TYPE_ABS)
UNARY_OP(__bf16, vbfloat, usqr_bf16, UNARY_TYPE_SQUARE)
UNARY_OP(__bf16, vbfloat, usqrt_bf16, UNARY_TYPE_SQRT)
UNARY_OP(__bf16, vbfloat, ursqrt_bf16, UNARY_TYPE_RSQRT)
UNARY_OP(__bf16, vbfloat, ugelu_bf16, UNARY_TYPE_GELU)
UNARY_OP(__bf16, vbfloat, urelu_bf16, UNARY_TYPE_RELU) 
UNARY_OP(__bf16, vbfloat, usilu_bf16, UNARY_TYPE_SILU) 
UNARY_OP(__bf16, vbfloat, utanh_bf16, UNARY_TYPE_TANH) 
UNARY_OP(__bf16, vbfloat, urecip_bf16, UNARY_TYPE_RECIP) 
UNARY_OP(__bf16, vbfloat, uelu_bf16, UNARY_TYPE_ELU) 

UNARY_OP(__fp16, vhalf, uneg_f16, UNARY_TYPE_NEG)
UNARY_OP(__fp16, vhalf, uexp_f16, UNARY_TYPE_EXP)
UNARY_OP(__fp16, vhalf, ulog_f16, UNARY_TYPE_LOG)
UNARY_OP(__fp16, vhalf, usin_f16, UNARY_TYPE_SIN)
UNARY_OP(__fp16, vhalf, ucos_f16, UNARY_TYPE_COS)
UNARY_OP(__fp16, vhalf, uabs_f16, UNARY_TYPE_ABS)
UNARY_OP(__fp16, vhalf, usqr_f16, UNARY_TYPE_SQUARE)
UNARY_OP(__fp16, vhalf, usqrt_f16, UNARY_TYPE_SQRT)
UNARY_OP(__fp16, vhalf, ursqrt_f16, UNARY_TYPE_RSQRT)
UNARY_OP(__fp16, vhalf, ugelu_f16, UNARY_TYPE_GELU)
UNARY_OP(__fp16, vhalf, urelu_f16, UNARY_TYPE_RELU)
UNARY_OP(__fp16, vhalf, usilu_f16, UNARY_TYPE_SILU)
UNARY_OP(__fp16, vhalf, utanh_f16, UNARY_TYPE_TANH)
UNARY_OP(__fp16, vhalf, urecip_f16, UNARY_TYPE_RECIP)
UNARY_OP(__fp16, vhalf, uelu_f16, UNARY_TYPE_ELU)


UNARY_OP(float, vfloat, uneg_f32, UNARY_TYPE_NEG)
UNARY_OP(float, vfloat, uexp_f32, UNARY_TYPE_EXP)
UNARY_OP(float, vfloat, ulog_f32, UNARY_TYPE_LOG)
UNARY_OP(float, vfloat, usin_f32, UNARY_TYPE_SIN)
UNARY_OP(float, vfloat, ucos_f32, UNARY_TYPE_COS)
UNARY_OP(float, vfloat, uabs_f32, UNARY_TYPE_ABS)
UNARY_OP(float, vfloat, usqr_f32, UNARY_TYPE_SQUARE)
UNARY_OP(float, vfloat, usqrt_f32, UNARY_TYPE_SQRT)
UNARY_OP(float, vfloat, ursqrt_f32, UNARY_TYPE_RSQRT)
UNARY_OP(float, vfloat, ugelu_f32, UNARY_TYPE_GELU)
UNARY_OP(float, vfloat, urelu_f32, UNARY_TYPE_RELU)
UNARY_OP(float, vfloat, usilu_f32, UNARY_TYPE_SILU)
UNARY_OP(float, vfloat, utanh_f32, UNARY_TYPE_TANH)
UNARY_OP(float, vfloat, urecip_f32, UNARY_TYPE_RECIP)
UNARY_OP(float, vfloat, uelu_f32, UNARY_TYPE_ELU)

#define PRINTHELPER(ARRAY, SZ, MSG) \
  printf(MSG); \
  for (int i=0; i< SZ; i++) \
    printf("%d, ", (int)ARRAY[i]); \
  printf("\n") \


// #define PRINTHELPER 
#if 0
template <typename T, int SRCRANK, int DSTRANK, int BPE>
__device__ void unary_kernel_non_contiguous(T* in, T* out, const size_t op_type, const size_t origin_numel, 
        const size_t numel, const size_t start_offset, const size_t origin_num_dims, const size_t num_dims, size_t* dims_and_layout) {
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx); 
    int dst_shape[DSTRANK];
    int dst_layout[DSTRANK];
    for (int j = 0; j < num_dims; ++j) {
      dst_shape[j] = dims_and_layout[j + 2 * origin_num_dims];
      dst_layout[j] = dims_and_layout[2 * origin_num_dims + num_dims + j];
    }
    
    // PRINTHELPER(dst_shape, DSTRANK, "\ndst_shape = ");

    if (op_type == 1 && origin_num_dims > 0) {
        int src_shape[SRCRANK];
        int src_layout[SRCRANK];
        for (int j = 0; j < origin_num_dims; ++j) {
          src_shape[j] = dims_and_layout[j];
          src_layout[j] = dims_and_layout[origin_num_dims + j];
        }
        // PRINTHELPER(src_shape, origin_num_dims, "\nsrc_shape = ");
        // PRINTHELPER(src_layout, origin_num_dims, "\nsrc layout = ");
        // PRINTHELPER(dst_layout, num_dims, "\ntranspose pattern, dst layout = ");
        tops::mdspan src_mem(tops::Global, in, src_shape);
        tops::mdspan dst_mem(tops::Global, out, dst_shape);
        tops::transpose(ctx, dst_mem, src_mem, dst_layout); //Optimization required!
    } else if (op_type == 2) { 
      int src_shape[DSTRANK];
      for (int j = DSTRANK - 1; j >=0; j--) {
        if (j - (DSTRANK - SRCRANK) < 0) {
          src_shape[j] = 1;
        } else {
          src_shape[j] = dims_and_layout[j - (DSTRANK - SRCRANK)];
        }
      }
      // PRINTHELPER(src_shape, DSTRANK, "\nsrc_shape = ");
      // PRINTHELPER(dst_shape, DSTRANK, "\nbroadcasting pattern, dst shape = ");

      if (origin_numel == 1) {
        tops::mdspan src_mem(tops::Global, in, 1);
        tops::mdspan dst_mem(tops::Global, out, numel);
        tops::broadcast(ctx, dst_mem, src_mem);
      } else {
        tops::mdspan src_mem(tops::Global, in, src_shape);
        tops::mdspan dst_mem(tops::Global, out, dst_shape);
        tops::broadcast(ctx, dst_mem, src_mem);
      }

    } else if (op_type == 3) { //TODO purmutation pattern
        // printf("\npurmute pattern %d!\n", numel);
        // PRINTHELPER(dst_layout, DSTRANK, "\npurmute pattern, dst layout = ");

    } else if (op_type == 4) {
        // PRINTHELPER(dst_shape, DSTRANK, "\nnarrow pattern, dst shape = ");
        int src_shape[SRCRANK];
        int offsets[SRCRANK];
        for (int j = 0; j < SRCRANK; ++j) {
          src_shape[j] = dims_and_layout[j];
          offsets[j] = src_shape[j]==dst_shape[j]?0:start_offset;
        }
        // PRINTHELPER(src_shape, DSTRANK, "\nnarrow pattern, src shape = ");
        // PRINTHELPER(offsets, DSTRANK, "\noffsets = ");

        tops::mdspan src_mem(tops::Global, in, src_shape);
        tops::mdspan dst_mem(tops::Global, out, dst_shape);
        tops::slice(ctx, dst_mem, src_mem, offsets);
    } else {
        printf("\nNot supported pattern type %d!\n", op_type);
    }

}

#endif

__device__ __forceinline__ unsigned int get_strided_index(
    unsigned int idx,
    const int num_dims,
    const int *dims,
    const int *strides
) {
    unsigned int strided_i = 0;
    for (unsigned int d = 0; d < num_dims; d++) {
        unsigned int dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}

template <int RANK>
__device__ __forceinline__ VecIndexType get_batch_strided_index(VecIndexType &indexes, VecIndexType results, VecIndexType dst_shape[], VecIndexType dst_strides[]) {
    VecIndexType vec_rem[RANK];
    vec_rem[0] = indexes;
    #pragma clang loop unroll(enable)
    for (int i = 0; i < RANK; i++) {
      unsigned int dim_idx = RANK - 1 - i;
      results = vadd(vmul(vrem(vec_rem[i], dst_shape[dim_idx]), dst_strides[dim_idx]), results); 
      vec_rem[i + 1] = vdiv(vec_rem[i], dst_shape[dim_idx]);
    }
    return results;
}


template <typename T, int RANK, int BPE>
__device__ void ucopy_single_thread(T* in, T* out, const size_t in_size, const size_t out_size, size_t* dims_and_strides) {
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx); 
    int dst_dim[RANK];
    int dst_strides[RANK];
    for (int j = 0; j < RANK; ++j) {
      dst_dim[j] = dims_and_strides[j];
      dst_strides[j] = dims_and_strides[RANK + j];
    }
    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNumEachBlock();
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * blockDim.x + threadIdx.x;
    int in_map_size = AlignUp(in_size, NUMS_SPLIT) * sizeof(T);
    int out_map_size = AlignUp(out_size, NUMS_SPLIT) * sizeof(T);
    auto src_l3_addr = map_mem(reinterpret_cast<generic_ptr>(in), in_map_size);
    auto dst_l3_addr = map_mem(reinterpret_cast<generic_ptr>(out), out_map_size);

    T* dst_l3 = reinterpret_cast<T*>(dst_l3_addr);
    T* src_l3 = reinterpret_cast<T*>(src_l3_addr);
    for (int i = 0; i < out_size; i++) {
        unsigned int strided_i = get_strided_index(i, RANK, dst_dim, dst_strides);
        if (strided_i < in_size) {
          dst_l3[i] = src_l3[strided_i];
        } 
    }
    unmap_mem(src_l3_addr);
    unmap_mem(dst_l3_addr);
  
}

template <typename T, int RANK, int BPE>
__device__ void ucopy_multithread(T* in, T* out, const size_t in_size, const size_t out_size, size_t* dims_and_strides) {
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx); 

    int dst_dim[RANK];
    int dst_strides[RANK];
    for (int j = 0; j < RANK; ++j) {
      dst_dim[j] = dims_and_strides[j];
      dst_strides[j] = dims_and_strides[RANK + j];
    }

    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNumEachBlock();
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * blockDim.x + threadIdx.x;

    const int TILESIZE = 128 * 1024 / BPE;
    __local__ __valigned__ T buffer[TILESIZE];
    tops::mdspan out_hbm(tops::Global, out, out_size);
    int N = out_size;
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
    int in_map_size = AlignUp(in_size, NUMS_SPLIT) * sizeof(T);
    auto src_l3_addr = map_mem(reinterpret_cast<generic_ptr>(in), in_map_size);
    T* src_l3 = reinterpret_cast<T*>(src_l3_addr);
    for (int i = 0; i < thread_step; i+=TILESIZE) {
      int bufsize = (i + TILESIZE < thread_step) ? TILESIZE : thread_step - i;
      int offset = thread_id * THREAD_STEP + i;
      for (int j = 0; j< bufsize; j++) {
        size_t idx = offset + j;
        unsigned int strided_i = get_strided_index(idx, RANK, dst_dim, dst_strides);
        buffer[j] = src_l3[strided_i];
      }
      tops::mdspan buffer_l1(tops::Private, buffer, bufsize);
      tops::deslice(ctx, out_hbm, buffer_l1, {offset});
    }
    unmap_mem(src_l3_addr);
}

#define SHARE_BUFFER_SIZE 1024 * 1024 * 16 //16MB
__shared__ char raw_cache[SHARE_BUFFER_SIZE];
template <typename T, int RANK, int BPE>
__device__ void ucopy_multithread_cache(T* in, T* out, const size_t in_size, const size_t out_size, size_t* dims_and_strides) {
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx); 
    __private_dte__ tops_dte_ctx_t cache_ctx;
    tops::dte_scope s1(cache_ctx);

    int dst_dim[RANK];
    int dst_strides[RANK];
    for (int j = 0; j < RANK; ++j) {
      dst_dim[j] = dims_and_strides[j];
      dst_strides[j] = dims_and_strides[RANK + j];
    }

    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNumEachBlock();
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * blockDim.x + threadIdx.x;

    const int TILESIZE = 128 * 1024 / BPE;
    __local__ __valigned__ T buffer[TILESIZE];
    tops::mdspan out_hbm(tops::Global, out, out_size);
    int N = out_size;
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
    bool cacheable = in_size > MAX_THREADS && in_size * sizeof(T) < SHARE_BUFFER_SIZE;

    T* src_cached = reinterpret_cast<T*>(raw_cache);
    mapped_ptr src_l3_addr;

    if (cacheable) {
      int offset = thread_id * THREAD_STEP;
      if (threadId == 0) {
        cache_ctx.config_memcpy(
          tops::mdspan(tops::Shared, src_cached, in_size),
          tops::mdspan(tops::Global, in, in_size));
        cache_ctx.trigger_and_wait();
      }
    } else {
      int in_map_size = AlignUp(in_size, NUMS_SPLIT) * sizeof(T);
      src_l3_addr = map_mem(reinterpret_cast<generic_ptr>(in), in_map_size);
      src_cached = reinterpret_cast<T*>(src_l3_addr);
    }

    __syncthreads();

    for (int i = 0; i < thread_step; i+=TILESIZE) {
      int bufsize = (i + TILESIZE < thread_step) ? TILESIZE : thread_step - i;
      int offset = thread_id * THREAD_STEP + i;
      for (int j = 0; j< bufsize; j++) {
        size_t idx = offset + j;
        unsigned int strided_i = get_strided_index(idx, RANK, dst_dim, dst_strides);
        buffer[j] = src_cached[strided_i];
      }
      tops::mdspan buffer_l1(tops::Private, buffer, bufsize);
      tops::deslice(ctx, out_hbm, buffer_l1, {offset});
    }
    if (!cacheable)
      unmap_mem(src_l3_addr);
}

#undef GATHER_COPY
#ifdef GATHER_COPY
template <int BPE>
__device__ __forceinline__ void gather_and_store(void* out, void* in,
                                                 VecIndexType vec_bpe,
                                                 VecIndexType vec_srt_start,
                                                 VecIndexType vec_srt_end,
                                                 VecIndexType strided_indexes) {
    using ResultValueType = FixedVecValueType<BPE>;
    
    auto mask_ge = vge<vbool64_t>(strided_indexes, vec_srt_start);
    auto mask_lt = vlt<vbool64_t>(strided_indexes, vec_srt_end);
    auto mask = mask_and(mask_ge, mask_lt);

    auto offsets = vmul(strided_indexes, vec_bpe);
    ResultValueType result;
    result = vgather_t(mask, in, offsets, result);
    vstore(result, out);
}
#endif

// #include "utils/unroller_helper.h"
template <typename T, int RANK, int BPE>
__device__ void ucopy_multithread_gather(T* in, T* out, const size_t in_size, const size_t out_size, size_t* dims_and_strides) {
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx); 
    __private_dte__ tops_dte_ctx_t cache_ctx;
    tops::dte_scope s1(cache_ctx);
    tops::mdspan out_hbm(tops::Global, out, out_size);

    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNumEachBlock();
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * blockDim.x + threadIdx.x;
    constexpr int TILESIZE = vector_length<va16u32x4>::value;
    constexpr int BUFFER_SIZE = TILESIZE * 1024 / BPE;

    // printf("\nin_size %lu, out_size %lu, TILESIZE %d", in_size, out_size, TILESIZE);
    int N = out_size;
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
    int dst_dim[RANK];
    int dst_strides[RANK];
    #pragma clang loop unroll(enable)
    for (int j = 0; j < RANK; ++j) {
      dst_dim[j] = dims_and_strides[j];
      dst_strides[j] = dims_and_strides[RANK + j];
    }

    bool cacheable = in_size > MAX_THREADS && in_size * sizeof(T) < SHARE_BUFFER_SIZE;

    T* src_cached = reinterpret_cast<T*>(raw_cache);
    mapped_ptr src_l3_addr;

    if (cacheable) {
      int offset = thread_id * THREAD_STEP;
      if (threadId == 0) {
        cache_ctx.config_memcpy(
          tops::mdspan(tops::Shared, src_cached, in_size),
          tops::mdspan(tops::Global, in, in_size));
        cache_ctx.trigger_and_wait();
      }
    } else {
      int in_map_size = AlignUp(in_size, NUMS_SPLIT) * sizeof(T);
      src_l3_addr = map_mem(reinterpret_cast<generic_ptr>(in), in_map_size);
      src_cached = reinterpret_cast<T*>(src_l3_addr);
    }

    __syncthreads();

    // int in_map_size = AlignUp(in_size, NUMS_SPLIT) * sizeof(T);
    // auto src_l3_addr = map_mem(reinterpret_cast<generic_ptr>(in), in_map_size);
    // T* src_l3 = reinterpret_cast<T*>(src_l3_addr);

    __local__ __valigned__ T buffer[BUFFER_SIZE];
    __local__ __valigned__ unsigned int idx_buffer[TILESIZE];

    va16u32x4 vec_dst_shape[RANK];
    va16u32x4 vec_dst_strides[RANK];

    #pragma clang loop unroll(enable)
    for (int i = 0; i < RANK; ++i) {
      vec_dst_shape[i] = vbroadcast<va16u32x4>((unsigned int)dst_dim[i]);
      vec_dst_strides[i] = vbroadcast<va16u32x4>((unsigned int)dst_strides[i]);
    }
    va16u32x4 vec_src_start = vbroadcast<va16u32x4>((unsigned int)0);
    va16u32x4 vec_src_end = vbroadcast<va16u32x4>((unsigned int)in_size);
    va16u32x4 vec_bpe = vbroadcast<va16u32x4>((unsigned int)sizeof(T));
    va16u32x4 idx_results = vbroadcast<va16u32x4>((unsigned int)0);
    constexpr IndexType NUM_UNROLLING = 1;
    T *pbuffer = buffer;
    size_t buf_written = 0;
    int init_offset = -1;
    for (int i = 0; i < thread_step; i+=TILESIZE) {
      int bufsize = (i + TILESIZE < thread_step) ? TILESIZE : thread_step - i;
      int offset = thread_id * THREAD_STEP + i;
      if (init_offset < 0) init_offset = offset;
      auto indexes = viota<va16u32x4>((unsigned int)offset);
      auto strided_indexes = get_batch_strided_index<RANK>(indexes, idx_results, vec_dst_shape, vec_dst_strides);
#ifdef GATHER_COPY
      if (in_size > TILESIZE && in_size == out_size) {
          gather_and_store<BPE>(pbuffer, src_cached, vec_bpe, vec_src_start,
                                    vec_src_end, strided_indexes);
      } else {
#endif
        vstore(strided_indexes, idx_buffer);
        #pragma clang loop unroll(enable)
        for (int j = 0; j< bufsize; j++) {
          pbuffer[j] = src_cached[idx_buffer[j]];
        }
#ifdef GATHER_COPY
      }
#endif
      buf_written += bufsize;
      pbuffer += bufsize;

      if (buf_written + TILESIZE >= BUFFER_SIZE || i + bufsize >= thread_step - 1 || bufsize < TILESIZE) {
        tops::mdspan buffer_l1(tops::Private, buffer, buf_written);
        tops::deslice(ctx, out_hbm, buffer_l1, {init_offset});
        buf_written = 0;
        pbuffer = buffer;
        init_offset = -1;
      }
    }

    if (!cacheable)
      unmap_mem(src_l3_addr);
  
}

#define UNARY_COPY_OP(KERNEL, TYPE, VT, FN_NAME, BPE) \
extern "C" __global__ void FN_NAME( \
    const size_t in_size, \
    const size_t out_size, \
    const size_t num_dims, \
    size_t *dims_and_strides, \
    TYPE *in, \
    TYPE *out) \
{ \
    bool cont = true; \
    __local__ __valigned__ size_t info[128]; \
    tops_dte_ctx_t ctx; \
    tops::dte_scope s(ctx); \
    if (dims_and_strides) { \
      tops::mdspan srcInfo(tops::Global, dims_and_strides, num_dims * 2); \
      tops::mdspan dstInfo(tops::Private, info, num_dims * 2); \
      tops::memcpy(ctx, dstInfo, srcInfo); \
    } \
    if (num_dims == 2)\
      KERNEL<TYPE, 2, BPE>(in, out, in_size, out_size, info); \
    else if (num_dims == 3)\
      KERNEL<TYPE, 3, BPE>(in, out, in_size, out_size, info); \
    else if (num_dims == 4)\
      KERNEL<TYPE, 4, BPE>(in, out, in_size, out_size, info); \
    else if (num_dims == 5)\
      KERNEL<TYPE, 5, BPE>(in, out, in_size, out_size, info); \
    else if (num_dims <= 1)\
      KERNEL<TYPE, 1, BPE>(in, out, in_size, out_size, info); \
} \


UNARY_COPY_OP(ucopy_multithread_cache, __bf16, vbfloat, ucopy_bf16, 2) 
UNARY_COPY_OP(ucopy_multithread_cache, __fp16, vhalf, ucopy_f16, 2)
UNARY_COPY_OP(ucopy_multithread_cache, float, vfloat, ucopy_f32, 4)
UNARY_COPY_OP(ucopy_multithread_cache, u_int8_t, vuchar, ucopy_u8, 2)
UNARY_COPY_OP(ucopy_multithread_cache, u_int32_t, vuint, ucopy_u32, 4)
UNARY_COPY_OP(ucopy_multithread_cache, double, vfloatx2, ucopy_f64, 8) //remove this for gather copy

// UNARY_OP(int8_t, vchar, ucopy_i8, UNARY_TYPE_COPY)
// UNARY_OP(u_int8_t, vuchar, ucopy_u8, UNARY_TYPE_COPY)
// UNARY_OP(int32_t, vint, ucopy_i32, UNARY_TYPE_COPY)

int test() {
  __fp16 *lhs_d, *out_d;
  int *shape_lhs_d;
  __fp16 *lhs_h, *out_h;
  size_t size_lhs = 64;
  size_t size_out = size_lhs;
  size_t dim = 1;
  topsHostMalloc((__fp16**)&lhs_h, size_lhs * sizeof(__fp16));
  topsHostMalloc((__fp16**)&out_h, size_out * sizeof(__fp16));

    for (size_t i = 0; i < size_lhs; i++) {
        lhs_h[i] = 0.5f;
    }
    for (size_t i = 0; i < size_out; i++) {
        out_h[i] = 0.0;
    }
  topsMalloc(&lhs_d, size_lhs * sizeof(__fp16));
  topsMalloc(&out_d, size_out * sizeof(__fp16));

  printf("info: copy Host2Device\n");
  topsMemcpy(lhs_d, lhs_h, size_lhs * sizeof(__fp16),
                  topsMemcpyHostToDevice);
  topsMemcpy(out_d, out_h, size_out * sizeof(__fp16),
                  topsMemcpyHostToDevice);
  // int grids = size_out/tile_size;
  // int threads = 6;
  // if (grids < 1) {
  //   grids = 1;
  // } else if (grids / 6 > 0) {
  //   grids = grids / 6;
  // } else {
  //   threads = 1;
  // }

  // ucopy_f16<<<dim3(1, 1, 1), dim3(1, 12, 1)>>>(size_out, lhs_d, out_d);

  printf("info: copy Device2Host\n");
  topsMemcpy(out_h, out_d, size_out * sizeof(__fp16), topsMemcpyDeviceToHost);

  for (size_t j = 0; j < size_out; j++) {
      printf("%.2f, ", out_h[j]);
  }
  topsHostFree(lhs_h);
  topsHostFree(out_h);
  topsFree(lhs_d);
  topsFree(out_d);
  return 0;
}

int main() {
    return test();
}