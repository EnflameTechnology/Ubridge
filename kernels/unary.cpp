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

#include <krt/scalar.h>
#include <krt/vector_mask.h>
#include <krt/dispatch.h>
#include <krt/leaptr.h>
#include <krt/vector_infra.h>

#if __GCU_ARCH__ < 300 
#include "pavo/vector_impl.h"
#include "pavo/vector_mask_impl.h"
#include "pavo/perfmon_impl.h"
#endif

#if __GCU_ARCH__ >= 300
#include "include/common/atomic_op.h"
// #include "/home/kernel/atomicop_pkg/src/include/common/atomic_op.h"
#endif

#include <tops.h>
#include <tops/tops_runtime.h>
#include <tops/topsrtc.h>
#include <tops/half.h>
#include <tops/bfloat.h>



#include "utils.h"
using namespace std;
using namespace tops;
#define tile_size 0x8000
#define PING_PONG_SIZE 2

namespace tops {
template <typename T>
__device__ __host__ __forceinline__ constexpr int hvlength() {
  return 128 / sizeof(T);
}

} // namespace tops

__device__ __forceinline__
int get_index() {
    std::size_t blockIndex = blockIdx.z*(gridDim.x*gridDim.y)
        + blockIdx.y*gridDim.x + blockIdx.x;
    std::size_t threadIndex = threadIdx.z*(blockDim.x*blockDim.y)
        + threadIdx.y*blockDim.x + threadIdx.x;
    return blockIndex*(blockDim.x*blockDim.y*blockDim.z) + threadIndex;
}


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

// template <typename T>
// __device__ void gelu(T* output, T* input, int num) {
//   using vtype = typename scalar_to_vector<T,
//                                           TOPS_VECTOR_LENGTH>::type;
//   const int vlength = vector_length<vtype>::value;
//   leaptr<vtype> intput_ptr = simple_leaptr<vtype>(input);
//   leaptr<vtype> output_ptr = simple_leaptr<vtype>(output);
//   int group_num = (num + vlength - 1) / vlength;
//   vtype v_input, v_output;
//   for (int i = 0; i < group_num; i++) {
//     v_input = intput_ptr.load();
//     v_output = vgelu(v_input);
//     output_ptr.store(v_output);
//   }
// }

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
  #if __GCU_ARCH__ >= 300 
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
        // gelu(out, in, len);
        
        break;
      }
    case UNARY_TYPE_RELU:
      {
        // relu(out, in , len);
        // max(out, in, )
        // dst = tops::vmax<VT>(src, tops::vzero<VT>());
        break;
      }
    case UNARY_TYPE_ELU:
      {
        // elu(out, in, len);
        break;
      }
    case UNARY_TYPE_SILU:
      {
        // elu(out, in, len);
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
        // dst = src;
        break;
      }
    default:
      break;
    }
  #endif

  #if __GCU_ARCH__ < 300 
  printf("len {%d}, type {%d}", len, tp);
  constexpr int vlen = tops::hvlength<VT>();

  generic_ptr src_addr = reinterpret_cast<generic_ptr>(in);
  generic_ptr dst_addr = reinterpret_cast<generic_ptr>(out);

  constexpr int bpe = sizeof(T);
  constexpr int vec_elems =
      sizeof(typename tops::unified_scalar<T>::type) * TOPS_VECTOR_LENGTH / bpe;
  using vtype = typename tops::scalar_to_vector<T, vec_elems>::type;

  // auto src_leaptr = leaptr<T>(src_addr, 1);
  leaptr<vtype> src_leaptr = simple_leaptr<vtype>(src_addr);
  leaptr<vtype> dst_leaptr = simple_leaptr<vtype>(dst_addr);

  int group_num = (len + vec_elems - 1) / vec_elems;

  vtype src, dst;

  for (int i = 0; i < group_num; i++) {
    src = src_leaptr.load();
    switch (tp) {
    case UNARY_TYPE_NEG:
      {
        dst = tops::vneg<VT>(src);
        break;
      }
    case UNARY_TYPE_EXP:
      {
        dst = tops::vexp<VT>(src);
        break;
      }
    case UNARY_TYPE_LOG:
      {
        dst = tops::vlog<VT>(src);
        break;
      }
    case UNARY_TYPE_SIN:
      {
        dst = tops::vsin<vbfloat>(src);
        break;
      }
    case UNARY_TYPE_COS:
      {
        dst = tops::vcos<VT>(src);
        break;
      }
    case UNARY_TYPE_ABS:
      {
        dst = tops::vabs<VT>(src);
        break;
      }
    case UNARY_TYPE_MUL:
      {
        dst = tops::vneg<VT>(src);
        break;
      }
    case UNARY_TYPE_SQUARE:
      {
        dst = tops::vmul<VT>(src, src);
        break;
      }
    case UNARY_TYPE_SQRT:
      {
        dst = tops::vsqrt<VT>(src);
        break;
      }
    case UNARY_TYPE_GELU:
      {
        dst = tops::vgelu<VT>(src);
        break;
      }
    case UNARY_TYPE_RELU:
      {
        dst = tops::vmax<VT>(src, tops::vzero<VT>());
        break;
      }
    // case UNARY_TYPE_ELU:
    //   break;
    case UNARY_TYPE_COPY:
      {
        dst = src;
        break;
      }
    default:
      break;
    }

    dst_leaptr.store(dst);
  }


// UNARY_OP1(tops::bfloat, vbfloat, uelu_bf16, elu_fwd(x[i], param))
#endif


}


template <typename T, typename VT>
__device__ void unary_kernel(T* in, T* out, int len, UNARY_TYPE tp) {
  tops_dte_ctx_t ctxs_in[PING_PONG_SIZE];
  tops_dte_ctx_t ctxs_out[PING_PONG_SIZE];
  tops::event evs_in[PING_PONG_SIZE];
  tops::event evs_out[PING_PONG_SIZE];
  __local__ __valigned__ T in_buffer[PING_PONG_SIZE][tile_size];
  __local__ __valigned__ T out_buffer[PING_PONG_SIZE][tile_size];
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
      unary_atomic<T, VT>(in_buffer[pp_flag], out_buffer[pp_flag],
                               thread_len, tp);
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

// #define UNARY_OP(TYPE, VT, FN_NAME, TP) \
// extern "C" __global__ void FN_NAME( \
//     const size_t numel, \
//     const size_t num_dims, \
//     const size_t *info, \
//     TYPE *inp, \
//     TYPE *out) \
// { \
//     unary_kernel<TYPE, VT>(inp, out, numel, TP); \
// } \

#define UNARY_OP(TYPE, VT, FN_NAME, TP) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    TYPE *inp, \
    TYPE *out) \
{ \
    unary_kernel<TYPE, VT>(inp, out, numel, TP); \
} \

// #define UNARY_OP(TYPE, VT, FN_NAME, FUNC) \
// extern "C" __global__ void FN_NAME( \
//     const size_t numel, \
//     const size_t num_dims, \
//     const size_t *info, \
//     TYPE *inp, \
//     TYPE *out) \
// { \
//     tops_dte_ctx_t ctx; \
//     tops::dte_scope s(ctx); \
//     std::size_t idx = get_index(); \
//     constexpr std::size_t num_len = tops::hvlength<VT>(); \
//     __valigned__ TYPE buffer1[num_len]; \
//     tops::mdspan buf1(tops::Private, &buffer1, num_len); \
//     tops::mdspan src1(tops::Global, inp + idx * num_len, num_len); \
//     tops::memcpy(ctx, buf1, src1); \
//     const auto &x = tops::vload<VT>(buffer1);  \
//     tops::mdspan dst(tops::Global, out + idx *num_len, num_len); \
//     tops::vstore(FUNC, buffer1);  \
//     tops::memcpy(ctx, dst, buf1); \
// } \


// #define UNARY_COPY_OP(TYPE, VT, FN_NAME)  \
// extern "C" __global__ void FN_NAME(  \
//     const size_t numel,  \
//     const size_t num_dims,  \
//     const size_t *info,  \
//     TYPE *inp,  \
//     TYPE *out) \
// {  \
//     tops_dte_ctx_t ctx; \
//     tops::dte_scope s(ctx); \
//     std::size_t idx = get_index(); \
//     constexpr std::size_t num_len = tops::hvlength<VT>(); \
//     __valigned__ TYPE buffer1[num_len]; \
//     tops::mdspan buf1(tops::Private, &buffer1, num_len); \
//     tops::mdspan src1(tops::Global, inp + idx * num_len, num_len); \
//     tops::memcpy(ctx, buf1, src1); \
//     tops::mdspan dst(tops::Global, out + idx *num_len, num_len);  \
//     tops::memcpy(ctx, dst, buf1); \
// }  \

// template<typename T>
// __device__ __forceinline__ T elu_fwd(T x, T alpha) {
//   if (x > static_cast<T>(0)) {
//     return x;
//   }
//   return alpha * (tops::exp<T>(x) - static_cast<T>(1));
// }

#define UNARY_OP1(TYPE, VT, FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    TYPE param, \
    TYPE *inp, \
    TYPE *out) \
{ \
    tops_dte_ctx_t ctx; \
    tops::dte_scope s(ctx); \
    std::size_t idx = get_index(); \
    constexpr std::size_t num_len = tops::hvlength<VT>(); \
    __valigned__ TYPE buffer1[num_len]; \
    tops::mdspan buf1(tops::Private, &buffer1, num_len); \
    __valigned__ TYPE buffer2[num_len]; \
    tops::mdspan buf2(tops::Private, &buffer2, num_len); \
    tops::mdspan src1(tops::Global, inp + idx * num_len, num_len); \
    tops::memcpy(ctx, buf1, src1); \
    const auto &x = tops::vload<VT>(buffer1);  \
    tops::mdspan dst(tops::Global, out + idx *num_len, num_len); \
    for (int i = 0; i < num_len; i++) { \
        buffer2[i] = FUNC; \
    } \
    tops::memcpy(ctx, dst, buf2); \
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
UNARY_OP(__bf16, vbfloat, uelu_bf16, UNARY_TYPE_ELU) 
UNARY_OP(__bf16, vbfloat, usilu_bf16, UNARY_TYPE_SILU) 
UNARY_OP(__bf16, vbfloat, utanh_bf16, UNARY_TYPE_TANH) 
UNARY_OP(__bf16, vbfloat, urecip_bf16, UNARY_TYPE_RECIP) 
UNARY_OP(__bf16, vbfloat, ucopy_bf16, UNARY_TYPE_COPY) 



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
UNARY_OP(__fp16, vhalf, uelu_f16, UNARY_TYPE_ELU)
UNARY_OP(__fp16, vhalf, usilu_f16, UNARY_TYPE_SILU)
UNARY_OP(__fp16, vhalf, utanh_f16, UNARY_TYPE_TANH)
UNARY_OP(__fp16, vhalf, urecip_f16, UNARY_TYPE_RECIP)
UNARY_OP(__fp16, vhalf, ucopy_f16, UNARY_TYPE_COPY)


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
UNARY_OP(float, vfloat, uelu_f32, UNARY_TYPE_ELU)
UNARY_OP(float, vfloat, usilu_f32, UNARY_TYPE_SILU)
UNARY_OP(float, vfloat, utanh_f32, UNARY_TYPE_TANH)
UNARY_OP(float, vfloat, urecip_f32, UNARY_TYPE_RECIP)
UNARY_OP(float, vfloat, ucopy_f32, UNARY_TYPE_COPY)


// UNARY_OP(int8_t, vchar, ucopy_i8, UNARY_TYPE_COPY)
// UNARY_OP(uint8_t, vuchar, ucopy_u8, UNARY_TYPE_COPY)
// UNARY_OP(int32_t, vint, ucopy_i32, UNARY_TYPE_COPY)
// UNARY_OP(uint32_t, vuint, ucopy_u32, UNARY_TYPE_COPY)

int test() {
  float *lhs_d, *out_d;
  int *shape_lhs_d;
  float *lhs_h, *out_h;
  std::size_t *tmp = new std::size_t[10];
  size_t size_lhs = 1024 * 4;
  size_t size_out = size_lhs;
  size_t dim = 1;
  topsHostMalloc((float**)&lhs_h, size_lhs * sizeof(float));
  topsHostMalloc((float**)&out_h, size_out * sizeof(float));

    for (size_t i = 0; i < size_lhs; i++) {
        lhs_h[i] = 0.5f;
    }
    for (size_t i = 0; i < size_out; i++) {
        out_h[i] = 0.0;
    }
  topsMalloc(&lhs_d, size_lhs * sizeof(float));
  topsMalloc(&out_d, size_out * sizeof(float));

  printf("info: copy Host2Device\n");
  topsMemcpy(lhs_d, lhs_h, size_lhs * sizeof(float),
                  topsMemcpyHostToDevice);
  topsMemcpy(out_d, out_h, size_out * sizeof(float),
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

  uneg_f32<<<dim3(grids, 1, 1), dim3(threads, 1, 1)>>>(size_out, lhs_d, out_d);

  printf("info: copy Device2Host\n");
  topsMemcpy(out_h, out_d, size_out * sizeof(float), topsMemcpyDeviceToHost);

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