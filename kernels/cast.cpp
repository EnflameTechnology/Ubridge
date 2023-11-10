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

#include <tops/tops_runtime.h>
#include <tops/topsrtc.h>
#include <krt/scalar.h>
#include <krt/vector_mask.h>
#include <krt/dispatch.h>
#include <krt/leaptr.h>
#include <krt/vector_infra.h>

#if __GCU_ARCH__ >= 300
#include "include/common/atomic_op.h"
// #include "/home/kernel/atomicop_pkg/src/include/common/atomic_op.h"
#endif

#include "utils.h"
#include "cast_atomic.h"
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

template <typename T, typename OUTT>
__device__ void cast_kernel(T* in, OUTT* out, int len) {
  tops_dte_ctx_t ctxs_in[PING_PONG_SIZE];
  tops_dte_ctx_t ctxs_out[PING_PONG_SIZE];
  tops_dte_ctx_t ctxs_cp;
  tops::event evs_in[PING_PONG_SIZE];
  tops::event evs_out[PING_PONG_SIZE];
  __local__ __valigned__ T in_buffer[PING_PONG_SIZE][tile_size];
  __local__ __valigned__ OUTT out_buffer[PING_PONG_SIZE][tile_size];

  int N = len;
  tops::mdspan output(tops::Global, out, N);

  int thread_num = GetThreadNum();
  int thread_id = GetThreadIdx();

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
      ctxs_in[pp_flag].config_memcpy(
        tops::mdspan(tops::Private, in_buffer[pp_flag],
                     thread_len_next2),
        tops::mdspan(tops::Global, in + thread_off_next2,
                     thread_len_next2));
    }

    // call atomic op here
    if (thread_len > 0) {
      cast_atomic<T, OUTT, RoundingMode_t::RM_DEFAULT>(reinterpret_cast<OUTT*>(out_buffer), reinterpret_cast<T*>(in_buffer), thread_len);

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

#define CAST_OP(TYPE, OUTTYPE, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    TYPE *inp, \
    OUTTYPE *out) \
{ \
    cast_kernel<TYPE, OUTTYPE>(inp, out, numel); \
} \


/*         1. bf16 -> fp16
 *         2. bf16 -> fp32
 *         3. bf16 -> int16
 *         4. bf16 -> int32
 *         5. bf16 -> int8

 *         6. fp16 -> bf16
 *         7. fp16 -> fp32
 *         8. fp16 -> int16
 *         9. fp16 -> int32
 *         10. fp16 -> int8

 *         11. fp32 -> bf16
 *         12. fp32 -> fp16
 *         13. fp32 -> int16
 *         14. fp32 -> int32
 *         15. fp32 -> int8

 *         16. int16 -> bf16
 *         17. int16 -> fp16
 *         18. int16 -> fp32
 *         19. int16 -> int32
 *         20. int16 -> int8

 *         21. int32 -> bf16
 *         22. int32 -> fp16
 *         23. int32 -> fp32
 *         24. int32 -> int16
 *         25. int32 -> int8
 *         26. int8 -> bf16
 *         27. int8 -> fp16
 *         28. int8 -> fp32
 *         29. int8 -> int16
 *         30. int8 -> int32

 *         31. uint8 -> uint16
 *         32. uint8 -> uint32
 *         33. uint8 -> bf16
 *         34. uint8 -> fp16
 *         35. uint8 -> fp32

 *         36. uint16 -> uint8
 *         37. uint16 -> uint32
 *         38. uint16 -> bf16
 *         39. uint16 -> fp16
 *         40. uint16 -> fp32
 *         41. uint32 -> uint8
 *         42. uint32 -> uint16
 *         43. uint32 -> bf16
 *         44. uint32 -> fp16
 *         45. uint32 -> fp32
 */

// CAST_OP(__bf16, int8_t, cast_bf16_i8)
// CAST_OP(__bf16, int16_t, cast_bf16_i16)
// CAST_OP(__bf16, int32_t, cast_bf16_i32)
// CAST_OP(__bf16, __fp16, cast_bf16_fp16)
// CAST_OP(__bf16, float, cast_bf16_fp32)

CAST_OP(__fp16, int8_t, cast_fp16_i8)
CAST_OP(__fp16, int16_t, cast_fp16_i16)
CAST_OP(__fp16, int32_t, cast_fp16_i32)
CAST_OP(__fp16, float, cast_fp16_fp32)
// CAST_OP(__fp16, __bf16, cast_fp16_bf16)

CAST_OP(float, int8_t, cast_f32_i8)
CAST_OP(float, int16_t, cast_f32_i16)
CAST_OP(float, int32_t, cast_f32_i32)
CAST_OP(float, __fp16, cast_f32_fp16)
// CAST_OP(float, __bf16, cast_f32_bf16)

CAST_OP(int8_t, int16_t, cast_i8_i16)
CAST_OP(int8_t, int32_t, cast_i8_i32)
CAST_OP(int8_t, float, cast_i8_fp32)
CAST_OP(int8_t, __fp16, cast_i8_fp16)
// CAST_OP(int8_t, __bf16, cast_i8_bf16)

CAST_OP(int16_t, int8_t, cast_i16_i8)
CAST_OP(int16_t, int32_t, cast_i16_i32)
CAST_OP(int16_t, float, cast_i16_fp32)
CAST_OP(int16_t, __fp16, cast_i16_fp16)
// CAST_OP(int16_t, __bf16, cast_i16_bf16)

CAST_OP(int32_t, int8_t, cast_i32_i8)
CAST_OP(int32_t, int16_t, cast_i32_i16)
CAST_OP(int32_t, float, cast_i32_fp32)
CAST_OP(int32_t, __fp16, cast_i32_fp16)
// CAST_OP(int32_t, __bf16, cast_i32_bf16)


CAST_OP(u_int8_t, u_int16_t, cast_u8_u16)
CAST_OP(u_int8_t, u_int32_t, cast_u8_u32)
CAST_OP(u_int8_t, float, cast_u8_f32)
CAST_OP(u_int8_t, __fp16, cast_u8_fp16)
// CAST_OP(u_int8_t, __bf16, cast_u8_bf16)

CAST_OP(u_int16_t, u_int8_t, cast_u16_u8)
CAST_OP(u_int16_t, u_int32_t, cast_u16_u32)
CAST_OP(u_int16_t, float, cast_u16_f32)
CAST_OP(u_int16_t, __fp16, cast_u16_fp16)
// CAST_OP(u_int16_t, __bf16, cast_u16_bf16)

CAST_OP(u_int32_t, u_int8_t, cast_u32_u8)
CAST_OP(u_int32_t, u_int16_t, cast_u32_u16)
CAST_OP(u_int32_t, float, cast_u32_f32)
CAST_OP(u_int32_t, __fp16, cast_u32_fp16)
// CAST_OP(u_int32_t, __bf16, cast_u32_bf16)

template<typename T, typename OUTT>
int test() {
  T *lhs_d; OUTT *out_d;
  int *shape_lhs_d;
  T *lhs_h; OUTT *out_h;
  size_t size_lhs = 1024;
  size_t size_out = size_lhs;
  size_t dim = 1;
  topsHostMalloc((T**)&lhs_h, size_lhs * sizeof(T));
  topsHostMalloc((T**)&out_h, size_out * sizeof(OUTT));
    T a = 0.5;
    OUTT zero = 0.0;
    for (size_t i = 0; i < size_lhs; i++) {
        lhs_h[i] = static_cast<T>(a);
    }
    for (size_t i = 0; i < size_out; i++) {
        out_h[i] = static_cast<OUTT>(zero);
    }
  topsMalloc(&lhs_d, size_lhs * sizeof(T));
  topsMalloc(&out_d, size_out * sizeof(OUTT));

  printf("info: copy Host2Device\n");
  topsMemcpy(lhs_d, lhs_h, size_lhs * sizeof(T),
                  topsMemcpyHostToDevice);
  topsMemcpy(out_d, out_h, size_out * sizeof(OUTT),
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

  cast_u32_f32<<<dim3(grids, 1, 1), dim3(threads, 1, 1)>>>(size_out, lhs_d, out_d);

  printf("info: copy Device2Host\n");
  topsMemcpy(out_h, out_d, size_out * sizeof(T), topsMemcpyDeviceToHost);

  for (size_t j = 0; j < size_out; j++) {
      printf("%.2f, ", static_cast<OUTT>(out_h[j]));
  }
  topsHostFree(lhs_h);
  topsHostFree(out_h);
  topsFree(lhs_d);
  topsFree(out_d);
  return 0;
}

int main() {
    return test<u_int32_t, float>();
}