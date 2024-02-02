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
#include <acore/atomic_op.h>
#include "utils.h"
using namespace std;
using namespace tops;
#define TILE_SIZE AlignDown(((VDMEM_SIZE) / 16), 256)

template <typename T, typename OUTT>
__device__ void cast_kernel(T* in, OUTT* out, int len) {
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx); 
    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNum();

    __local__ __valigned__ T buffer1[TILE_SIZE];
    __local__ __valigned__ OUTT buffer2[TILE_SIZE];
    tops::mdspan buffer_l1(tops::Private, buffer1, TILE_SIZE);

    tops::mdspan out_hbm(tops::Global, out, len);
    int N = len;
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

    for (int i = 0; i < thread_step; i+=TILE_SIZE) {
      int bufsize = (i + TILE_SIZE < thread_step) ? TILE_SIZE : thread_step - i;
      int offset = thread_id * THREAD_STEP + i;
      tops::memcpy(ctx, buffer_l1, tops::mdspan(tops::Global, in + offset, bufsize));
      convert<OUTT, T>(reinterpret_cast<OUTT*>(buffer2), buffer1, bufsize);
      tops::memcpy(ctx, tops::mdspan(tops::Global, out + offset, bufsize), tops::mdspan(tops::Private, buffer2, bufsize));
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
CAST_OP(__bf16, int16_t, cast_bf16_i16)
CAST_OP(__bf16, int32_t, cast_bf16_i32)
CAST_OP(__bf16, __fp16, cast_bf16_f16)
CAST_OP(__bf16, float, cast_bf16_f32)

// CAST_OP(__fp16, int8_t, cast_f16_i8)
CAST_OP(__fp16, int16_t, cast_f16_i16)
CAST_OP(__fp16, int32_t, cast_f16_i32)
CAST_OP(__fp16, float, cast_f16_f32)
CAST_OP(__fp16, __bf16, cast_f16_bf16)

// CAST_OP(float, int8_t, cast_f32_i8)
CAST_OP(float, int16_t, cast_f32_i16)
CAST_OP(float, int32_t, cast_f32_i32)
CAST_OP(float, __fp16, cast_f32_f16)
CAST_OP(float, __bf16, cast_f32_bf16)

// CAST_OP(int8_t, int16_t, cast_i8_i16)
// CAST_OP(int8_t, int32_t, cast_i8_i32)
// CAST_OP(int8_t, float, cast_i8_f32)
// CAST_OP(int8_t, __fp16, cast_i8_f16)
// CAST_OP(int8_t, __bf16, cast_i8_bf16)

// CAST_OP(int16_t, int8_t, cast_i16_i8)
CAST_OP(int16_t, int32_t, cast_i16_i32)
CAST_OP(int16_t, float, cast_i16_f32)
CAST_OP(int16_t, __fp16, cast_i16_f16)
CAST_OP(int16_t, __bf16, cast_i16_bf16)

// CAST_OP(int32_t, int8_t, cast_i32_i8)
CAST_OP(int32_t, int16_t, cast_i32_i16)
CAST_OP(int32_t, float, cast_i32_f32)
CAST_OP(int32_t, __fp16, cast_i32_f16)
CAST_OP(int32_t, __bf16, cast_i32_bf16)


CAST_OP(u_int8_t, u_int16_t, cast_u8_u16)
CAST_OP(u_int8_t, u_int32_t, cast_u8_u32)
CAST_OP(u_int8_t, float, cast_u8_f32)
CAST_OP(u_int8_t, __fp16, cast_u8_f16)
CAST_OP(u_int8_t, __bf16, cast_u8_bf16)

CAST_OP(u_int16_t, u_int8_t, cast_u16_u8)
CAST_OP(u_int16_t, u_int32_t, cast_u16_u32)
CAST_OP(u_int16_t, float, cast_u16_f32)
CAST_OP(u_int16_t, __fp16, cast_u16_f16)
CAST_OP(u_int16_t, __bf16, cast_u16_bf16)

CAST_OP(u_int32_t, u_int8_t, cast_u32_u8)
CAST_OP(u_int32_t, u_int16_t, cast_u32_u16)
CAST_OP(u_int32_t, float, cast_u32_f32)
CAST_OP(u_int32_t, __fp16, cast_u32_f16)
CAST_OP(u_int32_t, __bf16, cast_u32_bf16)

template<typename T, typename OUTT>
int test() {
  T *lhs_d; OUTT *out_d;
  int *shape_lhs_d;
  T *lhs_h; OUTT *out_h;
  size_t size_lhs = 64*4096;
  size_t size_out = size_lhs;
  size_t dim = 1;
  topsHostMalloc((T**)&lhs_h, size_lhs * sizeof(T));
  topsHostMalloc((T**)&out_h, size_out * sizeof(OUTT));
    // T a = 0.5;
    OUTT zero = 0.0;
    for (size_t i = 0; i < size_lhs; i++) {
        lhs_h[i] = static_cast<T>(0.5);
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

  cast_f16_f32<<<dim3(1, 1, 1), dim3(12, 1, 1)>>>(size_out, lhs_d, out_d);

  printf("info: copy Device2Host\n");
  topsMemcpy(out_h, out_d, size_out * sizeof(OUTT), topsMemcpyDeviceToHost);

  for (size_t j = 0; j < size_out; j++) {
      OUTT dif = static_cast<OUTT>(0.5) - static_cast<OUTT>(out_h[j]);
      if (dif > static_cast<OUTT>(0.0000001f))
        printf("%.5f, ", dif);
  }
  topsHostFree(lhs_h);
  topsHostFree(out_h);
  topsFree(lhs_d);
  topsFree(out_d);
  return 0;
}

int main() {
    return test<__fp16, float>();
}