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
 * @file    ternary.cpp
 * @brief
 *
 * @author  Guoqing Bao
 * @date    2023-12-02 to 2024-02-29
 * @version V0.1
 * @par     Copyright (c) Enflame Tech Company.
 * @par     History: use custom atomic_where
 * @par     Comments: gcu kernel for ternary/where operations.
 */

#include <stdio.h>
#include <tops.h>
#include <tops/half.h>
#include <algorithm>
#include <vector>
#include "tops/tops_runtime.h"
#include "utils.h"
#include "tcle.h"
using namespace tcle;
using namespace std;
#define TILE_SIZE AlignDown(((VDMEM_SIZE) / 32), 256)

template <typename ID_TYPENAME, typename T>
__device__ __forceinline__ void atomic_where(ID_TYPENAME* ids_ptr, T* src_ptr1, T* src_ptr2, 
                                            T* dst_ptr,
                                            unsigned int elements) {
    for (int i=0 ;i< elements; i++) {
      dst_ptr[i] = ids_ptr[i] ? src_ptr1[i] : src_ptr2[i];
    }

}

template <typename T>
__device__ __forceinline__ void atomic_where_u8(unsigned char* ids_ptr, T* src_ptr1, T* src_ptr2, 
                                            T* dst_ptr,
                                            unsigned int elements) {
  constexpr int vlength = (sizeof(int) / sizeof(T)) * TCLE_MAX_VECTOR_LENGTH;
  using vtype = typename altivector<T, vlength>::VT;
  using mask_vtype = typename altivector_to_mask<vtype>::type;
  using cond_vtype = typename altivector<unsigned char, vlength>::VT;

  tcle::leaptr<vtype> intput_ptr = tcle::simple_leaptr<vtype>(src_ptr1);
  tcle::leaptr<vtype> other_ptr = tcle::simple_leaptr<vtype>(src_ptr2);
  tcle::leaptr<vtype> output_ptr = tcle::simple_leaptr<vtype>(dst_ptr);

  int group_num = (elements + vlength - 1) / vlength;
  cond_vtype cond;
  mask_vtype mask;
  vtype casted_cond, v_input, v_other;
  vtype v0 = (vtype)(0);
  for (int i = 0; i < group_num; i++) {
    cond = tcle::load<cond_vtype>((__TCLE_AS__ char *)ids_ptr);
    ids_ptr += vlength;
    v_input = intput_ptr.load();
    v_other = other_ptr.load();
    casted_cond = cvt<vtype>(cond);
    mask = (casted_cond == v0);
    v_input = vsel(mask, v_other, v_input);
    output_ptr.store(v_input);
  }
}

template <typename ID_TYPENAME, typename T>
__device__ void where_kernel(ID_TYPENAME* ids, T* in1, T* in2, T* out, const size_t len) {
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx); 
    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNum();

    __local__ __valigned__ ID_TYPENAME ids_buffer[TILE_SIZE];
    __local__ __valigned__ T buffer1[TILE_SIZE];
    __local__ __valigned__ T buffer2[TILE_SIZE];
    __local__ __valigned__ T bufferO[TILE_SIZE];
    tops::mdspan ids_buffer_l1(tops::Private, ids_buffer, TILE_SIZE);
    tops::mdspan buffer1_l1(tops::Private, buffer1, TILE_SIZE);
    tops::mdspan buffer2_l1(tops::Private, buffer2, TILE_SIZE);
    tops::mdspan bufferO_l1(tops::Private, bufferO, TILE_SIZE);

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
      tops::memcpy(ctx, ids_buffer_l1, tops::mdspan(tops::Global, ids + offset, bufsize));
      tops::memcpy(ctx, buffer1_l1, tops::mdspan(tops::Global, in1 + offset, bufsize));
      tops::memcpy(ctx, buffer2_l1, tops::mdspan(tops::Global, in2 + offset, bufsize));
      if (std::is_same<ID_TYPENAME, u_int8_t>::value && std::is_same<T, float>::value) {
        atomic_where_u8<float>(reinterpret_cast<unsigned char*>(ids_buffer), reinterpret_cast<float*>(buffer1), reinterpret_cast<float*>(buffer2), reinterpret_cast<float*>(bufferO), bufsize);
      }  else {
        atomic_where<ID_TYPENAME, T>(ids_buffer, buffer1, buffer2, bufferO, bufsize);
      }
      tops::memcpy(ctx, tops::mdspan(tops::Global, out + offset, bufsize), tops::mdspan(tops::Private, bufferO, bufsize));
    }
}

#define WHERE_OP(TYPE, ID_TYPE, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    ID_TYPE* ids, \
    TYPE *in1, \
    TYPE *in2, \
    TYPE *out, \
    const size_t element_num) \
{ \
    where_kernel<ID_TYPE, TYPE>(ids, in1, in2, out, element_num); \
} \

WHERE_OP(float, int64_t, where_i64_f32)
WHERE_OP(double, int64_t, where_i64_f64)
WHERE_OP(uint8_t, int64_t, where_i64_u8)
WHERE_OP(uint32_t, int64_t, where_i64_u32)
WHERE_OP(int64_t, int64_t, where_i64_i64)

WHERE_OP(float, uint32_t, where_u32_f32)
WHERE_OP(double, uint32_t, where_u32_f64)
WHERE_OP(uint8_t, uint32_t, where_u32_u8)
WHERE_OP(uint32_t, uint32_t, where_u32_u32)
WHERE_OP(int64_t, uint32_t, where_u32_i64)

WHERE_OP(float, uint8_t, where_u8_f32)
WHERE_OP(double, uint8_t, where_u8_f64)
WHERE_OP(uint8_t, uint8_t, where_u8_u8)
WHERE_OP(uint32_t, uint8_t, where_u8_u32)
WHERE_OP(int64_t, uint8_t, where_u8_i64)
WHERE_OP(__fp16, uint8_t, where_u8_f16)
WHERE_OP(__bf16, uint8_t, where_u8_bf16)

int main(void) {
  topsError_t err = topsSuccess;
  int N = 64 * 4096 + 5;
  int in_size = N * sizeof(float);

  uint8_t *host_ids = reinterpret_cast<uint8_t*>(aligned_alloc(128, N * sizeof(uint8_t)));
  float *host_in1 = reinterpret_cast<float*>(aligned_alloc(128, in_size));
  float *host_in2 = reinterpret_cast<float*>(aligned_alloc(128, in_size));
  float *host_out = reinterpret_cast<float*>(aligned_alloc(128, in_size));

  for (int j =0; j< N; j ++) {
    host_ids[j] = (j % 2 == 1) ? 1:0;
    host_in1[j] = 5.0;
    host_in2[j] = 1.0;
  }

  float *dev_in1 = NULL;
  CHECK(topsMalloc(reinterpret_cast<void **>(&dev_in1), in_size));

  float *dev_in2 = NULL;
  CHECK(topsMalloc(reinterpret_cast<void **>(&dev_in2), in_size));
  
  float *dev_out = NULL;
  CHECK(topsMalloc(reinterpret_cast<void **>(&dev_out), in_size));

  uint8_t *dev_ids = NULL;
  CHECK(topsMalloc(reinterpret_cast<void **>(&dev_ids), N * sizeof(uint8_t)));

  CHECK(topsMemcpy(dev_in1, host_in1, in_size, topsMemcpyHostToDevice));
  CHECK(topsMemcpy(dev_in2, host_in2, in_size, topsMemcpyHostToDevice));
  CHECK(topsMemcpy(dev_ids, host_ids, N * sizeof(uint8_t), topsMemcpyHostToDevice));
  CHECK(topsMemset(dev_out, 0, in_size));

  printf("call where kernel!!!!!!!!!!!!!\n");

  float time = 0.0;
  float total_time = 0.0;
  topsEvent_t start, stop;
  int ITERATION = 20;
  for (int i=0; i< ITERATION; i++) {
    CHECK(topsEventCreate(&start));
    CHECK(topsEventCreate(&stop));

    CHECK(topsEventRecord(start));

    where_u8_f32<<<1, 12>>>(dev_ids, dev_in1, dev_in2, dev_out, N);

    CHECK(topsGetLastError());

    CHECK(topsEventRecord(stop));
    CHECK(topsEventSynchronize(stop));
    CHECK(topsEventElapsedTime(&time, start, stop));
    total_time += time;
  }
  
  printf("Avg Time taken: %g ms --------\n", total_time/ITERATION);


  if (topsGetLastError() != topsSuccess) {
    fprintf(stderr, "Failed to launch kernel (error code %s)!\n",
            topsGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  CHECK(topsMemcpy(host_out, dev_out, in_size, topsMemcpyDeviceToHost));

  // for (int i = 0; i < N; i++) {
  //   fprintf(stderr, "Result  element %d, %f!\n", i, host_out[i]);
  // }

  printf("Test PASSED\n");
  
  CHECK(topsFree(dev_in1));
  CHECK(topsFree(dev_in2));
  CHECK(topsFree(dev_ids));
  CHECK(topsFree(dev_out));
  free(host_in1);
  free(host_in2);
  free(host_ids);
  free(host_out);

  return 0;
}
