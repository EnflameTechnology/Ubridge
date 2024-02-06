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
 * @date    2023-12-02 to 2024-02-04
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

template <typename ID_TYPENAME, typename T>
__device__ __forceinline__ void where_kernel(ID_TYPENAME* ids, T* in1, T* in2, T* out, const size_t element_num) {
    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNum();
    // const int BLOCK_SIZE = TILE_SIZE;
    int N = element_num / TILE_SIZE;
    __local__ __valigned__ ID_TYPENAME ids_buffer[TILE_SIZE];
    __local__ __valigned__ T in_buffer1[TILE_SIZE];
    __local__ __valigned__ T in_buffer2[TILE_SIZE];
    __local__ __valigned__ T out_buffer[TILE_SIZE];

    int block_size = TILE_SIZE;
    int remains = 0;
    if (N < 1) {
      N = 1;
      block_size = element_num;
    } else if (element_num % TILE_SIZE > 0) {
      remains = element_num % TILE_SIZE;
    }
    tops_dte_ctx_t ctxs_in[3];
    tops_dte_ctx_t ctx;

    tops_dte_ctx_t ctxs_out;
    ctxs_in[0].init();
    ctxs_in[1].init();
    ctxs_in[2].init();
    ctxs_out.init();
    tops::dte_scope s_in0(ctxs_in[0]);
    tops::dte_scope s_in1(ctxs_in[1]);
    tops::dte_scope s_in2(ctxs_in[2]);

    tops::dte_scope s_out0(ctxs_out);
    tops::dte_scope s_out01(ctx);

    tops::mdspan hbm_ids(tops::Global, ids, N, block_size);
    tops::mdspan hbm_in1(tops::Global, in1, N, block_size);
    tops::mdspan hbm_in2(tops::Global, in2, N, block_size);
    tops::mdspan hbm_out(tops::Global, out, N, block_size);

    tops::mdspan l1_ids(tops::Private, ids_buffer, 1, block_size);
    tops::mdspan l1_in1(tops::Private, in_buffer1, 1, block_size);
    tops::mdspan l1_in2(tops::Private, in_buffer2, 1, block_size);
    tops::mdspan l1_out(tops::Private, out_buffer, 1, block_size);

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

    // printf("N %d, MAX_THREADS %d, THREAD_STEP %d, thread_step %d", N, MAX_THREADS, THREAD_STEP, thread_step);

    for (int i = 0; i < thread_step; i++) {
      int idx = thread_id * THREAD_STEP + i;
      if (idx < N) {
          ctxs_in[0].config_slice(l1_ids, hbm_ids, {0, 0});
          ctxs_in[0].set_src_offset(0, idx);
          ctxs_in[0].trigger_and_wait();

          ctxs_in[1].config_slice(l1_in1, hbm_in1, {0, 0});
          ctxs_in[1].set_src_offset(0, idx);
          ctxs_in[1].trigger_and_wait();

          ctxs_in[2].config_slice(l1_in2, hbm_in2, {0, 0});
          ctxs_in[2].set_src_offset(0, idx);
          ctxs_in[2].trigger_and_wait();

          atomic_where<ID_TYPENAME, T>(ids_buffer, in_buffer1, in_buffer2, out_buffer, block_size);

        // printf("\nInput buffer1: ");
        // for (int j=0; j<block_size; j++) {
        //   printf("%.5f ", static_cast<float>(in_buffer1[j]));
        // }
        // printf("\nInput buffer2: ");
        // for (int j=0; j<block_size; j++) {
        //   printf("%.5f ", static_cast<float>(in_buffer2[j]));
        // }
        // printf("\nOut buffer: ");
        // for (int j=0; j<block_size; j++) {
        //   printf("%.5f ", static_cast<float>(out_buffer[j]));
        // }

          ctxs_out.config_deslice(hbm_out, l1_out, {0, 0});
          ctxs_out.set_dst_offset(0, idx);
          ctxs_out.trigger_and_wait();
      }
    }

    if (remains > 0 && thread_id == MAX_THREADS - 1) {
        tops::mdspan srcIds(tops::Global, ids + N * block_size, remains);
        tops::mdspan src1(tops::Global, in1 + N * block_size, remains);
        tops::mdspan src2(tops::Global, in2 + N * block_size, remains);

        tops::mdspan l1_ids(tops::Private, ids_buffer, remains);
        tops::mdspan l1_in1(tops::Private, in_buffer1, remains);
        tops::mdspan l1_in2(tops::Private, in_buffer2, remains);
        tops::mdspan l1_out(tops::Private, out_buffer, remains);

        tops::memcpy(ctx, l1_ids, srcIds);
        tops::memcpy(ctx, l1_in1, src1);
        tops::memcpy(ctx, l1_in2, src2);
        atomic_where<ID_TYPENAME, T>(ids_buffer, in_buffer1, in_buffer2, out_buffer, remains);

        tops::mdspan dstOut(tops::Global, out + N * block_size, remains);
        tops::memcpy(ctx, dstOut, l1_out);
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
