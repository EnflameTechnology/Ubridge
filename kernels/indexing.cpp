/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.

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
#include <tops/half.h>
#include <algorithm>
#include <vector>
#include "tops/tops_runtime.h"
#include "utils.h"
using namespace std;
#define TILE_SIZE AlignDown(((VDMEM_SIZE) / 16), 256)

template <typename ID_TYPENAME, typename T>
__device__ __forceinline__ void index_select_kernel(const size_t id_numel,
    ID_TYPENAME *ids, T *inp, T *out,
    const size_t left_size, const size_t dim_size, const size_t right_size) {
    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNumEachBlock();
    int N = id_numel;
    __local__ __valigned__ ID_TYPENAME ids_buffer[4096];

    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx);

    tops::mdspan l1_ids(tops::Private, ids_buffer, N);
    tops::mdspan hbm_ids(tops::Global, ids, N);

    // printf("id_numel %d, left_size %d, dim_size %d, right_size %d \n", N, left_size, dim_size, right_size);

    // for (int i=0; i< N; i++)
    //     printf("%d ", ids_buffer[i]);
    tops::memcpy(ctx, l1_ids, hbm_ids);

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


    for (int i = 0; i < thread_step; i++) {
      int idx = thread_id * THREAD_STEP + i;
      if (idx < N) {
        for (unsigned int j = 0; j < left_size; ++j) {
            int _idx = ids_buffer[idx];
            tops::mdspan hbm_inp(tops::Global, inp + (j * dim_size + _idx) * right_size, right_size);
            tops::mdspan hbm_out(tops::Global, out + (idx + j * N) * right_size, right_size);
            tops::memcpy(ctx, hbm_out, hbm_inp);
            // memcpy(&out[(i + j * numel) * right_size], &inp[(j * dim_size + ids[i]) * right_size], right_size * sizeof(T));
        }
      }
    }


}

#define IS_OP(TYPE, ID_TYPE, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t id_numel, \
    ID_TYPE* ids, \
    TYPE *inp, \
    TYPE *out, \
    const size_t left_size, \
    const size_t dim_size, \
    const size_t right_size) \
{ \
    index_select_kernel<ID_TYPE, TYPE>(id_numel, ids, inp, out, left_size, dim_size, right_size); \
} \

IS_OP(__bf16, int64_t, is_i64_bf16)
IS_OP(__bf16, uint32_t, is_u32_bf16)
IS_OP(__bf16, uint8_t, is_u8_bf16)

IS_OP(__fp16, int64_t, is_i64_f16)
IS_OP(__fp16, uint32_t, is_u32_f16)
IS_OP(__fp16, uint8_t, is_u8_f16)

IS_OP(float, int64_t, is_i64_f32)
IS_OP(double, int64_t, is_i64_f64)
IS_OP(uint8_t, int64_t, is_i64_u8)
IS_OP(uint32_t, int64_t, is_i64_u32)
IS_OP(int64_t, int64_t, is_i64_i64)

IS_OP(float, uint32_t, is_u32_f32)
IS_OP(double, uint32_t, is_u32_f64)
IS_OP(uint8_t, uint32_t, is_u32_u8)
IS_OP(int64_t, uint32_t, is_u32_i64)
IS_OP(uint32_t, uint32_t, is_u32_u32)

IS_OP(float, uint8_t, is_u8_f32)
IS_OP(double, uint8_t, is_u8_f64)
IS_OP(uint8_t, uint8_t, is_u8_u8)
IS_OP(uint32_t, uint8_t, is_u8_u32)
IS_OP(int64_t, uint8_t, is_u8_i64)

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

    // where_u8_f32<<<1, 12>>>(dev_ids, dev_in1, dev_in2, dev_out, N);

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
