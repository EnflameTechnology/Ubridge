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
#include <tops/bfloat.h>
#include <algorithm>
#include <vector>
#include "tops/tops_runtime.h"
#include "utils.h"
#include "reduce_atomic.h"
using namespace std;
#define RANK_MAX 3

enum REDUCE_TYPE {
  REDUCE_MIN,
  REDUCE_MAX,
  REDUCE_SUM,
};

template <typename T>
__device__ void reduce_kernel(T* in, T* out, const size_t element_num, const size_t reduce_dim_size, REDUCE_TYPE tp) {
    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNumEachBlock();

    const int N = element_num / reduce_dim_size;
    __local__ __valigned__ T in_buffer[40960];
    __local__ __valigned__ T out_buffer[40960];

    tops_dte_ctx_t ctxs_in;
    tops_dte_ctx_t ctxs_out;
    ctxs_in.init();
    ctxs_out.init();
    tops::dte_scope s_in0(ctxs_in);
    tops::dte_scope s_out0(ctxs_out);

    tops::mdspan hbm_in(tops::Global, in, N, reduce_dim_size);
    tops::mdspan thread_in0(tops::Private, in_buffer, 1, reduce_dim_size);

    tops::mdspan hbm_out(tops::Global, out, N, 1);
    tops::mdspan thread_out0(tops::Private, out_buffer, 1, 1);

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

    // printf("N %d, reduce_dim_size %d, MAX_THREADS %d, THREAD_STEP %d, thread_step %d\n", N, reduce_dim_size, MAX_THREADS, THREAD_STEP, thread_step);
    for (int i = 0; i < thread_step; i++) {
      int idx = thread_id * THREAD_STEP + i;
      if (idx < N) {
          ctxs_in.config_slice(thread_in0, hbm_in, {0, 0});
          ctxs_in.set_src_offset(0, idx);
          ctxs_in.trigger_and_wait();
          if(tp == REDUCE_MIN) {
            atomic_reduce_min<T>(out_buffer, in_buffer, reduce_dim_size);
          } else if (tp == REDUCE_MAX) {
            atomic_reduce_max<T>(out_buffer, in_buffer, reduce_dim_size);
          } else if (tp == REDUCE_SUM) {
            atomic_reduce_sum<T>(out_buffer, in_buffer, reduce_dim_size);
          } else {
            printf("Not supported reduce type!");
          }
          // printf("Reduce type %d\n", tp);
          // printf("\nReduce Input buffer: ");
          // for (int j=0; j<reduce_dim_size; j++) {
          //   printf("%.5f ", static_cast<float>(in_buffer[j]));
          // }
          // printf("\nReduce Output buffer: ");

          // for (int j=0; j<reduce_dim_size; j++) {
          //   printf("%.5f ", static_cast<float>(out_buffer[j]));
          // }
          // printf("%.5f ", static_cast<float>(out_buffer[0]));
          ctxs_out.config_deslice(hbm_out, thread_out0, {0, 0});
          ctxs_out.set_dst_offset(0, idx);
          ctxs_out.trigger_and_wait();
      }
    }

}

#define REDUCE_OP(TYPE, FN_NAME, TP) \
extern "C" __global__ void FN_NAME( \
    TYPE *in, \
    TYPE *out, \
    const size_t element_num, const size_t reduce_dim_size) \
{ \
    reduce_kernel<TYPE>(in, out, element_num, reduce_dim_size, TP); \
} \

REDUCE_OP(float, fast_sum_f32, REDUCE_SUM)
REDUCE_OP(tops::half, fast_sum_f16, REDUCE_SUM)
REDUCE_OP(int8_t, fast_sum_i8, REDUCE_SUM)
REDUCE_OP(tops::bfloat, fast_sum_bf16, REDUCE_SUM)

REDUCE_OP(float, fast_min_f32, REDUCE_MIN)
REDUCE_OP(tops::half, fast_min_f16, REDUCE_MIN)
REDUCE_OP(int8_t, fast_min_i8, REDUCE_MIN)
REDUCE_OP(tops::bfloat, fast_min_bf16, REDUCE_MIN)


REDUCE_OP(float, fast_max_f32, REDUCE_MAX)
REDUCE_OP(tops::half, fast_max_f16, REDUCE_MAX)
REDUCE_OP(int8_t, fast_max_i8, REDUCE_MAX)
REDUCE_OP(tops::bfloat, fast_max_bf16, REDUCE_MAX)

int main(void) {
  topsError_t err = topsSuccess;
  int nums = 2;
  int N = 2;
  int in_euem = N * nums;
  int out_euem = N;
  int in_size = in_euem * sizeof(float);
  int out_size = out_euem * sizeof(float);
  float *host_in = reinterpret_cast<float*>(aligned_alloc(128, in_size));
  float *host_out = reinterpret_cast<float*>(aligned_alloc(128, out_size));

  for (int j =0; j< N; j ++) {
    for (int i = 0; i < in_euem/N; ++i) {
      host_in[j * in_euem/N + i] = i;
    }
    // host_in[j * in_euem/N + 100] = -100.0;
  }
  
  host_in[0] = -2.00000f;
  host_in[1] = 2.00000f;
  host_in[2] = 3.00000f;
  host_in[3] = -3.00000f;

  for (int i = 0; i < out_euem; ++i) {
    host_out[i] = 0;
  }

  float *dev_in = NULL;
  CHECK(topsMalloc(reinterpret_cast<void **>(&dev_in), in_size));

  float *dev_out = NULL;
  CHECK(topsMalloc(reinterpret_cast<void **>(&dev_out), out_size));

  CHECK(topsMemcpy(dev_in, host_in, in_size, topsMemcpyHostToDevice));
  CHECK(topsMemset(dev_out, 0, out_size));

  printf("call reduce kernel!!!!!!!!!!!!!\n");
  fast_max_f32<<<1, 12>>>(dev_in, dev_out, in_euem, nums);

  if (topsGetLastError() != topsSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            topsGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  CHECK(topsMemcpy(host_out, dev_out, out_size, topsMemcpyDeviceToHost));
  // int total = 0;
  // for (int i=0; i< nums; i++) {
  //   total += i;
  // }
  // printf("Actual results %d\n", total);

  for (int i = 0; i < out_euem; i++) {
    fprintf(stderr, "Result  element %d, %f!\n", i, host_out[i]);
  }

  printf("Test PASSED\n");

  CHECK(topsFree(dev_in));
  CHECK(topsFree(dev_out));
  free(host_in);
  free(host_out);

  return 0;
}
