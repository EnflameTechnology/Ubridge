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
 * @file    reduce.cpp
 * @brief
 *
 * @author  Guoqing Bao
 * @date    2023-12-12 to 2024-01-31
 * @version V0.1
 * @par     Copyright (c) Enflame Tech Company.
 * @par     History: add fused layernorm & softmax kernel, optimize reduce sum kernel
 * @par     Comments: gcu kernel for reduce operations, including reduce sum, reduce max, reduce min, softmax, layernorm.
 */
#include <stdio.h>
#include <tops.h>
#include <tops/half.h>
#include <tops/bfloat.h>
#include <algorithm>
#include <vector>
#include "tops/tops_runtime.h"
#include "utils/utils.h"
#include "utils/reduce_atomic.h"
#include <acore_op.h>
using namespace std;
#define RANK_MAX 3

enum REDUCE_TYPE {
  REDUCE_MIN,
  REDUCE_MAX,
  REDUCE_SUM,
};

template <typename T>
__forceinline__ __device__ void reduce_kernel(T* in, T* out, const size_t element_num, const size_t reduce_dim_size, REDUCE_TYPE tp) {
    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNum();

    const int N = element_num / reduce_dim_size;
    __local__ __valigned__ T in_buffer[41984];
    __local__ __valigned__ T out_buffer[41984];
    __shared__ char raw_cache[SHARE_BUFFER_SIZE];

    tops_dte_ctx_t ctxs_in;
    tops_dte_ctx_t ctxs_out;
    ctxs_in.init();
    ctxs_out.init();
    tops::dte_scope s_in0(ctxs_in);
    tops::dte_scope s_out0(ctxs_out);

    bool cachable = element_num * sizeof(T) < SHARE_BUFFER_SIZE;
    T* sharedBuffer = reinterpret_cast<T*>(raw_cache);

    if (GetThreadIdxInBlock() == 0 && cachable) {
      tops::memcpy(ctxs_in, tops::mdspan(tops::Shared, sharedBuffer, element_num), tops::mdspan(tops::Global, in, element_num));
    }
    __syncthreads();

    tops::mdspan hbm_in(tops::Global, in, N, reduce_dim_size);
    tops::mdspan shared_in(tops::Shared, sharedBuffer, N, reduce_dim_size);

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
          ctxs_in.config_slice(thread_in0, cachable ? shared_in : hbm_in, {0, 0});
          ctxs_in.set_src_offset(0, idx);
          ctxs_in.trigger_and_wait();
          if(tp == REDUCE_MIN) {
            atomic_reduce_min<T>(out_buffer, in_buffer, reduce_dim_size);
          } else if (tp == REDUCE_MAX) {
            atomic_reduce_max<T>(out_buffer, in_buffer, reduce_dim_size);
          } else if (tp == REDUCE_SUM) {
            atomic_reduce_sum<T>(out_buffer, in_buffer, reduce_dim_size);
          } 
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


#define TILE_SIZE 1024 * 16
template <typename T>
__device__ void softmax_kernel(T *input, T* output, 
    size_t chunks, size_t last_dim_size) {
    __local__ __valigned__ float buffer1[TILE_SIZE];
    __local__ __valigned__ float buffer2[TILE_SIZE];
    __local__ __valigned__ T bufferTmp[TILE_SIZE];
    __shared__ char raw_cache[SHARE_BUFFER_SIZE];

    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx);

    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNum();
    int N = chunks;

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

    bool cachable = chunks * last_dim_size * sizeof(T) < SHARE_BUFFER_SIZE;
    T* sharedBuffer = reinterpret_cast<T*>(raw_cache);

    if (GetThreadIdxInBlock() == 0 && cachable) {
      tops::memcpy(ctx, tops::mdspan(tops::Shared, sharedBuffer, chunks * last_dim_size), tops::mdspan(tops::Global, input, chunks * last_dim_size));
    }
    __syncthreads();

    // printf("N %d, MAX_THREADS %d, THREAD_STEP %d, thread_step %d, chunks %lu, last_dim_size %lu \n", 
    //     N, MAX_THREADS, THREAD_STEP, thread_step, chunks, last_dim_size);
    //yi = exp(xi - max)/(sum(exp(xi - max))
    for (int i = 0; i < thread_step; i++) {
      int offset = thread_id * THREAD_STEP + i;
      if (offset >= N) {
        break;
      }
      // printf("offset %lu\n", offset * last_dim_size);
      tops::mdspan l1_input(tops::Private, bufferTmp, last_dim_size);
      tops::mdspan hbm_input(tops::Global, input + offset * last_dim_size, last_dim_size);
      tops::mdspan shared_input(tops::Shared, sharedBuffer + offset * last_dim_size, last_dim_size);

      tops::memcpy(ctx, l1_input, cachable ? shared_input : hbm_input);
      convert<float, T>(reinterpret_cast<float*>(buffer1), reinterpret_cast<T*>(bufferTmp), last_dim_size);
      
      atomic_reduce_max(reinterpret_cast<float*>(buffer2), reinterpret_cast<float*>(buffer1), last_dim_size);
      
      float max_value = buffer2[0];
      sub(reinterpret_cast<float*>(buffer2), reinterpret_cast<float*>(buffer1), max_value, last_dim_size);
      exp(reinterpret_cast<float*>(buffer1), reinterpret_cast<float*>(buffer2), last_dim_size);
      atomic_reduce_sum(reinterpret_cast<float*>(buffer2), reinterpret_cast<float*>(buffer1), last_dim_size);
      float sum_exp = buffer2[0];
      tops::mdspan hbm_output(tops::Global, output + offset * last_dim_size, last_dim_size);
      div(reinterpret_cast<float*>(buffer2), reinterpret_cast<float*>(buffer1), sum_exp, last_dim_size);
      convert<T, float>(reinterpret_cast<T*>(bufferTmp), reinterpret_cast<float*>(buffer2), last_dim_size);
      tops::mdspan l1_output(tops::Private, bufferTmp, last_dim_size);
      tops::memcpy(ctx, hbm_output, l1_output);
    }
}

extern "C" __global__ void softmax_f16(__fp16 *input, __fp16 *output,
    size_t chunks, size_t last_dim_size) {
      softmax_kernel<__fp16>(input, output, chunks, last_dim_size);
}

extern "C" __global__ void softmax_bf16(__bf16 *input, __bf16 *output,
    size_t chunks, size_t last_dim_size) {
      softmax_kernel<__bf16>(input, output, chunks, last_dim_size);
}

extern "C" __global__ void softmax_f32(float *input, float *output,
    size_t chunks, size_t last_dim_size) {
      softmax_kernel<float>(input, output, chunks, last_dim_size);
}

// extern "C" __global__ void softmax_f64(double *input, double *output,
//     size_t elements) {
//       softmax_kernel<double>(input, output, elements);
// }

// extern "C" __global__ void softmax_u8(uint8_t *input, uint8_t* output, 
//     size_t elements) {
//       softmax_kernel<uint8_t>(input, output, elements);
// }



#define TILE_SIZE 1024 * 16
template <typename T>
__device__ void layernorm_kernel(T *input, T* output, T* weight, T* bias,
    size_t chunks, size_t last_dims_size, float epsilon, int remove_mean, int affine) {
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx);
    __local__ __valigned__ float buffer1[TILE_SIZE];
    __local__ __valigned__ float buffer2[TILE_SIZE];
    __local__ __valigned__ float buffer3[TILE_SIZE];
    __local__ __valigned__ T bufferTmp[TILE_SIZE];
    __local__ __valigned__ T bufferWeight[TILE_SIZE];
    __local__ __valigned__ T bufferBias[TILE_SIZE];
    __local__ __valigned__ T bufferOut[TILE_SIZE];
    __local__ __valigned__ float bufTmp[256];
    __local__ __valigned__ float bufTmp1[256];
    __shared__ char raw_cache[SHARE_BUFFER_SIZE];
    tops::mdspan l1_weight(tops::Private, bufferWeight, last_dims_size);
    tops::mdspan l1_bias(tops::Private, bufferBias, last_dims_size);
    tops::mdspan hbm_weight(tops::Global, weight, last_dims_size);
    tops::memcpy(ctx, l1_weight, hbm_weight);
    if (affine >0) {
      tops::mdspan hbm_bias(tops::Global, bias, last_dims_size);
      tops::memcpy(ctx, l1_bias, hbm_bias);
    }

    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNum();
    int N = chunks;
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

    //mean = sum(xi)/N
    //varience = sum((xi - mean)^2)/N
    //yi = (xi - mean) / sqrt(varience + epsilon)
    //yi = weight * yi + bias

    for (int i = 0; i < thread_step; i++) {
      int offset = thread_id * THREAD_STEP + i;
      if (offset >= N) { break; }
      tops::mdspan l1_input(tops::Private, bufferTmp, last_dims_size);
      tops::mdspan hbm_input(tops::Global, input + offset * last_dims_size, last_dims_size);
      tops::memcpy(ctx, l1_input, hbm_input);
      if (remove_mean == 0) { //rmsnorm
        convert<float, T>(reinterpret_cast<float*>(buffer2), reinterpret_cast<T*>(bufferTmp), last_dims_size);
      } else { //layernorm
        convert<float, T>(reinterpret_cast<float*>(buffer1), reinterpret_cast<T*>(bufferTmp), last_dims_size);
        atomic_reduce_sum(reinterpret_cast<float*>(buffer2), reinterpret_cast<float*>(buffer1), last_dims_size);
        float mean_value = buffer2[0] / last_dims_size;
        // buffer2 -> xi - mean
        sub(reinterpret_cast<float*>(buffer2), reinterpret_cast<float*>(buffer1), mean_value, last_dims_size);
      }
      // buffer1 -> (xi - mean)^2
      mul(reinterpret_cast<float*>(buffer1), reinterpret_cast<float*>(buffer2), reinterpret_cast<float*>(buffer2), last_dims_size);
      atomic_reduce_sum(reinterpret_cast<float*>(buffer3), reinterpret_cast<float*>(buffer1), last_dims_size);
      float var_value = buffer3[0] / last_dims_size; //varience
      bufTmp[0] = var_value + epsilon;
      sqrt(reinterpret_cast<float*>(bufTmp1), reinterpret_cast<float*>(bufTmp), 32); //sqrt(varience + epsilon)
      // buffer1 -> (xi - mean) / sqrt(varience + epsilon)
      div(reinterpret_cast<float*>(buffer1), reinterpret_cast<float*>(buffer2), bufTmp1[0], last_dims_size);
      convert<T, float>(reinterpret_cast<T*>(bufferOut), reinterpret_cast<float*>(buffer1), last_dims_size);
      tops::mdspan hbm_output(tops::Global, output + offset * last_dims_size, last_dims_size);
      mul(reinterpret_cast<T*>(bufferTmp), reinterpret_cast<T*>(bufferOut), reinterpret_cast<T*>(bufferWeight), last_dims_size); 
      if (affine > 0) { // + bias
        add(reinterpret_cast<T*>(bufferOut), reinterpret_cast<T*>(bufferTmp), reinterpret_cast<T*>(bufferBias), last_dims_size);
        tops::mdspan l1_output(tops::Private, bufferOut, last_dims_size);
        tops::memcpy(ctx, hbm_output, l1_output);
      } else {
        tops::mdspan l1_output(tops::Private, bufferTmp, last_dims_size);
        tops::memcpy(ctx, hbm_output, l1_output);
      }
    }
}

extern "C" __global__ void layernorm_f16(__fp16 *input, __fp16 *output, __fp16* weight, __fp16* bias,
    size_t chunks, size_t last_dim_size, float epsilon, int remove_mean, int affine) {
      layernorm_kernel<__fp16>(input, output, weight, bias, chunks, last_dim_size, epsilon, remove_mean, affine);
}

extern "C" __global__ void layernorm_bf16(__bf16 *input, __bf16 *output, __bf16* weight, __bf16* bias,
    size_t chunks, size_t last_dim_size, float epsilon, int remove_mean, int affine) {
      layernorm_kernel<__bf16>(input, output, weight, bias, chunks, last_dim_size, epsilon, remove_mean, affine);
}

extern "C" __global__ void layernorm_f32(float *input, float *output, float* weight, float* bias,
    size_t chunks, size_t last_dim_size, float epsilon, int remove_mean, int affine) {
      layernorm_kernel<float>(input, output, weight, bias, chunks, last_dim_size, epsilon, remove_mean, affine);
}

int main(void) {
#ifdef KERNEL_TEST
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
#endif
  return 0;
}
