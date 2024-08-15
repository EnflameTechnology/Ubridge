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
#define SHARE_REMAIN_BUFFER_SIZE 2 * 1024 * 1024
#define TILE_SIZE 1024 * 16

#define REDUCE_OP_KENEL(T, TO, FN_NAME, FUNC) \
__device__ __forceinline__ void FN_NAME( \
    T *in, \
    TO *out, \
    TO *out_share, \
    char* raw_cache, \
    const size_t element_num, const size_t reduce_dim_size, const bool use_shared_output) \
{ \
    int thread_id = GetThreadIdx();\
    int MAX_THREADS = GetThreadNum();\
    const int N1 = element_num / reduce_dim_size;\
    const int TILESIZE = reduce_dim_size < TILE_SIZE ? reduce_dim_size : TILE_SIZE;\
    const int N2 = reduce_dim_size % TILESIZE ==0 ? reduce_dim_size / TILESIZE : reduce_dim_size / TILESIZE + 1;\
    const int N = N1 * N2; \
    __local__ __valigned__ T in_buffer[TILE_SIZE];\
    __local__ __valigned__ TO out_buffer[TILE_SIZE];\
    __shared__ __valigned__ TO share_output[TILE_SIZE];\
    tops_dte_ctx_t ctxs_in;\
    tops_dte_ctx_t ctxs_out;\
    ctxs_in.init();\
    ctxs_out.init();\
    tops::dte_scope s_in0(ctxs_in);\
    tops::dte_scope s_out0(ctxs_out);\
    bool cachable = element_num * sizeof(T) < SHARE_BUFFER_SIZE - SHARE_REMAIN_BUFFER_SIZE;\
    T* sharedBuffer = reinterpret_cast<T*>(raw_cache);\
    if (GetThreadIdxInBlock() == 0 && cachable) {\
      tops::memcpy(ctxs_in, tops::mdspan(tops::Shared, sharedBuffer, element_num), tops::mdspan(tops::Global, in, element_num));\
    }\
    __syncthreads();\
    tops::mdspan hbm_in(tops::Global, in, N1, N2, TILESIZE);\
    tops::mdspan shared_in(tops::Shared, sharedBuffer, N1, N2, TILESIZE);\
    tops::mdspan hbm_out(tops::Global, out, N1);\
    tops::mdspan l2_out(tops::Shared, out_share, N1);\
    tops::mdspan share_out(tops::Shared, share_output, N1, N2);\
    int THREAD_STEP = 1;\
    int thread_step = 1;\
    if (N > MAX_THREADS) {\
      THREAD_STEP = N / MAX_THREADS;\
      thread_step = THREAD_STEP;\
      if (N % MAX_THREADS != 0) {\
        if (thread_id == MAX_THREADS - 1) {\
          thread_step += N % MAX_THREADS; \
        }\
      }\
    }\
    for (int i = 0; i < thread_step; i++) {\
      int idx = thread_id * THREAD_STEP + i;\
      if (idx >= N) break;\
      int batch_idx = idx / N2;\
      int section_idx = idx % N2;\
      if (batch_idx < N1) {\
          int bufsize = (section_idx * TILESIZE + TILESIZE < reduce_dim_size) ? TILESIZE : reduce_dim_size - section_idx * TILESIZE;\
          tops::mdspan thread_in0(tops::Private, in_buffer, 1, 1, bufsize);\
          ctxs_in.config_slice(thread_in0, cachable ? shared_in : hbm_in, {batch_idx, section_idx, 0});\
          ctxs_in.trigger_and_wait();\
          FUNC<T>(out_buffer, in_buffer, bufsize);\
          share_output[idx] = out_buffer[0];\
      }\
    }\
    __syncthreads();\
    TO* pBuffer = out_buffer;\
    if (thread_id == 0) {\
      for (int batch_idx=0; batch_idx<N1; batch_idx++) {\
          tops::mdspan thread_in0(tops::Private, in_buffer, 1, N2);\
          ctxs_in.config_slice(thread_in0, share_out, {batch_idx, 0});\
          ctxs_in.trigger_and_wait();\
          FUNC<T>(pBuffer, in_buffer, N2);\
          pBuffer += 1;\
      }\
      tops::mdspan thread_out0(tops::Private, out_buffer, N1);\
      tops::memcpy(ctxs_out, use_shared_output ? l2_out : hbm_out, thread_out0);\
    }\
}\


REDUCE_OP_KENEL(float, float, fast_sum_f32_kernel, atomic_reduce_sum)
REDUCE_OP_KENEL(tops::half, tops::half, fast_sum_f16_kernel, atomic_reduce_sum)
REDUCE_OP_KENEL(int8_t, int8_t, fast_sum_i8_kernel, atomic_reduce_sum)
REDUCE_OP_KENEL(tops::bfloat, tops::bfloat, fast_sum_bf16_kernel, atomic_reduce_sum)

REDUCE_OP_KENEL(float, float, fast_min_f32_kernel, atomic_reduce_min)
REDUCE_OP_KENEL(tops::half, tops::half, fast_min_f16_kernel, atomic_reduce_min)
REDUCE_OP_KENEL(int8_t, int8_t, fast_min_i8_kernel, atomic_reduce_min)
REDUCE_OP_KENEL(tops::bfloat, tops::bfloat, fast_min_bf16_kernel, atomic_reduce_min)


REDUCE_OP_KENEL(float, float, fast_max_f32_kernel, atomic_reduce_max)
REDUCE_OP_KENEL(tops::half, tops::half, fast_max_f16_kernel, atomic_reduce_max)
REDUCE_OP_KENEL(int8_t, int8_t, fast_max_i8_kernel, atomic_reduce_max)
REDUCE_OP_KENEL(tops::bfloat, tops::bfloat, fast_max_bf16_kernel, atomic_reduce_max)

REDUCE_OP_KENEL(float, u_int32_t, fast_argmax_f32_kernel, atomic_reduce_argmax)
REDUCE_OP_KENEL(tops::half, u_int32_t, fast_argmax_f16_kernel, atomic_reduce_argmax)
REDUCE_OP_KENEL(int8_t, u_int32_t, fast_argmax_i8_kernel, atomic_reduce_argmax)
REDUCE_OP_KENEL(tops::bfloat, u_int32_t, fast_argmax_bf16_kernel, atomic_reduce_argmax)

REDUCE_OP_KENEL(float, u_int32_t, fast_argmin_f32_kernel, atomic_reduce_argmin)
REDUCE_OP_KENEL(tops::half, u_int32_t, fast_argmin_f16_kernel, atomic_reduce_argmin)
REDUCE_OP_KENEL(int8_t, u_int32_t, fast_argmin_i8_kernel, atomic_reduce_argmin)
REDUCE_OP_KENEL(tops::bfloat, u_int32_t, fast_argmin_bf16_kernel, atomic_reduce_argmin)


#define REDUCE_OP(T, TO, FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    T *in, \
    TO *out, \
    const size_t element_num, const size_t reduce_dim_size) \
{\
    __shared__ char raw_cache[SHARE_BUFFER_SIZE - SHARE_REMAIN_BUFFER_SIZE];\
    __shared__ TO out_share[TILE_SIZE];\
    FN_NAME##_kernel(in, out, out_share, raw_cache, element_num, reduce_dim_size, false);\
}\

REDUCE_OP(float, float, fast_sum_f32, atomic_reduce_sum)
REDUCE_OP(tops::half, tops::half, fast_sum_f16, atomic_reduce_sum)
REDUCE_OP(int8_t, int8_t, fast_sum_i8, atomic_reduce_sum)
REDUCE_OP(tops::bfloat, tops::bfloat, fast_sum_bf16, atomic_reduce_sum)

REDUCE_OP(float, float, fast_min_f32, atomic_reduce_min)
REDUCE_OP(tops::half, tops::half, fast_min_f16, atomic_reduce_min)
REDUCE_OP(int8_t, int8_t, fast_min_i8, atomic_reduce_min)
REDUCE_OP(tops::bfloat, tops::bfloat, fast_min_bf16, atomic_reduce_min)


REDUCE_OP(float, float, fast_max_f32, atomic_reduce_max)
REDUCE_OP(tops::half, tops::half, fast_max_f16, atomic_reduce_max)
REDUCE_OP(int8_t, int8_t, fast_max_i8, atomic_reduce_max)
REDUCE_OP(tops::bfloat, tops::bfloat, fast_max_bf16, atomic_reduce_max)

REDUCE_OP(float, u_int32_t, fast_argmax_f32, atomic_reduce_argmax)
REDUCE_OP(tops::half, u_int32_t, fast_argmax_f16, atomic_reduce_argmax)
REDUCE_OP(int8_t, u_int32_t, fast_argmax_i8, atomic_reduce_argmax)
REDUCE_OP(tops::bfloat, u_int32_t, fast_argmax_bf16, atomic_reduce_argmax)

REDUCE_OP(float, u_int32_t, fast_argmin_f32, atomic_reduce_argmin)
REDUCE_OP(tops::half, u_int32_t, fast_argmin_f16, atomic_reduce_argmin)
REDUCE_OP(int8_t, u_int32_t, fast_argmin_i8, atomic_reduce_argmin)
REDUCE_OP(tops::bfloat, u_int32_t, fast_argmin_bf16, atomic_reduce_argmin)


#define SOFTMAX_OP(T, TT, FN_NAME, MAX_KERNEL) \
extern "C" __global__ void FN_NAME(T *input, T* output, int batch,\
    int chunks, int last_dim_size) {\
    __local__ __valigned__ T bufferTmp[TILE_SIZE];\
    __local__ __valigned__ T buffer1[TILE_SIZE];\
    __local__ __valigned__ float buffer2[TILE_SIZE];\
    __local__ __valigned__ float buffer3[TILE_SIZE];\
    __shared__ T share_max_out[TILE_SIZE];\
    __shared__ T share_exp_sum_out[TILE_SIZE];\
    __shared__ char raw_cache[SHARE_BUFFER_SIZE - SHARE_REMAIN_BUFFER_SIZE];\
    tops_dte_ctx_t ctx;\
    tops::dte_scope s(ctx);\
    int element_num = batch * chunks * last_dim_size;\
    int stride = chunks * last_dim_size;\
    int thread_id = GetThreadIdx();\
    int MAX_THREADS = GetThreadNum();\
    int N = batch * chunks;\
    int THREAD_STEP = 1;\
    int thread_step = 1;\
    if (N > MAX_THREADS) {\
      THREAD_STEP = N / MAX_THREADS;\
      thread_step = THREAD_STEP;\
      if (N % MAX_THREADS != 0) {\
        if (thread_id == MAX_THREADS - 1) {\
          thread_step += N % MAX_THREADS; \
        }\
      }\
    }\
    MAX_KERNEL(reinterpret_cast<TT*>(input), reinterpret_cast<TT*>(output), reinterpret_cast<TT*>(share_max_out), reinterpret_cast<char*>(raw_cache), element_num, last_dim_size, true);\
    __syncthreads();\
    bool output_cachable = element_num * sizeof(T) < SHARE_BUFFER_SIZE - SHARE_REMAIN_BUFFER_SIZE;\
    T* outputShare = reinterpret_cast<T*>(raw_cache);\
    for (int i = 0; i < thread_step; i++) {\
      int idx = thread_id * THREAD_STEP + i;\
      if (idx >= N) break;\
      int batch_idx = idx / chunks;\
      int offset = idx % chunks;\
      T* input_cur = input + batch_idx * stride;\
      T* output_cur = output + batch_idx * stride;\
      T* output_cur_share = outputShare + batch_idx * stride;\
      float exp_sum = 0.0;\
      for (int k = 0; k < last_dim_size; k+=TILE_SIZE) {\
          int bufsize = (k + TILE_SIZE < last_dim_size) ? TILE_SIZE : last_dim_size - k;\
          tops::mdspan l1_input(tops::Private, bufferTmp, bufsize);\
          tops::mdspan hbm_input(tops::Global, input_cur + offset * last_dim_size + k, bufsize);\
          tops::memcpy(ctx, l1_input, hbm_input);\
          convert<float, T>(reinterpret_cast<float*>(buffer2), reinterpret_cast<T*>(bufferTmp), bufsize);\
          sub(reinterpret_cast<float*>(buffer3), reinterpret_cast<float*>(buffer2), (float)share_max_out[idx], bufsize);\
          exp(reinterpret_cast<float*>(buffer2), reinterpret_cast<float*>(buffer3), bufsize);\
          atomic_reduce_sum(reinterpret_cast<float*>(buffer3), reinterpret_cast<float*>(buffer2), bufsize);\
          tops::mdspan hbm_output(tops::Global, output_cur + offset * last_dim_size + k, bufsize);\
          tops::mdspan l2_output(tops::Shared, output_cur_share + offset * last_dim_size + k, bufsize);\
          convert<T, float>(reinterpret_cast<T*>(bufferTmp), reinterpret_cast<float*>(buffer2), bufsize);\
          tops::mdspan l1_output(tops::Private, bufferTmp, bufsize);\
          tops::memcpy(ctx, output_cachable? l2_output : hbm_output, l1_output);\
          exp_sum += buffer3[0];\
      }\
      for (int k = 0; k < last_dim_size; k+=TILE_SIZE) {\
          int bufsize = (k + TILE_SIZE < last_dim_size) ? TILE_SIZE : last_dim_size - k;\
          tops::mdspan l1_input_output(tops::Private, bufferTmp, bufsize);\
          tops::mdspan hbm_input_output(tops::Global, output_cur + offset * last_dim_size + k, bufsize);\
          tops::mdspan l2_input_output(tops::Shared, output_cur_share + offset * last_dim_size + k, bufsize);\
          tops::memcpy(ctx, l1_input_output, output_cachable? l2_input_output : hbm_input_output);\
          convert<float, T>(reinterpret_cast<float*>(buffer2), reinterpret_cast<T*>(bufferTmp), bufsize);\
          div(reinterpret_cast<float*>(buffer3), reinterpret_cast<float*>(buffer2), exp_sum, bufsize);\
          convert<T, float>(reinterpret_cast<T*>(bufferTmp), reinterpret_cast<float*>(buffer3), bufsize);\
          tops::memcpy(ctx, hbm_input_output, l1_input_output);\
      }\
    }\
}\

extern "C" __global__ void softmax_f32(float *input, float* output, int batch,
    int chunks, int last_dim_size) {
    __local__ __valigned__ float bufferTmp[TILE_SIZE];
    __local__ __valigned__ float buffer1[TILE_SIZE];
    __local__ __valigned__ float buffer2[TILE_SIZE];
    __local__ __valigned__ float buffer3[TILE_SIZE];
    __shared__ float share_max_out[TILE_SIZE];
    __shared__ char raw_cache[SHARE_BUFFER_SIZE - SHARE_REMAIN_BUFFER_SIZE];
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx);
    int element_num = batch * chunks * last_dim_size;
    int stride = chunks * last_dim_size;
    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNum();
    int N = batch * chunks;
    int THREAD_STEP = 1;
    int thread_step = 1;
    if (N > MAX_THREADS) {
      THREAD_STEP = N / MAX_THREADS;
      thread_step = THREAD_STEP;
      if (N % MAX_THREADS != 0) {
        if (thread_id == MAX_THREADS - 1) {
          thread_step += N % MAX_THREADS; 
        }
      }
    }
    fast_max_f32_kernel(reinterpret_cast<float*>(input), reinterpret_cast<float*>(output), reinterpret_cast<float*>(share_max_out), reinterpret_cast<char*>(raw_cache), element_num, last_dim_size, true);
    __syncthreads();

    bool output_cachable = element_num * sizeof(float) < SHARE_BUFFER_SIZE - SHARE_REMAIN_BUFFER_SIZE;
    float* outputShare = reinterpret_cast<float*>(raw_cache);

    // if (thread_id==0)
    //   printf("max value for softmax %.5f [%d, %d, %d]\n", (float)share_max_out[0], batch, chunks, last_dim_size);
    for (int i = 0; i < thread_step; i++) {
      int idx = thread_id * THREAD_STEP + i;
      int batch_idx = idx / chunks;
      int offset = idx % chunks;
      float* input_cur = input + batch_idx * stride;
      float* output_cur = output + batch_idx * stride;
      float* output_cur_share = outputShare + batch_idx * stride;
      float exp_sum = 0.0;
      if (idx >= N) break;
      for (int k = 0; k < last_dim_size; k+=TILE_SIZE) {
          int bufsize = (k + TILE_SIZE < last_dim_size) ? TILE_SIZE : last_dim_size - k;
          tops::mdspan l1_input(tops::Private, bufferTmp, bufsize);
          tops::mdspan hbm_input(tops::Global, input_cur + offset * last_dim_size + k, bufsize);
          tops::memcpy(ctx, l1_input, hbm_input);
          sub(reinterpret_cast<float*>(buffer1), reinterpret_cast<float*>(bufferTmp), (float)share_max_out[idx], bufsize);
          exp(reinterpret_cast<float*>(buffer2), reinterpret_cast<float*>(buffer1), bufsize);
          atomic_reduce_sum(reinterpret_cast<float*>(buffer3), reinterpret_cast<float*>(buffer2), bufsize);
          tops::mdspan hbm_output(tops::Global, output_cur + offset * last_dim_size + k, bufsize);
          tops::mdspan l2_output(tops::Shared, output_cur_share + offset * last_dim_size + k, bufsize);
          tops::mdspan l1_output(tops::Private, buffer2, bufsize);
          tops::memcpy(ctx, output_cachable? l2_output : hbm_output, l1_output);
          exp_sum += buffer3[0];
      }
      for (int k = 0; k < last_dim_size; k+=TILE_SIZE) {
          int bufsize = (k + TILE_SIZE < last_dim_size) ? TILE_SIZE : last_dim_size - k;
          tops::mdspan l1_input(tops::Private, bufferTmp, bufsize);
          tops::mdspan l1_output(tops::Private, buffer1, bufsize);
          tops::mdspan hbm_input_output(tops::Global, output_cur + offset * last_dim_size + k, bufsize);
          tops::mdspan l2_input_output(tops::Shared, output_cur_share + offset * last_dim_size + k, bufsize);
          tops::memcpy(ctx, l1_input, output_cachable? l2_input_output : hbm_input_output);
          div(reinterpret_cast<float*>(buffer1), reinterpret_cast<float*>(bufferTmp), exp_sum, bufsize);
          tops::memcpy(ctx, hbm_input_output, l1_output);
      }
    }
}

    // printf("N %d, MAX_THREADS %d, THREAD_STEP %d, thread_step %d, chunks %lu, last_dim_size %lu \n", 
    //     N, MAX_THREADS, THREAD_STEP, thread_step, chunks, last_dim_size);
    //yi = exp(xi - max)/(sum(exp(xi - max))

      // if (results_sum - 1.0 > 0.01 || 1.0 - results_sum > 0.01)
      //   printf("Invalid softmax result %.5f in batch_idx %d for [%d, %d, %d]\n\n", results_sum, batch_idx, batch, chunks, last_dim_size);

// SOFTMAX_OP(float, float, softmax_f32, fast_max_f32_kernel)
SOFTMAX_OP(__fp16, tops::half, softmax_f16, fast_max_f16_kernel)
SOFTMAX_OP(__bf16, tops::bfloat, softmax_bf16, fast_max_bf16_kernel)

// extern "C" __global__ void softmax_f64(double *input, double *output,
//     size_t elements) {
//       softmax_kernel<double>(input, output, elements);
// }

// extern "C" __global__ void softmax_u8(uint8_t *input, uint8_t* output, 
//     size_t elements) {
//       softmax_kernel<uint8_t>(input, output, elements);
// }



#define TILE_SIZE_NORM 1024 * 32
template <typename T>
__device__ void layernorm_kernel(T *input, T* output, T* weight, T* bias, int batch,
    int chunks, int last_dims_size, float epsilon, int remove_mean, int affine) {
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx);
    __local__ __valigned__ float buffer1[TILE_SIZE_NORM];
    __local__ __valigned__ float buffer2[TILE_SIZE_NORM];
    __local__ __valigned__ float buffer3[TILE_SIZE_NORM];
    __local__ __valigned__ T bufferTmp[TILE_SIZE_NORM];
    __local__ __valigned__ T bufferWeight[TILE_SIZE_NORM];
    __local__ __valigned__ T bufferBias[TILE_SIZE_NORM];
    __local__ __valigned__ T bufferOut[TILE_SIZE_NORM];
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
    int stride = chunks * last_dims_size;
    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNum();
    int N = batch * chunks;
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
      int idx = thread_id * THREAD_STEP + i;
      if (idx >= N) { break; }

      int batch_idx = idx / chunks;
      int offset = idx % chunks;
      T* input_cur = input + batch_idx * stride;
      T* output_cur = output + batch_idx * stride;

      tops::mdspan l1_input(tops::Private, bufferTmp, last_dims_size);
      tops::mdspan hbm_input(tops::Global, input_cur + offset * last_dims_size, last_dims_size);
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
      tops::mdspan hbm_output(tops::Global, output_cur + offset * last_dims_size, last_dims_size);
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

extern "C" __global__ void layernorm_f16(__fp16 *input, __fp16 *output, __fp16* weight, __fp16* bias, int batch,
    int chunks, int last_dim_size, float epsilon, int remove_mean, int affine) {
      layernorm_kernel<__fp16>(input, output, weight, bias, batch, chunks, last_dim_size, epsilon, remove_mean, affine);
}

extern "C" __global__ void layernorm_bf16(__bf16 *input, __bf16 *output, __bf16* weight, __bf16* bias, int batch,
    int chunks, int last_dim_size, float epsilon, int remove_mean, int affine) {
      layernorm_kernel<__bf16>(input, output, weight, bias, batch, chunks, last_dim_size, epsilon, remove_mean, affine);
}

extern "C" __global__ void layernorm_f32(float *input, float *output, float* weight, float* bias, int batch,
    int chunks, int last_dim_size, float epsilon, int remove_mean, int affine) {
      layernorm_kernel<float>(input, output, weight, bias, batch, chunks, last_dim_size, epsilon, remove_mean, affine);
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
