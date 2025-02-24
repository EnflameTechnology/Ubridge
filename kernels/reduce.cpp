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
#define TILE_SIZE 1024 * 32
#define REDUCE_TILE_SIZE 1024 * 256

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
    const int N = element_num / reduce_dim_size;\
    __local__ __valigned__ T in_buffer[REDUCE_TILE_SIZE];\
    __local__ __valigned__ TO out_buffer[1024];\
    tops_dte_ctx_t ctxs_in;\
    tops_dte_ctx_t ctxs_out;\
    ctxs_in.init();\
    ctxs_out.init();\
    tops::dte_scope s_in0(ctxs_in);\
    tops::dte_scope s_out0(ctxs_out);\
    bool use_l2 = reduce_dim_size > REDUCE_TILE_SIZE && element_num * sizeof(T) < SHARE_BUFFER_SIZE - SHARE_REMAIN_BUFFER_SIZE;\
    T* sharedBuffer = reinterpret_cast<T*>(raw_cache);\
    int THREAD_STEP = 1;\
    int thread_step = 1;\
    GetThreadStep(N, thread_step, THREAD_STEP);\
    for (int i = 0; i < thread_step; i++) {\
      int idx = thread_id * THREAD_STEP + i;\
      if (idx >= N) break;\
      tops::mdspan thread_in0(tops::Private, in_buffer, reduce_dim_size);\
      tops::mdspan hbm_in(tops::Global, in + idx * reduce_dim_size, reduce_dim_size);\
      tops::mdspan share_in(tops::Shared, sharedBuffer + idx * reduce_dim_size, reduce_dim_size);\
      tops::memcpy(ctxs_in, use_l2? share_in : thread_in0, hbm_in);\
      FUNC<T>(out_buffer, use_l2? sharedBuffer + idx * reduce_dim_size : in_buffer, reduce_dim_size);\
      out_share[idx] = out_buffer[0];\
      if (!use_shared_output) {\
        tops::mdspan thread_out0(tops::Private, out_buffer, 1);\
        tops::mdspan hbm_out(tops::Global, out + idx, 1);\
        tops::memcpy(ctxs_out, hbm_out, thread_out0);\
      }\
    }\
    __syncthreads();\
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
    extern __shared__ char raw_cache[];\
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
    extern __shared__ char raw_cache[];\
    tops_dte_ctx_t ctx;\
    tops::dte_scope s(ctx);\
    int element_num = batch * chunks * last_dim_size;\
    int stride = chunks * last_dim_size;\
    int thread_id = GetThreadIdx();\
    int MAX_THREADS = GetThreadNum();\
    int N = batch * chunks;\
    int THREAD_STEP = 1;\
    int thread_step = 1;\
    GetThreadStep(N, thread_step, THREAD_STEP);\
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
          int aligned_bufsize = ALIGN_UP(bufsize, SIP_VECTOR_LENGTH);\
          tops::mdspan l1_input(tops::Private, bufferTmp, bufsize);\
          tops::mdspan hbm_input(tops::Global, input_cur + offset * last_dim_size + k, bufsize);\
          tops::memcpy(ctx, l1_input, hbm_input);\
          sub(reinterpret_cast<T*>(buffer1), reinterpret_cast<T*>(bufferTmp), (T)share_max_out[idx], aligned_bufsize);\
          exp(reinterpret_cast<T*>(bufferTmp), reinterpret_cast<T*>(buffer1), aligned_bufsize);\
          convert<float, T>(reinterpret_cast<float*>(buffer2), reinterpret_cast<T*>(bufferTmp), bufsize);\
          atomic_reduce_sum(reinterpret_cast<float*>(buffer3), reinterpret_cast<float*>(buffer2), bufsize);\
          tops::mdspan hbm_output(tops::Global, output_cur + offset * last_dim_size + k, bufsize);\
          tops::mdspan l2_output(tops::Shared, output_cur_share + offset * last_dim_size + k, bufsize);\
          tops::mdspan l1_output(tops::Private, bufferTmp, bufsize);\
          tops::memcpy(ctx, output_cachable? l2_output : hbm_output, l1_output);\
          exp_sum += buffer3[0];\
      }\
      for (int k = 0; k < last_dim_size; k+=TILE_SIZE) {\
          int bufsize = (k + TILE_SIZE < last_dim_size) ? TILE_SIZE : last_dim_size - k;\
          int aligned_bufsize = ALIGN_UP(bufsize, SIP_VECTOR_LENGTH);\
          tops::mdspan l1_input_output(tops::Private, bufferTmp, bufsize);\
          tops::mdspan hbm_input_output(tops::Global, output_cur + offset * last_dim_size + k, bufsize);\
          tops::mdspan l2_input_output(tops::Shared, output_cur_share + offset * last_dim_size + k, bufsize);\
          tops::memcpy(ctx, l1_input_output, output_cachable? l2_input_output : hbm_input_output);\
          convert<float, T>(reinterpret_cast<float*>(buffer2), reinterpret_cast<T*>(bufferTmp), bufsize);\
          div(reinterpret_cast<float*>(buffer3), reinterpret_cast<float*>(buffer2), exp_sum, aligned_bufsize);\
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
    extern __shared__ char raw_cache[];
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx);
    int element_num = batch * chunks * last_dim_size;
    int stride = chunks * last_dim_size;
    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNum();
    int N = batch * chunks;
    int THREAD_STEP = 1;
    int thread_step = 1;
    GetThreadStep(N, thread_step, THREAD_STEP);
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
          int aligned_bufsize = ALIGN_UP(bufsize, SIP_VECTOR_LENGTH);
          sub(reinterpret_cast<float*>(buffer1), reinterpret_cast<float*>(bufferTmp), (float)share_max_out[idx], aligned_bufsize);
          exp(reinterpret_cast<float*>(buffer2), reinterpret_cast<float*>(buffer1), aligned_bufsize);
          atomic_reduce_sum(reinterpret_cast<float*>(buffer3), reinterpret_cast<float*>(buffer2), bufsize);
          tops::mdspan hbm_output(tops::Global, output_cur + offset * last_dim_size + k, bufsize);
          tops::mdspan l2_output(tops::Shared, output_cur_share + offset * last_dim_size + k, bufsize);
          tops::mdspan l1_output(tops::Private, buffer2, bufsize);
          tops::memcpy(ctx, output_cachable? l2_output : hbm_output, l1_output);
          exp_sum += buffer3[0];
      }
      for (int k = 0; k < last_dim_size; k+=TILE_SIZE) {
          int bufsize = (k + TILE_SIZE < last_dim_size) ? TILE_SIZE : last_dim_size - k;
          int aligned_bufsize = ALIGN_UP(bufsize, SIP_VECTOR_LENGTH);
          tops::mdspan l1_input(tops::Private, bufferTmp, bufsize);
          tops::mdspan l1_output(tops::Private, buffer1, bufsize);
          tops::mdspan hbm_input_output(tops::Global, output_cur + offset * last_dim_size + k, bufsize);
          tops::mdspan l2_input_output(tops::Shared, output_cur_share + offset * last_dim_size + k, bufsize);
          tops::memcpy(ctx, l1_input, output_cachable? l2_input_output : hbm_input_output);
          div(reinterpret_cast<float*>(buffer1), reinterpret_cast<float*>(bufferTmp), exp_sum, aligned_bufsize);
          tops::memcpy(ctx, hbm_input_output, l1_output);
      }
    }
}

SOFTMAX_OP(__fp16, tops::half, softmax_f16, fast_max_f16_kernel)
SOFTMAX_OP(__bf16, tops::bfloat, softmax_bf16, fast_max_bf16_kernel)

#define TILE_SIZE_NORM 1024 * 48
template <typename T>
__device__ void layernorm_kernel(T *input, T* output, T* weight, T* bias, int batch,
    int chunks, int last_dims_size, float epsilon, int remove_mean, int affine) {
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx);
    __local__ __valigned__ float buffer1[TILE_SIZE_NORM];
    __local__ __valigned__ float buffer2[TILE_SIZE_NORM];
    __local__ __valigned__ float buffer3[128];
    __local__ __valigned__ T bufferTmp[TILE_SIZE_NORM];
    __local__ __valigned__ T bufferWeight[TILE_SIZE_NORM];
    __local__ __valigned__ T bufferBias[TILE_SIZE_NORM];
    __local__ __valigned__ T bufferOut[TILE_SIZE_NORM];
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
    GetThreadStep(N, thread_step, THREAD_STEP);

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
      // buffer1 -> (xi - mean) / sqrt(varience + epsilon)
      div(reinterpret_cast<float*>(buffer1), reinterpret_cast<float*>(buffer2), sqrtf(var_value + epsilon), last_dims_size);
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

int main() {}