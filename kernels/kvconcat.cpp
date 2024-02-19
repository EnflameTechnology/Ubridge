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
 * @file    kvconcat.cpp
 * @brief
 *
 * @author  Guoqing Bao
 * @date    2024-01-10 to 2024-02-02
 * @version V0.1
 * @par     Copyright (c) Enflame Tech Company.
 * @par     History: Remove PING_PONG logic & support 3-dim kvconcat
 * @par     Comments: gcu kernel for faster key-value concatination in LLM models.
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

template <typename T>
__device__ __forceinline__ void kvconcat_dim0_kernel(T *ltensor, T* rtensor, T *out,
    size_t* dims, size_t dim_size) {
    __local__ __valigned__ size_t dim_buf[10];
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx);
    tops::mdspan hbm_dims(tops::Global, dims, dim_size * 2);
    tops::mdspan l1_dims(tops::Private, dim_buf, dim_size * 2);
    tops::memcpy(ctx, l1_dims, hbm_dims);

    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNum();
    int LN = dim_size > 3 ? dim_buf[0] * dim_buf[1] : dim_buf[0];
    int RN = dim_size > 3 ? dim_buf[4] * dim_buf[5] : dim_buf[3];
    int lstride = dim_size > 3 ? dim_buf[2] * dim_buf[3] : dim_buf[1] * dim_buf[2];
    int rstride = dim_size > 3 ? dim_buf[6] * dim_buf[7]: dim_buf[4] * dim_buf[5];
    
    tops::memcpy(ctx, tops::mdspan(tops::Global, out, LN * lstride), tops::mdspan(tops::Global, ltensor, LN * lstride));
    tops::memcpy(ctx, tops::mdspan(tops::Global, out + LN * lstride, RN * rstride), tops::mdspan(tops::Global, rtensor, RN * rstride));

}
template <typename T>
__device__ __forceinline__ void kvconcat_dim2_kernel(T *ltensor, T* rtensor, T *out,
    size_t* dims, size_t dim_size) {
    __local__ __valigned__ size_t dim_buf[10];
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx);
    tops::mdspan hbm_dims(tops::Global, dims, dim_size * 2);
    tops::mdspan l1_dims(tops::Private, dim_buf, dim_size * 2);
    tops::memcpy(ctx, l1_dims, hbm_dims);

    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNum();
    int N = dim_size > 3 ? dim_buf[0] * dim_buf[1] : dim_buf[0];
    int lstride = dim_size > 3 ? dim_buf[2] * dim_buf[3] : dim_buf[1] * dim_buf[2];
    int rstride = dim_size > 3 ? dim_buf[6] * dim_buf[7]: dim_buf[4] * dim_buf[5];
    int out_stride = lstride + rstride;

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
          tops::mdspan hbm_left(tops::Global, ltensor + idx * lstride, lstride);
          tops::mdspan hbm_right(tops::Global, rtensor + idx * rstride, rstride);
          tops::mdspan hbm_out(tops::Global, out + idx * out_stride, lstride);
          // printf("\ncopy left %d to %d [size %d]", idx * lstride, idx * out_stride, lstride);
          // printf("\ncopy right %d to %d [size %d]", idx * rstride, idx * out_stride + lstride, rstride);
          // printf("\r\n");
          tops::memcpy(ctx, hbm_out, hbm_left);
          tops::mdspan hbm_out1(tops::Global, out + idx * out_stride + lstride, rstride);
          tops::memcpy(ctx, hbm_out1, hbm_right);

      }
    }
}

extern "C" __global__ void kvconcat_f16(__fp16 *ltensor, __fp16* rtensor, __fp16 *out,
    size_t* dims, size_t dim_size, int concat_dim) {
    if (concat_dim == 2)
      kvconcat_dim2_kernel<__fp16>(ltensor, rtensor, out, dims, dim_size);
    else if (concat_dim == 0) {
        if (blockIdx.x == 0 && threadIdx.x ==0) 
          kvconcat_dim0_kernel<__fp16>(ltensor, rtensor, out, dims, dim_size);
    }
    else {
      printf("Not support kvconcat on dim %d\n", concat_dim);
    }
}

extern "C" __global__ void kvconcat_bf16(__bf16 *ltensor, __bf16* rtensor, __bf16 *out,
    size_t* dims, size_t dim_size, int concat_dim) {
    if (concat_dim == 2)
      kvconcat_dim2_kernel<__bf16>(ltensor, rtensor, out, dims, dim_size);
    else if (concat_dim == 0) {
      if (blockIdx.x == 0 && threadIdx.x ==0) 
        kvconcat_dim0_kernel<__bf16>(ltensor, rtensor, out, dims, dim_size);
    }
    else {
      printf("Not support kvconcat on dim %d\n", concat_dim);
    }
}

extern "C" __global__ void kvconcat_f32(float *ltensor, float* rtensor, float *out,
    size_t* dims, size_t dim_size, int concat_dim) {
    if (concat_dim == 2)
      kvconcat_dim2_kernel<float>(ltensor, rtensor, out, dims, dim_size);
    else if (concat_dim == 0) {
      if (blockIdx.x == 0 && threadIdx.x ==0) 
        kvconcat_dim0_kernel<float>(ltensor, rtensor, out, dims, dim_size);
    }
    else {
      printf("Not support kvconcat on dim %d\n", concat_dim);
    }
}

extern "C" __global__ void kvconcat_f64(double *ltensor, double* rtensor, double *out,
    size_t* dims, size_t dim_size, int concat_dim) {
    if (concat_dim == 2)
      kvconcat_dim2_kernel<double>(ltensor, rtensor, out, dims, dim_size);
    else if (concat_dim == 0) {
      if (blockIdx.x == 0 && threadIdx.x ==0) 
        kvconcat_dim0_kernel<double>(ltensor, rtensor, out, dims, dim_size);
    }
    else {
      printf("Not support kvconcat on dim %d\n", concat_dim);
    }
}

extern "C" __global__ void kvconcat_u8(uint8_t *ltensor, uint8_t* rtensor, uint8_t *out,
    size_t* dims, size_t dim_size, int concat_dim) {
    if (concat_dim == 2)
      kvconcat_dim2_kernel<uint8_t>(ltensor, rtensor, out, dims, dim_size);
    else if (concat_dim == 0) {
      if (blockIdx.x == 0 && threadIdx.x ==0) 
        kvconcat_dim0_kernel<uint8_t>(ltensor, rtensor, out, dims, dim_size);
    }
    else {
      printf("Not support kvconcat on dim %d\n", concat_dim);
    }
}

int main() {}