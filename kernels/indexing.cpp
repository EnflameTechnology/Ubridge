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
 * @file    indexing.cpp
 * @brief
 *
 * @author  Guoqing Bao
 * @date    2023-11-23 to 2024-01-02
 * @version V0.1
 * @par     Copyright (c) Enflame Tech Company.
 * @par     History: 
 * @par     Comments: gcu kernel for index selection.
 */

#include <stdio.h>
#include <tops.h>
#include <tops/half.h>
#include <tops/bfloat.h>
#include <algorithm>
#include <vector>
#include "tops/tops_runtime.h"
#include "utils/utils.h"
using namespace std;
#define TILE_SIZE AlignDown(((VDMEM_VALID_SIZE) / 16), 256)
#define MAX_IDS_SIZE 40960

template <typename ID_TYPENAME, typename T>
__device__ __forceinline__ void index_select_kernel(const size_t id_numel,
    ID_TYPENAME *ids, T *inp, T *out,
    const size_t left_size, const size_t dim_size, const size_t right_size) {
    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNum();
    int N = id_numel;
    __local__ __valigned__ ID_TYPENAME ids_buffer[MAX_IDS_SIZE];

    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx);
    tops::mdspan l1_ids(tops::Private, ids_buffer, N);
    tops::mdspan hbm_ids(tops::Global, ids, N);
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
    index_select_kernel<ID_TYPE, TYPE> \
    (id_numel, ids, inp, out, left_size, dim_size, right_size); \
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

int main() {}