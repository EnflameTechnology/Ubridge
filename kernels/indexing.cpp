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
 * @date    2023-11-23 to 2025-02-17
 * @version V0.1
 * @par     Copyright (c) Enflame Tech Company.
 * @par     History: 
 * @par     Comments: gcu kernels for index selection and gather.
 */

#include <stdio.h>
#include <tops.h>
#include <tops/half.h>
#include <tops/bfloat.h>
#include <algorithm>
#include <vector>
#include "tops/tops_runtime.h"
#include "utils/utils.h"
#include <krt/mmu.h>

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
    GetThreadStep(N, thread_step, THREAD_STEP);

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


#define TEMPLATE_ALIGN_UP(a, b) (((a + b - 1) / b) * b)
#define L1_ALIGN_SIZE (128)
template<typename T, typename I>
__device__ void gather(
    const size_t numel,
    I *ids,
    T *inp,
    T *out,
    const size_t left_size,
    const size_t src_dim_size,
    const size_t ids_dim_size,
    const size_t right_size
) {
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx);
    __local__ __valigned__ char buffer[VDMEM_VALID_SIZE];
    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNum();
    int thread_step = 1;
    int THREAD_STEP = 1;
    extern __shared__ char l2_buffer[];
    I* ids_buffer = reinterpret_cast<I*>(buffer);
    tops::mdspan l1_ids(tops::Private, ids_buffer, numel);
    tops::mdspan hbm_ids(tops::Global, ids, numel);
    tops::memcpy(ctx, l1_ids, hbm_ids);

    GetThreadStep(numel, thread_step, THREAD_STEP);
    // if (numel * sizeof(T) +  ids_dim_size * sizeof(I) < VDMEM_VALID_SIZE) {
    //   //input on L1 cache
    //   T* inp_buffer = reinterpret_cast<T*>(buffer + TEMPLATE_ALIGN_UP(ids_dim_size * sizeof(I), L1_ALIGN_SIZE));
    //   T* out_buffer = reinterpret_cast<T*>(l2_buffer);
    //   tops::mdspan l1_inp(tops::Private, inp_buffer, numel);
    //   tops::mdspan hbm_inp(tops::Global, inp, numel);
    //   tops::memcpy(ctx, l1_inp, hbm_inp);

    //   for (int idx = 0; idx < thread_step; idx++) {
    //     int i = thread_id * THREAD_STEP + idx;
    //     if (i < numel) {
    //         size_t post = i % right_size;
    //         size_t idx_ = ids_buffer[i];
    //         size_t pre = i / (right_size * ids_dim_size);
    //         size_t src_i = (pre * src_dim_size + idx_) * right_size + post;
    //         out_buffer[i] = inp_buffer[src_i];
    //     }
    //   }
    //   __syncthreads();
    //   tops::mdspan l2_out(tops::Shared, out_buffer + thread_id * THREAD_STEP, thread_step);
    //   tops::mdspan hbm_out(tops::Global, out + thread_id * THREAD_STEP, thread_step);
    //   tops::memcpy(ctx, hbm_out, l2_out);
    // } else {
      int in_map_size = AlignUp(numel, L1_ALIGN_SIZE) * sizeof(T);
      auto src_l3_addr = map_mem(reinterpret_cast<generic_ptr>(inp), in_map_size);
      T* src_hbm = reinterpret_cast<T*>(src_l3_addr);
      auto src_l3_addr1 = map_mem(reinterpret_cast<generic_ptr>(out), in_map_size);
      T* out_hbm = reinterpret_cast<T*>(src_l3_addr1);

      for (int idx = 0; idx < thread_step; idx++) {
        int i = thread_id * THREAD_STEP + idx;
        if (i < numel) {
          size_t post = i % right_size;
          size_t idx_ = ids_buffer[i];
          size_t pre = i / (right_size * ids_dim_size);
          size_t src_i = (pre * src_dim_size + idx_) * right_size + post;
          out_hbm[i] = src_hbm[src_i];
        }
      }
      unmap_mem(src_l3_addr);
      unmap_mem(src_l3_addr1);
    // }
}

#define GATHER_OP(TYPENAME, INDEX_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t numel,  \
    INDEX_TYPENAME *ids, \
    TYPENAME *inp, \
    TYPENAME *out, \
    const size_t left_size, \
    const size_t src_dim_size, \
    const size_t ids_dim_size, \
    const size_t right_size \
) { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size); } \

GATHER_OP(__fp16, int64_t, gather_i64_f16)
GATHER_OP(__bf16, int64_t, gather_i64_bf16)
GATHER_OP(float, int64_t, gather_i64_f32)
GATHER_OP(double, int64_t, gather_i64_f64)
GATHER_OP(uint8_t, int64_t, gather_i64_u8)
GATHER_OP(uint32_t, int64_t, gather_i64_u32)
GATHER_OP(int64_t, int64_t, gather_i64_i64)

GATHER_OP(__fp16, uint32_t, gather_u32_f16)
GATHER_OP(__bf16, uint32_t, gather_u32_bf16)
GATHER_OP(float, uint32_t, gather_u32_f32)
GATHER_OP(double, uint32_t, gather_u32_f64)
GATHER_OP(uint8_t, uint32_t, gather_u32_u8)
GATHER_OP(int64_t, uint32_t, gather_u32_i64)
GATHER_OP(uint32_t, uint32_t, gather_u32_u32)

GATHER_OP(__fp16, uint8_t, gather_u8_f16)
GATHER_OP(__bf16, uint8_t, gather_u8_bf16)
GATHER_OP(float, uint8_t, gather_u8_f32)
GATHER_OP(double, uint8_t, gather_u8_f64)
GATHER_OP(uint8_t, uint8_t, gather_u8_u8)
GATHER_OP(uint32_t, uint8_t, gather_u8_u32)
GATHER_OP(int64_t, uint8_t, gather_u8_i64)



template<typename T, typename I>
__device__ void index_add(
    I *ids,
    const size_t ids_dim_size,
    T *inp,
    T *out,
    const size_t left_size,
    const size_t src_dim_size,
    const size_t dst_dim_size,
    const size_t right_size
) {
    extern __shared__ char l2_buffer[];
    const size_t numel = left_size * right_size;
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx);
    __local__ __valigned__ char buffer[VDMEM_VALID_SIZE];
    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNum();
    int thread_step = 1;
    int THREAD_STEP = 1;
    I* ids_buffer = reinterpret_cast<I*>(buffer);
    tops::mdspan l1_ids(tops::Private, ids_buffer, ids_dim_size);
    tops::mdspan hbm_ids(tops::Global, ids, ids_dim_size);
    tops::memcpy(ctx, l1_ids, hbm_ids);

    GetThreadStep(numel, thread_step, THREAD_STEP);
    // if (2 * numel * sizeof(T) < SHARE_BUFFER_SIZE && ids_dim_size * sizeof(I) < VDMEM_VALID_SIZE) {
    //   T* inp_buffer = reinterpret_cast<T*>(l2_buffer);
    //   T* out_buffer = reinterpret_cast<T*>(l2_buffer + TEMPLATE_ALIGN_UP(numel * sizeof(T), L1_ALIGN_SIZE));

    //   //input/output on L2 cache
    //   tops::mdspan l2_inp(tops::Shared, inp_buffer + thread_id * THREAD_STEP, thread_step);
    //   tops::mdspan hbm_inp(tops::Global, inp + thread_id * THREAD_STEP, thread_step);
    //   tops::memcpy(ctx, l2_inp, hbm_inp);
    //   tops::mdspan l2_out(tops::Shared, out_buffer + thread_id * THREAD_STEP, thread_step);
    //   tops::mdspan hbm_out(tops::Global, out + thread_id * THREAD_STEP, thread_step);
    //   tops::memcpy(ctx, l2_out, hbm_out);
    //   __syncthreads();

    //   for (int idx = 0; idx < thread_step; idx++) {
    //     int i = thread_id * THREAD_STEP + idx;
    //     if (i < numel) {
    //         const size_t pre = i / right_size;
    //         const size_t post = i % right_size;
    //         for (unsigned int j = 0; j < ids_dim_size; ++j) {
    //             const size_t idx_ = (size_t)ids_buffer[j];
    //             const size_t src_i = (pre * ids_dim_size + j) * right_size + post;
    //             const size_t dst_i = (pre * dst_dim_size + idx_) * right_size + post;
    //             out_buffer[dst_i] += inp_buffer[src_i];
    //         }
    //     }
    //   }
    //   __syncthreads();
    //   tops::memcpy(ctx, hbm_out, l2_out);
    // } else {
        int in_map_size = AlignUp(numel, L1_ALIGN_SIZE) * sizeof(T);
        auto src_l3_addr = map_mem(reinterpret_cast<generic_ptr>(inp), in_map_size);
        T* src_hbm = reinterpret_cast<T*>(src_l3_addr);
        auto src_l3_addr1 = map_mem(reinterpret_cast<generic_ptr>(out), in_map_size);
        T* out_hbm = reinterpret_cast<T*>(src_l3_addr1);
        for (int idx = 0; idx < thread_step; idx++) {
          int i = thread_id * THREAD_STEP + idx;
          if (i < numel) {
            const size_t pre = i / right_size;
            const size_t post = i % right_size;
            for (unsigned int j = 0; j < ids_dim_size; ++j) {
                const size_t idx = ids_buffer[j];
                const size_t src_i = (pre * ids_dim_size + j) * right_size + post;
                const size_t dst_i = (pre * dst_dim_size + idx) * right_size + post;
                out_hbm[dst_i] += src_hbm[src_i];
            }
          }
      }
      unmap_mem(src_l3_addr);
      unmap_mem(src_l3_addr1);
    // }
}

#define IA_OP(TYPENAME, INDEX_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    INDEX_TYPENAME *ids, \
    const size_t ids_dim_size, \
    TYPENAME *inp, \
    TYPENAME *out, \
    const size_t left_size, \
    const size_t src_dim_size, \
    const size_t dst_dim_size, \
    const size_t right_size \
) { index_add(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size); } \


IA_OP(__fp16, int64_t, ia_i64_f16)
IA_OP(__bf16, int64_t, ia_i64_bf16)
IA_OP(float, int64_t, ia_i64_f32)
IA_OP(double, int64_t, ia_i64_f64)
IA_OP(uint8_t, int64_t, ia_i64_u8)
IA_OP(int64_t, int64_t, ia_i64_i64)
IA_OP(uint32_t, int64_t, ia_i64_u32)

IA_OP(__fp16, uint32_t, ia_u32_f16)
IA_OP(__bf16, uint32_t, ia_u32_bf16)
IA_OP(float, uint32_t, ia_u32_f32)
IA_OP(double, uint32_t, ia_u32_f64)
IA_OP(uint8_t, uint32_t, ia_u32_u8)
IA_OP(int64_t, uint32_t, ia_u32_i64)
IA_OP(uint32_t, uint32_t, ia_u32_u32)

IA_OP(__fp16, uint8_t, ia_u8_f16)
IA_OP(__bf16, uint8_t, ia_u8_bf16)
IA_OP(float, uint8_t, ia_u8_f32)
IA_OP(double, uint8_t, ia_u8_f64)
IA_OP(uint8_t, uint8_t, ia_u8_u8)
IA_OP(uint32_t, uint8_t, ia_u8_u32)
IA_OP(int64_t, uint8_t, ia_u8_i64)

int main() {}