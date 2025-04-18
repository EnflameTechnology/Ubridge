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
#include <acore_op.h>
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
    const size_t numel, //number of output (gathered elements)
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

    if (left_size * src_dim_size * sizeof(T) +  numel * sizeof(I) < VDMEM_VALID_SIZE) {
      GetThreadStep(left_size, thread_step, THREAD_STEP);
      //input on L1 cache
      T* inp_buffer = reinterpret_cast<T*>(buffer + TEMPLATE_ALIGN_UP(numel * sizeof(I), L1_ALIGN_SIZE));
      T* out_buffer = reinterpret_cast<T*>(l2_buffer);
      tops::mdspan l1_inp(tops::Private, inp_buffer, left_size * src_dim_size);
      tops::mdspan hbm_inp(tops::Global, inp, left_size * src_dim_size);
      tops::memcpy(ctx, l1_inp, hbm_inp);

      for (int idx = 0; idx < thread_step; idx++) {
        int i = thread_id * THREAD_STEP + idx;
        if (i < left_size) {
          for (int j = 0; j < ids_dim_size; j++) {
            size_t idx_ = ids_buffer[i * ids_dim_size + j];
            size_t src_i = i * src_dim_size + idx_;
            out_buffer[i * ids_dim_size + j] = inp_buffer[src_i];
          }
        }
      }
      __syncthreads();
      if (thread_id == 0) {
        tops::mdspan l2_out(tops::Shared, out_buffer, left_size * ids_dim_size);
        tops::mdspan hbm_out(tops::Global, out, left_size * ids_dim_size);
        tops::memcpy(ctx, hbm_out, l2_out);
      }
    } else {
      GetThreadStep(numel, thread_step, THREAD_STEP);
      int in_map_size = AlignUp(left_size * src_dim_size, L1_ALIGN_SIZE) * sizeof(T);
      auto src_l3_addr = map_mem_ex(reinterpret_cast<generic_ptr>(inp), in_map_size);
      T* src_hbm = reinterpret_cast<T*>(src_l3_addr);
      int out_map_size = AlignUp(numel, L1_ALIGN_SIZE) * sizeof(T);
      auto src_l3_addr1 = map_mem_ex(reinterpret_cast<generic_ptr>(out), out_map_size);
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
      unmap_mem_ex(src_l3_addr, in_map_size);
      unmap_mem_ex(src_l3_addr1, out_map_size);
    }
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
    GetThreadStep(ids_dim_size, thread_step, THREAD_STEP);
    if (ids_dim_size * sizeof(I) + 3 * right_size * sizeof(T) < VDMEM_VALID_SIZE) {
      int mem_size = (TEMPLATE_ALIGN_UP(ids_dim_size * sizeof(I), L1_ALIGN_SIZE) + L1_ALIGN_SIZE);
      T* inp_buffer = reinterpret_cast<T*>(buffer + mem_size);
      mem_size += (TEMPLATE_ALIGN_UP(right_size * sizeof(T), L1_ALIGN_SIZE) + 2 * L1_ALIGN_SIZE);
      T* out_buffer = reinterpret_cast<T*>(buffer + mem_size);
      mem_size += (TEMPLATE_ALIGN_UP(right_size * sizeof(T), L1_ALIGN_SIZE)+ 3 * L1_ALIGN_SIZE);
      T* final_buffer = reinterpret_cast<T*>(buffer + mem_size);

      for (int idx_ = 0; idx_ < thread_step; idx_++) {
        int i = thread_id * THREAD_STEP + idx_;
        if (i < ids_dim_size) {
            int idx = ids_buffer[i];
            if (idx < dst_dim_size) {
              //a row of input/output on L1 cache
              tops::mdspan l1_inp(tops::Private, inp_buffer, right_size);
              tops::mdspan hbm_inp(tops::Global, inp + i * right_size, right_size);
              tops::memcpy(ctx, l1_inp, hbm_inp);
              tops::mdspan l1_out(tops::Private, out_buffer, right_size);
              tops::mdspan hbm_out(tops::Global, out + idx * right_size, right_size);
              tops::memcpy(ctx, l1_out, hbm_out);
              add(reinterpret_cast<T*>(final_buffer),
                  reinterpret_cast<T*>(inp_buffer),
                  reinterpret_cast<T*>(out_buffer),
                  right_size);
              tops::mdspan l1_final(tops::Private, final_buffer, right_size);
              tops::memcpy(ctx, hbm_out, l1_final);
            }
        }
      }
    } else {
        int in_map_size = AlignUp(numel, L1_ALIGN_SIZE) * sizeof(T);
        auto src_l3_addr = map_mem_ex(reinterpret_cast<generic_ptr>(inp), in_map_size);
        T* src_hbm = reinterpret_cast<T*>(src_l3_addr);
        int out_map_size = AlignUp(dst_dim_size * right_size, L1_ALIGN_SIZE) * sizeof(T);
        auto out_l3_addr = map_mem_ex(reinterpret_cast<generic_ptr>(out), out_map_size);
        T* out_hbm = reinterpret_cast<T*>(out_l3_addr);
        for (int idx_ = 0; idx_ < thread_step; idx_++) {
          int i = thread_id * THREAD_STEP + idx_;
          if (i < ids_dim_size) {
              int idx = ids_buffer[i];
              if (idx < dst_dim_size) {
                T* inp_buffer = src_hbm + i * right_size;
                T* out_buffer = out_hbm + idx * right_size;
                add(reinterpret_cast<T*>(out_buffer),
                    reinterpret_cast<T*>(inp_buffer),
                    reinterpret_cast<T*>(out_buffer),
                    right_size);
              }
          }
        }
        unmap_mem_ex(src_l3_addr, in_map_size);
        unmap_mem_ex(out_l3_addr, out_map_size);
    }
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
// IA_OP(double, int64_t, ia_i64_f64)
IA_OP(uint8_t, int64_t, ia_i64_u8)
// IA_OP(int64_t, int64_t, ia_i64_i64)
IA_OP(uint32_t, int64_t, ia_i64_u32)

IA_OP(__fp16, uint32_t, ia_u32_f16)
IA_OP(__bf16, uint32_t, ia_u32_bf16)
IA_OP(float, uint32_t, ia_u32_f32)
// IA_OP(double, uint32_t, ia_u32_f64)
IA_OP(uint8_t, uint32_t, ia_u32_u8)
// IA_OP(int64_t, uint32_t, ia_u32_i64)
IA_OP(uint32_t, uint32_t, ia_u32_u32)

IA_OP(__fp16, uint8_t, ia_u8_f16)
IA_OP(__bf16, uint8_t, ia_u8_bf16)
IA_OP(float, uint8_t, ia_u8_f32)
// IA_OP(double, uint8_t, ia_u8_f64)
IA_OP(uint8_t, uint8_t, ia_u8_u8)
IA_OP(uint32_t, uint8_t, ia_u8_u32)
// IA_OP(int64_t, uint8_t, ia_u8_i64)

int main() {}