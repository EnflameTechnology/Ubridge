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
 * @file    embedding.cpp
 * @brief
 *
 * @author  Guoqing Bao
 * @date    2024-01-08 to 2024-06-13
 * @version V0.1
 * @par     Copyright (c) Enflame Tech Company.
 * @par     History: Support partial rotary embedding, vectorized compute
 * @par     Comments: gcu kernel for rotary embedding (partial rotary embedding also supported).
 */
#include <stdio.h>
#include <tops.h>
#include <tops/half.h>
#include <tops/bfloat.h>
#include <tops/tops_runtime.h>
#include "utils/utils.h"
#include <acore_op.h>
#include "utils/vector_ex.h"
using namespace std;


template <typename T>
__device__ __forceinline__  void apply_rotary_qkv(T *q_arr, T *k_arr, T *cos_ptr, T *sin_ptr,
                                int rot_offset, int embed_dim, int gpt_geox,
                                int j, int nq, int nk) {

  int x_index, y_index;
  T cos, sin;
  if (gpt_geox) {
    // GPT-NeoX style
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = cos_ptr[x_index];
    sin = sin_ptr[x_index];
  } else {
    // GPT-J style
    x_index = 2 * rot_offset;
    y_index = x_index + 1;
    cos = cos_ptr[x_index / 2];
    sin = sin_ptr[x_index / 2];
  }

  T x = q_arr[x_index];
  T y = q_arr[y_index];
  if (j < nq) {
    q_arr[x_index] = x * cos - y * sin;
    q_arr[y_index] = y * cos + x * sin;
  }

  if (j < nk) {
    x = k_arr[x_index];
    y = k_arr[y_index];
    k_arr[x_index] = x * cos - y * sin;
    k_arr[y_index] = y * cos + x * sin;
  }
}

template <typename T>
__device__ __forceinline__  void apply_rotary(T *arr, T *cos_ptr, T *sin_ptr,
                                int rot_offset, int embed_dim, int gpt_geox) {

  int x_index, y_index;
  T cos, sin;
  if (gpt_geox) {
    // GPT-NeoX style
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = cos_ptr[x_index];
    sin = sin_ptr[x_index];
  } else {
    // GPT-J style
    x_index = 2 * rot_offset;
    y_index = x_index + 1;
    cos = cos_ptr[x_index / 2];
    sin = sin_ptr[x_index / 2];
  }

  T x = arr[x_index];
  T y = arr[y_index];
  arr[x_index] = x * cos - y * sin;
  arr[y_index] = y * cos + x * sin;
}


template <typename T>
__device__ __forceinline__  void apply_rotary_qkv_batch(T *q_arr, T *k_arr, T *cos_ptr, T *sin_ptr,
                                va16u32x4 offsets, va16u32x4 rot_offsets, va16u32x4 embed_dims, 
                                const va16u32x4 VEC1, const va16u32x4 VEC2, va16u32x4 vec_bpe,
                                int gpt_geox, int j, int nq, int nk) {
  using vtype = typename tops::scalar_to_vector<T, TOPS_VECTOR_LENGTH>::type;
  va16u32x4 x_index, y_index, x_index_;
  vtype cos, sin, x, y;
  if (gpt_geox) {
    // GPT-NeoX style
    x_index = rot_offsets;
    y_index = vadd(embed_dims, rot_offsets);
    x_index_ = vmul(x_index, vec_bpe);
    cos = vgather<vtype>(cos_ptr, x_index_);
    sin = vgather<vtype>(sin_ptr, x_index_);
  } else {
    // GPT-J style
    x_index = vmul(VEC2, rot_offsets);
    y_index = vadd(x_index, VEC1);
    x_index_ = vmul(vdiv(x_index, VEC2), vec_bpe);
    cos = vgather<vtype>(cos_ptr, x_index_);
    sin = vgather<vtype>(sin_ptr, x_index_);
  }

  x_index = vmul(vadd(x_index, offsets), vec_bpe);
  y_index = vmul(vadd(y_index, offsets), vec_bpe);

  if (j < nq) {
    x = vgather<vtype>(q_arr, x_index);
    y = vgather<vtype>(q_arr, y_index);
    auto v1 = vsub(vmul(x, cos), vmul(y, sin));
    auto v2 = vadd(vmul(y, cos), vmul(x, sin));
    vscatter<vtype>(q_arr, v1, x_index);
    vscatter<vtype>(q_arr, v2, y_index);
  }

  if (j < nk) {
    x = vgather<vtype>(k_arr, x_index); 
    y = vgather<vtype>(k_arr, y_index);
    auto v1 = vsub(vmul(x, cos), vmul(y, sin));
    auto v2 = vadd(vmul(y, cos), vmul(x, sin));
    vscatter<vtype>(k_arr, v1, x_index);
    vscatter<vtype>(k_arr, v2, y_index);
  }
}

template <typename T, typename CST>
__device__ void rope(T* query, T* key, CST* cos_sin, int cos_sin_stride, int* index_positions, int batch,
    int num_tokens, int q_heads, int k_heads, int hidden_size, int split_dim, int gpt_geox) {
    int q_stride = q_heads * hidden_size;
    int k_stride = k_heads * hidden_size;
    int q_stride_whole = q_stride * num_tokens;
    int k_stride_whole = k_stride * num_tokens;
    if (split_dim == hidden_size || split_dim <= 0 || split_dim > hidden_size) {
      split_dim = hidden_size;
    }
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx);
    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNum();
    int THREAD_STEP = 1;
    int thread_step = 1;
    int N = batch * num_tokens;
    GetThreadStep(N, thread_step, THREAD_STEP);
    __local__ __valigned__ CST bufCosSin[1024 * 48];
    __local__ __valigned__ T bufQuery[1024 * 48];
    __local__ __valigned__ T bufKey[1024 * 48];
    __local__ __valigned__ float bufCosSinf32[1024 * 48];
    __local__ __valigned__ float bufQueryf32[1024 * 48];
    __local__ __valigned__ float bufKeyf32[1024 * 48];
    __local__ int positions[4096];

    tops::memcpy(ctx, tops::mdspan(tops::Private, positions, batch), 
            tops::mdspan(tops::Global, index_positions, batch));

    tops::mdspan cos_sin_l1(tops::Private, bufCosSin, hidden_size);
    tops::mdspan query_l1(tops::Private, bufQuery, q_stride);
    tops::mdspan key_l1(tops::Private, bufKey, k_stride);
    int32_t embed_dim = split_dim / 2;
    int nq = q_heads * embed_dim;
    int nk = k_heads * embed_dim;
    int max_qk = nq > nk ? nq : nk;
    constexpr int TILESIZE = vector_length<va16u32x4>::value;
    va16u32x4 embed_dims = vbroadcast<va16u32x4>((unsigned int)embed_dim);
    va16u32x4 hidden_sizes = vbroadcast<va16u32x4>((unsigned int)hidden_size);
    const va16u32x4 VEC1 = vbroadcast<va16u32x4>((unsigned int)1);
    const va16u32x4 VEC2 = vbroadcast<va16u32x4>((unsigned int)2);
    va16u32x4 vec_bpe = vbroadcast<va16u32x4>((unsigned int)sizeof(float));
    for (int p = 0; p < thread_step; p++) {
      int idx = thread_id * THREAD_STEP + p;
      if (idx >= N) break;
      int batch_idx = idx / num_tokens;
      CST* cos_sin_cur = cos_sin + positions[batch_idx] * cos_sin_stride;
      int i = idx % num_tokens;
      if (i < num_tokens) {
        auto query_sub0 = query + batch_idx * q_stride_whole + i * q_stride;
        auto key_sub0 = key + batch_idx * k_stride_whole + i * k_stride;
        tops::mdspan hbm_query(tops::Global, query_sub0, q_stride);
        tops::mdspan hbm_key(tops::Global, key_sub0, k_stride);
        tops::memcpy(ctx, query_l1, hbm_query);
        tops::memcpy(ctx, key_l1, hbm_key);
        auto cos_sin_cache = cos_sin_cur + i * split_dim;
        tops::mdspan hbm_cos_sin(tops::Global, cos_sin_cache, split_dim);
        tops::memcpy(ctx, cos_sin_l1, hbm_cos_sin);
        if (sizeof(T) < 4) {
          convert<float, T>(reinterpret_cast<float*>(bufQueryf32), reinterpret_cast<T*>(bufQuery), q_stride);
          convert<float, T>(reinterpret_cast<float*>(bufKeyf32), reinterpret_cast<T*>(bufKey), k_stride);
        }
        if (sizeof(CST) < 4) {
          convert<float, T>(reinterpret_cast<float*>(bufCosSinf32), reinterpret_cast<T*>(bufCosSin), split_dim);
        }
        auto cos_ptr = sizeof(CST) < 4 ? bufCosSinf32 : reinterpret_cast<float*>(bufCosSin);
        auto sin_ptr = cos_ptr + embed_dim;
        auto query_ptr = sizeof(T) < 4 ? bufQueryf32 : reinterpret_cast<float*>(bufQuery);
        auto key_ptr = sizeof(T) < 4 ? bufKeyf32 : reinterpret_cast<float*>(bufKey);
        for (int j = 0; j < max_qk; j+=TILESIZE) { //This is safe even though nk != nq (redundant compute)
          int bufsize = (j + TILESIZE < max_qk) ? TILESIZE : max_qk - j;
          if (bufsize == TILESIZE) {
            auto indexes = viota<va16u32x4>((unsigned int)j);
            auto rot_offsets = vrem(indexes, embed_dims);
            auto head_idxes = vdiv(indexes, embed_dims);
            auto offsets = vmul(head_idxes, hidden_sizes);
            apply_rotary_qkv_batch<float>(query_ptr, key_ptr, cos_ptr, sin_ptr, 
                                        offsets, rot_offsets, embed_dims, VEC1, VEC2, vec_bpe, gpt_geox, j, nq, nk);
          } else {
            for (int k = j; k < max_qk; k++) {
              int head_idx = k / embed_dim;
              int offset = head_idx * hidden_size;
              int rot_offset = k % embed_dim;
              auto q_arr = query_ptr + offset;
              auto k_arr = key_ptr + offset;
              apply_rotary_qkv(q_arr, k_arr, cos_ptr, sin_ptr, rot_offset, embed_dim, gpt_geox, k, nq, nk);
            }
            break;
          }

        }
        if (sizeof(T) < 4) {
          convert<T, float>(reinterpret_cast<T*>(bufQuery), reinterpret_cast<float*>(bufQueryf32), q_stride);
          convert<T, float>(reinterpret_cast<T*>(bufKey), reinterpret_cast<float*>(bufKeyf32), k_stride);
        }
        tops::memcpy(ctx, hbm_query, query_l1);
        tops::memcpy(ctx, hbm_key, key_l1);
      }
    }  // loop in num_tokens
}

extern "C" __global__ void  rope_f32(float *query, float *key, float *cos_sin, int cos_sin_stride, int* index_positions,
                      int batch, int num_tokens, int q_heads, int k_heads, int hidden_size, int split_dim, int gpt_geox) {
      rope<float, float>(
        query, key, cos_sin, cos_sin_stride, index_positions, batch,
        num_tokens, q_heads, k_heads, hidden_size, split_dim, gpt_geox);
}

extern "C" __global__ void  rope_f16(__fp16*query, __fp16 *key, __fp16 *cos_sin, int cos_sin_stride, int* index_positions,
                      int batch, int num_tokens, int q_heads, int k_heads, int hidden_size, int split_dim, int gpt_geox) {
    rope<__fp16, __fp16>(
      query, key, cos_sin, cos_sin_stride, index_positions, batch,
      num_tokens, q_heads, k_heads, hidden_size, split_dim, gpt_geox);
}

extern "C" __global__ void  rope_bf16(__bf16 *query, __bf16 *key, __bf16 *cos_sin, int cos_sin_stride, int* index_positions,
                       int batch, int num_tokens, int q_heads, int k_heads, int hidden_size, int split_dim, int gpt_geox) {
    rope<__bf16, __bf16>(
      query, key, cos_sin, cos_sin_stride, index_positions, batch,
      num_tokens, q_heads, k_heads, hidden_size, split_dim, gpt_geox);
}

extern "C" __global__ void  rope_f32_bf16(__bf16 *query, __bf16 *key, float *cos_sin, int cos_sin_stride, int* index_positions,
                       int batch, int num_tokens, int q_heads, int k_heads, int hidden_size, int split_dim, int gpt_geox) {
    rope<__bf16, float>(
      query, key, cos_sin, cos_sin_stride, index_positions, batch,
      num_tokens, q_heads, k_heads, hidden_size, split_dim, gpt_geox);
}

extern "C" __global__ void  rope_f32_f16(__fp16 *query, __fp16 *key, float *cos_sin, int cos_sin_stride, int* index_positions,
                       int batch, int num_tokens, int q_heads, int k_heads, int hidden_size, int split_dim, int gpt_geox) {
    rope<__fp16, float>(
      query, key, cos_sin, cos_sin_stride, index_positions, batch,
      num_tokens, q_heads, k_heads, hidden_size, split_dim, gpt_geox);
}

int main() {}