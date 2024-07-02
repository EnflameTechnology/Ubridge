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

template <typename T>
__device__ void rope(T* query, T* key, T* cos_sin, int cos_sin_stride, int index_pos,
    int num_tokens, int q_heads, int k_heads, int hidden_size, int split_dim, int gpt_geox) {
    int q_stride = q_heads * hidden_size;
    int k_stride = k_heads * hidden_size;
    T* cos_sin_cur = cos_sin + index_pos * cos_sin_stride;
    if (split_dim == hidden_size || split_dim <= 0 || split_dim > hidden_size) {
      split_dim = hidden_size;
    }
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx);
    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNum();
    int THREAD_STEP = 1;
    int thread_step = 1;
    int N = num_tokens;
    if (N > MAX_THREADS) {
      THREAD_STEP = N / MAX_THREADS;
      thread_step = THREAD_STEP;
      if (N % MAX_THREADS != 0) {
        if (thread_id == MAX_THREADS - 1) {
          thread_step += N % MAX_THREADS; //last thread also process remains
        }
      }
    }
    __local__ __valigned__ T bufCosSin[1024 * 48];
    __local__ __valigned__ T bufQuery[1024 * 48];
    __local__ __valigned__ T bufKey[1024 * 48];
    __local__ __valigned__ float bufCosSinf32[1024 * 48];
    __local__ __valigned__ float bufQueryf32[1024 * 48];
    __local__ __valigned__ float bufKeyf32[1024 * 48];


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
      int i = thread_id * THREAD_STEP + p;
      if (i < N) {
        auto query_sub0 = query + i * q_stride;
        auto key_sub0 = key + i * k_stride;
        tops::mdspan hbm_query(tops::Global, query_sub0, q_stride);
        tops::mdspan hbm_key(tops::Global, key_sub0, k_stride);
        tops::memcpy(ctx, query_l1, hbm_query);
        tops::memcpy(ctx, key_l1, hbm_key);
        auto cos_sin_cache = cos_sin_cur + i * split_dim;
        tops::mdspan hbm_cos_sin(tops::Global, cos_sin_cache, split_dim);
        tops::memcpy(ctx, cos_sin_l1, hbm_cos_sin);
        convert<float, T>(reinterpret_cast<float*>(bufCosSinf32), reinterpret_cast<T*>(bufCosSin), split_dim);
        convert<float, T>(reinterpret_cast<float*>(bufQueryf32), reinterpret_cast<T*>(bufQuery), q_stride);
        convert<float, T>(reinterpret_cast<float*>(bufKeyf32), reinterpret_cast<T*>(bufKey), k_stride);
        auto cos_ptr = bufCosSinf32;
        auto sin_ptr = bufCosSinf32 + embed_dim;
        for (int j = 0; j < max_qk; j+=TILESIZE) { //This is safe even though nk != nq (redundant compute)
          int bufsize = (j + TILESIZE < max_qk) ? TILESIZE : max_qk - j;
          if (bufsize == TILESIZE) {
            auto indexes = viota<va16u32x4>((unsigned int)j);
            auto rot_offsets = vrem(indexes, embed_dims);
            auto head_idxes = vdiv(indexes, embed_dims);
            auto offsets = vmul(head_idxes, hidden_sizes);
            apply_rotary_qkv_batch<float>(bufQueryf32, bufKeyf32, cos_ptr, sin_ptr, 
                                        offsets, rot_offsets, embed_dims, VEC1, VEC2, vec_bpe, gpt_geox, j, nq, nk);
          } else {
            for (int k = j; k < max_qk; k++) {
              int head_idx = k / embed_dim;
              int offset = head_idx * hidden_size;
              int rot_offset = k % embed_dim;
              auto q_arr = bufQueryf32 + offset;
              auto k_arr = bufKeyf32 + offset;
              apply_rotary_qkv(q_arr, k_arr, cos_ptr, sin_ptr, rot_offset, embed_dim, gpt_geox, k, nq, nk);
            }
            break;
          }

        }
        convert<T, float>(reinterpret_cast<T*>(bufQuery), reinterpret_cast<float*>(bufQueryf32), q_stride);
        convert<T, float>(reinterpret_cast<T*>(bufKey), reinterpret_cast<float*>(bufKeyf32), k_stride);
        tops::memcpy(ctx, hbm_query, query_l1);
        tops::memcpy(ctx, hbm_key, key_l1);
      }
    }  // loop in num_tokens
}

extern "C" __global__ void  rope_f32(float *query, float *key, float *cos_sin, int cos_sin_stride, int index_pos,
                      int num_tokens, int q_heads, int k_heads, int hidden_size, int split_dim, int gpt_geox) {
    int q_stride = q_heads * hidden_size;
    int k_stride = k_heads * hidden_size;
    float* cos_sin_cur = cos_sin + index_pos * cos_sin_stride;
    if (split_dim == hidden_size || split_dim <= 0 || split_dim > hidden_size) {
      split_dim = hidden_size;
    }
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx);
    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNum();
    int THREAD_STEP = 1;
    int thread_step = 1;
    int N = num_tokens;
    if (N > MAX_THREADS) {
      THREAD_STEP = N / MAX_THREADS;
      thread_step = THREAD_STEP;
      if (N % MAX_THREADS != 0) {
        if (thread_id == MAX_THREADS - 1) {
          thread_step += N % MAX_THREADS; //last thread also process remains
        }
      }
    }
    __local__ __valigned__ float bufCosSin[1024 * 64];
    __local__ __valigned__ float bufQuery[1024 * 64];
    __local__ __valigned__ float bufKey[1024 * 64];

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
      int i = thread_id * THREAD_STEP + p;
      if (i < N) {
        auto query_sub0 = query + i * q_stride;
        auto key_sub0 = key + i * k_stride;
        tops::mdspan hbm_query(tops::Global, query_sub0, q_stride);
        tops::mdspan hbm_key(tops::Global, key_sub0, k_stride);
        tops::memcpy(ctx, query_l1, hbm_query);
        tops::memcpy(ctx, key_l1, hbm_key);
        auto cos_sin_cache = cos_sin_cur + i * cos_sin_stride;
        tops::mdspan hbm_cos_sin(tops::Global, cos_sin_cache, cos_sin_stride);
        tops::memcpy(ctx, cos_sin_l1, hbm_cos_sin);

        auto cos_ptr = bufCosSin;
        auto sin_ptr = bufCosSin + embed_dim;
        for (int j = 0; j < max_qk; j+=TILESIZE) { //This is safe even though nk != nq (redundant compute)
          int bufsize = (j + TILESIZE < max_qk) ? TILESIZE : max_qk - j;
          if (bufsize == TILESIZE) {
            auto indexes = viota<va16u32x4>((unsigned int)j);
            auto rot_offsets = vrem(indexes, embed_dims);
            auto head_idxes = vdiv(indexes, embed_dims);
            auto offsets = vmul(head_idxes, hidden_sizes);
            apply_rotary_qkv_batch<float>(bufQuery, bufKey, cos_ptr, sin_ptr, 
                                        offsets, rot_offsets, embed_dims, VEC1, VEC2, vec_bpe, gpt_geox, j, nq, nk);
          } else {
            for (int k = j; k < max_qk; k++) {
              int head_idx = k / embed_dim;
              int offset = head_idx * hidden_size;
              int rot_offset = k % embed_dim;
              auto q_arr = bufQuery + offset;
              auto k_arr = bufKey + offset;
              apply_rotary_qkv(q_arr, k_arr, cos_ptr, sin_ptr, rot_offset, embed_dim, gpt_geox, k, nq, nk);
            }
            break;
          }

        }
        tops::memcpy(ctx, hbm_query, query_l1);
        tops::memcpy(ctx, hbm_key, key_l1);
      }
    }  // loop in num_tokens
}

extern "C" __global__ void  rope_f16(__fp16*query, __fp16 *key, __fp16 *cos_sin, int cos_sin_stride, int index_pos,
                      int num_tokens, int q_heads, int k_heads, int hidden_size, int split_dim, int gpt_geox) {
    rope<__fp16>(
      query, key, cos_sin, cos_sin_stride, index_pos,
      num_tokens, q_heads, k_heads, hidden_size, split_dim, gpt_geox);
}

extern "C" __global__ void  rope_bf16(__bf16 *query, __bf16 *key, __bf16 *cos_sin, int cos_sin_stride, int index_pos,
                      int num_tokens, int q_heads, int k_heads, int hidden_size, int split_dim, int gpt_geox) {
    rope<__bf16>(
      query, key, cos_sin, cos_sin_stride, index_pos,
      num_tokens, q_heads, k_heads, hidden_size, split_dim, gpt_geox);
}

#ifdef KERNEL_TEST
template <typename T>
__forceinline__ void apply_rotary_embedding(T *arr, T *cos_ptr, T *sin_ptr,
                                int rot_offset, int embed_dim, int gpt_geox) {}


template <>
__forceinline__  void apply_rotary_embedding(float *arr, float *cos_ptr, float *sin_ptr,
                                int rot_offset, int embed_dim, int gpt_geox) {

  int x_index, y_index;
  float cos, sin;
  if (gpt_geox) {
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = cos_ptr[x_index];
    sin = sin_ptr[x_index];
  } else {
    // GPT-J style rotary embedding.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = cos_ptr[x_index / 2];
    sin = sin_ptr[x_index / 2];
  }

  float x = arr[x_index];
  float y = arr[y_index];
  arr[x_index] = x * cos - y * sin;
  arr[y_index] = y * cos + x * sin;
}

template <>
__forceinline__  void apply_rotary_embedding(tops::half *arr,
                                tops::half *cos_ptr,
                                tops::half *sin_ptr, int rot_offset,
                                int embed_dim, int gpt_geox) {
  int x_index, y_index;
  tops::half cos, sin;
  if (gpt_geox) {
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = cos_ptr[x_index];
    sin = sin_ptr[x_index];
  } else {
    // GPT-J style rotary embedding.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = cos_ptr[x_index / 2];
    sin = sin_ptr[x_index / 2];
  }


  tops::half x = arr[x_index];
  tops::half y = arr[y_index];
  float x_fp32 = float(x);
  float y_fp32 = float(y);

  float sin_fp32 = float(sin);
  float cos_fp32 = float(cos);
  float tmp_x = x_fp32 * cos_fp32 - y_fp32 * sin_fp32;
  float tmp_y = y_fp32 * cos_fp32 + x_fp32 * sin_fp32;
  arr[x_index] = tops::half(tmp_x);
  arr[y_index] = tops::half(tmp_y);
}

template <typename T>
void rope_cpu(T *query, T *key, T *cos_sin,
                      int num_tokens, int q_heads, int hidden_size, int gpt_geox) {
    int stride = q_heads * hidden_size;
    for (int i = 0; i < num_tokens; i++) {
      // int32_t pos = positions[i];

      auto query_sub0 = query + i * stride;
      auto key_sub0 = key + i * stride;
      auto cos_sin_cache_sub0 = cos_sin + i * hidden_size;

      int32_t embed_dim = hidden_size / 2;
      auto cos_cache_sub0 = cos_sin_cache_sub0;
      auto sin_cache_sub0 = cos_sin_cache_sub0 + embed_dim;

      int nq = q_heads * embed_dim;

      for (int j = 0; j < nq; j++) {
        int head_idx = j / embed_dim;
        int offset = head_idx * hidden_size;
        int rot_offset = j % embed_dim;
        auto query_sub1 = query_sub0 + offset;
        apply_rotary_embedding(reinterpret_cast<T*>(query_sub1), reinterpret_cast<T*>(cos_cache_sub0), 
                      reinterpret_cast<T*>(sin_cache_sub0),
                                    rot_offset, embed_dim, gpt_geox);
      }

      int nk = q_heads * embed_dim;
      for (int j = 0; j < nk; j++) {
        int head_idx = j / embed_dim;
        int offset = head_idx * hidden_size;
        int rot_offset = j % embed_dim;
        auto key_sub1 = key_sub0 + offset;
        apply_rotary_embedding(reinterpret_cast<T*>(key_sub1), reinterpret_cast<T*>(cos_cache_sub0), 
              reinterpret_cast<T*>(sin_cache_sub0),
                                    rot_offset, embed_dim, gpt_geox);
      }
    }  // loop in num_tokens

}

template <typename T>
void test() {
  int num_tokens = 13;
  int q_heads = 32;
  int hidden_size = 128;
  int qsize = num_tokens * q_heads * hidden_size;
  printf("start the test...\n");
  int gpt_neox = 1;
  int in_size = qsize * sizeof(T);
  T *query = reinterpret_cast<T*>(aligned_alloc(256, in_size));
  T *key = reinterpret_cast<T*>(aligned_alloc(256, in_size));

  for (int j =0; j< qsize; j ++) {
    query[j] = T(0.50); 
  }

  for (int j =0; j< qsize; j ++) {
    key[j] = T(0.25);
  }

  int cos_sin_size = num_tokens * hidden_size;
  float *cos_sin = reinterpret_cast<float*>(aligned_alloc(128, cos_sin_size * sizeof(float)));
  for (int j =0; j< cos_sin_size; j ++) {
    cos_sin[j] = 0.9;
  }
  printf("Calculating gcu results...\n");

  T *query_d = NULL;
  T *key_d = NULL;
  float *cos_sin_d = NULL;

  CHECK(topsMalloc(reinterpret_cast<void **>(&query_d), in_size));
  CHECK(topsMalloc(reinterpret_cast<void **>(&key_d), in_size));
  CHECK(topsMalloc(reinterpret_cast<void **>(&cos_sin_d), cos_sin_size * sizeof(T)));

  CHECK(topsMemcpy(query_d, query, in_size, topsMemcpyHostToDevice));
  CHECK(topsMemcpy(key_d, key, in_size, topsMemcpyHostToDevice));
  CHECK(topsMemcpy(cos_sin_d, cos_sin, cos_sin_size * sizeof(float), topsMemcpyHostToDevice));
  printf("Launching kernel...\n");

  rope_f16<<<1, 12>>>(query_d, key_d, cos_sin_d, num_tokens, q_heads, q_heads, hidden_size, 0, gpt_neox);

  T *query_o = reinterpret_cast<T*>(aligned_alloc(128, in_size));
  T *key_o = reinterpret_cast<T*>(aligned_alloc(128, in_size));

  CHECK(topsMemcpy(query_o, query_d, in_size, topsMemcpyDeviceToHost));
  CHECK(topsMemcpy(key_o, key_d, in_size, topsMemcpyDeviceToHost));
  printf("Calculating cpu results...\n");
  // rope_cpu<T>(query, key, cos_sin, num_tokens, q_heads, hidden_size, gpt_neox);

  for (int i = 0; i < qsize; i++) {
    // if (abs(float(query[i]) - float(query_o[i])) >0.00001 || abs(float(key[i]) - float(key_o[i])) > 0.00001)
    //   fprintf(stderr, "Result dif %d, [%.5f, %.5f]!\n", i, 
    //         abs(float(query[i]) - float(query_o[i])), abs(float(key[i]) - float(key_o[i])));
    // else
    printf("%.5f ", float(query_o[i]));
      // printf("\n query %d [%.5f, %.5f], key [%.5f, %.5f]", i, 
      //       float(query[i]), float(query_o[i]), float(key[i]), float(key_o[i]));
  }

}
#endif

int main() {
#ifdef KERNEL_TEST
  test<__fp16>();
#endif
}