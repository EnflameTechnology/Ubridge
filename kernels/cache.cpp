/**
 * Copyright 2020-2021 Enflame. All Rights Reserved.
 *
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
#include <tops/dte_chain.h>
#include <tops/half.h>
#include <tops/bfloat.h>
#include "tops/tops_runtime.h"
#include "utils/utils.h"
using namespace tops;

template <typename T>
__device__ void reshape_and_cache_kernel(T* key, T* value, int64_t* slot_mapping,
                                         T* key_cache, T* value_cache,
                                         int num_tokens, int num_heads,
                                         int head_size, int num_blocks,
                                         int block_size, int x,
                                         int key_stride0, int value_stride0) {
  int thread_num = GetThreadNum();
  int thread_id = GetThreadIdx();
  
  constexpr int sip_size = VDMEM_VALID_SIZE;
  int head_size_div_x = head_size / x;
  int key_cache_stride0 = num_heads * head_size_div_x * block_size * x;
  int value_cache_stride0 = num_heads * head_size * block_size;

  __local__ __valigned__ char sip_mem[sip_size];
  char* sip_slot_mapping_addr = sip_mem;

  // cast ptr
  int64_t* sip_slot_mapping_ptr = reinterpret_cast<int64_t*>(sip_slot_mapping_addr);

  int tokens_per_sip = CeilDiv(num_tokens, thread_num);
  int tokens_start = thread_id * tokens_per_sip;
  if (tokens_start >= num_tokens) return;

  int tokens_remain = num_tokens - tokens_start;
  int tokens_process =
      tokens_remain > tokens_per_sip ? tokens_per_sip : tokens_remain;
  int tokens_end = tokens_start + tokens_process;

  tops::mdspan key_hbm(tops::Global, key, num_tokens, num_heads, head_size);
  tops::mdspan value_hbm(tops::Global, value, num_tokens, num_heads, head_size);
  tops::mdspan slot_mapping_hbm(tops::Global, slot_mapping, num_tokens);
  tops::mdspan key_cache_hbm(tops::Global, key_cache, num_blocks, num_heads,
                             head_size_div_x, block_size, x);
  tops::mdspan value_cache_hbm(tops::Global, value_cache, num_blocks, num_heads,
                               head_size, block_size);

  tops_dte_ctx_t ctxs_key;
  tops_dte_ctx_t ctxs_value;
  tops_dte_ctx_t ctxs_slot_mapping;
  tops::dte_scope s_key(ctxs_key);
  tops::dte_scope s_value(ctxs_value);
  tops::dte_scope s_slot_mapping(ctxs_slot_mapping);

  tops::event ev_key;
  tops::event ev_value;

  tops::mdspan slot_mapping_hbm_split(tops::Global, slot_mapping + tokens_start,
                                      tokens_process);
  tops::mdspan slot_mapping_sip(tops::Private, sip_slot_mapping_ptr,
                                tokens_process);
  tops::memcpy(ctxs_slot_mapping, slot_mapping_sip, slot_mapping_hbm_split);

  for (int token_idx = tokens_start; token_idx < tokens_end; token_idx++) {
    int slot_idx = (int)sip_slot_mapping_ptr[token_idx - tokens_start];

    if (slot_idx < 0) continue;

    int block_idx = slot_idx / block_size;
    int block_offset = slot_idx % block_size;

    tops::mdspan key_hbm_split(tops::Global, key + (token_idx * key_stride0),
                               num_heads, 1, head_size);
    tops::mdspan key_cache_hbm_split(
        tops::Global, key_cache + (block_idx * key_cache_stride0),
        num_heads, block_size, head_size);
  // if (token_idx > tokens_start) ev_key.wait();
  // ev_key = tops::deslice_async(ctxs_key, key_cache_hbm_split, key_hbm_split,
  //                              {0, 0, block_offset, 0});
    tops::deslice(ctxs_key, key_cache_hbm_split, key_hbm_split,
                  {0, block_offset, 0});

    tops::mdspan value_hbm_split(tops::Global,
                                 value + (token_idx * value_stride0),
                                 num_heads, 1, head_size);
    tops::mdspan value_cache_hbm_split(
        tops::Global, value_cache + (block_idx * value_cache_stride0),
        num_heads, block_size, head_size);
    tops::deslice(ctxs_value, value_cache_hbm_split,
                  value_hbm_split, {0, block_offset, 0});
  }
}

#define CACHE_KERNEL(T, TYPENAME) \
extern "C" __global__ void reshape_and_cache_##TYPENAME( \
  T *key,              \
  T *value,            \
  T *key_cache,       \
  T *value_cache,      \
  int64_t* slot_mapping,  \
  int32_t num_tokens,\
  int32_t num_heads,\
  int32_t head_size,\
  int32_t num_blocks, \
  int32_t block_size,\
  int32_t x,\
  int32_t key_stride,\
  int32_t value_stride) \
{\
  reshape_and_cache_kernel<T>(key, value, slot_mapping, key_cache, value_cache, \
    num_tokens, num_heads, head_size, num_blocks, \
    block_size, x, key_stride, value_stride);\
}

CACHE_KERNEL(__fp16, f16)
CACHE_KERNEL(__bf16, bf16)
CACHE_KERNEL(float, f32)


