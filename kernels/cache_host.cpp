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
#include <acore_op.h>
#include "utils/utils.h"
#include <tops.h>
#include <tops/bfloat.h>
#include <tops/dte_chain.h>
#include <tops/half.h>
#include <tops/topscc_types.h>
#include <tops/tops_runtime.h>
using namespace tops;
#define LOOP_DTE 8

template <typename T>
__global__ void reshape_and_cache_flash_kernel(
        T* key, T* value, int64_t* slot_mapping, T* key_cache, T* value_cache,
        int num_tokens, int num_heads, int head_size, int num_blocks,
        int block_size, int key_stride0, int value_stride0, int block_stride) {
  int thread_id = GetThreadIdx();
  int thread_num = GetThreadNum();
  __local__ __valigned__ char l1_buffer[VDMEM_SIZE];
  char* sip_slot_mapping_addr = l1_buffer;
  int64_t* sip_slot_mapping_ptr = reinterpret_cast<int64_t*>(sip_slot_mapping_addr);

  int tokens_per_thread = CeilDiv(num_tokens, thread_num);
  int tokens_start = thread_id * tokens_per_thread;
  if (tokens_start >= num_tokens) return;
  int tokens_left = num_tokens - tokens_start;
  int tokens_process =
      tokens_left > tokens_per_thread ? tokens_per_thread : tokens_left;
  int tokens_end = tokens_start + tokens_process;
  int total_block = num_blocks * block_size;

  int tokens_loop = (tokens_process / LOOP_DTE) * LOOP_DTE;
  int tokens_loop_end = tokens_start + tokens_loop;
  int tokens_loop_remain = tokens_process - tokens_loop;

  tops::private_dte ctxs_key[LOOP_DTE];
  tops::private_dte ctxs_value[LOOP_DTE];
  for (int i=0; i<LOOP_DTE; i++) {
    ctxs_key[i].init();
    ctxs_value[i].init();
  }
  tops::private_dte ctxs_slot_mapping;
  ctxs_slot_mapping.init();

  tops::event key_ev[LOOP_DTE];
  tops::event value_ev[LOOP_DTE];
  tops::event ev_slot_mapping;

  tops::mdspan slot_mapping_hbm_split(
        tops::Global, slot_mapping + tokens_start, tokens_process);
  tops::mdspan slot_mapping_sip(tops::Private, sip_slot_mapping_ptr,
                                tokens_process);
  ev_slot_mapping = tops::memcpy_async(ctxs_slot_mapping, slot_mapping_sip,
                                                    slot_mapping_hbm_split);
  tops::mdspan key_hbm(tops::Global, key, 1, num_heads, head_size);
  tops::mdspan value_hbm(tops::Global, value, 1, num_heads, head_size);
  tops::mdspan key_cache_hbm(tops::Global, key_cache,
                        block_size, num_heads, head_size);
  tops::mdspan value_cache_hbm(tops::Global, value_cache,
                        block_size, num_heads, head_size);

  for (int i = 0; i < LOOP_DTE; i++) {
    ctxs_key[i].config_deslice(key_cache_hbm, key_hbm, {0, 0, 0});
    ctxs_value[i].config_deslice(value_cache_hbm, value_hbm, {0, 0, 0});
  }

  ev_slot_mapping.wait();
  for (int token_idx = tokens_start; token_idx < tokens_loop_end;
      token_idx += LOOP_DTE) {
    for (int i = 0; i < LOOP_DTE; i++) {
      int sub_token_idx = token_idx + i;
      int slot_idx = (int)sip_slot_mapping_ptr[sub_token_idx - tokens_start];
      if (slot_idx < 0 || slot_idx >= total_block) continue;

      int block_idx = slot_idx / block_size;
      int block_offset = slot_idx % block_size;
      // op_assert(block_idx < num_blocks, "slot mapping value invalid");

      ctxs_key[i].set_src_addr(key + (sub_token_idx * key_stride0));
      ctxs_key[i].set_dst_addr(key_cache + (block_idx * block_stride));
      ctxs_key[i].set_dst_offset(0, block_offset);
      key_ev[i] = ctxs_key[i].trigger();

      ctxs_value[i].set_src_addr(value + (sub_token_idx * value_stride0));
      ctxs_value[i].set_dst_addr(value_cache + (block_idx * block_stride));
      ctxs_value[i].set_dst_offset(0, block_offset);
      value_ev[i] = ctxs_value[i].trigger();
    }
    for (int i = 0; i < LOOP_DTE; i++) {
        int sub_token_idx = token_idx + i;
        int slot_idx = (int)sip_slot_mapping_ptr[sub_token_idx - tokens_start];
        if (slot_idx >= 0 && slot_idx < total_block) {
            key_ev[i].wait();
            value_ev[i].wait();
        }
    }
  }

  if (tokens_loop_remain > 0) {
    for (int token_idx = tokens_loop_end; token_idx < tokens_end;
        token_idx++) {
      int i = token_idx - tokens_loop_end;
      int slot_idx = (int)sip_slot_mapping_ptr[token_idx - tokens_start];

      if (slot_idx < 0 || slot_idx >= total_block) continue;

      int block_idx = slot_idx / block_size;
      int block_offset = slot_idx % block_size;

      // op_assert(block_idx < num_blocks, "slot mapping value invalid");

      ctxs_key[i].set_src_addr(key + (token_idx * key_stride0));
      ctxs_key[i].set_dst_addr(key_cache + (block_idx * block_stride));
      ctxs_key[i].set_dst_offset(0, block_offset);
      key_ev[i] = ctxs_key[i].trigger();

      ctxs_value[i].set_src_addr(value + (token_idx * value_stride0));
      ctxs_value[i].set_dst_addr(value_cache + (block_idx * block_stride));
      ctxs_value[i].set_dst_offset(0, block_offset);
      value_ev[i] = ctxs_value[i].trigger();
    }
    for (int i = 0; i < tokens_loop_remain; i++) {
      int slot_idx = (int)sip_slot_mapping_ptr[tokens_loop + i];
      if (slot_idx >= 0 && slot_idx < total_block) {
          key_ev[i].wait();
          value_ev[i].wait();
      }
    }
  }
}

#define CALL_RESHAPE_AND_CACHE_FLASH_KERNEL(data_type)                        \
  reshape_and_cache_flash_kernel<data_type><<<numBlocks, dimBlocks, 0, stream>>>( \
      reinterpret_cast<data_type*>(dev_key),                                  \
      reinterpret_cast<data_type*>(dev_value),                                \
      reinterpret_cast<int64_t*>(dev_slot_mapping),                           \
      reinterpret_cast<data_type*>(dev_key_cache),                            \
      reinterpret_cast<data_type*>(dev_value_cache), num_tokens, num_heads,   \
      head_size, num_blocks, block_size, key_stride0, value_stride0,          \
      block_stride);

extern "C" void reshape_and_cache_flash_host(
    dim3 numBlocks, dim3 dimBlocks, void* dev_key, void* dev_value,
    void* dev_slot_mapping, void* dev_key_cache, void* dev_value_cache,
    int dataType, int num_tokens, int num_heads, int head_size, int num_blocks,
    int block_size, int key_stride0, int value_stride0, int block_stride,
    int page_stride, int head_stride, topsStream_t stream) {
  if (dataType == TOPSOP_DATA_FP32) {
    CALL_RESHAPE_AND_CACHE_FLASH_KERNEL(float);
  } else if (dataType == TOPSOP_DATA_FP16) {
    CALL_RESHAPE_AND_CACHE_FLASH_KERNEL(tops::half);
  } else if (dataType == TOPSOP_DATA_BF16) {
    CALL_RESHAPE_AND_CACHE_FLASH_KERNEL(tops::bfloat);
  }
}
