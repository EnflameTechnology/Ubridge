/**
 * Copyright 2020-2024 Enflame. All Rights Reserved.
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
#include "utils/moe_kernel_compat.h"
#if defined(__GCU_ARCH__)
#include <tcle.h>
using tcle::FenceType;
#endif
#include "utils/moe_indices_check.h"

using namespace tops;
using namespace cc_kernel;

#ifndef ALIGN_UP_128
#define ALIGN_UP_128(a) (((a) + 127) & (~127))
#endif
#ifndef ALIGN_UP_64
#define ALIGN_UP_64(a) (((a) + 63) & (~63))
#endif

template <typename T>
__global__ void invoke_fused_moe_kernel(
    T* C, T* A, T* B, T* bias, float* topk_weights, int32_t* sorted_ids,
    int32_t* experts_ids, int32_t* num_tokens_post_pad, int32_t M, int32_t K,
    int32_t N, int32_t E, int32_t topk_ids_size, int32_t top_k,
    int32_t block_size, int32_t mul_routed_weight, int32_t bias_flag) {
  int32_t thread_num = GetThreadNum();
  int32_t thread_id = GetThreadIdx();
  int32_t thread_id_in_block = GetThreadIdxInBlock();

  tops::shared_dte shared_A_ctx;
  tops::private_dte private_A_ctx;
  tops::private_dte private_B_ctx;
  tops::private_dte private_C_ctx;
  tops::private_dte private_C_fill_0_ctx;
  tops::private_dte private_bias_ctx;
  tops::private_dte private_ctx;
  extern __shared__ __valigned__ char shared_buf[];
  T* shared_A = reinterpret_cast<T*>(shared_buf);
  float* shared_topk_weight =
      reinterpret_cast<float*>(shared_A + block_size * K);

  __local__ __attribute__((aligned(512))) char local_buf[VDMEM_SIZE];
  const int32_t group_size = 32;
  int32_t group_num = (block_size + group_size - 1) / group_size;
  group_num = group_num == 0 ? 1 : group_num;

  int32_t n_per_thread = (N + thread_num - 1) / thread_num;
  n_per_thread = n_per_thread == 0 ? 1 : n_per_thread;
  n_per_thread = ALIGN_UP_64(n_per_thread);
  int32_t n_start = n_per_thread * thread_id;
  int32_t n_end = std::min(n_start + n_per_thread, N);

  int32_t sub_k = (std::min(K, 5120) & (~63));
  if (sub_k == 0) {
    sub_k = 64;
  }
  constexpr int32_t sub_n = 64;

  // if mul_routed_weight == false, A [M, K], C [M, topk, N]
  // if mul_routed_weight == true, A [M*topk, K], C[M, topk, N]
  int32_t topk_str = (mul_routed_weight == 1) ? 1 : top_k;

  int32_t padded_tokens_num;
  tops::memcpy(private_ctx, mdspan(Private, &padded_tokens_num, 1),
               mdspan(Global, num_tokens_post_pad, 1));
  // support expert parallel
  // some rank's padded_tokens_num is 0, will do nothing
  if (padded_tokens_num == 0) {
    return;
  }
  op_assert(padded_tokens_num >= block_size, "Error: invalid padded_tokens_num! "
      "padded_tokens_num must be >= block_size. "
      "padded_tokens_num = %d < block_size = %d", padded_tokens_num, block_size);
  int32_t num_blocks = padded_tokens_num / block_size;

  int32_t sorted_buf_size = ALIGN_UP_128(block_size);
  int32_t experts_buf_size = ALIGN_UP_128(num_blocks);

  int32_t* local_sorted_ids = reinterpret_cast<int32_t*>(local_buf);
  int32_t* local_experts_ids = local_sorted_ids + sorted_buf_size;
  int32_t* local_workspace = local_experts_ids + experts_buf_size;
  T* local_A = reinterpret_cast<T*>(local_workspace + 512);
  T* local_B = local_A + group_size * sub_k;
  T* local_C = local_B + sub_n * sub_k;
  T* local_bias = local_C + group_size * sub_n * 2;

  tops::memcpy(private_ctx, mdspan(Private, local_experts_ids, num_blocks),
               mdspan(Global, experts_ids, num_blocks));
  indices_check(reinterpret_cast<int *>(local_experts_ids), num_blocks, E, -1);

  // linearcpy weights L3->L2

  if (thread_id_in_block == 0) {
    shared_A_ctx.init();
    shared_A_ctx.config_memcpy(mdspan(Shared, shared_A, K),
                               mdspan(Global, A, K));
    if (mul_routed_weight == 1) {
      tops::memcpy(private_ctx,
                   mdspan(Shared, shared_topk_weight, topk_ids_size),
                   mdspan(Global, topk_weights, topk_ids_size));
    }
  }
  int32_t local_A_shape[2] = {group_size, sub_k};
  int32_t shared_A_shape[2] = {block_size, K};
  private_A_ctx.config_slice(mdspan(Private, local_A, local_A_shape),
                             mdspan(Shared, shared_A, shared_A_shape), {0, 0});
  int32_t local_B_shape[2] = {sub_n, sub_k};
  int32_t global_B_shape[2] = {N, K};
  private_B_ctx.config_slice(mdspan(Private, local_B, local_B_shape),
                                       mdspan(Global, B, global_B_shape),
                                       {0, 0});
  private_C_ctx.config_memcpy(mdspan(Global, C, sub_n),
                              mdspan(Private, local_C, sub_n));
  private_C_fill_0_ctx.config_memset(mdspan(Global, C, sub_n), 0);
  private_bias_ctx.config_memcpy(mdspan(Private, local_bias, sub_n),
                                 mdspan(Global, bias, sub_n));
  int32_t launch_times = 0;
  using u_t = typename UnderlyingType<T>::type;
  for (int block_id = 0; block_id < num_blocks; block_id++) {
    // sorted_ids in this block
    tops::memcpy(
        private_ctx, mdspan(Private, local_sorted_ids, block_size),
        mdspan(Global, sorted_ids + block_id * block_size, block_size));
    // gather lhs L3->L2
    __syncthreads();
    if (thread_id_in_block == 0) {
      for (int token_id = 0; token_id < block_size; token_id++) {
        int32_t sorted_id = local_sorted_ids[token_id];
        if (sorted_id < topk_ids_size && sorted_id >= 0) {
          shared_A_ctx.set_dst_addr(shared_A + token_id * K);
          shared_A_ctx.set_src_addr(A + sorted_id / topk_str * K);
          shared_A_ctx.trigger_and_wait();
        }
      }
    }
    __syncthreads();
    int32_t expert_id = local_experts_ids[block_id];
    if (expert_id == -1) {
      if (n_end > n_start) {
        for (int inner_id = 0; inner_id < block_size; inner_id++) {
          int32_t sorted_id = local_sorted_ids[inner_id];
          if (sorted_id >= topk_ids_size || sorted_id < 0) {
            break;
          }
          int dst_offset = sorted_id * N + n_start;

          private_C_fill_0_ctx.set_dst_addr(C + dst_offset);
          private_C_fill_0_ctx.set_total_size((n_end - n_start) * sizeof(T));
          private_C_fill_0_ctx.trigger_and_wait();
        }
      }
      continue;
    }
    int32_t expert_offset = expert_id * N * K;
    private_B_ctx.set_src_addr(B + expert_offset);
    for (int group_id = 0; group_id < group_num; group_id++) {
      int32_t cur_token_num =
          std::min(group_size, block_size - group_id * group_size);
      for (int n_idx = n_start; n_idx < n_end; n_idx += sub_n) {
        private_B_ctx.set_src_offset(0, n_idx);
        int32_t n_rem = std::min(sub_n, N - n_idx);
        if (bias_flag == 1) {
          private_bias_ctx.set_src_addr(expert_id * N + bias + n_idx);
          private_bias_ctx.set_total_size(n_rem * sizeof(T));
          private_bias_ctx.trigger_and_wait();
        }
        for (int k_idx = 0; k_idx < K; k_idx += sub_k) {
          // slice and auto_pad lhs L2->L1
          private_A_ctx.set_src_offset(0, group_id * group_size);
          private_A_ctx.set_src_offset(1, k_idx);
          private_A_ctx.trigger_and_wait();

          // transpose_and_deslice rhs L3->L1
          private_B_ctx.set_src_offset(1, k_idx);
          private_B_ctx.trigger_and_wait();

          // matmul
          int32_t acc_flag = k_idx == 0 ? 0 : 1;
          int32_t store_flag = (k_idx + sub_k) < K ? 0 : 1;
          if (mul_routed_weight == 1) {
            matmul<32, MK_NK>(reinterpret_cast<float*>(local_C),
                       reinterpret_cast<u_t*>(local_A),
                       reinterpret_cast<u_t*>(local_B),
                       reinterpret_cast<u_t*>(local_bias), local_workspace,
                       sub_k, sub_n, acc_flag, store_flag, bias_flag, 0,
                       launch_times);
          } else {
            matmul<32, MK_NK>(reinterpret_cast<u_t*>(local_C),
                       reinterpret_cast<u_t*>(local_A),
                       reinterpret_cast<u_t*>(local_B),
                       reinterpret_cast<u_t*>(local_bias), local_workspace,
                       sub_k, sub_n, acc_flag, store_flag, bias_flag, 0,
                       launch_times);
          }
          launch_times = 1;
        }
        tcle::fence<FenceType::L1_VDMEM>();
        // mul_routed_weight
        if (mul_routed_weight == 1) {
          for (int i = 0; i < cur_token_num; i++) {
            int32_t token_id = local_sorted_ids[i + group_id * group_size];
            if (token_id < topk_ids_size && token_id >= 0) {
              float weight = shared_topk_weight[token_id];
              mul(reinterpret_cast<u_t*>(local_C) + i * sub_n,
                  reinterpret_cast<float*>(local_C) + i * sub_n, weight, sub_n);
              private_C_ctx.set_dst_addr(C + token_id * N + n_idx);
              private_C_ctx.set_src_addr(local_C + i * sub_n);
              private_C_ctx.set_total_size(n_rem * sizeof(T));
              private_C_ctx.trigger_and_wait();
            }
          }
        } else {
          // scatter dst
          for (int i = 0; i < cur_token_num; i++) {
            int32_t token_id = local_sorted_ids[i + group_id * group_size];
            if (token_id < topk_ids_size && token_id >= 0) {
              private_C_ctx.set_dst_addr(C + token_id * N + n_idx);
              private_C_ctx.set_src_addr(local_C + i * sub_n);
              private_C_ctx.set_total_size(n_rem * sizeof(T));
              private_C_ctx.trigger_and_wait();
            }
          }
        }
      }
    }
  }

  return;
}

template <typename T>
__global__ void invoke_fused_moe_kernel_splitm(
    T* C, T* A, T* B, T* bias, float* topk_weights, int32_t* sorted_ids,
    int32_t* experts_ids, int32_t* num_tokens_post_pad, int32_t M, int32_t K,
    int32_t N, int32_t E, int32_t topk_ids_size, int32_t top_k,
    int32_t block_size, int32_t mul_routed_weight, int32_t bias_flag) {
  tops::private_dte ctx;
  ctx.init();
  tops::private_dte ctx_B;
  ctx_B.init();
  tops::private_dte private_C_fill_0_ctx;
  private_C_fill_0_ctx.init();
  tops::private_dte ctx_A;
  ctx_A.init();
  tops::private_dte ctx_C;
  ctx_C.init();

  // if mul_routed_weight == false, A [M, K], C [M, topk, N]
  // if mul_routed_weight == true, A [M*topk, K], C[M, topk, N]
  int32_t topk_str = (mul_routed_weight == 1) ? 1 : top_k;
  int32_t K_align = AlignUp(K, 64);

  int32_t sub_k_a = K_align;

  int32_t padded_tokens_num;
  tops::memcpy(ctx, mdspan(Private, &padded_tokens_num, 1),
               mdspan(Global, num_tokens_post_pad, 1));
  if (padded_tokens_num == 0) {
    return;
  }
  op_assert(padded_tokens_num >= block_size, "Error: invalid padded_tokens_num! "
      "padded_tokens_num must be >= block_size. "
      "padded_tokens_num = %d < block_size = %d", padded_tokens_num, block_size);
  int32_t num_blocks = padded_tokens_num / block_size;
  int thread_num = GetThreadNum();
  int thread_id = GetThreadIdx();
  int blocks_per_thread = (num_blocks + thread_num - 1) / thread_num;
  int start_block = blocks_per_thread * thread_id;
  int end_block = std::min(start_block + blocks_per_thread, num_blocks);
  const int32_t group_size = 32;
  int32_t group_num = (block_size + group_size - 1) / group_size;
  group_num = group_num == 0 ? 1 : group_num;

  __local__ __attribute__((aligned(512))) char local_buf[VDMEM_SIZE];
  int32_t offset = 0;
  const int32_t bpe4 = sizeof(int32_t);
  int32_t* local_sorted_ids = reinterpret_cast<int32_t*>(local_buf);
  offset += AlignUp(blocks_per_thread * block_size * bpe4, 512);
  int32_t* local_experts_ids = reinterpret_cast<int32_t*>(local_buf + offset);
  offset += AlignUp(blocks_per_thread * bpe4, 512);
  float* local_topk_weights = reinterpret_cast<float*>(local_buf + offset);
  if (mul_routed_weight == 1) {
    offset += AlignUp(topk_ids_size * bpe4, 512);
  }
  T* local_A = reinterpret_cast<T*>(local_buf + offset);
  offset += AlignUp(group_size * sub_k_a * sizeof(T), 512);
  int32_t* local_workspace = reinterpret_cast<int32_t*>(local_buf + offset);
  offset += 2048;

  int32_t sub_n;
  const int32_t sub_n_align = 64;
  int32_t left_buf_size = VDMEM_SIZE - offset;
  sub_n = AlignDown(
      (left_buf_size / sizeof(T)) /
          (sub_k_a + group_size * topk_str * sizeof(float) / sizeof(T)),
      sub_n_align);
  op_assert(sub_n > 0,
            "sub_n=0! left=%d sub_k_a=%d gs=%d ts=%d offset=%d",
            left_buf_size, sub_k_a, group_size, topk_str, offset);

  T* local_B = reinterpret_cast<T*>(local_buf + offset);
  offset += AlignUp(sub_n * sub_k_a * sizeof(T), 512);
  float* local_C = reinterpret_cast<float*>(local_buf + offset);
  offset += AlignUp(group_size * topk_str * sub_n * sizeof(float), 512);
  T* local_bias = reinterpret_cast<T*>(local_buf + offset);

  int32_t local_B_shape[2] = {sub_n, sub_k_a};
  int32_t global_B_shape[2] = {N, K};
  ctx_B.config_slice(mdspan(Private, local_B, local_B_shape),
                     mdspan(Global, B, global_B_shape), {0, 0});

  if (end_block - start_block > 0) {
    tops::memcpy(
        ctx, mdspan(Private, local_experts_ids, end_block - start_block),
        mdspan(Global, experts_ids + start_block, end_block - start_block));
    indices_check(
        reinterpret_cast<int *>(local_experts_ids),
        end_block - start_block, E, -1);
  }

  private_C_fill_0_ctx.config_memset(mdspan(Global, C, N), 0);

  if (mul_routed_weight == 1) {
    tops::memcpy(ctx, mdspan(Private, local_topk_weights, topk_ids_size),
                 mdspan(Global, topk_weights, topk_ids_size));
  }
  int32_t matmul_kernel_launch_times = 0;
  for (int block_id = start_block; block_id < end_block; block_id++) {
    tops::memcpy(
        ctx, mdspan(Private, local_sorted_ids, block_size),
        mdspan(Global, sorted_ids + block_id * block_size, block_size));
    int32_t expert_id = local_experts_ids[block_id - start_block];
    if (expert_id == -1) {
      for (int inner_id = 0; inner_id < block_size; inner_id++) {
        int32_t sorted_id = local_sorted_ids[inner_id];
        if (sorted_id >= topk_ids_size || sorted_id < 0) {
          break;
        }
        int dst_offset = sorted_id * N;

        private_C_fill_0_ctx.set_dst_addr(C + dst_offset);
        private_C_fill_0_ctx.set_total_size(N * sizeof(T));
        private_C_fill_0_ctx.trigger_and_wait();
      }
      continue;
    }
    op_assert(expert_id >= 0 && expert_id < E,
              "fused_moe: expert_id=%d out of range [0,%d)", expert_id, E);
    for (int group_id = 0; group_id < group_num; group_id++) {
      int32_t cur_token_num =
          std::min(group_size, block_size - group_id * group_size);
      if (cur_token_num <= 0) {
        continue;
      }

      u_int32_t pad_size_a = K_align - K;
      u_int32_t pad_low_a[] = {0};
      u_int32_t pad_high_a[] = {pad_size_a};
      u_int32_t pad_mid_a[] = {0};
      ctx_A.config_pad(mdspan(Private, local_A, sub_k_a),
                       mdspan(Global, A, K),
                       pad_low_a, pad_high_a, pad_mid_a, 0);
      for (int inner_id = 0; inner_id < cur_token_num; inner_id++) {
        int32_t real_idx = group_id * group_size + inner_id;
        int32_t sorted_id = local_sorted_ids[real_idx];
        if (sorted_id < topk_ids_size && sorted_id >= 0) {
          ctx_A.set_dst_addr(local_A + inner_id * sub_k_a);
          ctx_A.set_src_addr(A + sorted_id / topk_str * K);
          ctx_A.trigger_and_wait();
        }
      }

      int32_t expert_offset = expert_id * N * K;
      ctx_B.set_src_addr(B + expert_offset);
      for (int n_idx = 0; n_idx < N; n_idx += sub_n) {
        int32_t sub_n_left = std::min(sub_n, N - n_idx);

        // load B
        ctx_B.set_src_offset(0, n_idx);
        ctx_B.trigger_and_wait();

        if (bias_flag == 1) {
          tops::memcpy(
              ctx, mdspan(Private, local_bias, sub_n_left),
              mdspan(Global, bias + expert_id * N + n_idx, sub_n_left));
        }

        // C = dot(A, B)
        using u_t = typename UnderlyingType<T>::type;
        u_t* local_C_ut = reinterpret_cast<u_t*>(local_C);
        if (mul_routed_weight == 1) {
          matmul<32, MK_NK>(local_C, reinterpret_cast<u_t*>(local_A),
                     reinterpret_cast<u_t*>(local_B),
                     reinterpret_cast<u_t*>(local_bias), local_workspace,
                     sub_k_a, sub_n, 0, 1, bias_flag, 0,
                     matmul_kernel_launch_times);
          for (int i = 0; i < cur_token_num; i++) {
            int32_t token_id = local_sorted_ids[i + group_id * group_size];
            float weight = local_topk_weights[token_id];
            if (token_id < topk_ids_size && token_id >= 0) {
              mul(local_C_ut + i * sub_n, local_C + i * sub_n,
                  weight, sub_n);
            }
          }
        } else {
          matmul<32, MK_NK>(local_C_ut, reinterpret_cast<u_t*>(local_A),
                     reinterpret_cast<u_t*>(local_B),
                     reinterpret_cast<u_t*>(local_bias), local_workspace,
                     sub_k_a, sub_n, 0, 1, bias_flag, 0,
                     matmul_kernel_launch_times);
        }
        matmul_kernel_launch_times = 1;
        tcle::fence<FenceType::L1_VDMEM>();

        ctx_C.config_memcpy(mdspan(Private, local_C_ut, sub_n_left),
                            mdspan(Global, C, sub_n_left));
        for (int inner_id = 0; inner_id < cur_token_num; inner_id++) {
          int32_t real_idx = group_id * group_size + inner_id;
          int32_t sorted_id = local_sorted_ids[real_idx];
          if (sorted_id < topk_ids_size && sorted_id >= 0) {
            ctx_C.set_dst_addr(C + sorted_id * N + n_idx);
            ctx_C.set_src_addr(local_C_ut + inner_id * sub_n);
            ctx_C.trigger_and_wait();
          }
        }
      }      // end of N loop
    }        // end of group loop
  }          // end of block loop
  return;
}

template <typename T>
void invoke_fused_moe_host(dim3 numBlocks, dim3 dimBlocks, T* C, T* A,
                           T* B, T* bias, float* topk_weights,
                           int32_t* sorted_ids, int32_t* experts_ids,
                           int32_t* num_tokens_post_pad, int32_t M, int32_t K,
                           int32_t N, int32_t E, int32_t topk_ids_size,
                           int32_t top_k, int32_t block_size,
                           int32_t mul_routed_weight, int32_t bias_flag,
                           topsStream_t stream) {
  int shared_buf_size = block_size * K * sizeof(T);
  if (mul_routed_weight == 1) {
    shared_buf_size += topk_ids_size * sizeof(float);
  }
  int32_t thread_num = numBlocks.x * dimBlocks.x;
  int32_t n_per_thread = (N + thread_num - 1) / thread_num;
  n_per_thread = ALIGN_UP_64(n_per_thread);

  if (K <= 5120 && block_size <= 1024 && top_k <= 64) {
    invoke_fused_moe_kernel_splitm<<<numBlocks, dimBlocks, 0, stream>>>(
        C, A, B, bias, topk_weights, sorted_ids, experts_ids,
        num_tokens_post_pad, M, K, N, E, topk_ids_size, top_k, block_size,
        mul_routed_weight, bias_flag);
    CC_KERNEL_LAUNCH_CHECK();
  } else {
    invoke_fused_moe_kernel<<<numBlocks, dimBlocks, shared_buf_size, stream>>>(
        C, A, B, bias, topk_weights, sorted_ids, experts_ids,
        num_tokens_post_pad, M, K, N, E, topk_ids_size, top_k, block_size,
        mul_routed_weight, bias_flag);
    CC_KERNEL_LAUNCH_CHECK();
  }
  // return topsSuccess;
}

template void invoke_fused_moe_host<tops::half>(
    dim3 numBlocks, dim3 dimBlocks, tops::half* C, tops::half* A, tops::half* B,
    tops::half* bias, float* topk_weights, int32_t* sorted_ids,
    int32_t* experts_ids, int32_t* num_tokens_post_pad, int32_t M, int32_t K,
    int32_t N, int32_t E, int32_t topk_ids_size, int32_t top_k,
    int32_t block_size, int32_t mul_routed_weight, int32_t bias_flag,
    topsStream_t stream);

template void invoke_fused_moe_host<tops::bfloat>(
    dim3 numBlocks, dim3 dimBlocks, tops::bfloat* C, tops::bfloat* A,
    tops::bfloat* B, tops::bfloat* bias, float* topk_weights,
    int32_t* sorted_ids, int32_t* experts_ids, int32_t* num_tokens_post_pad,
    int32_t M, int32_t K, int32_t N, int32_t E, int32_t topk_ids_size,
    int32_t top_k, int32_t block_size, int32_t mul_routed_weight,
    int32_t bias_flag, topsStream_t stream);

extern "C" void invoke_fused_moe_f16(
    __fp16* C, __fp16* A, __fp16* B, __fp16* bias, float* topk_weights,
    int32_t* sorted_ids, int32_t* experts_ids, int32_t* num_tokens_post_pad,
    int32_t M, int32_t K, int32_t N, int32_t E, int32_t topk_ids_size,
    int32_t top_k, int32_t block_size, int32_t mul_routed_weight,
    int32_t bias_flag, void* stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  constexpr int kDeviceBlocks = 2;
  constexpr int kThreadsPerBlock = 12;
  invoke_fused_moe_host<tops::half>(
      dim3(kDeviceBlocks, 1, 1), dim3(kThreadsPerBlock, 1, 1),
      reinterpret_cast<tops::half*>(C), reinterpret_cast<tops::half*>(A),
      reinterpret_cast<tops::half*>(B), reinterpret_cast<tops::half*>(bias),
      topk_weights, sorted_ids, experts_ids, num_tokens_post_pad, M, K, N, E,
      topk_ids_size, top_k, block_size, mul_routed_weight, bias_flag, stream);
}

extern "C" void invoke_fused_moe_bf16(
    __bf16* C, __bf16* A, __bf16* B, __bf16* bias, float* topk_weights,
    int32_t* sorted_ids, int32_t* experts_ids, int32_t* num_tokens_post_pad,
    int32_t M, int32_t K, int32_t N, int32_t E, int32_t topk_ids_size,
    int32_t top_k, int32_t block_size, int32_t mul_routed_weight,
    int32_t bias_flag, void* stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  constexpr int kDeviceBlocks = 2;
  constexpr int kThreadsPerBlock = 12;
  invoke_fused_moe_host<tops::bfloat>(
      dim3(kDeviceBlocks, 1, 1), dim3(kThreadsPerBlock, 1, 1),
      reinterpret_cast<tops::bfloat*>(C), reinterpret_cast<tops::bfloat*>(A),
      reinterpret_cast<tops::bfloat*>(B), reinterpret_cast<tops::bfloat*>(bias),
      topk_weights, sorted_ids, experts_ids, num_tokens_post_pad, M, K, N, E,
      topk_ids_size, top_k, block_size, mul_routed_weight, bias_flag, stream);
}
