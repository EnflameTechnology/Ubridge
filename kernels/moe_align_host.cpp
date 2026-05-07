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
#include <krt/mmu.h>
#include <tops/tops_runtime.h>
#include "utils/moe_kernel_compat.h"
#if defined(__GCU_ARCH__)
#include <tcle.h>
using tcle::FenceType;
#endif
#include "utils/moe_indices_check.h"

using namespace tops;

__global__ void moe_align_block_size_kernel(
    int32_t *sorted_ids, int32_t *experts_ids, int32_t *num_tokens_post_pad,
    int32_t *topk_ids, int32_t *real_token_num, int32_t *expert_map,
    int32_t num_experts, int32_t block_size, int32_t token_num, int32_t topk);
__device__ int moe_align_block_size_scalar_kernel(
    int32_t *sorted_ids, int32_t *experts_ids, int32_t *num_tokens_post_pad,
    int32_t *topk_ids, int32_t *local_buf, int32_t *expert_map,
    int32_t num_experts, int32_t block_size, int32_t token_num, int32_t topk,
    int32_t pad_value);

__global__ void moe_align_block_size_parallel_kernel(
    int32_t *sorted_ids, int32_t *experts_ids, int32_t *num_tokens_post_pad,
    int32_t *topk_ids, int32_t *real_token_num, int32_t *expert_map,
    int32_t num_experts, int32_t block_size, int32_t token_num, int32_t topk);

void moe_align_block_size_host(
    dim3 numBlocks, dim3 dimBlocks, int32_t *sorted_ids, int32_t *experts_ids,
    int32_t *num_tokens_post_pad, int32_t *topk_ids, int32_t *real_token_num,
    int32_t *expert_map, int32_t num_experts, int32_t block_size,
    int32_t token_num, int32_t topk, topsStream_t stream) {
  int32_t M = token_num * topk;
  int32_t max_padded_num = M + num_experts * (block_size - 1);
  int32_t max_block_num = max_padded_num / block_size;

  int32_t l1_used_num = 128; // num_tokens_post_pad
  // experts ids num
  l1_used_num += moe_host_align_up(M / block_size + 1 + num_experts, 128);
  l1_used_num += moe_host_align_up(M + 15 * 64, 128); // topk ids num
  l1_used_num += moe_host_align_up(num_experts, 128); // experts count
  l1_used_num += moe_host_align_up(num_experts * 2 + 1, 128); // cumsum count
  if (expert_map) {
    l1_used_num += moe_host_align_up(num_experts, 128);
  }

  if (token_num * topk <= 4096 && (block_size & (block_size - 1)) == 0 &&
      ((uint32_t)l1_used_num <= VDMEM_SIZE / sizeof(int32_t))) {
    int shared_buf_size = max_padded_num * sizeof(int32_t);
    moe_align_block_size_kernel<<<1, 1, shared_buf_size, stream>>>(
        sorted_ids, experts_ids, num_tokens_post_pad, topk_ids, real_token_num,
        expert_map, num_experts, block_size, token_num, topk);
    CC_KERNEL_LAUNCH_CHECK();
  } else {
    int32_t thread_num_in_block = dimBlocks.x * dimBlocks.y * dimBlocks.z;
    int32_t shared_buf_size =
        (moe_host_align_up(num_experts, 128) * thread_num_in_block + max_padded_num) *
        sizeof(int32_t);
    int32_t topk_ids_tile_size =
        (token_num * topk + thread_num_in_block - 1) / thread_num_in_block;
    int32_t topk_ids_l1_buf_size = moe_host_align_up(topk_ids_tile_size + 15 * 64, 128);
    int32_t num_experts_aligned = moe_host_align_up(num_experts, 128);
    int32_t l1_used_size =
        (topk_ids_l1_buf_size + num_experts_aligned * 5 + max_block_num) *
        sizeof(int32_t);

    if (l1_used_size > VDMEM_SIZE) {
      shared_buf_size += max_block_num * sizeof(int32_t);
    }
    moe_align_block_size_parallel_kernel<<<1, dimBlocks, shared_buf_size,
                                           stream>>>(
        sorted_ids, experts_ids, num_tokens_post_pad, topk_ids, real_token_num,
        expert_map, num_experts, block_size, token_num, topk);
    CC_KERNEL_LAUNCH_CHECK();
  }
}

__device__ void check_ids(int32_t *topk_ids, int32_t *expert_map,
                          int32_t num_experts, int32_t token_num,
                          bool has_expert_map = false) {
  using vt = typename tcle::simple_altivector<int32_t>::VT;
  vt v0 = (vt)(0);
  vt v1 = (vt)(1);
  vt lower_bound = (vt)(-1);
  vt vnum_experts = (vt)(num_experts);
  vt vcnt = v0;
  vt* vptr = reinterpret_cast<vt*>(topk_ids);
  constexpr int vlen = tcle::altivector_step<vt>();
  int aligned_num = AlignDown(token_num, vlen);
  for (int i = 0; i < aligned_num; i += vlen) {
    vt vtopk_ids = *vptr++;
    vt vcmp = tcle::vsel(vtopk_ids >= vnum_experts || vtopk_ids < v0, v1, v0);
    vcnt += vcmp;
  }
  int32_t cnt = tcle::redsum<0>(vcnt)[0];
  for (int i = aligned_num; i < token_num; i++) {
    int32_t topk_id = topk_ids[i];
    if (topk_id >= num_experts || topk_id < 0) {
      cnt++;
    }
  }
  if (cnt != 0) {
    for (int i = 0; i < token_num; i++) {
      int32_t topk_id = topk_ids[i];
      if (topk_id >= num_experts || topk_id < 0) {
        op_assert(0, "topk_ids[%d] = %d, out of range[0, %d] !", i, topk_id,
                  num_experts - 1);
      }
    }
  }

  if (has_expert_map) {
    vt vcnt = v0;
    vt *expert_vptr = reinterpret_cast<vt *>(expert_map);
    int num_experts_aligned = AlignDown(num_experts, vlen);
    for (int i = 0; i < num_experts_aligned; i += vlen) {
      vt vexpert_ids = *expert_vptr++;
      vt vcmp = tcle::vsel(
          vexpert_ids >= vnum_experts || vexpert_ids < lower_bound, v1, v0);
      vcnt += vcmp;
    }
    cnt = tcle::redsum<0>(vcnt)[0];

    for (int i = num_experts_aligned; i < num_experts; i++) {
      int32_t expert_id = expert_map[i];
      if (expert_id >= num_experts || expert_id < -1) {
        cnt++;
      }
    }
    if (cnt != 0) {
      for (int i = 0; i < num_experts; i++) {
        int32_t expert_id = expert_map[i];
        if (expert_id >= num_experts || expert_id < -1) {
          op_assert(0, "expert_map[%d] = %d, out of range[-1, %d] !", i,
                    expert_id, num_experts - 1);
        }
      }
    }
  }
}

__device__ void cnt_experts(int32_t *expert_cnt_buf, /* [expert_num] */
                            int32_t *topk_ids,       /* [token_num] */
                            int32_t *idx_table,      /* [expert_num + 1] */
                            int32_t expert_id, int32_t num_experts,
                            int32_t token_num) {
#if 1
  // vector version
  using da_type_u32 = typename scalar_to_vector<int32_t, 64>::type;
  da_type_u32 vbpe = vbroadcast<da_type_u32>(4);
  da_type_u32 vids, voff;

  auto ids_ptr = simple_leaptr<da_type_u32>(topk_ids);
  auto off_st_ptr = simple_leaptr<da_type_u32>(topk_ids);
  auto off_ld_ptr = simple_leaptr<da_type_u32>(topk_ids);

  // index to offset
  if (expert_id == 0) {
    for (int i = 0; i < token_num + 15 * 64; i += 64) {
      vids = ids_ptr.load();
      voff = vids * vbpe;
      off_st_ptr.store(voff);
    }
  }

  // accumulate the token num for each expert
  auto exp_id_ptr = reinterpret_cast<__DTU_INTRIN_AS__ char *>(idx_table);

  for (int i = 0; i < token_num; i += 15 * 64) {
    auto vcnt = vzero<da_type_u32>();
    for (int j = 0; j < 15; j++) {
      auto voff = off_ld_ptr.load();
      auto vids = __dtu_m_vldxda_dual_u32(exp_id_ptr, voff);
      vcnt += vids;
    }

    da_type_u32 vmask = vbroadcast<da_type_u32>(15);
    for (int j = 0; j < 8; j++) {
      int32_t cnt = vreduce_sum<int32_t>(vand(vshri(vcnt, j * 4), vmask));
      expert_cnt_buf[j] += cnt;
    }
  }
#else
  // scalar version
  for (int i = 0; i < token_num; i++) {
    int32_t expert_id = topk_ids[i];
    expert_cnt_buf[expert_id]++;
  }
#endif
}

__device__ void
experts_cnt_to_padded_cumsum(int32_t *cumsum_buf,     /* [expert_num + 1] */
                             int32_t *expert_cnt_buf, /* [expert_num] */
                             int32_t num_experts, int32_t block_size,
                             int32_t token_num) {
  cumsum_buf[0] = 0;
  int32_t cnt_tar_base =
      static_cast<int32_t>(reinterpret_cast<long>(expert_cnt_buf)) >> 6;
  int32_t cumsum_tar_base =
      static_cast<int32_t>(reinterpret_cast<long>(cumsum_buf)) >> 6;
  tar_t cnt_targ = __dtu_s_movsr2targ((cnt_tar_base << 16) | cnt_tar_base);
  tar_t cumsum_targ =
      __dtu_s_movsr2targ((cumsum_tar_base << 16) | cumsum_tar_base);
  tar_t cnt_tari = __dtu_s_movsr2tari(0x00010001, cnt_targ);
  tar_t cumsum_tari = __dtu_s_movsr2tari(0x00010001, cumsum_targ);
  v64i8 vcnt[16];
  v64i8 vdec_block_size = __dtu_s_movr2vr_dup(block_size - 1);
  v64i8 vneg_block_size = __dtu_s_movr2vr_dup(-block_size);
  v64i8 vcumsum = __dtu_s_movr2vr_dup(0);
  int lpr = __dtu_movs_lpr();
  __dtu_c_movsr2lpr(1);
  for (int i = 0; i < num_experts; i += 16) {
    vcnt[0] = __dtu_s_tvld_itar(cnt_targ, cnt_tari);
    vcnt[0] = __dtu_v_vadd_a_s32(vcnt[0], vdec_block_size);
    vcnt[0] = __dtu_v_vand_a_u32(vcnt[0], vneg_block_size);
    vcnt[1] = __dtu_l_movsfti_l_b(vcnt[0], 4);
    vcnt[2] = __dtu_l_movsfti_l_b(vcnt[0], 8);
    vcnt[3] = __dtu_l_movsfti_l_b(vcnt[0], 12);
    vcnt[4] = __dtu_l_movsfti_l_qw(vcnt[0], 1);
    vcnt[5] = __dtu_l_movsfti_l_b(vcnt[4], 4);
    vcnt[6] = __dtu_l_movsfti_l_b(vcnt[4], 8);
    vcnt[7] = __dtu_l_movsfti_l_b(vcnt[4], 12);
    vcnt[8] = __dtu_l_movsfti_l_qw(vcnt[4], 1);
    vcnt[9] = __dtu_l_movsfti_l_b(vcnt[8], 4);
    vcnt[10] = __dtu_l_movsfti_l_b(vcnt[8], 8);
    vcnt[11] = __dtu_l_movsfti_l_b(vcnt[8], 12);
    vcnt[12] = __dtu_l_movsfti_l_qw(vcnt[8], 1);
    vcnt[13] = __dtu_l_movsfti_l_b(vcnt[12], 4);
    vcnt[14] = __dtu_l_movsfti_l_b(vcnt[12], 8);
    vcnt[15] = __dtu_l_movsfti_l_b(vcnt[12], 12);

    vcumsum = __dtu_s_lpr_movr2vr_dup(cumsum_buf[i - 1]);
    __dtu_c_movsr2lpr(0);

    vcnt[0] = __dtu_v_vadd_a_s32(vcnt[0], vcnt[1]);
    vcnt[2] = __dtu_v_vadd_a_s32(vcnt[2], vcnt[3]);
    vcnt[4] = __dtu_v_vadd_a_s32(vcnt[4], vcnt[5]);
    vcnt[6] = __dtu_v_vadd_a_s32(vcnt[6], vcnt[7]);
    vcnt[8] = __dtu_v_vadd_a_s32(vcnt[8], vcnt[9]);
    vcnt[10] = __dtu_v_vadd_a_s32(vcnt[10], vcnt[11]);
    vcnt[12] = __dtu_v_vadd_a_s32(vcnt[12], vcnt[13]);
    vcnt[14] = __dtu_v_vadd_a_s32(vcnt[14], vcnt[15]);

    vcnt[0] = __dtu_v_vadd_a_s32(vcnt[0], vcnt[2]);
    vcnt[4] = __dtu_v_vadd_a_s32(vcnt[4], vcnt[6]);
    vcnt[8] = __dtu_v_vadd_a_s32(vcnt[8], vcnt[10]);
    vcnt[12] = __dtu_v_vadd_a_s32(vcnt[12], vcnt[14]);

    vcnt[0] = __dtu_v_vadd_a_s32(vcnt[0], vcnt[4]);
    vcnt[8] = __dtu_v_vadd_a_s32(vcnt[8], vcnt[12]);

    vcnt[0] = __dtu_v_vadd_a_s32(vcnt[0], vcnt[8]);
    vcumsum = __dtu_v_vadd_a_s32(vcumsum, vcnt[0]);
    __dtu_m_tvst_itar(vcumsum, cumsum_targ, cumsum_tari);
  }
  __dtu_c_movsr2lpr(lpr);
}

__device__ void gen_sorted_ids(int32_t *sorted_ids,  /* [token_num] */
                               int32_t *experts_ids, /* [token_num] */
                               int32_t *cumsum_buf,  /* [expert_num] */
                               int32_t *topk_ids,    /* [token_num] */
                               int32_t *block_off,   /* [expert_num] */
                               int32_t *expert_map,  /* [expert_num] */
                               int32_t token_num, int32_t block_size,
                               int32_t num_experts,
                               bool has_expert_map = false) {
  cumsum_buf[0] = 0;
  int sft_val = 31 - __dtu_l_bc_lz_a_s32(block_size);

  for (int32_t i = 1; i < num_experts + 1; i++) {
    int32_t block_num = (cumsum_buf[i] - cumsum_buf[i - 1]) >> sft_val;
    for (int32_t j = 0; j < block_num; j++) {
      *experts_ids++ = has_expert_map ? expert_map[i - 1] : i - 1;
    }
  }

  for (int i = 0; i < num_experts; i++) {
    block_off[i] = 0;
  }

  char *cumsum_ptr = reinterpret_cast<char *>(cumsum_buf);
  char *block_off_ptr = reinterpret_cast<char *>(block_off);
  for (int32_t i = 0; i < token_num; i++) {
    int32_t idx = topk_ids[i];
    int32_t cumsum_val = *reinterpret_cast<int32_t *>(cumsum_ptr + idx);
    int32_t block_off_val = *reinterpret_cast<int32_t *>(block_off_ptr + idx);
    sorted_ids[cumsum_val + block_off_val] = i;
    *reinterpret_cast<int32_t *>(block_off_ptr + idx) += 1;
  }
  __dtu_l_movs_barrier_fence(0xc);
}

__global__ void moe_align_block_size_parallel_kernel(
    int32_t *sorted_ids, int32_t *experts_ids, int32_t *num_tokens_post_pad,
    int32_t *topk_ids, int32_t *real_token_num, int32_t *expert_map,
    int32_t num_experts, int32_t block_size, int32_t token_num, int32_t topk) {
  const int kMemSize = VDMEM_SIZE / sizeof(int32_t);
  __local__ __valigned__ int32_t local_buf[kMemSize];
  extern __shared__ __valigned__ int32_t shared_buf[];
  int32_t thread_num_each_block = GetThreadNumEachBlock();
  int32_t thread_id_in_block = GetThreadIdxInBlock();
  int32_t pad_value = token_num * topk;
  int32_t max_padded_num = pad_value + num_experts * (block_size - 1);
  int32_t max_block_num = max_padded_num / block_size;
  if (real_token_num != nullptr) {
    tops::private_dte real_token_num_ctx;
    real_token_num_ctx.init();
    int32_t real_token_num_val = 0;
    tops::memcpy(real_token_num_ctx, mdspan(Private, &real_token_num_val, 1),
                 mdspan(Global, real_token_num, 1));
    op_assert(real_token_num_val >= 0, "real_token_num cannot be negative! "
              "real_token_num=%d", real_token_num_val);
    op_assert(real_token_num_val <= token_num,
              "real_token_num exceeds token_num! "
              "real_token_num=%d, token_num=%d",
              real_token_num_val, token_num);
    token_num = real_token_num_val;
    tops::private_dte ctx_set_sorted_ids;
    ctx_set_sorted_ids.init();
    tops::private_dte ctx_set_experts_ids;
    ctx_set_experts_ids.init();
    tops::event event_set_experts_ids;
    if (thread_id_in_block == 0) {
      ctx_set_experts_ids.config_memset(
          mdspan(Shared, experts_ids, max_block_num), 0);
      event_set_experts_ids = ctx_set_experts_ids.trigger();
    }
    int32_t size_per_thread =
        (max_padded_num + thread_num_each_block - 1) / thread_num_each_block;
    int32_t set_from = thread_id_in_block * size_per_thread;
    int32_t rem_size = max_padded_num - set_from;
    int32_t set_size = rem_size < size_per_thread ? rem_size : size_per_thread;
    tops::private_dte ctx_set;
    ctx_set.init();
    if (rem_size > 0) {
      tops::memset(ctx_set_sorted_ids,
                   mdspan(Global, sorted_ids + set_from, set_size), pad_value);
    }
    if (thread_id_in_block == 0) {
      tops::wait(event_set_experts_ids);
    }
  }
  if (token_num <= 0) {
    mapped_ptr num_padded_mapped = tops::map_mem_m(num_tokens_post_pad, 1);
    int32_t *num_padded_ptr = reinterpret_cast<int32_t *>(num_padded_mapped);
    *num_padded_ptr = 0;
    tops::unmap_mem_m(num_padded_mapped, 1);
    tcle::fence<FenceType::L2_MEM>();
    return;
  } else if (token_num * topk <= 1024) {
    int status = moe_align_block_size_scalar_kernel(
        sorted_ids, experts_ids, num_tokens_post_pad, topk_ids, local_buf,
        expert_map, num_experts, block_size, token_num, topk, pad_value);
    if (status == 0) {
      return;
    }
  }
  int32_t topk_ids_total = token_num * topk;
  int32_t num_experts_aligned = AlignUp(num_experts, 128);
  int32_t topk_ids_per_thread =
      (topk_ids_total + thread_num_each_block - 1) / thread_num_each_block;
  int32_t topk_ids_offset = topk_ids_per_thread * thread_id_in_block;
  int32_t topk_ids_rem = topk_ids_total - topk_ids_offset;
  int32_t topk_ids_num =
      topk_ids_per_thread < topk_ids_rem ? topk_ids_per_thread : topk_ids_rem;

  int32_t *shared_sorted_ids =
      shared_buf + thread_num_each_block * num_experts_aligned;
  int32_t *shared_experts_ids = shared_sorted_ids + max_padded_num;
  using vtype = tcle::simple_altivector<unsigned int>::VT;
  const int vlen = tcle::altivector_step<vtype>();
  int32_t offset = AlignUp(topk_ids_num + 15 * 64, vlen);
  int32_t *local_topk_ids = local_buf;
  int32_t *local_experts_off = local_topk_ids + offset;
  int32_t *local_experts_cnt = local_experts_off + num_experts_aligned;
  int32_t *local_const_buf = local_experts_cnt + num_experts_aligned;
  int32_t *local_expert_map = local_const_buf + num_experts_aligned * 3;
  int32_t *local_experts_ids = local_expert_map;
  bool use_shared_experts_ids = false;
  int32_t use_l1_buf_size = (offset + num_experts_aligned * 5 + max_block_num) *
      sizeof(int32_t);
  if (use_l1_buf_size > VDMEM_SIZE) {
    use_shared_experts_ids = true;
  }

  tops::private_dte ctx_expert_map;
  ctx_expert_map.init();
  tops::event event_expert_map;
  if (expert_map) {
    local_experts_ids = local_expert_map + num_experts_aligned;
    ctx_expert_map.config_memcpy(mdspan(Private, local_expert_map, num_experts),
                                 mdspan(Global, expert_map, num_experts));
    event_expert_map = ctx_expert_map.trigger();
  }

  tops::private_dte dte_ctx1;
  dte_ctx1.init();
  tops::event event1;
  tops::private_dte dte_ctx0;
  dte_ctx0.init();
  tops::event event0;

  if (thread_id_in_block == 0) {
    dte_ctx1.config_memset(mdspan(Shared, shared_sorted_ids, max_padded_num),
                           pad_value);
    event1 = dte_ctx1.trigger();
  }

  if (topk_ids_num > 0) {
    u_int32_t pad_low[] = {0};
    u_int32_t pad_high[] = {15 * 64};
    u_int32_t pad_mid[] = {0};
    dte_ctx0.config_pad(
        mdspan(Private, local_topk_ids, topk_ids_num + 15 * 64),
        mdspan(Global, topk_ids + topk_ids_offset, topk_ids_num), pad_low,
        pad_high, pad_mid, num_experts);
    dte_ctx0.trigger_and_wait();
    bool has_expert_map = (expert_map != nullptr);
    if (expert_map) {
      tops::wait(event_expert_map);
    }
    check_ids(local_topk_ids, local_expert_map, num_experts, topk_ids_num,
              has_expert_map);

    // stage1: counting the token_num for each expert
    vtype *vptr = reinterpret_cast<vtype *>(local_experts_off);
    vtype v0 = (vtype)(0);
    for (int i = 0; i < num_experts_aligned; i += vlen) {
      *vptr++ = v0; // expert_off
      *vptr++ = v0; // expert_cnt
      *vptr++ = v0; // const_buf
      *vptr++ = v0; // const_buf
      *vptr++ = v0; // cumsum_buf
    }
    int32_t *expert_id_cfg = local_const_buf + num_experts;
    expert_id_cfg[0] = 0x1;
    expert_id_cfg[1] = 0x10;
    expert_id_cfg[2] = 0x100;
    expert_id_cfg[3] = 0x1000;
    expert_id_cfg[4] = 0x10000;
    expert_id_cfg[5] = 0x100000;
    expert_id_cfg[6] = 0x1000000;
    expert_id_cfg[7] = 0x10000000;
    for (int i = 0; i < num_experts; i += 8) {
      int32_t *expert_cnt_buf = local_experts_cnt + i;
      int32_t *expert_id_buf = local_const_buf + num_experts - i;
      cnt_experts(expert_cnt_buf, local_topk_ids, expert_id_buf, i, num_experts,
                  topk_ids_num);
    }
    __dtu_l_movs_barrier_fence(0x3);
    dte_ctx0.config_memcpy(
        mdspan(Shared, shared_buf + thread_id_in_block * num_experts_aligned,
               num_experts_aligned),
        mdspan(Private, local_experts_cnt, num_experts_aligned));
    event0 = dte_ctx0.trigger();
    // recover the topk_ids
    vtype *topk_ids_vptr = reinterpret_cast<vtype *>(local_topk_ids);
    vtype v2 = (vtype)(2);
    for (int i = 0; i < topk_ids_num; i += vlen) {
      auto vtopk_ids = *topk_ids_vptr;
      vtopk_ids = vtopk_ids >> v2;
      *topk_ids_vptr++ = vtopk_ids;
    }
    tops::wait(event0);
  } else {
    dte_ctx0.config_memset(
        mdspan(Shared, shared_buf + thread_id_in_block * num_experts_aligned,
               num_experts_aligned),
        0);
    dte_ctx0.trigger_and_wait();
  }
  if (thread_id_in_block == 0) {
    tops::wait(event1);
  }
  __syncthreads();

  if (topk_ids_num > 0) {
    // stage2: merge the expert_cnt of each thread
    vtype *expert_cnt_vptr = reinterpret_cast<vtype *>(local_experts_cnt);
    vtype *expert_off_vptr = reinterpret_cast<vtype *>(local_experts_off);
    for (int i = 0; i < num_experts; i += vlen) {
      vtype *shared_buf_vptr = reinterpret_cast<vtype *>(shared_buf + i);
      vtype vexpert_cnt = (vtype)(0);
      vtype vexpert_off = (vtype)(0);
      for (int j = 0; j < thread_num_each_block; j++) {
        vexpert_cnt += *shared_buf_vptr;
        if (j < thread_id_in_block) {
          vexpert_off += *shared_buf_vptr;
        }
        shared_buf_vptr += (num_experts_aligned / vlen);
      }
      *expert_cnt_vptr++ = vexpert_cnt;
      *expert_off_vptr++ = vexpert_off;
    }
    // stage3: caculating the offset for each expert of each thread
    int32_t *cumsum_buf = local_const_buf; // reuse the const buffer

    int32_t *expert_ids_ptr = local_experts_ids;
    if (use_shared_experts_ids) {
      expert_ids_ptr = shared_experts_ids;
    }
    int32_t padded_size = 0;
    for (int i = 0; i < num_experts; i++) {
      int32_t expert_cnt = local_experts_cnt[i];
      int32_t expert_padded_size = 0;
      for (int j = 0; j < expert_cnt; j += block_size) {
        expert_padded_size += block_size;
      }
      if (thread_id_in_block == 0) {
        if (expert_map) {
          for (int j = 0; j < expert_cnt; j += block_size) {
            *expert_ids_ptr++ = local_expert_map[i];
          }
        } else {
          for (int j = 0; j < expert_cnt; j += block_size) {
            *expert_ids_ptr++ = i;
          }
        }
      }
      cumsum_buf[i] = padded_size;
      padded_size += expert_padded_size;
    }
    int32_t block_cnt = padded_size / block_size;
    if (thread_id_in_block == 0) {
      dte_ctx0.config_memcpy(mdspan(Global, num_tokens_post_pad, 1),
                             mdspan(Private, &padded_size, 1));
      event0 = dte_ctx0.trigger();

      if (use_shared_experts_ids) {
        dte_ctx1.config_slice(mdspan(experts_ids, max_block_num),
                              mdspan(shared_experts_ids, block_cnt), {0});
      } else {
        dte_ctx1.config_slice(mdspan(experts_ids, max_block_num),
                              mdspan(local_experts_ids, block_cnt), {0});
      }
      event1 = dte_ctx1.trigger();
    }

    for (int i = 0; i < num_experts; i++) {
      local_experts_off[i] = local_experts_off[i] + cumsum_buf[i];
    }

    for (int i = 0; i < topk_ids_num; i++) {
      int32_t topk_id = local_topk_ids[i];
      int32_t expert_off = local_experts_off[topk_id];
      shared_sorted_ids[expert_off] = i + topk_ids_offset;
      local_experts_off[topk_id] = expert_off + 1;
    }
    if (thread_id_in_block == 0) {
      tops::wait(event0);
      tops::wait(event1);
    }
  }
  __syncthreads();
  tcle::fence<FenceType::L2_MEM>();
  if (thread_id_in_block == 0) {
    tops::memcpy(dte_ctx0, mdspan(Global, sorted_ids, max_padded_num),
                 mdspan(Shared, shared_sorted_ids, max_padded_num));
  }
}

__device__ int moe_align_block_size_scalar_kernel(
    int32_t *sorted_ids, int32_t *experts_ids, int32_t *num_tokens_post_pad,
    int32_t *topk_ids, int32_t *local_buf, int32_t *expert_map,
    int32_t num_experts, int32_t block_size, int32_t token_num, int32_t topk,
    int32_t pad_value) {
  int32_t sorted_ids_buf_size = pad_value + num_experts * (block_size - 1);
  int32_t expert_ids_buf_size = sorted_ids_buf_size / block_size;

  using vtype = tcle::simple_altivector<unsigned int>::VT;
  const int vlength = tcle::altivector_step<vtype>();

  int max_padded_num_in_block = AlignUp(token_num, block_size);
  max_padded_num_in_block = AlignUp(max_padded_num_in_block, vlength);
  int max_padded_num = token_num * topk + num_experts * (block_size - 1);
  int max_block_num = max_padded_num / block_size;

  int32_t offset = 0;
  int32_t *local_experts_cnt = local_buf;
  offset += AlignUp(num_experts, vlength);
  int32_t *local_sorted_ids = local_buf + offset;
  offset += max_padded_num_in_block * num_experts;
  int32_t *local_experts_ids = local_buf + offset;
  offset += AlignUp(max_block_num, vlength);
  int32_t *local_topk_ids = local_buf + offset;
  offset += AlignUp(token_num * topk, vlength);
  int32_t *local_expert_map = local_buf + offset;
  if (expert_map) {
    offset += AlignUp(num_experts, vlength);
  }

  const int kMemSize = VDMEM_SIZE / sizeof(int32_t);
  if (offset > kMemSize) {
    return -1;
  }

  tops::private_dte dte_ctx0;
  tops::private_dte dte_ctx1;
  tops::private_dte dte_ctx2;
  dte_ctx0.init();
  dte_ctx1.init();
  dte_ctx2.init();
  tops::event event0;
  tops::event event2;

  dte_ctx0.config_memcpy(mdspan(Private, local_topk_ids, token_num * topk),
                         mdspan(Global, topk_ids, token_num * topk));
  event0 = dte_ctx0.trigger();

  if (expert_map) {
    dte_ctx2.config_memcpy(mdspan(Private, local_expert_map, num_experts),
                           mdspan(Global, expert_map, num_experts));
    event2 = dte_ctx2.trigger();
  }

  // init local_sorted_ids
  vtype vpad_value = (vtype)(pad_value);
  vtype *vptr = reinterpret_cast<vtype *>(local_sorted_ids);
  for (int i = 0; i < max_padded_num_in_block * num_experts / vlength; i++) {
    vptr[i] = vpad_value;
  }

  // init offset of each expert sorted_ids_buf
  vtype vexperts_off = tcle::mid<vtype, 0>(0) *
                       (vtype)(max_padded_num_in_block * sizeof(int32_t));
  vtype vsorted_ids_base =
      (vtype)(static_cast<int32_t>(reinterpret_cast<long>(local_sorted_ids)));
  vtype vexperts_base = vsorted_ids_base + vexperts_off;
  vtype *experts_cnt_ptr = reinterpret_cast<vtype *>(local_experts_cnt);
  for (int i = 0; i < (num_experts + vlength - 1) / vlength; i++) {
    experts_cnt_ptr[i] = vexperts_base;
    vexperts_base += (vtype)(max_padded_num_in_block * sizeof(int32_t) * vlength);
  }
  // *experts_cnt_ptr = vexperts_base;
  tops::wait(event0);
  if (expert_map) {
    tops::wait(event2);
  }
  bool has_expert_map = (expert_map != nullptr);
  check_ids(local_topk_ids, local_expert_map, num_experts, token_num * topk,
            has_expert_map);

  // topk_ids to sorted_ids of each expert
  for (int i = 0; i < token_num * topk; i++) {
    int32_t topk_id = local_topk_ids[i];
    *reinterpret_cast<int32_t *>(local_experts_cnt[topk_id]) = i;
    local_experts_cnt[topk_id] += sizeof(int32_t); // offset to next
  }

  // offset to num_sorted_ids of each expert
  vexperts_base = vsorted_ids_base + vexperts_off;
  for (int i = 0; i < (num_experts + vlength - 1) / vlength; i++) {
    vexperts_off = experts_cnt_ptr[i];
    vexperts_off -= vexperts_base;
    vexperts_off >>= (vtype)(2); // sizeof(int32_t)
    experts_cnt_ptr[i] = vexperts_off;
    vexperts_base += (vtype)(max_padded_num_in_block * sizeof(int32_t) * vlength);
  }

  // gather the sorted_ids of each expert to local_sorted_ids
  dte_ctx0.config_memcpy(
      mdspan(Global, sorted_ids, max_padded_num_in_block),
      mdspan(Private, local_sorted_ids, max_padded_num_in_block));
  dte_ctx1.config_memcpy(
      mdspan(Global, sorted_ids, max_padded_num_in_block),
      mdspan(Private, local_sorted_ids, max_padded_num_in_block));
  tops::event memcpy_ev0;
  tops::event memcpy_ev1;
  int32_t pp_flag = 0;
  int32_t triggered = 0;
  int32_t block_cnt = 0;
  for (int i = 0; i < num_experts; i++) {
    int32_t num_sorted_ids = local_experts_cnt[i];
    if (num_sorted_ids > 0) {
      int32_t block_num = (num_sorted_ids + block_size - 1) / block_size;
      for (int j = block_cnt; j < block_cnt + block_num; j++) {
        if (expert_map) {
          local_experts_ids[j] = local_expert_map[i];
        } else {
          local_experts_ids[j] = i;
        }
      }

      if (pp_flag == 0) {
        if (triggered > 0) {
          tops::wait(memcpy_ev0);
        }
        dte_ctx0.set_dst_addr(sorted_ids + block_cnt * block_size);
        dte_ctx0.set_src_addr(local_sorted_ids + i * max_padded_num_in_block);
        dte_ctx0.set_total_size(block_num * block_size * sizeof(int32_t));
        memcpy_ev0 = dte_ctx0.trigger();
        pp_flag = 1;
      } else {
        if (triggered > 1) {
          tops::wait(memcpy_ev1);
        }
        dte_ctx1.set_dst_addr(sorted_ids + block_cnt * block_size);
        dte_ctx1.set_src_addr(local_sorted_ids + i * max_padded_num_in_block);
        dte_ctx1.set_total_size(block_num * block_size * sizeof(int32_t));
        memcpy_ev1 = dte_ctx1.trigger();
        pp_flag = 0;
      }
      triggered++;
      block_cnt += block_num;
    }
  }
  if (triggered > 0) {
    tops::wait(memcpy_ev0);
    if (triggered > 1) {
      tops::wait(memcpy_ev1);
    }
  }
  int num_padded = block_cnt * block_size;
  if (num_padded < sorted_ids_buf_size) {
    dte_ctx0.config_memset(
        mdspan(Global, sorted_ids + num_padded,
               sorted_ids_buf_size - num_padded),
        pad_value);
    memcpy_ev0 = dte_ctx0.trigger();
  }

  // experts_ids
  dte_ctx1.config_slice(mdspan(Global, experts_ids, expert_ids_buf_size),
                        mdspan(Private, local_experts_ids, block_cnt), {0});
  memcpy_ev1 = dte_ctx1.trigger();

  // num_tokens_post_pad
  mapped_ptr num_padded_mapped = tops::map_mem_m(num_tokens_post_pad, 1);
  int32_t *num_padded_ptr = reinterpret_cast<int32_t *>(num_padded_mapped);
  num_padded_ptr[0] = num_padded;
  tops::unmap_mem_m(num_padded_mapped, 1);

  tops::wait(memcpy_ev1);
  if (num_padded < sorted_ids_buf_size) {
    tops::wait(memcpy_ev0);
  }
  tcle::fence<FenceType::L2_MEM>();
  return 0;
}

__global__ void moe_align_block_size_kernel(
    int32_t *sorted_ids, int32_t *experts_ids, int32_t *num_tokens_post_pad,
    int32_t *topk_ids, int32_t *real_token_num, int32_t *expert_map,
    int32_t num_experts, int32_t block_size, int32_t token_num, int32_t topk) {
  const int kMemSize = VDMEM_SIZE / sizeof(int32_t);
  // constexpr int local_mem_size = 1 * 1024 * 1024 / sizeof(int32_t);
  __local__ __attribute__((aligned(512)))
  int32_t local_buf[kMemSize];
  extern __shared__ __valigned__ int32_t shared_buf[];
  int32_t pad_value = token_num * topk;
  int32_t max_padded_num = token_num * topk + num_experts * (block_size - 1);
  int32_t max_block_num = max_padded_num / block_size;
  tops::private_dte ctx_set_sorted_ids;
  ctx_set_sorted_ids.init();
  ctx_set_sorted_ids.config_memset(mdspan(Global, sorted_ids, max_padded_num),
                                   pad_value);
  tops::private_dte ctx_set_experts_ids;
  ctx_set_experts_ids.init();
  ctx_set_experts_ids.config_memset(mdspan(Global, experts_ids, max_block_num),
                                    0);
  if (real_token_num != nullptr) {
    tops::private_dte real_token_num_ctx;
    real_token_num_ctx.init();
    int32_t real_token_num_val = 0;
    tops::memcpy(real_token_num_ctx, mdspan(Private, &real_token_num_val, 1),
                 mdspan(Global, real_token_num, 1));
    op_assert(real_token_num_val >= 0, "real_token_num cannot be negative! "
              "real_token_num=%d", real_token_num_val);
    op_assert(real_token_num_val <= token_num,
              "real_token_num exceeds token_num! "
              "real_token_num=%d, token_num=%d",
              real_token_num_val, token_num);
    token_num = real_token_num_val;
  }
  if (token_num == 0) {
    mapped_ptr num_padded_mapped = tops::map_mem_m(num_tokens_post_pad, 1);
    int32_t *num_padded_ptr = reinterpret_cast<int32_t *>(num_padded_mapped);
    *num_padded_ptr = 0;
    tops::unmap_mem_m(num_padded_mapped, 1);
    tcle::fence<FenceType::L2_MEM>();
    ctx_set_sorted_ids.trigger_and_wait();
    ctx_set_experts_ids.trigger_and_wait();
    return;
  } else if (token_num * topk <= 1024) {
    int status = moe_align_block_size_scalar_kernel(
        sorted_ids, experts_ids, num_tokens_post_pad, topk_ids, local_buf,
        expert_map, num_experts, block_size, token_num, topk, pad_value);
    if (status == 0) {
      return;
    }
  }
  tops::event event_set_sorted_ids = ctx_set_sorted_ids.trigger();
  tops::event event_set_experts_ids = ctx_set_experts_ids.trigger();
  token_num *= topk;

  // auto AlignUp = [](int32_t x, int32_t y) -> int32_t {
  //   return ((x + y - 1) / y) * y;
  // };

  int32_t offset = 0;
  // indicate which expert each block belongs to
  int32_t *local_experts_ids = local_buf + offset;
  offset += AlignUp(token_num / block_size + 1 + num_experts, 128);
  // total number of valid token after padding
  int32_t *local_num_tokens_post_pad = local_buf + offset;
  offset += 128;
  // input token ids
  int32_t *local_topk_ids = local_buf + offset;
  offset += AlignUp(token_num + 15 * 64, 128);
  // number of tokens for each expert
  int32_t *local_experts_cnt = local_buf + offset;
  offset += AlignUp(num_experts, 128);
  // const buffer used to count the number of tokens for each expert
  int32_t *local_const_buf = local_buf + offset;
  offset += AlignUp(num_experts * 2 + 1, 128);
  int32_t *local_expert_map = local_buf + offset;

  tops::private_dte ctx_expert_map;
  ctx_expert_map.init();
  tops::event event_expert_map;
  if (expert_map) {
    ctx_expert_map.config_memcpy(mdspan(Private, local_expert_map, num_experts),
                                 mdspan(Global, expert_map, num_experts));
    event_expert_map = ctx_expert_map.trigger();
  }

  // dte in
  tops::private_dte ctx;
  ctx.init();
  u_int32_t pad_low[] = {0};
  u_int32_t pad_high[] = {15 * 64};
  u_int32_t pad_mid[] = {0};
  ctx.config_pad(mdspan(Private, local_topk_ids, token_num + 15 * 64),
                 mdspan(Global, topk_ids, token_num), pad_low, pad_high,
                 pad_mid, num_experts);
  ctx.trigger_and_wait();
  if (expert_map) {
    tops::wait(event_expert_map);
  }
  bool has_expert_map = (expert_map != nullptr);
  check_ids(local_topk_ids, local_expert_map, num_experts, token_num,
            has_expert_map);

  // stage1: counting the token_num for each expert
  // config expert id const buffer
  ctx.config_memset(mdspan(Private, local_experts_cnt,
    AlignUp(num_experts, 128) + AlignUp(num_experts * 2 + 1, 128)), 0);
  ctx.trigger_and_wait();
  int32_t *expert_id_cfg = local_const_buf + num_experts;
  expert_id_cfg[0] = 0x1;
  expert_id_cfg[1] = 0x10;
  expert_id_cfg[2] = 0x100;
  expert_id_cfg[3] = 0x1000;
  expert_id_cfg[4] = 0x10000;
  expert_id_cfg[5] = 0x100000;
  expert_id_cfg[6] = 0x1000000;
  expert_id_cfg[7] = 0x10000000;
  for (int i = 0; i < num_experts; i += 8) {
    int32_t *expert_cnt_buf = local_experts_cnt + i;
    int32_t *expert_id_buf = local_const_buf + num_experts - i;
    cnt_experts(expert_cnt_buf, local_topk_ids, expert_id_buf, i, num_experts,
                token_num);
  }
  // stage2: caculating the offset for each expert
  int32_t *cumsum_buf = local_const_buf; // reuse the const buffer
  experts_cnt_to_padded_cumsum(cumsum_buf, local_experts_cnt, num_experts,
                               block_size, num_experts);
  int32_t padded_size = cumsum_buf[num_experts - 1];
  *local_num_tokens_post_pad = padded_size;
  int32_t block_num = padded_size / block_size;

  // memset
  ctx.config_memset(mdspan(Shared, shared_buf, padded_size), pad_value);
  ctx.trigger_and_wait();
  // stage3: scatter the token ids to the corresponding expert
  gen_sorted_ids(shared_buf, local_experts_ids, cumsum_buf - 1, local_topk_ids,
                 local_const_buf + num_experts, local_expert_map, token_num,
                 block_size, num_experts, has_expert_map);

  // dte out
  tops::wait(event_set_sorted_ids);
  ctx.config_memcpy(mdspan(Global, sorted_ids, padded_size),
                    mdspan(Shared, shared_buf, padded_size));
  ctx.trigger_and_wait();

  tops::wait(event_set_experts_ids);
  ctx.config_memcpy(mdspan(Global, experts_ids, block_num),
                    mdspan(Private, local_experts_ids, block_num));
  ctx.trigger_and_wait();

  ctx.config_memcpy(mdspan(Global, num_tokens_post_pad, 1),
                    mdspan(Private, local_num_tokens_post_pad, 1));
  ctx.trigger_and_wait();

  return;
}

extern "C" void moe_align_block_size(
    int32_t *sorted_ids, int32_t *experts_ids, int32_t *num_tokens_post_pad,
    int32_t *topk_ids, int32_t *real_token_num, int32_t *expert_map,
    int32_t num_experts, int32_t block_size, int32_t token_num, int32_t topk,
    void *stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  constexpr int kDeviceBlocks = 2;
  constexpr int kThreadsPerBlock = 12;
  (void)moe_align_block_size_host(
      dim3(kDeviceBlocks, 1, 1), dim3(kThreadsPerBlock, 1, 1), sorted_ids,
      experts_ids, num_tokens_post_pad, topk_ids, real_token_num, expert_map,
      num_experts, block_size, token_num, topk, stream);
}
