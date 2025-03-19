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
 *
* @par Copyright (c) Enflame Tech Company.
* @par History: revised from TopsOp/Aten for candle-gcu
*/

#include <stdio.h>
#include <tops.h>
#include <tops/half.h>
#include <tops/bfloat.h>
#include "tops/tops_runtime.h"
#include "utils/utils.h"
#include "third_party/matmul_general.h"
#include "third_party/matmul_decoding.h"
#include <acore_op.h>

using namespace std;
#define MAX_NUM 1540096 - 1024
#define TEMPLATE_ALIGN_UP(a, b) (((a + b - 1) / b) * b)
#define L1_ALIGN_SIZE (128)


template <typename lhs_t, typename rhs_t>
__device__ __attribute__((noinline)) void matmul_kernel(lhs_t *lhs, rhs_t *rhs, rhs_t *out,
      int input_dtype, int input_batch,
      int input_m, int input_k, int input_n,
      int lhs_multicore, int rhs_multicore, int batch_multicore,
      int lhs_transpose, int rhs_transpose,
      float alpha, float beta, float addmm_beta, 
      int sip_m, int sip_k, int sip_n, int broadcasted_weight, char* buffer_sip) {
  int thread_num = GetThreadNum();
  int thread_id = GetThreadIdx();
  int sip_cnt_raw = thread_num;
  int thread_idx = thread_id;
  tops_dte_ctx_t dte_lhs[2];
  tops_dte_ctx_t dte_rhs[2];
  tops_dte_ctx_t dte_lhs_trans[2];
  tops_dte_ctx_t dte_rhs_trans[2];
  tops_dte_ctx_t dte_out;
  tops_dte_ctx_t dte_b;
  tops_dte_ctx_t dte_cache;

  dte_lhs[0].init();
  dte_lhs[1].init();
  dte_rhs[0].init();
  dte_rhs[1].init();
  dte_lhs_trans[0].init();
  dte_lhs_trans[1].init();
  dte_rhs_trans[0].init();
  dte_rhs_trans[1].init();
  dte_out.init();
  dte_b.init();
  dte_cache.init();

  tops::event event_lhs0;
  tops::event event_lhs1;
  tops::event event_rhs0;
  tops::event event_rhs1;
  tops::event e_lhs_trans_0;
  tops::event e_lhs_trans_1;
  tops::event e_rhs_trans_0;
  tops::event e_rhs_trans_1;
  tops::event e_bais;
  tops::event e_out;
  tops::event e_beta_output;

  auto data_type = input_dtype;
  auto weight_data_type = input_dtype;
  auto B = input_batch;
  auto M = input_m;
  auto K = input_k;
  auto N = input_n;
  auto is_lhs_split = lhs_multicore;
  auto is_rhs_split = rhs_multicore;
  auto is_batch_split = batch_multicore;
  auto transa = lhs_transpose;
  auto transb = rhs_transpose;

  int enable_act = 0;
  auto subm_size = sip_m;
  auto subn_size = sip_n;
  auto subk_size = sip_k;
  int enable_quant = 0;
  float scale = 1.0;

  tops::half quant_value(scale);

  int32_t hbm_lhs_shape[4] = {1, B, M, K};
  int32_t hbm_rhs_shape[4] = {1, broadcasted_weight > 0 ? 1 : B, K, N};
  int32_t hbm_out_shape[4] = {1, B, M, N};
  if (transa == 1) {
    hbm_lhs_shape[2] = K;
    hbm_lhs_shape[3] = M;
  }
  if (transb == 1) {
    hbm_rhs_shape[2] = N;
    hbm_rhs_shape[3] = K;
  }
  int32_t hbm_bais_shape[1] = {N};
  int32_t sip_lhs_shape[4] = {1, 1, subm_size, subk_size};
  int32_t sip_rhs_shape[4] = {1, 1, subk_size, subn_size};
  int32_t sip_out_shape[4] = {1, 1, subm_size, subn_size};
  int32_t sip_lhs_trans_shape[4] = {1, 1, subk_size, subm_size};
  int32_t sip_rhs_trans_shape[4] = {1, 1, subn_size, subk_size};

  int32_t sip_lhs_size = sip_lhs_shape[0] * sip_lhs_shape[1] *
                         sip_lhs_shape[2] * sip_lhs_shape[3] * sizeof(lhs_t);
  int32_t sip_rhs_size = sip_rhs_shape[0] * sip_rhs_shape[1] *
                         sip_rhs_shape[2] * sip_rhs_shape[3] * sizeof(lhs_t);
  int32_t sip_out_size = sip_out_shape[0] * sip_out_shape[1] *
                         sip_out_shape[2] * sip_out_shape[3] *
                         sizeof(float);  // use fp32
  int32_t sip_lhs_trans_size = transa == 0 ? 0 : sip_lhs_trans_shape[0] * sip_lhs_trans_shape[1] *
                               sip_lhs_trans_shape[2] * sip_lhs_trans_shape[3] *
                               sizeof(lhs_t);
  int32_t sip_rhs_trans_size = transb == 0 ? 0 : sip_rhs_trans_shape[0] * sip_rhs_trans_shape[1] *
                               sip_rhs_trans_shape[2] * sip_rhs_trans_shape[3] *
                               sizeof(lhs_t);

  tops::mdspan hbm_lhs(tops::Global, lhs, hbm_lhs_shape);
  tops::mdspan hbm_rhs(tops::Global, rhs, hbm_rhs_shape);
  tops::mdspan hbm_out(tops::Global, out, hbm_out_shape);


  lhs_t *buffer_sip_lhs0_trans =
      reinterpret_cast<lhs_t *>(reinterpret_cast<char *>(buffer_sip));
  lhs_t *buffer_sip_lhs1_trans = reinterpret_cast<lhs_t *>(
      (reinterpret_cast<char *>(buffer_sip_lhs0_trans)) +
      TEMPLATE_ALIGN_UP(sip_lhs_trans_size, L1_ALIGN_SIZE));
  rhs_t *buffer_sip_rhs0_trans = reinterpret_cast<rhs_t *>(
      (reinterpret_cast<char *>(buffer_sip_lhs1_trans)) +
      TEMPLATE_ALIGN_UP(sip_lhs_trans_size, L1_ALIGN_SIZE));
  rhs_t *buffer_sip_rhs1_trans = reinterpret_cast<rhs_t *>(
      (reinterpret_cast<char *>(buffer_sip_rhs0_trans)) +
      TEMPLATE_ALIGN_UP(sip_rhs_trans_size, L1_ALIGN_SIZE));
  lhs_t *buffer_sip_lhs0 = reinterpret_cast<lhs_t *>(
      (reinterpret_cast<char *>(buffer_sip_rhs1_trans)) +
      TEMPLATE_ALIGN_UP(sip_rhs_trans_size, L1_ALIGN_SIZE));
  lhs_t *buffer_sip_lhs1 =
      reinterpret_cast<lhs_t *>((reinterpret_cast<char *>(buffer_sip_lhs0)) +
                                TEMPLATE_ALIGN_UP(sip_lhs_size, L1_ALIGN_SIZE));
  rhs_t *buffer_sip_rhs0 =
      reinterpret_cast<rhs_t *>((reinterpret_cast<char *>(buffer_sip_lhs1)) +
                                TEMPLATE_ALIGN_UP(sip_lhs_size, L1_ALIGN_SIZE));
  rhs_t *buffer_sip_rhs1 =
      reinterpret_cast<rhs_t *>((reinterpret_cast<char *>(buffer_sip_rhs0)) +
                                TEMPLATE_ALIGN_UP(sip_rhs_size, L1_ALIGN_SIZE));
  float *buffer_private_out0 =
      reinterpret_cast<float *>((reinterpret_cast<char *>(buffer_sip_rhs1)) +
                                TEMPLATE_ALIGN_UP(sip_rhs_size, L1_ALIGN_SIZE));
  float *buffer_private_out1 = reinterpret_cast<float *>(
      (reinterpret_cast<char *>(buffer_private_out0)) +
      TEMPLATE_ALIGN_UP(sip_out_size, L1_ALIGN_SIZE));

  long lhs_addr = (long)buffer_sip_lhs0;
  long rhs_addr = (long)buffer_sip_rhs0;
  long cur_out_addr = (long)buffer_private_out0;

  int32_t beta_output_sdma_wait = 0;

  int weight_offset = 0;
  if ((data_type == TOPSOP_DATA_FP16) && (weight_data_type == TOPSOP_DATA_I8)) {
    weight_offset = subk_size * subn_size;
  }
  tops::mdspan private_lhs0(tops::Private, buffer_sip_lhs0, sip_lhs_shape);
  tops::mdspan private_lhs1(tops::Private, buffer_sip_lhs1, sip_lhs_shape);
  tops::mdspan private_rhs0(tops::Private, buffer_sip_rhs0 + weight_offset,
                            sip_rhs_shape);
  tops::mdspan private_rhs1(tops::Private, buffer_sip_rhs1 + weight_offset,
                            sip_rhs_shape);
  tops::mdspan private_lhs0_trans(tops::Private, buffer_sip_lhs0_trans,
                                  sip_lhs_trans_shape);
  tops::mdspan private_lhs1_trans(tops::Private, buffer_sip_lhs1_trans,
                                  sip_lhs_trans_shape);
  tops::mdspan private_rhs0_trans(tops::Private,
                                  buffer_sip_rhs0_trans + weight_offset,
                                  sip_rhs_trans_shape);
  tops::mdspan private_rhs1_trans(tops::Private,
                                  buffer_sip_rhs1_trans + weight_offset,
                                  sip_rhs_trans_shape);
  tops::mdspan private_out0(tops::Private, buffer_private_out0, sip_out_shape);
  tops::mdspan private_out1(tops::Private, buffer_private_out1, sip_out_shape);

//   lhs_t *buffer_sip_out0_f16 = reinterpret_cast<lhs_t *>(buffer_private_out0);
//   tops::mdspan sip_out0_f16(tops::Private, buffer_sip_out0_f16, sip_out_shape);
//   lhs_t *buffer_sip_out1_f16 = reinterpret_cast<lhs_t *>(buffer_private_out1);
//   tops::mdspan sip_out1_f16(tops::Private, buffer_sip_out1_f16, sip_out_shape);

//   tops::mdspan sip_bias0(tops::Private, buffer_sip_bias0, N);
  auto M_SIP_LOOP_CNT_TASKS = M / subm_size + (M % subm_size > 0 ? 1 : 0);
  auto N_SIP_LOOP_CNT_TASKS = N / subn_size + (N % subn_size > 0 ? 1 : 0);
  auto subk_count = K / subk_size + (K % subk_size > 0 ? 1 : 0);
  auto xhs_multicore = 0;
  if ((is_lhs_split == 1) || (is_rhs_split == 1) || (is_batch_split == 1)) {
    xhs_multicore = 1;
  }
  auto sip_cnt = (xhs_multicore == 0) ? 1 : sip_cnt_raw;
  auto sdma_tasks_num =
      (is_lhs_split == 1) ? M_SIP_LOOP_CNT_TASKS : N_SIP_LOOP_CNT_TASKS;
  if (is_batch_split == 1) {
    sdma_tasks_num = B;
  }
  auto sip_num_used = (sdma_tasks_num > sip_cnt) ? sip_cnt : sdma_tasks_num;
  auto lhs_loop_step = (is_lhs_split == 1) ? sip_num_used : 1;
  auto rhs_loop_step = (is_rhs_split == 1) ? sip_num_used : 1;
  auto reminder = sdma_tasks_num % sip_cnt;
  auto loop_len_this_sip = (thread_idx < reminder)
                               ? (sdma_tasks_num / sip_cnt + 1)
                               : (sdma_tasks_num / sip_cnt);
  if (loop_len_this_sip == 0) {
    return;
  }
  int batch_hmb_offset = 0;
  if (is_batch_split == 1) {
    batch_hmb_offset = (thread_idx < reminder)
                           ? (thread_idx * loop_len_this_sip)
                           : (thread_idx * loop_len_this_sip + reminder);
  }
  auto subm_count =
      is_lhs_split == 1 ? loop_len_this_sip : M_SIP_LOOP_CNT_TASKS;
  auto subn_count =
      is_rhs_split == 1 ? loop_len_this_sip : N_SIP_LOOP_CNT_TASKS;
  auto Batch_SIP_LOOP_CNT = (is_batch_split == 1) ? loop_len_this_sip : B;

  if (thread_idx < sip_num_used) {
    if (transa == 0) {
    } else {
      dte_lhs_trans[0].connect(dte_lhs[0]);
      dte_lhs_trans[1].connect(dte_lhs[1]);
    }
    if (transb == 0) {
    } else {
      dte_rhs_trans[0].connect(dte_rhs[0]);
      dte_rhs_trans[1].connect(dte_rhs[1]);
    }

    for (auto b_idx = 0; b_idx < Batch_SIP_LOOP_CNT; b_idx++) {
      auto m_hbm_offset = (is_lhs_split == 1) ? thread_idx * subm_size : 0;
      auto n_hbm_offset = (is_rhs_split == 1) ? thread_idx * subn_size : 0;
      auto batch_offset = batch_hmb_offset + b_idx;
      auto r_batch_offset = broadcasted_weight > 0 ? 0 : batch_offset;

      if (transa == 0) {
        event_lhs0 = tops::slice_async(dte_lhs[0], private_lhs0, hbm_lhs,
                                       {0, batch_offset, m_hbm_offset, 0});
      } else {
        e_lhs_trans_0 =
            tops::slice_async(dte_lhs_trans[0], private_lhs0_trans, hbm_lhs,
                              {0, batch_offset, 0, m_hbm_offset});
        // tops::wait(e_lhs_trans_0);
        event_lhs0 = tops::transpose_async(dte_lhs[0], private_lhs0,
                                           private_lhs0_trans, {0, 1, 3, 2});
      }

      if (transb == 0) {
        event_rhs0 = tops::slice_async(dte_rhs[0], private_rhs0, hbm_rhs,
                                       {0, r_batch_offset, 0, n_hbm_offset});
      } else {
        e_rhs_trans_0 =
            tops::slice_async(dte_rhs_trans[0], private_rhs0_trans, hbm_rhs,
                              {0, r_batch_offset, n_hbm_offset, 0});
        // tops::wait(e_rhs_trans_0);

        event_rhs0 = tops::transpose_async(dte_rhs[0], private_rhs0,
                                           private_rhs0_trans, {0, 1, 3, 2});
      }
      // need to delete the output load or initialize
      bool flag_no_beta = true;
      int *tmp_beta = reinterpret_cast<int *>(&beta);
      flag_no_beta = ((*tmp_beta) == 0);
      e_beta_output = tops::memset_async(dte_b, private_out0, 0.0f);

      beta_output_sdma_wait = 1;
      int next_n_sip_idx_temp, next_subm_index, next_m_sip_idx_temp;

      for (auto subm_index = 0;
           subm_index < subm_count;
           subm_index++) {
        int subm_index_global;
        if (is_lhs_split) {
          subm_index_global =
              subm_index * lhs_loop_step + thread_idx;
        } else {
          subm_index_global = subm_index;
        }
        int subm_offset_global = subm_index_global * subm_size;
        for (auto subn_index = 0;
             subn_index < subn_count;
             subn_index++) {
          int subn_index_global;
          if (is_rhs_split) {
            subn_index_global =
                subn_index * rhs_loop_step + thread_idx;
          } else {
            subn_index_global = subn_index;
          }

          int subn_offset_global = subn_index_global * subn_size;

          auto subout_index =
              subm_index * subn_count +
              subn_index;
          auto cur_private_out = ((subout_index % 2) == 0)
                                     ? &private_out0
                                     : &private_out1;
          cur_out_addr =
              (long)((subout_index % 2) == 0 ? buffer_private_out0
                                                         : buffer_private_out1);
          auto cur_private_output_ptr =
              ((subout_index % 2) == 0 ? buffer_private_out0
                                                   : buffer_private_out1);
          for (auto subk_index = 0;
               subk_index < subk_count;
               subk_index++) {
            // no parallel split k
            // auto k_sip_offset = subk_index * subk_size;
            auto global_loop_index =
                subm_index * subn_count *
                    subk_count +
                subn_index * subk_count +
                subk_index;

            auto next_private_lhs =
                (global_loop_index % 2) ? &private_lhs0 : &private_lhs1;
            auto next_private_rhs =
                (global_loop_index % 2) ? &private_rhs0 : &private_rhs1;
            auto next_private_out = (subout_index % 2) == 1
                                        ? &private_out0
                                        : &private_out1;
            auto next_private_lhs_trans = (global_loop_index % 2) == 1
                                              ? &private_lhs0_trans
                                              : &private_lhs1_trans;
            auto next_private_rhs_trans = (global_loop_index % 2) == 1
                                              ? &private_rhs0_trans
                                              : &private_rhs1_trans;

            int next_subk_index, next_subn_index,
                next_subm_index;

            if (subk_index + 1 < subk_count) {
              next_subk_index = subk_index + 1;
              next_subn_index = subn_index;
              next_subm_index = subm_index;
            } else {
              // subk_index + 1 == subk_count
              next_subk_index = 0;
              if (subn_index + 1 < subn_count) {
                next_subn_index = subn_index + 1;
                next_subm_index = subm_index;
              } else {
                // subn_index + 1 == subn_count
                next_subn_index = 0;
                next_subm_index = subm_index + 1;
              }
            }

            auto next_subk_offset_global =
                next_subk_index * subk_size;
            // auto next_subn_offset_global = next_subn_index *
            // subn_size;
            auto next_subn_offset_global =
                is_rhs_split ? ((next_subn_index *
                                     rhs_loop_step +
                                 thread_idx) *
                                subn_size)
                             : (next_subn_index * subn_size);
            // auto next_subm_offset_global = next_subm_index *
            // subm_size;
            auto next_subm_offset_global =
                is_lhs_split ? ((next_subm_index *
                                     lhs_loop_step +
                                 thread_idx) *
                                subm_size)
                             : (next_subm_index * subm_size);

            auto cur_event_lhs =
                (global_loop_index % 2) == 0 ? event_lhs0 : event_lhs1;
            tops::wait(cur_event_lhs);
            auto cur_event_rhs =
                (global_loop_index % 2) == 0 ? event_rhs0 : event_rhs1;
            tops::wait(cur_event_rhs);

            // need to delete for no output load or initialize
            if (beta_output_sdma_wait == 1) {
              tops::wait(e_beta_output);
            }
            beta_output_sdma_wait = 0;

            auto next_event_lhs =
                (global_loop_index % 2) == 1 ? event_lhs0 : event_lhs1;
            auto next_event_rhs =
                (global_loop_index % 2) == 1 ? event_rhs0 : event_rhs1;

            if ((next_subk_offset_global < K) &&
                (next_subn_offset_global < N) &&
                (next_subm_offset_global < M)) {
              if (transa == 0) {
                next_event_lhs =
                    tops::slice_async(dte_lhs[0], *next_private_lhs, hbm_lhs,
                                      {0, batch_offset, next_subm_offset_global,
                                       next_subk_offset_global});
              } else {
                e_lhs_trans_0 = tops::slice_async(
                    dte_lhs_trans[0], *next_private_lhs_trans, hbm_lhs,
                    {0, batch_offset, next_subk_offset_global,
                     next_subm_offset_global});
                // tops::wait(e_lhs_trans_0);
                next_event_lhs = tops::transpose_async(
                    dte_lhs[0], *next_private_lhs, *next_private_lhs_trans,
                    {0, 1, 3, 2});
              }

              if (transb == 0) {
                next_event_rhs =
                    tops::slice_async(dte_rhs[0], *next_private_rhs, hbm_rhs,
                                      {0, r_batch_offset, next_subk_offset_global,
                                       next_subn_offset_global});
              } else {
                e_rhs_trans_0 = tops::slice_async(
                    dte_rhs_trans[0], *next_private_rhs_trans, hbm_rhs,
                    {0, r_batch_offset, next_subn_offset_global,
                     next_subk_offset_global});
                // tops::wait(e_rhs_trans_0);
                next_event_rhs = tops::transpose_async(
                    dte_rhs[0], *next_private_rhs, *next_private_rhs_trans,
                    {0, 1, 3, 2});
              }
            }
            if ((next_subn_offset_global < N) &&
                (next_subm_offset_global < M) &&
                (next_subk_offset_global == 0)) {
              // need to initialize
              e_beta_output =
                  tops::memset_async(dte_b, *next_private_out, 0.0f);

              beta_output_sdma_wait = 1;
            }

            auto nacc_flag = subk_index == 0 ? 1 : 0;
            auto store_flag =
                subk_index + 1 == subk_count ? 1 : 0;
            lhs_addr = long((global_loop_index % 2) == 0 ? buffer_sip_lhs0
                                                         : buffer_sip_lhs1);
            auto cur_buffer_sip_rhs = (global_loop_index % 2) == 0
                                          ? buffer_sip_rhs0
                                          : buffer_sip_rhs1;
            rhs_addr = long(cur_buffer_sip_rhs);
            if (data_type == TOPSOP_DATA_FP32) {
              c_func_smatmul_general(
                  lhs_addr, rhs_addr, cur_out_addr, subm_size, subn_size,
                  subk_size);
            } else if (data_type == TOPSOP_DATA_FP16) {
                c_func_hmatmul_general(
                    lhs_addr, rhs_addr, cur_out_addr, subm_size, subn_size,
                    subk_size);
            } else if (data_type == TOPSOP_DATA_I8) {
              c_func_matmul_general_int8(
                  lhs_addr, rhs_addr, cur_out_addr, subm_size, subn_size,
                  subk_size);
            } else if (data_type == TOPSOP_DATA_BF16) {
              c_func_bfmatmul_general(
                  lhs_addr, rhs_addr, cur_out_addr, subm_size, subn_size,
                  subk_size);
            }
          }  // K loop

          if (data_type == TOPSOP_DATA_FP16) {
            convert(reinterpret_cast<__fp16 *>(cur_private_output_ptr),
                    reinterpret_cast<float *>(cur_private_output_ptr),
                    subm_size * subn_size);

          } else if (data_type == TOPSOP_DATA_I8) {
            convert(reinterpret_cast<char *>(cur_private_output_ptr),
                    reinterpret_cast<int32_t *>(cur_private_output_ptr),
                    subm_size * subn_size);
          } else if (data_type == TOPSOP_DATA_BF16) {
            convert(reinterpret_cast<__bf16 *>(cur_private_output_ptr),
                    reinterpret_cast<float *>(cur_private_output_ptr),
                    subm_size * subn_size);
          }
          e_out = tops::deslice_async(
              dte_out, hbm_out, *cur_private_out,
              {0, batch_offset, subm_offset_global, subn_offset_global});
          tops::wait(e_out);
        }  // N loop
      }    // M loop
    }      // batch loop
  }        // sip_id branch

  dte_lhs[0].destroy();
  dte_lhs[1].destroy();
  dte_rhs[0].destroy();
  dte_rhs[1].destroy();
  dte_lhs_trans[0].destroy();
  dte_lhs_trans[1].destroy();
  dte_rhs_trans[0].destroy();
  dte_rhs_trans[1].destroy();
  dte_out.destroy();
//   dte_bias.destroy();
  dte_b.destroy();
}  // func

template <typename lhs_t, typename rhs_t, typename out_t, typename bias_t, typename scale_t>
__device__ __attribute__((noinline)) void matmul_kernel_batch(lhs_t *lhs, rhs_t *rhs, out_t *out, 
      bias_t* bias, scale_t* scale, lhs_t* zeros,
      int input_dtype, int input_batch,
      int input_m, int input_k, int input_n,
      int lhs_multicore, int rhs_multicore, int batch_multicore,
      int lhs_transpose, int rhs_transpose,
      float alpha, float beta, float addmm_beta, 
      int sip_m, int sip_k, int sip_n, int broadcasted_weight, int group_size, char* buffer_sip) {
  int quant_type = 0;
  int enable_quant = 0;
  int enable_bias = 0;

  if (std::is_same<lhs_t, __fp16>::value ||
      std::is_same<lhs_t, __bf16>::value ) {
      if (std::is_same<rhs_t, uint8_t>::value) {
        quant_type = 1; //QuantType::W4A16;
        enable_quant = 1;
      } else if (std::is_same<rhs_t, int8_t>::value) {
        quant_type = 2; //QuantType::W8A16;
        enable_quant = 1;
      }
  }

  int thread_num = GetThreadNum();
  int thread_id = GetThreadIdx();
  int sip_cnt_raw = thread_num;
  int thread_idx = thread_id;
  tops_dte_ctx_t dte_lhs[2];
  tops_dte_ctx_t dte_rhs[2];
  tops_dte_ctx_t dte_lhs_trans[2];
  tops_dte_ctx_t dte_rhs_trans[2];
  tops_dte_ctx_t dte_out;
  tops_dte_ctx_t dte_bias;
  tops_dte_ctx_t dte_scale;
  tops_dte_ctx_t dte_zeros;


  dte_lhs[0].init();
  dte_lhs[1].init();
  dte_rhs[0].init();
  dte_rhs[1].init();
  dte_lhs_trans[0].init();
  dte_lhs_trans[1].init();
  dte_rhs_trans[0].init();
  dte_rhs_trans[1].init();
  dte_out.init();
  dte_bias.init();
  dte_scale.init();
  dte_zeros.init();


  tops::event event_lhs0;
  tops::event event_lhs1;
  tops::event event_rhs0;
  tops::event event_rhs1;
  tops::event e_lhs_trans_0;
  tops::event e_lhs_trans_1;
  tops::event e_rhs_trans_0;
  tops::event e_rhs_trans_1;
  tops::event event_bias0;
  tops::event event_bias1;
  tops::event event_scale0;
  tops::event event_scale1;
  tops::event event_zeros0;
  tops::event event_zeros1;
  tops::event e_out;

  auto data_type = input_dtype;
  auto weight_data_type = input_dtype;
  auto B = input_batch;
  auto M = input_m;
  auto K = input_k;
  auto N = input_n;
  auto is_lhs_split = lhs_multicore;
  auto is_rhs_split = rhs_multicore;
  auto is_batch_split = batch_multicore;

  int enable_act = 0;
  auto subm_size = sip_m;
  auto subn_size = sip_n;
  auto subk_size = sip_k;

  group_size = group_size == -1 ? K : group_size;
  int group_num = K / group_size;
  int k_group_num = CeilDiv(subk_size, group_size);
  if (group_size == -1) {
    if (K < subk_size) {
      k_group_num = 1;
    }
  }


  auto need_trans_lhs = lhs_transpose;
  auto need_trans_rhs = rhs_transpose;
  int32_t rhs_k = K;
  int32_t rhs_subk_size = subk_size;
  if (quant_type == 1) {
    rhs_subk_size = subk_size / 2;
    if ((K % 128) == 0) {
      rhs_k = K / 2;
    } else {
      rhs_k = K / 2 + 32;
    }
  }


  int32_t hbm_lhs_shape[4] = {1, B, M, K};
  int32_t hbm_rhs_shape[4] = {1, broadcasted_weight > 0 ? 1 : B, rhs_k, N};
  int32_t hbm_out_shape[4] = {1, B, M, N};
  int32_t unit_n = 128;
  if (data_type == TOPSOP_DATA_FP32) {
    unit_n = 64;
  }
  if (need_trans_lhs) {
    hbm_lhs_shape[2] = K;
    hbm_lhs_shape[3] = M;
  }

  if (need_trans_rhs) {
    hbm_rhs_shape[2] = N;
    hbm_rhs_shape[3] = rhs_k;
  }
  int32_t sip_lhs_shape[4] = {1, 1, subm_size, subk_size};
  int32_t sip_rhs_shape[4] = {1, 1, rhs_subk_size, subn_size};
  int32_t sip_out_shape[4] = {1, 1, subm_size, subn_size};
  int32_t sip_lhs_trans_shape[4] = {1, 1, subk_size, subm_size};
  int32_t sip_rhs_trans_shape[4] = {1, 1, subn_size, rhs_subk_size};

  int32_t sip_bias_shape[1] = {subn_size};
  int32_t sip_scale_shape[2] = {k_group_num, subn_size};
  int32_t sip_zeros_shape[2] = {k_group_num, subn_size};
  int32_t sip_lhs_size = sip_lhs_shape[0] * sip_lhs_shape[1] *
                         sip_lhs_shape[2] * sip_lhs_shape[3] * sizeof(lhs_t);
  int32_t sip_rhs_size = sip_rhs_shape[0] * sip_rhs_shape[1] *
                         sip_rhs_shape[2] * sip_rhs_shape[3] * sizeof(rhs_t);
  int32_t sip_out_size = sip_out_shape[0] * sip_out_shape[1] *
                         sip_out_shape[2] * sip_out_shape[3] *
                         sizeof(out_t);  // use fp32
  if (quant_type == 1) {
    sip_rhs_size = sip_rhs_size * 2;
  }
  int32_t sip_lhs_trans_size = need_trans_lhs > 0 ? sip_lhs_size : 0;
  int32_t sip_rhs_trans_size = need_trans_rhs > 0 ? sip_rhs_size : 0;
  int32_t sip_bias_size = 0;
  int32_t sip_scale_size = 0;
  int32_t sip_zeros_size = 0;
  if (enable_bias) {
    sip_bias_size = subn_size * sizeof(bias_t);
  }
  if (enable_quant) {
    sip_scale_size = k_group_num * subn_size * sizeof(scale_t);
  }
  if (quant_type == 1) {
    sip_scale_size = k_group_num * subn_size * sizeof(scale_t);
    sip_zeros_size = k_group_num * subn_size * sizeof(scale_t);
  }
  tops::mdspan hbm_lhs(tops::Global, lhs, hbm_lhs_shape);
  tops::mdspan hbm_rhs(tops::Global, rhs, hbm_rhs_shape);
  tops::mdspan hbm_out(tops::Global, out, hbm_out_shape);
  // tops::mdspan hbm_pre_gelu(tops::Global, pre_gelu, hbm_out_shape);
  tops::mdspan hbm_bias(tops::Global, bias, N);
  tops::mdspan hbm_scale(tops::Global, scale, group_num, N);
  tops::mdspan hbm_zeros(tops::Global, zeros, group_num, N);
  // __local__ __valigned__ char buffer_sip[VDMEM_SIZE];
  // workspace is 2KB
  int* local_workspace =
      reinterpret_cast<int*>(reinterpret_cast<char*>(buffer_sip));
  lhs_t* buffer_sip_lhs0_trans =
      reinterpret_cast<lhs_t*>(reinterpret_cast<char*>(local_workspace)) + 2048;
  lhs_t* buffer_sip_lhs1_trans = reinterpret_cast<lhs_t*>(
      (reinterpret_cast<char*>(buffer_sip_lhs0_trans)) +
      TEMPLATE_ALIGN_UP(sip_lhs_trans_size, L1_ALIGN_SIZE));
  rhs_t* buffer_sip_rhs0_trans = reinterpret_cast<rhs_t*>(
      (reinterpret_cast<char*>(buffer_sip_lhs1_trans)) +
      TEMPLATE_ALIGN_UP(sip_lhs_trans_size, L1_ALIGN_SIZE));
  rhs_t* buffer_sip_rhs1_trans = reinterpret_cast<rhs_t*>(
      (reinterpret_cast<char*>(buffer_sip_rhs0_trans)) +
      TEMPLATE_ALIGN_UP(sip_rhs_trans_size, L1_ALIGN_SIZE));
  lhs_t* buffer_sip_lhs0 = reinterpret_cast<lhs_t*>(
      (reinterpret_cast<char*>(buffer_sip_rhs1_trans)) +
      TEMPLATE_ALIGN_UP(sip_rhs_trans_size, L1_ALIGN_SIZE));
  lhs_t* buffer_sip_lhs1 =
      reinterpret_cast<lhs_t*>((reinterpret_cast<char*>(buffer_sip_lhs0)) +
                               TEMPLATE_ALIGN_UP(sip_lhs_size, L1_ALIGN_SIZE));
  out_t* buffer_private_out0 =
      reinterpret_cast<out_t*>((reinterpret_cast<char*>(buffer_sip_lhs1)) +
                               TEMPLATE_ALIGN_UP(sip_lhs_size, L1_ALIGN_SIZE));
  out_t* buffer_private_out1 =
      reinterpret_cast<out_t*>((reinterpret_cast<char*>(buffer_private_out0)) +
                               TEMPLATE_ALIGN_UP(sip_out_size, L1_ALIGN_SIZE));
  bias_t* buffer_sip_bias0 =
      reinterpret_cast<bias_t*>((reinterpret_cast<char*>(buffer_private_out1)) +
                               TEMPLATE_ALIGN_UP(sip_out_size, L1_ALIGN_SIZE));
  bias_t* buffer_sip_bias1 =
      reinterpret_cast<bias_t*>((reinterpret_cast<char*>(buffer_sip_bias0)) +
                               TEMPLATE_ALIGN_UP(sip_bias_size, L1_ALIGN_SIZE));
  scale_t* buffer_sip_scale0 =
      reinterpret_cast<scale_t*>((reinterpret_cast<char*>(buffer_sip_bias1)) +
                               TEMPLATE_ALIGN_UP(sip_bias_size, L1_ALIGN_SIZE));
  scale_t* buffer_sip_scale1 = reinterpret_cast<bias_t*>(
      (reinterpret_cast<char*>(buffer_sip_scale0)) +
      TEMPLATE_ALIGN_UP(sip_scale_size, L1_ALIGN_SIZE));
  scale_t* buffer_sip_zeros0 = reinterpret_cast<scale_t*>(
      (reinterpret_cast<char*>(buffer_sip_scale1)) +
      TEMPLATE_ALIGN_UP(sip_scale_size, L1_ALIGN_SIZE));
  scale_t* buffer_sip_zeros1 = reinterpret_cast<scale_t*>(
      (reinterpret_cast<char*>(buffer_sip_zeros0)) +
      TEMPLATE_ALIGN_UP(sip_zeros_size, L1_ALIGN_SIZE));
  rhs_t* buffer_sip_rhs0 = reinterpret_cast<rhs_t*>(
      (reinterpret_cast<char*>(buffer_sip_zeros1)) +
      TEMPLATE_ALIGN_UP(sip_zeros_size, L1_ALIGN_SIZE));
  rhs_t* buffer_sip_rhs1 =
      reinterpret_cast<rhs_t*>((reinterpret_cast<char*>(buffer_sip_rhs0)) +
                               TEMPLATE_ALIGN_UP(sip_rhs_size, L1_ALIGN_SIZE));
  rhs_t* private_rhs_requant_buff =
      reinterpret_cast<rhs_t*>((reinterpret_cast<char*>(buffer_sip_rhs1)) +
                               TEMPLATE_ALIGN_UP(sip_rhs_size, L1_ALIGN_SIZE));

  tops::mdspan private_lhs0(tops::Private, buffer_sip_lhs0, sip_lhs_shape);
  tops::mdspan private_lhs1(tops::Private, buffer_sip_lhs1, sip_lhs_shape);
  tops::mdspan private_rhs0(tops::Private, buffer_sip_rhs0,
                            sip_rhs_shape);
  tops::mdspan private_rhs1(tops::Private, buffer_sip_rhs1,
                            sip_rhs_shape);
  tops::mdspan private_lhs0_trans(tops::Private, buffer_sip_lhs0_trans,
                                  sip_lhs_trans_shape);
  tops::mdspan private_lhs1_trans(tops::Private, buffer_sip_lhs1_trans,
                                  sip_lhs_trans_shape);
  tops::mdspan private_rhs0_trans(tops::Private,
                                  buffer_sip_rhs0_trans,
                                  sip_rhs_trans_shape);
  tops::mdspan private_rhs1_trans(tops::Private,
                                  buffer_sip_rhs1_trans,
                                  sip_rhs_trans_shape);
  tops::mdspan private_out0(tops::Private, buffer_private_out0, sip_out_shape);
  tops::mdspan private_out1(tops::Private, buffer_private_out1, sip_out_shape);
  tops::mdspan private_bias0(tops::Private, buffer_sip_bias0, subn_size);
  tops::mdspan private_bias1(tops::Private, buffer_sip_bias1, subn_size);
  tops::mdspan private_scale0(tops::Private, buffer_sip_scale0, k_group_num,
                              subn_size);
  tops::mdspan private_scale1(tops::Private, buffer_sip_scale1, k_group_num,
                              subn_size);
  tops::mdspan private_zeros0(tops::Private, buffer_sip_zeros0, k_group_num,
                              subn_size);
  tops::mdspan private_zeros1(tops::Private, buffer_sip_zeros1, k_group_num,
                              subn_size);
  auto M_SIP_LOOP_CNT_TASKS = M / subm_size + (M % subm_size > 0 ? 1 : 0);
  auto N_SIP_LOOP_CNT_TASKS = N / subn_size + (N % subn_size > 0 ? 1 : 0);
  auto subk_count = K / subk_size + (K % subk_size > 0 ? 1 : 0);
  auto xhs_multicore = 0;
  if ((is_lhs_split == 1) || (is_rhs_split == 1) || (is_batch_split == 1)) {
    xhs_multicore = 1;
  }
  auto sip_cnt = (xhs_multicore == 0) ? 1 : sip_cnt_raw;
  auto sdma_tasks_num =
      (is_lhs_split == 1) ? M_SIP_LOOP_CNT_TASKS : N_SIP_LOOP_CNT_TASKS;
  if (is_batch_split == 1) {
    sdma_tasks_num = B;
  }
  auto sip_num_used = (sdma_tasks_num > sip_cnt) ? sip_cnt : sdma_tasks_num;
  auto lhs_loop_step = (is_lhs_split == 1) ? sip_num_used : 1;
  auto rhs_loop_step = (is_rhs_split == 1) ? sip_num_used : 1;
  auto reminder = sdma_tasks_num % sip_cnt;
  auto loop_len_this_sip = (thread_idx < reminder)
                               ? (sdma_tasks_num / sip_cnt + 1)
                               : (sdma_tasks_num / sip_cnt);
  if (loop_len_this_sip == 0) {
    return;
  }
  int batch_hmb_offset = 0;
  if (is_batch_split == 1) {
    batch_hmb_offset = (thread_idx < reminder)
                           ? (thread_idx * loop_len_this_sip)
                           : (thread_idx * loop_len_this_sip + reminder);
  }
  auto subm_count =
      is_lhs_split == 1 ? loop_len_this_sip : M_SIP_LOOP_CNT_TASKS;
  auto subn_count =
      is_rhs_split == 1 ? loop_len_this_sip : N_SIP_LOOP_CNT_TASKS;
  auto Batch_SIP_LOOP_CNT = (is_batch_split == 1) ? loop_len_this_sip : B;
  int vab_offset = (enable_quant == 1) ? 512 : 0;
  if (thread_idx < sip_num_used) {
    if (need_trans_lhs) {
      dte_lhs_trans[0].connect(dte_lhs[0]);
      dte_lhs_trans[1].connect(dte_lhs[1]);
    }
    if (need_trans_rhs) {
      dte_rhs_trans[0].connect(dte_rhs[0]);
      dte_rhs_trans[1].connect(dte_rhs[1]);
    }
    int launch_times = 0;
    for (auto b_idx = 0; b_idx < Batch_SIP_LOOP_CNT; b_idx++) {
      auto m_hbm_offset = (is_lhs_split == 1) ? thread_idx * subm_size : 0;
      auto n_hbm_offset = (is_rhs_split == 1) ? thread_idx * subn_size : 0;
      auto batch_offset = batch_hmb_offset + b_idx;
      auto r_batch_offset = broadcasted_weight > 0 ? 0 : batch_offset;
      if (need_trans_lhs) {
        e_lhs_trans_0 =
            tops::slice_async(dte_lhs_trans[0], private_lhs0_trans, hbm_lhs,
                              {0, batch_offset, 0, m_hbm_offset});
        event_lhs0 = tops::transpose_async(dte_lhs[0], private_lhs0,
                                           private_lhs0_trans, {0, 1, 3, 2});
      } else {
        event_lhs0 = tops::slice_async(dte_lhs[0], private_lhs0, hbm_lhs,
                                       {0, batch_offset, m_hbm_offset, 0});
      }

      if (need_trans_rhs) {
        e_rhs_trans_0 =
            tops::slice_async(dte_rhs_trans[0], private_rhs0_trans, hbm_rhs,
                              {0, r_batch_offset, n_hbm_offset, 0});
        event_rhs0 = tops::transpose_async(dte_rhs[0], private_rhs0,
                                           private_rhs0_trans, {0, 1, 3, 2});
      } else {
        event_rhs0 = tops::slice_async(dte_rhs[0], private_rhs0, hbm_rhs,
                                       {0, r_batch_offset, 0, n_hbm_offset});
      }
      // load bias, scale
      if (enable_bias) {
        event_bias0 = tops::slice_async(dte_bias, private_bias0, hbm_bias,
                                        {n_hbm_offset});
      }
      if (enable_quant) {
        event_scale0 = tops::slice_async(dte_scale, private_scale0, hbm_scale,
                                         {0, n_hbm_offset});
        if (quant_type == 1) {
          event_zeros0 = tops::slice_async(dte_zeros, private_zeros0, hbm_zeros,
                                           {0, n_hbm_offset});
        }
      }
      int next_n_sip_idx_temp, next_subm_index, next_m_sip_idx_temp;
      for (auto subm_index = 0;
           subm_index < subm_count;
           subm_index++) {
        int subm_index_global;
        if (is_lhs_split) {
          subm_index_global =
              subm_index * lhs_loop_step + thread_idx;
        } else {
          subm_index_global = subm_index;
        }
        int subm_offset_global = subm_index_global * subm_size;
        for (auto subn_index = 0;
             subn_index < subn_count;
             subn_index++) {
          int subn_index_global;
          if (is_rhs_split) {
            subn_index_global =
                subn_index * rhs_loop_step + thread_idx;
          } else {
            subn_index_global = subn_index;
          }

          int subn_offset_global = subn_index_global * subn_size;

          auto subout_index =
              subm_index * subn_count +
              subn_index;
          auto cur_private_out = ((subout_index % 2) == 0)
                                     ? &private_out0
                                     : &private_out1;
          auto cur_private_out_ptr =
              ((subout_index % 2) == 0 ? buffer_private_out0
                                                   : buffer_private_out1);
          for (auto subk_index = 0;
               subk_index < subk_count;
               subk_index++) {
            auto global_loop_index =
                subm_index * subn_count *
                    subk_count +
                subn_index * subk_count +
                subk_index;
            auto next_private_lhs =
                (global_loop_index % 2) ? &private_lhs0 : &private_lhs1;
            auto next_private_rhs =
                (global_loop_index % 2) ? &private_rhs0 : &private_rhs1;
            auto next_private_out = (subout_index % 2) == 1
                                        ? &private_out0
                                        : &private_out1;
            auto next_private_lhs_trans = (global_loop_index % 2) == 1
                                              ? &private_lhs0_trans
                                              : &private_lhs1_trans;
            auto next_private_rhs_trans = (global_loop_index % 2) == 1
                                              ? &private_rhs0_trans
                                              : &private_rhs1_trans;
            auto next_private_bias =
                (global_loop_index % 2) ? &private_bias0 : &private_bias1;
            auto next_private_scale =
                (global_loop_index % 2) ? &private_scale0 : &private_scale1;
            auto next_private_zeros =
                (global_loop_index % 2) ? &private_zeros0 : &private_zeros1;

            int next_subk_index, next_subn_index,
                next_subm_index;

            if (subk_index + 1 < subk_count) {
              next_subk_index = subk_index + 1;
              next_subn_index = subn_index;
              next_subm_index = subm_index;
            } else {
              next_subk_index = 0;
              if (subn_index + 1 < subn_count) {
                next_subn_index = subn_index + 1;
                next_subm_index = subm_index;
              } else {
                next_subn_index = 0;
                next_subm_index = subm_index + 1;
              }
            }
            auto next_subk_offset_global =
                next_subk_index * subk_size;
            auto next_rhs_subk_offset_global =
                next_subk_index * rhs_subk_size;
            auto next_subn_offset_global =
                is_rhs_split ? ((next_subn_index *
                                     rhs_loop_step +
                                 thread_idx) *
                                subn_size)
                             : (next_subn_index * subn_size);
            auto next_subm_offset_global =
                is_lhs_split ? ((next_subm_index *
                                     lhs_loop_step +
                                 thread_idx) *
                                subm_size)
                             : (next_subm_index * subm_size);

            auto cur_event_lhs =
                (global_loop_index % 2) == 0 ? event_lhs0 : event_lhs1;
            tops::wait(cur_event_lhs);
            auto cur_event_rhs =
                (global_loop_index % 2) == 0 ? event_rhs0 : event_rhs1;
            tops::wait(cur_event_rhs);
            auto cur_event_bias =
                (global_loop_index % 2) == 0 ? event_bias0 : event_bias1;
            if (enable_bias) {
              tops::wait(cur_event_bias);
            }
            auto cur_event_scale =
                (global_loop_index % 2) == 0 ? event_scale0 : event_scale1;
            auto cur_event_zeros =
                (global_loop_index % 2) == 0 ? event_zeros0 : event_zeros1;
            if (enable_quant) {
              tops::wait(cur_event_scale);
              if (quant_type == 1) {
                tops::wait(cur_event_zeros);
              }
            }
            auto next_event_lhs =
                (global_loop_index % 2) == 1 ? event_lhs0 : event_lhs1;
            auto next_event_rhs =
                (global_loop_index % 2) == 1 ? event_rhs0 : event_rhs1;
            auto next_event_bias =
                (global_loop_index % 2) == 1 ? event_bias0 : event_bias1;
            auto next_event_scale =
                (global_loop_index % 2) == 1 ? event_scale0 : event_scale1;
            auto next_event_zeros =
                (global_loop_index % 2) == 1 ? event_zeros0 : event_zeros1;
            if ((next_subk_offset_global < K) &&
                (next_subn_offset_global < N) &&
                (next_subm_offset_global < M)) {
              if (need_trans_lhs) {
                e_lhs_trans_0 = tops::slice_async(
                    dte_lhs_trans[0], *next_private_lhs_trans, hbm_lhs,
                    {0, batch_offset, next_subk_offset_global,
                     next_subm_offset_global});
                next_event_lhs = tops::transpose_async(
                    dte_lhs[0], *next_private_lhs, *next_private_lhs_trans,
                    {0, 1, 3, 2});
              } else {
                next_event_lhs =
                    tops::slice_async(dte_lhs[0], *next_private_lhs, hbm_lhs,
                                      {0, batch_offset, next_subm_offset_global,
                                       next_subk_offset_global});
              }

              if (need_trans_rhs) {
                e_rhs_trans_0 = tops::slice_async(
                    dte_rhs_trans[0], *next_private_rhs_trans, hbm_rhs,
                    {0, r_batch_offset, next_subn_offset_global,
                     next_rhs_subk_offset_global});
                next_event_rhs = tops::transpose_async(
                    dte_rhs[0], *next_private_rhs, *next_private_rhs_trans,
                    {0, 1, 3, 2});
              } else {
                next_event_rhs = tops::slice_async(
                    dte_rhs[0], *next_private_rhs, hbm_rhs,
                    {0, r_batch_offset, next_rhs_subk_offset_global,
                     next_subn_offset_global});
              }
              if (enable_bias) {
                next_event_bias =
                    tops::slice_async(dte_bias, *next_private_bias, hbm_bias,
                                      {next_subn_offset_global});
              }
              if (enable_quant) {
                next_event_scale =
                    tops::slice_async(dte_scale, *next_private_scale, hbm_scale,
                                      {next_subk_offset_global / group_size,
                                       next_subn_offset_global});
                if (quant_type == 1) {
                next_event_zeros =
                    tops::slice_async(dte_zeros, *next_private_zeros, hbm_zeros,
                                      {next_subk_offset_global / group_size,
                                       next_subn_offset_global});
                }
              }
            }
            auto cur_private_lhs_ptr =
                ((global_loop_index % 2) == 0 ? buffer_sip_lhs0
                                              : buffer_sip_lhs1);
            auto cur_private_rhs_ptr = (global_loop_index % 2) == 0
                                           ? buffer_sip_rhs0
                                           : buffer_sip_rhs1;
            auto cur_private_bias_ptr = (global_loop_index % 2) == 0
                                            ? buffer_sip_bias0
                                            : buffer_sip_bias1;
            auto cur_private_scale_ptr = (global_loop_index % 2) == 0
                                             ? buffer_sip_scale0
                                             : buffer_sip_scale1;
            auto cur_private_zeros_ptr = (global_loop_index % 2) == 0
                                             ? buffer_sip_zeros0
                                             : buffer_sip_zeros1;
            if (quant_type == 1 || quant_type == 2) {
              if (quant_type == 1) {
                dequant(reinterpret_cast<lhs_t*>(private_rhs_requant_buff),
                        reinterpret_cast<unsigned char*>(cur_private_rhs_ptr),
                        reinterpret_cast<lhs_t*>(cur_private_scale_ptr),
                        reinterpret_cast<lhs_t*>(cur_private_zeros_ptr),
                        rhs_subk_size, subn_size, group_size);
              } else {
                for (int group_idx = 0; group_idx < k_group_num; group_idx++) {
                  int group_k = std::min(group_size, subk_size);
                  if (group_size == -1) {
                    if (K < subk_size) {
                      group_k = subk_size;
                    }
                  }
                lhs_t* group_dequant_rhs_ptr =
                    reinterpret_cast<lhs_t*>(private_rhs_requant_buff) +
                    group_idx * group_k * subn_size;
                char* group_rhs_ptr =
                    reinterpret_cast<char*>(cur_private_rhs_ptr) +
                    group_idx * group_k * subn_size;
                scale_t* group_scale_ptr =
                    cur_private_scale_ptr + group_idx * subn_size;
                mul_not_inplace<1, lhs_t, char, lhs_t>(
                  reinterpret_cast<lhs_t*>(group_dequant_rhs_ptr),
                  reinterpret_cast<char*>(group_rhs_ptr),
                  reinterpret_cast<lhs_t*>(group_scale_ptr), group_k,
                  subn_size);
                }
              }
            }

            auto store_flag =
                subk_index + 1 == subk_count ? 1 : 0;
            int acc_flag = (subk_index == 0) ? 0 : 1;

            out_t* dst_ptr = reinterpret_cast<out_t*>(cur_private_out_ptr);
            lhs_t* lhs_ptr = reinterpret_cast<lhs_t*>(cur_private_lhs_ptr);
            lhs_t* rhs_ptr =
                reinterpret_cast<lhs_t*>(((quant_type == 1) ||
                                            (quant_type == 2))
                                               ? private_rhs_requant_buff
                                               : cur_private_rhs_ptr);
            bias_t* bias_ptr =
                reinterpret_cast<bias_t*>(cur_private_bias_ptr);

            if (sip_m % 256 == 0) {
              matmul<256, MK_KN>(dst_ptr, lhs_ptr, rhs_ptr, bias_ptr, local_workspace,
                            subk_size, subn_size, acc_flag, store_flag, enable_bias,
                            vab_offset, launch_times);
            } else if (sip_m % 128 == 0) {
              matmul<128, MK_KN>(dst_ptr, lhs_ptr, rhs_ptr, bias_ptr, local_workspace,
                            subk_size, subn_size, acc_flag, store_flag, enable_bias,
                            vab_offset, launch_times);
            } else if (sip_m % 96 == 0) {
              matmul<96, MK_KN>(dst_ptr, lhs_ptr, rhs_ptr, bias_ptr, local_workspace,
                            subk_size, subn_size, acc_flag, store_flag, enable_bias,
                            vab_offset, launch_times);
            } else if (sip_m % 64 == 0) {
              matmul<64, MK_KN>(dst_ptr, lhs_ptr, rhs_ptr, bias_ptr, local_workspace,
                            subk_size, subn_size, acc_flag, store_flag, enable_bias,
                            vab_offset, launch_times);
            } else if (sip_m % 32 == 0) {
              matmul<32, MK_KN>(dst_ptr, lhs_ptr, rhs_ptr, bias_ptr, local_workspace,
              subk_size, subn_size, acc_flag, store_flag, enable_bias,
              vab_offset, launch_times);
            } else {
              matmul<1, MK_KN>(dst_ptr, lhs_ptr, rhs_ptr, bias_ptr, local_workspace,
                subk_size, subn_size, acc_flag, store_flag, enable_bias,
                vab_offset, launch_times);
            }

            launch_times += 1;
          }  // K loop
          __dtu_movs_barrier_all();
          #if __GCU_ARCH__ == 400
                  tcle::fence<0xd2>();
          #elif __GCU_ARCH__ == 300
                  tcle::fence<0x3>();
          #endif
          e_out = tops::deslice_async(
              dte_out, hbm_out, *cur_private_out,
              {0, batch_offset, subm_offset_global, subn_offset_global});
          tops::wait(e_out);
        }  // N loop
      }    // M loop
    }      // batch loop
  }        // sip_id branch

  dte_lhs[0].destroy();
  dte_lhs[1].destroy();
  dte_rhs[0].destroy();
  dte_rhs[1].destroy();
  dte_lhs_trans[0].destroy();
  dte_lhs_trans[1].destroy();
  dte_rhs_trans[0].destroy();
  dte_rhs_trans[1].destroy();
  dte_out.destroy();
  dte_bias.destroy();
  dte_scale.destroy();
  dte_zeros.destroy();
}  // func


#define MATMUL_OP(TYPE, FN_NAME) \
extern "C" __global__ void FN_NAME(TYPE *in_a, TYPE *in_b, TYPE *out,\
                                            int input_dtype, int input_batch,\
                                            int input_m, int input_k, int input_n,\
                                            int lhs_multicore, int rhs_multicore, int batch_multicore,\
                                            int lhs_transpose, int rhs_transpose,\
                                            float alpha, float beta, float addmm_beta, \
                                            int sip_m, int sip_k, int sip_n, int broadcasted_weight) {\
    __local__ __valigned__ char buffer_sip[VDMEM_VALID_SIZE];\
    extern __shared__ char l2_buffer[];\
    int32_t M_align = CeilDiv(input_m, sip_m) * sip_m;\
    int32_t K_align = CeilDiv(input_k, sip_k) * sip_k; \
    int LSIZE = M_align * K_align * sizeof(TYPE); \
    int R_SIP_SIZE = (sip_k * sip_n * 2 + M_align * sip_n * 2) * sizeof(TYPE);\
    if (LSIZE + R_SIP_SIZE < VDMEM_VALID_SIZE) { \
      if (rhs_transpose > 0 ) { \
        matmul_kernel_trans_avoid<TYPE, TYPE, TYPE, TYPE, TYPE>(in_a, in_b, out, out, out, out, input_dtype, input_batch, input_m, input_k, input_n, lhs_multicore, rhs_multicore, batch_multicore, \
          lhs_transpose, rhs_transpose, alpha, beta, addmm_beta, sip_m, sip_k, sip_n, broadcasted_weight, -1, buffer_sip, l2_buffer);\
      } else { \
        matmul_kernel_lhs_l1<TYPE, TYPE, TYPE, TYPE, TYPE>(in_a, in_b, out, out, out, out, input_dtype, input_batch, input_m, input_k, input_n, lhs_multicore, rhs_multicore, batch_multicore, \
          lhs_transpose, rhs_transpose, alpha, beta, addmm_beta, sip_m, sip_k, sip_n, broadcasted_weight, -1, buffer_sip, l2_buffer);\
      }\
    } else \
      matmul_kernel_batch<TYPE, TYPE, TYPE, TYPE, TYPE>(in_a, in_b, out, out, out, out, input_dtype, input_batch, input_m, input_k, input_n, lhs_multicore, rhs_multicore, batch_multicore, \
        lhs_transpose, rhs_transpose, alpha, beta, addmm_beta, sip_m, sip_k, sip_n, broadcasted_weight, -1, buffer_sip);\
}\


extern "C" __global__ void matmul_f32(float *in_a, float *in_b, float *out, 
                                            int input_dtype, int input_batch,
                                            int input_m, int input_k, int input_n,
                                            int lhs_multicore, int rhs_multicore, int batch_multicore,
                                            int lhs_transpose, int rhs_transpose,
                                            float alpha, float beta, float addmm_beta, 
                                            int sip_m, int sip_k, int sip_n, int broadcasted_weight) {
    __local__ __valigned__ char buffer_sip[VDMEM_VALID_SIZE];
    matmul_kernel<float, float>(in_a, in_b, out, input_dtype, input_batch, input_m, input_k, input_n, lhs_multicore, rhs_multicore, batch_multicore, 
        lhs_transpose, rhs_transpose, alpha, beta, addmm_beta, sip_m, sip_k, sip_n, broadcasted_weight, buffer_sip);
}

MATMUL_OP(__fp16, matmul_f16)
MATMUL_OP(__bf16, matmul_bf16)

#define MATMUL_OP_QUANT(TYPE, TYPE_WEIGHT, TYPE_SCALE, FN_NAME) \
extern "C" __global__ void FN_NAME(TYPE *in_a, TYPE_WEIGHT *in_b, TYPE *out, \
                                            TYPE *scales, TYPE *zeros, \
                                            int input_dtype, int input_batch,\
                                            int input_m, int input_k, int input_n,\
                                            int lhs_multicore, int rhs_multicore, int batch_multicore,\
                                            int lhs_transpose, int rhs_transpose,\
                                            float alpha, float beta, float addmm_beta, \
                                            int sip_m, int sip_k, int sip_n, int broadcasted_weight, int group_size) {\
    __local__ __valigned__ char buffer_sip[VDMEM_VALID_SIZE];\
    extern __shared__ char l2_buffer[];\
    int32_t M_align = CeilDiv(input_m, sip_m) * sip_m;\
    int32_t K_align = CeilDiv(input_k, sip_k) * sip_k; \
    int LSIZE = M_align * K_align * sizeof(TYPE); \
    int SCALE_ZERO_SIZE = group_size > 0 ? (K_align / group_size) * sip_n * sizeof(TYPE) * 2 * 2 : 0; \
    int R_SIP_SIZE = sip_k * sip_n * 2 * sizeof(TYPE) + M_align * sip_n * 2 * sizeof(TYPE); \
    int DUMMY_SIZE = 32 * 1024; \
    if (LSIZE + R_SIP_SIZE + SCALE_ZERO_SIZE + DUMMY_SIZE < VDMEM_VALID_SIZE) { \
      matmul_kernel_lhs_l1<TYPE, TYPE_WEIGHT, TYPE, TYPE, TYPE>(in_a, in_b, out, out, scales, zeros, input_dtype, input_batch, input_m, input_k, input_n, lhs_multicore, rhs_multicore, batch_multicore, \
        lhs_transpose, rhs_transpose, alpha, beta, addmm_beta, sip_m, sip_k, sip_n, broadcasted_weight, group_size, buffer_sip, l2_buffer);\
    } else \
      matmul_kernel_batch<TYPE, TYPE_WEIGHT, TYPE, TYPE, TYPE>(in_a, in_b, out, out, scales, zeros, input_dtype, input_batch, input_m, input_k, input_n, lhs_multicore, rhs_multicore, batch_multicore, \
        lhs_transpose, rhs_transpose, alpha, beta, addmm_beta, sip_m, sip_k, sip_n, broadcasted_weight, group_size, buffer_sip);\
}\

MATMUL_OP_QUANT(__fp16, uint8_t, __fp16, matmul_f16_4bit)
MATMUL_OP_QUANT(__bf16, uint8_t, __bf16, matmul_bf16_4bit)

MATMUL_OP_QUANT(__fp16, int8_t, __fp16, matmul_f16_8bit)
MATMUL_OP_QUANT(__bf16, int8_t, __bf16, matmul_bf16_8bit)

int main() {}