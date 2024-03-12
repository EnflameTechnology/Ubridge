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
#include "matmul_general.h"
#include "tops/tops_runtime.h"
#include "utils.h"
#include <acore/acore_op.h>

#define SHARE_BUFFER_SIZE 1024 * 1024 * 24 //24MB

using namespace std;
#define MAX_NUM 1540096 - 1024
#define TEMPLATE_ALIGN_UP(a, b) (((a + b - 1) / b) * b)
#define L1_ALIGN_SIZE (128)
typedef enum {
  TOPSOP_DATA_NONE = -1,  /**< TOPSOP_DATA_NONE -1  */
  TOPSOP_DATA_I8 = 0,     /**< TOPSOP_DATA_I8 0  */
  TOPSOP_DATA_U8,         /**< TOPSOP_DATA_U8 1  */
  TOPSOP_DATA_I16,        /**< TOPSOP_DATA_I16 2  */
  TOPSOP_DATA_U16,        /**< TOPSOP_DATA_U16 3  */
  TOPSOP_DATA_FP16,       /**< TOPSOP_DATA_FP16 4  */
  TOPSOP_DATA_BF16,       /**< TOPSOP_DATA_BF16 5  */
  TOPSOP_DATA_I32,        /**< TOPSOP_DATA_I32 6  */
  TOPSOP_DATA_U32,        /**< TOPSOP_DATA_U32 7  */
  TOPSOP_DATA_FP32,       /**< TOPSOP_DATA_FP32 8  */
  TOPSOP_DATA_EF32,       /**< TOPSOP_DATA_EF32 9  */
  TOPSOP_DATA_TF32,       /**< TOPSOP_DATA_TF32 10  */
  TOPSOP_DATA_I64,        /**< TOPSOP_DATA_I64 11  */
  TOPSOP_DATA_U64,        /**< TOPSOP_DATA_U64 12  */
  TOPSOP_DATA_F64,        /**< TOPSOP_DATA_F64 13  */
  TOPSOP_DATA_PRED,       /**< TOPSOP_DATA_PRED 14  */
  TOPSOP_DATA_I4,         /**< TOPSOP_DATA_I4 15  */
} topsopDataType_t;


template <typename lhs_t, typename rhs_t>
__device__ __forceinline__ void matmul_kernel(lhs_t *lhs, rhs_t *rhs, rhs_t *out,
      int input_dtype, int input_batch,
      int input_m, int input_k, int input_n,
      int lhs_multicore, int rhs_multicore, int batch_multicore,
      int lhs_transpose, int rhs_transpose,
      float alpha, float beta, float addmm_beta, 
      int sip_m, int sip_k, int sip_n, int broadcasted_weight, char* buffer_sip, char* raw_cache) {
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
  int32_t sip_lhs_trans_size = sip_lhs_trans_shape[0] * sip_lhs_trans_shape[1] *
                               sip_lhs_trans_shape[2] * sip_lhs_trans_shape[3] *
                               sizeof(lhs_t);
  int32_t sip_rhs_trans_size = sip_rhs_trans_shape[0] * sip_rhs_trans_shape[1] *
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

  lhs_t *buffer_sip_out0_f16 = reinterpret_cast<lhs_t *>(buffer_private_out0);
  tops::mdspan sip_out0_f16(tops::Private, buffer_sip_out0_f16, sip_out_shape);
  lhs_t *buffer_sip_out1_f16 = reinterpret_cast<lhs_t *>(buffer_private_out1);
  tops::mdspan sip_out1_f16(tops::Private, buffer_sip_out1_f16, sip_out_shape);

//   tops::mdspan sip_bias0(tops::Private, buffer_sip_bias0, N);
  auto M_SIP_LOOP_CNT_TASKS = M / subm_size + (M % subm_size > 0 ? 1 : 0);
  auto N_SIP_LOOP_CNT_TASKS = N / subn_size + (N % subn_size > 0 ? 1 : 0);
  auto subk_count_each_thread = K / subk_size + (K % subk_size > 0 ? 1 : 0);
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
  auto lhs_loop_step_each_thread = (is_lhs_split == 1) ? sip_num_used : 1;
  auto rhs_loop_step_each_thread = (is_rhs_split == 1) ? sip_num_used : 1;
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
  auto subm_count_each_thread =
      is_lhs_split == 1 ? loop_len_this_sip : M_SIP_LOOP_CNT_TASKS;
  auto subn_count_each_thread =
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
      int next_n_sip_idx_temp, next_subm_index_each_thread, next_m_sip_idx_temp;

      for (auto subm_index_each_thread = 0;
           subm_index_each_thread < subm_count_each_thread;
           subm_index_each_thread++) {
        int subm_index_global;
        if (is_lhs_split) {
          subm_index_global =
              subm_index_each_thread * lhs_loop_step_each_thread + thread_idx;
        } else {
          subm_index_global = subm_index_each_thread;
        }
        int subm_offset_global = subm_index_global * subm_size;
        for (auto subn_index_each_thread = 0;
             subn_index_each_thread < subn_count_each_thread;
             subn_index_each_thread++) {
          int subn_index_global;
          if (is_rhs_split) {
            subn_index_global =
                subn_index_each_thread * rhs_loop_step_each_thread + thread_idx;
          } else {
            subn_index_global = subn_index_each_thread;
          }

          int subn_offset_global = subn_index_global * subn_size;

          auto subout_index_each_thread =
              subm_index_each_thread * subn_count_each_thread +
              subn_index_each_thread;
          auto cur_private_out = ((subout_index_each_thread % 2) == 0)
                                     ? &private_out0
                                     : &private_out1;
          cur_out_addr =
              (long)((subout_index_each_thread % 2) == 0 ? buffer_private_out0
                                                         : buffer_private_out1);
          auto cur_private_output_ptr =
              ((subout_index_each_thread % 2) == 0 ? buffer_private_out0
                                                   : buffer_private_out1);
          for (auto subk_index_each_thread = 0;
               subk_index_each_thread < subk_count_each_thread;
               subk_index_each_thread++) {
            // no parallel split k
            // auto k_sip_offset = subk_index_each_thread * subk_size;
            auto global_loop_index =
                subm_index_each_thread * subn_count_each_thread *
                    subk_count_each_thread +
                subn_index_each_thread * subk_count_each_thread +
                subk_index_each_thread;

            auto next_private_lhs =
                (global_loop_index % 2) ? &private_lhs0 : &private_lhs1;
            auto next_private_rhs =
                (global_loop_index % 2) ? &private_rhs0 : &private_rhs1;
            auto next_private_out = (subout_index_each_thread % 2) == 1
                                        ? &private_out0
                                        : &private_out1;
            auto next_private_lhs_trans = (global_loop_index % 2) == 1
                                              ? &private_lhs0_trans
                                              : &private_lhs1_trans;
            auto next_private_rhs_trans = (global_loop_index % 2) == 1
                                              ? &private_rhs0_trans
                                              : &private_rhs1_trans;

            int next_subk_index_each_thread, next_subn_index_each_thread,
                next_subm_index_each_thread;

            if (subk_index_each_thread + 1 < subk_count_each_thread) {
              next_subk_index_each_thread = subk_index_each_thread + 1;
              next_subn_index_each_thread = subn_index_each_thread;
              next_subm_index_each_thread = subm_index_each_thread;
            } else {
              // subk_index_each_thread + 1 == subk_count_each_thread
              next_subk_index_each_thread = 0;
              if (subn_index_each_thread + 1 < subn_count_each_thread) {
                next_subn_index_each_thread = subn_index_each_thread + 1;
                next_subm_index_each_thread = subm_index_each_thread;
              } else {
                // subn_index_each_thread + 1 == subn_count_each_thread
                next_subn_index_each_thread = 0;
                next_subm_index_each_thread = subm_index_each_thread + 1;
              }
            }

            auto next_subk_offset_global =
                next_subk_index_each_thread * subk_size;
            // auto next_subn_offset_global = next_subn_index_each_thread *
            // subn_size;
            auto next_subn_offset_global =
                is_rhs_split ? ((next_subn_index_each_thread *
                                     rhs_loop_step_each_thread +
                                 thread_idx) *
                                subn_size)
                             : (next_subn_index_each_thread * subn_size);
            // auto next_subm_offset_global = next_subm_index_each_thread *
            // subm_size;
            auto next_subm_offset_global =
                is_lhs_split ? ((next_subm_index_each_thread *
                                     lhs_loop_step_each_thread +
                                 thread_idx) *
                                subm_size)
                             : (next_subm_index_each_thread * subm_size);

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

            auto nacc_flag = subk_index_each_thread == 0 ? 1 : 0;
            auto store_flag =
                subk_index_each_thread + 1 == subk_count_each_thread ? 1 : 0;
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
              if (weight_data_type == TOPSOP_DATA_I8) {
                convert(reinterpret_cast<__fp16 *>(cur_buffer_sip_rhs),
                        reinterpret_cast<char *>(cur_buffer_sip_rhs +
                                                 subk_size * subn_size),
                        subk_size * subn_size);
                c_func_hmatmul_general(
                    lhs_addr, rhs_addr, cur_out_addr, subm_size, subn_size,
                    subk_size);
              } else {
                c_func_hmatmul_general(
                    lhs_addr, rhs_addr, cur_out_addr, subm_size, subn_size,
                    subk_size);
              }
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
            if (weight_data_type == TOPSOP_DATA_I8) {
              if (enable_quant == 1) {
                mul_fp32_scalar(cur_private_output_ptr, cur_private_output_ptr,
                                *reinterpret_cast<float *>(&scale),
                                subm_size * subn_size);
              }
            }
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

template <typename lhs_t, typename rhs_t, int SUBM>
__device__ __forceinline__ void matmul_kernel_aligned(lhs_t *lhs, rhs_t *rhs, rhs_t *out,
      int input_dtype, int input_batch,
      int input_m, int input_k, int input_n,
      int lhs_multicore, int rhs_multicore, int batch_multicore,
      int lhs_transpose, int rhs_transpose,
      float alpha, float beta, float addmm_beta, 
      int sip_m, int sip_k, int sip_n, int broadcasted_weight, char* buffer_sip, char* raw_cache) {
  int thread_num = GetThreadNum();
  int thread_id = GetThreadIdx();
  int sip_cnt_raw = thread_num;
  int thread_idx = thread_id;
  tops_dte_ctx_t dte_lhs[2];
  tops_dte_ctx_t dte_rhs[2];
  tops_dte_ctx_t dte_lhs_trans[2];
  tops_dte_ctx_t dte_rhs_trans[2];
  tops_dte_ctx_t dte_out;
  tops_dte_ctx_t dte_cache;
  // tops_dte_ctx_t dte_scale;
  // tops_dte_ctx_t dte_b;

  dte_lhs[0].init();
  dte_lhs[1].init();
  dte_rhs[0].init();
  dte_rhs[1].init();
  dte_lhs_trans[0].init();
  dte_lhs_trans[1].init();
  dte_rhs_trans[0].init();
  dte_rhs_trans[1].init();
  dte_out.init();
  dte_cache.init();
  // dte_bias.init();
  // dte_scale.init();

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

  int enable_bias = 0;
  int enable_act = 0;
  auto subm_size = sip_m;
  auto subn_size = sip_n;
  auto subk_size = sip_k;
  int enable_quant = 0;

  auto need_trans_lhs = transa;
  auto need_trans_rhs = transb;
  int32_t hbm_lhs_shape[4] = {1, B, M, K};
  int32_t hbm_rhs_shape[4] = {1, broadcasted_weight > 0 ? 1 : B, K, N};
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
    hbm_rhs_shape[3] = K;
  }
  int32_t sip_lhs_shape[4] = {1, 1, subm_size, subk_size};
  int32_t sip_rhs_shape[4] = {1, 1, subk_size, subn_size};
  int32_t sip_out_shape[4] = {1, 1, subm_size, subn_size};
  int32_t sip_lhs_trans_shape[4] = {1, 1, subk_size, subm_size};
  int32_t sip_rhs_trans_shape[4] = {1, 1, subn_size, subk_size};

  int32_t sip_bias_shape[1] = {subn_size};
  int32_t sip_scale_shape[1] = {subn_size};
  int32_t sip_lhs_size = sip_lhs_shape[0] * sip_lhs_shape[1] *
                         sip_lhs_shape[2] * sip_lhs_shape[3] * sizeof(lhs_t);
  int32_t sip_rhs_size = sip_rhs_shape[0] * sip_rhs_shape[1] *
                         sip_rhs_shape[2] * sip_rhs_shape[3] * sizeof(lhs_t);
  int32_t sip_out_size = sip_out_shape[0] * sip_out_shape[1] *
                         sip_out_shape[2] * sip_out_shape[3] *
                         sizeof(lhs_t);  // use fp32
  int32_t sip_lhs_trans_size = sip_lhs_size;
  int32_t sip_rhs_trans_size = sip_rhs_size;
  int32_t sip_bias_size = 0;
  int32_t sip_scale_size = 0;
  if (enable_bias) {
    sip_bias_size = subn_size * sizeof(lhs_t);
  }
  if (enable_quant) {
    sip_scale_size = subn_size * sizeof(lhs_t);
  }
  tops::mdspan hbm_lhs(tops::Global, lhs, hbm_lhs_shape);
  tops::mdspan hbm_rhs(tops::Global, rhs, hbm_rhs_shape);
  tops::mdspan hbm_out(tops::Global, out, hbm_out_shape);
  int right_size = hbm_rhs_shape[1] * hbm_rhs_shape[2] * hbm_rhs_shape[3];
  tops::mdspan cached_rhs(tops::Shared, reinterpret_cast<rhs_t*>(raw_cache), hbm_rhs_shape);
  bool weight_cache = right_size * sizeof(rhs_t) < SHARE_BUFFER_SIZE;
  if (thread_id == 0 && weight_cache) {
    tops::memcpy(dte_cache, cached_rhs, hbm_rhs);
  }
  __syncthreads();
  // tops::mdspan hbm_pre_gelu(tops::Global, pre_gelu, hbm_out_shape);
  // tops::mdspan hbm_bias(tops::Global, bias, N);
  // tops::mdspan hbm_scale(tops::Global, scale, N);
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
  rhs_t* buffer_sip_rhs0 =
      reinterpret_cast<rhs_t*>((reinterpret_cast<char*>(buffer_sip_lhs1)) +
                               TEMPLATE_ALIGN_UP(sip_lhs_size, L1_ALIGN_SIZE));
  rhs_t* buffer_sip_rhs1 =
      reinterpret_cast<rhs_t*>((reinterpret_cast<char*>(buffer_sip_rhs0)) +
                               TEMPLATE_ALIGN_UP(sip_rhs_size, L1_ALIGN_SIZE));
  lhs_t* buffer_private_out0 =
      reinterpret_cast<lhs_t*>((reinterpret_cast<char*>(buffer_sip_rhs1)) +
                               TEMPLATE_ALIGN_UP(sip_rhs_size, L1_ALIGN_SIZE));
  lhs_t* buffer_private_out1 =
      reinterpret_cast<lhs_t*>((reinterpret_cast<char*>(buffer_private_out0)) +
                               TEMPLATE_ALIGN_UP(sip_out_size, L1_ALIGN_SIZE));
  // lhs_t* buffer_sip_bias0 =
  //     reinterpret_cast<lhs_t*>((reinterpret_cast<char*>(buffer_private_out1)) +
  //                              TEMPLATE_ALIGN_UP(sip_out_size, L1_ALIGN_SIZE));
  // lhs_t* buffer_sip_bias1 =
  //     reinterpret_cast<lhs_t*>((reinterpret_cast<char*>(buffer_sip_bias0)) +
  //                              TEMPLATE_ALIGN_UP(sip_bias_size, L1_ALIGN_SIZE));
  // lhs_t* buffer_sip_scale0 =
  //     reinterpret_cast<lhs_t*>((reinterpret_cast<char*>(buffer_sip_bias1)) +
  //                              TEMPLATE_ALIGN_UP(sip_bias_size, L1_ALIGN_SIZE));
  // lhs_t* buffer_sip_scale1 = reinterpret_cast<lhs_t*>(
  //     (reinterpret_cast<char*>(buffer_sip_scale0)) +
  //     TEMPLATE_ALIGN_UP(sip_scale_size, L1_ALIGN_SIZE));
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
  // tops::mdspan private_bias0(tops::Private, buffer_sip_bias0, subn_size);
  // tops::mdspan private_bias1(tops::Private, buffer_sip_bias1, subn_size);
  // tops::mdspan private_scale0(tops::Private, buffer_sip_scale0, subn_size);
  // tops::mdspan private_scale1(tops::Private, buffer_sip_scale1, subn_size);
  auto M_SIP_LOOP_CNT_TASKS = M / subm_size + (M % subm_size > 0 ? 1 : 0);
  auto N_SIP_LOOP_CNT_TASKS = N / subn_size + (N % subn_size > 0 ? 1 : 0);
  auto subk_count_each_thread = K / subk_size + (K % subk_size > 0 ? 1 : 0);
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
  auto lhs_loop_step_each_thread = (is_lhs_split == 1) ? sip_num_used : 1;
  auto rhs_loop_step_each_thread = (is_rhs_split == 1) ? sip_num_used : 1;
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
  auto subm_count_each_thread =
      is_lhs_split == 1 ? loop_len_this_sip : M_SIP_LOOP_CNT_TASKS;
  auto subn_count_each_thread =
      is_rhs_split == 1 ? loop_len_this_sip : N_SIP_LOOP_CNT_TASKS;
  auto Batch_SIP_LOOP_CNT = (is_batch_split == 1) ? loop_len_this_sip : B;
  int vab_offset = (enable_quant == 1) ? 256 : 0;
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
            tops::slice_async(dte_rhs_trans[0], private_rhs0_trans, weight_cache ? cached_rhs: hbm_rhs,
                              {0, r_batch_offset, n_hbm_offset, 0});
        event_rhs0 = tops::transpose_async(dte_rhs[0], private_rhs0,
                                           private_rhs0_trans, {0, 1, 3, 2});
      } else {
        event_rhs0 = tops::slice_async(dte_rhs[0], private_rhs0, weight_cache ? cached_rhs: hbm_rhs,
                                       {0, r_batch_offset, 0, n_hbm_offset});
      }
      // load bias, scale
      // if (enable_bias) {
      //   event_bias0 = tops::slice_async(dte_bias, private_bias0, hbm_bias,
      //                                   {n_hbm_offset});
      // }
      // if (enable_quant) {
      //   event_scale0 = tops::slice_async(dte_scale, private_scale0, hbm_scale,
      //                                    {n_hbm_offset});
      // }
      int next_n_sip_idx_temp, next_subm_index_each_thread, next_m_sip_idx_temp;
      for (auto subm_index_each_thread = 0;
           subm_index_each_thread < subm_count_each_thread;
           subm_index_each_thread++) {
        int subm_index_global;
        if (is_lhs_split) {
          subm_index_global =
              subm_index_each_thread * lhs_loop_step_each_thread + thread_idx;
        } else {
          subm_index_global = subm_index_each_thread;
        }
        int subm_offset_global = subm_index_global * subm_size;
        for (auto subn_index_each_thread = 0;
             subn_index_each_thread < subn_count_each_thread;
             subn_index_each_thread++) {
          int subn_index_global;
          if (is_rhs_split) {
            subn_index_global =
                subn_index_each_thread * rhs_loop_step_each_thread + thread_idx;
          } else {
            subn_index_global = subn_index_each_thread;
          }

          int subn_offset_global = subn_index_global * subn_size;

          auto subout_index_each_thread =
              subm_index_each_thread * subn_count_each_thread +
              subn_index_each_thread;
          auto cur_private_out = ((subout_index_each_thread % 2) == 0)
                                     ? &private_out0
                                     : &private_out1;
          auto cur_private_out_ptr =
              ((subout_index_each_thread % 2) == 0 ? buffer_private_out0
                                                   : buffer_private_out1);
          for (auto subk_index_each_thread = 0;
               subk_index_each_thread < subk_count_each_thread;
               subk_index_each_thread++) {
            auto global_loop_index =
                subm_index_each_thread * subn_count_each_thread *
                    subk_count_each_thread +
                subn_index_each_thread * subk_count_each_thread +
                subk_index_each_thread;

            auto next_private_lhs =
                (global_loop_index % 2) ? &private_lhs0 : &private_lhs1;
            auto next_private_rhs =
                (global_loop_index % 2) ? &private_rhs0 : &private_rhs1;
            auto next_private_out = (subout_index_each_thread % 2) == 1
                                        ? &private_out0
                                        : &private_out1;
            auto next_private_lhs_trans = (global_loop_index % 2) == 1
                                              ? &private_lhs0_trans
                                              : &private_lhs1_trans;
            auto next_private_rhs_trans = (global_loop_index % 2) == 1
                                              ? &private_rhs0_trans
                                              : &private_rhs1_trans;
            // auto next_private_bias =
            //     (global_loop_index % 2) ? &private_bias0 : &private_bias1;
            // auto next_private_scale =
            //     (global_loop_index % 2) ? &private_scale0 : &private_scale1;

            int next_subk_index_each_thread, next_subn_index_each_thread,
                next_subm_index_each_thread;

            if (subk_index_each_thread + 1 < subk_count_each_thread) {
              next_subk_index_each_thread = subk_index_each_thread + 1;
              next_subn_index_each_thread = subn_index_each_thread;
              next_subm_index_each_thread = subm_index_each_thread;
            } else {
              next_subk_index_each_thread = 0;
              if (subn_index_each_thread + 1 < subn_count_each_thread) {
                next_subn_index_each_thread = subn_index_each_thread + 1;
                next_subm_index_each_thread = subm_index_each_thread;
              } else {
                next_subn_index_each_thread = 0;
                next_subm_index_each_thread = subm_index_each_thread + 1;
              }
            }

            auto next_subk_offset_global =
                next_subk_index_each_thread * subk_size;

            auto next_subn_offset_global =
                is_rhs_split ? ((next_subn_index_each_thread *
                                     rhs_loop_step_each_thread +
                                 thread_idx) *
                                subn_size)
                             : (next_subn_index_each_thread * subn_size);
            auto next_subm_offset_global =
                is_lhs_split ? ((next_subm_index_each_thread *
                                     lhs_loop_step_each_thread +
                                 thread_idx) *
                                subm_size)
                             : (next_subm_index_each_thread * subm_size);

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
            if (enable_quant) {
              tops::wait(cur_event_scale);
            }

            auto next_event_lhs =
                (global_loop_index % 2) == 1 ? event_lhs0 : event_lhs1;
            auto next_event_rhs =
                (global_loop_index % 2) == 1 ? event_rhs0 : event_rhs1;
            auto next_event_bias =
                (global_loop_index % 2) == 1 ? event_bias0 : event_bias1;
            auto next_event_scale =
                (global_loop_index % 2) == 1 ? event_scale0 : event_scale1;

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
                    dte_rhs_trans[0], *next_private_rhs_trans, weight_cache ? cached_rhs: hbm_rhs,
                    {0, r_batch_offset, next_subn_offset_global,
                     next_subk_offset_global});
                next_event_rhs = tops::transpose_async(
                    dte_rhs[0], *next_private_rhs, *next_private_rhs_trans,
                    {0, 1, 3, 2});
              } else {
                next_event_rhs =
                    tops::slice_async(dte_rhs[0], *next_private_rhs, weight_cache ? cached_rhs: hbm_rhs,
                                      {0, r_batch_offset, next_subk_offset_global,
                                       next_subn_offset_global});
              }
              // if (enable_bias) {
              //   next_event_bias =
              //       tops::slice_async(dte_bias, *next_private_bias, hbm_bias,
              //                         {next_subn_offset_global});
              // }
              // if (enable_quant) {
              //   next_event_scale =
              //       tops::slice_async(dte_scale, *next_private_scale, hbm_scale,
              //                         {next_subn_offset_global});
              // }
            }
            auto cur_private_lhs_ptr =
                ((global_loop_index % 2) == 0 ? buffer_sip_lhs0
                                              : buffer_sip_lhs1);
            auto cur_private_rhs_ptr = (global_loop_index % 2) == 0
                                           ? buffer_sip_rhs0
                                           : buffer_sip_rhs1;
            // auto cur_private_bias_ptr = (global_loop_index % 2) == 0
            //                                 ? buffer_sip_bias0
            //                                 : buffer_sip_bias1;
            // auto cur_private_scale_ptr = (global_loop_index % 2) == 0
            //                                  ? buffer_sip_scale0
            //                                  : buffer_sip_scale1;
            // if (weight_data_type == TOPSOP_DATA_I8) {
            //   mul<1>(reinterpret_cast<__fp16*>(cur_private_rhs_ptr),
            //          reinterpret_cast<char*>(cur_private_rhs_ptr +
            //                                  subk_size * subn_size),
            //          reinterpret_cast<__fp16*>(cur_private_scale_ptr),
            //          subk_size, subn_size);
            // }
            auto store_flag =
                subk_index_each_thread + 1 == subk_count_each_thread ? 1 : 0;
            int acc_flag = (subk_index_each_thread == 0) ? 0 : 1;
            using u_t = typename UnderlyingType<lhs_t>::type;
            u_t* dst_ptr = reinterpret_cast<u_t*>(cur_private_out_ptr);
            u_t* lhs_ptr = reinterpret_cast<u_t*>(cur_private_lhs_ptr);
            u_t* rhs_ptr = reinterpret_cast<u_t*>(cur_private_rhs_ptr);
            // u_t* bias_ptr = reinterpret_cast<u_t*>(cur_private_bias_ptr);
            matmul<SUBM>(dst_ptr, lhs_ptr, rhs_ptr, rhs_ptr, local_workspace,
                         subk_size, subn_size, acc_flag, store_flag,
                         enable_bias, vab_offset, launch_times);
            launch_times += 1;
          }  // K loop
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
  dte_cache.destroy();
  // dte_bias.destroy();
  // dte_scale.destroy();
}  // func

extern "C" __global__ void matmul_f32(float *in_a, float *in_b, float *out, 
                                            int input_dtype, int input_batch,
                                            int input_m, int input_k, int input_n,
                                            int lhs_multicore, int rhs_multicore, int batch_multicore,
                                            int lhs_transpose, int rhs_transpose,
                                            float alpha, float beta, float addmm_beta, 
                                            int sip_m, int sip_k, int sip_n, int broadcasted_weight) {
    __local__ __valigned__ char buffer_sip[VDMEM_SIZE];
    __shared__ char raw_cache[SHARE_BUFFER_SIZE];
    matmul_kernel<float, float>(in_a, in_b, out, input_dtype, input_batch, input_m, input_k, input_n, lhs_multicore, rhs_multicore, batch_multicore, 
        lhs_transpose, rhs_transpose, alpha, beta, addmm_beta, sip_m, sip_k, sip_n, broadcasted_weight, buffer_sip, raw_cache);
}

extern "C" __global__ void matmul_f16(tops::half *in_a, tops::half *in_b, tops::half *out,  
                                                int input_dtype, int input_batch,
                                                int input_m, int input_k, int input_n,
                                                int lhs_multicore, int rhs_multicore, int batch_multicore,
                                                int lhs_transpose, int rhs_transpose,
                                                float alpha, float beta, float addmm_beta, 
                                                int sip_m, int sip_k, int sip_n, int broadcasted_weight) {
    __local__ __valigned__ char buffer_sip[VDMEM_SIZE];
    __shared__ char raw_cache[SHARE_BUFFER_SIZE];
    if (sip_m % 64 == 0 || input_m % 64 == 0)
      matmul_kernel_aligned<tops::half, tops::half, 64>(in_a, in_b, out, input_dtype, input_batch, input_m, input_k, input_n, lhs_multicore, rhs_multicore, batch_multicore, 
        lhs_transpose, rhs_transpose, alpha, beta, addmm_beta, sip_m, sip_k, sip_n, broadcasted_weight, buffer_sip, raw_cache);
    else if (sip_m % 32 == 0 || input_m % 32 == 0)
      matmul_kernel_aligned<tops::half, tops::half, 32>(in_a, in_b, out, input_dtype, input_batch, input_m, input_k, input_n, lhs_multicore, rhs_multicore, batch_multicore, 
        lhs_transpose, rhs_transpose, alpha, beta, addmm_beta, sip_m, sip_k, sip_n, broadcasted_weight, buffer_sip, raw_cache);
    else
      matmul_kernel<tops::half, tops::half>(in_a, in_b, out, input_dtype, input_batch, input_m, input_k, input_n, lhs_multicore, rhs_multicore, batch_multicore, 
        lhs_transpose, rhs_transpose, alpha, beta, addmm_beta, sip_m, sip_k, sip_n, broadcasted_weight, buffer_sip, raw_cache);
}


extern "C" __global__ void matmul_bf16(tops::bfloat *in_a,
                                                   tops::bfloat *in_b,
                                                   tops::bfloat *out,
                                                  int input_dtype, int input_batch,
                                                  int input_m, int input_k, int input_n,
                                                  int lhs_multicore, int rhs_multicore, int batch_multicore,
                                                  int lhs_transpose, int rhs_transpose,
                                                  float alpha, float beta, float addmm_beta, 
                                                  int sip_m, int sip_k, int sip_n, int broadcasted_weight) {
    __local__ __valigned__ char buffer_sip[VDMEM_SIZE];
    __shared__ char raw_cache[SHARE_BUFFER_SIZE];
    if (sip_m % 64 == 0 || input_m % 64 == 0)
      matmul_kernel_aligned<tops::bfloat, tops::bfloat, 64>(in_a, in_b, out, input_dtype, input_batch, input_m, input_k, input_n, lhs_multicore, rhs_multicore, batch_multicore, 
        lhs_transpose, rhs_transpose, alpha, beta, addmm_beta, sip_m, sip_k, sip_n, broadcasted_weight, buffer_sip, raw_cache);
    else if (sip_m % 32 == 0 || input_m % 32 == 0)
      matmul_kernel_aligned<tops::bfloat, tops::bfloat, 32>(in_a, in_b, out, input_dtype, input_batch, input_m, input_k, input_n, lhs_multicore, rhs_multicore, batch_multicore, 
        lhs_transpose, rhs_transpose, alpha, beta, addmm_beta, sip_m, sip_k, sip_n, broadcasted_weight, buffer_sip, raw_cache);
    else
      matmul_kernel<tops::bfloat, tops::bfloat>(in_a, in_b, out, input_dtype, input_batch, input_m, input_k, input_n, lhs_multicore, rhs_multicore, batch_multicore, 
        lhs_transpose, rhs_transpose, alpha, beta, addmm_beta, sip_m, sip_k, sip_n, broadcasted_weight, buffer_sip, raw_cache);
}

extern "C" __global__ void matmul_i8(int8_t *in_a, int8_t *in_b,
                                             int8_t *out, 
                                            int input_dtype, int input_batch,
                                            int input_m, int input_k, int input_n,
                                            int lhs_multicore, int rhs_multicore, int batch_multicore,
                                            int lhs_transpose, int rhs_transpose,
                                            float alpha, float beta, float addmm_beta, 
                                            int sip_m, int sip_k, int sip_n, int broadcasted_weight) {
    __local__ __valigned__ char buffer_sip[VDMEM_SIZE];
    __shared__ char raw_cache[SHARE_BUFFER_SIZE];
    matmul_kernel<int8_t, int8_t>(in_a, in_b, out, input_dtype, input_batch, input_m, input_k, input_n, lhs_multicore, rhs_multicore, batch_multicore, 
        lhs_transpose, rhs_transpose, alpha, beta, addmm_beta, sip_m, sip_k, sip_n, broadcasted_weight, buffer_sip, raw_cache);
}

int main() {

}