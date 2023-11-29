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
#include <tops/half.h>
#include <tops/bfloat.h>
#include "gemm_general.h"
#include "tops/tops_runtime.h"
// #include "op_aten_gemm_tuner.h"
#include "utils.h"
#include "dot_core_kernels.h"
#include "include/common/atomic_op.h"
using namespace std;
#define MAX_NUM 1571840
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

template <typename Type1, typename Type2>
__device__ void gemm_kernel(Type1 *lhs, Type2 *rhs, Type1 *out, Type1 *bias,
      int input_dtype, int input_batch,
      int input_m, int input_k, int input_n,
      int lhs_multicore, int rhs_multicore, int batch_multicore,
      int lhs_transpose, int rhs_transpose,
      float alpha, float beta, float addmm_beta, int enable_bais,
      int sip_m, int sip_k, int sip_n) {
  int thread_num = GetThreadNum();
  int thread_id = GetThreadIdx();
  // CCPRINTF("thread_num =%d, thread_id=%d\n", thread_num, thread_id);

  int sip_cnt_raw = thread_num;
  int sip_idx = thread_id;
  // 声明dte
  tops_dte_ctx_t ctx_lhs_0;
  tops_dte_ctx_t ctx_lhs_1;
  tops_dte_ctx_t ctx_rhs_0;
  tops_dte_ctx_t ctx_rhs_1;
  tops_dte_ctx_t ctx_lhs_trans_0;
  tops_dte_ctx_t ctx_lhs_trans_1;
  tops_dte_ctx_t ctx_rhs_trans_0;
  tops_dte_ctx_t ctx_rhs_trans_1;
  tops_dte_ctx_t ctx_out;
  tops_dte_ctx_t ctx_bias;
  tops_dte_ctx_t ctx_b;

  tops::dte_scope s_lhs_0(ctx_lhs_0);
  tops::dte_scope s_lhs_1(ctx_lhs_1);
  tops::dte_scope s_rhs_0(ctx_rhs_0);
  tops::dte_scope s_rhs_1(ctx_rhs_1);
  tops::dte_scope s_lhs_trans_0(ctx_lhs_trans_0);
  tops::dte_scope s_lhs_trans_1(ctx_lhs_trans_1);
  tops::dte_scope s_rhs_trans_0(ctx_rhs_trans_0);
  tops::dte_scope s_rhs_trans_1(ctx_rhs_trans_1);
  tops::dte_scope s_out(ctx_out);
  tops::dte_scope s_bias(ctx_bias);
  tops::dte_scope s_b(ctx_b);

  tops::event e_lhs_0;
  tops::event e_lhs_1;
  tops::event e_rhs_0;
  tops::event e_rhs_1;
  tops::event e_lhs_trans_0;
  tops::event e_lhs_trans_1;
  tops::event e_rhs_trans_0;
  tops::event e_rhs_trans_1;
  tops::event e_bais;
  tops::event e_out;
  tops::event e_b;

  auto data_type = input_dtype;
  auto weight_data_type = input_dtype;
  auto B = input_batch;
  auto M = input_m;
  auto K = input_k;
  auto N = input_n;
  auto transa = lhs_transpose;
  auto transb = rhs_transpose;
  int enable_act = 0;
  auto M_SIP = sip_m;
  auto N_SIP = sip_n;
  auto K_SIP = sip_k;
  // CCPRINTF("gemm para %lu,%d\n", sizeof(para), enable_act);
  // CCPRINTF("gemm param  is %d,%d,%d,%d,%d,%d,%d,%d,%f,%f,%f,%d\n", B,M, N,K,
  // lhs_multicore, rhs_multicore,
  // transa,transb, alpha,beta, addmm_beta,enable_bais);
  int32_t hbm_lhs_shape[4] = {1, B, M, K};
  int32_t hbm_rhs_shape[4] = {1, B, K, N};
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
  int32_t sip_lhs_shape[4] = {1, 1, M_SIP, K_SIP};
  int32_t sip_rhs_shape[4] = {1, 1, K_SIP, N_SIP};
  int32_t sip_out_shape[4] = {1, 1, M_SIP, N_SIP};
  int32_t sip_lhs_trans_shape[4] = {1, 1, K_SIP, M_SIP};
  int32_t sip_rhs_trans_shape[4] = {1, 1, N_SIP, K_SIP};

  int32_t sip_bias_shape[1] = {N};
  int32_t sip_lhs_size = sip_lhs_shape[0] * sip_lhs_shape[1] *
                         sip_lhs_shape[2] * sip_lhs_shape[3] * sizeof(Type1);
  int32_t sip_rhs_size = sip_rhs_shape[0] * sip_rhs_shape[1] *
                         sip_rhs_shape[2] * sip_rhs_shape[3] * sizeof(Type1);
  int32_t sip_out_size = sip_out_shape[0] * sip_out_shape[1] *
                         sip_out_shape[2] * sip_out_shape[3] * sizeof(Type1);
  int32_t sip_lhs_trans_size = sip_lhs_trans_shape[0] * sip_lhs_trans_shape[1] *
                               sip_lhs_trans_shape[2] * sip_lhs_trans_shape[3] *
                               sizeof(Type1);
  int32_t sip_rhs_trans_size = sip_rhs_trans_shape[0] * sip_rhs_trans_shape[1] *
                               sip_rhs_trans_shape[2] * sip_rhs_trans_shape[3] *
                               sizeof(Type1);

  tops::mdspan hbm_lhs(tops::Global, lhs, hbm_lhs_shape);
  tops::mdspan hbm_rhs(tops::Global, rhs, hbm_rhs_shape);
  tops::mdspan hbm_out(tops::Global, out, hbm_out_shape);
  // tops::mdspan hbm_pre_gelu(tops::Global, pre_gelu, hbm_out_shape);
  tops::mdspan hbm_bias(tops::Global, bias, N);
  __local__ __valigned__ char buffer_sip[MAX_NUM];

  // float *buffer_sip_lhs0 = buffer_sip;
  Type1 *buffer_sip_lhs0_trans =
      reinterpret_cast<Type1 *>(reinterpret_cast<char *>(buffer_sip));
  Type1 *buffer_sip_lhs1_trans = reinterpret_cast<Type1 *>(
      (reinterpret_cast<char *>(buffer_sip_lhs0_trans)) +
      TEMPLATE_ALIGN_UP(sip_lhs_trans_size, L1_ALIGN_SIZE));
  Type2 *buffer_sip_rhs0_trans = reinterpret_cast<Type2 *>(
      (reinterpret_cast<char *>(buffer_sip_lhs1_trans)) +
      TEMPLATE_ALIGN_UP(sip_lhs_trans_size, L1_ALIGN_SIZE));
  Type2 *buffer_sip_rhs1_trans = reinterpret_cast<Type2 *>(
      (reinterpret_cast<char *>(buffer_sip_rhs0_trans)) +
      TEMPLATE_ALIGN_UP(sip_rhs_trans_size, L1_ALIGN_SIZE));
  Type1 *buffer_sip_lhs0 = reinterpret_cast<Type1 *>(
      (reinterpret_cast<char *>(buffer_sip_rhs1_trans)) +
      TEMPLATE_ALIGN_UP(sip_rhs_trans_size, L1_ALIGN_SIZE));
  Type1 *buffer_sip_lhs1 =
      reinterpret_cast<Type1 *>((reinterpret_cast<char *>(buffer_sip_lhs0)) +
                                TEMPLATE_ALIGN_UP(sip_lhs_size, L1_ALIGN_SIZE));
  Type2 *buffer_sip_rhs0 =
      reinterpret_cast<Type2 *>((reinterpret_cast<char *>(buffer_sip_lhs1)) +
                                TEMPLATE_ALIGN_UP(sip_lhs_size, L1_ALIGN_SIZE));
  Type2 *buffer_sip_rhs1 =
      reinterpret_cast<Type2 *>((reinterpret_cast<char *>(buffer_sip_rhs0)) +
                                TEMPLATE_ALIGN_UP(sip_rhs_size, L1_ALIGN_SIZE));
  Type1 *buffer_sip_out0 =
      reinterpret_cast<Type1 *>((reinterpret_cast<char *>(buffer_sip_rhs1)) +
                                TEMPLATE_ALIGN_UP(sip_rhs_size, L1_ALIGN_SIZE));
  Type1 *buffer_sip_out1 =
      reinterpret_cast<Type1 *>((reinterpret_cast<char *>(buffer_sip_out0)) +
                                TEMPLATE_ALIGN_UP(sip_out_size, L1_ALIGN_SIZE));
  Type1 *buffer_sip_bias0 =
      reinterpret_cast<Type1 *>((reinterpret_cast<char *>(buffer_sip_out1)) +
                                TEMPLATE_ALIGN_UP(sip_out_size, L1_ALIGN_SIZE));

  long lhs_addr = (long)buffer_sip_lhs0;
  long rhs_addr = (long)buffer_sip_rhs0;
  long out_addr = (long)buffer_sip_out0;
  long bias_addr = (long)buffer_sip_bias0;
  int32_t l_sdma_wait = 0;
  int32_t r_sdma_wait = 0;
  int32_t b_sdma_wait = 0;
  int32_t o_sdma_wait = 0;

  int weight_offset = 0;
  if ((data_type == TOPSOP_DATA_FP16) && (weight_data_type == TOPSOP_DATA_I8)) {
    weight_offset = K_SIP * N_SIP;
  }
  tops::mdspan sip_lhs0(tops::Private, buffer_sip_lhs0, sip_lhs_shape);
  tops::mdspan sip_lhs1(tops::Private, buffer_sip_lhs1, sip_lhs_shape);
  tops::mdspan sip_rhs0(tops::Private, buffer_sip_rhs0 + weight_offset,
                        sip_rhs_shape);
  tops::mdspan sip_rhs1(tops::Private, buffer_sip_rhs1 + weight_offset,
                        sip_rhs_shape);
  tops::mdspan sip_lhs0_trans(tops::Private, buffer_sip_lhs0_trans,
                              sip_lhs_trans_shape);
  tops::mdspan sip_lhs1_trans(tops::Private, buffer_sip_lhs1_trans,
                              sip_lhs_trans_shape);
  tops::mdspan sip_rhs0_trans(tops::Private,
                              buffer_sip_rhs0_trans + weight_offset,
                              sip_rhs_trans_shape);
  tops::mdspan sip_rhs1_trans(tops::Private,
                              buffer_sip_rhs1_trans + weight_offset,
                              sip_rhs_trans_shape);
  tops::mdspan sip_out0(tops::Private, buffer_sip_out0, sip_out_shape);
  tops::mdspan sip_out1(tops::Private, buffer_sip_out1, sip_out_shape);
  tops::mdspan sip_bias0(tops::Private, buffer_sip_bias0, N);
  auto M_SIP_LOOP_CNT_TASKS = M / M_SIP + (M % M_SIP > 0 ? 1 : 0);
  auto N_SIP_LOOP_CNT_TASKS = N / N_SIP + (N % N_SIP > 0 ? 1 : 0);
  auto K_SIP_LOOP_CNT = K / K_SIP + (K % K_SIP > 0 ? 1 : 0);
  auto xhs_multicore = 0;
  if ((lhs_multicore == 1) || (rhs_multicore == 1) || (batch_multicore == 1)) {
    xhs_multicore = 1;
  }
  auto sip_cnt = (xhs_multicore == 0) ? 1 : sip_cnt_raw;
  auto sdma_tasks_num =
      (lhs_multicore == 1) ? M_SIP_LOOP_CNT_TASKS : N_SIP_LOOP_CNT_TASKS;
  if (batch_multicore == 1) {
    sdma_tasks_num = B;
  }
  auto sip_num_used = (sdma_tasks_num > sip_cnt) ? sip_cnt : sdma_tasks_num;
  auto sip_num_lhs = (lhs_multicore == 1) ? sip_num_used : 1;
  auto sip_num_rhs = (rhs_multicore == 1) ? sip_num_used : 1;
  auto reminder = sdma_tasks_num % sip_cnt;
  auto loop_len_this_sip = (sip_idx < reminder) ? (sdma_tasks_num / sip_cnt + 1)
                                                : (sdma_tasks_num / sip_cnt);
  if (loop_len_this_sip == 0) {
    return;
  }
  int batch_hmb_offset = 0;
  if (batch_multicore == 1) {
    batch_hmb_offset = (sip_idx < reminder)
                           ? (sip_idx * loop_len_this_sip)
                           : (sip_idx * loop_len_this_sip + reminder);
  }
  auto M_SIP_LOOP_CNT =
      lhs_multicore == 1 ? loop_len_this_sip : M_SIP_LOOP_CNT_TASKS;
  auto N_SIP_LOOP_CNT =
      rhs_multicore == 1 ? loop_len_this_sip : N_SIP_LOOP_CNT_TASKS;
  auto Batch_SIP_LOOP_CNT = (batch_multicore == 1) ? loop_len_this_sip : B;

  if (sip_idx < sip_num_used) {
    // CCPRINTF("sip parallel: batch_split_flag: %d, lhs_split_flag: %d,
    // rhs_split_flag: %d, sip_num_used: %d\n",
    //         batch_multicore, lhs_multicore, rhs_multicore, sip_num_used);
    // CCPRINTF("param is M: %d, N: %d, K: %d, M_SIP: %d, N_SIP: %d, K_SIP: %d\n
    // Batch_SIP_LOOP_CNT: %d, M_SIP_LOOP_CNT: %d, N_SIP_LOOP_CNT: %d,
    // K_SIP_LOOP_CNT: %d\n",
    //       M, N, K, M_SIP, N_SIP, K_SIP, Batch_SIP_LOOP_CNT, M_SIP_LOOP_CNT,
    //       N_SIP_LOOP_CNT, K_SIP_LOOP_CNT);
    if (enable_bais) {
      // tops::slice(ctx_bias, sip_bias0, hbm_bias,{0});
      tops::memcpy(ctx_bias, sip_bias0, hbm_bias);
    }
    for (auto b_idx = 0; b_idx < Batch_SIP_LOOP_CNT; b_idx++) {
      auto m_hbm_offset = (lhs_multicore == 1) ? sip_idx * M_SIP : 0;
      auto n_hbm_offset = (rhs_multicore == 1) ? sip_idx * N_SIP : 0;
      auto batch_offset = batch_hmb_offset + b_idx;
      if (transa == 0) {
        e_lhs_0 = tops::slice_async(ctx_lhs_0, sip_lhs0, hbm_lhs,
                                    {0, batch_offset, m_hbm_offset, 0});
      } else {
        e_lhs_trans_0 =
            tops::slice_async(ctx_lhs_trans_0, sip_lhs0_trans, hbm_lhs,
                              {0, batch_offset, 0, m_hbm_offset});
        tops::wait(e_lhs_trans_0);
        e_lhs_0 = tops::transpose_async(ctx_lhs_0, sip_lhs0, sip_lhs0_trans,
                                        {0, 1, 3, 2});
      }

      if (transb == 0) {
        e_rhs_0 = tops::slice_async(ctx_rhs_0, sip_rhs0, hbm_rhs,
                                    {0, batch_offset, 0, n_hbm_offset});
      } else {
        e_rhs_trans_0 =
            tops::slice_async(ctx_rhs_trans_0, sip_rhs0_trans, hbm_rhs,
                              {0, batch_offset, n_hbm_offset, 0});
        tops::wait(e_rhs_trans_0);

        e_rhs_0 = tops::transpose_async(ctx_rhs_0, sip_rhs0, sip_rhs0_trans,
                                        {0, 1, 3, 2});
      }
      bool flag_no_beta = true;
      int *tmp_beta = reinterpret_cast<int *>(&beta);
      flag_no_beta = ((*tmp_beta) == 0);
      if (flag_no_beta) {
        e_b = tops::memset_async(ctx_b, sip_out0, (Type1)0.0);
      } else {
        e_b = tops::slice_async(ctx_b, sip_out0, hbm_out,
                                {0, batch_offset, m_hbm_offset, n_hbm_offset});
      }

      l_sdma_wait = 1;
      r_sdma_wait = 1;
      b_sdma_wait = 1;
      o_sdma_wait = 0;
      // tops::wait(e_lhs_0);
      int next_n_sip_idx_temp, next_m_sip_idx, next_m_sip_idx_temp;

      for (auto m_sip_idx = 0; m_sip_idx < M_SIP_LOOP_CNT; m_sip_idx++) {
        // auto m_sip_offset = m_sip_idx * M_SIP;
        auto m_sip_offset = lhs_multicore
                                ? ((m_sip_idx * sip_num_lhs + sip_idx) * M_SIP)
                                : (m_sip_idx * M_SIP);
        for (auto n_sip_idx = 0; n_sip_idx < N_SIP_LOOP_CNT; n_sip_idx++) {
          // auto n_sip_offset = n_sip_idx * N_SIP;
          auto n_sip_offset =
              rhs_multicore ? ((n_sip_idx * sip_num_rhs + sip_idx) * N_SIP)
                            : (n_sip_idx * N_SIP);
          auto osdma_task_mn_idx = m_sip_idx * N_SIP_LOOP_CNT + n_sip_idx;
          auto cur_out_sip =
              ((osdma_task_mn_idx % 2) == 0) ? &sip_out0 : &sip_out1;
          out_addr = (long)((osdma_task_mn_idx % 2) == 0 ? buffer_sip_out0
                                                         : buffer_sip_out1);
          auto pre_gelu_buffer =
              ((osdma_task_mn_idx % 2) == 0 ? buffer_sip_out0
                                            : buffer_sip_out1);
          for (auto k_sip_idx = 0; k_sip_idx < K_SIP_LOOP_CNT; k_sip_idx++) {
            auto k_sip_offset = k_sip_idx * K_SIP;
            auto sdma_task_mnc_idx =
                m_sip_idx * N_SIP_LOOP_CNT * K_SIP_LOOP_CNT +
                n_sip_idx * K_SIP_LOOP_CNT + k_sip_idx;

            auto next_lhs_sip = (sdma_task_mnc_idx % 2) ? &sip_lhs0 : &sip_lhs1;
            auto next_rhs_sip = (sdma_task_mnc_idx % 2) ? &sip_rhs0 : &sip_rhs1;
            auto next_out_sip = (osdma_task_mn_idx % 2) ? &sip_out0 : &sip_out1;
            auto next_lhs_sip_trans =
                (sdma_task_mnc_idx % 2) ? &sip_lhs0_trans : &sip_lhs1_trans;
            auto next_rhs_sip_trans =
                (sdma_task_mnc_idx % 2) ? &sip_rhs0_trans : &sip_rhs1_trans;

            auto next_k_sip_idx =
                (k_sip_idx + 1 == K_SIP_LOOP_CNT) ? 0 : k_sip_idx + 1;
            if ((k_sip_idx + 1 == K_SIP_LOOP_CNT) &&
                (n_sip_idx + 1 == N_SIP_LOOP_CNT)) {
              next_n_sip_idx_temp = 0;
            } else {
              next_n_sip_idx_temp = n_sip_idx + 1;
            }
            auto next_n_sip_idx = (k_sip_idx + 1 < K_SIP_LOOP_CNT)
                                      ? n_sip_idx
                                      : next_n_sip_idx_temp;
            if ((k_sip_idx + 1 == K_SIP_LOOP_CNT) &&
                (n_sip_idx + 1 == N_SIP_LOOP_CNT) &&
                (m_sip_idx + 1 == M_SIP_LOOP_CNT)) {
              next_m_sip_idx_temp = M_SIP_LOOP_CNT;
            } else {
              next_m_sip_idx_temp = m_sip_idx + 1;
            }
            if ((k_sip_idx + 1 < K_SIP_LOOP_CNT) ||
                (n_sip_idx + 1 < N_SIP_LOOP_CNT)) {
              next_m_sip_idx = m_sip_idx;
            } else {
              next_m_sip_idx = next_m_sip_idx_temp;
            }

            auto next_k_sip_offset = next_k_sip_idx * K_SIP;
            // auto next_n_sip_offset = next_n_sip_idx * N_SIP;
            auto next_n_sip_offset =
                rhs_multicore
                    ? ((next_n_sip_idx * sip_num_rhs + sip_idx) * N_SIP)
                    : (next_n_sip_idx * N_SIP);
            // auto next_m_sip_offset = next_m_sip_idx * M_SIP;
            auto next_m_sip_offset =
                lhs_multicore
                    ? ((next_m_sip_idx * sip_num_lhs + sip_idx) * M_SIP)
                    : (next_m_sip_idx * M_SIP);

            if (l_sdma_wait == 1) {
              tops::wait(e_lhs_0);
            }
            if (r_sdma_wait == 1) {
              tops::wait(e_rhs_0);
            }
            if (b_sdma_wait == 1) {
              tops::wait(e_b);
            }
            l_sdma_wait = 0;
            r_sdma_wait = 0;
            b_sdma_wait = 0;

            if ((next_k_sip_offset < K) && (next_n_sip_offset < N) &&
                (next_m_sip_offset < M)) {
              if (transa == 0) {
                e_lhs_0 = tops::slice_async(
                    ctx_lhs_0, *next_lhs_sip, hbm_lhs,
                    {0, batch_offset, next_m_sip_offset, next_k_sip_offset});
              } else {
                e_lhs_trans_0 = tops::slice_async(
                    ctx_lhs_trans_0, *next_lhs_sip_trans, hbm_lhs,
                    {0, batch_offset, next_k_sip_offset, next_m_sip_offset});
                tops::wait(e_lhs_trans_0);
                e_lhs_0 =
                    tops::transpose_async(ctx_lhs_0, *next_lhs_sip,
                                          *next_lhs_sip_trans, {0, 1, 3, 2});
              }
              l_sdma_wait = 1;
              if (transb == 0) {
                e_rhs_0 = tops::slice_async(
                    ctx_rhs_0, *next_rhs_sip, hbm_rhs,
                    {0, batch_offset, next_k_sip_offset, next_n_sip_offset});
              } else {
                e_rhs_trans_0 = tops::slice_async(
                    ctx_rhs_trans_0, *next_rhs_sip_trans, hbm_rhs,
                    {0, batch_offset, next_n_sip_offset, next_k_sip_offset});
                tops::wait(e_rhs_trans_0);
                e_rhs_0 =
                    tops::transpose_async(ctx_rhs_0, *next_rhs_sip,
                                          *next_rhs_sip_trans, {0, 1, 3, 2});
              }
              r_sdma_wait = 1;
            }
            if ((next_n_sip_offset < N) && (next_m_sip_offset < M) &&
                (next_k_sip_offset == 0)) {
              if (flag_no_beta) {
                e_b = tops::memset_async(ctx_b, *next_out_sip, (Type1)0.0);
              } else {
                e_b = tops::slice_async(
                    ctx_b, *next_out_sip, hbm_out,
                    {0, batch_offset, next_m_sip_offset, next_n_sip_offset});
              }
              // CCPRINTF("beta: %f, beta == 0: %d\n", beta, beta == 0);
              // e_b =  tops::memset_async(ctx_b, *next_out_sip, (Type1)0.0);

              b_sdma_wait = 1;
            }

            auto nacc_flag = k_sip_idx == 0 ? 1 : 0;
            auto store_flag = k_sip_idx + 1 == K_SIP_LOOP_CNT ? 1 : 0;
            lhs_addr = long((sdma_task_mnc_idx % 2) == 0 ? buffer_sip_lhs0
                                                         : buffer_sip_lhs1);
            auto next_buffer_sip_rhs = (sdma_task_mnc_idx % 2) == 0
                                           ? buffer_sip_rhs0
                                           : buffer_sip_rhs1;
            rhs_addr = long(next_buffer_sip_rhs);
            // CCPRINTF("input lhs,rhs,out is %f,%f,%f\n",
            // buffer_sip_lhs0[0],buffer_sip_rhs0[0],buffer_sip_out0[0]);
            if (data_type == TOPSOP_DATA_FP32) {
              c_func_sgemm_general(lhs_addr, rhs_addr, out_addr, M_SIP, N_SIP,
                                   K_SIP, nacc_flag, store_flag, 0, 0, alpha,
                                   beta, addmm_beta, enable_bais, bias_addr,
                                   n_sip_offset);
            } else if (data_type == TOPSOP_DATA_FP16) {
              if (weight_data_type == TOPSOP_DATA_I8) {
                convert(reinterpret_cast<__fp16 *>(next_buffer_sip_rhs),
                        reinterpret_cast<char *>(next_buffer_sip_rhs +
                                                 K_SIP * N_SIP),
                        K_SIP * N_SIP);
                c_func_hgemm_general(lhs_addr, rhs_addr, out_addr, M_SIP, N_SIP,
                                     K_SIP, nacc_flag, store_flag, 0, 0, alpha,
                                     beta, addmm_beta, enable_bais, bias_addr,
                                     n_sip_offset);
              } else {
                c_func_hgemm_general(lhs_addr, rhs_addr, out_addr, M_SIP, N_SIP,
                                     K_SIP, nacc_flag, store_flag, 0, 0, alpha,
                                     beta, addmm_beta, enable_bais, bias_addr,
                                     n_sip_offset);
              }
            } else if (data_type == TOPSOP_DATA_I8) {
              c_func_gemm_general_int8(lhs_addr, rhs_addr, out_addr, M_SIP,
                                       N_SIP, K_SIP, nacc_flag, store_flag, 0,
                                       0, alpha, beta, enable_bais, bias_addr,
                                       n_sip_offset);
            } else if (data_type == TOPSOP_DATA_BF16) {
              c_func_bfgemm_general(lhs_addr, rhs_addr, out_addr, M_SIP, N_SIP,
                                    K_SIP, nacc_flag, store_flag, 0, 0, alpha,
                                    beta, addmm_beta, enable_bais, bias_addr,
                                    n_sip_offset);
            }
          }  // K loop
          // if (enable_act == 1) {
          //   // call act + store
          //   e_out = tops::deslice_async(
          //       ctx_out, hbm_pre_gelu, *cur_out_sip,
          //       {0, batch_offset, m_sip_offset, n_sip_offset});
          //   tops::wait(e_out);
          //   gelu_wrapper(pre_gelu_buffer, pre_gelu_buffer, M_SIP * N_SIP);
          // }
          e_out = tops::deslice_async(
              ctx_out, hbm_out, *cur_out_sip,
              {0, batch_offset, m_sip_offset, n_sip_offset});
          tops::wait(e_out);
        }  // N loop
      }    // M loop
    }      // batch loop

  }
}

extern "C" __global__ void gemm_f32(float *in_a, float *in_b,
                                            float *out, float *bias,
                                            int input_dtype, int input_batch,
                                            int input_m, int input_k, int input_n,
                                            int lhs_multicore, int rhs_multicore, int batch_multicore,
                                            int lhs_transpose, int rhs_transpose,
                                            float alpha, float beta, float addmm_beta, int enable_bais,
                                            int sip_m, int sip_k, int sip_n) {
      gemm_kernel<float, float>(in_a, in_b, out, bias, input_dtype, input_batch, input_m, input_k, input_n, lhs_multicore, rhs_multicore, batch_multicore, 
      lhs_transpose, rhs_transpose, alpha, beta, addmm_beta, enable_bais, sip_m, sip_k, sip_n);
}

extern "C" __global__ void gemm_f16(__fp16 *in_a, __fp16 *in_b, __fp16 *out,  __fp16 *bias,
                                                int input_dtype, int input_batch,
                                                int input_m, int input_k, int input_n,
                                                int lhs_multicore, int rhs_multicore, int batch_multicore,
                                                int lhs_transpose, int rhs_transpose,
                                                float alpha, float beta, float addmm_beta, int enable_bais,
                                                int sip_m, int sip_k, int sip_n) {
      gemm_kernel<__fp16, __fp16>(in_a, in_b, out, bias, input_dtype, input_batch, input_m, input_k, input_n, lhs_multicore, rhs_multicore, batch_multicore, 
      lhs_transpose, rhs_transpose, alpha, beta, addmm_beta, enable_bais, sip_m, sip_k, sip_n);
}

extern "C" __global__ void gemm_bf16(tops::bfloat *in_a,
                                                   tops::bfloat *in_b,
                                                   tops::bfloat *out,
                                                   tops::bfloat *bias,
                                                  int input_dtype, int input_batch,
                                                  int input_m, int input_k, int input_n,
                                                  int lhs_multicore, int rhs_multicore, int batch_multicore,
                                                  int lhs_transpose, int rhs_transpose,
                                                  float alpha, float beta, float addmm_beta, int enable_bais,
                                                  int sip_m, int sip_k, int sip_n) {
      gemm_kernel<tops::bfloat, tops::bfloat>(in_a, in_b, out, bias, input_dtype, input_batch, input_m, input_k, input_n, lhs_multicore, rhs_multicore, batch_multicore, 
      lhs_transpose, rhs_transpose, alpha, beta, addmm_beta, enable_bais, sip_m, sip_k, sip_n);
}

extern "C" __global__ void gemm_i8(int8_t *in_a, int8_t *in_b,
                                             int8_t *out, int8_t *bias,
                                            int input_dtype, int input_batch,
                                            int input_m, int input_k, int input_n,
                                            int lhs_multicore, int rhs_multicore, int batch_multicore,
                                            int lhs_transpose, int rhs_transpose,
                                            float alpha, float beta, float addmm_beta, int enable_bais,
                                            int sip_m, int sip_k, int sip_n) {
      gemm_kernel<int8_t, int8_t>(in_a, in_b, out, bias, input_dtype, input_batch, input_m, input_k, input_n, lhs_multicore, rhs_multicore, batch_multicore, 
      lhs_transpose, rhs_transpose, alpha, beta, addmm_beta, enable_bais, sip_m, sip_k, sip_n);
}

int main(void) {
  // topsError_t err = topsSuccess;
  // // Print the vector length to be used, and compute its size
  // int ITERATION = 20;
  // int b = 13;
  // int m = 13;
  // int n = 4096;
  // int k = 4096;
  // bool check = false;
  // DATA<__fp16> data(b, m, k, n, 8, check);

  // AtenGemmInfo info;
  // info.data_type = TOPSOP_DATA_FP16;
  // info.out_data_type = TOPSOP_DATA_FP16;
  // info.is_batch = true;
  // info.batch = b;
  // info.M = m;
  // info.K = k;
  // info.N = n;
  // info.transa = false;
  // info.transb = false;
  // AtenGemmTune tune;
  // AtenGemmTuner tuner;
  // tuner.Tuner(info, &tune);
  // GEMM_OP_PARAS gemm_op_para_;
  // printf("Tuning results: csb_batch (%d), sip_batch (%d), lhs_csb_k (%d), rhs_csb_k (%d), \
  //         lhs_csb_m (%d), rhs_csb_n (%d), sip_m (%d), sip_k (%d), sip_n (%d), batch_multicore (%d), \
  //         lhs_multicore (%d), rhs_multicore (%d), cdma_lhs_pingpong (%d), sdma_rhs_pingpong (%d)\n",
  //         tune.csb_batch, tune.sip_batch, tune.lhs_csb_k, tune.rhs_csb_k, tune.lhs_csb_m, tune.rhs_csb_n, 
  //         tune.sip_m, tune.sip_k, tune.sip_n, tune.batch_multicore, tune.lhs_multicore, tune.rhs_multicore,
  //          tune.cdma_lhs_pingpong, tune.cdma_rhs_pingpong, tune.sdma_lhs_pingpong, tune.sdma_rhs_pingpong);
  //   // gemm_op_para_.input_dtype = info.data_type;  // 0
  //   // gemm_op_para_.output_dtype = info.out_data_type;
  //   // gemm_op_para_.csb_batch = tune.csb_batch;
  //   // gemm_op_para_.sip_batch = tune.sip_batch;
  //   // gemm_op_para_.lhs_csb_k = tune.lhs_csb_k;
  //   // gemm_op_para_.rhs_csb_k = tune.rhs_csb_k;  // 5
  //   // gemm_op_para_.lhs_csb_m = tune.lhs_csb_m;
  //   // gemm_op_para_.rhs_csb_n = tune.rhs_csb_n;
  //   // gemm_op_para_.sip_m = tune.sip_m;
  //   // gemm_op_para_.sip_k = tune.sip_k;
  //   // gemm_op_para_.sip_n = tune.sip_n;  // 10
  //   // gemm_op_para_.batch_multicore = tune.batch_multicore;
  //   // gemm_op_para_.lhs_multicore = tune.lhs_multicore;
  //   // gemm_op_para_.rhs_multicore = tune.rhs_multicore;
  //   // gemm_op_para_.lhs_pingpong = tune.cdma_lhs_pingpong;
  //   // gemm_op_para_.rhs_pingpong = tune.cdma_rhs_pingpong;
  //   // gemm_op_para_.sdma_lhs_pingpong = tune.sdma_lhs_pingpong;  // 15
  //   // gemm_op_para_.sdma_rhs_pingpong = tune.sdma_rhs_pingpong;
  //   // gemm_op_para_.rhs_repeat_copy = tune.rhs_repeatcopy;
  //   // gemm_op_para_.lhs_transpose = tune.lhs_tranpose;
  //   // gemm_op_para_.rhs_transpose = tune.rhs_tranpose;
  //   // gemm_op_para_.out_transpose = tune.out_tranpose;  // 20
  //   // gemm_op_para_.alpha = 1.0;
  //   // gemm_op_para_.beta = 0.0;
  //   // gemm_op_para_.coef = static_cast<float>(0);
  //   // gemm_op_para_.act_mode = TOPSOP_ACTIVATION_NONE;
  //   // gemm_op_para_.bias = 0;
  //   // gemm_op_para_.act_mode = 0;



  // __fp16 *h_bias =
  //     reinterpret_cast<__fp16 *>(aligned_alloc(4096, n * sizeof(__fp16)));
  // for (int i=0; i< n; i++) {
  //   h_bias[i] = 0.0;
  // }
  // __fp16 *d_Bias = NULL;
  // CHECK(topsMalloc(reinterpret_cast<void **>(&d_Bias), n * sizeof(__fp16)));
  // CHECK(
  //     topsMemcpy(d_Bias, h_bias, n * sizeof(__fp16), topsMemcpyHostToDevice));

  // printf("call kernel!!!!!!!!!!!!!\n");
  // float time = 0.0;
  // float total_time = 0.0;
  // topsEvent_t start, stop;
  // for (int i=0; i< ITERATION; i++) {
  //   CHECK(topsEventCreate(&start));
  //   CHECK(topsEventCreate(&stop));

  //   CHECK(topsEventRecord(start));

  //   gemm_f16<<<1, 12>>>(data.lhs_d, data.rhs_d, data.out_d, d_Bias, info.data_type, b, m, k, n, tune.lhs_multicore, tune.rhs_multicore,
  //     tune.batch_multicore, tune.lhs_tranpose, tune.rhs_tranpose, 1.0, 0.0, 0.0, 0, tune.sip_m, tune.sip_k, tune.sip_n);
  //   CHECK(topsEventRecord(stop));
  //   CHECK(topsEventSynchronize(stop));
  //   CHECK(topsEventElapsedTime(&time, start, stop));
  //   total_time += time;
  // }
  // printf("Time costs %.2f ms\n", total_time/ITERATION);

  // CHECK(topsGetLastError());
  // CHECK(topsMemcpy(data.out_h, data.out_d, data.size_out*sizeof(__fp16),
  //   topsMemcpyDeviceToHost));
  
  // //CPU/GPU check_data
  // if (check) { 
  //   printf("Compare with CPU data...\n");
  //   check_data<__fp16>(data.out_h, data.expected, b * m * n);
  // }

  // // for (int j=0; j< b * m * n; j++ ){
  // //   printf("%.2f ", data.out_h[j]);
  // // }
  return 0;
}