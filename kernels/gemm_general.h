/* Copyright 2020-2023 Enflame. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// #include <sip30intrin.h>
#include <stdio.h>
#define TAR_ADDR_WARP(addr, ss) (((addr)) | ((((addr) + (ss)) << 16)))
#define TAR_OFF_WARP(offset1, offset2) (((offset1) << 16) | ((offset2)&0xffff))


struct GEMM_OP_PARAS {
  int input_dtype; // 0
  int output_dtype;
  int csb_batch;
  int sip_batch;
  int lhs_csb_k;
  int rhs_csb_k; // 5
  int lhs_csb_m;
  int rhs_csb_n;
  int sip_m;
  int sip_k;
  int sip_n; // 10
  int batch_multicore;
  int lhs_multicore;
  int rhs_multicore;
  int lhs_pingpong;
  int rhs_pingpong;
  int sdma_lhs_pingpong; // 15
  int sdma_rhs_pingpong;
  int rhs_repeat_copy;
  int lhs_transpose;
  int rhs_transpose;
  int out_transpose; // 20
  float alpha;
  float beta;
  float addmm_beta;
  float coef;
  int act_mode;
  int bias; // 25
  int act_en;
  int input_batch;
  int input_m;
  int input_k;
  int input_n;
};

__attribute__((device)) extern "C" void c_func_hgemm_general(
    int a_addr, int b_addr, int c_addr, int M, int N, int K, int nacc_flag,
    int stroe_flag, int alpha_enable, int beta_enable, float alpha, float beta,
    float addmm_beta, int bias_en, int bias_addr, int cur_n) {
#if __GCU_ARCH__ >= 300

  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_lv_d(0);
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_s2(0);
  __dtu_c_movsr2vab_m_d(0);

  int BPE = 2;
  smr_t smr0, smr1;
  v64i8 vr0, vr1, vr2, vr3, vr4, vr5, vr6, vr7, vr8, vr9, vr10, vr11, vr12,
      vr13, vr14, vr15, vr16, vr17, vr18, vr19, vr20, vr21, vr22, vr23, vr24,
      vr25, vr26, vr27, vr28, vr29, vr30, vr31;
  v16f32 vr_alpha = __dtu_s_movr2vr_dup_f32(alpha);

  va16f32x4 qacc[64];
  va16f16x2 c_dacc[64];

  va16f32 va_alpha0, va_alpha1, va_alpha2, va_alpha3;
  va_alpha0 = __dtu_l_movvr2va(vr_alpha);
  va_alpha1 = __dtu_l_movvr2va(vr_alpha);
  va_alpha2 = __dtu_l_movvr2va(vr_alpha);
  va_alpha3 = __dtu_l_movvr2va(vr_alpha);
  va16f32x4 qa_alpha =
      __dtu_insertva2qa_f32(va_alpha0, va_alpha1, va_alpha2, va_alpha3);

  v16f32 vr_scale = __dtu_s_movr2vr_dup_f32(beta);
  va16f32 vacc_beta0, vacc_beta1, vacc_beta2, vacc_beta3;
  vacc_beta0 = __dtu_l_movvr2va(vr_scale);
  vacc_beta1 = __dtu_l_movvr2va(vr_scale);
  vacc_beta2 = __dtu_l_movvr2va(vr_scale);
  vacc_beta3 = __dtu_l_movvr2va(vr_scale);
  va16f32x4 qa_beta =
      __dtu_insertva2qa_f32(vacc_beta0, vacc_beta1, vacc_beta2, vacc_beta3);

  if (bias_en == 0) {
    vr_scale = __dtu_s_movr2vr_dup_f32(0.0f);
  } else {
    vr_scale = __dtu_s_movr2vr_dup_f32(addmm_beta);
  }
  vacc_beta0 = __dtu_l_movvr2va(vr_scale);
  vacc_beta1 = __dtu_l_movvr2va(vr_scale);
  vacc_beta2 = __dtu_l_movvr2va(vr_scale);
  vacc_beta3 = __dtu_l_movvr2va(vr_scale);
  va16f32x4 qa_bias =
      __dtu_insertva2qa_f32(vacc_beta0, vacc_beta1, vacc_beta2, vacc_beta3);
  va16f16x2 bs_dacc;

  auto k_unit = K >> 5;
  auto n_unit = N >> 6;
  auto on_unit = N >> 6;
  // vpt parallel in rhs
  int lt_addr = a_addr >> 6;
  int rt_addr = b_addr >> 7;
  int ot_addr = c_addr >> 7;
  int offset = 0;
  tar_t lt_base = __dtu_c_movsr2targ(TAR_ADDR_WARP(lt_addr, 0));
  offset = TAR_OFF_WARP(k_unit, k_unit);
  tar_t lt_off0 = __dtu_c_movsr2tari(offset, lt_base);
  offset = TAR_OFF_WARP(1 - 63 * k_unit, 1 - 63 * k_unit);  // next k
  tar_t lt_off1 = __dtu_c_movsr2tari(offset, lt_base);
  offset = TAR_OFF_WARP(1 - 64 * k_unit, 1 - 64 * k_unit);  //  new n
  tar_t lt_off2 = __dtu_c_movsr2tari(offset, lt_base);
  offset = TAR_OFF_WARP(1, 1);  // end k end n new m
  tar_t lt_off3 = __dtu_c_movsr2tari(offset, lt_base);

  tar_t rt_base = __dtu_c_movsr2targ((rt_addr) | ((rt_addr) + 1) << 16);
  offset = TAR_OFF_WARP(n_unit, n_unit);
  tar_t rt_off0 = __dtu_c_movsr2tari(offset, rt_base);
  offset = TAR_OFF_WARP(2 - (K - 1) * n_unit, 2 - (K - 1) * n_unit);
  tar_t rt_off1 = __dtu_c_movsr2tari(offset, rt_base);  // new n
  offset = TAR_OFF_WARP(2 - K * n_unit, 2 - K * n_unit);
  tar_t rt_off2 = __dtu_c_movsr2tari(offset, rt_base);  // new m

  auto bn_unit = N >> 6;
  int bt_addr = (c_addr >> 7) | ((c_addr >> 7) + 1) << 16;
  tar_t bt_base = __dtu_c_movsr2targ(bt_addr);
  offset = (bn_unit << 16) | bn_unit;
  tar_t bt_off0 = __dtu_c_movsr2tari(offset, bt_base);
  offset = (2 - 63 * bn_unit) & 0xffff;
  offset = (offset << 16) | offset;
  tar_t bt_off1 = __dtu_c_movsr2tari(offset, bt_base);
  offset = (2 << 16) | 2;
  tar_t bt_off2 = __dtu_c_movsr2tari(offset, bt_base);

  int biast_addr = ((bias_addr + cur_n * 2) >> 7) |
                   (((bias_addr + cur_n * 2) >> 7) + 1) << 16;
  tar_t biast_base = __dtu_c_movsr2targ(biast_addr);
  offset = (2 << 16) | 2;
  tar_t biast_off0 = __dtu_c_movsr2tari(offset, biast_base);
  offset = (2 - bn_unit) & 0xffff;
  offset = (offset << 16) | offset;
  tar_t biast_off1 = __dtu_c_movsr2tari(offset, bt_base);

  tar_t ot_base = __dtu_c_movsr2targ((ot_addr) | ((ot_addr) + 1) << 16);
  offset = TAR_OFF_WARP(on_unit, on_unit);
  tar_t ot_off0 = __dtu_c_movsr2tari(offset, ot_base);
  offset = TAR_OFF_WARP(2 - 63 * on_unit, 2 - 63 * on_unit);
  tar_t ot_off1 = __dtu_c_movsr2tari(offset, ot_base);  // new n
  offset = TAR_OFF_WARP(2, 2);
  tar_t ot_off2 = __dtu_c_movsr2tari(offset, ot_base);  // new m

  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 0);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 1);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 2);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 3);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 4);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 5);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 6);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 7);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 8);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 9);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 10);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 11);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 12);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 13);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 14);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 15);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 16);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 17);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 18);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 19);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 20);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 21);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 22);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 23);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 24);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 25);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 26);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 27);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 28);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 29);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 30);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 31);
  // m0k0
  vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr31 = __dtu_s_tvld_itar(lt_base, lt_off0);

  int naccovr = 0x10001;
  if (nacc_flag ^ 1) {
    naccovr = 0x1;
  }
  __dtu_c_movsr2naccovr(naccovr);
  __dtu_c_movsr2vab_m_s2(0);
  int vab_shift = 0;
// fp16 vmm2 mode17: [32, 64] * [64, 128] = [64, 128]
#pragma clang loop unroll(full)
  for (int m = 0; m < M; m += 64) {
    for (int n = 0; n < N - 128; n += 128) {  // VPT PARA DIM
      __dtu_c_movsr2naccovr(naccovr);
      for (int k = 0; k < K - 64; k += 64) {
        // smr1
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 0);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 1);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 2);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 3);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 4);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 5);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 6);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 7);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 8);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 9);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 10);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 11);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 12);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 13);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 14);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 15);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 16);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 17);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 18);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 19);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 20);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 21);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 22);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 23);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 24);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 25);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 26);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 27);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 28);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 29);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 30);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 31);
        // m0k0 * smr0
        qacc[0] = __dtu_m_vmm2_mode17_f16(qacc[0], vr0, smr0);
        qacc[1] = __dtu_m_vmm2_mode17_f16(qacc[1], vr1, smr0);
        qacc[2] = __dtu_m_vmm2_mode17_f16(qacc[2], vr2, smr0);
        qacc[3] = __dtu_m_vmm2_mode17_f16(qacc[3], vr3, smr0);
        qacc[4] = __dtu_m_vmm2_mode17_f16(qacc[4], vr4, smr0);
        qacc[5] = __dtu_m_vmm2_mode17_f16(qacc[5], vr5, smr0);
        qacc[6] = __dtu_m_vmm2_mode17_f16(qacc[6], vr6, smr0);
        qacc[7] = __dtu_m_vmm2_mode17_f16(qacc[7], vr7, smr0);
        qacc[8] = __dtu_m_vmm2_mode17_f16(qacc[8], vr8, smr0);
        qacc[9] = __dtu_m_vmm2_mode17_f16(qacc[9], vr9, smr0);
        qacc[10] = __dtu_m_vmm2_mode17_f16(qacc[10], vr10, smr0);
        qacc[11] = __dtu_m_vmm2_mode17_f16(qacc[11], vr11, smr0);
        qacc[12] = __dtu_m_vmm2_mode17_f16(qacc[12], vr12, smr0);
        qacc[13] = __dtu_m_vmm2_mode17_f16(qacc[13], vr13, smr0);
        qacc[14] = __dtu_m_vmm2_mode17_f16(qacc[14], vr14, smr0);
        qacc[15] = __dtu_m_vmm2_mode17_f16(qacc[15], vr15, smr0);
        qacc[16] = __dtu_m_vmm2_mode17_f16(qacc[16], vr16, smr0);
        qacc[17] = __dtu_m_vmm2_mode17_f16(qacc[17], vr17, smr0);
        qacc[18] = __dtu_m_vmm2_mode17_f16(qacc[18], vr18, smr0);
        qacc[19] = __dtu_m_vmm2_mode17_f16(qacc[19], vr19, smr0);
        qacc[20] = __dtu_m_vmm2_mode17_f16(qacc[20], vr20, smr0);
        qacc[21] = __dtu_m_vmm2_mode17_f16(qacc[21], vr21, smr0);
        qacc[22] = __dtu_m_vmm2_mode17_f16(qacc[22], vr22, smr0);
        qacc[23] = __dtu_m_vmm2_mode17_f16(qacc[23], vr23, smr0);
        qacc[24] = __dtu_m_vmm2_mode17_f16(qacc[24], vr24, smr0);
        qacc[25] = __dtu_m_vmm2_mode17_f16(qacc[25], vr25, smr0);
        qacc[26] = __dtu_m_vmm2_mode17_f16(qacc[26], vr26, smr0);
        qacc[27] = __dtu_m_vmm2_mode17_f16(qacc[27], vr27, smr0);
        qacc[28] = __dtu_m_vmm2_mode17_f16(qacc[28], vr28, smr0);
        qacc[29] = __dtu_m_vmm2_mode17_f16(qacc[29], vr29, smr0);
        qacc[30] = __dtu_m_vmm2_mode17_f16(qacc[30], vr30, smr0);
        qacc[31] = __dtu_m_vmm2_mode17_f16(qacc[31], vr31, smr0);
        // m1k0
        vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr31 = __dtu_s_tvld_itar(lt_base, lt_off1);
        // m1k0 * smr0
        qacc[32] = __dtu_m_vmm2_mode17_f16(qacc[32], vr0, smr0);
        qacc[33] = __dtu_m_vmm2_mode17_f16(qacc[33], vr1, smr0);
        qacc[34] = __dtu_m_vmm2_mode17_f16(qacc[34], vr2, smr0);
        qacc[35] = __dtu_m_vmm2_mode17_f16(qacc[35], vr3, smr0);
        qacc[36] = __dtu_m_vmm2_mode17_f16(qacc[36], vr4, smr0);
        qacc[37] = __dtu_m_vmm2_mode17_f16(qacc[37], vr5, smr0);
        qacc[38] = __dtu_m_vmm2_mode17_f16(qacc[38], vr6, smr0);
        qacc[39] = __dtu_m_vmm2_mode17_f16(qacc[39], vr7, smr0);
        qacc[40] = __dtu_m_vmm2_mode17_f16(qacc[40], vr8, smr0);
        qacc[41] = __dtu_m_vmm2_mode17_f16(qacc[41], vr9, smr0);
        qacc[42] = __dtu_m_vmm2_mode17_f16(qacc[42], vr10, smr0);
        qacc[43] = __dtu_m_vmm2_mode17_f16(qacc[43], vr11, smr0);
        qacc[44] = __dtu_m_vmm2_mode17_f16(qacc[44], vr12, smr0);
        qacc[45] = __dtu_m_vmm2_mode17_f16(qacc[45], vr13, smr0);
        qacc[46] = __dtu_m_vmm2_mode17_f16(qacc[46], vr14, smr0);
        qacc[47] = __dtu_m_vmm2_mode17_f16(qacc[47], vr15, smr0);
        qacc[48] = __dtu_m_vmm2_mode17_f16(qacc[48], vr16, smr0);
        qacc[49] = __dtu_m_vmm2_mode17_f16(qacc[49], vr17, smr0);
        qacc[50] = __dtu_m_vmm2_mode17_f16(qacc[50], vr18, smr0);
        qacc[51] = __dtu_m_vmm2_mode17_f16(qacc[51], vr19, smr0);
        qacc[52] = __dtu_m_vmm2_mode17_f16(qacc[52], vr20, smr0);
        qacc[53] = __dtu_m_vmm2_mode17_f16(qacc[53], vr21, smr0);
        qacc[54] = __dtu_m_vmm2_mode17_f16(qacc[54], vr22, smr0);
        qacc[55] = __dtu_m_vmm2_mode17_f16(qacc[55], vr23, smr0);
        qacc[56] = __dtu_m_vmm2_mode17_f16(qacc[56], vr24, smr0);
        qacc[57] = __dtu_m_vmm2_mode17_f16(qacc[57], vr25, smr0);
        qacc[58] = __dtu_m_vmm2_mode17_f16(qacc[58], vr26, smr0);
        qacc[59] = __dtu_m_vmm2_mode17_f16(qacc[59], vr27, smr0);
        qacc[60] = __dtu_m_vmm2_mode17_f16(qacc[60], vr28, smr0);
        qacc[61] = __dtu_m_vmm2_mode17_f16(qacc[61], vr29, smr0);
        qacc[62] = __dtu_m_vmm2_mode17_f16(qacc[62], vr30, smr0);
        qacc[63] = __dtu_m_vmm2_mode17_f16(qacc[63], vr31, smr0);

        // m0k1
        vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr31 = __dtu_s_tvld_itar(lt_base, lt_off0);
        // next k unit smr0
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 0);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 1);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 2);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 3);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 4);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 5);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 6);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 7);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 8);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 9);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 10);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 11);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 12);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 13);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 14);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 15);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 16);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 17);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 18);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 19);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 20);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 21);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 22);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 23);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 24);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 25);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 26);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 27);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 28);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 29);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 30);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 31);
        __dtu_c_movsr2naccovr(0x1);
        // m0k1 * smr1
        qacc[0] = __dtu_m_vmm2_mode17_f16(qacc[0], vr0, smr1);
        qacc[1] = __dtu_m_vmm2_mode17_f16(qacc[1], vr1, smr1);
        qacc[2] = __dtu_m_vmm2_mode17_f16(qacc[2], vr2, smr1);
        qacc[3] = __dtu_m_vmm2_mode17_f16(qacc[3], vr3, smr1);
        qacc[4] = __dtu_m_vmm2_mode17_f16(qacc[4], vr4, smr1);
        qacc[5] = __dtu_m_vmm2_mode17_f16(qacc[5], vr5, smr1);
        qacc[6] = __dtu_m_vmm2_mode17_f16(qacc[6], vr6, smr1);
        qacc[7] = __dtu_m_vmm2_mode17_f16(qacc[7], vr7, smr1);
        qacc[8] = __dtu_m_vmm2_mode17_f16(qacc[8], vr8, smr1);
        qacc[9] = __dtu_m_vmm2_mode17_f16(qacc[9], vr9, smr1);
        qacc[10] = __dtu_m_vmm2_mode17_f16(qacc[10], vr10, smr1);
        qacc[11] = __dtu_m_vmm2_mode17_f16(qacc[11], vr11, smr1);
        qacc[12] = __dtu_m_vmm2_mode17_f16(qacc[12], vr12, smr1);
        qacc[13] = __dtu_m_vmm2_mode17_f16(qacc[13], vr13, smr1);
        qacc[14] = __dtu_m_vmm2_mode17_f16(qacc[14], vr14, smr1);
        qacc[15] = __dtu_m_vmm2_mode17_f16(qacc[15], vr15, smr1);
        qacc[16] = __dtu_m_vmm2_mode17_f16(qacc[16], vr16, smr1);
        qacc[17] = __dtu_m_vmm2_mode17_f16(qacc[17], vr17, smr1);
        qacc[18] = __dtu_m_vmm2_mode17_f16(qacc[18], vr18, smr1);
        qacc[19] = __dtu_m_vmm2_mode17_f16(qacc[19], vr19, smr1);
        qacc[20] = __dtu_m_vmm2_mode17_f16(qacc[20], vr20, smr1);
        qacc[21] = __dtu_m_vmm2_mode17_f16(qacc[21], vr21, smr1);
        qacc[22] = __dtu_m_vmm2_mode17_f16(qacc[22], vr22, smr1);
        qacc[23] = __dtu_m_vmm2_mode17_f16(qacc[23], vr23, smr1);
        qacc[24] = __dtu_m_vmm2_mode17_f16(qacc[24], vr24, smr1);
        qacc[25] = __dtu_m_vmm2_mode17_f16(qacc[25], vr25, smr1);
        qacc[26] = __dtu_m_vmm2_mode17_f16(qacc[26], vr26, smr1);
        qacc[27] = __dtu_m_vmm2_mode17_f16(qacc[27], vr27, smr1);
        qacc[28] = __dtu_m_vmm2_mode17_f16(qacc[28], vr28, smr1);
        qacc[29] = __dtu_m_vmm2_mode17_f16(qacc[29], vr29, smr1);
        qacc[30] = __dtu_m_vmm2_mode17_f16(qacc[30], vr30, smr1);
        qacc[31] = __dtu_m_vmm2_mode17_f16(qacc[31], vr31, smr1);
        // m1k1
        vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr31 = __dtu_s_tvld_itar(lt_base, lt_off1);
        // m1k1 * smr1
        qacc[32] = __dtu_m_vmm2_mode17_f16(qacc[32], vr0, smr1);
        qacc[33] = __dtu_m_vmm2_mode17_f16(qacc[33], vr1, smr1);
        qacc[34] = __dtu_m_vmm2_mode17_f16(qacc[34], vr2, smr1);
        qacc[35] = __dtu_m_vmm2_mode17_f16(qacc[35], vr3, smr1);
        qacc[36] = __dtu_m_vmm2_mode17_f16(qacc[36], vr4, smr1);
        qacc[37] = __dtu_m_vmm2_mode17_f16(qacc[37], vr5, smr1);
        qacc[38] = __dtu_m_vmm2_mode17_f16(qacc[38], vr6, smr1);
        qacc[39] = __dtu_m_vmm2_mode17_f16(qacc[39], vr7, smr1);
        qacc[40] = __dtu_m_vmm2_mode17_f16(qacc[40], vr8, smr1);
        qacc[41] = __dtu_m_vmm2_mode17_f16(qacc[41], vr9, smr1);
        qacc[42] = __dtu_m_vmm2_mode17_f16(qacc[42], vr10, smr1);
        qacc[43] = __dtu_m_vmm2_mode17_f16(qacc[43], vr11, smr1);
        qacc[44] = __dtu_m_vmm2_mode17_f16(qacc[44], vr12, smr1);
        qacc[45] = __dtu_m_vmm2_mode17_f16(qacc[45], vr13, smr1);
        qacc[46] = __dtu_m_vmm2_mode17_f16(qacc[46], vr14, smr1);
        qacc[47] = __dtu_m_vmm2_mode17_f16(qacc[47], vr15, smr1);
        qacc[48] = __dtu_m_vmm2_mode17_f16(qacc[48], vr16, smr1);
        qacc[49] = __dtu_m_vmm2_mode17_f16(qacc[49], vr17, smr1);
        qacc[50] = __dtu_m_vmm2_mode17_f16(qacc[50], vr18, smr1);
        qacc[51] = __dtu_m_vmm2_mode17_f16(qacc[51], vr19, smr1);
        qacc[52] = __dtu_m_vmm2_mode17_f16(qacc[52], vr20, smr1);
        qacc[53] = __dtu_m_vmm2_mode17_f16(qacc[53], vr21, smr1);
        qacc[54] = __dtu_m_vmm2_mode17_f16(qacc[54], vr22, smr1);
        qacc[55] = __dtu_m_vmm2_mode17_f16(qacc[55], vr23, smr1);
        qacc[56] = __dtu_m_vmm2_mode17_f16(qacc[56], vr24, smr1);
        qacc[57] = __dtu_m_vmm2_mode17_f16(qacc[57], vr25, smr1);
        qacc[58] = __dtu_m_vmm2_mode17_f16(qacc[58], vr26, smr1);
        qacc[59] = __dtu_m_vmm2_mode17_f16(qacc[59], vr27, smr1);
        qacc[60] = __dtu_m_vmm2_mode17_f16(qacc[60], vr28, smr1);
        qacc[61] = __dtu_m_vmm2_mode17_f16(qacc[61], vr29, smr1);
        qacc[62] = __dtu_m_vmm2_mode17_f16(qacc[62], vr30, smr1);
        qacc[63] = __dtu_m_vmm2_mode17_f16(qacc[63], vr31, smr1);
        // next k unit m0k0
        vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr31 = __dtu_s_tvld_itar(lt_base, lt_off0);
      }  // end kcout-1
      // last k unit
      // smr1
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 0);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 1);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 2);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 3);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 4);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 5);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 6);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 7);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 8);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 9);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 10);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 11);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 12);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 13);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 14);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 15);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 16);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 17);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 18);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 19);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 20);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 21);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 22);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 23);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 24);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 25);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 26);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 27);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 28);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 29);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 30);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off1, 31);
      // m0k0 * smr0
      qacc[0] = __dtu_m_vmm2_mode17_f16(qacc[0], vr0, smr0);
      qacc[1] = __dtu_m_vmm2_mode17_f16(qacc[1], vr1, smr0);
      qacc[2] = __dtu_m_vmm2_mode17_f16(qacc[2], vr2, smr0);
      qacc[3] = __dtu_m_vmm2_mode17_f16(qacc[3], vr3, smr0);
      qacc[4] = __dtu_m_vmm2_mode17_f16(qacc[4], vr4, smr0);
      qacc[5] = __dtu_m_vmm2_mode17_f16(qacc[5], vr5, smr0);
      qacc[6] = __dtu_m_vmm2_mode17_f16(qacc[6], vr6, smr0);
      qacc[7] = __dtu_m_vmm2_mode17_f16(qacc[7], vr7, smr0);
      qacc[8] = __dtu_m_vmm2_mode17_f16(qacc[8], vr8, smr0);
      qacc[9] = __dtu_m_vmm2_mode17_f16(qacc[9], vr9, smr0);
      qacc[10] = __dtu_m_vmm2_mode17_f16(qacc[10], vr10, smr0);
      qacc[11] = __dtu_m_vmm2_mode17_f16(qacc[11], vr11, smr0);
      qacc[12] = __dtu_m_vmm2_mode17_f16(qacc[12], vr12, smr0);
      qacc[13] = __dtu_m_vmm2_mode17_f16(qacc[13], vr13, smr0);
      qacc[14] = __dtu_m_vmm2_mode17_f16(qacc[14], vr14, smr0);
      qacc[15] = __dtu_m_vmm2_mode17_f16(qacc[15], vr15, smr0);
      qacc[16] = __dtu_m_vmm2_mode17_f16(qacc[16], vr16, smr0);
      qacc[17] = __dtu_m_vmm2_mode17_f16(qacc[17], vr17, smr0);
      qacc[18] = __dtu_m_vmm2_mode17_f16(qacc[18], vr18, smr0);
      qacc[19] = __dtu_m_vmm2_mode17_f16(qacc[19], vr19, smr0);
      qacc[20] = __dtu_m_vmm2_mode17_f16(qacc[20], vr20, smr0);
      qacc[21] = __dtu_m_vmm2_mode17_f16(qacc[21], vr21, smr0);
      qacc[22] = __dtu_m_vmm2_mode17_f16(qacc[22], vr22, smr0);
      qacc[23] = __dtu_m_vmm2_mode17_f16(qacc[23], vr23, smr0);
      qacc[24] = __dtu_m_vmm2_mode17_f16(qacc[24], vr24, smr0);
      qacc[25] = __dtu_m_vmm2_mode17_f16(qacc[25], vr25, smr0);
      qacc[26] = __dtu_m_vmm2_mode17_f16(qacc[26], vr26, smr0);
      qacc[27] = __dtu_m_vmm2_mode17_f16(qacc[27], vr27, smr0);
      qacc[28] = __dtu_m_vmm2_mode17_f16(qacc[28], vr28, smr0);
      qacc[29] = __dtu_m_vmm2_mode17_f16(qacc[29], vr29, smr0);
      qacc[30] = __dtu_m_vmm2_mode17_f16(qacc[30], vr30, smr0);
      qacc[31] = __dtu_m_vmm2_mode17_f16(qacc[31], vr31, smr0);
      // m1k0
      vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr31 = __dtu_s_tvld_itar(lt_base, lt_off1);
      // m1k0 * smr0
      qacc[32] = __dtu_m_vmm2_mode17_f16(qacc[32], vr0, smr0);
      qacc[33] = __dtu_m_vmm2_mode17_f16(qacc[33], vr1, smr0);
      qacc[34] = __dtu_m_vmm2_mode17_f16(qacc[34], vr2, smr0);
      qacc[35] = __dtu_m_vmm2_mode17_f16(qacc[35], vr3, smr0);
      qacc[36] = __dtu_m_vmm2_mode17_f16(qacc[36], vr4, smr0);
      qacc[37] = __dtu_m_vmm2_mode17_f16(qacc[37], vr5, smr0);
      qacc[38] = __dtu_m_vmm2_mode17_f16(qacc[38], vr6, smr0);
      qacc[39] = __dtu_m_vmm2_mode17_f16(qacc[39], vr7, smr0);
      qacc[40] = __dtu_m_vmm2_mode17_f16(qacc[40], vr8, smr0);
      qacc[41] = __dtu_m_vmm2_mode17_f16(qacc[41], vr9, smr0);
      qacc[42] = __dtu_m_vmm2_mode17_f16(qacc[42], vr10, smr0);
      qacc[43] = __dtu_m_vmm2_mode17_f16(qacc[43], vr11, smr0);
      qacc[44] = __dtu_m_vmm2_mode17_f16(qacc[44], vr12, smr0);
      qacc[45] = __dtu_m_vmm2_mode17_f16(qacc[45], vr13, smr0);
      qacc[46] = __dtu_m_vmm2_mode17_f16(qacc[46], vr14, smr0);
      qacc[47] = __dtu_m_vmm2_mode17_f16(qacc[47], vr15, smr0);
      qacc[48] = __dtu_m_vmm2_mode17_f16(qacc[48], vr16, smr0);
      qacc[49] = __dtu_m_vmm2_mode17_f16(qacc[49], vr17, smr0);
      qacc[50] = __dtu_m_vmm2_mode17_f16(qacc[50], vr18, smr0);
      qacc[51] = __dtu_m_vmm2_mode17_f16(qacc[51], vr19, smr0);
      qacc[52] = __dtu_m_vmm2_mode17_f16(qacc[52], vr20, smr0);
      qacc[53] = __dtu_m_vmm2_mode17_f16(qacc[53], vr21, smr0);
      qacc[54] = __dtu_m_vmm2_mode17_f16(qacc[54], vr22, smr0);
      qacc[55] = __dtu_m_vmm2_mode17_f16(qacc[55], vr23, smr0);
      qacc[56] = __dtu_m_vmm2_mode17_f16(qacc[56], vr24, smr0);
      qacc[57] = __dtu_m_vmm2_mode17_f16(qacc[57], vr25, smr0);
      qacc[58] = __dtu_m_vmm2_mode17_f16(qacc[58], vr26, smr0);
      qacc[59] = __dtu_m_vmm2_mode17_f16(qacc[59], vr27, smr0);
      qacc[60] = __dtu_m_vmm2_mode17_f16(qacc[60], vr28, smr0);
      qacc[61] = __dtu_m_vmm2_mode17_f16(qacc[61], vr29, smr0);
      qacc[62] = __dtu_m_vmm2_mode17_f16(qacc[62], vr30, smr0);
      qacc[63] = __dtu_m_vmm2_mode17_f16(qacc[63], vr31, smr0);

      // m0k1
      vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr31 = __dtu_s_tvld_itar(lt_base, lt_off0);  // end k new n
      // next n unit smr0
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 0);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 1);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 2);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 3);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 4);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 5);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 6);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 7);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 8);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 9);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 10);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 11);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 12);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 13);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 14);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 15);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 16);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 17);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 18);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 19);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 20);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 21);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 22);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 23);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 24);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 25);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 26);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 27);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 28);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 29);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 30);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 31);
      __dtu_c_movsr2naccovr(0x1);
      // m0k1 * smr1
      qacc[0] = __dtu_m_vmm2_mode17_f16(qacc[0], vr0, smr1);
      qacc[1] = __dtu_m_vmm2_mode17_f16(qacc[1], vr1, smr1);
      qacc[2] = __dtu_m_vmm2_mode17_f16(qacc[2], vr2, smr1);
      qacc[3] = __dtu_m_vmm2_mode17_f16(qacc[3], vr3, smr1);
      qacc[4] = __dtu_m_vmm2_mode17_f16(qacc[4], vr4, smr1);
      qacc[5] = __dtu_m_vmm2_mode17_f16(qacc[5], vr5, smr1);
      qacc[6] = __dtu_m_vmm2_mode17_f16(qacc[6], vr6, smr1);
      qacc[7] = __dtu_m_vmm2_mode17_f16(qacc[7], vr7, smr1);
      qacc[8] = __dtu_m_vmm2_mode17_f16(qacc[8], vr8, smr1);
      qacc[9] = __dtu_m_vmm2_mode17_f16(qacc[9], vr9, smr1);
      qacc[10] = __dtu_m_vmm2_mode17_f16(qacc[10], vr10, smr1);
      qacc[11] = __dtu_m_vmm2_mode17_f16(qacc[11], vr11, smr1);
      qacc[12] = __dtu_m_vmm2_mode17_f16(qacc[12], vr12, smr1);
      qacc[13] = __dtu_m_vmm2_mode17_f16(qacc[13], vr13, smr1);
      qacc[14] = __dtu_m_vmm2_mode17_f16(qacc[14], vr14, smr1);
      qacc[15] = __dtu_m_vmm2_mode17_f16(qacc[15], vr15, smr1);
      qacc[16] = __dtu_m_vmm2_mode17_f16(qacc[16], vr16, smr1);
      qacc[17] = __dtu_m_vmm2_mode17_f16(qacc[17], vr17, smr1);
      qacc[18] = __dtu_m_vmm2_mode17_f16(qacc[18], vr18, smr1);
      qacc[19] = __dtu_m_vmm2_mode17_f16(qacc[19], vr19, smr1);
      qacc[20] = __dtu_m_vmm2_mode17_f16(qacc[20], vr20, smr1);
      qacc[21] = __dtu_m_vmm2_mode17_f16(qacc[21], vr21, smr1);
      qacc[22] = __dtu_m_vmm2_mode17_f16(qacc[22], vr22, smr1);
      qacc[23] = __dtu_m_vmm2_mode17_f16(qacc[23], vr23, smr1);
      qacc[24] = __dtu_m_vmm2_mode17_f16(qacc[24], vr24, smr1);
      qacc[25] = __dtu_m_vmm2_mode17_f16(qacc[25], vr25, smr1);
      qacc[26] = __dtu_m_vmm2_mode17_f16(qacc[26], vr26, smr1);
      qacc[27] = __dtu_m_vmm2_mode17_f16(qacc[27], vr27, smr1);
      qacc[28] = __dtu_m_vmm2_mode17_f16(qacc[28], vr28, smr1);
      qacc[29] = __dtu_m_vmm2_mode17_f16(qacc[29], vr29, smr1);
      qacc[30] = __dtu_m_vmm2_mode17_f16(qacc[30], vr30, smr1);
      qacc[31] = __dtu_m_vmm2_mode17_f16(qacc[31], vr31, smr1);
      // m1k1
      vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr31 = __dtu_s_tvld_itar(lt_base, lt_off2);
      // m1k1 * smr1
      qacc[32] = __dtu_m_vmm2_mode17_f16(qacc[32], vr0, smr1);
      qacc[33] = __dtu_m_vmm2_mode17_f16(qacc[33], vr1, smr1);
      qacc[34] = __dtu_m_vmm2_mode17_f16(qacc[34], vr2, smr1);
      qacc[35] = __dtu_m_vmm2_mode17_f16(qacc[35], vr3, smr1);
      qacc[36] = __dtu_m_vmm2_mode17_f16(qacc[36], vr4, smr1);
      qacc[37] = __dtu_m_vmm2_mode17_f16(qacc[37], vr5, smr1);
      qacc[38] = __dtu_m_vmm2_mode17_f16(qacc[38], vr6, smr1);
      qacc[39] = __dtu_m_vmm2_mode17_f16(qacc[39], vr7, smr1);
      qacc[40] = __dtu_m_vmm2_mode17_f16(qacc[40], vr8, smr1);
      qacc[41] = __dtu_m_vmm2_mode17_f16(qacc[41], vr9, smr1);
      qacc[42] = __dtu_m_vmm2_mode17_f16(qacc[42], vr10, smr1);
      qacc[43] = __dtu_m_vmm2_mode17_f16(qacc[43], vr11, smr1);
      qacc[44] = __dtu_m_vmm2_mode17_f16(qacc[44], vr12, smr1);
      qacc[45] = __dtu_m_vmm2_mode17_f16(qacc[45], vr13, smr1);
      qacc[46] = __dtu_m_vmm2_mode17_f16(qacc[46], vr14, smr1);
      qacc[47] = __dtu_m_vmm2_mode17_f16(qacc[47], vr15, smr1);
      qacc[48] = __dtu_m_vmm2_mode17_f16(qacc[48], vr16, smr1);
      qacc[49] = __dtu_m_vmm2_mode17_f16(qacc[49], vr17, smr1);
      qacc[50] = __dtu_m_vmm2_mode17_f16(qacc[50], vr18, smr1);
      qacc[51] = __dtu_m_vmm2_mode17_f16(qacc[51], vr19, smr1);
      qacc[52] = __dtu_m_vmm2_mode17_f16(qacc[52], vr20, smr1);
      qacc[53] = __dtu_m_vmm2_mode17_f16(qacc[53], vr21, smr1);
      qacc[54] = __dtu_m_vmm2_mode17_f16(qacc[54], vr22, smr1);
      qacc[55] = __dtu_m_vmm2_mode17_f16(qacc[55], vr23, smr1);
      qacc[56] = __dtu_m_vmm2_mode17_f16(qacc[56], vr24, smr1);
      qacc[57] = __dtu_m_vmm2_mode17_f16(qacc[57], vr25, smr1);
      qacc[58] = __dtu_m_vmm2_mode17_f16(qacc[58], vr26, smr1);
      qacc[59] = __dtu_m_vmm2_mode17_f16(qacc[59], vr27, smr1);
      qacc[60] = __dtu_m_vmm2_mode17_f16(qacc[60], vr28, smr1);
      qacc[61] = __dtu_m_vmm2_mode17_f16(qacc[61], vr29, smr1);
      qacc[62] = __dtu_m_vmm2_mode17_f16(qacc[62], vr30, smr1);
      qacc[63] = __dtu_m_vmm2_mode17_f16(qacc[63], vr31, smr1);
      // next n unit m0k0
      vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr31 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vab_shift += 512;
      __dtu_c_movsr2vab_m_s1(vab_shift);
      __dtu_c_movsr2vab_m_d(vab_shift);
    }  // end ncount-1
    __dtu_c_movsr2naccovr(naccovr);
    for (int k = 0; k < K - 64; k += 64) {
      // smr1
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 0);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 1);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 2);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 3);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 4);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 5);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 6);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 7);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 8);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 9);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 10);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 11);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 12);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 13);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 14);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 15);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 16);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 17);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 18);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 19);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 20);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 21);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 22);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 23);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 24);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 25);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 26);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 27);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 28);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 29);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 30);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 31);
      // m0k0 * smr0
      qacc[0] = __dtu_m_vmm2_mode17_f16(qacc[0], vr0, smr0);
      qacc[1] = __dtu_m_vmm2_mode17_f16(qacc[1], vr1, smr0);
      qacc[2] = __dtu_m_vmm2_mode17_f16(qacc[2], vr2, smr0);
      qacc[3] = __dtu_m_vmm2_mode17_f16(qacc[3], vr3, smr0);
      qacc[4] = __dtu_m_vmm2_mode17_f16(qacc[4], vr4, smr0);
      qacc[5] = __dtu_m_vmm2_mode17_f16(qacc[5], vr5, smr0);
      qacc[6] = __dtu_m_vmm2_mode17_f16(qacc[6], vr6, smr0);
      qacc[7] = __dtu_m_vmm2_mode17_f16(qacc[7], vr7, smr0);
      qacc[8] = __dtu_m_vmm2_mode17_f16(qacc[8], vr8, smr0);
      qacc[9] = __dtu_m_vmm2_mode17_f16(qacc[9], vr9, smr0);
      qacc[10] = __dtu_m_vmm2_mode17_f16(qacc[10], vr10, smr0);
      qacc[11] = __dtu_m_vmm2_mode17_f16(qacc[11], vr11, smr0);
      qacc[12] = __dtu_m_vmm2_mode17_f16(qacc[12], vr12, smr0);
      qacc[13] = __dtu_m_vmm2_mode17_f16(qacc[13], vr13, smr0);
      qacc[14] = __dtu_m_vmm2_mode17_f16(qacc[14], vr14, smr0);
      qacc[15] = __dtu_m_vmm2_mode17_f16(qacc[15], vr15, smr0);
      qacc[16] = __dtu_m_vmm2_mode17_f16(qacc[16], vr16, smr0);
      qacc[17] = __dtu_m_vmm2_mode17_f16(qacc[17], vr17, smr0);
      qacc[18] = __dtu_m_vmm2_mode17_f16(qacc[18], vr18, smr0);
      qacc[19] = __dtu_m_vmm2_mode17_f16(qacc[19], vr19, smr0);
      qacc[20] = __dtu_m_vmm2_mode17_f16(qacc[20], vr20, smr0);
      qacc[21] = __dtu_m_vmm2_mode17_f16(qacc[21], vr21, smr0);
      qacc[22] = __dtu_m_vmm2_mode17_f16(qacc[22], vr22, smr0);
      qacc[23] = __dtu_m_vmm2_mode17_f16(qacc[23], vr23, smr0);
      qacc[24] = __dtu_m_vmm2_mode17_f16(qacc[24], vr24, smr0);
      qacc[25] = __dtu_m_vmm2_mode17_f16(qacc[25], vr25, smr0);
      qacc[26] = __dtu_m_vmm2_mode17_f16(qacc[26], vr26, smr0);
      qacc[27] = __dtu_m_vmm2_mode17_f16(qacc[27], vr27, smr0);
      qacc[28] = __dtu_m_vmm2_mode17_f16(qacc[28], vr28, smr0);
      qacc[29] = __dtu_m_vmm2_mode17_f16(qacc[29], vr29, smr0);
      qacc[30] = __dtu_m_vmm2_mode17_f16(qacc[30], vr30, smr0);
      qacc[31] = __dtu_m_vmm2_mode17_f16(qacc[31], vr31, smr0);
      // m1k0
      vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr31 = __dtu_s_tvld_itar(lt_base, lt_off1);
      // m1k0 * smr0
      qacc[32] = __dtu_m_vmm2_mode17_f16(qacc[32], vr0, smr0);
      qacc[33] = __dtu_m_vmm2_mode17_f16(qacc[33], vr1, smr0);
      qacc[34] = __dtu_m_vmm2_mode17_f16(qacc[34], vr2, smr0);
      qacc[35] = __dtu_m_vmm2_mode17_f16(qacc[35], vr3, smr0);
      qacc[36] = __dtu_m_vmm2_mode17_f16(qacc[36], vr4, smr0);
      qacc[37] = __dtu_m_vmm2_mode17_f16(qacc[37], vr5, smr0);
      qacc[38] = __dtu_m_vmm2_mode17_f16(qacc[38], vr6, smr0);
      qacc[39] = __dtu_m_vmm2_mode17_f16(qacc[39], vr7, smr0);
      qacc[40] = __dtu_m_vmm2_mode17_f16(qacc[40], vr8, smr0);
      qacc[41] = __dtu_m_vmm2_mode17_f16(qacc[41], vr9, smr0);
      qacc[42] = __dtu_m_vmm2_mode17_f16(qacc[42], vr10, smr0);
      qacc[43] = __dtu_m_vmm2_mode17_f16(qacc[43], vr11, smr0);
      qacc[44] = __dtu_m_vmm2_mode17_f16(qacc[44], vr12, smr0);
      qacc[45] = __dtu_m_vmm2_mode17_f16(qacc[45], vr13, smr0);
      qacc[46] = __dtu_m_vmm2_mode17_f16(qacc[46], vr14, smr0);
      qacc[47] = __dtu_m_vmm2_mode17_f16(qacc[47], vr15, smr0);
      qacc[48] = __dtu_m_vmm2_mode17_f16(qacc[48], vr16, smr0);
      qacc[49] = __dtu_m_vmm2_mode17_f16(qacc[49], vr17, smr0);
      qacc[50] = __dtu_m_vmm2_mode17_f16(qacc[50], vr18, smr0);
      qacc[51] = __dtu_m_vmm2_mode17_f16(qacc[51], vr19, smr0);
      qacc[52] = __dtu_m_vmm2_mode17_f16(qacc[52], vr20, smr0);
      qacc[53] = __dtu_m_vmm2_mode17_f16(qacc[53], vr21, smr0);
      qacc[54] = __dtu_m_vmm2_mode17_f16(qacc[54], vr22, smr0);
      qacc[55] = __dtu_m_vmm2_mode17_f16(qacc[55], vr23, smr0);
      qacc[56] = __dtu_m_vmm2_mode17_f16(qacc[56], vr24, smr0);
      qacc[57] = __dtu_m_vmm2_mode17_f16(qacc[57], vr25, smr0);
      qacc[58] = __dtu_m_vmm2_mode17_f16(qacc[58], vr26, smr0);
      qacc[59] = __dtu_m_vmm2_mode17_f16(qacc[59], vr27, smr0);
      qacc[60] = __dtu_m_vmm2_mode17_f16(qacc[60], vr28, smr0);
      qacc[61] = __dtu_m_vmm2_mode17_f16(qacc[61], vr29, smr0);
      qacc[62] = __dtu_m_vmm2_mode17_f16(qacc[62], vr30, smr0);
      qacc[63] = __dtu_m_vmm2_mode17_f16(qacc[63], vr31, smr0);

      // m0k1
      vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr31 = __dtu_s_tvld_itar(lt_base, lt_off0);
      // next k unit smr0
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 0);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 1);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 2);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 3);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 4);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 5);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 6);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 7);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 8);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 9);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 10);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 11);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 12);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 13);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 14);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 15);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 16);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 17);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 18);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 19);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 20);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 21);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 22);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 23);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 24);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 25);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 26);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 27);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 28);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 29);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 30);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 31);
      __dtu_c_movsr2naccovr(0x1);
      // m0k1 * smr1
      qacc[0] = __dtu_m_vmm2_mode17_f16(qacc[0], vr0, smr1);
      qacc[1] = __dtu_m_vmm2_mode17_f16(qacc[1], vr1, smr1);
      qacc[2] = __dtu_m_vmm2_mode17_f16(qacc[2], vr2, smr1);
      qacc[3] = __dtu_m_vmm2_mode17_f16(qacc[3], vr3, smr1);
      qacc[4] = __dtu_m_vmm2_mode17_f16(qacc[4], vr4, smr1);
      qacc[5] = __dtu_m_vmm2_mode17_f16(qacc[5], vr5, smr1);
      qacc[6] = __dtu_m_vmm2_mode17_f16(qacc[6], vr6, smr1);
      qacc[7] = __dtu_m_vmm2_mode17_f16(qacc[7], vr7, smr1);
      qacc[8] = __dtu_m_vmm2_mode17_f16(qacc[8], vr8, smr1);
      qacc[9] = __dtu_m_vmm2_mode17_f16(qacc[9], vr9, smr1);
      qacc[10] = __dtu_m_vmm2_mode17_f16(qacc[10], vr10, smr1);
      qacc[11] = __dtu_m_vmm2_mode17_f16(qacc[11], vr11, smr1);
      qacc[12] = __dtu_m_vmm2_mode17_f16(qacc[12], vr12, smr1);
      qacc[13] = __dtu_m_vmm2_mode17_f16(qacc[13], vr13, smr1);
      qacc[14] = __dtu_m_vmm2_mode17_f16(qacc[14], vr14, smr1);
      qacc[15] = __dtu_m_vmm2_mode17_f16(qacc[15], vr15, smr1);
      qacc[16] = __dtu_m_vmm2_mode17_f16(qacc[16], vr16, smr1);
      qacc[17] = __dtu_m_vmm2_mode17_f16(qacc[17], vr17, smr1);
      qacc[18] = __dtu_m_vmm2_mode17_f16(qacc[18], vr18, smr1);
      qacc[19] = __dtu_m_vmm2_mode17_f16(qacc[19], vr19, smr1);
      qacc[20] = __dtu_m_vmm2_mode17_f16(qacc[20], vr20, smr1);
      qacc[21] = __dtu_m_vmm2_mode17_f16(qacc[21], vr21, smr1);
      qacc[22] = __dtu_m_vmm2_mode17_f16(qacc[22], vr22, smr1);
      qacc[23] = __dtu_m_vmm2_mode17_f16(qacc[23], vr23, smr1);
      qacc[24] = __dtu_m_vmm2_mode17_f16(qacc[24], vr24, smr1);
      qacc[25] = __dtu_m_vmm2_mode17_f16(qacc[25], vr25, smr1);
      qacc[26] = __dtu_m_vmm2_mode17_f16(qacc[26], vr26, smr1);
      qacc[27] = __dtu_m_vmm2_mode17_f16(qacc[27], vr27, smr1);
      qacc[28] = __dtu_m_vmm2_mode17_f16(qacc[28], vr28, smr1);
      qacc[29] = __dtu_m_vmm2_mode17_f16(qacc[29], vr29, smr1);
      qacc[30] = __dtu_m_vmm2_mode17_f16(qacc[30], vr30, smr1);
      qacc[31] = __dtu_m_vmm2_mode17_f16(qacc[31], vr31, smr1);
      // m1k1
      vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr31 = __dtu_s_tvld_itar(lt_base, lt_off1);
      // m1k1 * smr1
      qacc[32] = __dtu_m_vmm2_mode17_f16(qacc[32], vr0, smr1);
      qacc[33] = __dtu_m_vmm2_mode17_f16(qacc[33], vr1, smr1);
      qacc[34] = __dtu_m_vmm2_mode17_f16(qacc[34], vr2, smr1);
      qacc[35] = __dtu_m_vmm2_mode17_f16(qacc[35], vr3, smr1);
      qacc[36] = __dtu_m_vmm2_mode17_f16(qacc[36], vr4, smr1);
      qacc[37] = __dtu_m_vmm2_mode17_f16(qacc[37], vr5, smr1);
      qacc[38] = __dtu_m_vmm2_mode17_f16(qacc[38], vr6, smr1);
      qacc[39] = __dtu_m_vmm2_mode17_f16(qacc[39], vr7, smr1);
      qacc[40] = __dtu_m_vmm2_mode17_f16(qacc[40], vr8, smr1);
      qacc[41] = __dtu_m_vmm2_mode17_f16(qacc[41], vr9, smr1);
      qacc[42] = __dtu_m_vmm2_mode17_f16(qacc[42], vr10, smr1);
      qacc[43] = __dtu_m_vmm2_mode17_f16(qacc[43], vr11, smr1);
      qacc[44] = __dtu_m_vmm2_mode17_f16(qacc[44], vr12, smr1);
      qacc[45] = __dtu_m_vmm2_mode17_f16(qacc[45], vr13, smr1);
      qacc[46] = __dtu_m_vmm2_mode17_f16(qacc[46], vr14, smr1);
      qacc[47] = __dtu_m_vmm2_mode17_f16(qacc[47], vr15, smr1);
      qacc[48] = __dtu_m_vmm2_mode17_f16(qacc[48], vr16, smr1);
      qacc[49] = __dtu_m_vmm2_mode17_f16(qacc[49], vr17, smr1);
      qacc[50] = __dtu_m_vmm2_mode17_f16(qacc[50], vr18, smr1);
      qacc[51] = __dtu_m_vmm2_mode17_f16(qacc[51], vr19, smr1);
      qacc[52] = __dtu_m_vmm2_mode17_f16(qacc[52], vr20, smr1);
      qacc[53] = __dtu_m_vmm2_mode17_f16(qacc[53], vr21, smr1);
      qacc[54] = __dtu_m_vmm2_mode17_f16(qacc[54], vr22, smr1);
      qacc[55] = __dtu_m_vmm2_mode17_f16(qacc[55], vr23, smr1);
      qacc[56] = __dtu_m_vmm2_mode17_f16(qacc[56], vr24, smr1);
      qacc[57] = __dtu_m_vmm2_mode17_f16(qacc[57], vr25, smr1);
      qacc[58] = __dtu_m_vmm2_mode17_f16(qacc[58], vr26, smr1);
      qacc[59] = __dtu_m_vmm2_mode17_f16(qacc[59], vr27, smr1);
      qacc[60] = __dtu_m_vmm2_mode17_f16(qacc[60], vr28, smr1);
      qacc[61] = __dtu_m_vmm2_mode17_f16(qacc[61], vr29, smr1);
      qacc[62] = __dtu_m_vmm2_mode17_f16(qacc[62], vr30, smr1);
      qacc[63] = __dtu_m_vmm2_mode17_f16(qacc[63], vr31, smr1);

      // next k unit m0k0
      vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr31 = __dtu_s_tvld_itar(lt_base, lt_off0);
    }  // end kcout-1
    // last k unit of last n unit
    // smr1
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 1);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 2);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 3);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 4);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 5);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 6);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 7);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 8);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 9);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 10);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 11);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 12);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 13);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 14);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 15);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 16);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 17);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 18);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 19);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 20);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 21);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 22);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 23);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 24);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 25);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 26);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 27);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 28);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 29);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 30);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off2, 31);
    // m0k0 * smr0
    qacc[0] = __dtu_m_vmm2_mode17_f16(qacc[0], vr0, smr0);
    qacc[1] = __dtu_m_vmm2_mode17_f16(qacc[1], vr1, smr0);
    qacc[2] = __dtu_m_vmm2_mode17_f16(qacc[2], vr2, smr0);
    qacc[3] = __dtu_m_vmm2_mode17_f16(qacc[3], vr3, smr0);
    qacc[4] = __dtu_m_vmm2_mode17_f16(qacc[4], vr4, smr0);
    qacc[5] = __dtu_m_vmm2_mode17_f16(qacc[5], vr5, smr0);
    qacc[6] = __dtu_m_vmm2_mode17_f16(qacc[6], vr6, smr0);
    qacc[7] = __dtu_m_vmm2_mode17_f16(qacc[7], vr7, smr0);
    qacc[8] = __dtu_m_vmm2_mode17_f16(qacc[8], vr8, smr0);
    qacc[9] = __dtu_m_vmm2_mode17_f16(qacc[9], vr9, smr0);
    qacc[10] = __dtu_m_vmm2_mode17_f16(qacc[10], vr10, smr0);
    qacc[11] = __dtu_m_vmm2_mode17_f16(qacc[11], vr11, smr0);
    qacc[12] = __dtu_m_vmm2_mode17_f16(qacc[12], vr12, smr0);
    qacc[13] = __dtu_m_vmm2_mode17_f16(qacc[13], vr13, smr0);
    qacc[14] = __dtu_m_vmm2_mode17_f16(qacc[14], vr14, smr0);
    qacc[15] = __dtu_m_vmm2_mode17_f16(qacc[15], vr15, smr0);
    qacc[16] = __dtu_m_vmm2_mode17_f16(qacc[16], vr16, smr0);
    qacc[17] = __dtu_m_vmm2_mode17_f16(qacc[17], vr17, smr0);
    qacc[18] = __dtu_m_vmm2_mode17_f16(qacc[18], vr18, smr0);
    qacc[19] = __dtu_m_vmm2_mode17_f16(qacc[19], vr19, smr0);
    qacc[20] = __dtu_m_vmm2_mode17_f16(qacc[20], vr20, smr0);
    qacc[21] = __dtu_m_vmm2_mode17_f16(qacc[21], vr21, smr0);
    qacc[22] = __dtu_m_vmm2_mode17_f16(qacc[22], vr22, smr0);
    qacc[23] = __dtu_m_vmm2_mode17_f16(qacc[23], vr23, smr0);
    qacc[24] = __dtu_m_vmm2_mode17_f16(qacc[24], vr24, smr0);
    qacc[25] = __dtu_m_vmm2_mode17_f16(qacc[25], vr25, smr0);
    qacc[26] = __dtu_m_vmm2_mode17_f16(qacc[26], vr26, smr0);
    qacc[27] = __dtu_m_vmm2_mode17_f16(qacc[27], vr27, smr0);
    qacc[28] = __dtu_m_vmm2_mode17_f16(qacc[28], vr28, smr0);
    qacc[29] = __dtu_m_vmm2_mode17_f16(qacc[29], vr29, smr0);
    qacc[30] = __dtu_m_vmm2_mode17_f16(qacc[30], vr30, smr0);
    qacc[31] = __dtu_m_vmm2_mode17_f16(qacc[31], vr31, smr0);
    // m1k0
    vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr31 = __dtu_s_tvld_itar(lt_base, lt_off1);
    // m1k0 * smr0
    qacc[32] = __dtu_m_vmm2_mode17_f16(qacc[32], vr0, smr0);
    qacc[33] = __dtu_m_vmm2_mode17_f16(qacc[33], vr1, smr0);
    qacc[34] = __dtu_m_vmm2_mode17_f16(qacc[34], vr2, smr0);
    qacc[35] = __dtu_m_vmm2_mode17_f16(qacc[35], vr3, smr0);
    qacc[36] = __dtu_m_vmm2_mode17_f16(qacc[36], vr4, smr0);
    qacc[37] = __dtu_m_vmm2_mode17_f16(qacc[37], vr5, smr0);
    qacc[38] = __dtu_m_vmm2_mode17_f16(qacc[38], vr6, smr0);
    qacc[39] = __dtu_m_vmm2_mode17_f16(qacc[39], vr7, smr0);
    qacc[40] = __dtu_m_vmm2_mode17_f16(qacc[40], vr8, smr0);
    qacc[41] = __dtu_m_vmm2_mode17_f16(qacc[41], vr9, smr0);
    qacc[42] = __dtu_m_vmm2_mode17_f16(qacc[42], vr10, smr0);
    qacc[43] = __dtu_m_vmm2_mode17_f16(qacc[43], vr11, smr0);
    qacc[44] = __dtu_m_vmm2_mode17_f16(qacc[44], vr12, smr0);
    qacc[45] = __dtu_m_vmm2_mode17_f16(qacc[45], vr13, smr0);
    qacc[46] = __dtu_m_vmm2_mode17_f16(qacc[46], vr14, smr0);
    qacc[47] = __dtu_m_vmm2_mode17_f16(qacc[47], vr15, smr0);
    qacc[48] = __dtu_m_vmm2_mode17_f16(qacc[48], vr16, smr0);
    qacc[49] = __dtu_m_vmm2_mode17_f16(qacc[49], vr17, smr0);
    qacc[50] = __dtu_m_vmm2_mode17_f16(qacc[50], vr18, smr0);
    qacc[51] = __dtu_m_vmm2_mode17_f16(qacc[51], vr19, smr0);
    qacc[52] = __dtu_m_vmm2_mode17_f16(qacc[52], vr20, smr0);
    qacc[53] = __dtu_m_vmm2_mode17_f16(qacc[53], vr21, smr0);
    qacc[54] = __dtu_m_vmm2_mode17_f16(qacc[54], vr22, smr0);
    qacc[55] = __dtu_m_vmm2_mode17_f16(qacc[55], vr23, smr0);
    qacc[56] = __dtu_m_vmm2_mode17_f16(qacc[56], vr24, smr0);
    qacc[57] = __dtu_m_vmm2_mode17_f16(qacc[57], vr25, smr0);
    qacc[58] = __dtu_m_vmm2_mode17_f16(qacc[58], vr26, smr0);
    qacc[59] = __dtu_m_vmm2_mode17_f16(qacc[59], vr27, smr0);
    qacc[60] = __dtu_m_vmm2_mode17_f16(qacc[60], vr28, smr0);
    qacc[61] = __dtu_m_vmm2_mode17_f16(qacc[61], vr29, smr0);
    qacc[62] = __dtu_m_vmm2_mode17_f16(qacc[62], vr30, smr0);
    qacc[63] = __dtu_m_vmm2_mode17_f16(qacc[63], vr31, smr0);

    // m0k1
    vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr31 = __dtu_s_tvld_itar(lt_base, lt_off0);
    // next m unit smr0
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 1);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 2);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 3);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 4);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 5);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 6);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 7);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 8);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 9);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 10);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 11);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 12);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 13);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 14);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 15);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 16);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 17);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 18);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 19);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 20);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 21);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 22);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 23);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 24);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 25);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 26);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 27);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 28);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 29);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 30);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 31);
    __dtu_c_movsr2naccovr(0x1);
    // m0k1 * smr1
    qacc[0] = __dtu_m_vmm2_mode17_f16(qacc[0], vr0, smr1);
    qacc[1] = __dtu_m_vmm2_mode17_f16(qacc[1], vr1, smr1);
    qacc[2] = __dtu_m_vmm2_mode17_f16(qacc[2], vr2, smr1);
    qacc[3] = __dtu_m_vmm2_mode17_f16(qacc[3], vr3, smr1);
    qacc[4] = __dtu_m_vmm2_mode17_f16(qacc[4], vr4, smr1);
    qacc[5] = __dtu_m_vmm2_mode17_f16(qacc[5], vr5, smr1);
    qacc[6] = __dtu_m_vmm2_mode17_f16(qacc[6], vr6, smr1);
    qacc[7] = __dtu_m_vmm2_mode17_f16(qacc[7], vr7, smr1);
    qacc[8] = __dtu_m_vmm2_mode17_f16(qacc[8], vr8, smr1);
    qacc[9] = __dtu_m_vmm2_mode17_f16(qacc[9], vr9, smr1);
    qacc[10] = __dtu_m_vmm2_mode17_f16(qacc[10], vr10, smr1);
    qacc[11] = __dtu_m_vmm2_mode17_f16(qacc[11], vr11, smr1);
    qacc[12] = __dtu_m_vmm2_mode17_f16(qacc[12], vr12, smr1);
    qacc[13] = __dtu_m_vmm2_mode17_f16(qacc[13], vr13, smr1);
    qacc[14] = __dtu_m_vmm2_mode17_f16(qacc[14], vr14, smr1);
    qacc[15] = __dtu_m_vmm2_mode17_f16(qacc[15], vr15, smr1);
    qacc[16] = __dtu_m_vmm2_mode17_f16(qacc[16], vr16, smr1);
    qacc[17] = __dtu_m_vmm2_mode17_f16(qacc[17], vr17, smr1);
    qacc[18] = __dtu_m_vmm2_mode17_f16(qacc[18], vr18, smr1);
    qacc[19] = __dtu_m_vmm2_mode17_f16(qacc[19], vr19, smr1);
    qacc[20] = __dtu_m_vmm2_mode17_f16(qacc[20], vr20, smr1);
    qacc[21] = __dtu_m_vmm2_mode17_f16(qacc[21], vr21, smr1);
    qacc[22] = __dtu_m_vmm2_mode17_f16(qacc[22], vr22, smr1);
    qacc[23] = __dtu_m_vmm2_mode17_f16(qacc[23], vr23, smr1);
    qacc[24] = __dtu_m_vmm2_mode17_f16(qacc[24], vr24, smr1);
    qacc[25] = __dtu_m_vmm2_mode17_f16(qacc[25], vr25, smr1);
    qacc[26] = __dtu_m_vmm2_mode17_f16(qacc[26], vr26, smr1);
    qacc[27] = __dtu_m_vmm2_mode17_f16(qacc[27], vr27, smr1);
    qacc[28] = __dtu_m_vmm2_mode17_f16(qacc[28], vr28, smr1);
    qacc[29] = __dtu_m_vmm2_mode17_f16(qacc[29], vr29, smr1);
    qacc[30] = __dtu_m_vmm2_mode17_f16(qacc[30], vr30, smr1);
    qacc[31] = __dtu_m_vmm2_mode17_f16(qacc[31], vr31, smr1);
    // m1k1
    vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr31 = __dtu_s_tvld_itar(lt_base, lt_off3);
    // m1k1 * smr1
    qacc[32] = __dtu_m_vmm2_mode17_f16(qacc[32], vr0, smr1);
    qacc[33] = __dtu_m_vmm2_mode17_f16(qacc[33], vr1, smr1);
    qacc[34] = __dtu_m_vmm2_mode17_f16(qacc[34], vr2, smr1);
    qacc[35] = __dtu_m_vmm2_mode17_f16(qacc[35], vr3, smr1);
    qacc[36] = __dtu_m_vmm2_mode17_f16(qacc[36], vr4, smr1);
    qacc[37] = __dtu_m_vmm2_mode17_f16(qacc[37], vr5, smr1);
    qacc[38] = __dtu_m_vmm2_mode17_f16(qacc[38], vr6, smr1);
    qacc[39] = __dtu_m_vmm2_mode17_f16(qacc[39], vr7, smr1);
    qacc[40] = __dtu_m_vmm2_mode17_f16(qacc[40], vr8, smr1);
    qacc[41] = __dtu_m_vmm2_mode17_f16(qacc[41], vr9, smr1);
    qacc[42] = __dtu_m_vmm2_mode17_f16(qacc[42], vr10, smr1);
    qacc[43] = __dtu_m_vmm2_mode17_f16(qacc[43], vr11, smr1);
    qacc[44] = __dtu_m_vmm2_mode17_f16(qacc[44], vr12, smr1);
    qacc[45] = __dtu_m_vmm2_mode17_f16(qacc[45], vr13, smr1);
    qacc[46] = __dtu_m_vmm2_mode17_f16(qacc[46], vr14, smr1);
    qacc[47] = __dtu_m_vmm2_mode17_f16(qacc[47], vr15, smr1);
    qacc[48] = __dtu_m_vmm2_mode17_f16(qacc[48], vr16, smr1);
    qacc[49] = __dtu_m_vmm2_mode17_f16(qacc[49], vr17, smr1);
    qacc[50] = __dtu_m_vmm2_mode17_f16(qacc[50], vr18, smr1);
    qacc[51] = __dtu_m_vmm2_mode17_f16(qacc[51], vr19, smr1);
    qacc[52] = __dtu_m_vmm2_mode17_f16(qacc[52], vr20, smr1);
    qacc[53] = __dtu_m_vmm2_mode17_f16(qacc[53], vr21, smr1);
    qacc[54] = __dtu_m_vmm2_mode17_f16(qacc[54], vr22, smr1);
    qacc[55] = __dtu_m_vmm2_mode17_f16(qacc[55], vr23, smr1);
    qacc[56] = __dtu_m_vmm2_mode17_f16(qacc[56], vr24, smr1);
    qacc[57] = __dtu_m_vmm2_mode17_f16(qacc[57], vr25, smr1);
    qacc[58] = __dtu_m_vmm2_mode17_f16(qacc[58], vr26, smr1);
    qacc[59] = __dtu_m_vmm2_mode17_f16(qacc[59], vr27, smr1);
    qacc[60] = __dtu_m_vmm2_mode17_f16(qacc[60], vr28, smr1);
    qacc[61] = __dtu_m_vmm2_mode17_f16(qacc[61], vr29, smr1);
    qacc[62] = __dtu_m_vmm2_mode17_f16(qacc[62], vr30, smr1);
    qacc[63] = __dtu_m_vmm2_mode17_f16(qacc[63], vr31, smr1);

    // next m unit m0k0
    vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr31 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vab_shift += 512;
    __dtu_c_movsr2vab_m_s1(vab_shift);
    __dtu_c_movsr2vab_m_d(vab_shift);
  }  // end mcount

  if (stroe_flag) {
    vab_shift = 0;
    __dtu_c_movsr2vab_lv_s(0);
    __dtu_c_movsr2vab_lv_d(0);
    __dtu_c_movsr2vab_m_s1(0);
    __dtu_c_movsr2vab_m_d(0);
#pragma clang loop unroll(disable)
    for (int m = 0; m < M; m = m + 64) {
#pragma clang loop unroll(disable)
      for (int n = 0; n < N - 128; n = n + 128) {
        bs_dacc = __dtu_l_tvldqa_f16_da(biast_base, biast_off0);
        c_dacc[0] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[1] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[2] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[3] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[4] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[5] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[6] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[7] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[8] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[9] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[10] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[11] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[12] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[13] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[14] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[15] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[16] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[17] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[18] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[19] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[20] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[21] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[22] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[23] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[24] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[25] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[26] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[27] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[28] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[29] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[30] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[31] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[32] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[33] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[34] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[35] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[36] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[37] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[38] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[39] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[40] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[41] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[42] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[43] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[44] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[45] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[46] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[47] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[48] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[49] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[50] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[51] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[52] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[53] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[54] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[55] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[56] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[57] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[58] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[59] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[60] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[61] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[62] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
        c_dacc[63] = __dtu_l_tvldqa_f16_da(bt_base, bt_off1);
        __dtu_c_movsr2vab_m_s2(0);
        qacc[0] = __dtu_m_mop_mul_f32_qa(qacc[0], qa_alpha);
        qacc[1] = __dtu_m_mop_mul_f32_qa(qacc[1], qa_alpha);
        qacc[2] = __dtu_m_mop_mul_f32_qa(qacc[2], qa_alpha);
        qacc[3] = __dtu_m_mop_mul_f32_qa(qacc[3], qa_alpha);
        qacc[4] = __dtu_m_mop_mul_f32_qa(qacc[4], qa_alpha);
        qacc[5] = __dtu_m_mop_mul_f32_qa(qacc[5], qa_alpha);
        qacc[6] = __dtu_m_mop_mul_f32_qa(qacc[6], qa_alpha);
        qacc[7] = __dtu_m_mop_mul_f32_qa(qacc[7], qa_alpha);
        qacc[8] = __dtu_m_mop_mul_f32_qa(qacc[8], qa_alpha);
        qacc[9] = __dtu_m_mop_mul_f32_qa(qacc[9], qa_alpha);
        qacc[10] = __dtu_m_mop_mul_f32_qa(qacc[10], qa_alpha);
        qacc[11] = __dtu_m_mop_mul_f32_qa(qacc[11], qa_alpha);
        qacc[12] = __dtu_m_mop_mul_f32_qa(qacc[12], qa_alpha);
        qacc[13] = __dtu_m_mop_mul_f32_qa(qacc[13], qa_alpha);
        qacc[14] = __dtu_m_mop_mul_f32_qa(qacc[14], qa_alpha);
        qacc[15] = __dtu_m_mop_mul_f32_qa(qacc[15], qa_alpha);
        qacc[16] = __dtu_m_mop_mul_f32_qa(qacc[16], qa_alpha);
        qacc[17] = __dtu_m_mop_mul_f32_qa(qacc[17], qa_alpha);
        qacc[18] = __dtu_m_mop_mul_f32_qa(qacc[18], qa_alpha);
        qacc[19] = __dtu_m_mop_mul_f32_qa(qacc[19], qa_alpha);
        qacc[20] = __dtu_m_mop_mul_f32_qa(qacc[20], qa_alpha);
        qacc[21] = __dtu_m_mop_mul_f32_qa(qacc[21], qa_alpha);
        qacc[22] = __dtu_m_mop_mul_f32_qa(qacc[22], qa_alpha);
        qacc[23] = __dtu_m_mop_mul_f32_qa(qacc[23], qa_alpha);
        qacc[24] = __dtu_m_mop_mul_f32_qa(qacc[24], qa_alpha);
        qacc[25] = __dtu_m_mop_mul_f32_qa(qacc[25], qa_alpha);
        qacc[26] = __dtu_m_mop_mul_f32_qa(qacc[26], qa_alpha);
        qacc[27] = __dtu_m_mop_mul_f32_qa(qacc[27], qa_alpha);
        qacc[28] = __dtu_m_mop_mul_f32_qa(qacc[28], qa_alpha);
        qacc[29] = __dtu_m_mop_mul_f32_qa(qacc[29], qa_alpha);
        qacc[30] = __dtu_m_mop_mul_f32_qa(qacc[30], qa_alpha);
        qacc[31] = __dtu_m_mop_mul_f32_qa(qacc[31], qa_alpha);
        qacc[32] = __dtu_m_mop_mul_f32_qa(qacc[32], qa_alpha);
        qacc[33] = __dtu_m_mop_mul_f32_qa(qacc[33], qa_alpha);
        qacc[34] = __dtu_m_mop_mul_f32_qa(qacc[34], qa_alpha);
        qacc[35] = __dtu_m_mop_mul_f32_qa(qacc[35], qa_alpha);
        qacc[36] = __dtu_m_mop_mul_f32_qa(qacc[36], qa_alpha);
        qacc[37] = __dtu_m_mop_mul_f32_qa(qacc[37], qa_alpha);
        qacc[38] = __dtu_m_mop_mul_f32_qa(qacc[38], qa_alpha);
        qacc[39] = __dtu_m_mop_mul_f32_qa(qacc[39], qa_alpha);
        qacc[40] = __dtu_m_mop_mul_f32_qa(qacc[40], qa_alpha);
        qacc[41] = __dtu_m_mop_mul_f32_qa(qacc[41], qa_alpha);
        qacc[42] = __dtu_m_mop_mul_f32_qa(qacc[42], qa_alpha);
        qacc[43] = __dtu_m_mop_mul_f32_qa(qacc[43], qa_alpha);
        qacc[44] = __dtu_m_mop_mul_f32_qa(qacc[44], qa_alpha);
        qacc[45] = __dtu_m_mop_mul_f32_qa(qacc[45], qa_alpha);
        qacc[46] = __dtu_m_mop_mul_f32_qa(qacc[46], qa_alpha);
        qacc[47] = __dtu_m_mop_mul_f32_qa(qacc[47], qa_alpha);
        qacc[48] = __dtu_m_mop_mul_f32_qa(qacc[48], qa_alpha);
        qacc[49] = __dtu_m_mop_mul_f32_qa(qacc[49], qa_alpha);
        qacc[50] = __dtu_m_mop_mul_f32_qa(qacc[50], qa_alpha);
        qacc[51] = __dtu_m_mop_mul_f32_qa(qacc[51], qa_alpha);
        qacc[52] = __dtu_m_mop_mul_f32_qa(qacc[52], qa_alpha);
        qacc[53] = __dtu_m_mop_mul_f32_qa(qacc[53], qa_alpha);
        qacc[54] = __dtu_m_mop_mul_f32_qa(qacc[54], qa_alpha);
        qacc[55] = __dtu_m_mop_mul_f32_qa(qacc[55], qa_alpha);
        qacc[56] = __dtu_m_mop_mul_f32_qa(qacc[56], qa_alpha);
        qacc[57] = __dtu_m_mop_mul_f32_qa(qacc[57], qa_alpha);
        qacc[58] = __dtu_m_mop_mul_f32_qa(qacc[58], qa_alpha);
        qacc[59] = __dtu_m_mop_mul_f32_qa(qacc[59], qa_alpha);
        qacc[60] = __dtu_m_mop_mul_f32_qa(qacc[60], qa_alpha);
        qacc[61] = __dtu_m_mop_mul_f32_qa(qacc[61], qa_alpha);
        qacc[62] = __dtu_m_mop_mul_f32_qa(qacc[62], qa_alpha);
        qacc[63] = __dtu_m_mop_mul_f32_qa(qacc[63], qa_alpha);

        qacc[0] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[0], qa_beta, qacc[0]);
        qacc[1] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[1], qa_beta, qacc[1]);
        qacc[2] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[2], qa_beta, qacc[2]);
        qacc[3] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[3], qa_beta, qacc[3]);
        qacc[4] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[4], qa_beta, qacc[4]);
        qacc[5] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[5], qa_beta, qacc[5]);
        qacc[6] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[6], qa_beta, qacc[6]);
        qacc[7] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[7], qa_beta, qacc[7]);
        qacc[8] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[8], qa_beta, qacc[8]);
        qacc[9] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[9], qa_beta, qacc[9]);
        qacc[10] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[10], qa_beta, qacc[10]);
        qacc[11] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[11], qa_beta, qacc[11]);
        qacc[12] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[12], qa_beta, qacc[12]);
        qacc[13] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[13], qa_beta, qacc[13]);
        qacc[14] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[14], qa_beta, qacc[14]);
        qacc[15] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[15], qa_beta, qacc[15]);
        qacc[16] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[16], qa_beta, qacc[16]);
        qacc[17] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[17], qa_beta, qacc[17]);
        qacc[18] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[18], qa_beta, qacc[18]);
        qacc[19] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[19], qa_beta, qacc[19]);
        qacc[20] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[20], qa_beta, qacc[20]);
        qacc[21] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[21], qa_beta, qacc[21]);
        qacc[22] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[22], qa_beta, qacc[22]);
        qacc[23] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[23], qa_beta, qacc[23]);
        qacc[24] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[24], qa_beta, qacc[24]);
        qacc[25] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[25], qa_beta, qacc[25]);
        qacc[26] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[26], qa_beta, qacc[26]);
        qacc[27] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[27], qa_beta, qacc[27]);
        qacc[28] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[28], qa_beta, qacc[28]);
        qacc[29] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[29], qa_beta, qacc[29]);
        qacc[30] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[30], qa_beta, qacc[30]);
        qacc[31] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[31], qa_beta, qacc[31]);
        qacc[32] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[32], qa_beta, qacc[32]);
        qacc[33] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[33], qa_beta, qacc[33]);
        qacc[34] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[34], qa_beta, qacc[34]);
        qacc[35] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[35], qa_beta, qacc[35]);
        qacc[36] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[36], qa_beta, qacc[36]);
        qacc[37] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[37], qa_beta, qacc[37]);
        qacc[38] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[38], qa_beta, qacc[38]);
        qacc[39] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[39], qa_beta, qacc[39]);
        qacc[40] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[40], qa_beta, qacc[40]);
        qacc[41] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[41], qa_beta, qacc[41]);
        qacc[42] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[42], qa_beta, qacc[42]);
        qacc[43] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[43], qa_beta, qacc[43]);
        qacc[44] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[44], qa_beta, qacc[44]);
        qacc[45] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[45], qa_beta, qacc[45]);
        qacc[46] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[46], qa_beta, qacc[46]);
        qacc[47] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[47], qa_beta, qacc[47]);
        qacc[48] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[48], qa_beta, qacc[48]);
        qacc[49] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[49], qa_beta, qacc[49]);
        qacc[50] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[50], qa_beta, qacc[50]);
        qacc[51] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[51], qa_beta, qacc[51]);
        qacc[52] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[52], qa_beta, qacc[52]);
        qacc[53] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[53], qa_beta, qacc[53]);
        qacc[54] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[54], qa_beta, qacc[54]);
        qacc[55] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[55], qa_beta, qacc[55]);
        qacc[56] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[56], qa_beta, qacc[56]);
        qacc[57] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[57], qa_beta, qacc[57]);
        qacc[58] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[58], qa_beta, qacc[58]);
        qacc[59] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[59], qa_beta, qacc[59]);
        qacc[60] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[60], qa_beta, qacc[60]);
        qacc[61] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[61], qa_beta, qacc[61]);
        qacc[62] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[62], qa_beta, qacc[62]);
        qacc[63] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[63], qa_beta, qacc[63]);
        // add bias
        if (bias_en == 1) {
          qacc[0] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[0]);
          qacc[1] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[1]);
          qacc[2] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[2]);
          qacc[3] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[3]);
          qacc[4] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[4]);
          qacc[5] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[5]);
          qacc[6] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[6]);
          qacc[7] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[7]);
          qacc[8] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[8]);
          qacc[9] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[9]);
          qacc[10] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[10]);
          qacc[11] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[11]);
          qacc[12] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[12]);
          qacc[13] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[13]);
          qacc[14] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[14]);
          qacc[15] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[15]);
          qacc[16] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[16]);
          qacc[17] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[17]);
          qacc[18] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[18]);
          qacc[19] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[19]);
          qacc[20] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[20]);
          qacc[21] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[21]);
          qacc[22] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[22]);
          qacc[23] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[23]);
          qacc[24] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[24]);
          qacc[25] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[25]);
          qacc[26] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[26]);
          qacc[27] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[27]);
          qacc[28] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[28]);
          qacc[29] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[29]);
          qacc[30] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[30]);
          qacc[31] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[31]);
          qacc[32] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[32]);
          qacc[33] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[33]);
          qacc[34] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[34]);
          qacc[35] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[35]);
          qacc[36] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[36]);
          qacc[37] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[37]);
          qacc[38] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[38]);
          qacc[39] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[39]);
          qacc[40] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[40]);
          qacc[41] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[41]);
          qacc[42] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[42]);
          qacc[43] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[43]);
          qacc[44] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[44]);
          qacc[45] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[45]);
          qacc[46] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[46]);
          qacc[47] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[47]);
          qacc[48] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[48]);
          qacc[49] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[49]);
          qacc[50] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[50]);
          qacc[51] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[51]);
          qacc[52] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[52]);
          qacc[53] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[53]);
          qacc[54] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[54]);
          qacc[55] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[55]);
          qacc[56] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[56]);
          qacc[57] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[57]);
          qacc[58] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[58]);
          qacc[59] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[59]);
          qacc[60] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[60]);
          qacc[61] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[61]);
          qacc[62] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[62]);
          qacc[63] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[63]);
        }

        c_dacc[0] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[0]);
        c_dacc[1] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[1]);
        c_dacc[2] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[2]);
        c_dacc[3] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[3]);
        c_dacc[4] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[4]);
        c_dacc[5] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[5]);
        c_dacc[6] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[6]);
        c_dacc[7] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[7]);
        c_dacc[8] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[8]);
        c_dacc[9] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[9]);
        c_dacc[10] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[10]);
        c_dacc[11] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[11]);
        c_dacc[12] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[12]);
        c_dacc[13] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[13]);
        c_dacc[14] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[14]);
        c_dacc[15] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[15]);
        c_dacc[16] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[16]);
        c_dacc[17] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[17]);
        c_dacc[18] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[18]);
        c_dacc[19] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[19]);
        c_dacc[20] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[20]);
        c_dacc[21] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[21]);
        c_dacc[22] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[22]);
        c_dacc[23] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[23]);
        c_dacc[24] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[24]);
        c_dacc[25] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[25]);
        c_dacc[26] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[26]);
        c_dacc[27] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[27]);
        c_dacc[28] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[28]);
        c_dacc[29] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[29]);
        c_dacc[30] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[30]);
        c_dacc[31] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[31]);
        c_dacc[32] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[32]);
        c_dacc[33] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[33]);
        c_dacc[34] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[34]);
        c_dacc[35] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[35]);
        c_dacc[36] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[36]);
        c_dacc[37] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[37]);
        c_dacc[38] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[38]);
        c_dacc[39] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[39]);
        c_dacc[40] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[40]);
        c_dacc[41] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[41]);
        c_dacc[42] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[42]);
        c_dacc[43] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[43]);
        c_dacc[44] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[44]);
        c_dacc[45] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[45]);
        c_dacc[46] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[46]);
        c_dacc[47] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[47]);
        c_dacc[48] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[48]);
        c_dacc[49] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[49]);
        c_dacc[50] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[50]);
        c_dacc[51] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[51]);
        c_dacc[52] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[52]);
        c_dacc[53] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[53]);
        c_dacc[54] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[54]);
        c_dacc[55] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[55]);
        c_dacc[56] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[56]);
        c_dacc[57] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[57]);
        c_dacc[58] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[58]);
        c_dacc[59] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[59]);
        c_dacc[60] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[60]);
        c_dacc[61] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[61]);
        c_dacc[62] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[62]);
        c_dacc[63] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[63]);
        __dtu_v_tvstda_f16_dual(c_dacc[0], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[1], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[2], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[3], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[4], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[5], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[6], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[7], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[8], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[9], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[10], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[11], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[12], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[13], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[14], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[15], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[16], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[17], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[18], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[19], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[20], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[21], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[22], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[23], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[24], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[25], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[26], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[27], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[28], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[29], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[30], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[31], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[32], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[33], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[34], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[35], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[36], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[37], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[38], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[39], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[40], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[41], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[42], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[43], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[44], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[45], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[46], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[47], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[48], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[49], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[50], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[51], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[52], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[53], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[54], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[55], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[56], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[57], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[58], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[59], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[60], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[61], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[62], ot_base, ot_off0);
        __dtu_v_tvstda_f16_dual(c_dacc[63], ot_base, ot_off1);

        vab_shift += 512;
        __dtu_c_movsr2vab_lv_s(vab_shift);
        __dtu_c_movsr2vab_lv_d(vab_shift);
        __dtu_c_movsr2vab_m_s1(vab_shift);
        __dtu_c_movsr2vab_m_d(vab_shift);
      }
      bs_dacc = __dtu_l_tvldqa_f16_da(biast_base, biast_off1);
      c_dacc[0] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[1] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[2] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[3] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[4] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[5] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[6] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[7] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[8] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[9] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[10] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[11] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[12] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[13] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[14] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[15] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[16] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[17] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[18] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[19] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[20] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[21] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[22] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[23] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[24] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[25] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[26] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[27] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[28] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[29] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[30] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[31] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[32] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[33] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[34] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[35] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[36] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[37] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[38] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[39] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[40] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[41] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[42] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[43] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[44] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[45] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[46] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[47] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[48] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[49] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[50] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[51] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[52] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[53] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[54] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[55] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[56] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[57] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[58] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[59] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[60] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[61] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[62] = __dtu_l_tvldqa_f16_da(bt_base, bt_off0);
      c_dacc[63] = __dtu_l_tvldqa_f16_da(bt_base, bt_off2);

      __dtu_c_movsr2vab_m_s2(0);
      qacc[0] = __dtu_m_mop_mul_f32_qa(qacc[0], qa_alpha);
      qacc[1] = __dtu_m_mop_mul_f32_qa(qacc[1], qa_alpha);
      qacc[2] = __dtu_m_mop_mul_f32_qa(qacc[2], qa_alpha);
      qacc[3] = __dtu_m_mop_mul_f32_qa(qacc[3], qa_alpha);
      qacc[4] = __dtu_m_mop_mul_f32_qa(qacc[4], qa_alpha);
      qacc[5] = __dtu_m_mop_mul_f32_qa(qacc[5], qa_alpha);
      qacc[6] = __dtu_m_mop_mul_f32_qa(qacc[6], qa_alpha);
      qacc[7] = __dtu_m_mop_mul_f32_qa(qacc[7], qa_alpha);
      qacc[8] = __dtu_m_mop_mul_f32_qa(qacc[8], qa_alpha);
      qacc[9] = __dtu_m_mop_mul_f32_qa(qacc[9], qa_alpha);
      qacc[10] = __dtu_m_mop_mul_f32_qa(qacc[10], qa_alpha);
      qacc[11] = __dtu_m_mop_mul_f32_qa(qacc[11], qa_alpha);
      qacc[12] = __dtu_m_mop_mul_f32_qa(qacc[12], qa_alpha);
      qacc[13] = __dtu_m_mop_mul_f32_qa(qacc[13], qa_alpha);
      qacc[14] = __dtu_m_mop_mul_f32_qa(qacc[14], qa_alpha);
      qacc[15] = __dtu_m_mop_mul_f32_qa(qacc[15], qa_alpha);
      qacc[16] = __dtu_m_mop_mul_f32_qa(qacc[16], qa_alpha);
      qacc[17] = __dtu_m_mop_mul_f32_qa(qacc[17], qa_alpha);
      qacc[18] = __dtu_m_mop_mul_f32_qa(qacc[18], qa_alpha);
      qacc[19] = __dtu_m_mop_mul_f32_qa(qacc[19], qa_alpha);
      qacc[20] = __dtu_m_mop_mul_f32_qa(qacc[20], qa_alpha);
      qacc[21] = __dtu_m_mop_mul_f32_qa(qacc[21], qa_alpha);
      qacc[22] = __dtu_m_mop_mul_f32_qa(qacc[22], qa_alpha);
      qacc[23] = __dtu_m_mop_mul_f32_qa(qacc[23], qa_alpha);
      qacc[24] = __dtu_m_mop_mul_f32_qa(qacc[24], qa_alpha);
      qacc[25] = __dtu_m_mop_mul_f32_qa(qacc[25], qa_alpha);
      qacc[26] = __dtu_m_mop_mul_f32_qa(qacc[26], qa_alpha);
      qacc[27] = __dtu_m_mop_mul_f32_qa(qacc[27], qa_alpha);
      qacc[28] = __dtu_m_mop_mul_f32_qa(qacc[28], qa_alpha);
      qacc[29] = __dtu_m_mop_mul_f32_qa(qacc[29], qa_alpha);
      qacc[30] = __dtu_m_mop_mul_f32_qa(qacc[30], qa_alpha);
      qacc[31] = __dtu_m_mop_mul_f32_qa(qacc[31], qa_alpha);
      qacc[32] = __dtu_m_mop_mul_f32_qa(qacc[32], qa_alpha);
      qacc[33] = __dtu_m_mop_mul_f32_qa(qacc[33], qa_alpha);
      qacc[34] = __dtu_m_mop_mul_f32_qa(qacc[34], qa_alpha);
      qacc[35] = __dtu_m_mop_mul_f32_qa(qacc[35], qa_alpha);
      qacc[36] = __dtu_m_mop_mul_f32_qa(qacc[36], qa_alpha);
      qacc[37] = __dtu_m_mop_mul_f32_qa(qacc[37], qa_alpha);
      qacc[38] = __dtu_m_mop_mul_f32_qa(qacc[38], qa_alpha);
      qacc[39] = __dtu_m_mop_mul_f32_qa(qacc[39], qa_alpha);
      qacc[40] = __dtu_m_mop_mul_f32_qa(qacc[40], qa_alpha);
      qacc[41] = __dtu_m_mop_mul_f32_qa(qacc[41], qa_alpha);
      qacc[42] = __dtu_m_mop_mul_f32_qa(qacc[42], qa_alpha);
      qacc[43] = __dtu_m_mop_mul_f32_qa(qacc[43], qa_alpha);
      qacc[44] = __dtu_m_mop_mul_f32_qa(qacc[44], qa_alpha);
      qacc[45] = __dtu_m_mop_mul_f32_qa(qacc[45], qa_alpha);
      qacc[46] = __dtu_m_mop_mul_f32_qa(qacc[46], qa_alpha);
      qacc[47] = __dtu_m_mop_mul_f32_qa(qacc[47], qa_alpha);
      qacc[48] = __dtu_m_mop_mul_f32_qa(qacc[48], qa_alpha);
      qacc[49] = __dtu_m_mop_mul_f32_qa(qacc[49], qa_alpha);
      qacc[50] = __dtu_m_mop_mul_f32_qa(qacc[50], qa_alpha);
      qacc[51] = __dtu_m_mop_mul_f32_qa(qacc[51], qa_alpha);
      qacc[52] = __dtu_m_mop_mul_f32_qa(qacc[52], qa_alpha);
      qacc[53] = __dtu_m_mop_mul_f32_qa(qacc[53], qa_alpha);
      qacc[54] = __dtu_m_mop_mul_f32_qa(qacc[54], qa_alpha);
      qacc[55] = __dtu_m_mop_mul_f32_qa(qacc[55], qa_alpha);
      qacc[56] = __dtu_m_mop_mul_f32_qa(qacc[56], qa_alpha);
      qacc[57] = __dtu_m_mop_mul_f32_qa(qacc[57], qa_alpha);
      qacc[58] = __dtu_m_mop_mul_f32_qa(qacc[58], qa_alpha);
      qacc[59] = __dtu_m_mop_mul_f32_qa(qacc[59], qa_alpha);
      qacc[60] = __dtu_m_mop_mul_f32_qa(qacc[60], qa_alpha);
      qacc[61] = __dtu_m_mop_mul_f32_qa(qacc[61], qa_alpha);
      qacc[62] = __dtu_m_mop_mul_f32_qa(qacc[62], qa_alpha);
      qacc[63] = __dtu_m_mop_mul_f32_qa(qacc[63], qa_alpha);

      qacc[0] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[0], qa_beta, qacc[0]);
      qacc[1] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[1], qa_beta, qacc[1]);
      qacc[2] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[2], qa_beta, qacc[2]);
      qacc[3] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[3], qa_beta, qacc[3]);
      qacc[4] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[4], qa_beta, qacc[4]);
      qacc[5] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[5], qa_beta, qacc[5]);
      qacc[6] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[6], qa_beta, qacc[6]);
      qacc[7] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[7], qa_beta, qacc[7]);
      qacc[8] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[8], qa_beta, qacc[8]);
      qacc[9] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[9], qa_beta, qacc[9]);
      qacc[10] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[10], qa_beta, qacc[10]);
      qacc[11] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[11], qa_beta, qacc[11]);
      qacc[12] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[12], qa_beta, qacc[12]);
      qacc[13] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[13], qa_beta, qacc[13]);
      qacc[14] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[14], qa_beta, qacc[14]);
      qacc[15] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[15], qa_beta, qacc[15]);
      qacc[16] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[16], qa_beta, qacc[16]);
      qacc[17] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[17], qa_beta, qacc[17]);
      qacc[18] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[18], qa_beta, qacc[18]);
      qacc[19] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[19], qa_beta, qacc[19]);
      qacc[20] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[20], qa_beta, qacc[20]);
      qacc[21] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[21], qa_beta, qacc[21]);
      qacc[22] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[22], qa_beta, qacc[22]);
      qacc[23] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[23], qa_beta, qacc[23]);
      qacc[24] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[24], qa_beta, qacc[24]);
      qacc[25] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[25], qa_beta, qacc[25]);
      qacc[26] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[26], qa_beta, qacc[26]);
      qacc[27] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[27], qa_beta, qacc[27]);
      qacc[28] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[28], qa_beta, qacc[28]);
      qacc[29] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[29], qa_beta, qacc[29]);
      qacc[30] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[30], qa_beta, qacc[30]);
      qacc[31] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[31], qa_beta, qacc[31]);
      qacc[32] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[32], qa_beta, qacc[32]);
      qacc[33] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[33], qa_beta, qacc[33]);
      qacc[34] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[34], qa_beta, qacc[34]);
      qacc[35] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[35], qa_beta, qacc[35]);
      qacc[36] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[36], qa_beta, qacc[36]);
      qacc[37] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[37], qa_beta, qacc[37]);
      qacc[38] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[38], qa_beta, qacc[38]);
      qacc[39] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[39], qa_beta, qacc[39]);
      qacc[40] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[40], qa_beta, qacc[40]);
      qacc[41] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[41], qa_beta, qacc[41]);
      qacc[42] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[42], qa_beta, qacc[42]);
      qacc[43] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[43], qa_beta, qacc[43]);
      qacc[44] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[44], qa_beta, qacc[44]);
      qacc[45] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[45], qa_beta, qacc[45]);
      qacc[46] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[46], qa_beta, qacc[46]);
      qacc[47] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[47], qa_beta, qacc[47]);
      qacc[48] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[48], qa_beta, qacc[48]);
      qacc[49] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[49], qa_beta, qacc[49]);
      qacc[50] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[50], qa_beta, qacc[50]);
      qacc[51] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[51], qa_beta, qacc[51]);
      qacc[52] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[52], qa_beta, qacc[52]);
      qacc[53] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[53], qa_beta, qacc[53]);
      qacc[54] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[54], qa_beta, qacc[54]);
      qacc[55] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[55], qa_beta, qacc[55]);
      qacc[56] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[56], qa_beta, qacc[56]);
      qacc[57] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[57], qa_beta, qacc[57]);
      qacc[58] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[58], qa_beta, qacc[58]);
      qacc[59] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[59], qa_beta, qacc[59]);
      qacc[60] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[60], qa_beta, qacc[60]);
      qacc[61] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[61], qa_beta, qacc[61]);
      qacc[62] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[62], qa_beta, qacc[62]);
      qacc[63] = __dtu_m_mop_mac_f32mix_f16_da(c_dacc[63], qa_beta, qacc[63]);
      // add bias
      if (bias_en == 1) {
        qacc[0] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[0]);
        qacc[1] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[1]);
        qacc[2] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[2]);
        qacc[3] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[3]);
        qacc[4] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[4]);
        qacc[5] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[5]);
        qacc[6] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[6]);
        qacc[7] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[7]);
        qacc[8] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[8]);
        qacc[9] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[9]);
        qacc[10] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[10]);
        qacc[11] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[11]);
        qacc[12] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[12]);
        qacc[13] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[13]);
        qacc[14] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[14]);
        qacc[15] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[15]);
        qacc[16] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[16]);
        qacc[17] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[17]);
        qacc[18] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[18]);
        qacc[19] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[19]);
        qacc[20] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[20]);
        qacc[21] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[21]);
        qacc[22] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[22]);
        qacc[23] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[23]);
        qacc[24] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[24]);
        qacc[25] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[25]);
        qacc[26] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[26]);
        qacc[27] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[27]);
        qacc[28] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[28]);
        qacc[29] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[29]);
        qacc[30] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[30]);
        qacc[31] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[31]);
        qacc[32] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[32]);
        qacc[33] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[33]);
        qacc[34] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[34]);
        qacc[35] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[35]);
        qacc[36] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[36]);
        qacc[37] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[37]);
        qacc[38] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[38]);
        qacc[39] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[39]);
        qacc[40] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[40]);
        qacc[41] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[41]);
        qacc[42] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[42]);
        qacc[43] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[43]);
        qacc[44] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[44]);
        qacc[45] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[45]);
        qacc[46] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[46]);
        qacc[47] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[47]);
        qacc[48] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[48]);
        qacc[49] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[49]);
        qacc[50] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[50]);
        qacc[51] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[51]);
        qacc[52] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[52]);
        qacc[53] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[53]);
        qacc[54] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[54]);
        qacc[55] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[55]);
        qacc[56] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[56]);
        qacc[57] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[57]);
        qacc[58] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[58]);
        qacc[59] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[59]);
        qacc[60] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[60]);
        qacc[61] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[61]);
        qacc[62] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[62]);
        qacc[63] = __dtu_m_mop_mac_f32mix_f16_da(bs_dacc, qa_bias, qacc[63]);
      }

      c_dacc[0] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[0]);
      c_dacc[1] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[1]);
      c_dacc[2] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[2]);
      c_dacc[3] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[3]);
      c_dacc[4] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[4]);
      c_dacc[5] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[5]);
      c_dacc[6] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[6]);
      c_dacc[7] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[7]);
      c_dacc[8] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[8]);
      c_dacc[9] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[9]);
      c_dacc[10] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[10]);
      c_dacc[11] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[11]);
      c_dacc[12] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[12]);
      c_dacc[13] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[13]);
      c_dacc[14] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[14]);
      c_dacc[15] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[15]);
      c_dacc[16] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[16]);
      c_dacc[17] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[17]);
      c_dacc[18] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[18]);
      c_dacc[19] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[19]);
      c_dacc[20] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[20]);
      c_dacc[21] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[21]);
      c_dacc[22] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[22]);
      c_dacc[23] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[23]);
      c_dacc[24] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[24]);
      c_dacc[25] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[25]);
      c_dacc[26] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[26]);
      c_dacc[27] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[27]);
      c_dacc[28] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[28]);
      c_dacc[29] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[29]);
      c_dacc[30] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[30]);
      c_dacc[31] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[31]);
      c_dacc[32] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[32]);
      c_dacc[33] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[33]);
      c_dacc[34] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[34]);
      c_dacc[35] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[35]);
      c_dacc[36] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[36]);
      c_dacc[37] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[37]);
      c_dacc[38] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[38]);
      c_dacc[39] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[39]);
      c_dacc[40] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[40]);
      c_dacc[41] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[41]);
      c_dacc[42] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[42]);
      c_dacc[43] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[43]);
      c_dacc[44] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[44]);
      c_dacc[45] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[45]);
      c_dacc[46] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[46]);
      c_dacc[47] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[47]);
      c_dacc[48] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[48]);
      c_dacc[49] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[49]);
      c_dacc[50] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[50]);
      c_dacc[51] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[51]);
      c_dacc[52] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[52]);
      c_dacc[53] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[53]);
      c_dacc[54] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[54]);
      c_dacc[55] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[55]);
      c_dacc[56] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[56]);
      c_dacc[57] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[57]);
      c_dacc[58] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[58]);
      c_dacc[59] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[59]);
      c_dacc[60] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[60]);
      c_dacc[61] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[61]);
      c_dacc[62] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[62]);
      c_dacc[63] = __dtu_m_mop_cvt_qa_rne_clamp_f32_f16(qacc[63]);

      __dtu_v_tvstda_f16_dual(c_dacc[0], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[1], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[2], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[3], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[4], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[5], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[6], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[7], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[8], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[9], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[10], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[11], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[12], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[13], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[14], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[15], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[16], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[17], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[18], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[19], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[20], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[21], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[22], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[23], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[24], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[25], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[26], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[27], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[28], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[29], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[30], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[31], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[32], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[33], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[34], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[35], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[36], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[37], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[38], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[39], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[40], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[41], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[42], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[43], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[44], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[45], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[46], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[47], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[48], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[49], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[50], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[51], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[52], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[53], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[54], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[55], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[56], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[57], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[58], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[59], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[60], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[61], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[62], ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(c_dacc[63], ot_base, ot_off2);
      vab_shift += 512;
      __dtu_c_movsr2vab_lv_s(vab_shift);
      __dtu_c_movsr2vab_lv_d(vab_shift);
      __dtu_c_movsr2vab_m_s1(vab_shift);
      __dtu_c_movsr2vab_m_d(vab_shift);
    }
  }

#endif
}




__attribute__((device)) extern "C" void c_func_sgemm_general(
    int a_addr, int b_addr, int c_addr, int M, int N, int K, int nacc_flag,
    int store_flag, int alpha_enable, int beta_enable, float alpha,
    float beta, float addmm_beta, int bias_en, int bias_addr, int cur_n) {

#if __GCU_ARCH__ >= 300
  v16f32 vr0, vr1, vr2, vr3, vr4, vr5, vr6, vr7, vr8, vr9, vr10, vr11, vr12,
      vr13, vr14, vr15, vr16, vr17, vr18, vr19, vr20, vr21, vr22, vr23, vr24,
      vr25, vr26, vr27, vr28, vr29, vr30, vr31;

  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_lv_d(0);
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_s2(0);
  __dtu_c_movsr2vab_m_d(0);

  va16f32x2 dacc_arr[32];
  va16f32x2 c_dacc_arr[32];

  va16f32x2 dacc_alpha, dacc_beta, dacc_bias, da_bs;
  va16f32 vacc_alpha0, vacc_alpha1, vacc_beta0, vacc_beta1;

  v16f32 vr_scale = __dtu_s_movr2vr_dup_f32(alpha);
  vacc_alpha0 = __dtu_l_movvr2va(vr_scale);
  vacc_alpha1 = __dtu_l_movvr2va(vr_scale);
  dacc_alpha = __dtu_insertva2da(vacc_alpha0, vacc_alpha1);

  vr_scale = __dtu_s_movr2vr_dup_f32(beta);
  vacc_beta0 = __dtu_l_movvr2va(vr_scale);
  vacc_beta1 = __dtu_l_movvr2va(vr_scale);
  dacc_beta = __dtu_insertva2da(vacc_beta0, vacc_beta1);

  if (bias_en == 0) {
    vr_scale = __dtu_s_movr2vr_dup_f32(0.0f);
  } else {
    vr_scale = __dtu_s_movr2vr_dup_f32(addmm_beta);
  }
  vacc_beta0 = __dtu_l_movvr2va(vr_scale);
  vacc_beta1 = __dtu_l_movvr2va(vr_scale);
  da_bs =
      __dtu_insertva2da(vacc_beta0, vacc_beta1);

  smr_t smr0;
  smr_t smr1;

  auto k_unit = K >> 4;
  auto n_unit = N >> 5;

  // vpt parallel in rhs
  int lhs_tar_addr = a_addr >> 6;
  int rhs_tar_addr = b_addr >> 7;
  int bias_or_out_tar_addr = c_addr >> 7;
  int te_bias_tar_addr = (bias_addr + cur_n * 4) >> 7;

  tar_t lhs_tar_base = __dtu_c_movsr2targ(TAR_ADDR_WARP(lhs_tar_addr, 0));
  int lhs_off0 = TAR_OFF_WARP(k_unit, k_unit);
  tar_t lhs_tar_off0 = __dtu_c_movsr2tari(lhs_off0, lhs_tar_base);
  int lhs_off1 = TAR_OFF_WARP(1 - 31 * k_unit, 1 - 31 * k_unit);  // next k
  tar_t lhs_tar_off1 = __dtu_c_movsr2tari(lhs_off1, lhs_tar_base);
  int lhs_off2 =
      TAR_OFF_WARP(1 - 32 * k_unit, 1 - 32 * k_unit);  //  end k new n
  tar_t lhs_tar_off2 = __dtu_c_movsr2tari(lhs_off2, lhs_tar_base);
  int lhs_off3 = TAR_OFF_WARP(1, 1);  // end k end n new m
  tar_t lhs_tar_off3 = __dtu_c_movsr2tari(lhs_off3, lhs_tar_base);

  tar_t rhs_tar_base =
      __dtu_c_movsr2targ((rhs_tar_addr) | ((rhs_tar_addr) + 1) << 16);
  int rhs_off0 = TAR_OFF_WARP(n_unit, n_unit);
  tar_t rhs_tar_off0 = __dtu_c_movsr2tari(rhs_off0, rhs_tar_base);
  int rhs_off1 = TAR_OFF_WARP(2 + n_unit - K * n_unit, 2 + n_unit - K * n_unit);
  tar_t rhs_tar_off1 = __dtu_c_movsr2tari(rhs_off1, rhs_tar_base);
  int rhs_off2 = TAR_OFF_WARP(2 - K * n_unit, 2 - K * n_unit);
  tar_t rhs_tar_off2 = __dtu_c_movsr2tari(rhs_off2, rhs_tar_base);

  tar_t bias_tar_base = __dtu_c_movsr2targ((bias_or_out_tar_addr) |
                                           ((bias_or_out_tar_addr) + 1) << 16);
  int bias_off0 = TAR_OFF_WARP(n_unit, n_unit);
  tar_t bias_tar_off0 = __dtu_c_movsr2tari(bias_off0, bias_tar_base);
  int bias_off1 = TAR_OFF_WARP(2 - 31 * n_unit, 2 - 31 * n_unit);
  tar_t bias_tar_off1 = __dtu_c_movsr2tari(bias_off1, bias_tar_base);
  int bias_off2 = TAR_OFF_WARP(2, 2);
  tar_t bias_tar_off2 = __dtu_c_movsr2tari(bias_off2, bias_tar_base);

  tar_t out_tar_base = __dtu_c_movsr2targ((bias_or_out_tar_addr) |
                                          ((bias_or_out_tar_addr) + 1) << 16);
  int out_off0 = TAR_OFF_WARP(n_unit, n_unit);
  tar_t out_tar_off0 = __dtu_c_movsr2tari(out_off0, out_tar_base);
  int out_off1 = TAR_OFF_WARP(2 - 31 * n_unit, 2 - 31 * n_unit);
  tar_t out_tar_off1 = __dtu_c_movsr2tari(out_off1, out_tar_base);
  int out_off2 = TAR_OFF_WARP(2, 2);
  tar_t out_tar_off2 = __dtu_c_movsr2tari(out_off2, out_tar_base);

  tar_t te_bias_tar_base = __dtu_c_movsr2targ((te_bias_tar_addr) |
                                           ((te_bias_tar_addr) + 1) << 16);
  int te_bias_off0 = TAR_OFF_WARP(2, 2);
  tar_t te_bias_tar_off0 = __dtu_c_movsr2tari(te_bias_off0, te_bias_tar_base);
  int te_bias_off1 = TAR_OFF_WARP(2 - n_unit, 2 - n_unit);
  tar_t te_bias_tar_off1 = __dtu_c_movsr2tari(te_bias_off1, te_bias_tar_base);

  smr0 =
      __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0, 0);
  smr0 =
      __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0, 1);
  smr0 =
      __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0, 2);
  smr0 =
      __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0, 3);
  smr0 =
      __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0, 4);
  smr0 =
      __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0, 5);
  smr0 =
      __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0, 6);
  smr0 =
      __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0, 7);
  smr0 =
      __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0, 8);
  smr0 =
      __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0, 9);
  smr0 =
      __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0, 10);
  smr0 =
      __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0, 11);
  smr0 =
      __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0, 12);
  smr0 =
      __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0, 13);
  smr0 =
      __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0, 14);
  smr0 =
      __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0, 15);

  vr0 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
  vr1 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
  vr2 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
  vr3 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
  vr4 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
  vr5 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
  vr6 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
  vr7 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
  vr8 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
  vr9 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
  vr10 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
  vr11 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
  vr12 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
  vr13 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
  vr14 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
  vr15 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);

  int naccovr = 0x10001;
  if (nacc_flag ^ 1) {
    naccovr = 0x1;
  }

  __dtu_c_movsr2naccovr(naccovr);

  __dtu_c_movsr2vab_m_s2(0);

  int vab_shift = 0;

// [32, 32] * [32, 64] = [32, 64]
#pragma clang loop unroll(full)
  for (int m = 0; m < M; m += 32) {
    for (int n = 0; n < N - 64; n += 64) {  // VPT PARA DIM
      __dtu_c_movsr2naccovr(naccovr);
      for (int k = 0; k < K - 32; k += 32) {
        // m0k0
        dacc_arr[0] = __dtu_m_vmm2_mode18_f32(dacc_arr[0], vr0, smr0);
        dacc_arr[1] = __dtu_m_vmm2_mode18_f32(dacc_arr[1], vr1, smr0);
        dacc_arr[2] = __dtu_m_vmm2_mode18_f32(dacc_arr[2], vr2, smr0);
        dacc_arr[3] = __dtu_m_vmm2_mode18_f32(dacc_arr[3], vr3, smr0);
        dacc_arr[4] = __dtu_m_vmm2_mode18_f32(dacc_arr[4], vr4, smr0);
        dacc_arr[5] = __dtu_m_vmm2_mode18_f32(dacc_arr[5], vr5, smr0);
        dacc_arr[6] = __dtu_m_vmm2_mode18_f32(dacc_arr[6], vr6, smr0);
        dacc_arr[7] = __dtu_m_vmm2_mode18_f32(dacc_arr[7], vr7, smr0);
        dacc_arr[8] = __dtu_m_vmm2_mode18_f32(dacc_arr[8], vr8, smr0);
        dacc_arr[9] = __dtu_m_vmm2_mode18_f32(dacc_arr[9], vr9, smr0);
        dacc_arr[10] = __dtu_m_vmm2_mode18_f32(dacc_arr[10], vr10, smr0);
        dacc_arr[11] = __dtu_m_vmm2_mode18_f32(dacc_arr[11], vr11, smr0);
        dacc_arr[12] = __dtu_m_vmm2_mode18_f32(dacc_arr[12], vr12, smr0);
        dacc_arr[13] = __dtu_m_vmm2_mode18_f32(dacc_arr[13], vr13, smr0);
        dacc_arr[14] = __dtu_m_vmm2_mode18_f32(dacc_arr[14], vr14, smr0);
        dacc_arr[15] = __dtu_m_vmm2_mode18_f32(dacc_arr[15], vr15, smr0);

        vr0 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr1 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr2 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr3 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr4 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr5 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr6 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr7 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr8 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr9 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr10 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr11 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr12 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr13 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr14 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr15 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off1);

        smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                   rhs_tar_off0, 0);
        smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                   rhs_tar_off0, 1);
        smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                   rhs_tar_off0, 2);
        smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                   rhs_tar_off0, 3);
        smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                   rhs_tar_off0, 4);
        smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                   rhs_tar_off0, 5);
        smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                   rhs_tar_off0, 6);
        smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                   rhs_tar_off0, 7);
        smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                   rhs_tar_off0, 8);
        smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                   rhs_tar_off0, 9);
        smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                   rhs_tar_off0, 10);
        smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                   rhs_tar_off0, 11);
        smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                   rhs_tar_off0, 12);
        smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                   rhs_tar_off0, 13);
        smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                   rhs_tar_off0, 14);
        smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                   rhs_tar_off0, 15);

        // m1k0
        dacc_arr[16] = __dtu_m_vmm2_mode18_f32(dacc_arr[16], vr0, smr0);
        dacc_arr[17] = __dtu_m_vmm2_mode18_f32(dacc_arr[17], vr1, smr0);
        dacc_arr[18] = __dtu_m_vmm2_mode18_f32(dacc_arr[18], vr2, smr0);
        dacc_arr[19] = __dtu_m_vmm2_mode18_f32(dacc_arr[19], vr3, smr0);
        dacc_arr[20] = __dtu_m_vmm2_mode18_f32(dacc_arr[20], vr4, smr0);
        dacc_arr[21] = __dtu_m_vmm2_mode18_f32(dacc_arr[21], vr5, smr0);
        dacc_arr[22] = __dtu_m_vmm2_mode18_f32(dacc_arr[22], vr6, smr0);
        dacc_arr[23] = __dtu_m_vmm2_mode18_f32(dacc_arr[23], vr7, smr0);
        dacc_arr[24] = __dtu_m_vmm2_mode18_f32(dacc_arr[24], vr8, smr0);
        dacc_arr[25] = __dtu_m_vmm2_mode18_f32(dacc_arr[25], vr9, smr0);
        dacc_arr[26] = __dtu_m_vmm2_mode18_f32(dacc_arr[26], vr10, smr0);
        dacc_arr[27] = __dtu_m_vmm2_mode18_f32(dacc_arr[27], vr11, smr0);
        dacc_arr[28] = __dtu_m_vmm2_mode18_f32(dacc_arr[28], vr12, smr0);
        dacc_arr[29] = __dtu_m_vmm2_mode18_f32(dacc_arr[29], vr13, smr0);
        dacc_arr[30] = __dtu_m_vmm2_mode18_f32(dacc_arr[30], vr14, smr0);
        dacc_arr[31] = __dtu_m_vmm2_mode18_f32(dacc_arr[31], vr15, smr0);

        vr0 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr1 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr2 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr3 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr4 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr5 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr6 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr7 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr8 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr9 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr10 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr11 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr12 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr13 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr14 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr15 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);

        __dtu_c_movsr2naccovr(0x1);

        // m0k1
        dacc_arr[0] = __dtu_m_vmm2_mode18_f32(dacc_arr[0], vr0, smr1);
        dacc_arr[1] = __dtu_m_vmm2_mode18_f32(dacc_arr[1], vr1, smr1);
        dacc_arr[2] = __dtu_m_vmm2_mode18_f32(dacc_arr[2], vr2, smr1);
        dacc_arr[3] = __dtu_m_vmm2_mode18_f32(dacc_arr[3], vr3, smr1);
        dacc_arr[4] = __dtu_m_vmm2_mode18_f32(dacc_arr[4], vr4, smr1);
        dacc_arr[5] = __dtu_m_vmm2_mode18_f32(dacc_arr[5], vr5, smr1);
        dacc_arr[6] = __dtu_m_vmm2_mode18_f32(dacc_arr[6], vr6, smr1);
        dacc_arr[7] = __dtu_m_vmm2_mode18_f32(dacc_arr[7], vr7, smr1);
        dacc_arr[8] = __dtu_m_vmm2_mode18_f32(dacc_arr[8], vr8, smr1);
        dacc_arr[9] = __dtu_m_vmm2_mode18_f32(dacc_arr[9], vr9, smr1);
        dacc_arr[10] = __dtu_m_vmm2_mode18_f32(dacc_arr[10], vr10, smr1);
        dacc_arr[11] = __dtu_m_vmm2_mode18_f32(dacc_arr[11], vr11, smr1);
        dacc_arr[12] = __dtu_m_vmm2_mode18_f32(dacc_arr[12], vr12, smr1);
        dacc_arr[13] = __dtu_m_vmm2_mode18_f32(dacc_arr[13], vr13, smr1);
        dacc_arr[14] = __dtu_m_vmm2_mode18_f32(dacc_arr[14], vr14, smr1);
        dacc_arr[15] = __dtu_m_vmm2_mode18_f32(dacc_arr[15], vr15, smr1);

        vr0 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr1 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr2 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr3 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr4 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr5 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr6 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr7 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr8 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr9 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr10 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr11 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr12 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr13 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr14 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr15 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off1);

        smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                   rhs_tar_off0, 0);
        smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                   rhs_tar_off0, 1);
        smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                   rhs_tar_off0, 2);
        smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                   rhs_tar_off0, 3);
        smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                   rhs_tar_off0, 4);
        smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                   rhs_tar_off0, 5);
        smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                   rhs_tar_off0, 6);
        smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                   rhs_tar_off0, 7);
        smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                   rhs_tar_off0, 8);
        smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                   rhs_tar_off0, 9);
        smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                   rhs_tar_off0, 10);
        smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                   rhs_tar_off0, 11);
        smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                   rhs_tar_off0, 12);
        smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                   rhs_tar_off0, 13);
        smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                   rhs_tar_off0, 14);
        smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                   rhs_tar_off0, 15);

        // m1k1
        dacc_arr[16] = __dtu_m_vmm2_mode18_f32(dacc_arr[16], vr0, smr1);
        dacc_arr[17] = __dtu_m_vmm2_mode18_f32(dacc_arr[17], vr1, smr1);
        dacc_arr[18] = __dtu_m_vmm2_mode18_f32(dacc_arr[18], vr2, smr1);
        dacc_arr[19] = __dtu_m_vmm2_mode18_f32(dacc_arr[19], vr3, smr1);
        dacc_arr[20] = __dtu_m_vmm2_mode18_f32(dacc_arr[20], vr4, smr1);
        dacc_arr[21] = __dtu_m_vmm2_mode18_f32(dacc_arr[21], vr5, smr1);
        dacc_arr[22] = __dtu_m_vmm2_mode18_f32(dacc_arr[22], vr6, smr1);
        dacc_arr[23] = __dtu_m_vmm2_mode18_f32(dacc_arr[23], vr7, smr1);
        dacc_arr[24] = __dtu_m_vmm2_mode18_f32(dacc_arr[24], vr8, smr1);
        dacc_arr[25] = __dtu_m_vmm2_mode18_f32(dacc_arr[25], vr9, smr1);
        dacc_arr[26] = __dtu_m_vmm2_mode18_f32(dacc_arr[26], vr10, smr1);
        dacc_arr[27] = __dtu_m_vmm2_mode18_f32(dacc_arr[27], vr11, smr1);
        dacc_arr[28] = __dtu_m_vmm2_mode18_f32(dacc_arr[28], vr12, smr1);
        dacc_arr[29] = __dtu_m_vmm2_mode18_f32(dacc_arr[29], vr13, smr1);
        dacc_arr[30] = __dtu_m_vmm2_mode18_f32(dacc_arr[30], vr14, smr1);
        dacc_arr[31] = __dtu_m_vmm2_mode18_f32(dacc_arr[31], vr15, smr1);

        vr0 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr1 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr2 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr3 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr4 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr5 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr6 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr7 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr8 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr9 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr10 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr11 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr12 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr13 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr14 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
        vr15 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      }  // end kcout-1

      // m0k0
      dacc_arr[0] = __dtu_m_vmm2_mode18_f32(dacc_arr[0], vr0, smr0);
      dacc_arr[1] = __dtu_m_vmm2_mode18_f32(dacc_arr[1], vr1, smr0);
      dacc_arr[2] = __dtu_m_vmm2_mode18_f32(dacc_arr[2], vr2, smr0);
      dacc_arr[3] = __dtu_m_vmm2_mode18_f32(dacc_arr[3], vr3, smr0);
      dacc_arr[4] = __dtu_m_vmm2_mode18_f32(dacc_arr[4], vr4, smr0);
      dacc_arr[5] = __dtu_m_vmm2_mode18_f32(dacc_arr[5], vr5, smr0);
      dacc_arr[6] = __dtu_m_vmm2_mode18_f32(dacc_arr[6], vr6, smr0);
      dacc_arr[7] = __dtu_m_vmm2_mode18_f32(dacc_arr[7], vr7, smr0);
      dacc_arr[8] = __dtu_m_vmm2_mode18_f32(dacc_arr[8], vr8, smr0);
      dacc_arr[9] = __dtu_m_vmm2_mode18_f32(dacc_arr[9], vr9, smr0);
      dacc_arr[10] = __dtu_m_vmm2_mode18_f32(dacc_arr[10], vr10, smr0);
      dacc_arr[11] = __dtu_m_vmm2_mode18_f32(dacc_arr[11], vr11, smr0);
      dacc_arr[12] = __dtu_m_vmm2_mode18_f32(dacc_arr[12], vr12, smr0);
      dacc_arr[13] = __dtu_m_vmm2_mode18_f32(dacc_arr[13], vr13, smr0);
      dacc_arr[14] = __dtu_m_vmm2_mode18_f32(dacc_arr[14], vr14, smr0);
      dacc_arr[15] = __dtu_m_vmm2_mode18_f32(dacc_arr[15], vr15, smr0);

      vr0 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr1 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr2 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr3 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr4 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr5 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr6 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr7 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr8 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr9 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr10 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr11 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr12 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr13 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr14 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr15 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off1);

      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 0);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 1);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 2);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 3);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 4);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 5);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 6);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 7);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 8);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 9);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 10);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 11);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 12);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 13);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 14);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off1, 15);

      // m1k0
      dacc_arr[16] = __dtu_m_vmm2_mode18_f32(dacc_arr[16], vr0, smr0);
      dacc_arr[17] = __dtu_m_vmm2_mode18_f32(dacc_arr[17], vr1, smr0);
      dacc_arr[18] = __dtu_m_vmm2_mode18_f32(dacc_arr[18], vr2, smr0);
      dacc_arr[19] = __dtu_m_vmm2_mode18_f32(dacc_arr[19], vr3, smr0);
      dacc_arr[20] = __dtu_m_vmm2_mode18_f32(dacc_arr[20], vr4, smr0);
      dacc_arr[21] = __dtu_m_vmm2_mode18_f32(dacc_arr[21], vr5, smr0);
      dacc_arr[22] = __dtu_m_vmm2_mode18_f32(dacc_arr[22], vr6, smr0);
      dacc_arr[23] = __dtu_m_vmm2_mode18_f32(dacc_arr[23], vr7, smr0);
      dacc_arr[24] = __dtu_m_vmm2_mode18_f32(dacc_arr[24], vr8, smr0);
      dacc_arr[25] = __dtu_m_vmm2_mode18_f32(dacc_arr[25], vr9, smr0);
      dacc_arr[26] = __dtu_m_vmm2_mode18_f32(dacc_arr[26], vr10, smr0);
      dacc_arr[27] = __dtu_m_vmm2_mode18_f32(dacc_arr[27], vr11, smr0);
      dacc_arr[28] = __dtu_m_vmm2_mode18_f32(dacc_arr[28], vr12, smr0);
      dacc_arr[29] = __dtu_m_vmm2_mode18_f32(dacc_arr[29], vr13, smr0);
      dacc_arr[30] = __dtu_m_vmm2_mode18_f32(dacc_arr[30], vr14, smr0);
      dacc_arr[31] = __dtu_m_vmm2_mode18_f32(dacc_arr[31], vr15, smr0);

      vr0 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr1 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr2 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr3 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr4 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr5 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr6 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr7 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr8 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr9 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr10 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr11 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr12 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr13 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr14 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr15 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);

      __dtu_c_movsr2naccovr(0x1);

      // m0k1
      dacc_arr[0] = __dtu_m_vmm2_mode18_f32(dacc_arr[0], vr0, smr1);
      dacc_arr[1] = __dtu_m_vmm2_mode18_f32(dacc_arr[1], vr1, smr1);
      dacc_arr[2] = __dtu_m_vmm2_mode18_f32(dacc_arr[2], vr2, smr1);
      dacc_arr[3] = __dtu_m_vmm2_mode18_f32(dacc_arr[3], vr3, smr1);
      dacc_arr[4] = __dtu_m_vmm2_mode18_f32(dacc_arr[4], vr4, smr1);
      dacc_arr[5] = __dtu_m_vmm2_mode18_f32(dacc_arr[5], vr5, smr1);
      dacc_arr[6] = __dtu_m_vmm2_mode18_f32(dacc_arr[6], vr6, smr1);
      dacc_arr[7] = __dtu_m_vmm2_mode18_f32(dacc_arr[7], vr7, smr1);
      dacc_arr[8] = __dtu_m_vmm2_mode18_f32(dacc_arr[8], vr8, smr1);
      dacc_arr[9] = __dtu_m_vmm2_mode18_f32(dacc_arr[9], vr9, smr1);
      dacc_arr[10] = __dtu_m_vmm2_mode18_f32(dacc_arr[10], vr10, smr1);
      dacc_arr[11] = __dtu_m_vmm2_mode18_f32(dacc_arr[11], vr11, smr1);
      dacc_arr[12] = __dtu_m_vmm2_mode18_f32(dacc_arr[12], vr12, smr1);
      dacc_arr[13] = __dtu_m_vmm2_mode18_f32(dacc_arr[13], vr13, smr1);
      dacc_arr[14] = __dtu_m_vmm2_mode18_f32(dacc_arr[14], vr14, smr1);
      dacc_arr[15] = __dtu_m_vmm2_mode18_f32(dacc_arr[15], vr15, smr1);

      vr0 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr1 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr2 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr3 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr4 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr5 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr6 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr7 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr8 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr9 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr10 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr11 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr12 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr13 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr14 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr15 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off2);

      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 0);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 1);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 2);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 3);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 4);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 5);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 6);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 7);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 8);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 9);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 10);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 11);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 12);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 13);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 14);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 15);

      // m1k1
      dacc_arr[16] = __dtu_m_vmm2_mode18_f32(dacc_arr[16], vr0, smr1);
      dacc_arr[17] = __dtu_m_vmm2_mode18_f32(dacc_arr[17], vr1, smr1);
      dacc_arr[18] = __dtu_m_vmm2_mode18_f32(dacc_arr[18], vr2, smr1);
      dacc_arr[19] = __dtu_m_vmm2_mode18_f32(dacc_arr[19], vr3, smr1);
      dacc_arr[20] = __dtu_m_vmm2_mode18_f32(dacc_arr[20], vr4, smr1);
      dacc_arr[21] = __dtu_m_vmm2_mode18_f32(dacc_arr[21], vr5, smr1);
      dacc_arr[22] = __dtu_m_vmm2_mode18_f32(dacc_arr[22], vr6, smr1);
      dacc_arr[23] = __dtu_m_vmm2_mode18_f32(dacc_arr[23], vr7, smr1);
      dacc_arr[24] = __dtu_m_vmm2_mode18_f32(dacc_arr[24], vr8, smr1);
      dacc_arr[25] = __dtu_m_vmm2_mode18_f32(dacc_arr[25], vr9, smr1);
      dacc_arr[26] = __dtu_m_vmm2_mode18_f32(dacc_arr[26], vr10, smr1);
      dacc_arr[27] = __dtu_m_vmm2_mode18_f32(dacc_arr[27], vr11, smr1);
      dacc_arr[28] = __dtu_m_vmm2_mode18_f32(dacc_arr[28], vr12, smr1);
      dacc_arr[29] = __dtu_m_vmm2_mode18_f32(dacc_arr[29], vr13, smr1);
      dacc_arr[30] = __dtu_m_vmm2_mode18_f32(dacc_arr[30], vr14, smr1);
      dacc_arr[31] = __dtu_m_vmm2_mode18_f32(dacc_arr[31], vr15, smr1);

      vr0 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr1 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr2 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr3 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr4 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr5 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr6 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr7 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr8 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr9 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr10 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr11 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr12 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr13 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr14 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr15 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);

      ////////////////////////////////////////////////////////////////////
      vab_shift += 256;
      __dtu_c_movsr2vab_m_s1(vab_shift);
      __dtu_c_movsr2vab_m_d(vab_shift);
      ////////////////////////////////////////////////////////////////////
    }  // end ncount-1

    __dtu_c_movsr2naccovr(naccovr);
    for (int k = 0; k < K - 32; k += 32) {
      // m0k0
      dacc_arr[0] = __dtu_m_vmm2_mode18_f32(dacc_arr[0], vr0, smr0);
      dacc_arr[1] = __dtu_m_vmm2_mode18_f32(dacc_arr[1], vr1, smr0);
      dacc_arr[2] = __dtu_m_vmm2_mode18_f32(dacc_arr[2], vr2, smr0);
      dacc_arr[3] = __dtu_m_vmm2_mode18_f32(dacc_arr[3], vr3, smr0);
      dacc_arr[4] = __dtu_m_vmm2_mode18_f32(dacc_arr[4], vr4, smr0);
      dacc_arr[5] = __dtu_m_vmm2_mode18_f32(dacc_arr[5], vr5, smr0);
      dacc_arr[6] = __dtu_m_vmm2_mode18_f32(dacc_arr[6], vr6, smr0);
      dacc_arr[7] = __dtu_m_vmm2_mode18_f32(dacc_arr[7], vr7, smr0);
      dacc_arr[8] = __dtu_m_vmm2_mode18_f32(dacc_arr[8], vr8, smr0);
      dacc_arr[9] = __dtu_m_vmm2_mode18_f32(dacc_arr[9], vr9, smr0);
      dacc_arr[10] = __dtu_m_vmm2_mode18_f32(dacc_arr[10], vr10, smr0);
      dacc_arr[11] = __dtu_m_vmm2_mode18_f32(dacc_arr[11], vr11, smr0);
      dacc_arr[12] = __dtu_m_vmm2_mode18_f32(dacc_arr[12], vr12, smr0);
      dacc_arr[13] = __dtu_m_vmm2_mode18_f32(dacc_arr[13], vr13, smr0);
      dacc_arr[14] = __dtu_m_vmm2_mode18_f32(dacc_arr[14], vr14, smr0);
      dacc_arr[15] = __dtu_m_vmm2_mode18_f32(dacc_arr[15], vr15, smr0);

      vr0 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr1 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr2 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr3 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr4 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr5 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr6 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr7 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr8 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr9 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr10 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr11 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr12 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr13 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr14 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr15 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off1);

      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 0);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 1);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 2);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 3);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 4);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 5);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 6);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 7);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 8);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 9);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 10);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 11);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 12);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 13);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 14);
      smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base,
                                                 rhs_tar_off0, 15);

      // m1k0
      dacc_arr[16] = __dtu_m_vmm2_mode18_f32(dacc_arr[16], vr0, smr0);
      dacc_arr[17] = __dtu_m_vmm2_mode18_f32(dacc_arr[17], vr1, smr0);
      dacc_arr[18] = __dtu_m_vmm2_mode18_f32(dacc_arr[18], vr2, smr0);
      dacc_arr[19] = __dtu_m_vmm2_mode18_f32(dacc_arr[19], vr3, smr0);
      dacc_arr[20] = __dtu_m_vmm2_mode18_f32(dacc_arr[20], vr4, smr0);
      dacc_arr[21] = __dtu_m_vmm2_mode18_f32(dacc_arr[21], vr5, smr0);
      dacc_arr[22] = __dtu_m_vmm2_mode18_f32(dacc_arr[22], vr6, smr0);
      dacc_arr[23] = __dtu_m_vmm2_mode18_f32(dacc_arr[23], vr7, smr0);
      dacc_arr[24] = __dtu_m_vmm2_mode18_f32(dacc_arr[24], vr8, smr0);
      dacc_arr[25] = __dtu_m_vmm2_mode18_f32(dacc_arr[25], vr9, smr0);
      dacc_arr[26] = __dtu_m_vmm2_mode18_f32(dacc_arr[26], vr10, smr0);
      dacc_arr[27] = __dtu_m_vmm2_mode18_f32(dacc_arr[27], vr11, smr0);
      dacc_arr[28] = __dtu_m_vmm2_mode18_f32(dacc_arr[28], vr12, smr0);
      dacc_arr[29] = __dtu_m_vmm2_mode18_f32(dacc_arr[29], vr13, smr0);
      dacc_arr[30] = __dtu_m_vmm2_mode18_f32(dacc_arr[30], vr14, smr0);
      dacc_arr[31] = __dtu_m_vmm2_mode18_f32(dacc_arr[31], vr15, smr0);

      vr0 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr1 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr2 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr3 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr4 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr5 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr6 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr7 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr8 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr9 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr10 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr11 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr12 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr13 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr14 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr15 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);

      __dtu_c_movsr2naccovr(0x1);

      // m0k1
      dacc_arr[0] = __dtu_m_vmm2_mode18_f32(dacc_arr[0], vr0, smr1);
      dacc_arr[1] = __dtu_m_vmm2_mode18_f32(dacc_arr[1], vr1, smr1);
      dacc_arr[2] = __dtu_m_vmm2_mode18_f32(dacc_arr[2], vr2, smr1);
      dacc_arr[3] = __dtu_m_vmm2_mode18_f32(dacc_arr[3], vr3, smr1);
      dacc_arr[4] = __dtu_m_vmm2_mode18_f32(dacc_arr[4], vr4, smr1);
      dacc_arr[5] = __dtu_m_vmm2_mode18_f32(dacc_arr[5], vr5, smr1);
      dacc_arr[6] = __dtu_m_vmm2_mode18_f32(dacc_arr[6], vr6, smr1);
      dacc_arr[7] = __dtu_m_vmm2_mode18_f32(dacc_arr[7], vr7, smr1);
      dacc_arr[8] = __dtu_m_vmm2_mode18_f32(dacc_arr[8], vr8, smr1);
      dacc_arr[9] = __dtu_m_vmm2_mode18_f32(dacc_arr[9], vr9, smr1);
      dacc_arr[10] = __dtu_m_vmm2_mode18_f32(dacc_arr[10], vr10, smr1);
      dacc_arr[11] = __dtu_m_vmm2_mode18_f32(dacc_arr[11], vr11, smr1);
      dacc_arr[12] = __dtu_m_vmm2_mode18_f32(dacc_arr[12], vr12, smr1);
      dacc_arr[13] = __dtu_m_vmm2_mode18_f32(dacc_arr[13], vr13, smr1);
      dacc_arr[14] = __dtu_m_vmm2_mode18_f32(dacc_arr[14], vr14, smr1);
      dacc_arr[15] = __dtu_m_vmm2_mode18_f32(dacc_arr[15], vr15, smr1);

      vr0 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr1 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr2 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr3 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr4 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr5 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr6 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr7 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr8 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr9 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr10 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr11 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr12 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr13 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr14 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr15 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off1);

      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 0);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 1);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 2);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 3);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 4);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 5);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 6);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 7);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 8);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 9);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 10);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 11);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 12);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 13);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 14);
      smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base,
                                                 rhs_tar_off0, 15);

      // m1k1
      dacc_arr[16] = __dtu_m_vmm2_mode18_f32(dacc_arr[16], vr0, smr1);
      dacc_arr[17] = __dtu_m_vmm2_mode18_f32(dacc_arr[17], vr1, smr1);
      dacc_arr[18] = __dtu_m_vmm2_mode18_f32(dacc_arr[18], vr2, smr1);
      dacc_arr[19] = __dtu_m_vmm2_mode18_f32(dacc_arr[19], vr3, smr1);
      dacc_arr[20] = __dtu_m_vmm2_mode18_f32(dacc_arr[20], vr4, smr1);
      dacc_arr[21] = __dtu_m_vmm2_mode18_f32(dacc_arr[21], vr5, smr1);
      dacc_arr[22] = __dtu_m_vmm2_mode18_f32(dacc_arr[22], vr6, smr1);
      dacc_arr[23] = __dtu_m_vmm2_mode18_f32(dacc_arr[23], vr7, smr1);
      dacc_arr[24] = __dtu_m_vmm2_mode18_f32(dacc_arr[24], vr8, smr1);
      dacc_arr[25] = __dtu_m_vmm2_mode18_f32(dacc_arr[25], vr9, smr1);
      dacc_arr[26] = __dtu_m_vmm2_mode18_f32(dacc_arr[26], vr10, smr1);
      dacc_arr[27] = __dtu_m_vmm2_mode18_f32(dacc_arr[27], vr11, smr1);
      dacc_arr[28] = __dtu_m_vmm2_mode18_f32(dacc_arr[28], vr12, smr1);
      dacc_arr[29] = __dtu_m_vmm2_mode18_f32(dacc_arr[29], vr13, smr1);
      dacc_arr[30] = __dtu_m_vmm2_mode18_f32(dacc_arr[30], vr14, smr1);
      dacc_arr[31] = __dtu_m_vmm2_mode18_f32(dacc_arr[31], vr15, smr1);

      vr0 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr1 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr2 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr3 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr4 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr5 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr6 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr7 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr8 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr9 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr10 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr11 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr12 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr13 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr14 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
      vr15 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);

      __dtu_c_movsr2naccovr(0x1);
    }  // end kcout-1

    // m0k0
    dacc_arr[0] = __dtu_m_vmm2_mode18_f32(dacc_arr[0], vr0, smr0);
    dacc_arr[1] = __dtu_m_vmm2_mode18_f32(dacc_arr[1], vr1, smr0);
    dacc_arr[2] = __dtu_m_vmm2_mode18_f32(dacc_arr[2], vr2, smr0);
    dacc_arr[3] = __dtu_m_vmm2_mode18_f32(dacc_arr[3], vr3, smr0);
    dacc_arr[4] = __dtu_m_vmm2_mode18_f32(dacc_arr[4], vr4, smr0);
    dacc_arr[5] = __dtu_m_vmm2_mode18_f32(dacc_arr[5], vr5, smr0);
    dacc_arr[6] = __dtu_m_vmm2_mode18_f32(dacc_arr[6], vr6, smr0);
    dacc_arr[7] = __dtu_m_vmm2_mode18_f32(dacc_arr[7], vr7, smr0);
    dacc_arr[8] = __dtu_m_vmm2_mode18_f32(dacc_arr[8], vr8, smr0);
    dacc_arr[9] = __dtu_m_vmm2_mode18_f32(dacc_arr[9], vr9, smr0);
    dacc_arr[10] = __dtu_m_vmm2_mode18_f32(dacc_arr[10], vr10, smr0);
    dacc_arr[11] = __dtu_m_vmm2_mode18_f32(dacc_arr[11], vr11, smr0);
    dacc_arr[12] = __dtu_m_vmm2_mode18_f32(dacc_arr[12], vr12, smr0);
    dacc_arr[13] = __dtu_m_vmm2_mode18_f32(dacc_arr[13], vr13, smr0);
    dacc_arr[14] = __dtu_m_vmm2_mode18_f32(dacc_arr[14], vr14, smr0);
    dacc_arr[15] = __dtu_m_vmm2_mode18_f32(dacc_arr[15], vr15, smr0);

    vr0 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr1 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr2 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr3 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr4 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr5 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr6 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr7 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr8 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr9 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr10 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr11 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr12 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr13 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr14 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr15 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off1);

    smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base, rhs_tar_off0,
                                               0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base, rhs_tar_off0,
                                               1);
    smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base, rhs_tar_off0,
                                               2);
    smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base, rhs_tar_off0,
                                               3);
    smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base, rhs_tar_off0,
                                               4);
    smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base, rhs_tar_off0,
                                               5);
    smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base, rhs_tar_off0,
                                               6);
    smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base, rhs_tar_off0,
                                               7);
    smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base, rhs_tar_off0,
                                               8);
    smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base, rhs_tar_off0,
                                               9);
    smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base, rhs_tar_off0,
                                               10);
    smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base, rhs_tar_off0,
                                               11);
    smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base, rhs_tar_off0,
                                               12);
    smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base, rhs_tar_off0,
                                               13);
    smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base, rhs_tar_off0,
                                               14);
    smr1 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr1, rhs_tar_base, rhs_tar_off2,
                                               15);

    // m1k0
    dacc_arr[16] = __dtu_m_vmm2_mode18_f32(dacc_arr[16], vr0, smr0);
    dacc_arr[17] = __dtu_m_vmm2_mode18_f32(dacc_arr[17], vr1, smr0);
    dacc_arr[18] = __dtu_m_vmm2_mode18_f32(dacc_arr[18], vr2, smr0);
    dacc_arr[19] = __dtu_m_vmm2_mode18_f32(dacc_arr[19], vr3, smr0);
    dacc_arr[20] = __dtu_m_vmm2_mode18_f32(dacc_arr[20], vr4, smr0);
    dacc_arr[21] = __dtu_m_vmm2_mode18_f32(dacc_arr[21], vr5, smr0);
    dacc_arr[22] = __dtu_m_vmm2_mode18_f32(dacc_arr[22], vr6, smr0);
    dacc_arr[23] = __dtu_m_vmm2_mode18_f32(dacc_arr[23], vr7, smr0);
    dacc_arr[24] = __dtu_m_vmm2_mode18_f32(dacc_arr[24], vr8, smr0);
    dacc_arr[25] = __dtu_m_vmm2_mode18_f32(dacc_arr[25], vr9, smr0);
    dacc_arr[26] = __dtu_m_vmm2_mode18_f32(dacc_arr[26], vr10, smr0);
    dacc_arr[27] = __dtu_m_vmm2_mode18_f32(dacc_arr[27], vr11, smr0);
    dacc_arr[28] = __dtu_m_vmm2_mode18_f32(dacc_arr[28], vr12, smr0);
    dacc_arr[29] = __dtu_m_vmm2_mode18_f32(dacc_arr[29], vr13, smr0);
    dacc_arr[30] = __dtu_m_vmm2_mode18_f32(dacc_arr[30], vr14, smr0);
    dacc_arr[31] = __dtu_m_vmm2_mode18_f32(dacc_arr[31], vr15, smr0);

    vr0 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr1 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr2 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr3 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr4 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr5 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr6 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr7 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr8 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr9 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr10 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr11 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr12 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr13 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr14 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr15 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);

    __dtu_c_movsr2naccovr(0x1);

    // m0k1
    dacc_arr[0] = __dtu_m_vmm2_mode18_f32(dacc_arr[0], vr0, smr1);
    dacc_arr[1] = __dtu_m_vmm2_mode18_f32(dacc_arr[1], vr1, smr1);
    dacc_arr[2] = __dtu_m_vmm2_mode18_f32(dacc_arr[2], vr2, smr1);
    dacc_arr[3] = __dtu_m_vmm2_mode18_f32(dacc_arr[3], vr3, smr1);
    dacc_arr[4] = __dtu_m_vmm2_mode18_f32(dacc_arr[4], vr4, smr1);
    dacc_arr[5] = __dtu_m_vmm2_mode18_f32(dacc_arr[5], vr5, smr1);
    dacc_arr[6] = __dtu_m_vmm2_mode18_f32(dacc_arr[6], vr6, smr1);
    dacc_arr[7] = __dtu_m_vmm2_mode18_f32(dacc_arr[7], vr7, smr1);
    dacc_arr[8] = __dtu_m_vmm2_mode18_f32(dacc_arr[8], vr8, smr1);
    dacc_arr[9] = __dtu_m_vmm2_mode18_f32(dacc_arr[9], vr9, smr1);
    dacc_arr[10] = __dtu_m_vmm2_mode18_f32(dacc_arr[10], vr10, smr1);
    dacc_arr[11] = __dtu_m_vmm2_mode18_f32(dacc_arr[11], vr11, smr1);
    dacc_arr[12] = __dtu_m_vmm2_mode18_f32(dacc_arr[12], vr12, smr1);
    dacc_arr[13] = __dtu_m_vmm2_mode18_f32(dacc_arr[13], vr13, smr1);
    dacc_arr[14] = __dtu_m_vmm2_mode18_f32(dacc_arr[14], vr14, smr1);
    dacc_arr[15] = __dtu_m_vmm2_mode18_f32(dacc_arr[15], vr15, smr1);

    vr0 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr1 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr2 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr3 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr4 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr5 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr6 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr7 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr8 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr9 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr10 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr11 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr12 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr13 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr14 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr15 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off3);

    smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0,
                                               0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0,
                                               1);
    smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0,
                                               2);
    smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0,
                                               3);
    smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0,
                                               4);
    smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0,
                                               5);
    smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0,
                                               6);
    smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0,
                                               7);
    smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0,
                                               8);
    smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0,
                                               9);
    smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0,
                                               10);
    smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0,
                                               11);
    smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0,
                                               12);
    smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0,
                                               13);
    smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0,
                                               14);
    smr0 = __dtu_v_ldsmr2_mem_v_mode18_f32_row(smr0, rhs_tar_base, rhs_tar_off0,
                                               15);

    // m1k1
    dacc_arr[16] = __dtu_m_vmm2_mode18_f32(dacc_arr[16], vr0, smr1);
    dacc_arr[17] = __dtu_m_vmm2_mode18_f32(dacc_arr[17], vr1, smr1);
    dacc_arr[18] = __dtu_m_vmm2_mode18_f32(dacc_arr[18], vr2, smr1);
    dacc_arr[19] = __dtu_m_vmm2_mode18_f32(dacc_arr[19], vr3, smr1);
    dacc_arr[20] = __dtu_m_vmm2_mode18_f32(dacc_arr[20], vr4, smr1);
    dacc_arr[21] = __dtu_m_vmm2_mode18_f32(dacc_arr[21], vr5, smr1);
    dacc_arr[22] = __dtu_m_vmm2_mode18_f32(dacc_arr[22], vr6, smr1);
    dacc_arr[23] = __dtu_m_vmm2_mode18_f32(dacc_arr[23], vr7, smr1);
    dacc_arr[24] = __dtu_m_vmm2_mode18_f32(dacc_arr[24], vr8, smr1);
    dacc_arr[25] = __dtu_m_vmm2_mode18_f32(dacc_arr[25], vr9, smr1);
    dacc_arr[26] = __dtu_m_vmm2_mode18_f32(dacc_arr[26], vr10, smr1);
    dacc_arr[27] = __dtu_m_vmm2_mode18_f32(dacc_arr[27], vr11, smr1);
    dacc_arr[28] = __dtu_m_vmm2_mode18_f32(dacc_arr[28], vr12, smr1);
    dacc_arr[29] = __dtu_m_vmm2_mode18_f32(dacc_arr[29], vr13, smr1);
    dacc_arr[30] = __dtu_m_vmm2_mode18_f32(dacc_arr[30], vr14, smr1);
    dacc_arr[31] = __dtu_m_vmm2_mode18_f32(dacc_arr[31], vr15, smr1);

    vr0 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr1 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr2 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr3 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr4 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr5 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr6 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr7 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr8 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr9 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr10 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr11 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr12 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr13 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr14 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);
    vr15 = __dtu_s_tvld_itar_f32(lhs_tar_base, lhs_tar_off0);

    ////////////////////////////////////////////////////////////////////

    vab_shift += 256;
    __dtu_c_movsr2vab_m_s1(vab_shift);
    __dtu_c_movsr2vab_m_d(vab_shift);
  }  // end mcount

  if (store_flag) {
    vab_shift = 0;
    __dtu_c_movsr2vab_lv_s(0);
    __dtu_c_movsr2vab_lv_d(0);
    __dtu_c_movsr2vab_m_s1(0);
    __dtu_c_movsr2vab_m_d(0);

// code block following can be optimized

#pragma clang loop unroll(disable)
    for (int m = 0; m < M; m = m + 32) {
#pragma clang loop unroll(disable)
      for (int n = 0; n < N - 64; n = n + 64) {
        c_dacc_arr[0] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[1] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[2] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[3] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[4] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[5] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[6] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[7] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[8] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[9] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[10] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[11] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[12] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[13] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[14] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[15] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[16] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[17] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[18] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[19] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[20] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[21] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[22] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[23] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[24] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[25] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[26] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[27] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[28] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[29] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[30] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
        c_dacc_arr[31] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off1);

        __dtu_c_movsr2vab_m_s2(0);
            dacc_arr[0] = __dtu_m_mop_mul_f32_da(dacc_arr[0], dacc_alpha);
            dacc_arr[1] = __dtu_m_mop_mul_f32_da(dacc_arr[1], dacc_alpha);
            dacc_arr[2] = __dtu_m_mop_mul_f32_da(dacc_arr[2], dacc_alpha);
            dacc_arr[3] = __dtu_m_mop_mul_f32_da(dacc_arr[3], dacc_alpha);
            dacc_arr[4] = __dtu_m_mop_mul_f32_da(dacc_arr[4], dacc_alpha);
            dacc_arr[5] = __dtu_m_mop_mul_f32_da(dacc_arr[5], dacc_alpha);
            dacc_arr[6] = __dtu_m_mop_mul_f32_da(dacc_arr[6], dacc_alpha);
            dacc_arr[7] = __dtu_m_mop_mul_f32_da(dacc_arr[7], dacc_alpha);
            dacc_arr[8] = __dtu_m_mop_mul_f32_da(dacc_arr[8], dacc_alpha);
            dacc_arr[9] = __dtu_m_mop_mul_f32_da(dacc_arr[9], dacc_alpha);
            dacc_arr[10] = __dtu_m_mop_mul_f32_da(dacc_arr[10], dacc_alpha);
            dacc_arr[11] = __dtu_m_mop_mul_f32_da(dacc_arr[11], dacc_alpha);
            dacc_arr[12] = __dtu_m_mop_mul_f32_da(dacc_arr[12], dacc_alpha);
            dacc_arr[13] = __dtu_m_mop_mul_f32_da(dacc_arr[13], dacc_alpha);
            dacc_arr[14] = __dtu_m_mop_mul_f32_da(dacc_arr[14], dacc_alpha);
            dacc_arr[15] = __dtu_m_mop_mul_f32_da(dacc_arr[15], dacc_alpha);
            dacc_arr[16] = __dtu_m_mop_mul_f32_da(dacc_arr[16], dacc_alpha);
            dacc_arr[17] = __dtu_m_mop_mul_f32_da(dacc_arr[17], dacc_alpha);
            dacc_arr[18] = __dtu_m_mop_mul_f32_da(dacc_arr[18], dacc_alpha);
            dacc_arr[19] = __dtu_m_mop_mul_f32_da(dacc_arr[19], dacc_alpha);
            dacc_arr[20] = __dtu_m_mop_mul_f32_da(dacc_arr[20], dacc_alpha);
            dacc_arr[21] = __dtu_m_mop_mul_f32_da(dacc_arr[21], dacc_alpha);
            dacc_arr[22] = __dtu_m_mop_mul_f32_da(dacc_arr[22], dacc_alpha);
            dacc_arr[23] = __dtu_m_mop_mul_f32_da(dacc_arr[23], dacc_alpha);
            dacc_arr[24] = __dtu_m_mop_mul_f32_da(dacc_arr[24], dacc_alpha);
            dacc_arr[25] = __dtu_m_mop_mul_f32_da(dacc_arr[25], dacc_alpha);
            dacc_arr[26] = __dtu_m_mop_mul_f32_da(dacc_arr[26], dacc_alpha);
            dacc_arr[27] = __dtu_m_mop_mul_f32_da(dacc_arr[27], dacc_alpha);
            dacc_arr[28] = __dtu_m_mop_mul_f32_da(dacc_arr[28], dacc_alpha);
            dacc_arr[29] = __dtu_m_mop_mul_f32_da(dacc_arr[29], dacc_alpha);
            dacc_arr[30] = __dtu_m_mop_mul_f32_da(dacc_arr[30], dacc_alpha);
            dacc_arr[31] = __dtu_m_mop_mul_f32_da(dacc_arr[31], dacc_alpha);
        
            c_dacc_arr[0] = __dtu_m_mop_mul_f32_da(c_dacc_arr[0], dacc_beta);
            c_dacc_arr[1] = __dtu_m_mop_mul_f32_da(c_dacc_arr[1], dacc_beta);
            c_dacc_arr[2] = __dtu_m_mop_mul_f32_da(c_dacc_arr[2], dacc_beta);
            c_dacc_arr[3] = __dtu_m_mop_mul_f32_da(c_dacc_arr[3], dacc_beta);
            c_dacc_arr[4] = __dtu_m_mop_mul_f32_da(c_dacc_arr[4], dacc_beta);
            c_dacc_arr[5] = __dtu_m_mop_mul_f32_da(c_dacc_arr[5], dacc_beta);
            c_dacc_arr[6] = __dtu_m_mop_mul_f32_da(c_dacc_arr[6], dacc_beta);
            c_dacc_arr[7] = __dtu_m_mop_mul_f32_da(c_dacc_arr[7], dacc_beta);
            c_dacc_arr[8] = __dtu_m_mop_mul_f32_da(c_dacc_arr[8], dacc_beta);
            c_dacc_arr[9] = __dtu_m_mop_mul_f32_da(c_dacc_arr[9], dacc_beta);
            c_dacc_arr[10] = __dtu_m_mop_mul_f32_da(c_dacc_arr[10], dacc_beta);
            c_dacc_arr[11] = __dtu_m_mop_mul_f32_da(c_dacc_arr[11], dacc_beta);
            c_dacc_arr[12] = __dtu_m_mop_mul_f32_da(c_dacc_arr[12], dacc_beta);
            c_dacc_arr[13] = __dtu_m_mop_mul_f32_da(c_dacc_arr[13], dacc_beta);
            c_dacc_arr[14] = __dtu_m_mop_mul_f32_da(c_dacc_arr[14], dacc_beta);
            c_dacc_arr[15] = __dtu_m_mop_mul_f32_da(c_dacc_arr[15], dacc_beta);
            c_dacc_arr[16] = __dtu_m_mop_mul_f32_da(c_dacc_arr[16], dacc_beta);
            c_dacc_arr[17] = __dtu_m_mop_mul_f32_da(c_dacc_arr[17], dacc_beta);
            c_dacc_arr[18] = __dtu_m_mop_mul_f32_da(c_dacc_arr[18], dacc_beta);
            c_dacc_arr[19] = __dtu_m_mop_mul_f32_da(c_dacc_arr[19], dacc_beta);
            c_dacc_arr[20] = __dtu_m_mop_mul_f32_da(c_dacc_arr[20], dacc_beta);
            c_dacc_arr[21] = __dtu_m_mop_mul_f32_da(c_dacc_arr[21], dacc_beta);
            c_dacc_arr[22] = __dtu_m_mop_mul_f32_da(c_dacc_arr[22], dacc_beta);
            c_dacc_arr[23] = __dtu_m_mop_mul_f32_da(c_dacc_arr[23], dacc_beta);
            c_dacc_arr[24] = __dtu_m_mop_mul_f32_da(c_dacc_arr[24], dacc_beta);
            c_dacc_arr[25] = __dtu_m_mop_mul_f32_da(c_dacc_arr[25], dacc_beta);
            c_dacc_arr[26] = __dtu_m_mop_mul_f32_da(c_dacc_arr[26], dacc_beta);
            c_dacc_arr[27] = __dtu_m_mop_mul_f32_da(c_dacc_arr[27], dacc_beta);
            c_dacc_arr[28] = __dtu_m_mop_mul_f32_da(c_dacc_arr[28], dacc_beta);
            c_dacc_arr[29] = __dtu_m_mop_mul_f32_da(c_dacc_arr[29], dacc_beta);
            c_dacc_arr[30] = __dtu_m_mop_mul_f32_da(c_dacc_arr[30], dacc_beta);
            c_dacc_arr[31] = __dtu_m_mop_mul_f32_da(c_dacc_arr[31], dacc_beta);


        __dtu_c_movsr2vab_m_s2(vab_shift);

        // alpha*[A*B] + beta*C
        dacc_arr[0] = __dtu_m_mop_add_f32_da(dacc_arr[0], c_dacc_arr[0]);
        dacc_arr[1] = __dtu_m_mop_add_f32_da(dacc_arr[1], c_dacc_arr[1]);
        dacc_arr[2] = __dtu_m_mop_add_f32_da(dacc_arr[2], c_dacc_arr[2]);
        dacc_arr[3] = __dtu_m_mop_add_f32_da(dacc_arr[3], c_dacc_arr[3]);
        dacc_arr[4] = __dtu_m_mop_add_f32_da(dacc_arr[4], c_dacc_arr[4]);
        dacc_arr[5] = __dtu_m_mop_add_f32_da(dacc_arr[5], c_dacc_arr[5]);
        dacc_arr[6] = __dtu_m_mop_add_f32_da(dacc_arr[6], c_dacc_arr[6]);
        dacc_arr[7] = __dtu_m_mop_add_f32_da(dacc_arr[7], c_dacc_arr[7]);
        dacc_arr[8] = __dtu_m_mop_add_f32_da(dacc_arr[8], c_dacc_arr[8]);
        dacc_arr[9] = __dtu_m_mop_add_f32_da(dacc_arr[9], c_dacc_arr[9]);
        dacc_arr[10] = __dtu_m_mop_add_f32_da(dacc_arr[10], c_dacc_arr[10]);
        dacc_arr[11] = __dtu_m_mop_add_f32_da(dacc_arr[11], c_dacc_arr[11]);
        dacc_arr[12] = __dtu_m_mop_add_f32_da(dacc_arr[12], c_dacc_arr[12]);
        dacc_arr[13] = __dtu_m_mop_add_f32_da(dacc_arr[13], c_dacc_arr[13]);
        dacc_arr[14] = __dtu_m_mop_add_f32_da(dacc_arr[14], c_dacc_arr[14]);
        dacc_arr[15] = __dtu_m_mop_add_f32_da(dacc_arr[15], c_dacc_arr[15]);
        dacc_arr[16] = __dtu_m_mop_add_f32_da(dacc_arr[16], c_dacc_arr[16]);
        dacc_arr[17] = __dtu_m_mop_add_f32_da(dacc_arr[17], c_dacc_arr[17]);
        dacc_arr[18] = __dtu_m_mop_add_f32_da(dacc_arr[18], c_dacc_arr[18]);
        dacc_arr[19] = __dtu_m_mop_add_f32_da(dacc_arr[19], c_dacc_arr[19]);
        dacc_arr[20] = __dtu_m_mop_add_f32_da(dacc_arr[20], c_dacc_arr[20]);
        dacc_arr[21] = __dtu_m_mop_add_f32_da(dacc_arr[21], c_dacc_arr[21]);
        dacc_arr[22] = __dtu_m_mop_add_f32_da(dacc_arr[22], c_dacc_arr[22]);
        dacc_arr[23] = __dtu_m_mop_add_f32_da(dacc_arr[23], c_dacc_arr[23]);
        dacc_arr[24] = __dtu_m_mop_add_f32_da(dacc_arr[24], c_dacc_arr[24]);
        dacc_arr[25] = __dtu_m_mop_add_f32_da(dacc_arr[25], c_dacc_arr[25]);
        dacc_arr[26] = __dtu_m_mop_add_f32_da(dacc_arr[26], c_dacc_arr[26]);
        dacc_arr[27] = __dtu_m_mop_add_f32_da(dacc_arr[27], c_dacc_arr[27]);
        dacc_arr[28] = __dtu_m_mop_add_f32_da(dacc_arr[28], c_dacc_arr[28]);
        dacc_arr[29] = __dtu_m_mop_add_f32_da(dacc_arr[29], c_dacc_arr[29]);
        dacc_arr[30] = __dtu_m_mop_add_f32_da(dacc_arr[30], c_dacc_arr[30]);
        dacc_arr[31] = __dtu_m_mop_add_f32_da(dacc_arr[31], c_dacc_arr[31]);

        vab_shift += 256;
        __dtu_c_movsr2vab_lv_s(vab_shift);
        __dtu_c_movsr2vab_lv_d(vab_shift);
        __dtu_c_movsr2vab_m_s1(vab_shift);
        __dtu_c_movsr2vab_m_d(vab_shift);
      }
      c_dacc_arr[0] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[1] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[2] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[3] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[4] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[5] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[6] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[7] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[8] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[9] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[10] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[11] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[12] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[13] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[14] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[15] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[16] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[17] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[18] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[19] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[20] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[21] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[22] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[23] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[24] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[25] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[26] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[27] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[28] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[29] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[30] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off0);
      c_dacc_arr[31] = __dtu_l_tvldqa_f32_da(bias_tar_base, bias_tar_off2);

      __dtu_c_movsr2vab_m_s2(0);

        dacc_arr[0] = __dtu_m_mop_mul_f32_da(dacc_arr[0], dacc_alpha);
        dacc_arr[1] = __dtu_m_mop_mul_f32_da(dacc_arr[1], dacc_alpha);
        dacc_arr[2] = __dtu_m_mop_mul_f32_da(dacc_arr[2], dacc_alpha);
        dacc_arr[3] = __dtu_m_mop_mul_f32_da(dacc_arr[3], dacc_alpha);
        dacc_arr[4] = __dtu_m_mop_mul_f32_da(dacc_arr[4], dacc_alpha);
        dacc_arr[5] = __dtu_m_mop_mul_f32_da(dacc_arr[5], dacc_alpha);
        dacc_arr[6] = __dtu_m_mop_mul_f32_da(dacc_arr[6], dacc_alpha);
        dacc_arr[7] = __dtu_m_mop_mul_f32_da(dacc_arr[7], dacc_alpha);
        dacc_arr[8] = __dtu_m_mop_mul_f32_da(dacc_arr[8], dacc_alpha);
        dacc_arr[9] = __dtu_m_mop_mul_f32_da(dacc_arr[9], dacc_alpha);
        dacc_arr[10] = __dtu_m_mop_mul_f32_da(dacc_arr[10], dacc_alpha);
        dacc_arr[11] = __dtu_m_mop_mul_f32_da(dacc_arr[11], dacc_alpha);
        dacc_arr[12] = __dtu_m_mop_mul_f32_da(dacc_arr[12], dacc_alpha);
        dacc_arr[13] = __dtu_m_mop_mul_f32_da(dacc_arr[13], dacc_alpha);
        dacc_arr[14] = __dtu_m_mop_mul_f32_da(dacc_arr[14], dacc_alpha);
        dacc_arr[15] = __dtu_m_mop_mul_f32_da(dacc_arr[15], dacc_alpha);
        dacc_arr[16] = __dtu_m_mop_mul_f32_da(dacc_arr[16], dacc_alpha);
        dacc_arr[17] = __dtu_m_mop_mul_f32_da(dacc_arr[17], dacc_alpha);
        dacc_arr[18] = __dtu_m_mop_mul_f32_da(dacc_arr[18], dacc_alpha);
        dacc_arr[19] = __dtu_m_mop_mul_f32_da(dacc_arr[19], dacc_alpha);
        dacc_arr[20] = __dtu_m_mop_mul_f32_da(dacc_arr[20], dacc_alpha);
        dacc_arr[21] = __dtu_m_mop_mul_f32_da(dacc_arr[21], dacc_alpha);
        dacc_arr[22] = __dtu_m_mop_mul_f32_da(dacc_arr[22], dacc_alpha);
        dacc_arr[23] = __dtu_m_mop_mul_f32_da(dacc_arr[23], dacc_alpha);
        dacc_arr[24] = __dtu_m_mop_mul_f32_da(dacc_arr[24], dacc_alpha);
        dacc_arr[25] = __dtu_m_mop_mul_f32_da(dacc_arr[25], dacc_alpha);
        dacc_arr[26] = __dtu_m_mop_mul_f32_da(dacc_arr[26], dacc_alpha);
        dacc_arr[27] = __dtu_m_mop_mul_f32_da(dacc_arr[27], dacc_alpha);
        dacc_arr[28] = __dtu_m_mop_mul_f32_da(dacc_arr[28], dacc_alpha);
        dacc_arr[29] = __dtu_m_mop_mul_f32_da(dacc_arr[29], dacc_alpha);
        dacc_arr[30] = __dtu_m_mop_mul_f32_da(dacc_arr[30], dacc_alpha);
        dacc_arr[31] = __dtu_m_mop_mul_f32_da(dacc_arr[31], dacc_alpha);

        c_dacc_arr[0] = __dtu_m_mop_mul_f32_da(c_dacc_arr[0], dacc_beta);
        c_dacc_arr[1] = __dtu_m_mop_mul_f32_da(c_dacc_arr[1], dacc_beta);
        c_dacc_arr[2] = __dtu_m_mop_mul_f32_da(c_dacc_arr[2], dacc_beta);
        c_dacc_arr[3] = __dtu_m_mop_mul_f32_da(c_dacc_arr[3], dacc_beta);
        c_dacc_arr[4] = __dtu_m_mop_mul_f32_da(c_dacc_arr[4], dacc_beta);
        c_dacc_arr[5] = __dtu_m_mop_mul_f32_da(c_dacc_arr[5], dacc_beta);
        c_dacc_arr[6] = __dtu_m_mop_mul_f32_da(c_dacc_arr[6], dacc_beta);
        c_dacc_arr[7] = __dtu_m_mop_mul_f32_da(c_dacc_arr[7], dacc_beta);
        c_dacc_arr[8] = __dtu_m_mop_mul_f32_da(c_dacc_arr[8], dacc_beta);
        c_dacc_arr[9] = __dtu_m_mop_mul_f32_da(c_dacc_arr[9], dacc_beta);
        c_dacc_arr[10] = __dtu_m_mop_mul_f32_da(c_dacc_arr[10], dacc_beta);
        c_dacc_arr[11] = __dtu_m_mop_mul_f32_da(c_dacc_arr[11], dacc_beta);
        c_dacc_arr[12] = __dtu_m_mop_mul_f32_da(c_dacc_arr[12], dacc_beta);
        c_dacc_arr[13] = __dtu_m_mop_mul_f32_da(c_dacc_arr[13], dacc_beta);
        c_dacc_arr[14] = __dtu_m_mop_mul_f32_da(c_dacc_arr[14], dacc_beta);
        c_dacc_arr[15] = __dtu_m_mop_mul_f32_da(c_dacc_arr[15], dacc_beta);
        c_dacc_arr[16] = __dtu_m_mop_mul_f32_da(c_dacc_arr[16], dacc_beta);
        c_dacc_arr[17] = __dtu_m_mop_mul_f32_da(c_dacc_arr[17], dacc_beta);
        c_dacc_arr[18] = __dtu_m_mop_mul_f32_da(c_dacc_arr[18], dacc_beta);
        c_dacc_arr[19] = __dtu_m_mop_mul_f32_da(c_dacc_arr[19], dacc_beta);
        c_dacc_arr[20] = __dtu_m_mop_mul_f32_da(c_dacc_arr[20], dacc_beta);
        c_dacc_arr[21] = __dtu_m_mop_mul_f32_da(c_dacc_arr[21], dacc_beta);
        c_dacc_arr[22] = __dtu_m_mop_mul_f32_da(c_dacc_arr[22], dacc_beta);
        c_dacc_arr[23] = __dtu_m_mop_mul_f32_da(c_dacc_arr[23], dacc_beta);
        c_dacc_arr[24] = __dtu_m_mop_mul_f32_da(c_dacc_arr[24], dacc_beta);
        c_dacc_arr[25] = __dtu_m_mop_mul_f32_da(c_dacc_arr[25], dacc_beta);
        c_dacc_arr[26] = __dtu_m_mop_mul_f32_da(c_dacc_arr[26], dacc_beta);
        c_dacc_arr[27] = __dtu_m_mop_mul_f32_da(c_dacc_arr[27], dacc_beta);
        c_dacc_arr[28] = __dtu_m_mop_mul_f32_da(c_dacc_arr[28], dacc_beta);
        c_dacc_arr[29] = __dtu_m_mop_mul_f32_da(c_dacc_arr[29], dacc_beta);
        c_dacc_arr[30] = __dtu_m_mop_mul_f32_da(c_dacc_arr[30], dacc_beta);
        c_dacc_arr[31] = __dtu_m_mop_mul_f32_da(c_dacc_arr[31], dacc_beta);

      __dtu_c_movsr2vab_m_s2(vab_shift);
      // alpha*[A*B] + beta*C
      dacc_arr[0] = __dtu_m_mop_add_f32_da(dacc_arr[0], c_dacc_arr[0]);
      dacc_arr[1] = __dtu_m_mop_add_f32_da(dacc_arr[1], c_dacc_arr[1]);
      dacc_arr[2] = __dtu_m_mop_add_f32_da(dacc_arr[2], c_dacc_arr[2]);
      dacc_arr[3] = __dtu_m_mop_add_f32_da(dacc_arr[3], c_dacc_arr[3]);
      dacc_arr[4] = __dtu_m_mop_add_f32_da(dacc_arr[4], c_dacc_arr[4]);
      dacc_arr[5] = __dtu_m_mop_add_f32_da(dacc_arr[5], c_dacc_arr[5]);
      dacc_arr[6] = __dtu_m_mop_add_f32_da(dacc_arr[6], c_dacc_arr[6]);
      dacc_arr[7] = __dtu_m_mop_add_f32_da(dacc_arr[7], c_dacc_arr[7]);
      dacc_arr[8] = __dtu_m_mop_add_f32_da(dacc_arr[8], c_dacc_arr[8]);
      dacc_arr[9] = __dtu_m_mop_add_f32_da(dacc_arr[9], c_dacc_arr[9]);
      dacc_arr[10] = __dtu_m_mop_add_f32_da(dacc_arr[10], c_dacc_arr[10]);
      dacc_arr[11] = __dtu_m_mop_add_f32_da(dacc_arr[11], c_dacc_arr[11]);
      dacc_arr[12] = __dtu_m_mop_add_f32_da(dacc_arr[12], c_dacc_arr[12]);
      dacc_arr[13] = __dtu_m_mop_add_f32_da(dacc_arr[13], c_dacc_arr[13]);
      dacc_arr[14] = __dtu_m_mop_add_f32_da(dacc_arr[14], c_dacc_arr[14]);
      dacc_arr[15] = __dtu_m_mop_add_f32_da(dacc_arr[15], c_dacc_arr[15]);
      dacc_arr[16] = __dtu_m_mop_add_f32_da(dacc_arr[16], c_dacc_arr[16]);
      dacc_arr[17] = __dtu_m_mop_add_f32_da(dacc_arr[17], c_dacc_arr[17]);
      dacc_arr[18] = __dtu_m_mop_add_f32_da(dacc_arr[18], c_dacc_arr[18]);
      dacc_arr[19] = __dtu_m_mop_add_f32_da(dacc_arr[19], c_dacc_arr[19]);
      dacc_arr[20] = __dtu_m_mop_add_f32_da(dacc_arr[20], c_dacc_arr[20]);
      dacc_arr[21] = __dtu_m_mop_add_f32_da(dacc_arr[21], c_dacc_arr[21]);
      dacc_arr[22] = __dtu_m_mop_add_f32_da(dacc_arr[22], c_dacc_arr[22]);
      dacc_arr[23] = __dtu_m_mop_add_f32_da(dacc_arr[23], c_dacc_arr[23]);
      dacc_arr[24] = __dtu_m_mop_add_f32_da(dacc_arr[24], c_dacc_arr[24]);
      dacc_arr[25] = __dtu_m_mop_add_f32_da(dacc_arr[25], c_dacc_arr[25]);
      dacc_arr[26] = __dtu_m_mop_add_f32_da(dacc_arr[26], c_dacc_arr[26]);
      dacc_arr[27] = __dtu_m_mop_add_f32_da(dacc_arr[27], c_dacc_arr[27]);
      dacc_arr[28] = __dtu_m_mop_add_f32_da(dacc_arr[28], c_dacc_arr[28]);
      dacc_arr[29] = __dtu_m_mop_add_f32_da(dacc_arr[29], c_dacc_arr[29]);
      dacc_arr[30] = __dtu_m_mop_add_f32_da(dacc_arr[30], c_dacc_arr[30]);
      dacc_arr[31] = __dtu_m_mop_add_f32_da(dacc_arr[31], c_dacc_arr[31]);

      vab_shift += 256;
      __dtu_c_movsr2vab_lv_s(vab_shift);
      __dtu_c_movsr2vab_lv_d(vab_shift);
      __dtu_c_movsr2vab_m_s1(vab_shift);
      __dtu_c_movsr2vab_m_d(vab_shift);
    }

    // add bias
    if (bias_en == 1) {
      vab_shift = 0;
      __dtu_c_movsr2vab_lv_s(0);
      __dtu_c_movsr2vab_lv_d(0);
      __dtu_c_movsr2vab_m_s1(0);
      __dtu_c_movsr2vab_m_d(0);

      __dtu_c_movsr2vab_m_s2(0);
#pragma clang loop unroll(disable)
      for (int m = 0; m < M; m = m + 32) {
#pragma clang loop unroll(disable)
        for (int n = 0; n < N - 64; n = n + 64) {
          dacc_bias = __dtu_l_tvldqa_f32_da(te_bias_tar_base, te_bias_tar_off0);
          // add bias
          dacc_arr[0] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[0]);
          dacc_arr[1] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[1]);
          dacc_arr[2] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[2]);
          dacc_arr[3] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[3]);
          dacc_arr[4] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[4]);
          dacc_arr[5] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[5]);
          dacc_arr[6] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[6]);
          dacc_arr[7] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[7]);
          dacc_arr[8] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[8]);
          dacc_arr[9] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[9]);
          dacc_arr[10] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[10]);
          dacc_arr[11] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[11]);
          dacc_arr[12] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[12]);
          dacc_arr[13] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[13]);
          dacc_arr[14] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[14]);
          dacc_arr[15] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[15]);
          dacc_arr[16] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[16]);
          dacc_arr[17] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[17]);
          dacc_arr[18] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[18]);
          dacc_arr[19] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[19]);
          dacc_arr[20] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[20]);
          dacc_arr[21] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[21]);
          dacc_arr[22] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[22]);
          dacc_arr[23] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[23]);
          dacc_arr[24] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[24]);
          dacc_arr[25] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[25]);
          dacc_arr[26] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[26]);
          dacc_arr[27] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[27]);
          dacc_arr[28] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[28]);
          dacc_arr[29] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[29]);
          dacc_arr[30] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[30]);
          dacc_arr[31] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[31]);
          vab_shift += 256;
          __dtu_c_movsr2vab_lv_s(vab_shift);
          __dtu_c_movsr2vab_lv_d(vab_shift);
          __dtu_c_movsr2vab_m_s1(vab_shift);
          __dtu_c_movsr2vab_m_d(vab_shift);
        }
        dacc_bias = __dtu_l_tvldqa_f32_da(te_bias_tar_base, te_bias_tar_off1);
        // add bias
        dacc_arr[0] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[0]);
        dacc_arr[1] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[1]);
        dacc_arr[2] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[2]);
        dacc_arr[3] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[3]);
        dacc_arr[4] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[4]);
        dacc_arr[5] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[5]);
        dacc_arr[6] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[6]);
        dacc_arr[7] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[7]);
        dacc_arr[8] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[8]);
        dacc_arr[9] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[9]);
        dacc_arr[10] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[10]);
        dacc_arr[11] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[11]);
        dacc_arr[12] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[12]);
        dacc_arr[13] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[13]);
        dacc_arr[14] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[14]);
        dacc_arr[15] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[15]);
        dacc_arr[16] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[16]);
        dacc_arr[17] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[17]);
        dacc_arr[18] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[18]);
        dacc_arr[19] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[19]);
        dacc_arr[20] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[20]);
        dacc_arr[21] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[21]);
        dacc_arr[22] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[22]);
        dacc_arr[23] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[23]);
        dacc_arr[24] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[24]);
        dacc_arr[25] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[25]);
        dacc_arr[26] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[26]);
        dacc_arr[27] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[27]);
        dacc_arr[28] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[28]);
        dacc_arr[29] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[29]);
        dacc_arr[30] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[30]);
        dacc_arr[31] = __dtu_m_mop_mac_f32_da(dacc_bias, da_bs, dacc_arr[31]);
        vab_shift += 256;
        __dtu_c_movsr2vab_lv_s(vab_shift);
        __dtu_c_movsr2vab_lv_d(vab_shift);
        __dtu_c_movsr2vab_m_s1(vab_shift);
        __dtu_c_movsr2vab_m_d(vab_shift);
      }
    }
    // vst dacc
    vab_shift = 0;
    __dtu_c_movsr2vab_lv_s(0);
    __dtu_c_movsr2vab_lv_d(0);
    __dtu_c_movsr2vab_m_s1(0);
    __dtu_c_movsr2vab_m_d(0);
#pragma clang loop unroll(disable)
    for (int m = 0; m < M; m = m + 32) {
#pragma clang loop unroll(disable)
      for (int n = 0; n < N - 64; n = n + 64) {
        // vst dacc
        __dtu_v_tvstda_f32_dual(dacc_arr[0], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[1], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[2], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[3], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[4], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[5], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[6], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[7], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[8], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[9], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[10], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[11], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[12], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[13], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[14], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[15], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[16], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[17], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[18], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[19], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[20], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[21], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[22], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[23], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[24], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[25], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[26], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[27], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[28], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[29], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[30], out_tar_base, out_tar_off0);
        __dtu_v_tvstda_f32_dual(dacc_arr[31], out_tar_base, out_tar_off1);

        vab_shift += 256;
        __dtu_c_movsr2vab_lv_s(vab_shift);
        __dtu_c_movsr2vab_lv_d(vab_shift);
        __dtu_c_movsr2vab_m_s1(vab_shift);
        __dtu_c_movsr2vab_m_d(vab_shift);
      }
      __dtu_v_tvstda_f32_dual(dacc_arr[0], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[1], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[2], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[3], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[4], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[5], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[6], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[7], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[8], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[9], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[10], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[11], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[12], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[13], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[14], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[15], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[16], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[17], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[18], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[19], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[20], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[21], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[22], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[23], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[24], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[25], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[26], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[27], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[28], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[29], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[30], out_tar_base, out_tar_off0);
      __dtu_v_tvstda_f32_dual(dacc_arr[31], out_tar_base, out_tar_off2);

      vab_shift += 256;
      __dtu_c_movsr2vab_lv_s(vab_shift);
      __dtu_c_movsr2vab_lv_d(vab_shift);
      __dtu_c_movsr2vab_m_s1(vab_shift);
      __dtu_c_movsr2vab_m_d(vab_shift);
    }
  }
#endif
}



__attribute__((device)) extern "C" void c_func_bfgemm_general(
    int a_addr, int b_addr, int c_addr, int M, int N, int K, int nacc_flag,
    int stroe_flag, int alpha_enable, int beta_enable, float alpha, float beta,
    float addmm_beta, int bias_en, int bias_addr, int cur_n) {
    // printf("TOPSOP_DATA_BF16 M is %d\n", M);
    // printf( "TOPSOP_DATA_BF16 is\n");
#if __GCU_ARCH__ >= 300

  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_lv_d(0);
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_s2(0);
  __dtu_c_movsr2vab_m_d(0);

  int BPE = 2;
  smr_t smr0, smr1;
  v64i8 vr0, vr1, vr2, vr3, vr4, vr5, vr6, vr7, vr8, vr9, vr10, vr11, vr12,
      vr13, vr14, vr15, vr16, vr17, vr18, vr19, vr20, vr21, vr22, vr23, vr24,
      vr25, vr26, vr27, vr28, vr29, vr30, vr31;
  v16f32 vr_alpha = __dtu_s_movr2vr_dup_f32(alpha);

  va16f32x4 qacc[64];
  va16bf16x2 c_dacc[64];

  va16f32 va_alpha0, va_alpha1, va_alpha2, va_alpha3;
  va_alpha0 = __dtu_l_movvr2va(vr_alpha);
  va_alpha1 = __dtu_l_movvr2va(vr_alpha);
  va_alpha2 = __dtu_l_movvr2va(vr_alpha);
  va_alpha3 = __dtu_l_movvr2va(vr_alpha);
  va16f32x4 qa_alpha =
      __dtu_insertva2qa_f32(va_alpha0, va_alpha1, va_alpha2, va_alpha3);

  v16f32 vr_scale = __dtu_s_movr2vr_dup_f32(beta);
  va16f32 vacc_beta0, vacc_beta1, vacc_beta2, vacc_beta3;
  vacc_beta0 = __dtu_l_movvr2va(vr_scale);
  vacc_beta1 = __dtu_l_movvr2va(vr_scale);
  vacc_beta2 = __dtu_l_movvr2va(vr_scale);
  vacc_beta3 = __dtu_l_movvr2va(vr_scale);
  va16f32x4 qa_beta =
      __dtu_insertva2qa_f32(vacc_beta0, vacc_beta1, vacc_beta2, vacc_beta3);

  if (bias_en == 0) {
    vr_scale = __dtu_s_movr2vr_dup_f32(0.0f);
  } else {
    vr_scale = __dtu_s_movr2vr_dup_f32(addmm_beta);
  }
  vacc_beta0 = __dtu_l_movvr2va(vr_scale);
  vacc_beta1 = __dtu_l_movvr2va(vr_scale);
  vacc_beta2 = __dtu_l_movvr2va(vr_scale);
  vacc_beta3 = __dtu_l_movvr2va(vr_scale);
  va16f32x4 qa_bias =
      __dtu_insertva2qa_f32(vacc_beta0, vacc_beta1, vacc_beta2, vacc_beta3);
  va16bf16x2 bs_dacc;

  auto k_unit = K >> 5;
  auto n_unit = N >> 6;
  auto on_unit = N >> 6;
  // vpt parallel in rhs
  int lt_addr = a_addr >> 6;
  int rt_addr = b_addr >> 7;
  int ot_addr = c_addr >> 7;
  int offset = 0;
  tar_t lt_base = __dtu_c_movsr2targ(TAR_ADDR_WARP(lt_addr, 0));
  offset = TAR_OFF_WARP(k_unit, k_unit);
  tar_t lt_off0 = __dtu_c_movsr2tari(offset, lt_base);
  offset = TAR_OFF_WARP(1 - 63 * k_unit, 1 - 63 * k_unit);  // next k
  tar_t lt_off1 = __dtu_c_movsr2tari(offset, lt_base);
  offset = TAR_OFF_WARP(1 - 64 * k_unit, 1 - 64 * k_unit);  //  new n
  tar_t lt_off2 = __dtu_c_movsr2tari(offset, lt_base);
  offset = TAR_OFF_WARP(1, 1);  // end k end n new m
  tar_t lt_off3 = __dtu_c_movsr2tari(offset, lt_base);

  tar_t rt_base = __dtu_c_movsr2targ((rt_addr) | ((rt_addr) + 1) << 16);
  offset = TAR_OFF_WARP(n_unit, n_unit);
  tar_t rt_off0 = __dtu_c_movsr2tari(offset, rt_base);
  offset = TAR_OFF_WARP(2 - (K - 1) * n_unit, 2 - (K - 1) * n_unit);
  tar_t rt_off1 = __dtu_c_movsr2tari(offset, rt_base);  // new n
  offset = TAR_OFF_WARP(2 - K * n_unit, 2 - K * n_unit);
  tar_t rt_off2 = __dtu_c_movsr2tari(offset, rt_base);  // new m

  auto bn_unit = N >> 6;
  int bt_addr = (c_addr >> 7) | ((c_addr >> 7) + 1) << 16;
  tar_t bt_base = __dtu_c_movsr2targ(bt_addr);
  offset = (bn_unit << 16) | bn_unit;
  tar_t bt_off0 = __dtu_c_movsr2tari(offset, bt_base);
  offset = (2 - 63 * bn_unit) & 0xffff;
  offset = (offset << 16) | offset;
  tar_t bt_off1 = __dtu_c_movsr2tari(offset, bt_base);
  offset = (2 << 16) | 2;
  tar_t bt_off2 = __dtu_c_movsr2tari(offset, bt_base);

  int biast_addr = ((bias_addr + cur_n * 2) >> 7) |
                   (((bias_addr + cur_n * 2) >> 7) + 1) << 16;
  tar_t biast_base = __dtu_c_movsr2targ(biast_addr);
  offset = (2 << 16) | 2;
  tar_t biast_off0 = __dtu_c_movsr2tari(offset, biast_base);
  offset = (2 - bn_unit) & 0xffff;
  offset = (offset << 16) | offset;
  tar_t biast_off1 = __dtu_c_movsr2tari(offset, bt_base);

  tar_t ot_base = __dtu_c_movsr2targ((ot_addr) | ((ot_addr) + 1) << 16);
  offset = TAR_OFF_WARP(on_unit, on_unit);
  tar_t ot_off0 = __dtu_c_movsr2tari(offset, ot_base);
  offset = TAR_OFF_WARP(2 - 63 * on_unit, 2 - 63 * on_unit);
  tar_t ot_off1 = __dtu_c_movsr2tari(offset, ot_base);  // new n
  offset = TAR_OFF_WARP(2, 2);
  tar_t ot_off2 = __dtu_c_movsr2tari(offset, ot_base);  // new m

  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 0);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 1);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 2);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 3);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 4);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 5);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 6);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 7);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 8);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 9);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 10);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 11);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 12);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 13);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 14);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 15);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 16);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 17);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 18);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 19);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 20);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 21);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 22);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 23);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 24);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 25);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 26);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 27);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 28);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 29);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 30);
  smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 31);
  // m0k0
  vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr31 = __dtu_s_tvld_itar(lt_base, lt_off0);

  int naccovr = 0x10001;
  if (nacc_flag ^ 1) {
    naccovr = 0x1;
  }
  __dtu_c_movsr2naccovr(naccovr);
  __dtu_c_movsr2vab_m_s2(0);
  int vab_shift = 0;
// bf16 vmm2 mode17: [32, 64] * [64, 128] = [64, 128]
#pragma clang loop unroll(full)
  for (int m = 0; m < M; m += 64) {
    for (int n = 0; n < N - 128; n += 128) {  // VPT PARA DIM
      __dtu_c_movsr2naccovr(naccovr);
      for (int k = 0; k < K - 64; k += 64) {
        // smr1
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 0);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 1);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 2);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 3);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 4);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 5);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 6);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 7);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 8);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 9);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 10);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 11);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 12);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 13);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 14);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 15);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 16);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 17);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 18);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 19);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 20);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 21);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 22);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 23);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 24);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 25);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 26);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 27);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 28);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 29);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 30);
        smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 31);
        // m0k0 * smr0
        qacc[0] = __dtu_m_vmm2_mode17_bf16(qacc[0], vr0, smr0);
        qacc[1] = __dtu_m_vmm2_mode17_bf16(qacc[1], vr1, smr0);
        qacc[2] = __dtu_m_vmm2_mode17_bf16(qacc[2], vr2, smr0);
        qacc[3] = __dtu_m_vmm2_mode17_bf16(qacc[3], vr3, smr0);
        qacc[4] = __dtu_m_vmm2_mode17_bf16(qacc[4], vr4, smr0);
        qacc[5] = __dtu_m_vmm2_mode17_bf16(qacc[5], vr5, smr0);
        qacc[6] = __dtu_m_vmm2_mode17_bf16(qacc[6], vr6, smr0);
        qacc[7] = __dtu_m_vmm2_mode17_bf16(qacc[7], vr7, smr0);
        qacc[8] = __dtu_m_vmm2_mode17_bf16(qacc[8], vr8, smr0);
        qacc[9] = __dtu_m_vmm2_mode17_bf16(qacc[9], vr9, smr0);
        qacc[10] = __dtu_m_vmm2_mode17_bf16(qacc[10], vr10, smr0);
        qacc[11] = __dtu_m_vmm2_mode17_bf16(qacc[11], vr11, smr0);
        qacc[12] = __dtu_m_vmm2_mode17_bf16(qacc[12], vr12, smr0);
        qacc[13] = __dtu_m_vmm2_mode17_bf16(qacc[13], vr13, smr0);
        qacc[14] = __dtu_m_vmm2_mode17_bf16(qacc[14], vr14, smr0);
        qacc[15] = __dtu_m_vmm2_mode17_bf16(qacc[15], vr15, smr0);
        qacc[16] = __dtu_m_vmm2_mode17_bf16(qacc[16], vr16, smr0);
        qacc[17] = __dtu_m_vmm2_mode17_bf16(qacc[17], vr17, smr0);
        qacc[18] = __dtu_m_vmm2_mode17_bf16(qacc[18], vr18, smr0);
        qacc[19] = __dtu_m_vmm2_mode17_bf16(qacc[19], vr19, smr0);
        qacc[20] = __dtu_m_vmm2_mode17_bf16(qacc[20], vr20, smr0);
        qacc[21] = __dtu_m_vmm2_mode17_bf16(qacc[21], vr21, smr0);
        qacc[22] = __dtu_m_vmm2_mode17_bf16(qacc[22], vr22, smr0);
        qacc[23] = __dtu_m_vmm2_mode17_bf16(qacc[23], vr23, smr0);
        qacc[24] = __dtu_m_vmm2_mode17_bf16(qacc[24], vr24, smr0);
        qacc[25] = __dtu_m_vmm2_mode17_bf16(qacc[25], vr25, smr0);
        qacc[26] = __dtu_m_vmm2_mode17_bf16(qacc[26], vr26, smr0);
        qacc[27] = __dtu_m_vmm2_mode17_bf16(qacc[27], vr27, smr0);
        qacc[28] = __dtu_m_vmm2_mode17_bf16(qacc[28], vr28, smr0);
        qacc[29] = __dtu_m_vmm2_mode17_bf16(qacc[29], vr29, smr0);
        qacc[30] = __dtu_m_vmm2_mode17_bf16(qacc[30], vr30, smr0);
        qacc[31] = __dtu_m_vmm2_mode17_bf16(qacc[31], vr31, smr0);
        // m1k0
        vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr31 = __dtu_s_tvld_itar(lt_base, lt_off1);
        // m1k0 * smr0
        qacc[32] = __dtu_m_vmm2_mode17_bf16(qacc[32], vr0, smr0);
        qacc[33] = __dtu_m_vmm2_mode17_bf16(qacc[33], vr1, smr0);
        qacc[34] = __dtu_m_vmm2_mode17_bf16(qacc[34], vr2, smr0);
        qacc[35] = __dtu_m_vmm2_mode17_bf16(qacc[35], vr3, smr0);
        qacc[36] = __dtu_m_vmm2_mode17_bf16(qacc[36], vr4, smr0);
        qacc[37] = __dtu_m_vmm2_mode17_bf16(qacc[37], vr5, smr0);
        qacc[38] = __dtu_m_vmm2_mode17_bf16(qacc[38], vr6, smr0);
        qacc[39] = __dtu_m_vmm2_mode17_bf16(qacc[39], vr7, smr0);
        qacc[40] = __dtu_m_vmm2_mode17_bf16(qacc[40], vr8, smr0);
        qacc[41] = __dtu_m_vmm2_mode17_bf16(qacc[41], vr9, smr0);
        qacc[42] = __dtu_m_vmm2_mode17_bf16(qacc[42], vr10, smr0);
        qacc[43] = __dtu_m_vmm2_mode17_bf16(qacc[43], vr11, smr0);
        qacc[44] = __dtu_m_vmm2_mode17_bf16(qacc[44], vr12, smr0);
        qacc[45] = __dtu_m_vmm2_mode17_bf16(qacc[45], vr13, smr0);
        qacc[46] = __dtu_m_vmm2_mode17_bf16(qacc[46], vr14, smr0);
        qacc[47] = __dtu_m_vmm2_mode17_bf16(qacc[47], vr15, smr0);
        qacc[48] = __dtu_m_vmm2_mode17_bf16(qacc[48], vr16, smr0);
        qacc[49] = __dtu_m_vmm2_mode17_bf16(qacc[49], vr17, smr0);
        qacc[50] = __dtu_m_vmm2_mode17_bf16(qacc[50], vr18, smr0);
        qacc[51] = __dtu_m_vmm2_mode17_bf16(qacc[51], vr19, smr0);
        qacc[52] = __dtu_m_vmm2_mode17_bf16(qacc[52], vr20, smr0);
        qacc[53] = __dtu_m_vmm2_mode17_bf16(qacc[53], vr21, smr0);
        qacc[54] = __dtu_m_vmm2_mode17_bf16(qacc[54], vr22, smr0);
        qacc[55] = __dtu_m_vmm2_mode17_bf16(qacc[55], vr23, smr0);
        qacc[56] = __dtu_m_vmm2_mode17_bf16(qacc[56], vr24, smr0);
        qacc[57] = __dtu_m_vmm2_mode17_bf16(qacc[57], vr25, smr0);
        qacc[58] = __dtu_m_vmm2_mode17_bf16(qacc[58], vr26, smr0);
        qacc[59] = __dtu_m_vmm2_mode17_bf16(qacc[59], vr27, smr0);
        qacc[60] = __dtu_m_vmm2_mode17_bf16(qacc[60], vr28, smr0);
        qacc[61] = __dtu_m_vmm2_mode17_bf16(qacc[61], vr29, smr0);
        qacc[62] = __dtu_m_vmm2_mode17_bf16(qacc[62], vr30, smr0);
        qacc[63] = __dtu_m_vmm2_mode17_bf16(qacc[63], vr31, smr0);

        // m0k1
        vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr31 = __dtu_s_tvld_itar(lt_base, lt_off0);
        // next k unit smr0
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 0);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 1);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 2);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 3);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 4);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 5);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 6);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 7);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 8);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 9);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 10);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 11);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 12);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 13);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 14);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 15);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 16);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 17);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 18);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 19);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 20);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 21);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 22);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 23);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 24);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 25);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 26);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 27);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 28);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 29);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 30);
        smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 31);
        __dtu_c_movsr2naccovr(0x1);
        // m0k1 * smr1
        qacc[0] = __dtu_m_vmm2_mode17_bf16(qacc[0], vr0, smr1);
        qacc[1] = __dtu_m_vmm2_mode17_bf16(qacc[1], vr1, smr1);
        qacc[2] = __dtu_m_vmm2_mode17_bf16(qacc[2], vr2, smr1);
        qacc[3] = __dtu_m_vmm2_mode17_bf16(qacc[3], vr3, smr1);
        qacc[4] = __dtu_m_vmm2_mode17_bf16(qacc[4], vr4, smr1);
        qacc[5] = __dtu_m_vmm2_mode17_bf16(qacc[5], vr5, smr1);
        qacc[6] = __dtu_m_vmm2_mode17_bf16(qacc[6], vr6, smr1);
        qacc[7] = __dtu_m_vmm2_mode17_bf16(qacc[7], vr7, smr1);
        qacc[8] = __dtu_m_vmm2_mode17_bf16(qacc[8], vr8, smr1);
        qacc[9] = __dtu_m_vmm2_mode17_bf16(qacc[9], vr9, smr1);
        qacc[10] = __dtu_m_vmm2_mode17_bf16(qacc[10], vr10, smr1);
        qacc[11] = __dtu_m_vmm2_mode17_bf16(qacc[11], vr11, smr1);
        qacc[12] = __dtu_m_vmm2_mode17_bf16(qacc[12], vr12, smr1);
        qacc[13] = __dtu_m_vmm2_mode17_bf16(qacc[13], vr13, smr1);
        qacc[14] = __dtu_m_vmm2_mode17_bf16(qacc[14], vr14, smr1);
        qacc[15] = __dtu_m_vmm2_mode17_bf16(qacc[15], vr15, smr1);
        qacc[16] = __dtu_m_vmm2_mode17_bf16(qacc[16], vr16, smr1);
        qacc[17] = __dtu_m_vmm2_mode17_bf16(qacc[17], vr17, smr1);
        qacc[18] = __dtu_m_vmm2_mode17_bf16(qacc[18], vr18, smr1);
        qacc[19] = __dtu_m_vmm2_mode17_bf16(qacc[19], vr19, smr1);
        qacc[20] = __dtu_m_vmm2_mode17_bf16(qacc[20], vr20, smr1);
        qacc[21] = __dtu_m_vmm2_mode17_bf16(qacc[21], vr21, smr1);
        qacc[22] = __dtu_m_vmm2_mode17_bf16(qacc[22], vr22, smr1);
        qacc[23] = __dtu_m_vmm2_mode17_bf16(qacc[23], vr23, smr1);
        qacc[24] = __dtu_m_vmm2_mode17_bf16(qacc[24], vr24, smr1);
        qacc[25] = __dtu_m_vmm2_mode17_bf16(qacc[25], vr25, smr1);
        qacc[26] = __dtu_m_vmm2_mode17_bf16(qacc[26], vr26, smr1);
        qacc[27] = __dtu_m_vmm2_mode17_bf16(qacc[27], vr27, smr1);
        qacc[28] = __dtu_m_vmm2_mode17_bf16(qacc[28], vr28, smr1);
        qacc[29] = __dtu_m_vmm2_mode17_bf16(qacc[29], vr29, smr1);
        qacc[30] = __dtu_m_vmm2_mode17_bf16(qacc[30], vr30, smr1);
        qacc[31] = __dtu_m_vmm2_mode17_bf16(qacc[31], vr31, smr1);
        // m1k1
        vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr31 = __dtu_s_tvld_itar(lt_base, lt_off1);
        // m1k1 * smr1
        qacc[32] = __dtu_m_vmm2_mode17_bf16(qacc[32], vr0, smr1);
        qacc[33] = __dtu_m_vmm2_mode17_bf16(qacc[33], vr1, smr1);
        qacc[34] = __dtu_m_vmm2_mode17_bf16(qacc[34], vr2, smr1);
        qacc[35] = __dtu_m_vmm2_mode17_bf16(qacc[35], vr3, smr1);
        qacc[36] = __dtu_m_vmm2_mode17_bf16(qacc[36], vr4, smr1);
        qacc[37] = __dtu_m_vmm2_mode17_bf16(qacc[37], vr5, smr1);
        qacc[38] = __dtu_m_vmm2_mode17_bf16(qacc[38], vr6, smr1);
        qacc[39] = __dtu_m_vmm2_mode17_bf16(qacc[39], vr7, smr1);
        qacc[40] = __dtu_m_vmm2_mode17_bf16(qacc[40], vr8, smr1);
        qacc[41] = __dtu_m_vmm2_mode17_bf16(qacc[41], vr9, smr1);
        qacc[42] = __dtu_m_vmm2_mode17_bf16(qacc[42], vr10, smr1);
        qacc[43] = __dtu_m_vmm2_mode17_bf16(qacc[43], vr11, smr1);
        qacc[44] = __dtu_m_vmm2_mode17_bf16(qacc[44], vr12, smr1);
        qacc[45] = __dtu_m_vmm2_mode17_bf16(qacc[45], vr13, smr1);
        qacc[46] = __dtu_m_vmm2_mode17_bf16(qacc[46], vr14, smr1);
        qacc[47] = __dtu_m_vmm2_mode17_bf16(qacc[47], vr15, smr1);
        qacc[48] = __dtu_m_vmm2_mode17_bf16(qacc[48], vr16, smr1);
        qacc[49] = __dtu_m_vmm2_mode17_bf16(qacc[49], vr17, smr1);
        qacc[50] = __dtu_m_vmm2_mode17_bf16(qacc[50], vr18, smr1);
        qacc[51] = __dtu_m_vmm2_mode17_bf16(qacc[51], vr19, smr1);
        qacc[52] = __dtu_m_vmm2_mode17_bf16(qacc[52], vr20, smr1);
        qacc[53] = __dtu_m_vmm2_mode17_bf16(qacc[53], vr21, smr1);
        qacc[54] = __dtu_m_vmm2_mode17_bf16(qacc[54], vr22, smr1);
        qacc[55] = __dtu_m_vmm2_mode17_bf16(qacc[55], vr23, smr1);
        qacc[56] = __dtu_m_vmm2_mode17_bf16(qacc[56], vr24, smr1);
        qacc[57] = __dtu_m_vmm2_mode17_bf16(qacc[57], vr25, smr1);
        qacc[58] = __dtu_m_vmm2_mode17_bf16(qacc[58], vr26, smr1);
        qacc[59] = __dtu_m_vmm2_mode17_bf16(qacc[59], vr27, smr1);
        qacc[60] = __dtu_m_vmm2_mode17_bf16(qacc[60], vr28, smr1);
        qacc[61] = __dtu_m_vmm2_mode17_bf16(qacc[61], vr29, smr1);
        qacc[62] = __dtu_m_vmm2_mode17_bf16(qacc[62], vr30, smr1);
        qacc[63] = __dtu_m_vmm2_mode17_bf16(qacc[63], vr31, smr1);
        // next k unit m0k0
        vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr31 = __dtu_s_tvld_itar(lt_base, lt_off0);
      }  // end kcout-1
      // last k unit
      // smr1
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 0);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 1);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 2);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 3);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 4);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 5);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 6);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 7);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 8);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 9);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 10);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 11);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 12);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 13);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 14);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 15);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 16);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 17);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 18);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 19);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 20);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 21);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 22);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 23);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 24);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 25);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 26);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 27);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 28);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 29);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 30);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off1, 31);
      // m0k0 * smr0
      qacc[0] = __dtu_m_vmm2_mode17_bf16(qacc[0], vr0, smr0);
      qacc[1] = __dtu_m_vmm2_mode17_bf16(qacc[1], vr1, smr0);
      qacc[2] = __dtu_m_vmm2_mode17_bf16(qacc[2], vr2, smr0);
      qacc[3] = __dtu_m_vmm2_mode17_bf16(qacc[3], vr3, smr0);
      qacc[4] = __dtu_m_vmm2_mode17_bf16(qacc[4], vr4, smr0);
      qacc[5] = __dtu_m_vmm2_mode17_bf16(qacc[5], vr5, smr0);
      qacc[6] = __dtu_m_vmm2_mode17_bf16(qacc[6], vr6, smr0);
      qacc[7] = __dtu_m_vmm2_mode17_bf16(qacc[7], vr7, smr0);
      qacc[8] = __dtu_m_vmm2_mode17_bf16(qacc[8], vr8, smr0);
      qacc[9] = __dtu_m_vmm2_mode17_bf16(qacc[9], vr9, smr0);
      qacc[10] = __dtu_m_vmm2_mode17_bf16(qacc[10], vr10, smr0);
      qacc[11] = __dtu_m_vmm2_mode17_bf16(qacc[11], vr11, smr0);
      qacc[12] = __dtu_m_vmm2_mode17_bf16(qacc[12], vr12, smr0);
      qacc[13] = __dtu_m_vmm2_mode17_bf16(qacc[13], vr13, smr0);
      qacc[14] = __dtu_m_vmm2_mode17_bf16(qacc[14], vr14, smr0);
      qacc[15] = __dtu_m_vmm2_mode17_bf16(qacc[15], vr15, smr0);
      qacc[16] = __dtu_m_vmm2_mode17_bf16(qacc[16], vr16, smr0);
      qacc[17] = __dtu_m_vmm2_mode17_bf16(qacc[17], vr17, smr0);
      qacc[18] = __dtu_m_vmm2_mode17_bf16(qacc[18], vr18, smr0);
      qacc[19] = __dtu_m_vmm2_mode17_bf16(qacc[19], vr19, smr0);
      qacc[20] = __dtu_m_vmm2_mode17_bf16(qacc[20], vr20, smr0);
      qacc[21] = __dtu_m_vmm2_mode17_bf16(qacc[21], vr21, smr0);
      qacc[22] = __dtu_m_vmm2_mode17_bf16(qacc[22], vr22, smr0);
      qacc[23] = __dtu_m_vmm2_mode17_bf16(qacc[23], vr23, smr0);
      qacc[24] = __dtu_m_vmm2_mode17_bf16(qacc[24], vr24, smr0);
      qacc[25] = __dtu_m_vmm2_mode17_bf16(qacc[25], vr25, smr0);
      qacc[26] = __dtu_m_vmm2_mode17_bf16(qacc[26], vr26, smr0);
      qacc[27] = __dtu_m_vmm2_mode17_bf16(qacc[27], vr27, smr0);
      qacc[28] = __dtu_m_vmm2_mode17_bf16(qacc[28], vr28, smr0);
      qacc[29] = __dtu_m_vmm2_mode17_bf16(qacc[29], vr29, smr0);
      qacc[30] = __dtu_m_vmm2_mode17_bf16(qacc[30], vr30, smr0);
      qacc[31] = __dtu_m_vmm2_mode17_bf16(qacc[31], vr31, smr0);
      // m1k0
      vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr31 = __dtu_s_tvld_itar(lt_base, lt_off1);
      // m1k0 * smr0
      qacc[32] = __dtu_m_vmm2_mode17_bf16(qacc[32], vr0, smr0);
      qacc[33] = __dtu_m_vmm2_mode17_bf16(qacc[33], vr1, smr0);
      qacc[34] = __dtu_m_vmm2_mode17_bf16(qacc[34], vr2, smr0);
      qacc[35] = __dtu_m_vmm2_mode17_bf16(qacc[35], vr3, smr0);
      qacc[36] = __dtu_m_vmm2_mode17_bf16(qacc[36], vr4, smr0);
      qacc[37] = __dtu_m_vmm2_mode17_bf16(qacc[37], vr5, smr0);
      qacc[38] = __dtu_m_vmm2_mode17_bf16(qacc[38], vr6, smr0);
      qacc[39] = __dtu_m_vmm2_mode17_bf16(qacc[39], vr7, smr0);
      qacc[40] = __dtu_m_vmm2_mode17_bf16(qacc[40], vr8, smr0);
      qacc[41] = __dtu_m_vmm2_mode17_bf16(qacc[41], vr9, smr0);
      qacc[42] = __dtu_m_vmm2_mode17_bf16(qacc[42], vr10, smr0);
      qacc[43] = __dtu_m_vmm2_mode17_bf16(qacc[43], vr11, smr0);
      qacc[44] = __dtu_m_vmm2_mode17_bf16(qacc[44], vr12, smr0);
      qacc[45] = __dtu_m_vmm2_mode17_bf16(qacc[45], vr13, smr0);
      qacc[46] = __dtu_m_vmm2_mode17_bf16(qacc[46], vr14, smr0);
      qacc[47] = __dtu_m_vmm2_mode17_bf16(qacc[47], vr15, smr0);
      qacc[48] = __dtu_m_vmm2_mode17_bf16(qacc[48], vr16, smr0);
      qacc[49] = __dtu_m_vmm2_mode17_bf16(qacc[49], vr17, smr0);
      qacc[50] = __dtu_m_vmm2_mode17_bf16(qacc[50], vr18, smr0);
      qacc[51] = __dtu_m_vmm2_mode17_bf16(qacc[51], vr19, smr0);
      qacc[52] = __dtu_m_vmm2_mode17_bf16(qacc[52], vr20, smr0);
      qacc[53] = __dtu_m_vmm2_mode17_bf16(qacc[53], vr21, smr0);
      qacc[54] = __dtu_m_vmm2_mode17_bf16(qacc[54], vr22, smr0);
      qacc[55] = __dtu_m_vmm2_mode17_bf16(qacc[55], vr23, smr0);
      qacc[56] = __dtu_m_vmm2_mode17_bf16(qacc[56], vr24, smr0);
      qacc[57] = __dtu_m_vmm2_mode17_bf16(qacc[57], vr25, smr0);
      qacc[58] = __dtu_m_vmm2_mode17_bf16(qacc[58], vr26, smr0);
      qacc[59] = __dtu_m_vmm2_mode17_bf16(qacc[59], vr27, smr0);
      qacc[60] = __dtu_m_vmm2_mode17_bf16(qacc[60], vr28, smr0);
      qacc[61] = __dtu_m_vmm2_mode17_bf16(qacc[61], vr29, smr0);
      qacc[62] = __dtu_m_vmm2_mode17_bf16(qacc[62], vr30, smr0);
      qacc[63] = __dtu_m_vmm2_mode17_bf16(qacc[63], vr31, smr0);

      // m0k1
      vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr31 = __dtu_s_tvld_itar(lt_base, lt_off0);  // end k new n
      // next n unit smr0
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 0);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 1);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 2);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 3);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 4);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 5);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 6);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 7);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 8);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 9);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 10);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 11);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 12);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 13);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 14);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 15);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 16);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 17);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 18);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 19);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 20);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 21);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 22);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 23);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 24);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 25);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 26);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 27);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 28);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 29);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 30);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 31);
      __dtu_c_movsr2naccovr(0x1);
      // m0k1 * smr1
      qacc[0] = __dtu_m_vmm2_mode17_bf16(qacc[0], vr0, smr1);
      qacc[1] = __dtu_m_vmm2_mode17_bf16(qacc[1], vr1, smr1);
      qacc[2] = __dtu_m_vmm2_mode17_bf16(qacc[2], vr2, smr1);
      qacc[3] = __dtu_m_vmm2_mode17_bf16(qacc[3], vr3, smr1);
      qacc[4] = __dtu_m_vmm2_mode17_bf16(qacc[4], vr4, smr1);
      qacc[5] = __dtu_m_vmm2_mode17_bf16(qacc[5], vr5, smr1);
      qacc[6] = __dtu_m_vmm2_mode17_bf16(qacc[6], vr6, smr1);
      qacc[7] = __dtu_m_vmm2_mode17_bf16(qacc[7], vr7, smr1);
      qacc[8] = __dtu_m_vmm2_mode17_bf16(qacc[8], vr8, smr1);
      qacc[9] = __dtu_m_vmm2_mode17_bf16(qacc[9], vr9, smr1);
      qacc[10] = __dtu_m_vmm2_mode17_bf16(qacc[10], vr10, smr1);
      qacc[11] = __dtu_m_vmm2_mode17_bf16(qacc[11], vr11, smr1);
      qacc[12] = __dtu_m_vmm2_mode17_bf16(qacc[12], vr12, smr1);
      qacc[13] = __dtu_m_vmm2_mode17_bf16(qacc[13], vr13, smr1);
      qacc[14] = __dtu_m_vmm2_mode17_bf16(qacc[14], vr14, smr1);
      qacc[15] = __dtu_m_vmm2_mode17_bf16(qacc[15], vr15, smr1);
      qacc[16] = __dtu_m_vmm2_mode17_bf16(qacc[16], vr16, smr1);
      qacc[17] = __dtu_m_vmm2_mode17_bf16(qacc[17], vr17, smr1);
      qacc[18] = __dtu_m_vmm2_mode17_bf16(qacc[18], vr18, smr1);
      qacc[19] = __dtu_m_vmm2_mode17_bf16(qacc[19], vr19, smr1);
      qacc[20] = __dtu_m_vmm2_mode17_bf16(qacc[20], vr20, smr1);
      qacc[21] = __dtu_m_vmm2_mode17_bf16(qacc[21], vr21, smr1);
      qacc[22] = __dtu_m_vmm2_mode17_bf16(qacc[22], vr22, smr1);
      qacc[23] = __dtu_m_vmm2_mode17_bf16(qacc[23], vr23, smr1);
      qacc[24] = __dtu_m_vmm2_mode17_bf16(qacc[24], vr24, smr1);
      qacc[25] = __dtu_m_vmm2_mode17_bf16(qacc[25], vr25, smr1);
      qacc[26] = __dtu_m_vmm2_mode17_bf16(qacc[26], vr26, smr1);
      qacc[27] = __dtu_m_vmm2_mode17_bf16(qacc[27], vr27, smr1);
      qacc[28] = __dtu_m_vmm2_mode17_bf16(qacc[28], vr28, smr1);
      qacc[29] = __dtu_m_vmm2_mode17_bf16(qacc[29], vr29, smr1);
      qacc[30] = __dtu_m_vmm2_mode17_bf16(qacc[30], vr30, smr1);
      qacc[31] = __dtu_m_vmm2_mode17_bf16(qacc[31], vr31, smr1);
      // m1k1
      vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr31 = __dtu_s_tvld_itar(lt_base, lt_off2);
      // m1k1 * smr1
      qacc[32] = __dtu_m_vmm2_mode17_bf16(qacc[32], vr0, smr1);
      qacc[33] = __dtu_m_vmm2_mode17_bf16(qacc[33], vr1, smr1);
      qacc[34] = __dtu_m_vmm2_mode17_bf16(qacc[34], vr2, smr1);
      qacc[35] = __dtu_m_vmm2_mode17_bf16(qacc[35], vr3, smr1);
      qacc[36] = __dtu_m_vmm2_mode17_bf16(qacc[36], vr4, smr1);
      qacc[37] = __dtu_m_vmm2_mode17_bf16(qacc[37], vr5, smr1);
      qacc[38] = __dtu_m_vmm2_mode17_bf16(qacc[38], vr6, smr1);
      qacc[39] = __dtu_m_vmm2_mode17_bf16(qacc[39], vr7, smr1);
      qacc[40] = __dtu_m_vmm2_mode17_bf16(qacc[40], vr8, smr1);
      qacc[41] = __dtu_m_vmm2_mode17_bf16(qacc[41], vr9, smr1);
      qacc[42] = __dtu_m_vmm2_mode17_bf16(qacc[42], vr10, smr1);
      qacc[43] = __dtu_m_vmm2_mode17_bf16(qacc[43], vr11, smr1);
      qacc[44] = __dtu_m_vmm2_mode17_bf16(qacc[44], vr12, smr1);
      qacc[45] = __dtu_m_vmm2_mode17_bf16(qacc[45], vr13, smr1);
      qacc[46] = __dtu_m_vmm2_mode17_bf16(qacc[46], vr14, smr1);
      qacc[47] = __dtu_m_vmm2_mode17_bf16(qacc[47], vr15, smr1);
      qacc[48] = __dtu_m_vmm2_mode17_bf16(qacc[48], vr16, smr1);
      qacc[49] = __dtu_m_vmm2_mode17_bf16(qacc[49], vr17, smr1);
      qacc[50] = __dtu_m_vmm2_mode17_bf16(qacc[50], vr18, smr1);
      qacc[51] = __dtu_m_vmm2_mode17_bf16(qacc[51], vr19, smr1);
      qacc[52] = __dtu_m_vmm2_mode17_bf16(qacc[52], vr20, smr1);
      qacc[53] = __dtu_m_vmm2_mode17_bf16(qacc[53], vr21, smr1);
      qacc[54] = __dtu_m_vmm2_mode17_bf16(qacc[54], vr22, smr1);
      qacc[55] = __dtu_m_vmm2_mode17_bf16(qacc[55], vr23, smr1);
      qacc[56] = __dtu_m_vmm2_mode17_bf16(qacc[56], vr24, smr1);
      qacc[57] = __dtu_m_vmm2_mode17_bf16(qacc[57], vr25, smr1);
      qacc[58] = __dtu_m_vmm2_mode17_bf16(qacc[58], vr26, smr1);
      qacc[59] = __dtu_m_vmm2_mode17_bf16(qacc[59], vr27, smr1);
      qacc[60] = __dtu_m_vmm2_mode17_bf16(qacc[60], vr28, smr1);
      qacc[61] = __dtu_m_vmm2_mode17_bf16(qacc[61], vr29, smr1);
      qacc[62] = __dtu_m_vmm2_mode17_bf16(qacc[62], vr30, smr1);
      qacc[63] = __dtu_m_vmm2_mode17_bf16(qacc[63], vr31, smr1);
      // next n unit m0k0
      vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr31 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vab_shift += 512;
      __dtu_c_movsr2vab_m_s1(vab_shift);
      __dtu_c_movsr2vab_m_d(vab_shift);
    }  // end ncount-1
    __dtu_c_movsr2naccovr(naccovr);
    for (int k = 0; k < K - 64; k += 64) {
      // smr1
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 0);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 1);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 2);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 3);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 4);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 5);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 6);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 7);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 8);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 9);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 10);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 11);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 12);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 13);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 14);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 15);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 16);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 17);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 18);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 19);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 20);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 21);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 22);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 23);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 24);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 25);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 26);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 27);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 28);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 29);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 30);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 31);
      // m0k0 * smr0
      qacc[0] = __dtu_m_vmm2_mode17_bf16(qacc[0], vr0, smr0);
      qacc[1] = __dtu_m_vmm2_mode17_bf16(qacc[1], vr1, smr0);
      qacc[2] = __dtu_m_vmm2_mode17_bf16(qacc[2], vr2, smr0);
      qacc[3] = __dtu_m_vmm2_mode17_bf16(qacc[3], vr3, smr0);
      qacc[4] = __dtu_m_vmm2_mode17_bf16(qacc[4], vr4, smr0);
      qacc[5] = __dtu_m_vmm2_mode17_bf16(qacc[5], vr5, smr0);
      qacc[6] = __dtu_m_vmm2_mode17_bf16(qacc[6], vr6, smr0);
      qacc[7] = __dtu_m_vmm2_mode17_bf16(qacc[7], vr7, smr0);
      qacc[8] = __dtu_m_vmm2_mode17_bf16(qacc[8], vr8, smr0);
      qacc[9] = __dtu_m_vmm2_mode17_bf16(qacc[9], vr9, smr0);
      qacc[10] = __dtu_m_vmm2_mode17_bf16(qacc[10], vr10, smr0);
      qacc[11] = __dtu_m_vmm2_mode17_bf16(qacc[11], vr11, smr0);
      qacc[12] = __dtu_m_vmm2_mode17_bf16(qacc[12], vr12, smr0);
      qacc[13] = __dtu_m_vmm2_mode17_bf16(qacc[13], vr13, smr0);
      qacc[14] = __dtu_m_vmm2_mode17_bf16(qacc[14], vr14, smr0);
      qacc[15] = __dtu_m_vmm2_mode17_bf16(qacc[15], vr15, smr0);
      qacc[16] = __dtu_m_vmm2_mode17_bf16(qacc[16], vr16, smr0);
      qacc[17] = __dtu_m_vmm2_mode17_bf16(qacc[17], vr17, smr0);
      qacc[18] = __dtu_m_vmm2_mode17_bf16(qacc[18], vr18, smr0);
      qacc[19] = __dtu_m_vmm2_mode17_bf16(qacc[19], vr19, smr0);
      qacc[20] = __dtu_m_vmm2_mode17_bf16(qacc[20], vr20, smr0);
      qacc[21] = __dtu_m_vmm2_mode17_bf16(qacc[21], vr21, smr0);
      qacc[22] = __dtu_m_vmm2_mode17_bf16(qacc[22], vr22, smr0);
      qacc[23] = __dtu_m_vmm2_mode17_bf16(qacc[23], vr23, smr0);
      qacc[24] = __dtu_m_vmm2_mode17_bf16(qacc[24], vr24, smr0);
      qacc[25] = __dtu_m_vmm2_mode17_bf16(qacc[25], vr25, smr0);
      qacc[26] = __dtu_m_vmm2_mode17_bf16(qacc[26], vr26, smr0);
      qacc[27] = __dtu_m_vmm2_mode17_bf16(qacc[27], vr27, smr0);
      qacc[28] = __dtu_m_vmm2_mode17_bf16(qacc[28], vr28, smr0);
      qacc[29] = __dtu_m_vmm2_mode17_bf16(qacc[29], vr29, smr0);
      qacc[30] = __dtu_m_vmm2_mode17_bf16(qacc[30], vr30, smr0);
      qacc[31] = __dtu_m_vmm2_mode17_bf16(qacc[31], vr31, smr0);
      // m1k0
      vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr31 = __dtu_s_tvld_itar(lt_base, lt_off1);
      // m1k0 * smr0
      qacc[32] = __dtu_m_vmm2_mode17_bf16(qacc[32], vr0, smr0);
      qacc[33] = __dtu_m_vmm2_mode17_bf16(qacc[33], vr1, smr0);
      qacc[34] = __dtu_m_vmm2_mode17_bf16(qacc[34], vr2, smr0);
      qacc[35] = __dtu_m_vmm2_mode17_bf16(qacc[35], vr3, smr0);
      qacc[36] = __dtu_m_vmm2_mode17_bf16(qacc[36], vr4, smr0);
      qacc[37] = __dtu_m_vmm2_mode17_bf16(qacc[37], vr5, smr0);
      qacc[38] = __dtu_m_vmm2_mode17_bf16(qacc[38], vr6, smr0);
      qacc[39] = __dtu_m_vmm2_mode17_bf16(qacc[39], vr7, smr0);
      qacc[40] = __dtu_m_vmm2_mode17_bf16(qacc[40], vr8, smr0);
      qacc[41] = __dtu_m_vmm2_mode17_bf16(qacc[41], vr9, smr0);
      qacc[42] = __dtu_m_vmm2_mode17_bf16(qacc[42], vr10, smr0);
      qacc[43] = __dtu_m_vmm2_mode17_bf16(qacc[43], vr11, smr0);
      qacc[44] = __dtu_m_vmm2_mode17_bf16(qacc[44], vr12, smr0);
      qacc[45] = __dtu_m_vmm2_mode17_bf16(qacc[45], vr13, smr0);
      qacc[46] = __dtu_m_vmm2_mode17_bf16(qacc[46], vr14, smr0);
      qacc[47] = __dtu_m_vmm2_mode17_bf16(qacc[47], vr15, smr0);
      qacc[48] = __dtu_m_vmm2_mode17_bf16(qacc[48], vr16, smr0);
      qacc[49] = __dtu_m_vmm2_mode17_bf16(qacc[49], vr17, smr0);
      qacc[50] = __dtu_m_vmm2_mode17_bf16(qacc[50], vr18, smr0);
      qacc[51] = __dtu_m_vmm2_mode17_bf16(qacc[51], vr19, smr0);
      qacc[52] = __dtu_m_vmm2_mode17_bf16(qacc[52], vr20, smr0);
      qacc[53] = __dtu_m_vmm2_mode17_bf16(qacc[53], vr21, smr0);
      qacc[54] = __dtu_m_vmm2_mode17_bf16(qacc[54], vr22, smr0);
      qacc[55] = __dtu_m_vmm2_mode17_bf16(qacc[55], vr23, smr0);
      qacc[56] = __dtu_m_vmm2_mode17_bf16(qacc[56], vr24, smr0);
      qacc[57] = __dtu_m_vmm2_mode17_bf16(qacc[57], vr25, smr0);
      qacc[58] = __dtu_m_vmm2_mode17_bf16(qacc[58], vr26, smr0);
      qacc[59] = __dtu_m_vmm2_mode17_bf16(qacc[59], vr27, smr0);
      qacc[60] = __dtu_m_vmm2_mode17_bf16(qacc[60], vr28, smr0);
      qacc[61] = __dtu_m_vmm2_mode17_bf16(qacc[61], vr29, smr0);
      qacc[62] = __dtu_m_vmm2_mode17_bf16(qacc[62], vr30, smr0);
      qacc[63] = __dtu_m_vmm2_mode17_bf16(qacc[63], vr31, smr0);

      // m0k1
      vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr31 = __dtu_s_tvld_itar(lt_base, lt_off0);
      // next k unit smr0
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 0);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 1);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 2);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 3);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 4);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 5);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 6);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 7);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 8);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 9);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 10);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 11);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 12);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 13);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 14);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 15);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 16);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 17);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 18);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 19);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 20);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 21);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 22);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 23);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 24);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 25);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 26);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 27);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 28);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 29);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 30);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 31);
      __dtu_c_movsr2naccovr(0x1);
      // m0k1 * smr1
      qacc[0] = __dtu_m_vmm2_mode17_bf16(qacc[0], vr0, smr1);
      qacc[1] = __dtu_m_vmm2_mode17_bf16(qacc[1], vr1, smr1);
      qacc[2] = __dtu_m_vmm2_mode17_bf16(qacc[2], vr2, smr1);
      qacc[3] = __dtu_m_vmm2_mode17_bf16(qacc[3], vr3, smr1);
      qacc[4] = __dtu_m_vmm2_mode17_bf16(qacc[4], vr4, smr1);
      qacc[5] = __dtu_m_vmm2_mode17_bf16(qacc[5], vr5, smr1);
      qacc[6] = __dtu_m_vmm2_mode17_bf16(qacc[6], vr6, smr1);
      qacc[7] = __dtu_m_vmm2_mode17_bf16(qacc[7], vr7, smr1);
      qacc[8] = __dtu_m_vmm2_mode17_bf16(qacc[8], vr8, smr1);
      qacc[9] = __dtu_m_vmm2_mode17_bf16(qacc[9], vr9, smr1);
      qacc[10] = __dtu_m_vmm2_mode17_bf16(qacc[10], vr10, smr1);
      qacc[11] = __dtu_m_vmm2_mode17_bf16(qacc[11], vr11, smr1);
      qacc[12] = __dtu_m_vmm2_mode17_bf16(qacc[12], vr12, smr1);
      qacc[13] = __dtu_m_vmm2_mode17_bf16(qacc[13], vr13, smr1);
      qacc[14] = __dtu_m_vmm2_mode17_bf16(qacc[14], vr14, smr1);
      qacc[15] = __dtu_m_vmm2_mode17_bf16(qacc[15], vr15, smr1);
      qacc[16] = __dtu_m_vmm2_mode17_bf16(qacc[16], vr16, smr1);
      qacc[17] = __dtu_m_vmm2_mode17_bf16(qacc[17], vr17, smr1);
      qacc[18] = __dtu_m_vmm2_mode17_bf16(qacc[18], vr18, smr1);
      qacc[19] = __dtu_m_vmm2_mode17_bf16(qacc[19], vr19, smr1);
      qacc[20] = __dtu_m_vmm2_mode17_bf16(qacc[20], vr20, smr1);
      qacc[21] = __dtu_m_vmm2_mode17_bf16(qacc[21], vr21, smr1);
      qacc[22] = __dtu_m_vmm2_mode17_bf16(qacc[22], vr22, smr1);
      qacc[23] = __dtu_m_vmm2_mode17_bf16(qacc[23], vr23, smr1);
      qacc[24] = __dtu_m_vmm2_mode17_bf16(qacc[24], vr24, smr1);
      qacc[25] = __dtu_m_vmm2_mode17_bf16(qacc[25], vr25, smr1);
      qacc[26] = __dtu_m_vmm2_mode17_bf16(qacc[26], vr26, smr1);
      qacc[27] = __dtu_m_vmm2_mode17_bf16(qacc[27], vr27, smr1);
      qacc[28] = __dtu_m_vmm2_mode17_bf16(qacc[28], vr28, smr1);
      qacc[29] = __dtu_m_vmm2_mode17_bf16(qacc[29], vr29, smr1);
      qacc[30] = __dtu_m_vmm2_mode17_bf16(qacc[30], vr30, smr1);
      qacc[31] = __dtu_m_vmm2_mode17_bf16(qacc[31], vr31, smr1);
      // m1k1
      vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr31 = __dtu_s_tvld_itar(lt_base, lt_off1);
      // m1k1 * smr1
      qacc[32] = __dtu_m_vmm2_mode17_bf16(qacc[32], vr0, smr1);
      qacc[33] = __dtu_m_vmm2_mode17_bf16(qacc[33], vr1, smr1);
      qacc[34] = __dtu_m_vmm2_mode17_bf16(qacc[34], vr2, smr1);
      qacc[35] = __dtu_m_vmm2_mode17_bf16(qacc[35], vr3, smr1);
      qacc[36] = __dtu_m_vmm2_mode17_bf16(qacc[36], vr4, smr1);
      qacc[37] = __dtu_m_vmm2_mode17_bf16(qacc[37], vr5, smr1);
      qacc[38] = __dtu_m_vmm2_mode17_bf16(qacc[38], vr6, smr1);
      qacc[39] = __dtu_m_vmm2_mode17_bf16(qacc[39], vr7, smr1);
      qacc[40] = __dtu_m_vmm2_mode17_bf16(qacc[40], vr8, smr1);
      qacc[41] = __dtu_m_vmm2_mode17_bf16(qacc[41], vr9, smr1);
      qacc[42] = __dtu_m_vmm2_mode17_bf16(qacc[42], vr10, smr1);
      qacc[43] = __dtu_m_vmm2_mode17_bf16(qacc[43], vr11, smr1);
      qacc[44] = __dtu_m_vmm2_mode17_bf16(qacc[44], vr12, smr1);
      qacc[45] = __dtu_m_vmm2_mode17_bf16(qacc[45], vr13, smr1);
      qacc[46] = __dtu_m_vmm2_mode17_bf16(qacc[46], vr14, smr1);
      qacc[47] = __dtu_m_vmm2_mode17_bf16(qacc[47], vr15, smr1);
      qacc[48] = __dtu_m_vmm2_mode17_bf16(qacc[48], vr16, smr1);
      qacc[49] = __dtu_m_vmm2_mode17_bf16(qacc[49], vr17, smr1);
      qacc[50] = __dtu_m_vmm2_mode17_bf16(qacc[50], vr18, smr1);
      qacc[51] = __dtu_m_vmm2_mode17_bf16(qacc[51], vr19, smr1);
      qacc[52] = __dtu_m_vmm2_mode17_bf16(qacc[52], vr20, smr1);
      qacc[53] = __dtu_m_vmm2_mode17_bf16(qacc[53], vr21, smr1);
      qacc[54] = __dtu_m_vmm2_mode17_bf16(qacc[54], vr22, smr1);
      qacc[55] = __dtu_m_vmm2_mode17_bf16(qacc[55], vr23, smr1);
      qacc[56] = __dtu_m_vmm2_mode17_bf16(qacc[56], vr24, smr1);
      qacc[57] = __dtu_m_vmm2_mode17_bf16(qacc[57], vr25, smr1);
      qacc[58] = __dtu_m_vmm2_mode17_bf16(qacc[58], vr26, smr1);
      qacc[59] = __dtu_m_vmm2_mode17_bf16(qacc[59], vr27, smr1);
      qacc[60] = __dtu_m_vmm2_mode17_bf16(qacc[60], vr28, smr1);
      qacc[61] = __dtu_m_vmm2_mode17_bf16(qacc[61], vr29, smr1);
      qacc[62] = __dtu_m_vmm2_mode17_bf16(qacc[62], vr30, smr1);
      qacc[63] = __dtu_m_vmm2_mode17_bf16(qacc[63], vr31, smr1);

      // next k unit m0k0
      vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr31 = __dtu_s_tvld_itar(lt_base, lt_off0);
    }  // end kcout-1
    // last k unit of last n unit
    // smr1
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 1);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 2);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 3);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 4);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 5);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 6);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 7);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 8);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 9);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 10);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 11);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 12);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 13);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 14);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 15);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 16);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 17);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 18);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 19);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 20);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 21);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 22);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 23);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 24);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 25);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 26);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 27);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 28);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 29);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 30);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off2, 31);
    // m0k0 * smr0
    qacc[0] = __dtu_m_vmm2_mode17_bf16(qacc[0], vr0, smr0);
    qacc[1] = __dtu_m_vmm2_mode17_bf16(qacc[1], vr1, smr0);
    qacc[2] = __dtu_m_vmm2_mode17_bf16(qacc[2], vr2, smr0);
    qacc[3] = __dtu_m_vmm2_mode17_bf16(qacc[3], vr3, smr0);
    qacc[4] = __dtu_m_vmm2_mode17_bf16(qacc[4], vr4, smr0);
    qacc[5] = __dtu_m_vmm2_mode17_bf16(qacc[5], vr5, smr0);
    qacc[6] = __dtu_m_vmm2_mode17_bf16(qacc[6], vr6, smr0);
    qacc[7] = __dtu_m_vmm2_mode17_bf16(qacc[7], vr7, smr0);
    qacc[8] = __dtu_m_vmm2_mode17_bf16(qacc[8], vr8, smr0);
    qacc[9] = __dtu_m_vmm2_mode17_bf16(qacc[9], vr9, smr0);
    qacc[10] = __dtu_m_vmm2_mode17_bf16(qacc[10], vr10, smr0);
    qacc[11] = __dtu_m_vmm2_mode17_bf16(qacc[11], vr11, smr0);
    qacc[12] = __dtu_m_vmm2_mode17_bf16(qacc[12], vr12, smr0);
    qacc[13] = __dtu_m_vmm2_mode17_bf16(qacc[13], vr13, smr0);
    qacc[14] = __dtu_m_vmm2_mode17_bf16(qacc[14], vr14, smr0);
    qacc[15] = __dtu_m_vmm2_mode17_bf16(qacc[15], vr15, smr0);
    qacc[16] = __dtu_m_vmm2_mode17_bf16(qacc[16], vr16, smr0);
    qacc[17] = __dtu_m_vmm2_mode17_bf16(qacc[17], vr17, smr0);
    qacc[18] = __dtu_m_vmm2_mode17_bf16(qacc[18], vr18, smr0);
    qacc[19] = __dtu_m_vmm2_mode17_bf16(qacc[19], vr19, smr0);
    qacc[20] = __dtu_m_vmm2_mode17_bf16(qacc[20], vr20, smr0);
    qacc[21] = __dtu_m_vmm2_mode17_bf16(qacc[21], vr21, smr0);
    qacc[22] = __dtu_m_vmm2_mode17_bf16(qacc[22], vr22, smr0);
    qacc[23] = __dtu_m_vmm2_mode17_bf16(qacc[23], vr23, smr0);
    qacc[24] = __dtu_m_vmm2_mode17_bf16(qacc[24], vr24, smr0);
    qacc[25] = __dtu_m_vmm2_mode17_bf16(qacc[25], vr25, smr0);
    qacc[26] = __dtu_m_vmm2_mode17_bf16(qacc[26], vr26, smr0);
    qacc[27] = __dtu_m_vmm2_mode17_bf16(qacc[27], vr27, smr0);
    qacc[28] = __dtu_m_vmm2_mode17_bf16(qacc[28], vr28, smr0);
    qacc[29] = __dtu_m_vmm2_mode17_bf16(qacc[29], vr29, smr0);
    qacc[30] = __dtu_m_vmm2_mode17_bf16(qacc[30], vr30, smr0);
    qacc[31] = __dtu_m_vmm2_mode17_bf16(qacc[31], vr31, smr0);
    // m1k0
    vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr31 = __dtu_s_tvld_itar(lt_base, lt_off1);
    // m1k0 * smr0
    qacc[32] = __dtu_m_vmm2_mode17_bf16(qacc[32], vr0, smr0);
    qacc[33] = __dtu_m_vmm2_mode17_bf16(qacc[33], vr1, smr0);
    qacc[34] = __dtu_m_vmm2_mode17_bf16(qacc[34], vr2, smr0);
    qacc[35] = __dtu_m_vmm2_mode17_bf16(qacc[35], vr3, smr0);
    qacc[36] = __dtu_m_vmm2_mode17_bf16(qacc[36], vr4, smr0);
    qacc[37] = __dtu_m_vmm2_mode17_bf16(qacc[37], vr5, smr0);
    qacc[38] = __dtu_m_vmm2_mode17_bf16(qacc[38], vr6, smr0);
    qacc[39] = __dtu_m_vmm2_mode17_bf16(qacc[39], vr7, smr0);
    qacc[40] = __dtu_m_vmm2_mode17_bf16(qacc[40], vr8, smr0);
    qacc[41] = __dtu_m_vmm2_mode17_bf16(qacc[41], vr9, smr0);
    qacc[42] = __dtu_m_vmm2_mode17_bf16(qacc[42], vr10, smr0);
    qacc[43] = __dtu_m_vmm2_mode17_bf16(qacc[43], vr11, smr0);
    qacc[44] = __dtu_m_vmm2_mode17_bf16(qacc[44], vr12, smr0);
    qacc[45] = __dtu_m_vmm2_mode17_bf16(qacc[45], vr13, smr0);
    qacc[46] = __dtu_m_vmm2_mode17_bf16(qacc[46], vr14, smr0);
    qacc[47] = __dtu_m_vmm2_mode17_bf16(qacc[47], vr15, smr0);
    qacc[48] = __dtu_m_vmm2_mode17_bf16(qacc[48], vr16, smr0);
    qacc[49] = __dtu_m_vmm2_mode17_bf16(qacc[49], vr17, smr0);
    qacc[50] = __dtu_m_vmm2_mode17_bf16(qacc[50], vr18, smr0);
    qacc[51] = __dtu_m_vmm2_mode17_bf16(qacc[51], vr19, smr0);
    qacc[52] = __dtu_m_vmm2_mode17_bf16(qacc[52], vr20, smr0);
    qacc[53] = __dtu_m_vmm2_mode17_bf16(qacc[53], vr21, smr0);
    qacc[54] = __dtu_m_vmm2_mode17_bf16(qacc[54], vr22, smr0);
    qacc[55] = __dtu_m_vmm2_mode17_bf16(qacc[55], vr23, smr0);
    qacc[56] = __dtu_m_vmm2_mode17_bf16(qacc[56], vr24, smr0);
    qacc[57] = __dtu_m_vmm2_mode17_bf16(qacc[57], vr25, smr0);
    qacc[58] = __dtu_m_vmm2_mode17_bf16(qacc[58], vr26, smr0);
    qacc[59] = __dtu_m_vmm2_mode17_bf16(qacc[59], vr27, smr0);
    qacc[60] = __dtu_m_vmm2_mode17_bf16(qacc[60], vr28, smr0);
    qacc[61] = __dtu_m_vmm2_mode17_bf16(qacc[61], vr29, smr0);
    qacc[62] = __dtu_m_vmm2_mode17_bf16(qacc[62], vr30, smr0);
    qacc[63] = __dtu_m_vmm2_mode17_bf16(qacc[63], vr31, smr0);

    // m0k1
    vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr31 = __dtu_s_tvld_itar(lt_base, lt_off0);
    // next m unit smr0
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 1);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 2);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 3);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 4);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 5);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 6);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 7);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 8);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 9);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 10);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 11);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 12);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 13);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 14);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 15);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 16);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 17);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 18);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 19);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 20);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 21);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 22);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 23);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 24);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 25);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 26);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 27);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 28);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 29);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 30);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 31);
    __dtu_c_movsr2naccovr(0x1);
    // m0k1 * smr1
    qacc[0] = __dtu_m_vmm2_mode17_bf16(qacc[0], vr0, smr1);
    qacc[1] = __dtu_m_vmm2_mode17_bf16(qacc[1], vr1, smr1);
    qacc[2] = __dtu_m_vmm2_mode17_bf16(qacc[2], vr2, smr1);
    qacc[3] = __dtu_m_vmm2_mode17_bf16(qacc[3], vr3, smr1);
    qacc[4] = __dtu_m_vmm2_mode17_bf16(qacc[4], vr4, smr1);
    qacc[5] = __dtu_m_vmm2_mode17_bf16(qacc[5], vr5, smr1);
    qacc[6] = __dtu_m_vmm2_mode17_bf16(qacc[6], vr6, smr1);
    qacc[7] = __dtu_m_vmm2_mode17_bf16(qacc[7], vr7, smr1);
    qacc[8] = __dtu_m_vmm2_mode17_bf16(qacc[8], vr8, smr1);
    qacc[9] = __dtu_m_vmm2_mode17_bf16(qacc[9], vr9, smr1);
    qacc[10] = __dtu_m_vmm2_mode17_bf16(qacc[10], vr10, smr1);
    qacc[11] = __dtu_m_vmm2_mode17_bf16(qacc[11], vr11, smr1);
    qacc[12] = __dtu_m_vmm2_mode17_bf16(qacc[12], vr12, smr1);
    qacc[13] = __dtu_m_vmm2_mode17_bf16(qacc[13], vr13, smr1);
    qacc[14] = __dtu_m_vmm2_mode17_bf16(qacc[14], vr14, smr1);
    qacc[15] = __dtu_m_vmm2_mode17_bf16(qacc[15], vr15, smr1);
    qacc[16] = __dtu_m_vmm2_mode17_bf16(qacc[16], vr16, smr1);
    qacc[17] = __dtu_m_vmm2_mode17_bf16(qacc[17], vr17, smr1);
    qacc[18] = __dtu_m_vmm2_mode17_bf16(qacc[18], vr18, smr1);
    qacc[19] = __dtu_m_vmm2_mode17_bf16(qacc[19], vr19, smr1);
    qacc[20] = __dtu_m_vmm2_mode17_bf16(qacc[20], vr20, smr1);
    qacc[21] = __dtu_m_vmm2_mode17_bf16(qacc[21], vr21, smr1);
    qacc[22] = __dtu_m_vmm2_mode17_bf16(qacc[22], vr22, smr1);
    qacc[23] = __dtu_m_vmm2_mode17_bf16(qacc[23], vr23, smr1);
    qacc[24] = __dtu_m_vmm2_mode17_bf16(qacc[24], vr24, smr1);
    qacc[25] = __dtu_m_vmm2_mode17_bf16(qacc[25], vr25, smr1);
    qacc[26] = __dtu_m_vmm2_mode17_bf16(qacc[26], vr26, smr1);
    qacc[27] = __dtu_m_vmm2_mode17_bf16(qacc[27], vr27, smr1);
    qacc[28] = __dtu_m_vmm2_mode17_bf16(qacc[28], vr28, smr1);
    qacc[29] = __dtu_m_vmm2_mode17_bf16(qacc[29], vr29, smr1);
    qacc[30] = __dtu_m_vmm2_mode17_bf16(qacc[30], vr30, smr1);
    qacc[31] = __dtu_m_vmm2_mode17_bf16(qacc[31], vr31, smr1);
    // m1k1
    vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr31 = __dtu_s_tvld_itar(lt_base, lt_off3);
    // m1k1 * smr1
    qacc[32] = __dtu_m_vmm2_mode17_bf16(qacc[32], vr0, smr1);
    qacc[33] = __dtu_m_vmm2_mode17_bf16(qacc[33], vr1, smr1);
    qacc[34] = __dtu_m_vmm2_mode17_bf16(qacc[34], vr2, smr1);
    qacc[35] = __dtu_m_vmm2_mode17_bf16(qacc[35], vr3, smr1);
    qacc[36] = __dtu_m_vmm2_mode17_bf16(qacc[36], vr4, smr1);
    qacc[37] = __dtu_m_vmm2_mode17_bf16(qacc[37], vr5, smr1);
    qacc[38] = __dtu_m_vmm2_mode17_bf16(qacc[38], vr6, smr1);
    qacc[39] = __dtu_m_vmm2_mode17_bf16(qacc[39], vr7, smr1);
    qacc[40] = __dtu_m_vmm2_mode17_bf16(qacc[40], vr8, smr1);
    qacc[41] = __dtu_m_vmm2_mode17_bf16(qacc[41], vr9, smr1);
    qacc[42] = __dtu_m_vmm2_mode17_bf16(qacc[42], vr10, smr1);
    qacc[43] = __dtu_m_vmm2_mode17_bf16(qacc[43], vr11, smr1);
    qacc[44] = __dtu_m_vmm2_mode17_bf16(qacc[44], vr12, smr1);
    qacc[45] = __dtu_m_vmm2_mode17_bf16(qacc[45], vr13, smr1);
    qacc[46] = __dtu_m_vmm2_mode17_bf16(qacc[46], vr14, smr1);
    qacc[47] = __dtu_m_vmm2_mode17_bf16(qacc[47], vr15, smr1);
    qacc[48] = __dtu_m_vmm2_mode17_bf16(qacc[48], vr16, smr1);
    qacc[49] = __dtu_m_vmm2_mode17_bf16(qacc[49], vr17, smr1);
    qacc[50] = __dtu_m_vmm2_mode17_bf16(qacc[50], vr18, smr1);
    qacc[51] = __dtu_m_vmm2_mode17_bf16(qacc[51], vr19, smr1);
    qacc[52] = __dtu_m_vmm2_mode17_bf16(qacc[52], vr20, smr1);
    qacc[53] = __dtu_m_vmm2_mode17_bf16(qacc[53], vr21, smr1);
    qacc[54] = __dtu_m_vmm2_mode17_bf16(qacc[54], vr22, smr1);
    qacc[55] = __dtu_m_vmm2_mode17_bf16(qacc[55], vr23, smr1);
    qacc[56] = __dtu_m_vmm2_mode17_bf16(qacc[56], vr24, smr1);
    qacc[57] = __dtu_m_vmm2_mode17_bf16(qacc[57], vr25, smr1);
    qacc[58] = __dtu_m_vmm2_mode17_bf16(qacc[58], vr26, smr1);
    qacc[59] = __dtu_m_vmm2_mode17_bf16(qacc[59], vr27, smr1);
    qacc[60] = __dtu_m_vmm2_mode17_bf16(qacc[60], vr28, smr1);
    qacc[61] = __dtu_m_vmm2_mode17_bf16(qacc[61], vr29, smr1);
    qacc[62] = __dtu_m_vmm2_mode17_bf16(qacc[62], vr30, smr1);
    qacc[63] = __dtu_m_vmm2_mode17_bf16(qacc[63], vr31, smr1);

    // next m unit m0k0
    vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr31 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vab_shift += 512;
    __dtu_c_movsr2vab_m_s1(vab_shift);
    __dtu_c_movsr2vab_m_d(vab_shift);
  }  // end mcount

  if (stroe_flag) {
    vab_shift = 0;
    __dtu_c_movsr2vab_lv_s(0);
    __dtu_c_movsr2vab_lv_d(0);
    __dtu_c_movsr2vab_m_s1(0);
    __dtu_c_movsr2vab_m_d(0);
#pragma clang loop unroll(disable)
    for (int m = 0; m < M; m = m + 64) {
#pragma clang loop unroll(disable)
      for (int n = 0; n < N - 128; n = n + 128) {
        bs_dacc = __dtu_l_tvldqa_bf16_da(biast_base, biast_off0);
        c_dacc[0] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[1] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[2] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[3] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[4] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[5] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[6] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[7] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[8] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[9] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[10] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[11] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[12] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[13] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[14] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[15] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[16] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[17] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[18] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[19] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[20] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[21] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[22] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[23] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[24] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[25] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[26] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[27] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[28] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[29] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[30] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[31] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[32] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[33] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[34] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[35] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[36] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[37] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[38] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[39] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[40] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[41] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[42] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[43] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[44] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[45] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[46] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[47] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[48] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[49] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[50] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[51] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[52] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[53] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[54] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[55] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[56] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[57] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[58] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[59] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[60] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[61] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[62] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
        c_dacc[63] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off1);
        __dtu_c_movsr2vab_m_s2(0);
        qacc[0] = __dtu_m_mop_mul_f32_qa(qacc[0], qa_alpha);
        qacc[1] = __dtu_m_mop_mul_f32_qa(qacc[1], qa_alpha);
        qacc[2] = __dtu_m_mop_mul_f32_qa(qacc[2], qa_alpha);
        qacc[3] = __dtu_m_mop_mul_f32_qa(qacc[3], qa_alpha);
        qacc[4] = __dtu_m_mop_mul_f32_qa(qacc[4], qa_alpha);
        qacc[5] = __dtu_m_mop_mul_f32_qa(qacc[5], qa_alpha);
        qacc[6] = __dtu_m_mop_mul_f32_qa(qacc[6], qa_alpha);
        qacc[7] = __dtu_m_mop_mul_f32_qa(qacc[7], qa_alpha);
        qacc[8] = __dtu_m_mop_mul_f32_qa(qacc[8], qa_alpha);
        qacc[9] = __dtu_m_mop_mul_f32_qa(qacc[9], qa_alpha);
        qacc[10] = __dtu_m_mop_mul_f32_qa(qacc[10], qa_alpha);
        qacc[11] = __dtu_m_mop_mul_f32_qa(qacc[11], qa_alpha);
        qacc[12] = __dtu_m_mop_mul_f32_qa(qacc[12], qa_alpha);
        qacc[13] = __dtu_m_mop_mul_f32_qa(qacc[13], qa_alpha);
        qacc[14] = __dtu_m_mop_mul_f32_qa(qacc[14], qa_alpha);
        qacc[15] = __dtu_m_mop_mul_f32_qa(qacc[15], qa_alpha);
        qacc[16] = __dtu_m_mop_mul_f32_qa(qacc[16], qa_alpha);
        qacc[17] = __dtu_m_mop_mul_f32_qa(qacc[17], qa_alpha);
        qacc[18] = __dtu_m_mop_mul_f32_qa(qacc[18], qa_alpha);
        qacc[19] = __dtu_m_mop_mul_f32_qa(qacc[19], qa_alpha);
        qacc[20] = __dtu_m_mop_mul_f32_qa(qacc[20], qa_alpha);
        qacc[21] = __dtu_m_mop_mul_f32_qa(qacc[21], qa_alpha);
        qacc[22] = __dtu_m_mop_mul_f32_qa(qacc[22], qa_alpha);
        qacc[23] = __dtu_m_mop_mul_f32_qa(qacc[23], qa_alpha);
        qacc[24] = __dtu_m_mop_mul_f32_qa(qacc[24], qa_alpha);
        qacc[25] = __dtu_m_mop_mul_f32_qa(qacc[25], qa_alpha);
        qacc[26] = __dtu_m_mop_mul_f32_qa(qacc[26], qa_alpha);
        qacc[27] = __dtu_m_mop_mul_f32_qa(qacc[27], qa_alpha);
        qacc[28] = __dtu_m_mop_mul_f32_qa(qacc[28], qa_alpha);
        qacc[29] = __dtu_m_mop_mul_f32_qa(qacc[29], qa_alpha);
        qacc[30] = __dtu_m_mop_mul_f32_qa(qacc[30], qa_alpha);
        qacc[31] = __dtu_m_mop_mul_f32_qa(qacc[31], qa_alpha);
        qacc[32] = __dtu_m_mop_mul_f32_qa(qacc[32], qa_alpha);
        qacc[33] = __dtu_m_mop_mul_f32_qa(qacc[33], qa_alpha);
        qacc[34] = __dtu_m_mop_mul_f32_qa(qacc[34], qa_alpha);
        qacc[35] = __dtu_m_mop_mul_f32_qa(qacc[35], qa_alpha);
        qacc[36] = __dtu_m_mop_mul_f32_qa(qacc[36], qa_alpha);
        qacc[37] = __dtu_m_mop_mul_f32_qa(qacc[37], qa_alpha);
        qacc[38] = __dtu_m_mop_mul_f32_qa(qacc[38], qa_alpha);
        qacc[39] = __dtu_m_mop_mul_f32_qa(qacc[39], qa_alpha);
        qacc[40] = __dtu_m_mop_mul_f32_qa(qacc[40], qa_alpha);
        qacc[41] = __dtu_m_mop_mul_f32_qa(qacc[41], qa_alpha);
        qacc[42] = __dtu_m_mop_mul_f32_qa(qacc[42], qa_alpha);
        qacc[43] = __dtu_m_mop_mul_f32_qa(qacc[43], qa_alpha);
        qacc[44] = __dtu_m_mop_mul_f32_qa(qacc[44], qa_alpha);
        qacc[45] = __dtu_m_mop_mul_f32_qa(qacc[45], qa_alpha);
        qacc[46] = __dtu_m_mop_mul_f32_qa(qacc[46], qa_alpha);
        qacc[47] = __dtu_m_mop_mul_f32_qa(qacc[47], qa_alpha);
        qacc[48] = __dtu_m_mop_mul_f32_qa(qacc[48], qa_alpha);
        qacc[49] = __dtu_m_mop_mul_f32_qa(qacc[49], qa_alpha);
        qacc[50] = __dtu_m_mop_mul_f32_qa(qacc[50], qa_alpha);
        qacc[51] = __dtu_m_mop_mul_f32_qa(qacc[51], qa_alpha);
        qacc[52] = __dtu_m_mop_mul_f32_qa(qacc[52], qa_alpha);
        qacc[53] = __dtu_m_mop_mul_f32_qa(qacc[53], qa_alpha);
        qacc[54] = __dtu_m_mop_mul_f32_qa(qacc[54], qa_alpha);
        qacc[55] = __dtu_m_mop_mul_f32_qa(qacc[55], qa_alpha);
        qacc[56] = __dtu_m_mop_mul_f32_qa(qacc[56], qa_alpha);
        qacc[57] = __dtu_m_mop_mul_f32_qa(qacc[57], qa_alpha);
        qacc[58] = __dtu_m_mop_mul_f32_qa(qacc[58], qa_alpha);
        qacc[59] = __dtu_m_mop_mul_f32_qa(qacc[59], qa_alpha);
        qacc[60] = __dtu_m_mop_mul_f32_qa(qacc[60], qa_alpha);
        qacc[61] = __dtu_m_mop_mul_f32_qa(qacc[61], qa_alpha);
        qacc[62] = __dtu_m_mop_mul_f32_qa(qacc[62], qa_alpha);
        qacc[63] = __dtu_m_mop_mul_f32_qa(qacc[63], qa_alpha);

        qacc[0] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[0], qa_beta, qacc[0]);
        qacc[1] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[1], qa_beta, qacc[1]);
        qacc[2] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[2], qa_beta, qacc[2]);
        qacc[3] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[3], qa_beta, qacc[3]);
        qacc[4] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[4], qa_beta, qacc[4]);
        qacc[5] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[5], qa_beta, qacc[5]);
        qacc[6] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[6], qa_beta, qacc[6]);
        qacc[7] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[7], qa_beta, qacc[7]);
        qacc[8] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[8], qa_beta, qacc[8]);
        qacc[9] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[9], qa_beta, qacc[9]);
        qacc[10] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[10], qa_beta, qacc[10]);
        qacc[11] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[11], qa_beta, qacc[11]);
        qacc[12] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[12], qa_beta, qacc[12]);
        qacc[13] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[13], qa_beta, qacc[13]);
        qacc[14] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[14], qa_beta, qacc[14]);
        qacc[15] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[15], qa_beta, qacc[15]);
        qacc[16] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[16], qa_beta, qacc[16]);
        qacc[17] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[17], qa_beta, qacc[17]);
        qacc[18] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[18], qa_beta, qacc[18]);
        qacc[19] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[19], qa_beta, qacc[19]);
        qacc[20] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[20], qa_beta, qacc[20]);
        qacc[21] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[21], qa_beta, qacc[21]);
        qacc[22] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[22], qa_beta, qacc[22]);
        qacc[23] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[23], qa_beta, qacc[23]);
        qacc[24] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[24], qa_beta, qacc[24]);
        qacc[25] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[25], qa_beta, qacc[25]);
        qacc[26] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[26], qa_beta, qacc[26]);
        qacc[27] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[27], qa_beta, qacc[27]);
        qacc[28] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[28], qa_beta, qacc[28]);
        qacc[29] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[29], qa_beta, qacc[29]);
        qacc[30] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[30], qa_beta, qacc[30]);
        qacc[31] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[31], qa_beta, qacc[31]);
        qacc[32] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[32], qa_beta, qacc[32]);
        qacc[33] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[33], qa_beta, qacc[33]);
        qacc[34] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[34], qa_beta, qacc[34]);
        qacc[35] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[35], qa_beta, qacc[35]);
        qacc[36] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[36], qa_beta, qacc[36]);
        qacc[37] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[37], qa_beta, qacc[37]);
        qacc[38] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[38], qa_beta, qacc[38]);
        qacc[39] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[39], qa_beta, qacc[39]);
        qacc[40] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[40], qa_beta, qacc[40]);
        qacc[41] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[41], qa_beta, qacc[41]);
        qacc[42] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[42], qa_beta, qacc[42]);
        qacc[43] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[43], qa_beta, qacc[43]);
        qacc[44] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[44], qa_beta, qacc[44]);
        qacc[45] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[45], qa_beta, qacc[45]);
        qacc[46] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[46], qa_beta, qacc[46]);
        qacc[47] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[47], qa_beta, qacc[47]);
        qacc[48] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[48], qa_beta, qacc[48]);
        qacc[49] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[49], qa_beta, qacc[49]);
        qacc[50] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[50], qa_beta, qacc[50]);
        qacc[51] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[51], qa_beta, qacc[51]);
        qacc[52] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[52], qa_beta, qacc[52]);
        qacc[53] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[53], qa_beta, qacc[53]);
        qacc[54] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[54], qa_beta, qacc[54]);
        qacc[55] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[55], qa_beta, qacc[55]);
        qacc[56] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[56], qa_beta, qacc[56]);
        qacc[57] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[57], qa_beta, qacc[57]);
        qacc[58] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[58], qa_beta, qacc[58]);
        qacc[59] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[59], qa_beta, qacc[59]);
        qacc[60] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[60], qa_beta, qacc[60]);
        qacc[61] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[61], qa_beta, qacc[61]);
        qacc[62] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[62], qa_beta, qacc[62]);
        qacc[63] =
            __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[63], qa_beta, qacc[63]);
        // add bias
        if (bias_en == 1) {
          qacc[0] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[0]);
          qacc[1] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[1]);
          qacc[2] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[2]);
          qacc[3] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[3]);
          qacc[4] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[4]);
          qacc[5] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[5]);
          qacc[6] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[6]);
          qacc[7] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[7]);
          qacc[8] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[8]);
          qacc[9] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[9]);
          qacc[10] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[10]);
          qacc[11] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[11]);
          qacc[12] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[12]);
          qacc[13] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[13]);
          qacc[14] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[14]);
          qacc[15] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[15]);
          qacc[16] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[16]);
          qacc[17] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[17]);
          qacc[18] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[18]);
          qacc[19] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[19]);
          qacc[20] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[20]);
          qacc[21] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[21]);
          qacc[22] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[22]);
          qacc[23] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[23]);
          qacc[24] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[24]);
          qacc[25] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[25]);
          qacc[26] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[26]);
          qacc[27] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[27]);
          qacc[28] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[28]);
          qacc[29] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[29]);
          qacc[30] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[30]);
          qacc[31] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[31]);
          qacc[32] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[32]);
          qacc[33] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[33]);
          qacc[34] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[34]);
          qacc[35] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[35]);
          qacc[36] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[36]);
          qacc[37] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[37]);
          qacc[38] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[38]);
          qacc[39] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[39]);
          qacc[40] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[40]);
          qacc[41] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[41]);
          qacc[42] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[42]);
          qacc[43] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[43]);
          qacc[44] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[44]);
          qacc[45] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[45]);
          qacc[46] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[46]);
          qacc[47] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[47]);
          qacc[48] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[48]);
          qacc[49] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[49]);
          qacc[50] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[50]);
          qacc[51] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[51]);
          qacc[52] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[52]);
          qacc[53] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[53]);
          qacc[54] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[54]);
          qacc[55] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[55]);
          qacc[56] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[56]);
          qacc[57] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[57]);
          qacc[58] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[58]);
          qacc[59] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[59]);
          qacc[60] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[60]);
          qacc[61] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[61]);
          qacc[62] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[62]);
          qacc[63] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[63]);
        }

        c_dacc[0] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[0]);
        c_dacc[1] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[1]);
        c_dacc[2] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[2]);
        c_dacc[3] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[3]);
        c_dacc[4] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[4]);
        c_dacc[5] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[5]);
        c_dacc[6] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[6]);
        c_dacc[7] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[7]);
        c_dacc[8] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[8]);
        c_dacc[9] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[9]);
        c_dacc[10] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[10]);
        c_dacc[11] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[11]);
        c_dacc[12] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[12]);
        c_dacc[13] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[13]);
        c_dacc[14] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[14]);
        c_dacc[15] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[15]);
        c_dacc[16] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[16]);
        c_dacc[17] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[17]);
        c_dacc[18] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[18]);
        c_dacc[19] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[19]);
        c_dacc[20] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[20]);
        c_dacc[21] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[21]);
        c_dacc[22] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[22]);
        c_dacc[23] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[23]);
        c_dacc[24] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[24]);
        c_dacc[25] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[25]);
        c_dacc[26] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[26]);
        c_dacc[27] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[27]);
        c_dacc[28] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[28]);
        c_dacc[29] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[29]);
        c_dacc[30] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[30]);
        c_dacc[31] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[31]);
        c_dacc[32] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[32]);
        c_dacc[33] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[33]);
        c_dacc[34] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[34]);
        c_dacc[35] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[35]);
        c_dacc[36] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[36]);
        c_dacc[37] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[37]);
        c_dacc[38] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[38]);
        c_dacc[39] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[39]);
        c_dacc[40] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[40]);
        c_dacc[41] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[41]);
        c_dacc[42] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[42]);
        c_dacc[43] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[43]);
        c_dacc[44] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[44]);
        c_dacc[45] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[45]);
        c_dacc[46] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[46]);
        c_dacc[47] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[47]);
        c_dacc[48] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[48]);
        c_dacc[49] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[49]);
        c_dacc[50] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[50]);
        c_dacc[51] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[51]);
        c_dacc[52] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[52]);
        c_dacc[53] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[53]);
        c_dacc[54] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[54]);
        c_dacc[55] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[55]);
        c_dacc[56] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[56]);
        c_dacc[57] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[57]);
        c_dacc[58] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[58]);
        c_dacc[59] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[59]);
        c_dacc[60] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[60]);
        c_dacc[61] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[61]);
        c_dacc[62] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[62]);
        c_dacc[63] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[63]);
        __dtu_v_tvstda_bf16_dual(c_dacc[0], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[1], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[2], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[3], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[4], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[5], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[6], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[7], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[8], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[9], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[10], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[11], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[12], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[13], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[14], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[15], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[16], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[17], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[18], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[19], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[20], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[21], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[22], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[23], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[24], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[25], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[26], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[27], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[28], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[29], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[30], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[31], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[32], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[33], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[34], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[35], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[36], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[37], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[38], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[39], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[40], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[41], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[42], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[43], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[44], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[45], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[46], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[47], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[48], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[49], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[50], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[51], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[52], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[53], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[54], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[55], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[56], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[57], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[58], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[59], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[60], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[61], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[62], ot_base, ot_off0);
        __dtu_v_tvstda_bf16_dual(c_dacc[63], ot_base, ot_off1);

        vab_shift += 512;
        __dtu_c_movsr2vab_lv_s(vab_shift);
        __dtu_c_movsr2vab_lv_d(vab_shift);
        __dtu_c_movsr2vab_m_s1(vab_shift);
        __dtu_c_movsr2vab_m_d(vab_shift);
      }
      bs_dacc = __dtu_l_tvldqa_bf16_da(biast_base, biast_off1);
      c_dacc[0] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[1] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[2] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[3] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[4] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[5] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[6] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[7] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[8] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[9] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[10] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[11] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[12] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[13] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[14] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[15] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[16] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[17] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[18] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[19] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[20] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[21] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[22] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[23] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[24] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[25] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[26] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[27] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[28] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[29] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[30] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[31] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[32] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[33] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[34] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[35] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[36] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[37] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[38] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[39] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[40] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[41] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[42] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[43] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[44] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[45] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[46] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[47] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[48] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[49] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[50] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[51] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[52] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[53] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[54] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[55] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[56] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[57] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[58] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[59] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[60] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[61] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[62] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off0);
      c_dacc[63] = __dtu_l_tvldqa_bf16_da(bt_base, bt_off2);

      __dtu_c_movsr2vab_m_s2(0);
      qacc[0] = __dtu_m_mop_mul_f32_qa(qacc[0], qa_alpha);
      qacc[1] = __dtu_m_mop_mul_f32_qa(qacc[1], qa_alpha);
      qacc[2] = __dtu_m_mop_mul_f32_qa(qacc[2], qa_alpha);
      qacc[3] = __dtu_m_mop_mul_f32_qa(qacc[3], qa_alpha);
      qacc[4] = __dtu_m_mop_mul_f32_qa(qacc[4], qa_alpha);
      qacc[5] = __dtu_m_mop_mul_f32_qa(qacc[5], qa_alpha);
      qacc[6] = __dtu_m_mop_mul_f32_qa(qacc[6], qa_alpha);
      qacc[7] = __dtu_m_mop_mul_f32_qa(qacc[7], qa_alpha);
      qacc[8] = __dtu_m_mop_mul_f32_qa(qacc[8], qa_alpha);
      qacc[9] = __dtu_m_mop_mul_f32_qa(qacc[9], qa_alpha);
      qacc[10] = __dtu_m_mop_mul_f32_qa(qacc[10], qa_alpha);
      qacc[11] = __dtu_m_mop_mul_f32_qa(qacc[11], qa_alpha);
      qacc[12] = __dtu_m_mop_mul_f32_qa(qacc[12], qa_alpha);
      qacc[13] = __dtu_m_mop_mul_f32_qa(qacc[13], qa_alpha);
      qacc[14] = __dtu_m_mop_mul_f32_qa(qacc[14], qa_alpha);
      qacc[15] = __dtu_m_mop_mul_f32_qa(qacc[15], qa_alpha);
      qacc[16] = __dtu_m_mop_mul_f32_qa(qacc[16], qa_alpha);
      qacc[17] = __dtu_m_mop_mul_f32_qa(qacc[17], qa_alpha);
      qacc[18] = __dtu_m_mop_mul_f32_qa(qacc[18], qa_alpha);
      qacc[19] = __dtu_m_mop_mul_f32_qa(qacc[19], qa_alpha);
      qacc[20] = __dtu_m_mop_mul_f32_qa(qacc[20], qa_alpha);
      qacc[21] = __dtu_m_mop_mul_f32_qa(qacc[21], qa_alpha);
      qacc[22] = __dtu_m_mop_mul_f32_qa(qacc[22], qa_alpha);
      qacc[23] = __dtu_m_mop_mul_f32_qa(qacc[23], qa_alpha);
      qacc[24] = __dtu_m_mop_mul_f32_qa(qacc[24], qa_alpha);
      qacc[25] = __dtu_m_mop_mul_f32_qa(qacc[25], qa_alpha);
      qacc[26] = __dtu_m_mop_mul_f32_qa(qacc[26], qa_alpha);
      qacc[27] = __dtu_m_mop_mul_f32_qa(qacc[27], qa_alpha);
      qacc[28] = __dtu_m_mop_mul_f32_qa(qacc[28], qa_alpha);
      qacc[29] = __dtu_m_mop_mul_f32_qa(qacc[29], qa_alpha);
      qacc[30] = __dtu_m_mop_mul_f32_qa(qacc[30], qa_alpha);
      qacc[31] = __dtu_m_mop_mul_f32_qa(qacc[31], qa_alpha);
      qacc[32] = __dtu_m_mop_mul_f32_qa(qacc[32], qa_alpha);
      qacc[33] = __dtu_m_mop_mul_f32_qa(qacc[33], qa_alpha);
      qacc[34] = __dtu_m_mop_mul_f32_qa(qacc[34], qa_alpha);
      qacc[35] = __dtu_m_mop_mul_f32_qa(qacc[35], qa_alpha);
      qacc[36] = __dtu_m_mop_mul_f32_qa(qacc[36], qa_alpha);
      qacc[37] = __dtu_m_mop_mul_f32_qa(qacc[37], qa_alpha);
      qacc[38] = __dtu_m_mop_mul_f32_qa(qacc[38], qa_alpha);
      qacc[39] = __dtu_m_mop_mul_f32_qa(qacc[39], qa_alpha);
      qacc[40] = __dtu_m_mop_mul_f32_qa(qacc[40], qa_alpha);
      qacc[41] = __dtu_m_mop_mul_f32_qa(qacc[41], qa_alpha);
      qacc[42] = __dtu_m_mop_mul_f32_qa(qacc[42], qa_alpha);
      qacc[43] = __dtu_m_mop_mul_f32_qa(qacc[43], qa_alpha);
      qacc[44] = __dtu_m_mop_mul_f32_qa(qacc[44], qa_alpha);
      qacc[45] = __dtu_m_mop_mul_f32_qa(qacc[45], qa_alpha);
      qacc[46] = __dtu_m_mop_mul_f32_qa(qacc[46], qa_alpha);
      qacc[47] = __dtu_m_mop_mul_f32_qa(qacc[47], qa_alpha);
      qacc[48] = __dtu_m_mop_mul_f32_qa(qacc[48], qa_alpha);
      qacc[49] = __dtu_m_mop_mul_f32_qa(qacc[49], qa_alpha);
      qacc[50] = __dtu_m_mop_mul_f32_qa(qacc[50], qa_alpha);
      qacc[51] = __dtu_m_mop_mul_f32_qa(qacc[51], qa_alpha);
      qacc[52] = __dtu_m_mop_mul_f32_qa(qacc[52], qa_alpha);
      qacc[53] = __dtu_m_mop_mul_f32_qa(qacc[53], qa_alpha);
      qacc[54] = __dtu_m_mop_mul_f32_qa(qacc[54], qa_alpha);
      qacc[55] = __dtu_m_mop_mul_f32_qa(qacc[55], qa_alpha);
      qacc[56] = __dtu_m_mop_mul_f32_qa(qacc[56], qa_alpha);
      qacc[57] = __dtu_m_mop_mul_f32_qa(qacc[57], qa_alpha);
      qacc[58] = __dtu_m_mop_mul_f32_qa(qacc[58], qa_alpha);
      qacc[59] = __dtu_m_mop_mul_f32_qa(qacc[59], qa_alpha);
      qacc[60] = __dtu_m_mop_mul_f32_qa(qacc[60], qa_alpha);
      qacc[61] = __dtu_m_mop_mul_f32_qa(qacc[61], qa_alpha);
      qacc[62] = __dtu_m_mop_mul_f32_qa(qacc[62], qa_alpha);
      qacc[63] = __dtu_m_mop_mul_f32_qa(qacc[63], qa_alpha);

      qacc[0] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[0], qa_beta, qacc[0]);
      qacc[1] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[1], qa_beta, qacc[1]);
      qacc[2] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[2], qa_beta, qacc[2]);
      qacc[3] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[3], qa_beta, qacc[3]);
      qacc[4] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[4], qa_beta, qacc[4]);
      qacc[5] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[5], qa_beta, qacc[5]);
      qacc[6] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[6], qa_beta, qacc[6]);
      qacc[7] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[7], qa_beta, qacc[7]);
      qacc[8] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[8], qa_beta, qacc[8]);
      qacc[9] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[9], qa_beta, qacc[9]);
      qacc[10] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[10], qa_beta, qacc[10]);
      qacc[11] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[11], qa_beta, qacc[11]);
      qacc[12] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[12], qa_beta, qacc[12]);
      qacc[13] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[13], qa_beta, qacc[13]);
      qacc[14] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[14], qa_beta, qacc[14]);
      qacc[15] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[15], qa_beta, qacc[15]);
      qacc[16] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[16], qa_beta, qacc[16]);
      qacc[17] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[17], qa_beta, qacc[17]);
      qacc[18] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[18], qa_beta, qacc[18]);
      qacc[19] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[19], qa_beta, qacc[19]);
      qacc[20] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[20], qa_beta, qacc[20]);
      qacc[21] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[21], qa_beta, qacc[21]);
      qacc[22] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[22], qa_beta, qacc[22]);
      qacc[23] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[23], qa_beta, qacc[23]);
      qacc[24] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[24], qa_beta, qacc[24]);
      qacc[25] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[25], qa_beta, qacc[25]);
      qacc[26] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[26], qa_beta, qacc[26]);
      qacc[27] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[27], qa_beta, qacc[27]);
      qacc[28] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[28], qa_beta, qacc[28]);
      qacc[29] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[29], qa_beta, qacc[29]);
      qacc[30] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[30], qa_beta, qacc[30]);
      qacc[31] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[31], qa_beta, qacc[31]);
      qacc[32] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[32], qa_beta, qacc[32]);
      qacc[33] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[33], qa_beta, qacc[33]);
      qacc[34] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[34], qa_beta, qacc[34]);
      qacc[35] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[35], qa_beta, qacc[35]);
      qacc[36] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[36], qa_beta, qacc[36]);
      qacc[37] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[37], qa_beta, qacc[37]);
      qacc[38] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[38], qa_beta, qacc[38]);
      qacc[39] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[39], qa_beta, qacc[39]);
      qacc[40] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[40], qa_beta, qacc[40]);
      qacc[41] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[41], qa_beta, qacc[41]);
      qacc[42] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[42], qa_beta, qacc[42]);
      qacc[43] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[43], qa_beta, qacc[43]);
      qacc[44] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[44], qa_beta, qacc[44]);
      qacc[45] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[45], qa_beta, qacc[45]);
      qacc[46] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[46], qa_beta, qacc[46]);
      qacc[47] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[47], qa_beta, qacc[47]);
      qacc[48] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[48], qa_beta, qacc[48]);
      qacc[49] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[49], qa_beta, qacc[49]);
      qacc[50] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[50], qa_beta, qacc[50]);
      qacc[51] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[51], qa_beta, qacc[51]);
      qacc[52] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[52], qa_beta, qacc[52]);
      qacc[53] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[53], qa_beta, qacc[53]);
      qacc[54] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[54], qa_beta, qacc[54]);
      qacc[55] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[55], qa_beta, qacc[55]);
      qacc[56] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[56], qa_beta, qacc[56]);
      qacc[57] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[57], qa_beta, qacc[57]);
      qacc[58] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[58], qa_beta, qacc[58]);
      qacc[59] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[59], qa_beta, qacc[59]);
      qacc[60] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[60], qa_beta, qacc[60]);
      qacc[61] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[61], qa_beta, qacc[61]);
      qacc[62] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[62], qa_beta, qacc[62]);
      qacc[63] = __dtu_m_mop_mac_f32mix_bf16_da(c_dacc[63], qa_beta, qacc[63]);
      // add bias
      if (bias_en == 1) {
        qacc[0] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[0]);
        qacc[1] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[1]);
        qacc[2] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[2]);
        qacc[3] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[3]);
        qacc[4] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[4]);
        qacc[5] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[5]);
        qacc[6] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[6]);
        qacc[7] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[7]);
        qacc[8] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[8]);
        qacc[9] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[9]);
        qacc[10] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[10]);
        qacc[11] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[11]);
        qacc[12] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[12]);
        qacc[13] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[13]);
        qacc[14] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[14]);
        qacc[15] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[15]);
        qacc[16] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[16]);
        qacc[17] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[17]);
        qacc[18] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[18]);
        qacc[19] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[19]);
        qacc[20] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[20]);
        qacc[21] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[21]);
        qacc[22] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[22]);
        qacc[23] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[23]);
        qacc[24] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[24]);
        qacc[25] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[25]);
        qacc[26] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[26]);
        qacc[27] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[27]);
        qacc[28] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[28]);
        qacc[29] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[29]);
        qacc[30] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[30]);
        qacc[31] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[31]);
        qacc[32] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[32]);
        qacc[33] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[33]);
        qacc[34] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[34]);
        qacc[35] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[35]);
        qacc[36] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[36]);
        qacc[37] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[37]);
        qacc[38] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[38]);
        qacc[39] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[39]);
        qacc[40] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[40]);
        qacc[41] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[41]);
        qacc[42] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[42]);
        qacc[43] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[43]);
        qacc[44] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[44]);
        qacc[45] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[45]);
        qacc[46] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[46]);
        qacc[47] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[47]);
        qacc[48] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[48]);
        qacc[49] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[49]);
        qacc[50] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[50]);
        qacc[51] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[51]);
        qacc[52] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[52]);
        qacc[53] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[53]);
        qacc[54] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[54]);
        qacc[55] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[55]);
        qacc[56] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[56]);
        qacc[57] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[57]);
        qacc[58] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[58]);
        qacc[59] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[59]);
        qacc[60] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[60]);
        qacc[61] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[61]);
        qacc[62] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[62]);
        qacc[63] = __dtu_m_mop_mac_f32mix_bf16_da(bs_dacc, qa_bias, qacc[63]);
      }

      c_dacc[0] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[0]);
      c_dacc[1] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[1]);
      c_dacc[2] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[2]);
      c_dacc[3] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[3]);
      c_dacc[4] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[4]);
      c_dacc[5] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[5]);
      c_dacc[6] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[6]);
      c_dacc[7] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[7]);
      c_dacc[8] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[8]);
      c_dacc[9] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[9]);
      c_dacc[10] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[10]);
      c_dacc[11] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[11]);
      c_dacc[12] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[12]);
      c_dacc[13] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[13]);
      c_dacc[14] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[14]);
      c_dacc[15] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[15]);
      c_dacc[16] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[16]);
      c_dacc[17] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[17]);
      c_dacc[18] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[18]);
      c_dacc[19] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[19]);
      c_dacc[20] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[20]);
      c_dacc[21] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[21]);
      c_dacc[22] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[22]);
      c_dacc[23] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[23]);
      c_dacc[24] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[24]);
      c_dacc[25] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[25]);
      c_dacc[26] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[26]);
      c_dacc[27] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[27]);
      c_dacc[28] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[28]);
      c_dacc[29] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[29]);
      c_dacc[30] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[30]);
      c_dacc[31] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[31]);
      c_dacc[32] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[32]);
      c_dacc[33] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[33]);
      c_dacc[34] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[34]);
      c_dacc[35] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[35]);
      c_dacc[36] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[36]);
      c_dacc[37] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[37]);
      c_dacc[38] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[38]);
      c_dacc[39] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[39]);
      c_dacc[40] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[40]);
      c_dacc[41] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[41]);
      c_dacc[42] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[42]);
      c_dacc[43] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[43]);
      c_dacc[44] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[44]);
      c_dacc[45] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[45]);
      c_dacc[46] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[46]);
      c_dacc[47] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[47]);
      c_dacc[48] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[48]);
      c_dacc[49] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[49]);
      c_dacc[50] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[50]);
      c_dacc[51] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[51]);
      c_dacc[52] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[52]);
      c_dacc[53] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[53]);
      c_dacc[54] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[54]);
      c_dacc[55] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[55]);
      c_dacc[56] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[56]);
      c_dacc[57] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[57]);
      c_dacc[58] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[58]);
      c_dacc[59] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[59]);
      c_dacc[60] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[60]);
      c_dacc[61] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[61]);
      c_dacc[62] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[62]);
      c_dacc[63] = __dtu_m_mop_cvt_qa_rne_clamp_f32_bf16(qacc[63]);

      __dtu_v_tvstda_bf16_dual(c_dacc[0], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[1], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[2], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[3], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[4], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[5], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[6], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[7], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[8], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[9], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[10], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[11], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[12], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[13], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[14], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[15], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[16], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[17], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[18], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[19], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[20], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[21], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[22], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[23], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[24], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[25], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[26], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[27], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[28], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[29], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[30], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[31], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[32], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[33], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[34], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[35], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[36], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[37], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[38], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[39], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[40], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[41], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[42], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[43], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[44], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[45], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[46], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[47], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[48], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[49], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[50], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[51], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[52], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[53], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[54], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[55], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[56], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[57], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[58], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[59], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[60], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[61], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[62], ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(c_dacc[63], ot_base, ot_off2);
      vab_shift += 512;
      __dtu_c_movsr2vab_lv_s(vab_shift);
      __dtu_c_movsr2vab_lv_d(vab_shift);
      __dtu_c_movsr2vab_m_s1(vab_shift);
      __dtu_c_movsr2vab_m_d(vab_shift);
    }
  }

#endif
}


// tune->sip_m = 32;
// tune->sip_k = 64;
// tune->sip_n = 128;
__attribute__((device)) extern "C" void c_func_gemm_general_int8(
  int a_addr, int b_addr, int c_addr, int M, int N, int K, int nacc_flag,
  int stroe_flag, int alpha_enable, int beta_enable, float alpha,
  float beta, int bias_en, int bias_addr, int cur_n) {
#if __GCU_ARCH__ >= 300
  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_lv_d(0);
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_s2(0);
  __dtu_c_movsr2vab_m_d(0);
  smr_t smr0, smr1;
  v64i8 vr0, vr1, vr2, vr3, vr4, vr5, vr6, vr7, vr8, vr9, vr10, vr11, vr12,
      vr13, vr14, vr15, vr16, vr17, vr18, vr19, vr20, vr21, vr22, vr23, vr24,
      vr25, vr26, vr27, vr28, vr29, vr30, vr31;
  v16f32 vr_alpha = __dtu_s_movr2vr_dup_f32(alpha);

  va16i32x4 qacc[32];
  va16f32x4 f_qacc[32];
  va16i8 vacc[32];

  va16f32 va_alpha0, va_alpha1, va_alpha2, va_alpha3;
  va_alpha0 = __dtu_l_movvr2va(vr_alpha);
  va_alpha1 = __dtu_l_movvr2va(vr_alpha);
  va_alpha2 = __dtu_l_movvr2va(vr_alpha);
  va_alpha3 = __dtu_l_movvr2va(vr_alpha);
  va16f32x4 qa_alpha =
      __dtu_insertva2qa_f32(va_alpha0, va_alpha1, va_alpha2, va_alpha3);

  v16f32 vr_scale = __dtu_s_movr2vr_dup_f32(beta);
  va16f32 vacc_beta0, vacc_beta1, vacc_beta2, vacc_beta3;
  vacc_beta0 = __dtu_l_movvr2va(vr_scale);
  vacc_beta1 = __dtu_l_movvr2va(vr_scale);
  vacc_beta2 = __dtu_l_movvr2va(vr_scale);
  vacc_beta3 = __dtu_l_movvr2va(vr_scale);
  va16f32x4 qa_beta =
      __dtu_insertva2qa_f32(vacc_beta0, vacc_beta1, vacc_beta2, vacc_beta3);

  auto k_unit = K >> 6;
  auto n_unit = N >> 6;
  auto n2_unit = N >> 7;
  auto on_unit = N >> 6;
  // vpt parallel in rhs
  int lt_addr = a_addr >> 6;
  int rt_addr = b_addr >> 6;
  int ot_addr = c_addr >> 6;
  int offset = 0;
  tar_t lt_base = __dtu_c_movsr2targ(TAR_ADDR_WARP(lt_addr, 0));
  offset = TAR_OFF_WARP(k_unit, k_unit);
  tar_t lt_off0 = __dtu_c_movsr2tari(offset, lt_base);
  offset = TAR_OFF_WARP(1 - 31 * k_unit, 1 - 31 * k_unit);  // next k
  tar_t lt_off1 = __dtu_c_movsr2tari(offset, lt_base);
  offset = TAR_OFF_WARP(1 - 32 * k_unit, 1 - 32 * k_unit);  //  new n
  tar_t lt_off2 = __dtu_c_movsr2tari(offset, lt_base);
  offset = TAR_OFF_WARP(1, 1);  // end k end n new m
  tar_t lt_off3 = __dtu_c_movsr2tari(offset, lt_base);

  tar_t rt_base = __dtu_c_movsr2targ((rt_addr) | ((rt_addr) + 1) << 16);
  offset = TAR_OFF_WARP(n_unit, n_unit);  // 2 row
  tar_t rt_off0 = __dtu_c_movsr2tari(offset, rt_base);
  offset = TAR_OFF_WARP(1 - (K >> 1) * n_unit, 1 - (K >> 1) * n_unit);
  tar_t rt_off1 = __dtu_c_movsr2tari(offset, rt_base);  // new n
  offset = TAR_OFF_WARP(1 + n2_unit - ((K >> 1) + 1) * n_unit,
                        1 + n2_unit - ((K >> 1) + 1) * n_unit);
  tar_t rt_off2 = __dtu_c_movsr2tari(offset, rt_base);  // new m

  tar_t ot_base = __dtu_c_movsr2targ((ot_addr) | ((ot_addr) + 1) << 16);
  offset = TAR_OFF_WARP(on_unit, on_unit);
  tar_t ot_off0 = __dtu_c_movsr2tari(offset, ot_base);
  offset = TAR_OFF_WARP(2 - 31 * on_unit, 2 - 31 * on_unit);
  tar_t ot_off1 = __dtu_c_movsr2tari(offset, ot_base);  // new n
  offset = TAR_OFF_WARP(2, 2);
  tar_t ot_off2 = __dtu_c_movsr2tari(offset, ot_base);  // new m

  tar_t bt_base = __dtu_c_movsr2targ((ot_addr) | ((ot_addr) + 1) << 16);
  offset = TAR_OFF_WARP(on_unit, on_unit);
  tar_t bt_off0 = __dtu_c_movsr2tari(offset, bt_base);
  offset = TAR_OFF_WARP(2 - 31 * on_unit, 2 - 31 * on_unit);
  tar_t bt_off1 = __dtu_c_movsr2tari(offset, bt_base);  // new n
  offset = TAR_OFF_WARP(2, 2);
  tar_t bt_off2 = __dtu_c_movsr2tari(offset, bt_base);  // new m

  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 0);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 1);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 2);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 3);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 4);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 5);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 6);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 7);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 8);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 9);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 10);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 11);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 12);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 13);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 14);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 15);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 16);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 17);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 18);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 19);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 20);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 21);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 22);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 23);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 24);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 25);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 26);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 27);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 28);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 29);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 30);
  smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 31);
  // m0k0
  vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
  vr31 = __dtu_s_tvld_itar(lt_base, lt_off1);

  int naccovr = 0x10001;
  if (nacc_flag ^ 1) {
    naccovr = 0x1;
  }
  __dtu_c_movsr2naccovr(naccovr);
  __dtu_c_movsr2vab_m_s2(0);
  int vab_shift = 0;
// fp16 vmm2 mode17: [32, 64] * [64, 128] = [64, 128]
#pragma clang loop unroll(full)
  for (int m = 0; m < M; m += 32) {
    for (int n = 0; n < N - 128; n += 128) {  // VPT PARA DIM
      __dtu_c_movsr2naccovr(naccovr);
      for (int k = 0; k < K - 128; k += 128) {
        // m0k0 * smr0
        qacc[0] = __dtu_m_vmm2_mode19_s8_acc0(qacc[0], vr0, smr0);
        qacc[1] = __dtu_m_vmm2_mode19_s8_acc0(qacc[1], vr1, smr0);
        qacc[2] = __dtu_m_vmm2_mode19_s8_acc0(qacc[2], vr2, smr0);
        qacc[3] = __dtu_m_vmm2_mode19_s8_acc0(qacc[3], vr3, smr0);
        qacc[4] = __dtu_m_vmm2_mode19_s8_acc0(qacc[4], vr4, smr0);
        qacc[5] = __dtu_m_vmm2_mode19_s8_acc0(qacc[5], vr5, smr0);
        qacc[6] = __dtu_m_vmm2_mode19_s8_acc0(qacc[6], vr6, smr0);
        qacc[7] = __dtu_m_vmm2_mode19_s8_acc0(qacc[7], vr7, smr0);
        qacc[8] = __dtu_m_vmm2_mode19_s8_acc0(qacc[8], vr8, smr0);
        qacc[9] = __dtu_m_vmm2_mode19_s8_acc0(qacc[9], vr9, smr0);
        qacc[10] = __dtu_m_vmm2_mode19_s8_acc0(qacc[10], vr10, smr0);
        qacc[11] = __dtu_m_vmm2_mode19_s8_acc0(qacc[11], vr11, smr0);
        qacc[12] = __dtu_m_vmm2_mode19_s8_acc0(qacc[12], vr12, smr0);
        qacc[13] = __dtu_m_vmm2_mode19_s8_acc0(qacc[13], vr13, smr0);
        qacc[14] = __dtu_m_vmm2_mode19_s8_acc0(qacc[14], vr14, smr0);
        qacc[15] = __dtu_m_vmm2_mode19_s8_acc0(qacc[15], vr15, smr0);
        qacc[16] = __dtu_m_vmm2_mode19_s8_acc0(qacc[16], vr16, smr0);
        qacc[17] = __dtu_m_vmm2_mode19_s8_acc0(qacc[17], vr17, smr0);
        qacc[18] = __dtu_m_vmm2_mode19_s8_acc0(qacc[18], vr18, smr0);
        qacc[19] = __dtu_m_vmm2_mode19_s8_acc0(qacc[19], vr19, smr0);
        qacc[20] = __dtu_m_vmm2_mode19_s8_acc0(qacc[20], vr20, smr0);
        qacc[21] = __dtu_m_vmm2_mode19_s8_acc0(qacc[21], vr21, smr0);
        qacc[22] = __dtu_m_vmm2_mode19_s8_acc0(qacc[22], vr22, smr0);
        qacc[23] = __dtu_m_vmm2_mode19_s8_acc0(qacc[23], vr23, smr0);
        qacc[24] = __dtu_m_vmm2_mode19_s8_acc0(qacc[24], vr24, smr0);
        qacc[25] = __dtu_m_vmm2_mode19_s8_acc0(qacc[25], vr25, smr0);
        qacc[26] = __dtu_m_vmm2_mode19_s8_acc0(qacc[26], vr26, smr0);
        qacc[27] = __dtu_m_vmm2_mode19_s8_acc0(qacc[27], vr27, smr0);
        qacc[28] = __dtu_m_vmm2_mode19_s8_acc0(qacc[28], vr28, smr0);
        qacc[29] = __dtu_m_vmm2_mode19_s8_acc0(qacc[29], vr29, smr0);
        qacc[30] = __dtu_m_vmm2_mode19_s8_acc0(qacc[30], vr30, smr0);
        qacc[31] = __dtu_m_vmm2_mode19_s8_acc0(qacc[31], vr31, smr0);
        // m0k1
        vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr31 = __dtu_s_tvld_itar(lt_base, lt_off1);
        // smr1
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 0);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 1);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 2);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 3);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 4);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 5);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 6);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 7);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 8);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 9);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 10);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 11);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 12);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 13);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 14);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 15);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 16);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 17);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 18);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 19);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 20);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 21);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 22);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 23);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 24);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 25);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 26);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 27);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 28);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 29);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 30);
        smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 31);
        __dtu_c_movsr2naccovr(0x1);
        // m0k1 * smr1
        qacc[0] = __dtu_m_vmm2_mode19_s8_acc0(qacc[0], vr0, smr1);
        qacc[1] = __dtu_m_vmm2_mode19_s8_acc0(qacc[1], vr1, smr1);
        qacc[2] = __dtu_m_vmm2_mode19_s8_acc0(qacc[2], vr2, smr1);
        qacc[3] = __dtu_m_vmm2_mode19_s8_acc0(qacc[3], vr3, smr1);
        qacc[4] = __dtu_m_vmm2_mode19_s8_acc0(qacc[4], vr4, smr1);
        qacc[5] = __dtu_m_vmm2_mode19_s8_acc0(qacc[5], vr5, smr1);
        qacc[6] = __dtu_m_vmm2_mode19_s8_acc0(qacc[6], vr6, smr1);
        qacc[7] = __dtu_m_vmm2_mode19_s8_acc0(qacc[7], vr7, smr1);
        qacc[8] = __dtu_m_vmm2_mode19_s8_acc0(qacc[8], vr8, smr1);
        qacc[9] = __dtu_m_vmm2_mode19_s8_acc0(qacc[9], vr9, smr1);
        qacc[10] = __dtu_m_vmm2_mode19_s8_acc0(qacc[10], vr10, smr1);
        qacc[11] = __dtu_m_vmm2_mode19_s8_acc0(qacc[11], vr11, smr1);
        qacc[12] = __dtu_m_vmm2_mode19_s8_acc0(qacc[12], vr12, smr1);
        qacc[13] = __dtu_m_vmm2_mode19_s8_acc0(qacc[13], vr13, smr1);
        qacc[14] = __dtu_m_vmm2_mode19_s8_acc0(qacc[14], vr14, smr1);
        qacc[15] = __dtu_m_vmm2_mode19_s8_acc0(qacc[15], vr15, smr1);
        qacc[16] = __dtu_m_vmm2_mode19_s8_acc0(qacc[16], vr16, smr1);
        qacc[17] = __dtu_m_vmm2_mode19_s8_acc0(qacc[17], vr17, smr1);
        qacc[18] = __dtu_m_vmm2_mode19_s8_acc0(qacc[18], vr18, smr1);
        qacc[19] = __dtu_m_vmm2_mode19_s8_acc0(qacc[19], vr19, smr1);
        qacc[20] = __dtu_m_vmm2_mode19_s8_acc0(qacc[20], vr20, smr1);
        qacc[21] = __dtu_m_vmm2_mode19_s8_acc0(qacc[21], vr21, smr1);
        qacc[22] = __dtu_m_vmm2_mode19_s8_acc0(qacc[22], vr22, smr1);
        qacc[23] = __dtu_m_vmm2_mode19_s8_acc0(qacc[23], vr23, smr1);
        qacc[24] = __dtu_m_vmm2_mode19_s8_acc0(qacc[24], vr24, smr1);
        qacc[25] = __dtu_m_vmm2_mode19_s8_acc0(qacc[25], vr25, smr1);
        qacc[26] = __dtu_m_vmm2_mode19_s8_acc0(qacc[26], vr26, smr1);
        qacc[27] = __dtu_m_vmm2_mode19_s8_acc0(qacc[27], vr27, smr1);
        qacc[28] = __dtu_m_vmm2_mode19_s8_acc0(qacc[28], vr28, smr1);
        qacc[29] = __dtu_m_vmm2_mode19_s8_acc0(qacc[29], vr29, smr1);
        qacc[30] = __dtu_m_vmm2_mode19_s8_acc0(qacc[30], vr30, smr1);
        qacc[31] = __dtu_m_vmm2_mode19_s8_acc0(qacc[31], vr31, smr1);
        // next k unit smr0
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 0);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 1);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 2);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 3);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 4);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 5);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 6);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 7);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 8);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 9);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 10);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 11);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 12);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 13);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 14);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 15);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 16);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 17);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 18);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 19);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 20);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 21);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 22);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 23);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 24);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 25);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 26);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 27);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 28);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 29);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 30);
        smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 31);
        // next k unit m0k0
        vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
        vr31 = __dtu_s_tvld_itar(lt_base, lt_off1);
      }  // end kcout-1
      // last k unit
      // m0k0 * smr0
      qacc[0] = __dtu_m_vmm2_mode19_s8_acc0(qacc[0], vr0, smr0);
      qacc[1] = __dtu_m_vmm2_mode19_s8_acc0(qacc[1], vr1, smr0);
      qacc[2] = __dtu_m_vmm2_mode19_s8_acc0(qacc[2], vr2, smr0);
      qacc[3] = __dtu_m_vmm2_mode19_s8_acc0(qacc[3], vr3, smr0);
      qacc[4] = __dtu_m_vmm2_mode19_s8_acc0(qacc[4], vr4, smr0);
      qacc[5] = __dtu_m_vmm2_mode19_s8_acc0(qacc[5], vr5, smr0);
      qacc[6] = __dtu_m_vmm2_mode19_s8_acc0(qacc[6], vr6, smr0);
      qacc[7] = __dtu_m_vmm2_mode19_s8_acc0(qacc[7], vr7, smr0);
      qacc[8] = __dtu_m_vmm2_mode19_s8_acc0(qacc[8], vr8, smr0);
      qacc[9] = __dtu_m_vmm2_mode19_s8_acc0(qacc[9], vr9, smr0);
      qacc[10] = __dtu_m_vmm2_mode19_s8_acc0(qacc[10], vr10, smr0);
      qacc[11] = __dtu_m_vmm2_mode19_s8_acc0(qacc[11], vr11, smr0);
      qacc[12] = __dtu_m_vmm2_mode19_s8_acc0(qacc[12], vr12, smr0);
      qacc[13] = __dtu_m_vmm2_mode19_s8_acc0(qacc[13], vr13, smr0);
      qacc[14] = __dtu_m_vmm2_mode19_s8_acc0(qacc[14], vr14, smr0);
      qacc[15] = __dtu_m_vmm2_mode19_s8_acc0(qacc[15], vr15, smr0);
      qacc[16] = __dtu_m_vmm2_mode19_s8_acc0(qacc[16], vr16, smr0);
      qacc[17] = __dtu_m_vmm2_mode19_s8_acc0(qacc[17], vr17, smr0);
      qacc[18] = __dtu_m_vmm2_mode19_s8_acc0(qacc[18], vr18, smr0);
      qacc[19] = __dtu_m_vmm2_mode19_s8_acc0(qacc[19], vr19, smr0);
      qacc[20] = __dtu_m_vmm2_mode19_s8_acc0(qacc[20], vr20, smr0);
      qacc[21] = __dtu_m_vmm2_mode19_s8_acc0(qacc[21], vr21, smr0);
      qacc[22] = __dtu_m_vmm2_mode19_s8_acc0(qacc[22], vr22, smr0);
      qacc[23] = __dtu_m_vmm2_mode19_s8_acc0(qacc[23], vr23, smr0);
      qacc[24] = __dtu_m_vmm2_mode19_s8_acc0(qacc[24], vr24, smr0);
      qacc[25] = __dtu_m_vmm2_mode19_s8_acc0(qacc[25], vr25, smr0);
      qacc[26] = __dtu_m_vmm2_mode19_s8_acc0(qacc[26], vr26, smr0);
      qacc[27] = __dtu_m_vmm2_mode19_s8_acc0(qacc[27], vr27, smr0);
      qacc[28] = __dtu_m_vmm2_mode19_s8_acc0(qacc[28], vr28, smr0);
      qacc[29] = __dtu_m_vmm2_mode19_s8_acc0(qacc[29], vr29, smr0);
      qacc[30] = __dtu_m_vmm2_mode19_s8_acc0(qacc[30], vr30, smr0);
      qacc[31] = __dtu_m_vmm2_mode19_s8_acc0(qacc[31], vr31, smr0);
      // m0k1
      vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr31 = __dtu_s_tvld_itar(lt_base, lt_off2);  // end k new n
      // smr1
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 0);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 1);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 2);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 3);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 4);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 5);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 6);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 7);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 8);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 9);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 10);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 11);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 12);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 13);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 14);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 15);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 16);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 17);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 18);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 19);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 20);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 21);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 22);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 23);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 24);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 25);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 26);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 27);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 28);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 29);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 30);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 31);
      __dtu_c_movsr2naccovr(0x1);
      // m0k1 * smr1
      qacc[0] = __dtu_m_vmm2_mode19_s8_acc0(qacc[0], vr0, smr1);
      qacc[1] = __dtu_m_vmm2_mode19_s8_acc0(qacc[1], vr1, smr1);
      qacc[2] = __dtu_m_vmm2_mode19_s8_acc0(qacc[2], vr2, smr1);
      qacc[3] = __dtu_m_vmm2_mode19_s8_acc0(qacc[3], vr3, smr1);
      qacc[4] = __dtu_m_vmm2_mode19_s8_acc0(qacc[4], vr4, smr1);
      qacc[5] = __dtu_m_vmm2_mode19_s8_acc0(qacc[5], vr5, smr1);
      qacc[6] = __dtu_m_vmm2_mode19_s8_acc0(qacc[6], vr6, smr1);
      qacc[7] = __dtu_m_vmm2_mode19_s8_acc0(qacc[7], vr7, smr1);
      qacc[8] = __dtu_m_vmm2_mode19_s8_acc0(qacc[8], vr8, smr1);
      qacc[9] = __dtu_m_vmm2_mode19_s8_acc0(qacc[9], vr9, smr1);
      qacc[10] = __dtu_m_vmm2_mode19_s8_acc0(qacc[10], vr10, smr1);
      qacc[11] = __dtu_m_vmm2_mode19_s8_acc0(qacc[11], vr11, smr1);
      qacc[12] = __dtu_m_vmm2_mode19_s8_acc0(qacc[12], vr12, smr1);
      qacc[13] = __dtu_m_vmm2_mode19_s8_acc0(qacc[13], vr13, smr1);
      qacc[14] = __dtu_m_vmm2_mode19_s8_acc0(qacc[14], vr14, smr1);
      qacc[15] = __dtu_m_vmm2_mode19_s8_acc0(qacc[15], vr15, smr1);
      qacc[16] = __dtu_m_vmm2_mode19_s8_acc0(qacc[16], vr16, smr1);
      qacc[17] = __dtu_m_vmm2_mode19_s8_acc0(qacc[17], vr17, smr1);
      qacc[18] = __dtu_m_vmm2_mode19_s8_acc0(qacc[18], vr18, smr1);
      qacc[19] = __dtu_m_vmm2_mode19_s8_acc0(qacc[19], vr19, smr1);
      qacc[20] = __dtu_m_vmm2_mode19_s8_acc0(qacc[20], vr20, smr1);
      qacc[21] = __dtu_m_vmm2_mode19_s8_acc0(qacc[21], vr21, smr1);
      qacc[22] = __dtu_m_vmm2_mode19_s8_acc0(qacc[22], vr22, smr1);
      qacc[23] = __dtu_m_vmm2_mode19_s8_acc0(qacc[23], vr23, smr1);
      qacc[24] = __dtu_m_vmm2_mode19_s8_acc0(qacc[24], vr24, smr1);
      qacc[25] = __dtu_m_vmm2_mode19_s8_acc0(qacc[25], vr25, smr1);
      qacc[26] = __dtu_m_vmm2_mode19_s8_acc0(qacc[26], vr26, smr1);
      qacc[27] = __dtu_m_vmm2_mode19_s8_acc0(qacc[27], vr27, smr1);
      qacc[28] = __dtu_m_vmm2_mode19_s8_acc0(qacc[28], vr28, smr1);
      qacc[29] = __dtu_m_vmm2_mode19_s8_acc0(qacc[29], vr29, smr1);
      qacc[30] = __dtu_m_vmm2_mode19_s8_acc0(qacc[30], vr30, smr1);
      qacc[31] = __dtu_m_vmm2_mode19_s8_acc0(qacc[31], vr31, smr1);
      // move to n unit smr0
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off1, 0);
      // next n unit smr0
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 0);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 1);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 2);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 3);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 4);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 5);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 6);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 7);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 8);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 9);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 10);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 11);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 12);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 13);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 14);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 15);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 16);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 17);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 18);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 19);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 20);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 21);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 22);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 23);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 24);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 25);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 26);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 27);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 28);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 29);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 30);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 31);
      // next n unit m0k0
      vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr31 = __dtu_s_tvld_itar(lt_base, lt_off1);

      vab_shift += 512;
      __dtu_c_movsr2vab_m_s1(vab_shift);
      __dtu_c_movsr2vab_m_d(vab_shift);
    }  // end ncount-1
    __dtu_c_movsr2naccovr(naccovr);
    for (int k = 0; k < K - 128; k += 128) {
      // m0k0 * smr0
      qacc[0] = __dtu_m_vmm2_mode19_s8_acc0(qacc[0], vr0, smr0);
      qacc[1] = __dtu_m_vmm2_mode19_s8_acc0(qacc[1], vr1, smr0);
      qacc[2] = __dtu_m_vmm2_mode19_s8_acc0(qacc[2], vr2, smr0);
      qacc[3] = __dtu_m_vmm2_mode19_s8_acc0(qacc[3], vr3, smr0);
      qacc[4] = __dtu_m_vmm2_mode19_s8_acc0(qacc[4], vr4, smr0);
      qacc[5] = __dtu_m_vmm2_mode19_s8_acc0(qacc[5], vr5, smr0);
      qacc[6] = __dtu_m_vmm2_mode19_s8_acc0(qacc[6], vr6, smr0);
      qacc[7] = __dtu_m_vmm2_mode19_s8_acc0(qacc[7], vr7, smr0);
      qacc[8] = __dtu_m_vmm2_mode19_s8_acc0(qacc[8], vr8, smr0);
      qacc[9] = __dtu_m_vmm2_mode19_s8_acc0(qacc[9], vr9, smr0);
      qacc[10] = __dtu_m_vmm2_mode19_s8_acc0(qacc[10], vr10, smr0);
      qacc[11] = __dtu_m_vmm2_mode19_s8_acc0(qacc[11], vr11, smr0);
      qacc[12] = __dtu_m_vmm2_mode19_s8_acc0(qacc[12], vr12, smr0);
      qacc[13] = __dtu_m_vmm2_mode19_s8_acc0(qacc[13], vr13, smr0);
      qacc[14] = __dtu_m_vmm2_mode19_s8_acc0(qacc[14], vr14, smr0);
      qacc[15] = __dtu_m_vmm2_mode19_s8_acc0(qacc[15], vr15, smr0);
      qacc[16] = __dtu_m_vmm2_mode19_s8_acc0(qacc[16], vr16, smr0);
      qacc[17] = __dtu_m_vmm2_mode19_s8_acc0(qacc[17], vr17, smr0);
      qacc[18] = __dtu_m_vmm2_mode19_s8_acc0(qacc[18], vr18, smr0);
      qacc[19] = __dtu_m_vmm2_mode19_s8_acc0(qacc[19], vr19, smr0);
      qacc[20] = __dtu_m_vmm2_mode19_s8_acc0(qacc[20], vr20, smr0);
      qacc[21] = __dtu_m_vmm2_mode19_s8_acc0(qacc[21], vr21, smr0);
      qacc[22] = __dtu_m_vmm2_mode19_s8_acc0(qacc[22], vr22, smr0);
      qacc[23] = __dtu_m_vmm2_mode19_s8_acc0(qacc[23], vr23, smr0);
      qacc[24] = __dtu_m_vmm2_mode19_s8_acc0(qacc[24], vr24, smr0);
      qacc[25] = __dtu_m_vmm2_mode19_s8_acc0(qacc[25], vr25, smr0);
      qacc[26] = __dtu_m_vmm2_mode19_s8_acc0(qacc[26], vr26, smr0);
      qacc[27] = __dtu_m_vmm2_mode19_s8_acc0(qacc[27], vr27, smr0);
      qacc[28] = __dtu_m_vmm2_mode19_s8_acc0(qacc[28], vr28, smr0);
      qacc[29] = __dtu_m_vmm2_mode19_s8_acc0(qacc[29], vr29, smr0);
      qacc[30] = __dtu_m_vmm2_mode19_s8_acc0(qacc[30], vr30, smr0);
      qacc[31] = __dtu_m_vmm2_mode19_s8_acc0(qacc[31], vr31, smr0);
      // m0k1
      vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr31 = __dtu_s_tvld_itar(lt_base, lt_off1);
      // smr1
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 0);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 1);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 2);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 3);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 4);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 5);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 6);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 7);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 8);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 9);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 10);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 11);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 12);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 13);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 14);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 15);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 16);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 17);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 18);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 19);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 20);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 21);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 22);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 23);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 24);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 25);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 26);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 27);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 28);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 29);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 30);
      smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 31);
      __dtu_c_movsr2naccovr(0x1);
      // m0k1 * smr1
      qacc[0] = __dtu_m_vmm2_mode19_s8_acc0(qacc[0], vr0, smr1);
      qacc[1] = __dtu_m_vmm2_mode19_s8_acc0(qacc[1], vr1, smr1);
      qacc[2] = __dtu_m_vmm2_mode19_s8_acc0(qacc[2], vr2, smr1);
      qacc[3] = __dtu_m_vmm2_mode19_s8_acc0(qacc[3], vr3, smr1);
      qacc[4] = __dtu_m_vmm2_mode19_s8_acc0(qacc[4], vr4, smr1);
      qacc[5] = __dtu_m_vmm2_mode19_s8_acc0(qacc[5], vr5, smr1);
      qacc[6] = __dtu_m_vmm2_mode19_s8_acc0(qacc[6], vr6, smr1);
      qacc[7] = __dtu_m_vmm2_mode19_s8_acc0(qacc[7], vr7, smr1);
      qacc[8] = __dtu_m_vmm2_mode19_s8_acc0(qacc[8], vr8, smr1);
      qacc[9] = __dtu_m_vmm2_mode19_s8_acc0(qacc[9], vr9, smr1);
      qacc[10] = __dtu_m_vmm2_mode19_s8_acc0(qacc[10], vr10, smr1);
      qacc[11] = __dtu_m_vmm2_mode19_s8_acc0(qacc[11], vr11, smr1);
      qacc[12] = __dtu_m_vmm2_mode19_s8_acc0(qacc[12], vr12, smr1);
      qacc[13] = __dtu_m_vmm2_mode19_s8_acc0(qacc[13], vr13, smr1);
      qacc[14] = __dtu_m_vmm2_mode19_s8_acc0(qacc[14], vr14, smr1);
      qacc[15] = __dtu_m_vmm2_mode19_s8_acc0(qacc[15], vr15, smr1);
      qacc[16] = __dtu_m_vmm2_mode19_s8_acc0(qacc[16], vr16, smr1);
      qacc[17] = __dtu_m_vmm2_mode19_s8_acc0(qacc[17], vr17, smr1);
      qacc[18] = __dtu_m_vmm2_mode19_s8_acc0(qacc[18], vr18, smr1);
      qacc[19] = __dtu_m_vmm2_mode19_s8_acc0(qacc[19], vr19, smr1);
      qacc[20] = __dtu_m_vmm2_mode19_s8_acc0(qacc[20], vr20, smr1);
      qacc[21] = __dtu_m_vmm2_mode19_s8_acc0(qacc[21], vr21, smr1);
      qacc[22] = __dtu_m_vmm2_mode19_s8_acc0(qacc[22], vr22, smr1);
      qacc[23] = __dtu_m_vmm2_mode19_s8_acc0(qacc[23], vr23, smr1);
      qacc[24] = __dtu_m_vmm2_mode19_s8_acc0(qacc[24], vr24, smr1);
      qacc[25] = __dtu_m_vmm2_mode19_s8_acc0(qacc[25], vr25, smr1);
      qacc[26] = __dtu_m_vmm2_mode19_s8_acc0(qacc[26], vr26, smr1);
      qacc[27] = __dtu_m_vmm2_mode19_s8_acc0(qacc[27], vr27, smr1);
      qacc[28] = __dtu_m_vmm2_mode19_s8_acc0(qacc[28], vr28, smr1);
      qacc[29] = __dtu_m_vmm2_mode19_s8_acc0(qacc[29], vr29, smr1);
      qacc[30] = __dtu_m_vmm2_mode19_s8_acc0(qacc[30], vr30, smr1);
      qacc[31] = __dtu_m_vmm2_mode19_s8_acc0(qacc[31], vr31, smr1);
      // next k unit smr0
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 0);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 1);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 2);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 3);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 4);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 5);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 6);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 7);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 8);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 9);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 10);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 11);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 12);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 13);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 14);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 15);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 16);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 17);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 18);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 19);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 20);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 21);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 22);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 23);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 24);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 25);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 26);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 27);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 28);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 29);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 30);
      smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 31);
      // next k unit m0k0
      vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
      vr31 = __dtu_s_tvld_itar(lt_base, lt_off1);
    }  // end kcout-1
    // last k unit of last n unit m0k0 * smr0
    qacc[0] = __dtu_m_vmm2_mode19_s8_acc0(qacc[0], vr0, smr0);
    qacc[1] = __dtu_m_vmm2_mode19_s8_acc0(qacc[1], vr1, smr0);
    qacc[2] = __dtu_m_vmm2_mode19_s8_acc0(qacc[2], vr2, smr0);
    qacc[3] = __dtu_m_vmm2_mode19_s8_acc0(qacc[3], vr3, smr0);
    qacc[4] = __dtu_m_vmm2_mode19_s8_acc0(qacc[4], vr4, smr0);
    qacc[5] = __dtu_m_vmm2_mode19_s8_acc0(qacc[5], vr5, smr0);
    qacc[6] = __dtu_m_vmm2_mode19_s8_acc0(qacc[6], vr6, smr0);
    qacc[7] = __dtu_m_vmm2_mode19_s8_acc0(qacc[7], vr7, smr0);
    qacc[8] = __dtu_m_vmm2_mode19_s8_acc0(qacc[8], vr8, smr0);
    qacc[9] = __dtu_m_vmm2_mode19_s8_acc0(qacc[9], vr9, smr0);
    qacc[10] = __dtu_m_vmm2_mode19_s8_acc0(qacc[10], vr10, smr0);
    qacc[11] = __dtu_m_vmm2_mode19_s8_acc0(qacc[11], vr11, smr0);
    qacc[12] = __dtu_m_vmm2_mode19_s8_acc0(qacc[12], vr12, smr0);
    qacc[13] = __dtu_m_vmm2_mode19_s8_acc0(qacc[13], vr13, smr0);
    qacc[14] = __dtu_m_vmm2_mode19_s8_acc0(qacc[14], vr14, smr0);
    qacc[15] = __dtu_m_vmm2_mode19_s8_acc0(qacc[15], vr15, smr0);
    qacc[16] = __dtu_m_vmm2_mode19_s8_acc0(qacc[16], vr16, smr0);
    qacc[17] = __dtu_m_vmm2_mode19_s8_acc0(qacc[17], vr17, smr0);
    qacc[18] = __dtu_m_vmm2_mode19_s8_acc0(qacc[18], vr18, smr0);
    qacc[19] = __dtu_m_vmm2_mode19_s8_acc0(qacc[19], vr19, smr0);
    qacc[20] = __dtu_m_vmm2_mode19_s8_acc0(qacc[20], vr20, smr0);
    qacc[21] = __dtu_m_vmm2_mode19_s8_acc0(qacc[21], vr21, smr0);
    qacc[22] = __dtu_m_vmm2_mode19_s8_acc0(qacc[22], vr22, smr0);
    qacc[23] = __dtu_m_vmm2_mode19_s8_acc0(qacc[23], vr23, smr0);
    qacc[24] = __dtu_m_vmm2_mode19_s8_acc0(qacc[24], vr24, smr0);
    qacc[25] = __dtu_m_vmm2_mode19_s8_acc0(qacc[25], vr25, smr0);
    qacc[26] = __dtu_m_vmm2_mode19_s8_acc0(qacc[26], vr26, smr0);
    qacc[27] = __dtu_m_vmm2_mode19_s8_acc0(qacc[27], vr27, smr0);
    qacc[28] = __dtu_m_vmm2_mode19_s8_acc0(qacc[28], vr28, smr0);
    qacc[29] = __dtu_m_vmm2_mode19_s8_acc0(qacc[29], vr29, smr0);
    qacc[30] = __dtu_m_vmm2_mode19_s8_acc0(qacc[30], vr30, smr0);
    qacc[31] = __dtu_m_vmm2_mode19_s8_acc0(qacc[31], vr31, smr0);
    // m0k1
    vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr31 = __dtu_s_tvld_itar(lt_base, lt_off3);  // end k  end n new m
    // smr1
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 1);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 2);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 3);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 4);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 5);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 6);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 7);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 8);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 9);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 10);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 11);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 12);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 13);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 14);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 15);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 16);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 17);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 18);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 19);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 20);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 21);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 22);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 23);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 24);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 25);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 26);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 27);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 28);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 29);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 30);
    smr1 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr1, rt_base, rt_off0, 31);
    __dtu_c_movsr2naccovr(0x1);
    // m0k1 * smr1
    qacc[0] = __dtu_m_vmm2_mode19_s8_acc0(qacc[0], vr0, smr1);
    qacc[1] = __dtu_m_vmm2_mode19_s8_acc0(qacc[1], vr1, smr1);
    qacc[2] = __dtu_m_vmm2_mode19_s8_acc0(qacc[2], vr2, smr1);
    qacc[3] = __dtu_m_vmm2_mode19_s8_acc0(qacc[3], vr3, smr1);
    qacc[4] = __dtu_m_vmm2_mode19_s8_acc0(qacc[4], vr4, smr1);
    qacc[5] = __dtu_m_vmm2_mode19_s8_acc0(qacc[5], vr5, smr1);
    qacc[6] = __dtu_m_vmm2_mode19_s8_acc0(qacc[6], vr6, smr1);
    qacc[7] = __dtu_m_vmm2_mode19_s8_acc0(qacc[7], vr7, smr1);
    qacc[8] = __dtu_m_vmm2_mode19_s8_acc0(qacc[8], vr8, smr1);
    qacc[9] = __dtu_m_vmm2_mode19_s8_acc0(qacc[9], vr9, smr1);
    qacc[10] = __dtu_m_vmm2_mode19_s8_acc0(qacc[10], vr10, smr1);
    qacc[11] = __dtu_m_vmm2_mode19_s8_acc0(qacc[11], vr11, smr1);
    qacc[12] = __dtu_m_vmm2_mode19_s8_acc0(qacc[12], vr12, smr1);
    qacc[13] = __dtu_m_vmm2_mode19_s8_acc0(qacc[13], vr13, smr1);
    qacc[14] = __dtu_m_vmm2_mode19_s8_acc0(qacc[14], vr14, smr1);
    qacc[15] = __dtu_m_vmm2_mode19_s8_acc0(qacc[15], vr15, smr1);
    qacc[16] = __dtu_m_vmm2_mode19_s8_acc0(qacc[16], vr16, smr1);
    qacc[17] = __dtu_m_vmm2_mode19_s8_acc0(qacc[17], vr17, smr1);
    qacc[18] = __dtu_m_vmm2_mode19_s8_acc0(qacc[18], vr18, smr1);
    qacc[19] = __dtu_m_vmm2_mode19_s8_acc0(qacc[19], vr19, smr1);
    qacc[20] = __dtu_m_vmm2_mode19_s8_acc0(qacc[20], vr20, smr1);
    qacc[21] = __dtu_m_vmm2_mode19_s8_acc0(qacc[21], vr21, smr1);
    qacc[22] = __dtu_m_vmm2_mode19_s8_acc0(qacc[22], vr22, smr1);
    qacc[23] = __dtu_m_vmm2_mode19_s8_acc0(qacc[23], vr23, smr1);
    qacc[24] = __dtu_m_vmm2_mode19_s8_acc0(qacc[24], vr24, smr1);
    qacc[25] = __dtu_m_vmm2_mode19_s8_acc0(qacc[25], vr25, smr1);
    qacc[26] = __dtu_m_vmm2_mode19_s8_acc0(qacc[26], vr26, smr1);
    qacc[27] = __dtu_m_vmm2_mode19_s8_acc0(qacc[27], vr27, smr1);
    qacc[28] = __dtu_m_vmm2_mode19_s8_acc0(qacc[28], vr28, smr1);
    qacc[29] = __dtu_m_vmm2_mode19_s8_acc0(qacc[29], vr29, smr1);
    qacc[30] = __dtu_m_vmm2_mode19_s8_acc0(qacc[30], vr30, smr1);
    qacc[31] = __dtu_m_vmm2_mode19_s8_acc0(qacc[31], vr31, smr1);
    // back to begin of [k, n]
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off2, 0);
    // next m unit smr0
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 1);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 2);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 3);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 4);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 5);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 6);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 7);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 8);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 9);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 10);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 11);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 12);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 13);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 14);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 15);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 16);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 17);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 18);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 19);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 20);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 21);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 22);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 23);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 24);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 25);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 26);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 27);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 28);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 29);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 30);
    smr0 = __dtu_v_ldsmr2_mem_v_mode19_s8_row(smr0, rt_base, rt_off0, 31);
    // next n unit m0k0
    vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr15 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr16 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr17 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr18 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr19 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr20 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr21 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr22 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr23 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr24 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr25 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr26 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr27 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr28 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr29 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr30 = __dtu_s_tvld_itar(lt_base, lt_off0);
    vr31 = __dtu_s_tvld_itar(lt_base, lt_off1);
    vab_shift += 512;
    __dtu_c_movsr2vab_m_s1(vab_shift);
    __dtu_c_movsr2vab_m_d(vab_shift);
  }  // end mcount
     // __DTU_INTRIN_FUNC_TYPE__ va16f32x4 __dtu_m_mop_mac_f32mix_s8_va(va16i8,
     // va16f32x4, va16f32x4);
  if (stroe_flag) {
    vab_shift = 0;
    __dtu_c_movsr2vab_lv_s(0);
    __dtu_c_movsr2vab_lv_d(0);
    __dtu_c_movsr2vab_m_s1(0);
    __dtu_c_movsr2vab_m_d(0);
#pragma clang loop unroll(disable)
    for (int m = 0; m < M; m = m + 32) {
#pragma clang loop unroll(disable)
      for (int n = 0; n < N - 128; n = n + 128) {
        __dtu_c_movsr2vab_m_s2(0);
        vacc[0] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[1] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[2] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[3] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[4] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[5] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[6] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[7] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[8] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[9] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[10] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[11] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[12] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[13] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[14] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[15] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[16] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[17] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[18] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[19] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[20] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[21] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[22] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[23] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[24] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[25] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[26] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[27] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[28] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[29] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[30] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
        vacc[31] = __dtu_l_tvldqa_s8_va(bt_base, bt_off1);
        f_qacc[0] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[0], qa_alpha);
        f_qacc[1] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[1], qa_alpha);
        f_qacc[2] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[2], qa_alpha);
        f_qacc[3] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[3], qa_alpha);
        f_qacc[4] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[4], qa_alpha);
        f_qacc[5] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[5], qa_alpha);
        f_qacc[6] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[6], qa_alpha);
        f_qacc[7] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[7], qa_alpha);
        f_qacc[8] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[8], qa_alpha);
        f_qacc[9] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[9], qa_alpha);
        f_qacc[10] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[10], qa_alpha);
        f_qacc[11] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[11], qa_alpha);
        f_qacc[12] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[12], qa_alpha);
        f_qacc[13] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[13], qa_alpha);
        f_qacc[14] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[14], qa_alpha);
        f_qacc[15] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[15], qa_alpha);
        f_qacc[16] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[16], qa_alpha);
        f_qacc[17] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[17], qa_alpha);
        f_qacc[18] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[18], qa_alpha);
        f_qacc[19] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[19], qa_alpha);
        f_qacc[20] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[20], qa_alpha);
        f_qacc[21] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[21], qa_alpha);
        f_qacc[22] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[22], qa_alpha);
        f_qacc[23] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[23], qa_alpha);
        f_qacc[24] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[24], qa_alpha);
        f_qacc[25] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[25], qa_alpha);
        f_qacc[26] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[26], qa_alpha);
        f_qacc[27] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[27], qa_alpha);
        f_qacc[28] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[28], qa_alpha);
        f_qacc[29] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[29], qa_alpha);
        f_qacc[30] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[30], qa_alpha);
        f_qacc[31] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[31], qa_alpha);

        f_qacc[0] = __dtu_m_mop_mac_f32mix_s8_va(vacc[0], qa_beta, f_qacc[0]);
        f_qacc[1] = __dtu_m_mop_mac_f32mix_s8_va(vacc[1], qa_beta, f_qacc[1]);
        f_qacc[2] = __dtu_m_mop_mac_f32mix_s8_va(vacc[2], qa_beta, f_qacc[2]);
        f_qacc[3] = __dtu_m_mop_mac_f32mix_s8_va(vacc[3], qa_beta, f_qacc[3]);
        f_qacc[4] = __dtu_m_mop_mac_f32mix_s8_va(vacc[4], qa_beta, f_qacc[4]);
        f_qacc[5] = __dtu_m_mop_mac_f32mix_s8_va(vacc[5], qa_beta, f_qacc[5]);
        f_qacc[6] = __dtu_m_mop_mac_f32mix_s8_va(vacc[6], qa_beta, f_qacc[6]);
        f_qacc[7] = __dtu_m_mop_mac_f32mix_s8_va(vacc[7], qa_beta, f_qacc[7]);
        f_qacc[8] = __dtu_m_mop_mac_f32mix_s8_va(vacc[8], qa_beta, f_qacc[8]);
        f_qacc[9] = __dtu_m_mop_mac_f32mix_s8_va(vacc[9], qa_beta, f_qacc[9]);
        f_qacc[10] =
            __dtu_m_mop_mac_f32mix_s8_va(vacc[10], qa_beta, f_qacc[10]);
        f_qacc[11] =
            __dtu_m_mop_mac_f32mix_s8_va(vacc[11], qa_beta, f_qacc[11]);
        f_qacc[12] =
            __dtu_m_mop_mac_f32mix_s8_va(vacc[12], qa_beta, f_qacc[12]);
        f_qacc[13] =
            __dtu_m_mop_mac_f32mix_s8_va(vacc[13], qa_beta, f_qacc[13]);
        f_qacc[14] =
            __dtu_m_mop_mac_f32mix_s8_va(vacc[14], qa_beta, f_qacc[14]);
        f_qacc[15] =
            __dtu_m_mop_mac_f32mix_s8_va(vacc[15], qa_beta, f_qacc[15]);
        f_qacc[16] =
            __dtu_m_mop_mac_f32mix_s8_va(vacc[16], qa_beta, f_qacc[16]);
        f_qacc[17] =
            __dtu_m_mop_mac_f32mix_s8_va(vacc[17], qa_beta, f_qacc[17]);
        f_qacc[18] =
            __dtu_m_mop_mac_f32mix_s8_va(vacc[18], qa_beta, f_qacc[18]);
        f_qacc[19] =
            __dtu_m_mop_mac_f32mix_s8_va(vacc[19], qa_beta, f_qacc[19]);
        f_qacc[20] =
            __dtu_m_mop_mac_f32mix_s8_va(vacc[20], qa_beta, f_qacc[20]);
        f_qacc[21] =
            __dtu_m_mop_mac_f32mix_s8_va(vacc[21], qa_beta, f_qacc[21]);
        f_qacc[22] =
            __dtu_m_mop_mac_f32mix_s8_va(vacc[22], qa_beta, f_qacc[22]);
        f_qacc[23] =
            __dtu_m_mop_mac_f32mix_s8_va(vacc[23], qa_beta, f_qacc[23]);
        f_qacc[24] =
            __dtu_m_mop_mac_f32mix_s8_va(vacc[24], qa_beta, f_qacc[24]);
        f_qacc[25] =
            __dtu_m_mop_mac_f32mix_s8_va(vacc[25], qa_beta, f_qacc[25]);
        f_qacc[26] =
            __dtu_m_mop_mac_f32mix_s8_va(vacc[26], qa_beta, f_qacc[26]);
        f_qacc[27] =
            __dtu_m_mop_mac_f32mix_s8_va(vacc[27], qa_beta, f_qacc[27]);
        f_qacc[28] =
            __dtu_m_mop_mac_f32mix_s8_va(vacc[28], qa_beta, f_qacc[28]);
        f_qacc[29] =
            __dtu_m_mop_mac_f32mix_s8_va(vacc[29], qa_beta, f_qacc[29]);
        f_qacc[30] =
            __dtu_m_mop_mac_f32mix_s8_va(vacc[30], qa_beta, f_qacc[30]);
        f_qacc[31] =
            __dtu_m_mop_mac_f32mix_s8_va(vacc[31], qa_beta, f_qacc[31]);

        // vst qacc
        qacc[0] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[0]);
        qacc[1] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[1]);
        qacc[2] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[2]);
        qacc[3] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[3]);
        qacc[4] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[4]);
        qacc[5] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[5]);
        qacc[6] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[6]);
        qacc[7] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[7]);
        qacc[8] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[8]);
        qacc[9] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[9]);
        qacc[10] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[10]);
        qacc[11] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[11]);
        qacc[12] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[12]);
        qacc[13] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[13]);
        qacc[14] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[14]);
        qacc[15] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[15]);
        qacc[16] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[16]);
        qacc[17] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[17]);
        qacc[18] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[18]);
        qacc[19] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[19]);
        qacc[20] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[20]);
        qacc[21] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[21]);
        qacc[22] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[22]);
        qacc[23] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[23]);
        qacc[24] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[24]);
        qacc[25] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[25]);
        qacc[26] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[26]);
        qacc[27] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[27]);
        qacc[28] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[28]);
        qacc[29] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[29]);
        qacc[30] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[30]);
        qacc[31] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[31]);

        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[0], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[1], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[2], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[3], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[4], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[5], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[6], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[7], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[8], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[9], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[10], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[11], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[12], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[13], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[14], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[15], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[16], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[17], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[18], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[19], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[20], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[21], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[22], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[23], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[24], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[25], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[26], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[27], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[28], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[29], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[30], ot_base, ot_off0);
        __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[31], ot_base, ot_off1);

        vab_shift += 512;
        __dtu_c_movsr2vab_lv_s(vab_shift);
        __dtu_c_movsr2vab_lv_d(vab_shift);
        __dtu_c_movsr2vab_m_s1(vab_shift);
        __dtu_c_movsr2vab_m_d(vab_shift);
      }
      __dtu_c_movsr2vab_m_s2(0);
      vacc[0] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[1] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[2] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[3] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[4] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[5] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[6] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[7] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[8] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[9] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[10] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[11] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[12] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[13] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[14] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[15] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[16] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[17] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[18] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[19] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[20] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[21] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[22] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[23] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[24] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[25] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[26] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[27] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[28] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[29] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[30] = __dtu_l_tvldqa_s8_va(bt_base, bt_off0);
      vacc[31] = __dtu_l_tvldqa_s8_va(bt_base, bt_off2);

      f_qacc[0] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[0], qa_alpha);
      f_qacc[1] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[1], qa_alpha);
      f_qacc[2] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[2], qa_alpha);
      f_qacc[3] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[3], qa_alpha);
      f_qacc[4] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[4], qa_alpha);
      f_qacc[5] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[5], qa_alpha);
      f_qacc[6] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[6], qa_alpha);
      f_qacc[7] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[7], qa_alpha);
      f_qacc[8] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[8], qa_alpha);
      f_qacc[9] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[9], qa_alpha);
      f_qacc[10] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[10], qa_alpha);
      f_qacc[11] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[11], qa_alpha);
      f_qacc[12] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[12], qa_alpha);
      f_qacc[13] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[13], qa_alpha);
      f_qacc[14] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[14], qa_alpha);
      f_qacc[15] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[15], qa_alpha);
      f_qacc[16] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[16], qa_alpha);
      f_qacc[17] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[17], qa_alpha);
      f_qacc[18] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[18], qa_alpha);
      f_qacc[19] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[19], qa_alpha);
      f_qacc[20] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[20], qa_alpha);
      f_qacc[21] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[21], qa_alpha);
      f_qacc[22] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[22], qa_alpha);
      f_qacc[23] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[23], qa_alpha);
      f_qacc[24] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[24], qa_alpha);
      f_qacc[25] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[25], qa_alpha);
      f_qacc[26] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[26], qa_alpha);
      f_qacc[27] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[27], qa_alpha);
      f_qacc[28] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[28], qa_alpha);
      f_qacc[29] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[29], qa_alpha);
      f_qacc[30] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[30], qa_alpha);
      f_qacc[31] = __dtu_m_mop_mul_f32mix_s32_qa(qacc[31], qa_alpha);

      f_qacc[0] = __dtu_m_mop_mac_f32mix_s8_va(vacc[0], qa_beta, f_qacc[0]);
      f_qacc[1] = __dtu_m_mop_mac_f32mix_s8_va(vacc[1], qa_beta, f_qacc[1]);
      f_qacc[2] = __dtu_m_mop_mac_f32mix_s8_va(vacc[2], qa_beta, f_qacc[2]);
      f_qacc[3] = __dtu_m_mop_mac_f32mix_s8_va(vacc[3], qa_beta, f_qacc[3]);
      f_qacc[4] = __dtu_m_mop_mac_f32mix_s8_va(vacc[4], qa_beta, f_qacc[4]);
      f_qacc[5] = __dtu_m_mop_mac_f32mix_s8_va(vacc[5], qa_beta, f_qacc[5]);
      f_qacc[6] = __dtu_m_mop_mac_f32mix_s8_va(vacc[6], qa_beta, f_qacc[6]);
      f_qacc[7] = __dtu_m_mop_mac_f32mix_s8_va(vacc[7], qa_beta, f_qacc[7]);
      f_qacc[8] = __dtu_m_mop_mac_f32mix_s8_va(vacc[8], qa_beta, f_qacc[8]);
      f_qacc[9] = __dtu_m_mop_mac_f32mix_s8_va(vacc[9], qa_beta, f_qacc[9]);
      f_qacc[10] = __dtu_m_mop_mac_f32mix_s8_va(vacc[10], qa_beta, f_qacc[10]);
      f_qacc[11] = __dtu_m_mop_mac_f32mix_s8_va(vacc[11], qa_beta, f_qacc[11]);
      f_qacc[12] = __dtu_m_mop_mac_f32mix_s8_va(vacc[12], qa_beta, f_qacc[12]);
      f_qacc[13] = __dtu_m_mop_mac_f32mix_s8_va(vacc[13], qa_beta, f_qacc[13]);
      f_qacc[14] = __dtu_m_mop_mac_f32mix_s8_va(vacc[14], qa_beta, f_qacc[14]);
      f_qacc[15] = __dtu_m_mop_mac_f32mix_s8_va(vacc[15], qa_beta, f_qacc[15]);
      f_qacc[16] = __dtu_m_mop_mac_f32mix_s8_va(vacc[16], qa_beta, f_qacc[16]);
      f_qacc[17] = __dtu_m_mop_mac_f32mix_s8_va(vacc[17], qa_beta, f_qacc[17]);
      f_qacc[18] = __dtu_m_mop_mac_f32mix_s8_va(vacc[18], qa_beta, f_qacc[18]);
      f_qacc[19] = __dtu_m_mop_mac_f32mix_s8_va(vacc[19], qa_beta, f_qacc[19]);
      f_qacc[20] = __dtu_m_mop_mac_f32mix_s8_va(vacc[20], qa_beta, f_qacc[20]);
      f_qacc[21] = __dtu_m_mop_mac_f32mix_s8_va(vacc[21], qa_beta, f_qacc[21]);
      f_qacc[22] = __dtu_m_mop_mac_f32mix_s8_va(vacc[22], qa_beta, f_qacc[22]);
      f_qacc[23] = __dtu_m_mop_mac_f32mix_s8_va(vacc[23], qa_beta, f_qacc[23]);
      f_qacc[24] = __dtu_m_mop_mac_f32mix_s8_va(vacc[24], qa_beta, f_qacc[24]);
      f_qacc[25] = __dtu_m_mop_mac_f32mix_s8_va(vacc[25], qa_beta, f_qacc[25]);
      f_qacc[26] = __dtu_m_mop_mac_f32mix_s8_va(vacc[26], qa_beta, f_qacc[26]);
      f_qacc[27] = __dtu_m_mop_mac_f32mix_s8_va(vacc[27], qa_beta, f_qacc[27]);
      f_qacc[28] = __dtu_m_mop_mac_f32mix_s8_va(vacc[28], qa_beta, f_qacc[28]);
      f_qacc[29] = __dtu_m_mop_mac_f32mix_s8_va(vacc[29], qa_beta, f_qacc[29]);
      f_qacc[30] = __dtu_m_mop_mac_f32mix_s8_va(vacc[30], qa_beta, f_qacc[30]);
      f_qacc[31] = __dtu_m_mop_mac_f32mix_s8_va(vacc[31], qa_beta, f_qacc[31]);

      // vst qacc
      qacc[0] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[0]);
      qacc[1] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[1]);
      qacc[2] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[2]);
      qacc[3] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[3]);
      qacc[4] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[4]);
      qacc[5] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[5]);
      qacc[6] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[6]);
      qacc[7] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[7]);
      qacc[8] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[8]);
      qacc[9] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[9]);
      qacc[10] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[10]);
      qacc[11] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[11]);
      qacc[12] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[12]);
      qacc[13] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[13]);
      qacc[14] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[14]);
      qacc[15] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[15]);
      qacc[16] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[16]);
      qacc[17] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[17]);
      qacc[18] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[18]);
      qacc[19] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[19]);
      qacc[20] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[20]);
      qacc[21] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[21]);
      qacc[22] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[22]);
      qacc[23] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[23]);
      qacc[24] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[24]);
      qacc[25] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[25]);
      qacc[26] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[26]);
      qacc[27] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[27]);
      qacc[28] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[28]);
      qacc[29] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[29]);
      qacc[30] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[30]);
      qacc[31] = __dtu_m_mop_cvt_qa_rne_f32_s32(f_qacc[31]);

      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[0], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[1], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[2], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[3], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[4], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[5], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[6], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[7], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[8], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[9], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[10], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[11], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[12], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[13], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[14], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[15], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[16], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[17], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[18], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[19], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[20], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[21], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[22], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[23], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[24], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[25], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[26], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[27], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[28], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[29], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[30], ot_base, ot_off0);
      __dtu_l_tvsta_cvt2s8_rnd_clamp(qacc[31], ot_base, ot_off2);

      vab_shift += 512;
      __dtu_c_movsr2vab_lv_s(vab_shift);
      __dtu_c_movsr2vab_lv_d(vab_shift);
      __dtu_c_movsr2vab_m_s1(vab_shift);
      __dtu_c_movsr2vab_m_d(vab_shift);
    }
  }
#endif
}


