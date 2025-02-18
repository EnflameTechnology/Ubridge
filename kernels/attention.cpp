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
#include <tops/dte_chain.h>
#include "tops/tops_runtime.h"
#include "utils/utils.h"
// #include "krt/scalar_type.h"
#include "utils/attention_gqa_kernels.h"

using namespace tops;
#define TAR32(low, high) (((low) & 0xffff) | (((high) & 0xffff) << 16))

#define ALIGN_512(x)  ((((x) + 511) >> 9) << 9)
#define ALIGN_256(x)  ((((x) + 255) >> 8) << 8)
#define ALIGN_128(x)  ((((x) + 127) >> 7) << 7)
#define ALIGN_64(x)  ((((x) + 63) >> 6) << 6)
#define ALIGN_32(x)  ((((x) + 31) >> 5) << 5)
#define ALIGN_16(x)  ((((x) + 15) >> 4) << 4)
#define IS_ALIGN_64(x)  (((x) & 0x3f) == 0)
#define IS_ALIGN_128(x)  (((x) & 0x7f) == 0)
#define MAX_BLOCK_TABLE_LEN  (16 * 1024)
#define SLICE_LEN   8192
#define SUM_OFFSET  32
#define F32_OUT   2
#define F16_OUT   1
#define NO_OUT    0

template <typename T>
// static
__device__ void call_k_dot_nk(
    float* out, int attn_offset, T* lhs, T* rhs,
    int K, int N, float scale, char *p_mid,
    float slope, int ctx_len, int alibi_enable);

template <typename T>
__device__ void call_convert(T* p_out, float* p_in, float* p_sum, int in_len);

template <>
__device__ void call_convert(
    tops::half* p_out, float* p_in, float* p_sum, int in_len) {
  auto qacc0 = __dtu_l_vldqa_f32_qa(reinterpret_cast<char*>(p_in));
  auto qacc1 = __dtu_l_vldqa_f32_qa(reinterpret_cast<char*>(p_in + 128));
  auto vacc0 = __dtu_l_vldla(reinterpret_cast<char*>(p_sum), 0);
  vacc0 = __dtu_m_msf_rec_f32(vacc0);
  auto dacc0 = __dtu_insertva2da_f32(vacc0, vacc0);

  auto dacc2 = __dtu_extractqa2da(qacc0, 0);
  auto dacc3 = __dtu_extractqa2da(qacc0, 1);
  auto dacc4 = __dtu_extractqa2da(qacc1, 0);
  auto dacc5 = __dtu_extractqa2da(qacc1, 1);
  // dacc2 = __dtu_m_mop_mul_f32_da(dacc2, dacc0);
  // dacc3 = __dtu_m_mop_mul_f32_da(dacc3, dacc0);
  // dacc4 = __dtu_m_mop_mul_f32_da(dacc4, dacc0);
  // dacc5 = __dtu_m_mop_mul_f32_da(dacc5, dacc0);

  int out_addr = reinterpret_cast<long>(p_out);
  out_addr = out_addr >> 6;
  tar_t ot_base = __dtu_c_movsr2targ(TAR32(out_addr, out_addr + 2));
  int offset = TAR32(1, 1);
  tar_t ot_off0 = __dtu_c_movsr2tari(offset, ot_base);
  offset = TAR32(3, 3);
  tar_t ot_off1 = __dtu_c_movsr2tari(offset, ot_base);

  __dtu_l_tvsta_cvt2fp16(dacc2, ot_base, ot_off0);
  __dtu_l_tvsta_cvt2fp16(dacc3, ot_base, ot_off1);
  if (in_len > 128) {
    __dtu_l_tvsta_cvt2fp16(dacc4, ot_base, ot_off0);
    __dtu_l_tvsta_cvt2fp16(dacc5, ot_base, ot_off1);
  }
}

template <>
__device__ void call_convert(
    tops::bfloat* p_out, float* p_in, float* p_sum, int in_len) {
  auto qacc0 = __dtu_l_vldqa_f32_qa(reinterpret_cast<char*>(p_in));
  auto qacc1 = __dtu_l_vldqa_f32_qa(reinterpret_cast<char*>(p_in + 128));
  auto vacc0 = __dtu_l_vldla(reinterpret_cast<char*>(p_sum), 0);
  vacc0 = __dtu_m_msf_rec_f32(vacc0);
  auto dacc0 = __dtu_insertva2da_f32(vacc0, vacc0);

  auto dacc2 = __dtu_extractqa2da(qacc0, 0);
  auto dacc3 = __dtu_extractqa2da(qacc0, 1);
  auto dacc4 = __dtu_extractqa2da(qacc1, 0);
  auto dacc5 = __dtu_extractqa2da(qacc1, 1);
  // dacc2 = __dtu_m_mop_mul_f32_da(dacc2, dacc0);
  // dacc3 = __dtu_m_mop_mul_f32_da(dacc3, dacc0);
  // dacc4 = __dtu_m_mop_mul_f32_da(dacc4, dacc0);
  // dacc5 = __dtu_m_mop_mul_f32_da(dacc5, dacc0);

  int out_addr = reinterpret_cast<long>(p_out);
  out_addr = out_addr >> 6;
  tar_t ot_base = __dtu_c_movsr2targ(TAR32(out_addr, out_addr + 2));
  int offset = TAR32(1, 1);
  tar_t ot_off0 = __dtu_c_movsr2tari(offset, ot_base);
  offset = TAR32(3, 3);
  tar_t ot_off1 = __dtu_c_movsr2tari(offset, ot_base);

  __dtu_l_tvsta_cvt2bf16(dacc2, ot_base, ot_off0);
  __dtu_l_tvsta_cvt2bf16(dacc3, ot_base, ot_off1);
  if (in_len > 128) {
    __dtu_l_tvsta_cvt2bf16(dacc4, ot_base, ot_off0);
    __dtu_l_tvsta_cvt2bf16(dacc5, ot_base, ot_off1);
  }
}

__device__
void call_update_softmax(
    float* p_max0, float* p_max1, float* p_in0, float* p_in1, int head_size) {
  auto max0 = __dtu_v_vldl(reinterpret_cast<char*>(p_max0), 0);
  auto max1 = __dtu_v_vldl(reinterpret_cast<char*>(p_max1), 0);
  auto sum0 = __dtu_v_vldl(reinterpret_cast<char*>(p_max0), 1);
  auto sum1 = __dtu_v_vldl(reinterpret_cast<char*>(p_max1), 1);

  auto val0 = __dtu_v_vldl(reinterpret_cast<char*>(p_in0), 0);
  auto val1 = __dtu_v_vldl(reinterpret_cast<char*>(p_in0), 1);
  auto val2 = __dtu_v_vldl(reinterpret_cast<char*>(p_in0), 2);
  auto val3 = __dtu_v_vldl(reinterpret_cast<char*>(p_in0), 3);
  auto val4 = __dtu_v_vldl(reinterpret_cast<char*>(p_in1), 0);
  auto val5 = __dtu_v_vldl(reinterpret_cast<char*>(p_in1), 1);
  auto val6 = __dtu_v_vldl(reinterpret_cast<char*>(p_in1), 2);
  auto val7 = __dtu_v_vldl(reinterpret_cast<char*>(p_in1), 3);

  v16f32 val[8];
  if (head_size > 128) {
    val[0] = __dtu_v_vldl(reinterpret_cast<char*>(p_in0), 4);
    val[1] = __dtu_v_vldl(reinterpret_cast<char*>(p_in0), 5);
    val[2] = __dtu_v_vldl(reinterpret_cast<char*>(p_in0), 6);
    val[3] = __dtu_v_vldl(reinterpret_cast<char*>(p_in0), 7);
    val[4] = __dtu_v_vldl(reinterpret_cast<char*>(p_in1), 4);
    val[5] = __dtu_v_vldl(reinterpret_cast<char*>(p_in1), 5);
    val[6] = __dtu_v_vldl(reinterpret_cast<char*>(p_in1), 6);
    val[7] = __dtu_v_vldl(reinterpret_cast<char*>(p_in1), 7);
    __dtu_c_movsr2lpr(0);
  } else {
    __dtu_c_movsr2lpr(1);
  }

  auto max = __dtu_v_vmax_a_f32(max0, max1);
  __dtu_l_vstl(max, reinterpret_cast<char*>(p_max0), 0);
  auto sub0 = __dtu_v_vsub_a_f32(max0, max);
  auto sub1 = __dtu_v_vsub_a_f32(max1, max);
  auto ln2 = __dtu_l_movr2va(0x3fb8aa3b);
  auto mul0 = __dtu_v_vmul_a_f32(sub0, ln2);
  auto mul1 = __dtu_v_vmul_a_f32(sub1, ln2);
  auto va0 = __dtu_l_movvr2va(mul0);
  auto va1 = __dtu_l_movvr2va(mul1);

  auto exp0 = __dtu_m_msf_exp_f32(va0);
  auto exp1 = __dtu_m_msf_exp_f32(va1);
  auto vr0 = __dtu_l_movva2vr_w(exp0);
  auto vr1 = __dtu_l_movva2vr_w(exp1);

  sum0 = __dtu_v_vmul_a_f32(sum0, vr0);
  sum1 = __dtu_v_vmul_a_f32(sum1, vr1);
  auto sum = __dtu_v_vadd_a_f32(sum0, sum1);
  auto sum_rec = __dtu_m_msf_rec_f32(sum);
  sum0 = __dtu_v_vmul_a_f32(sum0, sum_rec);
  sum1 = __dtu_v_vmul_a_f32(sum1, sum_rec);

  val0 = __dtu_v_vmul_a_f32(val0, sum0);
  val1 = __dtu_v_vmul_a_f32(val1, sum0);
  val2 = __dtu_v_vmul_a_f32(val2, sum0);
  val3 = __dtu_v_vmul_a_f32(val3, sum0);
  val4 = __dtu_v_vmul_a_f32(val4, sum1);
  val5 = __dtu_v_vmul_a_f32(val5, sum1);
  val6 = __dtu_v_vmul_a_f32(val6, sum1);
  val7 = __dtu_v_vmul_a_f32(val7, sum1);

  if (head_size > 128) {
    val[0] = __dtu_v_vmul_a_f32(val[0], sum0);
    val[1] = __dtu_v_vmul_a_f32(val[1], sum0);
    val[2] = __dtu_v_vmul_a_f32(val[2], sum0);
    val[3] = __dtu_v_vmul_a_f32(val[3], sum0);
    val[4] = __dtu_v_vmul_a_f32(val[4], sum1);
    val[5] = __dtu_v_vmul_a_f32(val[5], sum1);
    val[6] = __dtu_v_vmul_a_f32(val[6], sum1);
    val[7] = __dtu_v_vmul_a_f32(val[7], sum1);
  }


  val0 = __dtu_v_vadd_a_f32(val0, val4);
  val1 = __dtu_v_vadd_a_f32(val1, val5);
  val2 = __dtu_v_vadd_a_f32(val2, val6);
  val3 = __dtu_v_vadd_a_f32(val3, val7);
  val[0] = __dtu_v_vadd_a_f32(val[0], val[4]);
  val[1] = __dtu_v_vadd_a_f32(val[1], val[5]);
  val[2] = __dtu_v_vadd_a_f32(val[2], val[6]);
  val[3] = __dtu_v_vadd_a_f32(val[3], val[7]);

  __dtu_l_vstl(sum, reinterpret_cast<char*>(p_max0), 1);

  __dtu_l_vstl(val0, reinterpret_cast<char*>(p_in0), 0);
  __dtu_l_vstl(val1, reinterpret_cast<char*>(p_in0), 1);
  __dtu_l_vstl(val2, reinterpret_cast<char*>(p_in0), 2);
  __dtu_l_vstl(val3, reinterpret_cast<char*>(p_in0), 3);
  __dtu_l_lpr_vstl(val[0], reinterpret_cast<char*>(p_in0), 4);
  __dtu_l_lpr_vstl(val[1], reinterpret_cast<char*>(p_in0), 5);
  __dtu_l_lpr_vstl(val[2], reinterpret_cast<char*>(p_in0), 6);
  __dtu_l_lpr_vstl(val[3], reinterpret_cast<char*>(p_in0), 7);
}


template <typename T>
__device__ void call_k_dot_kn(
    T* out, float* lhs, T* rhs, float* p_max,
    int K, int N, int softmax_offset, int ctx_len, int st_ind);

template <>
__device__ void call_k_dot_nk(
    float* out, int attn_offset, tops::half* lhs, tops::half* rhs,
    int K, int N, float scale, char *p_mid,
    float slope, int alibi_idx, int alibi_enable) {
  int out_addr = reinterpret_cast<long>(out) + attn_offset * 4;
  int lhs_addr = reinterpret_cast<long>(lhs);
  int rhs_addr = reinterpret_cast<long>(rhs);

  auto k_unit = K >> 6;
  lhs_addr = lhs_addr >> 7;
  rhs_addr = rhs_addr >> 7;
  out_addr = out_addr >> 7;

  tar_t lt_base =
      __dtu_c_movsr2targ(TAR32(lhs_addr, lhs_addr));
  int offset = TAR32(1, 1);
  tar_t lt_off0 = __dtu_c_movsr2tari(offset, lt_base);

  auto next_stride = 1 - k_unit * N;
  tar_t rt_base =
      __dtu_c_movsr2targ(TAR32(rhs_addr, rhs_addr + 32 * k_unit));
  offset = TAR32(k_unit, k_unit);
  tar_t rt_off0 = __dtu_c_movsr2tari(offset, rt_base);
  offset = TAR32(k_unit + k_unit * 32, k_unit + k_unit * 32);
  tar_t rt_off1 = __dtu_c_movsr2tari(offset, rt_base);
  offset = TAR32(next_stride, next_stride);
  tar_t rt_off2 = __dtu_c_movsr2tari(offset, rt_base);

  tar_t ot_base = __dtu_c_movsr2targ(TAR32(out_addr, out_addr + 1));
  offset = TAR32(2, 2);
  tar_t ot_off0 = __dtu_c_movsr2tari(offset, ot_base);

  smr_t smr0, smr1;
  va16f32x2 dacc0, dacc1;
  va16f32x2 v_mid, da_mid, v_ctx_len;
  using vtype = typename scalar_to_vector<float, TOPS_VECTOR_LENGTH/2>::type;
  vtype v_scale = vbroadcast<vtype>(scale);
  vtype v_slope = vbroadcast<vtype>(slope);
  da_mid = __dtu_l_vldqa_s32_da(p_mid);

  int vab_shift = 0;
  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_lv_d(0);
  __dtu_c_movsr2vab_m_s2(0);
  __dtu_c_movsr2naccovr(0x10001);

  int cnt = K / 64 - 2;

#pragma clang loop unroll(disable)
  for (int k = 0; k < K; k += 64) {
    __dtu_c_movsr2vab_m_s1(0);
    __dtu_c_movsr2vab_m_d(0);
    vab_shift = 0;

    int mpr = 1 - __dtu_srli_a(cnt, 31);
    cnt--;
    __dtu_c_movsr2mpr(mpr);

    dacc1 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 1);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 2);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 3);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 4);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 5);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 6);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 7);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 8);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 9);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 10);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 11);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 12);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 13);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 14);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 15);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 16);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 17);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 18);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 19);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 20);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 21);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 22);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 23);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 24);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 25);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 26);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 27);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 28);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 29);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 30);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off1, 31);

#pragma clang loop unroll(disable)
    for (int n = 0; n < N - 64; n += 64) {
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 0);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 1);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 2);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 3);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 4);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 5);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 6);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 7);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 8);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 9);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 10);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 11);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 12);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 13);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 14);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 15);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 16);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 17);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 18);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 19);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 20);
      dacc0 = __dtu_m_vmm2_mode16_f16_nacc(dacc0, dacc1, smr0);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 21);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 22);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 23);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 24);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 25);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 26);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 27);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 28);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 29);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off0, 30);
      smr1 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr1, rt_base, rt_off1, 31);
      __dtu_v_swap_smr(smr0, smr1);
      dacc0 = __dtu_m_mpr_mop_mul_f32_da(dacc0, v_scale);

      vab_shift += 16;
      __dtu_c_movsr2vab_m_s1(vab_shift);
      __dtu_c_movsr2vab_m_d(vab_shift);
    }

    dacc0 = __dtu_m_vmm2_mode16_f16_nacc(dacc0, dacc1, smr0);
    dacc0 = __dtu_m_mpr_mop_mul_f32_da(dacc0, v_scale);
    __dtu_c_movsr2naccovr(0x1);

    rt_base = __dtu_v_taradd(rt_base, rt_off2);
  }

  vab_shift = 0;
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_s2(0);
  __dtu_c_movsr2vab_m_d(0);
  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_lv_d(0);
  if (alibi_enable != 1) {
#pragma clang loop unroll(disable)
    for (int n = 0; n < N; n += 64) {
      __dtu_v_tvstda_f32_dual(dacc0, ot_base, ot_off0);

      vab_shift += 16;
      __dtu_c_movsr2vab_lv_s(vab_shift);
    }
  } else {
    int pos = -alibi_idx;
#pragma clang loop unroll(disable)
    for (int n = 0; n < N; n += 64) {
      __dtu_c_movsr2vab_m_s2(0);
      v_ctx_len = __dtu_l_movr2da_s32(pos);
      pos += 64;
      v_mid = __dtu_m_mop_add_s32_da(v_ctx_len, da_mid);
      v_mid = __dtu_m_mop_mul_f32mix_s32_da(v_mid, v_slope);

      __dtu_c_movsr2vab_m_s2(vab_shift);
      dacc0 = __dtu_m_mop_add_f32_da(dacc0, v_mid);
      __dtu_v_tvstda_f32_dual(dacc0, ot_base, ot_off0);

      vab_shift += 16;
      __dtu_c_movsr2vab_m_s1(vab_shift);
      __dtu_c_movsr2vab_m_d(vab_shift);
      __dtu_c_movsr2vab_lv_s(vab_shift);
      __dtu_c_movsr2vab_lv_d(vab_shift);
    }
  }
}

__device__
void call_softmax_qa(float* dst_ptr, float* in_ptr, float* p_max,
                    int dim0, int ctx_len) {
  int out_addr = reinterpret_cast<long>(dst_ptr);
  int in_addr = reinterpret_cast<long>(in_ptr);
  int buffer_ptr = reinterpret_cast<long>(p_max);

  int dim_align32 = ALIGN_32(dim0);
  int dim_2048 = (dim0 + 2048 - 1) & 0xfffff800;
  const int fp32_bpe = 4;
  auto v_inf = __dtu_s_movr2vr_dup(0xff800000);
  auto padding_addr_v = reinterpret_cast<char*>(in_ptr) +
                         dim_align32 * fp32_bpe;
  auto padding_addr_w = reinterpret_cast<int*>(in_ptr) + ctx_len;

  for (int i = 0; i < (dim_align32 - ctx_len); i++) {
    padding_addr_w[i] = 0xff800000;
  }

  for (int i = 0; i < (dim_2048 - dim_align32) / 32; i++) {
    __dtu_l_vstl(v_inf, padding_addr_v + i * 128, 0);
  }

  // __attribute__((aligned(128))) char buffer[128];
  // __DTU_INTRIN_AS__ char* buffer_ptr =
  //     reinterpret_cast<__DTU_INTRIN_AS__ char*>(buffer);

  asm volatile (
    " l.movsr2targ.ext.t0 ta_g0, %0         \n"
    " l.movsr2targ.ext.t1 ta_g0, %1         \n"
    " l.movsr2tari.ext.t0 [ta_g0, 1], %2    \n"
    " l.movsr2tari.ext.t1 [ta_g0, 1], %2    \n"
    " l.movsr2tari.ext.t0 [ta_g0, 2], %3    \n"
    " l.movsr2tari.ext.t1 [ta_g0, 2], %3    \n"
    :
    : "r" (in_addr >> 8),
      "r" ((in_addr >> 8) + 1),
      "r" (2),
      "r" (-(dim_2048 >> 6))
    :
    );

  // ld exp
  asm volatile (
    " l.movsr2targ.ext.t0 ta_g1, %0         \n"
    " l.movsr2targ.ext.t1 ta_g1, %1         \n"
    " l.movsr2tari.ext.t0 [ta_g1, 1], %2    \n"
    " l.movsr2tari.ext.t1 [ta_g1, 1], %2    \n"
    :
    : "r" (out_addr >> 8),
      "r" ((out_addr >> 8) + 1),
      "r" (2)
    :
    );

  int out_tar_addr = out_addr >> 7;
  int loop_times = (dim_2048 >> 7);

  asm volatile (
    " m.smrclr smr0                       \n"
    // " l.ldi_hi r10, 0x3c00            \n"
    // " l.ldi_lo r10, 0x3c00            \n"
    " l.ldi_hi.m1 r11, 0x3f80             \n"
    // " s.movr2vr.dup vr0, r10          \n"
    " l.movr2da dacc2, r11                      \n"
    " m.ldsmr2.mode18.f.row smr0, dacc2, 0      \n"


    " m.ldsmr2.mode18.f.row smr1, dacc2, 0     \n"
    " l.movsr2targ.ext.t0 ta_g2, %0         \n"
    " m.ldsmr2.mode18.f.row smr1, dacc2, 1     \n"
    " l.movsr2targ.ext.t1 ta_g2, %1         \n"
    " m.ldsmr2.mode18.f.row smr1, dacc2, 2     \n"
    " l.movsr2targ.ext.t0 ta_g3, %0         \n"
    " m.ldsmr2.mode18.f.row smr1, dacc2, 3     \n"
    " l.movsr2targ.ext.t1 ta_g3, %1         \n"
    " m.ldsmr2.mode18.f.row smr1, dacc2, 4     \n"
    " l.movsr2tari.ext.t0 [ta_g2, 1], %2    \n"
    " m.ldsmr2.mode18.f.row smr1, dacc2, 5     \n"
    " l.movsr2tari.ext.t1 [ta_g2, 1], %2    \n"
    " m.ldsmr2.mode18.f.row smr1, dacc2, 6     \n"
    " l.movsr2tari.ext.t0 [ta_g2, 3], %3    \n"
    " m.ldsmr2.mode18.f.row smr1, dacc2, 7     \n"
    " l.movsr2tari.ext.t1 [ta_g2, 3], %3    \n"
    " m.ldsmr2.mode18.f.row smr1, dacc2, 8     \n"
    " l.movsr2tari.ext.t0 [ta_g3, 1], %2    \n"
    " m.ldsmr2.mode18.f.row smr1, dacc2, 9     \n"
    " l.movsr2tari.ext.t1 [ta_g3, 1], %2    \n"
    " m.ldsmr2.mode18.f.row smr1, dacc2, 10     \n"
    " l.movsr2tari.ext.t0 [ta_g3, 3], %3    \n"
    " m.ldsmr2.mode18.f.row smr1, dacc2, 11     \n"
    " l.movsr2tari.ext.t1 [ta_g3, 3], %3    \n"
    " m.ldsmr2.mode18.f.row smr1, dacc2, 12     \n"
    " l.ldi_hi.m1 r11, 0xff80  \n"
    " m.ldsmr2.mode18.f.row smr1, dacc2, 13     \n"
    // " l.ldi_lo r11, 0xfc00  \n"
    " m.ldsmr2.mode18.f.row smr1, dacc2, 14     \n"
    " m.ldsmr2.mode18.f.row smr1, dacc2, 15     \n"

    " m.tvldqa.w.qa qacc1, [ta_g0, 1]     \n"
    " l.movr2qa qacc28, r11                  \n"
    " m.tvldqa.w.qa qacc2, [ta_g0, 1]     \n"
    " l.movr2qa qacc29, r11                  \n"
    " m.tvldqa.w.qa qacc3, [ta_g0, 1]     \n"
    " l.movr2qa qacc30, r11                  \n"
    " m.tvldqa.w.qa qacc4, [ta_g0, 1]     \n"
    " l.movr2qa qacc31, r11                  \n"
    " m.tvldqa.w.qa qacc5, [ta_g0, 1]     \n"
    " l.movr2qa qacc32, r11                  \n"
    " m.tvldqa.w.qa qacc6, [ta_g0, 1]     \n"
    " l.movr2qa qacc33, r11                  \n"
    " m.tvldqa.w.qa qacc7, [ta_g0, 1]     \n"
    " l.movr2qa qacc34, r11                  \n"
    " m.tvldqa.w.qa qacc8, [ta_g0, 1]     \n"
    " l.movr2qa qacc35, r11                  \n"
    " m.tvldqa.w.qa qacc9, [ta_g0, 1]     \n"
    " l.movr2qa qacc36, r11                  \n"
    " m.tvldqa.w.qa qacc10, [ta_g0, 1]     \n"
    " l.movr2qa qacc37, r11                  \n"
    " m.tvldqa.w.qa qacc11, [ta_g0, 1]     \n"
    " l.movr2qa qacc38, r11                  \n"
    " m.tvldqa.w.qa qacc12, [ta_g0, 1]     \n"
    " l.movr2qa qacc39, r11                  \n"
    " m.tvldqa.w.qa qacc13, [ta_g0, 1]     \n"
    " l.movr2qa qacc40, r11                  \n"
    " m.tvldqa.w.qa qacc14, [ta_g0, 1]     \n"
    " l.movr2qa qacc41, r11                  \n"
    " m.tvldqa.w.qa qacc15, [ta_g0, 1]     \n"
    " l.movr2qa qacc42, r11                  \n"
    " m.tvldqa.w.qa qacc16, [ta_g0, 1]     \n"
    " l.movr2qa qacc43, r11                  \n"
    :
    : "r" (out_tar_addr),
      "r" (out_tar_addr + 2),
      "r" (1),
      "r" (3)
    : "r10", "r11"
    );

#pragma clang loop unroll(disable)
  for (int i = 1; i < loop_times / 16; i++) {
    asm volatile (
      " m.mop.max.f32.qa qacc28, qacc28, qacc1        \n"

      " m.mop.max.f32.qa qacc29, qacc29, qacc2        \n"
      " l.tvldqa.w.qa qacc1, [ta_g0, 1]    \n"

      " m.mop.max.f32.qa qacc30, qacc30, qacc3        \n"
      " l.tvldqa.w.qa qacc2, [ta_g0, 1]    \n"

      " m.mop.max.f32.qa qacc31, qacc31, qacc4        \n"
      " l.tvldqa.w.qa qacc3, [ta_g0, 1]    \n"

      " m.mop.max.f32.qa qacc32, qacc32, qacc5        \n"
      " l.tvldqa.w.qa qacc4, [ta_g0, 1]    \n"

      " m.mop.max.f32.qa qacc33, qacc33, qacc6        \n"
      " l.tvldqa.w.qa qacc5, [ta_g0, 1]    \n"

      " m.mop.max.f32.qa qacc34, qacc34, qacc7        \n"
      " l.tvldqa.w.qa qacc6, [ta_g0, 1]    \n"

      " m.mop.max.f32.qa qacc35, qacc35, qacc8        \n"
      " l.tvldqa.w.qa qacc7, [ta_g0, 1]    \n"

      " m.mop.max.f32.qa qacc36, qacc36, qacc9        \n"
      " l.tvldqa.w.qa qacc8, [ta_g0, 1]    \n"

      " m.mop.max.f32.qa qacc37, qacc37, qacc10        \n"
      " l.tvldqa.w.qa qacc9, [ta_g0, 1]    \n"

      " m.mop.max.f32.qa qacc38, qacc38, qacc11        \n"
      " l.tvldqa.w.qa qacc10, [ta_g0, 1]    \n"

      " m.mop.max.f32.qa qacc39, qacc39, qacc12        \n"
      " l.tvldqa.w.qa qacc11, [ta_g0, 1]    \n"

      " m.mop.max.f32.qa qacc40, qacc40, qacc13        \n"
      " l.tvldqa.w.qa qacc12, [ta_g0, 1]    \n"

      " m.mop.max.f32.qa qacc41, qacc41, qacc14        \n"
      " l.tvldqa.w.qa qacc13, [ta_g0, 1]    \n"

      " m.mop.max.f32.qa qacc42, qacc42, qacc15        \n"
      " l.tvldqa.w.qa qacc14, [ta_g0, 1]    \n"

      " m.mop.max.f32.qa qacc43, qacc43, qacc16        \n"
      " l.tvldqa.w.qa qacc15, [ta_g0, 1]    \n"

      " l.tvldqa.w.qa qacc16, [ta_g0, 1]    \n"
    );
  }

  asm volatile (
    // reduce max to qa
    " m.mop.max.f32.qa qacc28, qacc28, qacc1        \n"
    " s.taradd  ta_g0, [ta_g0, 2]   \n"
    " m.mop.max.f32.qa qacc29, qacc29, qacc2        \n"
    " m.mop.max.f32.qa qacc30, qacc30, qacc3        \n"
    " m.mop.max.f32.qa qacc31, qacc31, qacc4        \n"
    " m.mop.max.f32.qa qacc32, qacc32, qacc5        \n"
    " m.mop.max.f32.qa qacc33, qacc33, qacc6        \n"
    " m.mop.max.f32.qa qacc34, qacc34, qacc7        \n"
    " m.mop.max.f32.qa qacc35, qacc35, qacc8        \n"

    " m.mop.max.f32.qa qacc36, qacc36, qacc9        \n"
    " m.mop.max.f32.qa qacc37, qacc37, qacc10        \n"
    " m.mop.max.f32.qa qacc38, qacc38, qacc11        \n"
    " m.mop.max.f32.qa qacc39, qacc39, qacc12        \n"
    " m.mop.max.f32.qa qacc40, qacc40, qacc13        \n"
    " m.mop.max.f32.qa qacc41, qacc41, qacc14        \n"
    " m.mop.max.f32.qa qacc42, qacc42, qacc15        \n"
    " m.mop.max.f32.qa qacc43, qacc43, qacc16        \n"

    " m.mop.max.f32.qa qacc28, qacc28, qacc29        \n"
    " m.mop.max.f32.qa qacc30, qacc30, qacc31        \n"
    " m.mop.max.f32.qa qacc32, qacc32, qacc33        \n"
    " m.mop.max.f32.qa qacc34, qacc34, qacc35        \n"
    " m.mop.max.f32.qa qacc36, qacc36, qacc37        \n"
    " m.mop.max.f32.qa qacc38, qacc38, qacc39        \n"
    " m.mop.max.f32.qa qacc40, qacc40, qacc41        \n"
    " m.mop.max.f32.qa qacc42, qacc42, qacc43        \n"

    " m.mop.max.f32.qa qacc28, qacc28, qacc30        \n"
    " m.mop.max.f32.qa qacc29, qacc32, qacc34        \n"
    " m.mop.max.f32.qa qacc30, qacc36, qacc38        \n"
    " m.mop.max.f32.qa qacc31, qacc40, qacc42        \n"

    " l.tvldqa.w.qa qacc16, [ta_g0, 1]    \n"
    " l.tvldqa.w.qa qacc17, [ta_g0, 1]    \n"
    " l.tvldqa.w.qa qacc18, [ta_g0, 1]    \n"
    " l.tvldqa.w.qa qacc19, [ta_g0, 1]    \n"
    " l.tvldqa.w.qa qacc20, [ta_g0, 1]    \n"
    " l.tvldqa.w.qa qacc21, [ta_g0, 1]    \n"
    " l.tvldqa.w.qa qacc22, [ta_g0, 1]    \n"
    " l.tvldqa.w.qa qacc23, [ta_g0, 1]    \n"

    " m.mop.max.f32.qa qacc28, qacc28, qacc29        \n"
    " m.mop.max.f32.qa qacc30, qacc30, qacc31        \n"

    " m.mop.max.f32.qa qacc1, qacc28, qacc30        \n"
    " l.tvldqa.w.qa qacc24, [ta_g0, 1]    \n"
    " l.tvldqa.w.qa qacc25, [ta_g0, 1]    \n"
    " l.tvldqa.w.qa qacc26, [ta_g0, 1]    \n"
    " l.tvldqa.w.qa qacc27, [ta_g0, 1]    \n"
    " l.tvldqa.w.qa qacc28, [ta_g0, 1]    \n"
    " l.tvldqa.w.qa qacc29, [ta_g0, 1]    \n"
    " l.tvldqa.w.qa qacc30, [ta_g0, 1]    \n"
    " l.tvldqa.w.qa qacc31, [ta_g0, 1]    \n"

    " l.movva2vr.w vr0, vacc4   \n"
    " l.movva2vr.w vr1, vacc5   \n"
    " l.movva2vr.w vr2, vacc6   \n"
    " l.movva2vr.w vr3, vacc7   \n"

    " v.vmaxa.f32 vr0, vr0, vr1 \n"
    " v.vmaxa.f32 vr2, vr2, vr3 \n"
    " v.vmaxa.f32 vr16, vr0, vr2 \n"
    " l.vclr.qa  qacc109 \n"
    " l.vclr.qa  qacc110 \n"
    " l.vclr.qa  qacc111 \n"
    " l.vclr.qa  qacc112 \n"
    " l.vclr.qa  qacc113 \n"
    " l.vclr.qa  qacc114 \n"
    " l.vclr.qa  qacc115 \n"
    " l.vclr.qa  qacc116 \n"

    " l.movsfti.r.qw vr17, vr16, 1  \n"
    " l.movsfti.r.qw vr18, vr16, 2  \n"
    " l.movsfti.r.qw vr19, vr16, 3  \n"

    " v.vmaxa.f32 vr16, vr16, vr17 \n"
    " v.vmaxa.f32 vr18, vr18, vr19 \n"
    " v.vmaxa.f32 vr16, vr16, vr18 \n"

    " l.movsfti.r.b vr18, vr16, 4  \n"
    " l.movsfti.r.b vr20, vr16, 8  \n"
    " l.movsfti.r.b vr22, vr16, 12  \n"

    " v.vmaxa.f32 vr16, vr16, vr18    \n"
    " v.vmaxa.f32 vr20, vr20, vr22    \n"
    " v.vmaxa.f32 vr0, vr16, vr20     \n"
    " l.addia.u r12, %[src], 0        \n"
    " l.addia.u r13, %[src], 64       \n"

    " s.vst.0 vr0, [r12]              \n"
    " l.movs.barrier.memory r11, barrier      \n"
    " s.vld.0.ldcopy vr0, [r12]       \n"
    " s.vld.0.ldcopy vr1, [r13]       \n"
    " l.vclr.qa  qacc117 \n"
    " l.vclr.qa  qacc118 \n"
    " l.vclr.qa  qacc119 \n"
    " l.vclr.qa  qacc120 \n"
    " l.vclr.qa  qacc121 \n"
    " l.vclr.qa  qacc122 \n"
    " l.vclr.qa  qacc123 \n"
    " l.vclr.qa  qacc124 \n"
    " v.vmaxa.f32 vr0, vr0, vr1       \n"

    " m.vmm2.mode18.f.nacc dacc10, vr0, smr0    \n"
    " l.ldi_hi r11, 0x3fb8        \n"
    " l.ldi_lo r11, 0xaa3b        \n"
    " l.ldi16.s r8, 1             \n"
    " l.addia.s r10, %[cnt], -2   \n"
    " c.movsr2spr VPR, r8         \n"

    " l.movda dacc6, dacc10       \n"  // qacc3
    " l.movda dacc7, dacc10       \n"
    " l.movda dacc8, dacc10       \n"  // qacc4
    " l.movda dacc9, dacc10       \n"
    " l.movr2qa qacc7, r11        \n"
    " l.movr2qa qacc8, r11        \n"
    " v.vstla.w vacc20, [r12]     \n"


    "11: \n"
      " if (vpr) v.tvstda.w.dual dacc148, [ta_g2, 1]   \n" //      2
      " m.mop.msub.f32.qa qacc16, qacc16, qacc3     \n"   //  0 3
      " l.srlia r9, r10, 31   \n"

      " if (vpr) v.tvstda.w.dual dacc149, [ta_g2, 3]   \n" //      2
      " m.mop.msub.f32.qa qacc17, qacc17, qacc3     \n"   //  1 3
      " l.addia.s r11, r10, 1 \n"

      " if (vpr) v.tvstda.w.dual dacc154, [ta_g2, 1]   \n" //      1
      " m.mop.msub.f32.qa qacc18, qacc18, qacc4     \n"   //  2 0
      " l.ldi16.s r8, 0     \n"

      " if (vpr) v.tvstda.w.dual dacc155, [ta_g2, 3]   \n" //      1
      " m.mop.msub.f32.qa qacc19, qacc19, qacc4     \n"   //  3 0
      " l.addia.s r10, r10, -1 \n"

      " if (vpr) v.tvstda.w.dual dacc156, [ta_g2, 1]   \n" //      2
      " m.mop.msub.f32.qa qacc20, qacc20, qacc3     \n"   //  0 3

      " if (vpr) v.tvstda.w.dual dacc157, [ta_g2, 3]   \n" //      2
      " m.mop.msub.f32.qa qacc21, qacc21, qacc3     \n"   //  1 3

      " if (vpr) v.tvstda.w.dual dacc162, [ta_g2, 1]   \n" //      1
      " m.mop.msub.f32.qa qacc22, qacc22, qacc4     \n"   //  2 0

      " if (vpr) v.tvstda.w.dual dacc163, [ta_g2, 3]   \n" //      1
      " m.mop.msub.f32.qa qacc23, qacc23, qacc4     \n"   //  3 0

      " if (vpr) v.tvstda.w.dual dacc164, [ta_g2, 1]   \n" //      2
      " m.mop.msub.f32.qa qacc24, qacc24, qacc3     \n"   //  0 3

      " if (vpr) v.tvstda.w.dual dacc165, [ta_g2, 3]   \n" //      2
      " m.mop.msub.f32.qa qacc25, qacc25, qacc3     \n"   //  1 3

      " if (vpr) v.tvstda.w.dual dacc170, [ta_g2, 1]   \n" //      1
      " m.mop.msub.f32.qa qacc26, qacc26, qacc4     \n"   //  2 0

      " if (vpr) v.tvstda.w.dual dacc171, [ta_g2, 3]   \n" //      1
      " m.mop.msub.f32.qa qacc27, qacc27, qacc4     \n"   //  3 0

      " if (vpr) v.tvstda.w.dual dacc172, [ta_g2, 1]   \n" //      2
      " m.mop.msub.f32.qa qacc28, qacc28, qacc3     \n"   //  0 3

      " if (vpr) v.tvstda.w.dual dacc173, [ta_g2, 3]   \n" //      2
      " m.mop.msub.f32.qa qacc29, qacc29, qacc3     \n"   //  1 3

      " if (vpr) v.tvstda.w.dual dacc178, [ta_g2, 1]   \n" //      1
      " m.mop.msub.f32.qa qacc30, qacc30, qacc4     \n"   //  2 0

      " if (vpr) v.tvstda.w.dual dacc179, [ta_g2, 3]   \n" //      1
      " m.mop.msub.f32.qa qacc31, qacc31, qacc4     \n"   //  3 0

      " if (vpr) v.tvstda.w.dual dacc180, [ta_g2, 1]    \n" //  2
      " m.mop.mdm.f32.qa qacc40, qacc16, qacc7          \n" //  0  3
      " if (vpr) v.tvstda.w.dual dacc181, [ta_g2, 3]    \n" //  2
      " m.mop.mdm.f32.qa qacc42, qacc17, qacc7          \n" //  1  3

      " if (vpr) v.tvstda.w.dual dacc186, [ta_g2, 1]    \n" //  1
      " m.mop.mdm.f32.qa qacc44, qacc18, qacc8          \n" //  2  0
      " if (vpr) v.tvstda.w.dual dacc187, [ta_g2, 3]    \n" //  1
      " m.mop.mdm.f32.qa qacc46, qacc19, qacc8          \n" //  3  0

      " if (vpr) v.tvstda.w.dual dacc188, [ta_g2, 1]    \n" //  2
      " m.mop.mdm.f32.qa qacc48, qacc20, qacc7          \n" //  0  3
      " if (vpr) v.tvstda.w.dual dacc189, [ta_g2, 3]    \n" //  2
      " m.mop.mdm.f32.qa qacc50, qacc21, qacc7          \n" //  1  3

      " if (vpr) v.tvstda.w.dual dacc194, [ta_g2, 1]    \n" //  1
      " m.mop.mdm.f32.qa qacc52, qacc22, qacc8          \n" //  2  0
      " if (vpr) v.tvstda.w.dual dacc195, [ta_g2, 3]    \n" //  1
      " m.mop.mdm.f32.qa qacc54, qacc23, qacc8          \n" //  3  0
      " c.movsr2spr LPR, r9 \n"

      " if (vpr) v.tvstda.w.dual dacc196, [ta_g2, 1]   \n"  //  2
      " m.mop.mdm.f32.qa qacc56, qacc24, qacc7     \n"  // 0  3
      " if (vpr) v.tvstda.w.dual dacc197, [ta_g2, 3]   \n"
      " m.mop.mdm.f32.qa qacc58, qacc25, qacc7     \n"  // 1  3

      " if (vpr) v.tvstda.w.dual dacc202, [ta_g2, 1]   \n"  //  1
      " m.mop.mdm.f32.qa qacc60, qacc26, qacc8     \n"  // 2  0
      " if (vpr) v.tvstda.w.dual dacc203, [ta_g2, 3]   \n"
      " m.mop.mdm.f32.qa qacc62, qacc27, qacc8     \n"  // 3  0

      " if (vpr) v.tvstda.w.dual dacc204, [ta_g2, 1]   \n"  //  2 
      " m.mop.mdm.f32.qa qacc64, qacc28, qacc7     \n"  // 0  3
      " if (vpr) v.tvstda.w.dual dacc205, [ta_g2, 3]   \n"
      " m.mop.mdm.f32.qa qacc66, qacc29, qacc7     \n"  // 1  3

      " if (vpr) v.tvstda.w.dual dacc210, [ta_g2, 1]   \n"  //  1
      " m.mop.mdm.f32.qa qacc68, qacc30, qacc8     \n"  // 2  0
      " if (vpr) v.tvstda.w.dual dacc211, [ta_g2, 3]   \n"
      " m.mop.mdm.f32.qa qacc70, qacc31, qacc8     \n"  // 3  0
      " c.movsr2spr VPR, r8   \n"


      //  0
      " if (lpr) l.tvldqa.w.qa qacc16, [ta_g0, 1]  \n"
      " m.msf.mode0 vacc296, vacc160     \n"    //  74 40
      " m.msf.mode0 vacc297, vacc161     \n"
      " m.msf.mode0 vacc298, vacc162     \n"
      " m.msf.mode0 vacc299, vacc163     \n"
      " if (lpr) l.tvldqa.w.qa qacc17, [ta_g0, 1]  \n"
      " m.msf.mode0 vacc308, vacc168     \n"
      " m.msf.mode0 vacc309, vacc169     \n"
      " m.msf.mode0 vacc310, vacc170     \n"
      " m.msf.mode0 vacc311, vacc171     \n"

      //  1
      " if (lpr) l.tvldqa.w.qa qacc18, [ta_g0, 1]  \n"
      " m.msf.mode0 vacc312, vacc176     \n"
      " m.msf.mode0 vacc313, vacc177     \n"
      " m.msf.mode0 vacc314, vacc178     \n"
      " m.msf.mode0 vacc315, vacc179     \n"
      " if (lpr) l.tvldqa.w.qa qacc19, [ta_g0, 1]  \n"
      " m.msf.mode0 vacc324, vacc184     \n"
      " m.msf.mode0 vacc325, vacc185     \n"
      " m.msf.mode0 vacc326, vacc186     \n"
      " m.msf.mode0 vacc327, vacc187     \n"

      //  2
      " if (lpr) l.tvldqa.w.qa qacc20, [ta_g0, 1]  \n"
      " m.msf.mode0 vacc328, vacc192     \n"
      " m.msf.mode0 vacc329, vacc193     \n"
      " m.msf.mode0 vacc330, vacc194     \n"
      " m.msf.mode0 vacc331, vacc195     \n"
      " if (lpr) l.tvldqa.w.qa qacc21, [ta_g0, 1]  \n"
      " m.msf.mode0 vacc340, vacc200     \n"
      " m.msf.mode0 vacc341, vacc201     \n"
      " m.msf.mode0 vacc342, vacc202     \n"
      " m.msf.mode0 vacc343, vacc203     \n"

      //  3
      " if (lpr) l.tvldqa.w.qa qacc22, [ta_g0, 1]  \n"
      " m.msf.mode0 vacc344, vacc208     \n"
      " m.msf.mode0 vacc345, vacc209     \n"
      " m.msf.mode0 vacc346, vacc210     \n"
      " m.msf.mode0 vacc347, vacc211     \n"
      " if (lpr) l.tvldqa.w.qa qacc23, [ta_g0, 1]  \n"
      " m.msf.mode0 vacc356, vacc216     \n"
      " m.msf.mode0 vacc357, vacc217     \n"
      " m.msf.mode0 vacc358, vacc218     \n"
      " m.msf.mode0 vacc359, vacc219     \n"

      //  4
      " m.msf.mode0 vacc360, vacc224     \n"
      " if (lpr) l.tvldqa.w.qa qacc24, [ta_g0, 1]  \n"
      " m.msf.mode0 vacc361, vacc225     \n"
      " if (lpr) l.tvldqa.w.qa qacc25, [ta_g0, 1]  \n"
      " m.msf.mode0 vacc362, vacc226     \n"
      " if (lpr) l.tvldqa.w.qa qacc26, [ta_g0, 1]  \n"
      " m.msf.mode0 vacc363, vacc227     \n"
      " if (lpr) l.tvldqa.w.qa qacc27, [ta_g0, 1]  \n"

      " m.msf.mode0 vacc372, vacc232     \n"
      " if (lpr) l.tvldqa.w.qa qacc28, [ta_g0, 1]  \n"
      " m.msf.mode0 vacc373, vacc233     \n"
      " if (lpr) l.tvldqa.w.qa qacc29, [ta_g0, 1]  \n"
      " m.msf.mode0 vacc374, vacc234     \n"
      " if (lpr) l.tvldqa.w.qa qacc30, [ta_g0, 1]  \n"
      " m.msf.mode0 vacc375, vacc235     \n"
      " if (lpr) l.tvldqa.w.qa qacc31, [ta_g0, 1]  \n"

      //  5
      " m.msf.mode0 vacc376, vacc240     \n"
      " m.msf.mode0 vacc377, vacc241     \n"
      " m.msf.mode0 vacc378, vacc242     \n"
      " m.msf.mode0 vacc379, vacc243     \n"
      " m.msf.mode0 vacc388, vacc248     \n"
      " m.msf.mode0 vacc389, vacc249     \n"
      " m.msf.mode0 vacc390, vacc250     \n"
      " m.msf.mode0 vacc391, vacc251     \n"

      //  6
      " m.msf.mode0 vacc392, vacc256     \n"
      " m.msf.mode0 vacc393, vacc257     \n"
      " m.msf.mode0 vacc394, vacc258     \n"
      " m.msf.mode0 vacc395, vacc259     \n"
      " m.msf.mode0 vacc404, vacc264     \n"
      " m.msf.mode0 vacc405, vacc265     \n"
      " m.msf.mode0 vacc406, vacc266     \n"
      " m.msf.mode0 vacc407, vacc267     \n"

      //  7
      " m.msf.mode0 vacc408, vacc272     \n"
      " m.msf.mode0 vacc409, vacc273     \n"
      " m.msf.mode0 vacc410, vacc274     \n"
      " m.msf.mode0 vacc411, vacc275     \n"
      " m.msf.mode0 vacc420, vacc280     \n"  //  105 70
      " m.msf.mode0 vacc421, vacc281     \n"
      " m.msf.mode0 vacc422, vacc282     \n"
      " m.msf.mode0 vacc423, vacc283     \n"

      " m.mop.madd.f32.qa qacc109, qacc109, qacc74     \n"   //  1 2
      " m.mop.madd.f32.qa qacc110, qacc110, qacc77     \n"   //  2 1
      " m.mop.madd.f32.qa qacc111, qacc111, qacc78     \n"  //  3 2
      " m.mop.madd.f32.qa qacc112, qacc112, qacc81     \n"  //  0 1
      " m.mop.madd.f32.qa qacc113, qacc113, qacc82     \n"
      " m.mop.madd.f32.qa qacc114, qacc114, qacc85     \n"
      " m.mop.madd.f32.qa qacc115, qacc115, qacc86     \n"
      " m.mop.madd.f32.qa qacc116, qacc116, qacc89     \n"
      " m.mop.madd.f32.qa qacc117, qacc117, qacc90     \n"
      " m.mop.madd.f32.qa qacc118, qacc118, qacc93     \n"
      " m.mop.madd.f32.qa qacc119, qacc119, qacc94     \n"
      " m.mop.madd.f32.qa qacc120, qacc120, qacc97     \n"
      " m.mop.madd.f32.qa qacc121, qacc121, qacc98     \n"
      " m.mop.madd.f32.qa qacc122, qacc122, qacc101     \n"
      " m.mop.madd.f32.qa qacc123, qacc123, qacc102     \n"
      " m.mop.madd.f32.qa qacc124, qacc124, qacc105     \n"

      " l.bne r0, r11, 11b   \n"
      :
      : [ cnt ] "r" (loop_times / 16),
        [ src ] "r" (buffer_ptr)
      : "r8", "r9", "r10", "r11", "r12", "r13"
    );

    asm volatile(
      " v.tvstda.w.dual dacc148, [ta_g2, 1]   \n" //      2
      " m.mop.madd.f32.qa qacc109, qacc109, qacc110     \n"
      " v.tvstda.w.dual dacc149, [ta_g2, 3]   \n" //      2
      " m.mop.madd.f32.qa qacc111, qacc111, qacc112     \n"  //  0 1

      " v.tvstda.w.dual dacc154, [ta_g2, 1]   \n" //      1
      " m.mop.madd.f32.qa qacc113, qacc113, qacc114     \n"
      " v.tvstda.w.dual dacc155, [ta_g2, 3]   \n" //      1
      " m.mop.madd.f32.qa qacc115, qacc115, qacc116     \n"  //  0 1

      " v.tvstda.w.dual dacc156, [ta_g2, 1]   \n" //      2
      " m.mop.madd.f32.qa qacc117, qacc117, qacc118     \n"
      " v.tvstda.w.dual dacc157, [ta_g2, 3]   \n" //      2
      " m.mop.madd.f32.qa qacc119, qacc119, qacc120     \n"  //  0 1

      " v.tvstda.w.dual dacc162, [ta_g2, 1]   \n" //      1
      " m.mop.madd.f32.qa qacc121, qacc121, qacc122     \n"
      " v.tvstda.w.dual dacc163, [ta_g2, 3]   \n" //      1
      " m.mop.madd.f32.qa qacc123, qacc123, qacc124     \n"  //  0 1

      " v.tvstda.w.dual dacc164, [ta_g2, 1]   \n" //      2
      " v.tvstda.w.dual dacc165, [ta_g2, 3]   \n" //      2
      " v.tvstda.w.dual dacc170, [ta_g2, 1]   \n" //      1
      " m.mop.madd.f32.qa qacc1, qacc109, qacc111       \n"
      " v.tvstda.w.dual dacc171, [ta_g2, 3]   \n" //      1
      " m.mop.madd.f32.qa qacc2, qacc113, qacc115       \n"
      " v.tvstda.w.dual dacc172, [ta_g2, 1]   \n" //      2
      " m.mop.madd.f32.qa qacc3, qacc117, qacc119       \n"
      " v.tvstda.w.dual dacc173, [ta_g2, 3]   \n" //      2
      " m.mop.madd.f32.qa qacc4, qacc121, qacc123       \n"
      " v.tvstda.w.dual dacc178, [ta_g2, 1]   \n" //      1
      " v.tvstda.w.dual dacc179, [ta_g2, 3]   \n" //      1

      " v.tvstda.w.dual dacc180, [ta_g2, 1]   \n" //      2
      " v.tvstda.w.dual dacc181, [ta_g2, 3]   \n" //      2
      " v.tvstda.w.dual dacc186, [ta_g2, 1]   \n" //      3
      " v.tvstda.w.dual dacc187, [ta_g2, 3]   \n" //      3
      " m.mop.madd.f32.qa qacc1, qacc1, qacc2     \n"
      " v.tvstda.w.dual dacc188, [ta_g2, 1]   \n" //      0
      " m.mop.madd.f32.qa qacc3, qacc3, qacc4     \n"
      " v.tvstda.w.dual dacc189, [ta_g2, 3]   \n" //      0
      " v.tvstda.w.dual dacc194, [ta_g2, 1]   \n" //      1
      " v.tvstda.w.dual dacc195, [ta_g2, 3]   \n" //      1

      " v.tvstda.w.dual dacc196, [ta_g2, 1]   \n" //      2
      " v.tvstda.w.dual dacc197, [ta_g2, 3]   \n" //      2
      " v.tvstda.w.dual dacc202, [ta_g2, 1]   \n" //      3
      " v.tvstda.w.dual dacc203, [ta_g2, 3]   \n" //      3
      " v.tvstda.w.dual dacc204, [ta_g2, 1]   \n" //      0
      " v.tvstda.w.dual dacc205, [ta_g2, 3]   \n" //      0
      " m.mop.madd.f32.qa qacc1, qacc1, qacc3     \n"
      " v.tvstda.w.dual dacc210, [ta_g2, 1]   \n" //      1
      " v.tvstda.w.dual dacc211, [ta_g2, 3]   \n" //      1

      " m.mop.madd.f32.va vacc4, vacc4, vacc5         \n"
      " m.mop.madd.f32.va vacc6, vacc6, vacc7         \n"
      " m.mop.madd.f32.va vacc4, vacc4, vacc6         \n"

      " l.addia.u r12, %[src], 0              \n"
      " l.addia.u r13, %[src], 64             \n"

      " l.movva2vr.w vr0, vacc4     \n"
      " s.vst.0 vr0, [r12]          \n"
      " l.movs.barrier.memory r11, barrier      \n"
      " s.vld.0.ldcopy vr0, [r12]   \n"
      " s.vld.0.ldcopy vr1, [r13]   \n"
      " v.vadda.f32 vr0, vr0, vr1   \n"

      // smr1
      " m.vmm2.mode18.f.nacc dacc2, vr0, smr1    \n"
      // " m.vmm2.mode18.f.nacc dacc3, vr0, smr1    \n"

      // " l.tvldqa.w.qa qacc18, [ta_g1, 1]   \n"  //  2
      // " l.tvldqa.w.qa qacc19, [ta_g1, 1]   \n"  //  3
      // " l.tvldqa.w.qa qacc20, [ta_g1, 1]   \n"  //  0
      // " l.tvldqa.w.qa qacc21, [ta_g1, 1]   \n"  //  1
      // " l.tvldqa.w.qa qacc22, [ta_g1, 1]   \n"  //  2
      // " l.tvldqa.w.qa qacc23, [ta_g1, 1]   \n"  //  3
      // " l.tvldqa.w.qa qacc24, [ta_g1, 1]   \n"  //  0
      // " l.tvldqa.w.qa qacc25, [ta_g1, 1]   \n"  //  1

      // " m.msf.mode6 vacc4, vacc4    \n"
      " v.vstla.w vacc4, [r12]      \n"
      // " m.msf.mode6 vacc5, vacc5   \n"
      // " m.msf.mode6 vacc6, vacc6   \n"
      // " m.msf.mode6 vacc7, vacc7   \n"

      // " l.tvldqa.w.qa qacc26, [ta_g1, 1]   \n"  //  2
      // " l.tvldqa.w.qa qacc27, [ta_g1, 1]   \n"  //  3
      // " l.tvldqa.w.qa qacc28, [ta_g1, 1]   \n"  //  0
      // " l.tvldqa.w.qa qacc29, [ta_g1, 1]   \n"  //  1
      // " l.tvldqa.w.qa qacc30, [ta_g1, 1]   \n"  //  2
      // " l.tvldqa.w.qa qacc31, [ta_g1, 1]   \n"  //  3
      // " l.tvldqa.w.qa qacc32, [ta_g1, 1]   \n"  //  0
      // " l.tvldqa.w.qa qacc33, [ta_g1, 1]   \n"  //  1

      // " m.mop.cvt.qa.rne.f32f16 dacc4, qacc1     \n"  // qa2
      // " l.ldi16.s r8, 1     \n"
      // " m.mop.cvt.qa.rne.f32f16 dacc5, qacc1     \n"
      // " l.addia.s r10, %[cnt], -2    \n"

      // " m.mop.cvt.qa.rne.f32f16 dacc6, qacc1     \n"  // qa3
      // " c.movsr2spr VPR, r8   \n"
      // " m.mop.cvt.qa.rne.f32f16 dacc7, qacc1     \n"
      // " m.mop.cvt.qa.rne.f32f16 dacc8, qacc1     \n"  // qa4
      // " m.mop.cvt.qa.rne.f32f16 dacc9, qacc1     \n"
      // " m.mop.cvt.qa.rne.f32f16 dacc10, qacc1     \n" // qa5
      // " m.mop.cvt.qa.rne.f32f16 dacc11, qacc1     \n"


      // "12: \n"
      // " if (vpr) v.tvstda.w.dual dacc72, [ta_g3, 1]    \n"
      // " l.srlia r9, r10, 31   \n"
      // " if (vpr) v.tvstda.w.dual dacc73, [ta_g3, 3]    \n"
      // " l.addia.s r11, r10, 1 \n"
      // " if (vpr) v.tvstda.w.dual dacc76, [ta_g3, 1]    \n"
      // " l.ldi16.s r8, 0     \n"
      // " if (vpr) v.tvstda.w.dual dacc77, [ta_g3, 3]    \n"
      // " l.addia.s r10, r10, -1    \n"

      // " if (vpr) v.tvstda.w.dual dacc80, [ta_g3, 1]    \n"
      // " if (vpr) v.tvstda.w.dual dacc81, [ta_g3, 3]    \n"
      // " if (vpr) v.tvstda.w.dual dacc84, [ta_g3, 1]    \n"
      // " if (vpr) v.tvstda.w.dual dacc85, [ta_g3, 3]    \n"

      // " if (vpr) v.tvstda.w.dual dacc88, [ta_g3, 1]    \n"
      // " if (vpr) v.tvstda.w.dual dacc89, [ta_g3, 3]    \n"
      // " if (vpr) v.tvstda.w.dual dacc92, [ta_g3, 1]    \n"
      // " if (vpr) v.tvstda.w.dual dacc93, [ta_g3, 3]    \n"
      // " c.movsr2spr LPR, r9   \n"

      // " if (vpr) v.tvstda.w.dual dacc96, [ta_g3, 1]    \n"
      // " if (vpr) v.tvstda.w.dual dacc97, [ta_g3, 3]    \n"
      // " if (vpr) v.tvstda.w.dual dacc100, [ta_g3, 1]    \n"
      // " if (vpr) v.tvstda.w.dual dacc101, [ta_g3, 3]    \n"

      // " if (vpr) v.tvstda.w.dual dacc104, [ta_g3, 1]    \n" //  0
      // " m.mop.mdm.f16.qa qacc36, qacc18, qacc3    \n"  //  2 3
      // " if (lpr) l.tvldqa.w.qa qacc18, [ta_g1, 1]      \n"

      // " if (vpr) v.tvstda.w.dual dacc105, [ta_g3, 3]    \n" //  0
      // " m.mop.mdm.f16.qa qacc38, qacc19, qacc2    \n"  //  3 2
      // " if (lpr) l.tvldqa.w.qa qacc19, [ta_g1, 1]      \n"

      // " if (vpr) v.tvstda.w.dual dacc108, [ta_g3, 1]    \n" //  2
      // " m.mop.mdm.f16.qa qacc40, qacc20, qacc5    \n"  //  0 1
      // " if (lpr) l.tvldqa.w.qa qacc20, [ta_g1, 1]      \n"

      // " if (vpr) v.tvstda.w.dual dacc109, [ta_g3, 3]    \n" //  2
      // " m.mop.mdm.f16.qa qacc42, qacc21, qacc4    \n"  //  1 0
      // " if (lpr) l.tvldqa.w.qa qacc21, [ta_g1, 1]      \n"

      // " if (vpr) v.tvstda.w.dual dacc112, [ta_g3, 1]    \n" //  0
      // " m.mop.mdm.f16.qa qacc44, qacc22, qacc3    \n"  //  2 3
      // " if (lpr) l.tvldqa.w.qa qacc22, [ta_g1, 1]      \n"

      // " if (vpr) v.tvstda.w.dual dacc113, [ta_g3, 3]    \n" //  0
      // " m.mop.mdm.f16.qa qacc46, qacc23, qacc2    \n"  //  3 2
      // " if (lpr) l.tvldqa.w.qa qacc23, [ta_g1, 1]      \n"

      // " if (vpr) v.tvstda.w.dual dacc116, [ta_g3, 1]    \n" //  2
      // " m.mop.mdm.f16.qa qacc48, qacc24, qacc5    \n"  //  0 1
      // " if (lpr) l.tvldqa.w.qa qacc24, [ta_g1, 1]      \n"

      // " if (vpr) v.tvstda.w.dual dacc117, [ta_g3, 3]    \n" //  2
      // " m.mop.mdm.f16.qa qacc50, qacc25, qacc4    \n"  //  1 0
      // " if (lpr) l.tvldqa.w.qa qacc25, [ta_g1, 1]      \n"

      // " if (vpr) v.tvstda.w.dual dacc120, [ta_g3, 1]    \n"
      // " m.mop.mdm.f16.qa qacc52, qacc26, qacc3    \n"
      // " if (lpr) l.tvldqa.w.qa qacc26, [ta_g1, 1]      \n"

      // " if (vpr) v.tvstda.w.dual dacc121, [ta_g3, 3]    \n"
      // " m.mop.mdm.f16.qa qacc54, qacc27, qacc2    \n"
      // " if (lpr) l.tvldqa.w.qa qacc27, [ta_g1, 1]      \n"

      // " if (vpr) v.tvstda.w.dual dacc124, [ta_g3, 1]    \n"
      // " m.mop.mdm.f16.qa qacc56, qacc28, qacc5    \n"
      // " if (lpr) l.tvldqa.w.qa qacc28, [ta_g1, 1]      \n"

      // " if (vpr) v.tvstda.w.dual dacc125, [ta_g3, 3]    \n"
      // " m.mop.mdm.f16.qa qacc58, qacc29, qacc4    \n"
      // " if (lpr) l.tvldqa.w.qa qacc29, [ta_g1, 1]      \n"

      // " if (vpr) v.tvstda.w.dual dacc128, [ta_g3, 1]    \n"
      // " m.mop.mdm.f16.qa qacc60, qacc30, qacc3    \n"
      // " if (lpr) l.tvldqa.w.qa qacc30, [ta_g1, 1]      \n"

      // " if (vpr) v.tvstda.w.dual dacc129, [ta_g3, 3]    \n"
      // " m.mop.mdm.f16.qa qacc62, qacc31, qacc2    \n"
      // " if (lpr) l.tvldqa.w.qa qacc31, [ta_g1, 1]      \n"

      // " if (vpr) v.tvstda.w.dual dacc132, [ta_g3, 1]    \n"
      // " m.mop.mdm.f16.qa qacc64, qacc32, qacc5    \n"
      // " if (lpr) l.tvldqa.w.qa qacc32, [ta_g1, 1]       \n"

      // " if (vpr) v.tvstda.w.dual dacc133, [ta_g3, 3]    \n"
      // " m.mop.mdm.f16.qa qacc66, qacc33, qacc4    \n"
      // " if (lpr) l.tvldqa.w.qa qacc33, [ta_g1, 1]       \n"
      // " c.movsr2spr VPR, r8   \n"
      // " l.bne r0, r11, 12b   \n"
      :
      : [ cnt ] "r" (loop_times / 16),
        [ src ] "r" (buffer_ptr + 128)
      : "r8", "r9", "r10", "r11", "r12", "r13"
      );

    // asm volatile(
    //   " v.tvstda.w.dual dacc72, [ta_g3, 1]    \n"
    //   " v.tvstda.w.dual dacc73, [ta_g3, 3]    \n"
    //   " v.tvstda.w.dual dacc76, [ta_g3, 1]    \n"
    //   " v.tvstda.w.dual dacc77, [ta_g3, 3]    \n"
    //   " v.tvstda.w.dual dacc80, [ta_g3, 1]    \n"
    //   " v.tvstda.w.dual dacc81, [ta_g3, 3]    \n"
    //   " v.tvstda.w.dual dacc84, [ta_g3, 1]    \n"
    //   " v.tvstda.w.dual dacc85, [ta_g3, 3]    \n"

    //   " v.tvstda.w.dual dacc88, [ta_g3, 1]    \n"
    //   " v.tvstda.w.dual dacc89, [ta_g3, 3]    \n"
    //   " v.tvstda.w.dual dacc92, [ta_g3, 1]    \n"
    //   " v.tvstda.w.dual dacc93, [ta_g3, 3]    \n"
    //   " v.tvstda.w.dual dacc96, [ta_g3, 1]    \n"
    //   " v.tvstda.w.dual dacc97, [ta_g3, 3]    \n"
    //   " v.tvstda.w.dual dacc100, [ta_g3, 1]    \n"
    //   " v.tvstda.w.dual dacc101, [ta_g3, 3]    \n"

    //   " v.tvstda.w.dual dacc104, [ta_g3, 1]    \n" //  1
    //   " v.tvstda.w.dual dacc105, [ta_g3, 3]    \n" //  1
    //   " v.tvstda.w.dual dacc108, [ta_g3, 1]    \n" //  2
    //   " v.tvstda.w.dual dacc109, [ta_g3, 3]    \n" //  2
    //   " v.tvstda.w.dual dacc112, [ta_g3, 1]    \n" //  3
    //   " v.tvstda.w.dual dacc113, [ta_g3, 3]    \n" //  3
    //   " v.tvstda.w.dual dacc116, [ta_g3, 1]    \n" //  0
    //   " v.tvstda.w.dual dacc117, [ta_g3, 3]    \n" //  0

    //   " v.tvstda.w.dual dacc120, [ta_g3, 1]    \n"
    //   " v.tvstda.w.dual dacc121, [ta_g3, 3]    \n"
    //   " v.tvstda.w.dual dacc124, [ta_g3, 1]    \n"
    //   " v.tvstda.w.dual dacc125, [ta_g3, 3]    \n"
    //   " v.tvstda.w.dual dacc128, [ta_g3, 1]    \n"
    //   " v.tvstda.w.dual dacc129, [ta_g3, 3]    \n"
    //   " v.tvstda.w.dual dacc132, [ta_g3, 1]    \n"
    //   " v.tvstda.w.dual dacc133, [ta_g3, 3]    \n"
    //   );
}

template <>
__device__ void __attribute__((dtu_maxinum_vacc(16))) call_k_dot_kn(
    tops::half* out, float* lhs, tops::half* rhs, float* p_max,
    int K, int N, int softmax_offset, int ctx_len, int st_ind) {
  int out_addr = reinterpret_cast<long>(out);
  int lhs_addr = reinterpret_cast<long>(lhs);
  int rhs_addr = reinterpret_cast<long>(rhs);

  int k = ctx_len - softmax_offset;
  auto vr_0 = __dtu_s_movr2vr_dup(0);
  if (K > k) {
    int loop_times = ((K - k) * N) >> 6;
    char* st_addr = reinterpret_cast<char*>(rhs + k * N);
    for (int i = 0; i < loop_times; i++) {
      __dtu_l_vstl(vr_0, st_addr + 128 * i, 0);
    }
  }

  int naccovr = 1;
  va16f32x4 qacc0;
  if (softmax_offset == 0) {
    naccovr = 0x10001;
    qacc0 = __dtu_l_vclr_f32_qa();
  }

  auto k_unit = K >> 5;
  auto n_unit = N >> 6;

  lhs_addr = lhs_addr >> 7;
  rhs_addr = rhs_addr >> 7;

  int rhs_off = 1;
  if (N <= 64) { rhs_off = 0; }

  tar_t lt_base = __dtu_c_movsr2targ(TAR32(lhs_addr, lhs_addr));
  int offset = TAR32(1, 1);
  tar_t lt_off0 = __dtu_c_movsr2tari(offset, lt_base);
  offset = TAR32(-k_unit, -k_unit);
  tar_t lt_off1 = __dtu_c_movsr2tari(offset, lt_base);

  tar_t rt_base = __dtu_c_movsr2targ(TAR32(rhs_addr, rhs_addr + rhs_off));
  offset = TAR32(n_unit, n_unit);
  tar_t rt_off0 = __dtu_c_movsr2tari(offset, rt_base);
  offset = TAR32(2 - (K) * n_unit, 2 - (K) * n_unit);
  tar_t rt_off1 = __dtu_c_movsr2tari(offset, rt_base);

  auto va0 = __dtu_l_vldqa_copy_f32_va(reinterpret_cast<char*>(p_max));
  va0 = __dtu_m_msf_rec_f32(va0);
  auto da = __dtu_insertva2da(va0, va0);
  auto v_rec = __dtu_l_movva2vr_cvt2fp16(da);

  smr_t smr0, smr1;

  int vab_shift = 0;
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_d(0);

#pragma clang loop unroll(disable)
  for (int n = 0; n < N; n += 128) {
    auto da0 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    auto vr0 = __dtu_l_movva2vr_cvt2fp16(da0);
    vr0 = __dtu_v_vmul_a_f16(vr0, v_rec);
    __dtu_c_movsr2naccovr(naccovr);

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

    auto da1 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    auto vr1 = __dtu_l_movva2vr_cvt2fp16(da1);
    vr1 = __dtu_v_vmul_a_f16(vr1, v_rec);
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

#pragma clang loop unroll(disable)
    for (int k = 0; k < K - 64; k += 64) {
      qacc0 = __dtu_m_vmm2_mode17_f16_nacc(qacc0, vr0, smr0);
      __dtu_c_movsr2naccovr(0x1);

      da0 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      vr0 = __dtu_l_movva2vr_cvt2fp16(da0);
      vr0 = __dtu_v_vmul_a_f16(vr0, v_rec);
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

      qacc0 = __dtu_m_vmm2_mode17_f16_nacc(qacc0, vr1, smr1);
      da1 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      vr1 = __dtu_l_movva2vr_cvt2fp16(da1);
      vr1 = __dtu_v_vmul_a_f16(vr1, v_rec);
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
    }

    qacc0 = __dtu_m_vmm2_mode17_f16_nacc(qacc0, vr0, smr0);
    __dtu_c_movsr2naccovr(0x1);
    qacc0 = __dtu_m_vmm2_mode17_f16_nacc(qacc0, vr1, smr1);

    vab_shift += 16;
    __dtu_c_movsr2vab_m_s1(vab_shift);
    __dtu_c_movsr2vab_m_d(vab_shift);

    lt_base = __dtu_v_taradd(lt_base, lt_off1);
    rt_base = __dtu_v_taradd(rt_base, rt_off1);
  }

  if (st_ind != NO_OUT) {
    vab_shift = 0;
    __dtu_c_movsr2vab_lv_s(0);

    out_addr = out_addr >> 7;
    tar_t ot_base = __dtu_c_movsr2targ(TAR32(out_addr, out_addr + 2));
    offset = TAR32(1, 1);
    tar_t ot_off0 = __dtu_c_movsr2tari(offset, ot_base);
    offset = TAR32(3, 3);
    tar_t ot_off1 = __dtu_c_movsr2tari(offset, ot_base);

#pragma clang loop unroll(disable)
    for (int n = 0; n < N; n += 128) {
      auto dacc2 = __dtu_extractqa2da(qacc0, 0);
      auto dacc3 = __dtu_extractqa2da(qacc0, 1);

      __dtu_v_tvstda_f32_dual(dacc2, ot_base, ot_off0);
      __dtu_v_tvstda_f32_dual(dacc3, ot_base, ot_off1);

      vab_shift += 16;
      __dtu_c_movsr2vab_lv_s(vab_shift);
    }
  }
}

template <>
__device__ void call_k_dot_nk(
    float* out, int attn_offset, tops::bfloat* lhs, tops::bfloat* rhs,
    int K, int N, float scale, char *p_mid,
    float slope, int alibi_idx, int alibi_enable) {
  int out_addr = reinterpret_cast<long>(out) + attn_offset * 4;
  int lhs_addr = reinterpret_cast<long>(lhs);
  int rhs_addr = reinterpret_cast<long>(rhs);

  auto k_unit = K >> 5;
  lhs_addr = lhs_addr >> 6;
  rhs_addr = rhs_addr >> 6;
  out_addr = out_addr >> 7;

  tar_t lt_base =
      __dtu_c_movsr2targ(TAR32(lhs_addr, lhs_addr));
  int offset = TAR32(1, 1);
  tar_t lt_off0 = __dtu_c_movsr2tari(offset, lt_base);

  auto next_stride = 1 - k_unit * N;
  tar_t rt_base =
      __dtu_c_movsr2targ(TAR32(rhs_addr, rhs_addr + 32 * k_unit));
  offset = TAR32(k_unit, k_unit);
  tar_t rt_off0 = __dtu_c_movsr2tari(offset, rt_base);
  offset = TAR32(k_unit + k_unit * 32, k_unit + k_unit * 32);
  tar_t rt_off1 = __dtu_c_movsr2tari(offset, rt_base);
  offset = TAR32(next_stride, next_stride);
  tar_t rt_off2 = __dtu_c_movsr2tari(offset, rt_base);

  tar_t ot_base = __dtu_c_movsr2targ(TAR32(out_addr, out_addr + 1));
  offset = TAR32(2, 2);
  tar_t ot_off0 = __dtu_c_movsr2tari(offset, ot_base);

  smr_t smr0, smr1;
  va16f32x2 dacc0;
  va16f32x2 v_mid, v_ctx_len;
  using vtype = typename scalar_to_vector<float, TOPS_VECTOR_LENGTH/2>::type;
  vtype v_scale = vbroadcast<vtype>(scale);
  vtype v_slope = vbroadcast<vtype>(slope);
  auto da_mid = __dtu_l_vldqa_s32_da(p_mid);

  int cnt = K / 32 - 2;
  int vab_shift = 0;
  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_lv_d(0);
  __dtu_c_movsr2vab_m_s2(0);
  __dtu_c_movsr2naccovr(0x10001);

  for (int k = 0; k < K; k += 32) {
    __dtu_c_movsr2vab_m_s1(0);
    __dtu_c_movsr2vab_m_d(0);
    vab_shift = 0;

    int mpr = 1 - __dtu_srli_a(cnt, 31);
    cnt--;
    __dtu_c_movsr2mpr(mpr);

    auto vacc1 = __dtu_l_tvlda(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 1);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 2);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 3);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 4);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 5);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 6);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 7);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 8);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 9);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 10);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 11);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 12);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 13);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 14);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 15);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 16);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 17);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 18);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 19);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 20);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 21);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 22);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 23);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 24);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 25);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 26);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 27);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 28);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 29);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off0, 30);
    smr0 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr0, rt_base, rt_off1, 31);

#pragma clang loop unroll(disable)
    for (int n = 0; n < N - 64; n += 64) {
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 0);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 1);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 2);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 3);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 4);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 5);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 6);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 7);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 8);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 9);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 10);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 11);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 12);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 13);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 14);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 15);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 16);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 17);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 18);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 19);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 20);
      dacc0 = __dtu_m_vmm_mode0_bf_nacc_vs0(dacc0, vacc1, smr0);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 21);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 22);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 23);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 24);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 25);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 26);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 27);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 28);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 29);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off0, 30);
      smr1 = __dtu_v_ldsmr2_mem_v_mode0_bf16_col(smr1, rt_base, rt_off1, 31);

      __dtu_v_swap_smr(smr0, smr1);
      dacc0 = __dtu_m_mpr_mop_mul_f32_da(dacc0, v_scale);

      vab_shift += 16;
      __dtu_c_movsr2vab_m_s1(vab_shift);
      __dtu_c_movsr2vab_m_d(vab_shift);
    }
    dacc0 = __dtu_m_vmm_mode0_bf_nacc_vs0(dacc0, vacc1, smr0);
    dacc0 = __dtu_m_mpr_mop_mul_f32_da(dacc0, v_scale);

    __dtu_c_movsr2naccovr(0x1);
    rt_base = __dtu_v_taradd(rt_base, rt_off2);
  }

  vab_shift = 0;
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_s2(0);
  __dtu_c_movsr2vab_m_d(0);
  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_lv_d(0);
  if (alibi_enable != 1) {
#pragma clang loop unroll(disable)
    for (int n = 0; n < N; n += 64) {
      __dtu_v_tvstda_f32_dual(dacc0, ot_base, ot_off0);

      vab_shift += 16;
      __dtu_c_movsr2vab_lv_s(vab_shift);
    }
  } else {
    int pos = -alibi_idx;
#pragma clang loop unroll(disable)
    for (int n = 0; n < N; n += 64) {
      __dtu_c_movsr2vab_m_s2(0);
      v_ctx_len = __dtu_l_movr2da_s32(pos);
      pos += 64;
      v_mid = __dtu_m_mop_add_s32_da(v_ctx_len, da_mid);
      v_mid = __dtu_m_mop_mul_f32mix_s32_da(v_mid, v_slope);

      __dtu_c_movsr2vab_m_s2(vab_shift);
      dacc0 = __dtu_m_mop_add_f32_da(dacc0, v_mid);
      __dtu_v_tvstda_f32_dual(dacc0, ot_base, ot_off0);

      vab_shift += 16;
      __dtu_c_movsr2vab_m_s1(vab_shift);
      __dtu_c_movsr2vab_m_d(vab_shift);
      __dtu_c_movsr2vab_lv_s(vab_shift);
      __dtu_c_movsr2vab_lv_d(vab_shift);
    }
  }
}

template <>
__device__ void __attribute__((dtu_maxinum_vacc(16))) call_k_dot_kn(
    tops::bfloat* out, float* lhs, tops::bfloat* rhs, float * p_max,
    int K, int N, int softmax_offset, int ctx_len, int st_ind) {
  int out_addr = reinterpret_cast<long>(out);
  int lhs_addr = reinterpret_cast<long>(lhs);
  int rhs_addr = reinterpret_cast<long>(rhs);

  int k = ctx_len - softmax_offset;
  auto vr_0 = __dtu_s_movr2vr_dup(0);
  if (K > k) {
    int loop_times = ((K - k) * N) >> 6;
    char* st_addr = reinterpret_cast<char*>(rhs + k * N);
    for (int i = 0; i < loop_times; i++) {
      __dtu_l_vstl(vr_0, st_addr + 128 * i, 0);
    }
  }

  int naccovr = 1;
  va16f32x4 qacc0;
  if (softmax_offset == 0) {
    naccovr = 0x10001;
    qacc0 = __dtu_l_vclr_f32_qa();
  }

  auto k_unit = K >> 5;
  auto n_unit = N >> 6;

  lhs_addr = lhs_addr >> 7;
  rhs_addr = rhs_addr >> 7;

  int rhs_off = 1;
  if (N <= 64) { rhs_off = 0; }

  tar_t lt_base = __dtu_c_movsr2targ(TAR32(lhs_addr, lhs_addr));
  int offset = TAR32(1, 1);
  tar_t lt_off0 = __dtu_c_movsr2tari(offset, lt_base);
  offset = TAR32(-k_unit, -k_unit);
  tar_t lt_off1 = __dtu_c_movsr2tari(offset, lt_base);

  tar_t rt_base = __dtu_c_movsr2targ(TAR32(rhs_addr, rhs_addr + rhs_off));
  offset = TAR32(n_unit, n_unit);
  tar_t rt_off0 = __dtu_c_movsr2tari(offset, rt_base);
  offset = TAR32(2 - (K) * n_unit, 2 - (K) * n_unit);
  tar_t rt_off1 = __dtu_c_movsr2tari(offset, rt_base);

  auto va0 = __dtu_l_vldqa_copy_f32_va(reinterpret_cast<char*>(p_max));
  va0 = __dtu_m_msf_rec_f32(va0);
  auto da = __dtu_insertva2da(va0, va0);
  auto v_rec = __dtu_l_movva2vr_cvt2bf16(da);

  smr_t smr0, smr1;

  int vab_shift = 0;
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_d(0);

#pragma clang loop unroll(disable)
  for (int n = 0; n < N; n += 128) {
    auto da0 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    auto vr0 = __dtu_l_movva2vr_cvt2bf16(da0);
    vr0 = __dtu_v_vmul_a_bf16(vr0, v_rec);
    __dtu_c_movsr2naccovr(naccovr);

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

    auto da1 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    auto vr1 = __dtu_l_movva2vr_cvt2bf16(da1);
    vr1 = __dtu_v_vmul_a_bf16(vr1, v_rec);
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

#pragma clang loop unroll(disable)
    for (int k = 0; k < K - 64; k += 64) {
      qacc0 = __dtu_m_vmm2_mode17_bf16(qacc0, vr0, smr0);
      __dtu_c_movsr2naccovr(0x1);

      da0 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      vr0 = __dtu_l_movva2vr_cvt2bf16(da0);
      vr0 = __dtu_v_vmul_a_bf16(vr0, v_rec);
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

      qacc0 = __dtu_m_vmm2_mode17_bf16(qacc0, vr1, smr1);
      da1 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      vr1 = __dtu_l_movva2vr_cvt2bf16(da1);
      vr1 = __dtu_v_vmul_a_bf16(vr1, v_rec);
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
    }

    qacc0 = __dtu_m_vmm2_mode17_bf16(qacc0, vr0, smr0);
    __dtu_c_movsr2naccovr(0x1);
    qacc0 = __dtu_m_vmm2_mode17_bf16(qacc0, vr1, smr1);

    vab_shift += 16;
    __dtu_c_movsr2vab_m_s1(vab_shift);
    __dtu_c_movsr2vab_m_d(vab_shift);

    lt_base = __dtu_v_taradd(lt_base, lt_off1);
    rt_base = __dtu_v_taradd(rt_base, rt_off1);
  }

  if (st_ind != NO_OUT) {
    vab_shift = 0;
    __dtu_c_movsr2vab_lv_s(0);

    out_addr = out_addr >> 7;
    tar_t ot_base = __dtu_c_movsr2targ(TAR32(out_addr, out_addr + 2));
    offset = TAR32(1, 1);
    tar_t ot_off0 = __dtu_c_movsr2tari(offset, ot_base);
    offset = TAR32(3, 3);
    tar_t ot_off1 = __dtu_c_movsr2tari(offset, ot_base);

#pragma clang loop unroll(disable)
    for (int n = 0; n < N; n += 128) {
      auto dacc2 = __dtu_extractqa2da(qacc0, 0);
      auto dacc3 = __dtu_extractqa2da(qacc0, 1);

      __dtu_v_tvstda_f32_dual(dacc2, ot_base, ot_off0);
      __dtu_v_tvstda_f32_dual(dacc3, ot_base, ot_off1);

      vab_shift += 16;
      __dtu_c_movsr2vab_lv_s(vab_shift);
    }
  }
}

__device__
void call_mid(char* dst_ptr) {
  auto qa_mid = __dtu_m_mid_m0_u32(0);
  auto da_mid0 = __dtu_extractqa2da(qa_mid, 0);
  auto da_mid1 = __dtu_extractqa2da(qa_mid, 1);
  __dtu_v_vstda_qaddr_s32_dual(da_mid0, dst_ptr);
  __dtu_v_vstda_qaddr_s32_dual(da_mid1, dst_ptr + 128);
}

// output: torch.Tensor,         # [num_seqs, num_heads, head_size][1, 8 ,128]
// query: torch.Tensor,          # [num_seqs, num_heads, head_size][1, 8 ,128]
// key_cache: torch.Tensor,
// # [num_blocks, num_kv_heads, head_size/x, block_size, x][31087,1,16,16,8]
// value_cache: torch.Tensor,
// # [num_blocks, num_kv_heads, head_size, block_size][31087,1,128,16]
// head_mapping: torch.Tensor,   # [num_heads][8]
// scale: float,                 # [0.08838834764831845]
// block_tables: torch.Tensor,   # [num_seqs, max_num_blocks_per_seq][1,36-160]
// context_lens: torch.Tensor,   # [num_seqs][1]
// block_size: int,              # [16]
// max_context_len: int,         # [512-2552]
// alibi_slopes: torch.Tensor    # [num_heads][nan]

template <typename T>
__device__ void paged_attention_v1_kernel(
    T *output,
    T *query, T *key_cache, T *value_cache,
    // int *head_mapping,
    float scale,
    int *block_tables, int *context_lens,
    int block_size, int max_context_len,
    float *alibi_slopes,
    int num_seqs, int num_heads, int head_size,
    int num_blocks, int num_kv_heads,
    int max_num_blocks_per_seq, int stride, int alibi_enable, float softscapping, char* buffer_sip) {
// krt_reset_clock();
  int thread_num = GetThreadNum();
  int thread_id = GetThreadIdx();
  int block_num = GetBlockNum();
  int block_id = GetBlockIdx();

  tops_dte_ctx_t ctxs_q;
  tops_dte_ctx_t ctxs_bt;
  tops_dte_ctx_t a_ctxs_k[MAX_WAIT_NUM];
  tops_dte_ctx_t ctxs_out;

  tops::event evs_q;
  tops::event evs_in;
  tops::event evs_out;
  tops::event evs_bt;
  tops::event evs_cl;
  tops::event evs_slope;
  tops::event a_evs_k[MAX_WAIT_NUM];

  // tops::dte_scope s_q(ctxs_q);
  // tops::dte_scope s_bt(ctxs_bt);
  // tops::dte_scope s_out(ctxs_out);

  // for (int i = 0; i < MAX_WAIT_NUM; i++) {
  //   a_ctxs_k[i].init();
  // }

  // int head_align32 = ALIGN_32(head_size);
  int head_align64 = ALIGN_64(head_size);
  int head_align128 = ALIGN_128(head_size);
  int32_t key_cache_shape[] =
      {1, 1, block_size, head_size};

  int32_t value_cache_shape[] =
      {1, 1, block_size, head_size};

  int32_t num_queries_per_kv = num_heads / num_kv_heads;

  int32_t query_shape[] = {1, 1, head_size};

  tops::mdspan l3_output(tops::Global, output, 1, 1, head_size);
  tops::mdspan l3_query(tops::Global, query, query_shape);
  tops::mdspan l3_key_cache(tops::Global, key_cache, key_cache_shape);
  tops::mdspan l3_value_cache(tops::Global, value_cache, value_cache_shape);
  tops::mdspan l3_context_lens(tops::Global, context_lens, num_seqs);
  tops::mdspan l3_alibi_slopes(tops::Global, alibi_slopes, num_heads);

  int BLOCKS_SIZE = (head_size == 256) ? 512 : 1024;
  int V_BLOCKS_SIZE = (head_size == 256) ? 512 : 1024;
  const int SOFTMAX_SIZE = 32 * 1024;
  int k_wait_num = BLOCKS_SIZE / block_size;
  k_wait_num =  (k_wait_num > MAX_WAIT_NUM) ? MAX_WAIT_NUM : k_wait_num;
  int v_wait_num = V_BLOCKS_SIZE / block_size;
  v_wait_num =  (v_wait_num > MAX_WAIT_NUM) ? MAX_WAIT_NUM : v_wait_num;
  int sip_offset = 0;

  int *p_block_tables = reinterpret_cast<int*>(buffer_sip + sip_offset);
  sip_offset += ALIGN_32(SLICE_LEN / block_size) * sizeof(int);

  int *p_context_lens = reinterpret_cast<int*>(buffer_sip + sip_offset);
  tops::mdspan s_context_lens(tops::Private, p_context_lens, num_seqs);
  sip_offset += ALIGN_32(num_seqs) * sizeof(int);

  float *p_alibi_slopes = reinterpret_cast<float*>(buffer_sip + sip_offset);
  tops::mdspan s_alibi_slopes(tops::Private, p_alibi_slopes, num_heads);
  sip_offset += ALIGN_32(num_heads) * sizeof(float);

  evs_cl = tops::memcpy_async(a_ctxs_k[2], s_context_lens, l3_context_lens);
  if (alibi_enable == 1) {
    evs_slope =
        tops::memcpy_async(a_ctxs_k[3], s_alibi_slopes, l3_alibi_slopes);
  }

  T *p_key_blocks = reinterpret_cast<T*>(buffer_sip + sip_offset);
  tops::mdspan s_key_block(
      tops::Private, p_key_blocks, 1, 1, block_size, head_align64);
  sip_offset += ALIGN_128(head_align64 * BLOCKS_SIZE * 2) * sizeof(T);
  T* p_key_buf[2] = {p_key_blocks, p_key_blocks + head_align64 * BLOCKS_SIZE};

  T *p_v_blocks = reinterpret_cast<T*>(p_key_blocks);
  tops::mdspan s_v_block(
      tops::Private, p_v_blocks, 1, 1, block_size, head_align64);
  // sip_offset += ALIGN_128(head_align64 * V_BLOCKS_SIZE * 2) * sizeof(T);
  T* p_val_buf[2] = {p_v_blocks, p_v_blocks + head_align64 * V_BLOCKS_SIZE};

  T *p_q = reinterpret_cast<T*>(buffer_sip + sip_offset);
  // head_size needs aligned to 64B, auto padding 0 when slice
  tops::mdspan sip_query(tops::Private, p_q, 1, 1, head_size);
  tops::mdspan sip_query64(tops::Private, p_q, 1, 1, head_align64);
  sip_offset += ALIGN_128(head_align64) * sizeof(T);
  sip_offset = ALIGN_256(sip_offset);

  float *p_attn = reinterpret_cast<float*>(buffer_sip + sip_offset);
  sip_offset += ALIGN_128(SOFTMAX_SIZE) * sizeof(float);
  sip_offset = ALIGN_512(sip_offset);

  T *p_out = reinterpret_cast<T*>(buffer_sip + sip_offset);
  tops::mdspan sip_out(tops::Private, p_out, 1, 1, head_size);
  sip_offset += ALIGN_128(head_align128) * sizeof(float);

  T *p_out_t = reinterpret_cast<T*>(buffer_sip + sip_offset);
  sip_offset += ALIGN_128(head_align128) * sizeof(float);
  sip_offset = ALIGN_256(sip_offset);

  char *p_mid = (buffer_sip + sip_offset);
  sip_offset += 256;

  float *p_max = reinterpret_cast<float*>(buffer_sip + sip_offset);
  sip_offset += 256;
  float *p_max_t = reinterpret_cast<float*>(buffer_sip + sip_offset);

  call_mid(p_mid);

  evs_cl.wait();
  if (alibi_enable == 1) { evs_slope.wait(); }

  if (IS_ALIGN_64(head_size)) {
    for (int i = 0; i < MAX_WAIT_NUM; i++) {
      a_ctxs_k[i].config_memcpy(s_key_block, l3_key_cache);
    }
  } else {
    for (int i = 0; i < MAX_WAIT_NUM; i++) {
      a_ctxs_k[i].config_slice(s_key_block, l3_key_cache, {0, 0, 0, 0});
    }
    evs_in = tops::memset_async(ctxs_out, sip_query64, (T)0.0);
  }

  ctxs_q.config_memcpy(sip_query, l3_query);

  int seq_start = 0;
  int seq_step = 1;
  if (num_seqs > 1) {
    thread_id = GetThreadIdxInBlock();
    thread_num = GetThreadNumEachBlock();
    seq_start = block_id;
    seq_step = block_num;
  }

  if (!IS_ALIGN_64(head_size)) { evs_in.wait(); }

  ctxs_out.config_memcpy(l3_output, sip_out);

  int sip_num_heads = (num_heads + thread_num - 1) / thread_num;
  int heads_start = thread_id * sip_num_heads;

  int heads_end = heads_start + sip_num_heads;
  heads_end = heads_end > num_heads ? num_heads : heads_end;

  for (int seq_idx = seq_start; seq_idx < num_seqs; seq_idx += seq_step) {
    auto p_query = query + seq_idx * stride;

    int ctx_len = p_context_lens[seq_idx];
    if ((ctx_len <= 0) || (heads_start >= num_heads)) continue;
    if (ctx_len > max_context_len)  ctx_len = max_context_len;
    int num_blocks_ = (ctx_len + block_size - 1) / block_size;
    int slices = (ctx_len + SLICE_LEN - 1) / SLICE_LEN;
    int slice_blocks = num_blocks_ / slices;
    slice_blocks = ALIGN_16(slice_blocks);
    int j_loop = (num_blocks_ + slice_blocks - 1) / slice_blocks;

    int bt_shape = (num_blocks_ <= slice_blocks) ? num_blocks_ : slice_blocks;
    tops::mdspan s_block_tables(
        tops::Private, p_block_tables, bt_shape);
    tops::mdspan l3_block_tables(tops::Global,
        block_tables + seq_idx * max_num_blocks_per_seq, bt_shape);
    evs_bt = tops::memcpy_async(ctxs_bt, s_block_tables, l3_block_tables);

    ctxs_q.set_src_addr(p_query + heads_start * head_size);
    evs_q = ctxs_q.trigger();

    int wait_times = 0;
    int k_ds_offset = 0;
    int kv_head_idx = heads_start / num_queries_per_kv;
    auto p_key_head = key_cache + kv_head_idx * head_size * block_size;
    evs_bt.wait();

    for (int i = 0; (i < num_blocks_) && (i < k_wait_num); i++) {
      int kv_block_idx = p_block_tables[i];
      auto p_key_src =
          p_key_head + kv_block_idx * num_kv_heads * head_size * block_size;
      auto p_key_dst = p_key_buf[0] + k_ds_offset;

      a_ctxs_k[wait_times].set_src_addr(p_key_src);
      a_ctxs_k[wait_times].set_dst_addr(p_key_dst);
      a_evs_k[wait_times] = a_ctxs_k[wait_times].trigger();

      wait_times++;
      k_ds_offset += head_align64 * block_size;
    }

    int pp_ind = 0;
    for (int head_idx = heads_start; head_idx < heads_end; head_idx++) {
      auto p_value_head = value_cache + kv_head_idx * head_size * block_size;
      float slope = 0.0f;
      if (alibi_enable == 1) { slope = p_alibi_slopes[head_idx]; }

      int end_blocks = slice_blocks;
      int start_blocks = 0;
      int alibi_idx = ctx_len;
      int left_len = ctx_len;

      evs_q.wait();

      for (int j = 0; j < j_loop; j++) {
        int attn_offset = 0;
        end_blocks = (end_blocks > num_blocks_) ? num_blocks_ : end_blocks;

        for (int i = k_wait_num; i < (end_blocks - start_blocks); i++) {
          int kv_block_idx = p_block_tables[i];
          auto p_key_src =
              p_key_head + kv_block_idx * num_kv_heads * head_size * block_size;

          if (wait_times >= k_wait_num) {
            for (int i = 0; i < wait_times; i++) {
              a_evs_k[i].wait();
            }

            wait_times = 0;
            k_ds_offset = 0;
          }

          auto p_key_dst = p_key_buf[1 - pp_ind] + k_ds_offset;
          a_ctxs_k[wait_times].set_src_addr(p_key_src);
          a_ctxs_k[wait_times].set_dst_addr(p_key_dst);
          a_evs_k[wait_times] = a_ctxs_k[wait_times].trigger();

          wait_times++;

          if ((wait_times >= k_wait_num) ||
              ((i + 1) >= (end_blocks - start_blocks))) {
            call_k_dot_nk(p_attn, attn_offset, p_q, p_key_buf[pp_ind],
                head_align64, block_size * k_wait_num, scale, p_mid,
                slope, alibi_idx, alibi_enable);

            pp_ind = 1 - pp_ind;
            attn_offset += block_size * k_wait_num;
            alibi_idx -= block_size * k_wait_num;
          }

          k_ds_offset += head_align64 * block_size;
        }

        int N = block_size * wait_times;
        for (int i = 0; i < wait_times; i++) {
          a_evs_k[i].wait();
        }

        wait_times = 0;
        int v_ds_offset = 0;
        int softmax_offset = 0;
        for (int i = 0;
            (i < (end_blocks - start_blocks)) && (i < v_wait_num); i++) {
          int kv_block_idx = p_block_tables[i];
          auto p_value_src = p_value_head +
              kv_block_idx * num_kv_heads * head_size * block_size;
          auto p_value_dst = p_val_buf[1-pp_ind] + v_ds_offset;

          a_ctxs_k[wait_times].set_src_addr(p_value_src);
          a_ctxs_k[wait_times].set_dst_addr(p_value_dst);
          a_evs_k[wait_times] = a_ctxs_k[wait_times].trigger();

          wait_times++;
          v_ds_offset += head_align64 * block_size;
        }

        // q = q * scale
        // attn = q @ key
        call_k_dot_nk(p_attn, attn_offset, p_q, p_key_buf[pp_ind],
            head_align64, ALIGN_64(N), scale, p_mid,
            slope, alibi_idx, alibi_enable);
        alibi_idx -= N;

        auto max_ptr = p_max;
        if (j > 0) { max_ptr = p_max_t; }
        call_softmax_qa(p_attn, p_attn, max_ptr, attn_offset + N, left_len);

        if ((j == 0) && (head_idx != heads_start)) {
          evs_out.wait();
        }

        for (int i = v_wait_num; i < (end_blocks - start_blocks); i++) {
          int kv_block_idx = p_block_tables[i];
          auto p_value_src = p_value_head +
              kv_block_idx * num_kv_heads * head_size * block_size;

          if (wait_times >= v_wait_num) {
            for (int i = 0; i < wait_times; i++) {
              a_evs_k[i].wait();
            }

            wait_times = 0;
            v_ds_offset = 0;
          }
          auto p_value_dst = p_val_buf[pp_ind] + v_ds_offset;

          a_ctxs_k[wait_times].set_src_addr(p_value_src);
          a_ctxs_k[wait_times].set_dst_addr(p_value_dst);
          a_evs_k[wait_times] = a_ctxs_k[wait_times].trigger();

          wait_times++;

          // o = attn @ value.transpose(-2, -1)
          if ((wait_times >= v_wait_num) ||
              ((i + 1) >= (end_blocks - start_blocks))) {
            call_k_dot_kn(p_out, p_attn + softmax_offset, p_val_buf[1-pp_ind],
                max_ptr + SUM_OFFSET, v_wait_num * block_size, head_align64,
                softmax_offset, left_len, NO_OUT);

            pp_ind = 1 - pp_ind;
            softmax_offset += v_wait_num * block_size;
          }

          v_ds_offset += head_align64 * block_size;
        }

        int K = block_size * wait_times;
        for (int i = 0; i < wait_times; i++) {
          a_evs_k[i].wait();
        }

        if (((head_idx + 1) != heads_end) || ((j + 1) != j_loop)) {
          wait_times = 0;
          k_ds_offset = 0;
          start_blocks += slice_blocks;
          if ((j + 1) == j_loop) { start_blocks = 0; }

          if (j_loop != 1) {
            int bt_shape = num_blocks_ - start_blocks;
            bt_shape = bt_shape > slice_blocks ? slice_blocks : bt_shape;
            tops::mdspan s_block_tables(
                tops::Private, p_block_tables, bt_shape);
            tops::mdspan l3_block_tables(tops::Global,
                block_tables + seq_idx * max_num_blocks_per_seq + start_blocks,
                bt_shape);
            evs_bt =
                tops::memcpy_async(ctxs_bt, s_block_tables, l3_block_tables);
          }

          if ((j + 1) == j_loop) {
            ctxs_q.set_src_addr(p_query + (head_idx + 1) * head_size);
            evs_q = ctxs_q.trigger();

            kv_head_idx = (head_idx + 1) / num_queries_per_kv;
          }

          p_key_head = key_cache + kv_head_idx * head_size * block_size;
          if (j_loop != 1) { evs_bt.wait(); }

          for (int i = 0;
               (i < (num_blocks_ - start_blocks)) && (i < k_wait_num); i++) {
            int kv_block_idx = p_block_tables[i];
            auto p_key_src = p_key_head +
                kv_block_idx * num_kv_heads * head_size * block_size;
            auto p_key_dst = p_key_buf[pp_ind] + k_ds_offset;

            a_ctxs_k[wait_times].set_src_addr(p_key_src);
            a_ctxs_k[wait_times].set_dst_addr(p_key_dst);
            a_evs_k[wait_times] = a_ctxs_k[wait_times].trigger();

            wait_times++;
            k_ds_offset += head_align64 * block_size;
          }
        }

        auto output_ptr = (j > 0) ? p_out_t : p_out;
        call_k_dot_kn(output_ptr, p_attn + softmax_offset, p_val_buf[1-pp_ind],
            max_ptr + SUM_OFFSET, ALIGN_64(K), head_align64,
            softmax_offset, left_len, F32_OUT);

        if (j > 0) {
          call_update_softmax(p_max, p_max_t, reinterpret_cast<float*>(p_out),
                              reinterpret_cast<float*>(p_out_t), head_align64);
        }

        left_len -= attn_offset + N;
        end_blocks += slice_blocks;
      } //  for j

      call_convert<T>(p_out, reinterpret_cast<float*>(p_out),
                      p_max + SUM_OFFSET, head_align64);

      // output[seq_idx, head_idx, :] = o
      auto p_output =
          output + seq_idx * num_heads * head_size + head_idx * head_size;
      ctxs_out.set_dst_addr(p_output);
      evs_out = ctxs_out.trigger();
    } //  for heads_end

    evs_out.wait();
  } // for seq_num

  // for (int i = 0; i < MAX_WAIT_NUM; i++) {
  //   a_ctxs_k[i].destroy();
  // }

// krt_close_clock();
// int duration = krt_clock();
// if (thread_id == 0)
//   printf("%d cycles\n", duration);
}

#define ATTENTION_KERNEL(T, TYPENAME) \
extern "C" __global__ void paged_attention_v1_##TYPENAME(\
  T *output,            \
  T *query,          \
  T *key_cache,       \
  T *value_cache,     \
  int32_t num_kv_heads,              \
  float scale,\
  int32_t *block_tables,    \
  int32_t *context_lens,    \
  int32_t block_size,\
  int32_t max_context_len,\
  int32_t num_seqs,\
  int32_t num_heads,\
  int32_t head_size,\
  int32_t max_num_blocks_per_seq,\
  int32_t q_stride,\
  int32_t kv_block_stride,\
  int32_t kv_head_stride,\
  int32_t num_blocks,\
  float softcapping \
  ) {\
    __local__ __valigned__ char buffer_sip[VDMEM_VALID_SIZE];\
    int alibi_enable = 0;\
    float* alibi_slopes = nullptr;\
    int head_radio = num_heads / num_kv_heads;\
    int head_rmd = num_heads % num_kv_heads;\
    bool is_gqa = ((head_radio <= 16) && (head_radio >= 8)) || \
                  (head_radio == 4);\
    is_gqa = is_gqa && (head_size == 128) && (head_rmd == 0);\
    is_gqa = is_gqa && (max_context_len <= 32768);\
    if (is_gqa) {\
      paged_attention_gqa_kernel<T>(output, query,\
          key_cache,\
          value_cache,\
          scale,\
          block_tables, context_lens,\
          block_size, max_context_len,\
          alibi_slopes,\
          num_seqs, num_heads, head_size,\
          num_blocks, num_kv_heads,\
          max_num_blocks_per_seq, q_stride, alibi_enable, softcapping, buffer_sip);\
    } else {\
      paged_attention_v1_kernel<T>(\
          output,\
          query,\
          key_cache,\
          value_cache,\
          scale,\
          block_tables, context_lens,\
          block_size, max_context_len,\
          alibi_slopes,\
          num_seqs, num_heads, head_size,\
          num_blocks, num_kv_heads,\
          max_num_blocks_per_seq, q_stride, alibi_enable, softcapping, buffer_sip);\
    }\
}\

ATTENTION_KERNEL(tops::half, f16)
ATTENTION_KERNEL(tops::bfloat, bf16)
