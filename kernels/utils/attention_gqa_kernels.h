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
#include "utils.h"
// #include "krt/scalar_type.h"

using namespace tops;

#define TAR32(low, high) (((low) & 0xffff) | (((high) & 0xffff) << 16))

#define ALIGN_4096(x)  ((((x) + 4095) >> 12) << 12)
#define ALIGN_2048(x)  ((((x) + 2047) >> 11) << 11)
#define ALIGN_256(x)  ((((x) + 255) >> 8) << 8)
#define ALIGN_128(x)  ((((x) + 127) >> 7) << 7)
#define ALIGN_64(x)  ((((x) + 63) >> 6) << 6)
#define ALIGN_32(x)  ((((x) + 31) >> 5) << 5)
#define ALIGN_16(x)  ((((x) + 15) >> 4) << 4)
#define MAX_WAIT_NUM  16
#define SMALL_SIZE    512
#define L1_MEM_SIZE   1310720   //  1.25M

#define ST_QA(qa)     \
      da0 = __dtu_extractqa2da_f32(qa, 0); \
      da1 = __dtu_extractqa2da_f32(qa, 1); \
      __dtu_v_tvstda_f32_dual(da0, out_tar, out_off0);  \
      __dtu_v_tvstda_f32_dual(da1, out_tar, out_off1);

__device__
void call_update_softmax(float* p_max, float* in, int dim1, int dim0) {
  int in_addr = reinterpret_cast<long>(in);
  tar_t in_tar = __dtu_c_movsr2targ(TAR32(in_addr >> 8, (in_addr >> 8) + 1));
  tar_t in_off0 = __dtu_c_movsr2tari(TAR32(2, 2), in_tar);

  tar_t ot_base = __dtu_c_movsr2targ(TAR32(in_addr >> 7, (in_addr >> 7) + 2));
  tar_t ot_off0 = __dtu_c_movsr2tari(TAR32(1, 1), ot_base);
  tar_t ot_off1 = __dtu_c_movsr2tari(TAR32(3, 3), ot_base);

  int max_addr = reinterpret_cast<long>(p_max);
  tar_t max_tar = __dtu_c_movsr2targ(TAR32(max_addr >> 6, max_addr >> 6));
  tar_t max_off0 = __dtu_c_movsr2tari(TAR32(1, 1), max_tar);

#pragma clang loop unroll(disable)
  for (int i = 0; i < dim1 / 16; i++) {
    auto vr0 = __dtu_s_tvld_itar(max_tar, max_off0);

    auto qa0 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa1 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa2 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa3 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa4 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa5 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa6 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa7 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa8 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa9 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa10 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa11 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa12 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa13 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa14 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa15 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);

    qa0 = __dtu_m_vdm_row_f32_qa(qa0, vr0, 0);
    qa1 = __dtu_m_vdm_row_f32_qa(qa1, vr0, 1);
    qa2 = __dtu_m_vdm_row_f32_qa(qa2, vr0, 2);
    qa3 = __dtu_m_vdm_row_f32_qa(qa3, vr0, 3);
    qa4 = __dtu_m_vdm_row_f32_qa(qa4, vr0, 4);
    qa5 = __dtu_m_vdm_row_f32_qa(qa5, vr0, 5);
    qa6 = __dtu_m_vdm_row_f32_qa(qa6, vr0, 6);
    qa7 = __dtu_m_vdm_row_f32_qa(qa7, vr0, 7);
    qa8 = __dtu_m_vdm_row_f32_qa(qa8, vr0, 8);
    qa9 = __dtu_m_vdm_row_f32_qa(qa9, vr0, 9);
    qa10 = __dtu_m_vdm_row_f32_qa(qa10, vr0, 10);
    qa11 = __dtu_m_vdm_row_f32_qa(qa11, vr0, 11);
    qa12 = __dtu_m_vdm_row_f32_qa(qa12, vr0, 12);
    qa13 = __dtu_m_vdm_row_f32_qa(qa13, vr0, 13);
    qa14 = __dtu_m_vdm_row_f32_qa(qa14, vr0, 14);
    qa15 = __dtu_m_vdm_row_f32_qa(qa15, vr0, 15);

    auto da0 = __dtu_extractqa2da(qa0, 0);
    auto da1 = __dtu_extractqa2da(qa0, 1);
    auto da2 = __dtu_extractqa2da(qa1, 0);
    auto da3 = __dtu_extractqa2da(qa1, 1);
    auto da4 = __dtu_extractqa2da(qa2, 0);
    auto da5 = __dtu_extractqa2da(qa2, 1);
    auto da6 = __dtu_extractqa2da(qa3, 0);
    auto da7 = __dtu_extractqa2da(qa3, 1);
    auto da8 = __dtu_extractqa2da(qa4, 0);
    auto da9 = __dtu_extractqa2da(qa4, 1);
    auto da10 = __dtu_extractqa2da(qa5, 0);
    auto da11 = __dtu_extractqa2da(qa5, 1);
    auto da12 = __dtu_extractqa2da(qa6, 0);
    auto da13 = __dtu_extractqa2da(qa6, 1);
    auto da14 = __dtu_extractqa2da(qa7, 0);
    auto da15 = __dtu_extractqa2da(qa7, 1);
    auto da16 = __dtu_extractqa2da(qa8, 0);
    auto da17 = __dtu_extractqa2da(qa8, 1);
    auto da18 = __dtu_extractqa2da(qa9, 0);
    auto da19 = __dtu_extractqa2da(qa9, 1);
    auto da20 = __dtu_extractqa2da(qa10, 0);
    auto da21 = __dtu_extractqa2da(qa10, 1);
    auto da22 = __dtu_extractqa2da(qa11, 0);
    auto da23 = __dtu_extractqa2da(qa11, 1);
    auto da24 = __dtu_extractqa2da(qa12, 0);
    auto da25 = __dtu_extractqa2da(qa12, 1);
    auto da26 = __dtu_extractqa2da(qa13, 0);
    auto da27 = __dtu_extractqa2da(qa13, 1);
    auto da28 = __dtu_extractqa2da(qa14, 0);
    auto da29 = __dtu_extractqa2da(qa14, 1);
    auto da30 = __dtu_extractqa2da(qa15, 0);
    auto da31 = __dtu_extractqa2da(qa15, 1);

    __dtu_v_tvstda_f32_dual(da0, ot_base, ot_off0);
    __dtu_v_tvstda_f32_dual(da1, ot_base, ot_off1);
    __dtu_v_tvstda_f32_dual(da2, ot_base, ot_off0);
    __dtu_v_tvstda_f32_dual(da3, ot_base, ot_off1);
    __dtu_v_tvstda_f32_dual(da4, ot_base, ot_off0);
    __dtu_v_tvstda_f32_dual(da5, ot_base, ot_off1);
    __dtu_v_tvstda_f32_dual(da6, ot_base, ot_off0);
    __dtu_v_tvstda_f32_dual(da7, ot_base, ot_off1);

    __dtu_v_tvstda_f32_dual(da8, ot_base, ot_off0);
    __dtu_v_tvstda_f32_dual(da9, ot_base, ot_off1);
    __dtu_v_tvstda_f32_dual(da10, ot_base, ot_off0);
    __dtu_v_tvstda_f32_dual(da11, ot_base, ot_off1);
    __dtu_v_tvstda_f32_dual(da12, ot_base, ot_off0);
    __dtu_v_tvstda_f32_dual(da13, ot_base, ot_off1);
    __dtu_v_tvstda_f32_dual(da14, ot_base, ot_off0);
    __dtu_v_tvstda_f32_dual(da15, ot_base, ot_off1);

    __dtu_v_tvstda_f32_dual(da16, ot_base, ot_off0);
    __dtu_v_tvstda_f32_dual(da17, ot_base, ot_off1);
    __dtu_v_tvstda_f32_dual(da18, ot_base, ot_off0);
    __dtu_v_tvstda_f32_dual(da19, ot_base, ot_off1);
    __dtu_v_tvstda_f32_dual(da20, ot_base, ot_off0);
    __dtu_v_tvstda_f32_dual(da21, ot_base, ot_off1);
    __dtu_v_tvstda_f32_dual(da22, ot_base, ot_off0);
    __dtu_v_tvstda_f32_dual(da23, ot_base, ot_off1);

    __dtu_v_tvstda_f32_dual(da24, ot_base, ot_off0);
    __dtu_v_tvstda_f32_dual(da25, ot_base, ot_off1);
    __dtu_v_tvstda_f32_dual(da26, ot_base, ot_off0);
    __dtu_v_tvstda_f32_dual(da27, ot_base, ot_off1);
    __dtu_v_tvstda_f32_dual(da28, ot_base, ot_off0);
    __dtu_v_tvstda_f32_dual(da29, ot_base, ot_off1);
    __dtu_v_tvstda_f32_dual(da30, ot_base, ot_off0);
    __dtu_v_tvstda_f32_dual(da31, ot_base, ot_off1);
  }
}

__device__
void call_get_max(float* p_out, float* p_max, int dim1, int dim0) {
  int out_addr = reinterpret_cast<long>(p_out);
  int in_addr = reinterpret_cast<long>(p_max);

  in_addr = in_addr >> 6;
  asm volatile (
    " l.movsr2targ.ext.t0 ta_g0, %0         \n"
    " l.movsr2targ.ext.t1 ta_g0, %1         \n"
    " l.movsr2tari.ext.t0 [ta_g0, 1], %2    \n"
    " l.movsr2tari.ext.t1 [ta_g0, 1], %2    \n"
    " l.movsr2tari.ext.t0 [ta_g0, 2], %3    \n"
    " l.movsr2tari.ext.t1 [ta_g0, 2], %3    \n"
    :
    : "r" (in_addr),
      "r" (in_addr + 1),
      "r" (2),
      "r" (-dim1 * 4)
    :
    );

  out_addr = out_addr >> 6;
  asm volatile (
    " l.movsr2targ.ext.t0 ta_g1, %0         \n"
    " l.movsr2targ.ext.t1 ta_g1, %1         \n"
    " l.movsr2tari.ext.t0 [ta_g1, 1], %2    \n"
    " l.movsr2tari.ext.t1 [ta_g1, 1], %2    \n"
    " l.movsr2tari.ext.t0 [ta_g1, 2], %3    \n"
    " l.movsr2tari.ext.t1 [ta_g1, 2], %3    \n"
    :
    : "r" (out_addr),
      "r" (out_addr + 1),
      "r" (4),
      "r" (-dim1 * 4)
    );

  asm volatile (
    " l.movsr2targ.ext.t0 ta_g2, %0         \n"
    " l.movsr2targ.ext.t1 ta_g2, %1         \n"
    " l.movsr2tari.ext.t0 [ta_g2, 1], %2    \n"
    " l.movsr2tari.ext.t1 [ta_g2, 1], %2    \n"
    :
    : "r" (out_addr + 2),
      "r" (out_addr + 3),
      "r" (4)
    );

  asm volatile (
    " l.ldi_hi r10, 0x3fb8            \n"
    " l.ldi_lo r10, 0xaa3b            \n"
    " s.movr2vr.dup vr0, r10          \n"

    " s.tvld_itar vr1, [ta_g0, 1]   \n"
    " s.tvld_itar vr2, [ta_g0, 1]   \n"
    " s.movvr vr3, vr1              \n"
    " l.addia.s r11, %0, 0    \n"

    "13:  \n"
    " v.vmaxa.f32 vr3, vr3, vr1     \n"
    " s.tvld_itar vr1, [ta_g0, 1]   \n"
    " s.tvld_itar vr2, [ta_g0, 1]   \n"
    " l.addia.s r11, r11, -1        \n"
    " l.bne r0, r11, 13b            \n"

    :
    : "r" (dim1)
    : "r10", "r11"
    );

  asm volatile (
    " l.movsr2targ.ext.t0 ta_g0, %0         \n"
    " l.movsr2targ.ext.t1 ta_g0, %1         \n"
    " l.addia.s r11, %2, 0    \n"

    " l.vclr.va vacc8      \n"
    " s.tvld_itar vr1, [ta_g0, 1]   \n"
    " s.tvld_itar vr2, [ta_g0, 1]   \n"

    "15:  \n"
    " v.vsuba.f32 vr1, vr1, vr3     \n"
    " v.vmula.f32 vr1, vr1, vr0     \n"
    " l.movvr2va vacc0, vr1         \n"
    " s.tvld_itar vr1, [ta_g0, 1]   \n"
    " l.movvr2va vacc9, vr2         \n"
    " s.tvld_itar vr2, [ta_g0, 1]   \n"
    " m.msf.mode0 vacc0, vacc0      \n"
    // " m.mop.mdm.f32.va vacc10, vacc9, vacc0     \n"
    " v.tvstda.w vacc0, [ta_g1, 1]  \n"
    " m.mop.mac.f32.va vacc8, vacc9, vacc0      \n"

    " l.addia.s r11, r11, -1        \n"
    " l.bne r0, r11, 15b            \n"

    :
    : "r" (in_addr),
      "r" (in_addr + 1),
      "r" (dim1)
    : "r11"
    );

  asm volatile (
    " v.taradd ta_g1, [ta_g1, 2]    \n"
    " l.addia.s r11, %0, 0          \n"
    " l.ldi16.s r10, 1              \n"

    " m.msf.mode6 vacc7, vacc8      \n"
    " l.tvlda vacc9, [ta_g1, 1]     \n"
    " c.movsr2spr VPR, r10          \n"
    " l.ldi16.s r10, 0              \n"

    "16:  \n"
    " if (vpr) v.tvstda.w vacc10, [ta_g2, 1]   \n"
    " m.mop.mdm.f32.va vacc10, vacc9, vacc7    \n"
    " l.tvlda vacc9, [ta_g1, 1]               \n"

    // " v.tvstda.w vacc7, [ta_g2, 1]   \n"

    " l.addia.s r11, r11, -1          \n"
    " c.movsr2spr VPR, r10            \n"
    " l.bne r0, r11, 16b              \n"

    " v.tvstda.w vacc10, [ta_g2, 1]   \n"

    :
    : "r" (dim1)
    : "r10", "r11"
    );
}

__device__
void call_alibi(float* dst_ptr, float* in_ptr,
    float *slopes, int dim1, int dim0, int ctx_len) {
  int out_addr = reinterpret_cast<long>(dst_ptr);
  int in_addr = reinterpret_cast<long>(in_ptr);

  int stride = dim0 / 64;
  int tar_addr = in_addr >> 8;
  tar_t in_tar = __dtu_s_movsr2targ(TAR32(tar_addr, tar_addr + 1));
  tar_t in_off0 = __dtu_s_movsr2tari(TAR32(stride, stride), in_tar);
  int off = -15 * stride + 2;
  tar_t in_off1 = __dtu_s_movsr2tari(TAR32(off, off), in_tar);
  off = 15 * stride;
  tar_t in_off2 = __dtu_s_movsr2tari(TAR32(off, off), in_tar);

  tar_addr = out_addr >> 7;
  tar_t out_tar = __dtu_s_movsr2targ(TAR32(tar_addr, tar_addr + 2));
  tar_t out_off0 = __dtu_s_movsr2tari(TAR32(1, 1), out_tar);
  off = dim0 / 32;
  tar_t out_off1 = __dtu_s_movsr2tari(TAR32(off - 1, off - 1), out_tar);
  off = (dim0 / 32) * 15;
  tar_t out_off2 = __dtu_s_movsr2tari(TAR32(-off + 3, -off + 3), out_tar);
  tar_t out_off3 = __dtu_s_movsr2tari(TAR32(3, 3), out_tar);

  auto qa_mid = __dtu_m_mid_m0_u32(0);
  auto v_ctx_len = __dtu_l_movr2qa_s32(-ctx_len);
  // auto v_mid = __dtu_m_mop_add_s32_qa(v_ctx_len, qa_mid);
  auto v_128 = __dtu_l_movr2qa_s32(128);
  using vtype = typename scalar_to_vector<float, TOPS_VECTOR_LENGTH>::type;

  for (int i = 0; i < dim1; i += 16) {
    auto qa0 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa1 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa2 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa3 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa4 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa5 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa6 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa7 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa8 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa9 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa10 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa11 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa12 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa13 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa14 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
    auto qa15 = __dtu_l_tvldqa_f32_qa(in_tar, in_off1);

    auto v_mid = __dtu_m_mop_add_s32_qa(v_ctx_len, qa_mid);

    vtype v_slope0 = vbroadcast<vtype>(slopes[0]);
    vtype v_slope1 = vbroadcast<vtype>(slopes[1]);
    vtype v_slope2 = vbroadcast<vtype>(slopes[2]);
    vtype v_slope3 = vbroadcast<vtype>(slopes[3]);
    vtype v_slope4 = vbroadcast<vtype>(slopes[4]);
    vtype v_slope5 = vbroadcast<vtype>(slopes[5]);
    vtype v_slope6 = vbroadcast<vtype>(slopes[6]);
    vtype v_slope7 = vbroadcast<vtype>(slopes[7]);

    vtype v_slope8 = vbroadcast<vtype>(slopes[8]);
    vtype v_slope9 = vbroadcast<vtype>(slopes[9]);
    vtype v_slope10 = vbroadcast<vtype>(slopes[10]);
    vtype v_slope11 = vbroadcast<vtype>(slopes[11]);
    vtype v_slope12 = vbroadcast<vtype>(slopes[12]);
    vtype v_slope13 = vbroadcast<vtype>(slopes[13]);
    vtype v_slope14 = vbroadcast<vtype>(slopes[14]);
    vtype v_slope15 = vbroadcast<vtype>(slopes[15]);

    slopes += 16;

    auto v_mid0 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope0);
    auto v_mid1 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope1);
    auto v_mid2 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope2);
    auto v_mid3 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope3);
    auto v_mid4 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope4);
    auto v_mid5 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope5);
    auto v_mid6 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope6);
    auto v_mid7 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope7);

    auto v_mid8 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope8);
    auto v_mid9 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope9);
    auto v_mid10 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope10);
    auto v_mid11 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope11);
    auto v_mid12 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope12);
    auto v_mid13 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope13);
    auto v_mid14 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope14);
    auto v_mid15 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope15);

    for (int j = 1; j < (dim0 / 128); j++) {
      qa0 = __dtu_m_mop_add_f32_qa(qa0, v_mid0);
      qa1 = __dtu_m_mop_add_f32_qa(qa1, v_mid1);
      qa2 = __dtu_m_mop_add_f32_qa(qa2, v_mid2);
      qa3 = __dtu_m_mop_add_f32_qa(qa3, v_mid3);
      qa4 = __dtu_m_mop_add_f32_qa(qa4, v_mid4);
      qa5 = __dtu_m_mop_add_f32_qa(qa5, v_mid5);
      qa6 = __dtu_m_mop_add_f32_qa(qa6, v_mid6);
      qa7 = __dtu_m_mop_add_f32_qa(qa7, v_mid7);

      qa8 = __dtu_m_mop_add_f32_qa(qa8, v_mid8);
      qa9 = __dtu_m_mop_add_f32_qa(qa9, v_mid9);
      qa10 = __dtu_m_mop_add_f32_qa(qa10, v_mid10);
      qa11 = __dtu_m_mop_add_f32_qa(qa11, v_mid11);
      qa12 = __dtu_m_mop_add_f32_qa(qa12, v_mid12);
      qa13 = __dtu_m_mop_add_f32_qa(qa13, v_mid13);
      qa14 = __dtu_m_mop_add_f32_qa(qa14, v_mid14);
      qa15 = __dtu_m_mop_add_f32_qa(qa15, v_mid15);

      auto da0 = __dtu_extractqa2da_f32(qa0, 0);
      auto da1 = __dtu_extractqa2da_f32(qa0, 1);
      __dtu_v_tvstda_f32_dual(da0, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da1, out_tar, out_off1);

      ST_QA(qa1)
      ST_QA(qa2)
      ST_QA(qa3)
      ST_QA(qa4)
      ST_QA(qa5)
      ST_QA(qa6)
      ST_QA(qa7)

      ST_QA(qa8)
      ST_QA(qa9)
      ST_QA(qa10)
      ST_QA(qa11)
      ST_QA(qa12)
      ST_QA(qa13)
      ST_QA(qa14)

      da0 = __dtu_extractqa2da_f32(qa15, 0);
      da1 = __dtu_extractqa2da_f32(qa15, 1);
      __dtu_v_tvstda_f32_dual(da0, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da1, out_tar, out_off2);

      qa0 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      qa1 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      qa2 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      qa3 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      qa4 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      qa5 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      qa6 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      qa7 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      qa8 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      qa9 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      qa10 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      qa11 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      qa12 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      qa13 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      qa14 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      qa15 = __dtu_l_tvldqa_f32_qa(in_tar, in_off1);

      v_mid = __dtu_m_mop_add_s32_qa(v_128, v_mid);

      v_mid0 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope0);
      v_mid1 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope1);
      v_mid2 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope2);
      v_mid3 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope3);
      v_mid4 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope4);
      v_mid5 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope5);
      v_mid6 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope6);
      v_mid7 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope7);

      v_mid8 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope8);
      v_mid9 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope9);
      v_mid10 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope10);
      v_mid11 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope11);
      v_mid12 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope12);
      v_mid13 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope13);
      v_mid14 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope14);
      v_mid15 = __dtu_m_mop_mul_f32mix_s32_qa(v_mid, v_slope15);
    }

    qa0 = __dtu_m_mop_add_f32_qa(qa0, v_mid0);
    qa1 = __dtu_m_mop_add_f32_qa(qa1, v_mid1);
    qa2 = __dtu_m_mop_add_f32_qa(qa2, v_mid2);
    qa3 = __dtu_m_mop_add_f32_qa(qa3, v_mid3);
    qa4 = __dtu_m_mop_add_f32_qa(qa4, v_mid4);
    qa5 = __dtu_m_mop_add_f32_qa(qa5, v_mid5);
    qa6 = __dtu_m_mop_add_f32_qa(qa6, v_mid6);
    qa7 = __dtu_m_mop_add_f32_qa(qa7, v_mid7);

    qa8 = __dtu_m_mop_add_f32_qa(qa8, v_mid8);
    qa9 = __dtu_m_mop_add_f32_qa(qa9, v_mid9);
    qa10 = __dtu_m_mop_add_f32_qa(qa10, v_mid10);
    qa11 = __dtu_m_mop_add_f32_qa(qa11, v_mid11);
    qa12 = __dtu_m_mop_add_f32_qa(qa12, v_mid12);
    qa13 = __dtu_m_mop_add_f32_qa(qa13, v_mid13);
    qa14 = __dtu_m_mop_add_f32_qa(qa14, v_mid14);
    qa15 = __dtu_m_mop_add_f32_qa(qa15, v_mid15);

    auto da0 = __dtu_extractqa2da_f32(qa0, 0);
    auto da1 = __dtu_extractqa2da_f32(qa0, 1);
    __dtu_v_tvstda_f32_dual(da0, out_tar, out_off0);
    __dtu_v_tvstda_f32_dual(da1, out_tar, out_off1);

    ST_QA(qa1)
    ST_QA(qa2)
    ST_QA(qa3)
    ST_QA(qa4)
    ST_QA(qa5)
    ST_QA(qa6)
    ST_QA(qa7)

    ST_QA(qa8)
    ST_QA(qa9)
    ST_QA(qa10)
    ST_QA(qa11)
    ST_QA(qa12)
    ST_QA(qa13)
    ST_QA(qa14)

    da0 = __dtu_extractqa2da_f32(qa15, 0);
    da1 = __dtu_extractqa2da_f32(qa15, 1);
    __dtu_v_tvstda_f32_dual(da0, out_tar, out_off0);
    __dtu_v_tvstda_f32_dual(da1, out_tar, out_off3);
  }
}

__device__
void call_softmax_vr(float* dst_ptr, float* in_ptr,
                     int dim1, int dim0, int ctx_len) {
  int out_addr = reinterpret_cast<long>(dst_ptr);
  int in_addr = reinterpret_cast<long>(in_ptr);

  if (ctx_len < dim0) {
    auto vdefault = __dtu_l_movr2qa_u32(0xff800000);
    auto mask_in = __dtu_m_mid_m0_u32(0);
    auto mask_val = __dtu_l_movr2qa_u32(ctx_len & 0x7f);
    auto vmask = __dtu_m_mop_mslt_u32_qa(mask_in, mask_val);
    int tar_addr = (in_addr + dim0 * 4 - 512) >> 8;
    int stride = dim0 / 64;
    tar_t in_tar = __dtu_s_movsr2targ(TAR32(tar_addr, tar_addr + 1));
    tar_t in_off0 = __dtu_s_movsr2tari(TAR32(stride, stride), in_tar);

    tar_addr = (in_addr + dim0 * 4 - 512) >> 7;
    stride = dim0 / 32;
    tar_t out_tar = __dtu_s_movsr2targ(TAR32(tar_addr, tar_addr + 2));
    tar_t out_off0 = __dtu_s_movsr2tari(TAR32(1, 1), out_tar);
    tar_t out_off1 = __dtu_s_movsr2tari(TAR32(stride - 1, stride - 1), out_tar);

#pragma clang loop unroll(disable)
    for (int i = 0; i < dim1; i += 16) {
      auto qa0 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa1 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa2 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa3 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa4 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa5 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa6 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa7 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa8 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa9 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa10 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa11 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa12 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa13 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa14 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa15 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);

      auto out0 = __dtu_m_mop_merge_f32_qa(qa0, vdefault, vmask);
      auto out1 = __dtu_m_mop_merge_f32_qa(qa1, vdefault, vmask);
      auto out2 = __dtu_m_mop_merge_f32_qa(qa2, vdefault, vmask);
      auto out3 = __dtu_m_mop_merge_f32_qa(qa3, vdefault, vmask);
      auto out4 = __dtu_m_mop_merge_f32_qa(qa4, vdefault, vmask);
      auto out5 = __dtu_m_mop_merge_f32_qa(qa5, vdefault, vmask);
      auto out6 = __dtu_m_mop_merge_f32_qa(qa6, vdefault, vmask);
      auto out7 = __dtu_m_mop_merge_f32_qa(qa7, vdefault, vmask);
      auto out8 = __dtu_m_mop_merge_f32_qa(qa8, vdefault, vmask);
      auto out9 = __dtu_m_mop_merge_f32_qa(qa9, vdefault, vmask);
      auto out10 = __dtu_m_mop_merge_f32_qa(qa10, vdefault, vmask);
      auto out11 = __dtu_m_mop_merge_f32_qa(qa11, vdefault, vmask);
      auto out12 = __dtu_m_mop_merge_f32_qa(qa12, vdefault, vmask);
      auto out13 = __dtu_m_mop_merge_f32_qa(qa13, vdefault, vmask);
      auto out14 = __dtu_m_mop_merge_f32_qa(qa14, vdefault, vmask);
      auto out15 = __dtu_m_mop_merge_f32_qa(qa15, vdefault, vmask);

      auto da0 = __dtu_extractqa2da_f32(out0, 0);
      auto da1 = __dtu_extractqa2da_f32(out0, 1);
      auto da2 = __dtu_extractqa2da_f32(out1, 0);
      auto da3 = __dtu_extractqa2da_f32(out1, 1);
      auto da4 = __dtu_extractqa2da_f32(out2, 0);
      auto da5 = __dtu_extractqa2da_f32(out2, 1);
      auto da6 = __dtu_extractqa2da_f32(out3, 0);
      auto da7 = __dtu_extractqa2da_f32(out3, 1);
      auto da8 = __dtu_extractqa2da_f32(out4, 0);
      auto da9 = __dtu_extractqa2da_f32(out4, 1);
      auto da10 = __dtu_extractqa2da_f32(out5, 0);
      auto da11 = __dtu_extractqa2da_f32(out5, 1);
      auto da12 = __dtu_extractqa2da_f32(out6, 0);
      auto da13 = __dtu_extractqa2da_f32(out6, 1);
      auto da14 = __dtu_extractqa2da_f32(out7, 0);
      auto da15 = __dtu_extractqa2da_f32(out7, 1);

      auto da16 = __dtu_extractqa2da_f32(out8, 0);
      auto da17 = __dtu_extractqa2da_f32(out8, 1);
      auto da18 = __dtu_extractqa2da_f32(out9, 0);
      auto da19 = __dtu_extractqa2da_f32(out9, 1);
      auto da20 = __dtu_extractqa2da_f32(out10, 0);
      auto da21 = __dtu_extractqa2da_f32(out10, 1);
      auto da22 = __dtu_extractqa2da_f32(out11, 0);
      auto da23 = __dtu_extractqa2da_f32(out11, 1);
      auto da24 = __dtu_extractqa2da_f32(out12, 0);
      auto da25 = __dtu_extractqa2da_f32(out12, 1);
      auto da26 = __dtu_extractqa2da_f32(out13, 0);
      auto da27 = __dtu_extractqa2da_f32(out13, 1);
      auto da28 = __dtu_extractqa2da_f32(out14, 0);
      auto da29 = __dtu_extractqa2da_f32(out14, 1);
      auto da30 = __dtu_extractqa2da_f32(out15, 0);
      auto da31 = __dtu_extractqa2da_f32(out15, 1);

      __dtu_v_tvstda_f32_dual(da0, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da1, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da2, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da3, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da4, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da5, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da6, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da7, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da8, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da9, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da10, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da11, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da12, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da13, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da14, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da15, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da16, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da17, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da18, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da19, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da20, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da21, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da22, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da23, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da24, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da25, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da26, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da27, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da28, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da29, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da30, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da31, out_tar, out_off1);
    }
  }

  int in_tar_addr = in_addr >> 6;
  int stride = dim0 / 16;
  const int unroll_time = 8;
  int back_off = -(unroll_time - 1) * stride + 1;
  int next_st_off = unroll_time * stride + 1;
  int next_ld_off = (2 * unroll_time - 1) * stride;

  in_tar_addr = TAR32(in_tar_addr, in_tar_addr + unroll_time * stride);
  tar_t in_tar = __dtu_s_movsr2targ(in_tar_addr);
  tar_t in_off0 = __dtu_s_movsr2tari(TAR32(stride, stride), in_tar);
  tar_t in_off1 = __dtu_s_movsr2tari(TAR32(back_off, back_off), in_tar);
  tar_t in_off2 = __dtu_s_movsr2tari(TAR32(-stride, -stride), in_tar);
  tar_t in_off3 = __dtu_s_movsr2tari(TAR32(next_ld_off, next_ld_off), in_tar);

  tar_t exp_tar = __dtu_s_movsr2targ(in_tar_addr);
  tar_t exp_off0 = __dtu_s_movsr2tari(TAR32(stride, stride), exp_tar);
  tar_t exp_off1 = __dtu_s_movsr2tari(TAR32(back_off, back_off), exp_tar);
  tar_t exp_off2 = __dtu_s_movsr2tari(TAR32(-stride, -stride), exp_tar);
  tar_t exp_off3 = __dtu_s_movsr2tari(TAR32(next_ld_off, next_ld_off), exp_tar);

  int out_tar_addr = out_addr >> 6;
  out_tar_addr = TAR32(out_tar_addr, out_tar_addr + unroll_time * stride);
  tar_t out_tar = __dtu_s_movsr2targ(out_tar_addr);
  tar_t out_off0 = __dtu_s_movsr2tari(TAR32(stride, stride), out_tar);
  tar_t out_off1 = __dtu_s_movsr2tari(TAR32(back_off, back_off), out_tar);
  tar_t out_off2 = __dtu_s_movsr2tari(TAR32(next_st_off, next_st_off), out_tar);

  for (int j = 0; j < dim1; j += unroll_time *2) {
    auto vr0 = __dtu_s_tvld_itar(in_tar, in_off0);
    auto vr1 = __dtu_s_tvld_itar(in_tar, in_off0);
    auto vr2 = __dtu_s_tvld_itar(in_tar, in_off0);
    auto vr3 = __dtu_s_tvld_itar(in_tar, in_off0);
    auto vr4 = __dtu_s_tvld_itar(in_tar, in_off0);
    auto vr5 = __dtu_s_tvld_itar(in_tar, in_off0);
    auto vr6 = __dtu_s_tvld_itar(in_tar, in_off0);
    auto vr7 = __dtu_s_tvld_itar(in_tar, in_off1);

    v32f16 vr[8];
    __dtu_c_movsr2vpr(1);

  #pragma clang loop unroll(disable)
    for (int i = 1; i < stride; i++) {
      vr0 = __dtu_v_vpr_vmax_a_f32(vr0, vr[0]);
      vr[0] = __dtu_s_tvld_itar(in_tar, in_off0);

      vr1 = __dtu_v_vpr_vmax_a_f32(vr1, vr[1]);
      vr[1] = __dtu_s_tvld_itar(in_tar, in_off0);

      vr2 = __dtu_v_vpr_vmax_a_f32(vr2, vr[2]);
      vr[2] = __dtu_s_tvld_itar(in_tar, in_off0);

      vr3 = __dtu_v_vpr_vmax_a_f32(vr3, vr[3]);
      vr[3] = __dtu_s_tvld_itar(in_tar, in_off0);

      vr4 = __dtu_v_vpr_vmax_a_f32(vr4, vr[4]);
      vr[4] = __dtu_s_tvld_itar(in_tar, in_off0);

      vr5 = __dtu_v_vpr_vmax_a_f32(vr5, vr[5]);
      vr[5] = __dtu_s_tvld_itar(in_tar, in_off0);

      vr6 = __dtu_v_vpr_vmax_a_f32(vr6, vr[6]);
      vr[6] = __dtu_s_tvld_itar(in_tar, in_off0);

      vr7 = __dtu_v_vpr_vmax_a_f32(vr7, vr[7]);
      vr[7] = __dtu_s_tvld_itar(in_tar, in_off1);

      __dtu_c_movsr2vpr(0);
    }

    in_tar = __dtu_v_taradd(in_tar, in_off2);
    vr0 = __dtu_v_vpr_vmax_a_f32(vr0, vr[0]);
    vr1 = __dtu_v_vpr_vmax_a_f32(vr1, vr[1]);
    vr2 = __dtu_v_vpr_vmax_a_f32(vr2, vr[2]);
    vr3 = __dtu_v_vpr_vmax_a_f32(vr3, vr[3]);
    vr4 = __dtu_v_vpr_vmax_a_f32(vr4, vr[4]);
    vr5 = __dtu_v_vpr_vmax_a_f32(vr5, vr[5]);
    vr6 = __dtu_v_vpr_vmax_a_f32(vr6, vr[6]);
    vr7 = __dtu_v_vpr_vmax_a_f32(vr7, vr[7]);

    vr[0] = __dtu_l_movsfti_r_qw(vr0, 2);
    vr[1] = __dtu_l_movsfti_r_qw(vr1, 2);
    vr[2] = __dtu_l_movsfti_r_qw(vr2, 2);
    vr[3] = __dtu_l_movsfti_r_qw(vr3, 2);
    vr[4] = __dtu_l_movsfti_r_qw(vr4, 2);
    vr[5] = __dtu_l_movsfti_r_qw(vr5, 2);
    vr[6] = __dtu_l_movsfti_r_qw(vr6, 2);
    vr[7] = __dtu_l_movsfti_r_qw(vr7, 2);

    vr0 = __dtu_v_vmax_a_f32(vr0, vr[0]);
    vr1 = __dtu_v_vmax_a_f32(vr1, vr[1]);
    vr2 = __dtu_v_vmax_a_f32(vr2, vr[2]);
    vr3 = __dtu_v_vmax_a_f32(vr3, vr[3]);
    vr4 = __dtu_v_vmax_a_f32(vr4, vr[4]);
    vr5 = __dtu_v_vmax_a_f32(vr5, vr[5]);
    vr6 = __dtu_v_vmax_a_f32(vr6, vr[6]);
    vr7 = __dtu_v_vmax_a_f32(vr7, vr[7]);

    vr[0] = __dtu_l_movsfti_r_qw(vr0, 1);
    vr[1] = __dtu_l_movsfti_r_qw(vr1, 1);
    vr[2] = __dtu_l_movsfti_r_qw(vr2, 1);
    vr[3] = __dtu_l_movsfti_r_qw(vr3, 1);
    vr[4] = __dtu_l_movsfti_r_qw(vr4, 1);
    vr[5] = __dtu_l_movsfti_r_qw(vr5, 1);
    vr[6] = __dtu_l_movsfti_r_qw(vr6, 1);
    vr[7] = __dtu_l_movsfti_r_qw(vr7, 1);

    vr0 = __dtu_v_vmax_a_f32(vr0, vr[0]);
    vr1 = __dtu_v_vmax_a_f32(vr1, vr[1]);
    vr2 = __dtu_v_vmax_a_f32(vr2, vr[2]);
    vr3 = __dtu_v_vmax_a_f32(vr3, vr[3]);
    vr4 = __dtu_v_vmax_a_f32(vr4, vr[4]);
    vr5 = __dtu_v_vmax_a_f32(vr5, vr[5]);
    vr6 = __dtu_v_vmax_a_f32(vr6, vr[6]);
    vr7 = __dtu_v_vmax_a_f32(vr7, vr[7]);

    vr[0] = __dtu_l_movsfti_r_b(vr0, 8);
    vr[1] = __dtu_l_movsfti_r_b(vr1, 8);
    vr[2] = __dtu_l_movsfti_r_b(vr2, 8);
    vr[3] = __dtu_l_movsfti_r_b(vr3, 8);
    vr[4] = __dtu_l_movsfti_r_b(vr4, 8);
    vr[5] = __dtu_l_movsfti_r_b(vr5, 8);
    vr[6] = __dtu_l_movsfti_r_b(vr6, 8);
    vr[7] = __dtu_l_movsfti_r_b(vr7, 8);

    vr0 = __dtu_v_vmax_a_f32(vr0, vr[0]);
    vr1 = __dtu_v_vmax_a_f32(vr1, vr[1]);
    vr2 = __dtu_v_vmax_a_f32(vr2, vr[2]);
    vr3 = __dtu_v_vmax_a_f32(vr3, vr[3]);
    vr4 = __dtu_v_vmax_a_f32(vr4, vr[4]);
    vr5 = __dtu_v_vmax_a_f32(vr5, vr[5]);
    vr6 = __dtu_v_vmax_a_f32(vr6, vr[6]);
    vr7 = __dtu_v_vmax_a_f32(vr7, vr[7]);

    vr[0] = __dtu_l_movsfti_r_b(vr0, 4);
    vr[1] = __dtu_l_movsfti_r_b(vr1, 4);
    vr[2] = __dtu_l_movsfti_r_b(vr2, 4);
    vr[3] = __dtu_l_movsfti_r_b(vr3, 4);
    vr[4] = __dtu_l_movsfti_r_b(vr4, 4);
    vr[5] = __dtu_l_movsfti_r_b(vr5, 4);
    vr[6] = __dtu_l_movsfti_r_b(vr6, 4);
    vr[7] = __dtu_l_movsfti_r_b(vr7, 4);

    vr0 = __dtu_v_vmax_a_f32(vr0, vr[0]);
    va16f32 va_max0 = __dtu_l_vclr_f32_va();
    vr1 = __dtu_v_vmax_a_f32(vr1, vr[1]);
    va16f32 va_max1 = __dtu_l_vclr_f32_va();
    vr2 = __dtu_v_vmax_a_f32(vr2, vr[2]);
    va16f32 va_max2 = __dtu_l_vclr_f32_va();
    vr3 = __dtu_v_vmax_a_f32(vr3, vr[3]);
    va16f32 va_max3 = __dtu_l_vclr_f32_va();
    vr4 = __dtu_v_vmax_a_f32(vr4, vr[4]);
    va16f32 va_max4 = __dtu_l_vclr_f32_va();
    vr5 = __dtu_v_vmax_a_f32(vr5, vr[5]);
    va16f32 va_max5 = __dtu_l_vclr_f32_va();
    vr6 = __dtu_v_vmax_a_f32(vr6, vr[6]);
    va16f32 va_max6 = __dtu_l_vclr_f32_va();
    vr7 = __dtu_v_vmax_a_f32(vr7, vr[7]);
    va16f32 va_max7 = __dtu_l_vclr_f32_va();

    smr_t smr0 = __dtu_m_clrsmr();
    auto vr31 = __dtu_s_movr2vr_dup(0x3f800000);
    auto va_ln2 = __dtu_l_movr2va(0x3fb8aa3b);  // 1/Ln2
    smr0 = __dtu_m_ldsmr_mode3_f_row(smr0, vr31, 0);

    va_max0 = __dtu_m_vmm_mode3_f_nacc_vs0(va_max0, vr0, smr0);
    va_max1 = __dtu_m_vmm_mode3_f_nacc_vs0(va_max1, vr1, smr0);
    va_max2 = __dtu_m_vmm_mode3_f_nacc_vs0(va_max2, vr2, smr0);
    va_max3 = __dtu_m_vmm_mode3_f_nacc_vs0(va_max3, vr3, smr0);
    va_max4 = __dtu_m_vmm_mode3_f_nacc_vs0(va_max4, vr4, smr0);
    va_max5 = __dtu_m_vmm_mode3_f_nacc_vs0(va_max5, vr5, smr0);
    va_max6 = __dtu_m_vmm_mode3_f_nacc_vs0(va_max6, vr6, smr0);
    va_max7 = __dtu_m_vmm_mode3_f_nacc_vs0(va_max7, vr7, smr0);

    auto va_sum0 = __dtu_l_vclr_f32_va();
    auto va_sum1 = __dtu_l_vclr_f32_va();
    auto va_sum2 = __dtu_l_vclr_f32_va();
    auto va_sum3 = __dtu_l_vclr_f32_va();
    auto va_sum4 = __dtu_l_vclr_f32_va();
    auto va_sum5 = __dtu_l_vclr_f32_va();
    auto va_sum6 = __dtu_l_vclr_f32_va();
    auto va_sum7 = __dtu_l_vclr_f32_va();

    auto va0 = __dtu_l_tvldqa_f32_va(in_tar, in_off0);
    auto va1 = __dtu_l_tvldqa_f32_va(in_tar, in_off0);
    auto va2 = __dtu_l_tvldqa_f32_va(in_tar, in_off0);
    auto va3 = __dtu_l_tvldqa_f32_va(in_tar, in_off0);
    auto va4 = __dtu_l_tvldqa_f32_va(in_tar, in_off0);
    auto va5 = __dtu_l_tvldqa_f32_va(in_tar, in_off0);
    auto va6 = __dtu_l_tvldqa_f32_va(in_tar, in_off0);
    auto va7 = __dtu_l_tvldqa_f32_va(in_tar, in_off1);

    int lpr_cnt = stride - 2;
    int lpr_ind = __dtu_srli_a(lpr_cnt, 31);
    __dtu_c_movsr2vpr(1);
    __dtu_c_movsr2lpr(lpr_ind);
    va32f16 va8, va9, va10, va11, va12, va13, va14, va15;

  #pragma clang loop unroll(disable)
    for (int i = 0; i < stride; i++) {
      __dtu_v_vpr_tvstda_f32(va8, exp_tar, exp_off0);
      va0 = __dtu_m_mop_sub_f32_va(va0, va_max0);
      __dtu_v_vpr_tvstda_f32(va9, exp_tar, exp_off0);
      va1 = __dtu_m_mop_sub_f32_va(va1, va_max1);
      __dtu_v_vpr_tvstda_f32(va10, exp_tar, exp_off0);
      va2 = __dtu_m_mop_sub_f32_va(va2, va_max2);
      __dtu_v_vpr_tvstda_f32(va11, exp_tar, exp_off0);
      va3 = __dtu_m_mop_sub_f32_va(va3, va_max3);
      __dtu_v_vpr_tvstda_f32(va12, exp_tar, exp_off0);
      va4 = __dtu_m_mop_sub_f32_va(va4, va_max4);
      __dtu_v_vpr_tvstda_f32(va13, exp_tar, exp_off0);
      va5 = __dtu_m_mop_sub_f32_va(va5, va_max5);
      __dtu_v_vpr_tvstda_f32(va14, exp_tar, exp_off0);
      va6 = __dtu_m_mop_sub_f32_va(va6, va_max6);
      __dtu_v_vpr_tvstda_f32(va15, exp_tar, exp_off1);
      va7 = __dtu_m_mop_sub_f32_va(va7, va_max7);
      __dtu_c_movsr2vpr(0);

      va8 = __dtu_m_mop_mul_f32_va(va0, va_ln2);
      va0 = __dtu_l_lpr_tvldqa_f32_va(in_tar, in_off0);
      va9 = __dtu_m_mop_mul_f32_va(va1, va_ln2);
      va1 = __dtu_l_lpr_tvldqa_f32_va(in_tar, in_off0);
      va10 = __dtu_m_mop_mul_f32_va(va2, va_ln2);
      va2 = __dtu_l_lpr_tvldqa_f32_va(in_tar, in_off0);
      va11 = __dtu_m_mop_mul_f32_va(va3, va_ln2);
      va3 = __dtu_l_lpr_tvldqa_f32_va(in_tar, in_off0);
      va12 = __dtu_m_mop_mul_f32_va(va4, va_ln2);
      va4 = __dtu_l_lpr_tvldqa_f32_va(in_tar, in_off0);
      va13 = __dtu_m_mop_mul_f32_va(va5, va_ln2);
      va5 = __dtu_l_lpr_tvldqa_f32_va(in_tar, in_off0);
      va14 = __dtu_m_mop_mul_f32_va(va6, va_ln2);
      va6 = __dtu_l_lpr_tvldqa_f32_va(in_tar, in_off0);
      va15 = __dtu_m_mop_mul_f32_va(va7, va_ln2);
      va7 = __dtu_l_lpr_tvldqa_f32_va(in_tar, in_off1);

      lpr_cnt--;
      lpr_ind = __dtu_srli_a(lpr_cnt, 31);
      __dtu_c_movsr2lpr(lpr_ind);

      va8 = __dtu_m_msf_exp_f32(va8);
      va9 = __dtu_m_msf_exp_f32(va9);
      va10 = __dtu_m_msf_exp_f32(va10);
      va11 = __dtu_m_msf_exp_f32(va11);
      va12 = __dtu_m_msf_exp_f32(va12);
      va13 = __dtu_m_msf_exp_f32(va13);
      va14 = __dtu_m_msf_exp_f32(va14);
      va15 = __dtu_m_msf_exp_f32(va15);

      va_sum0 = __dtu_m_mop_add_f32_va(va_sum0, va8);
      va_sum1 = __dtu_m_mop_add_f32_va(va_sum1, va9);
      va_sum2 = __dtu_m_mop_add_f32_va(va_sum2, va10);
      va_sum3 = __dtu_m_mop_add_f32_va(va_sum3, va11);
      va_sum4 = __dtu_m_mop_add_f32_va(va_sum4, va12);
      va_sum5 = __dtu_m_mop_add_f32_va(va_sum5, va13);
      va_sum6 = __dtu_m_mop_add_f32_va(va_sum6, va14);
      va_sum7 = __dtu_m_mop_add_f32_va(va_sum7, va15);
    }

    in_tar = __dtu_v_taradd(in_tar, in_off3);
    __dtu_v_tvstda_f32(va8, exp_tar, exp_off0);
    __dtu_v_tvstda_f32(va9, exp_tar, exp_off0);
    __dtu_v_tvstda_f32(va10, exp_tar, exp_off0);
    __dtu_v_tvstda_f32(va11, exp_tar, exp_off0);
    __dtu_v_tvstda_f32(va12, exp_tar, exp_off0);
    __dtu_v_tvstda_f32(va13, exp_tar, exp_off0);
    __dtu_v_tvstda_f32(va14, exp_tar, exp_off0);
    __dtu_v_tvstda_f32(va15, exp_tar, exp_off1);
    exp_tar = __dtu_v_taradd(exp_tar, exp_off2);

    smr_t smr1 = __dtu_m_clrsmr();
    vr31 = __dtu_s_movr2vr_dup(0x3f800000);
    smr1 = __dtu_m_ldsmr_mode3_f_row(smr1, vr31, 0);
    smr1 = __dtu_m_ldsmr_mode3_f_row(smr1, vr31, 1);
    smr1 = __dtu_m_ldsmr_mode3_f_row(smr1, vr31, 2);
    smr1 = __dtu_m_ldsmr_mode3_f_row(smr1, vr31, 3);
    smr1 = __dtu_m_ldsmr_mode3_f_row(smr1, vr31, 4);
    smr1 = __dtu_m_ldsmr_mode3_f_row(smr1, vr31, 5);
    smr1 = __dtu_m_ldsmr_mode3_f_row(smr1, vr31, 6);
    smr1 = __dtu_m_ldsmr_mode3_f_row(smr1, vr31, 7);
    smr1 = __dtu_m_ldsmr_mode3_f_row(smr1, vr31, 8);
    smr1 = __dtu_m_ldsmr_mode3_f_row(smr1, vr31, 9);
    smr1 = __dtu_m_ldsmr_mode3_f_row(smr1, vr31, 10);
    smr1 = __dtu_m_ldsmr_mode3_f_row(smr1, vr31, 11);
    smr1 = __dtu_m_ldsmr_mode3_f_row(smr1, vr31, 12);
    smr1 = __dtu_m_ldsmr_mode3_f_row(smr1, vr31, 13);
    smr1 = __dtu_m_ldsmr_mode3_f_row(smr1, vr31, 14);
    smr1 = __dtu_m_ldsmr_mode3_f_row(smr1, vr31, 15);

    auto vacc0 = __dtu_l_vclr_va();
    auto vacc1 = __dtu_l_vclr_va();
    auto vacc2 = __dtu_l_vclr_va();
    auto vacc3 = __dtu_l_vclr_va();
    auto vacc4 = __dtu_l_vclr_va();
    auto vacc5 = __dtu_l_vclr_va();
    auto vacc6 = __dtu_l_vclr_va();
    auto vacc7 = __dtu_l_vclr_va();

    vacc0 = __dtu_m_vmm_mode3_f_nacc_vs0(vacc0, va_sum0, smr1);
    vacc1 = __dtu_m_vmm_mode3_f_nacc_vs0(vacc1, va_sum1, smr1);
    vacc2 = __dtu_m_vmm_mode3_f_nacc_vs0(vacc2, va_sum2, smr1);
    vacc3 = __dtu_m_vmm_mode3_f_nacc_vs0(vacc3, va_sum3, smr1);
    vacc4 = __dtu_m_vmm_mode3_f_nacc_vs0(vacc4, va_sum4, smr1);
    vacc5 = __dtu_m_vmm_mode3_f_nacc_vs0(vacc5, va_sum5, smr1);
    vacc6 = __dtu_m_vmm_mode3_f_nacc_vs0(vacc6, va_sum6, smr1);
    vacc7 = __dtu_m_vmm_mode3_f_nacc_vs0(vacc7, va_sum7, smr1);

    vacc0 = __dtu_m_msf_rec_f32(vacc0);
    vacc1 = __dtu_m_msf_rec_f32(vacc1);
    vacc2 = __dtu_m_msf_rec_f32(vacc2);
    vacc3 = __dtu_m_msf_rec_f32(vacc3);
    vacc4 = __dtu_m_msf_rec_f32(vacc4);
    vacc5 = __dtu_m_msf_rec_f32(vacc5);
    vacc6 = __dtu_m_msf_rec_f32(vacc6);
    vacc7 = __dtu_m_msf_rec_f32(vacc7);

    va0 = __dtu_l_tvldqa_f32_va(exp_tar, exp_off0);
    va1 = __dtu_l_tvldqa_f32_va(exp_tar, exp_off0);
    va2 = __dtu_l_tvldqa_f32_va(exp_tar, exp_off0);
    va3 = __dtu_l_tvldqa_f32_va(exp_tar, exp_off0);
    va4 = __dtu_l_tvldqa_f32_va(exp_tar, exp_off0);
    va5 = __dtu_l_tvldqa_f32_va(exp_tar, exp_off0);
    va6 = __dtu_l_tvldqa_f32_va(exp_tar, exp_off0);
    va7 = __dtu_l_tvldqa_f32_va(exp_tar, exp_off1);

    __dtu_c_movsr2vpr(1);
    lpr_cnt = stride - 2;
    lpr_ind = __dtu_srli_a(lpr_cnt, 31);
    __dtu_c_movsr2lpr(lpr_ind);
    va16f32 va[8];
  #pragma clang loop unroll(disable)
    for (int i = 0; i < stride; i++) {
      __dtu_v_vpr_tvsta_w(va[0], out_tar, out_off0);
      va[0] = __dtu_m_mop_mul_f32_va(va0, vacc0);
      va0 = __dtu_l_lpr_tvldqa_f32_va(exp_tar, exp_off0);

      __dtu_v_vpr_tvsta_w(va[1], out_tar, out_off0);
      va[1] = __dtu_m_mop_mul_f32_va(va1, vacc1);
      va1 = __dtu_l_lpr_tvldqa_f32_va(exp_tar, exp_off0);

      __dtu_v_vpr_tvsta_w(va[2], out_tar, out_off0);
      va[2] = __dtu_m_mop_mul_f32_va(va2, vacc2);
      va2 = __dtu_l_lpr_tvldqa_f32_va(exp_tar, exp_off0);

      __dtu_v_vpr_tvsta_w(va[3], out_tar, out_off0);
      va[3] = __dtu_m_mop_mul_f32_va(va3, vacc3);
      va3 = __dtu_l_lpr_tvldqa_f32_va(exp_tar, exp_off0);

      __dtu_v_vpr_tvsta_w(va[4], out_tar, out_off0);
      va[4] = __dtu_m_mop_mul_f32_va(va4, vacc4);
      va4 = __dtu_l_lpr_tvldqa_f32_va(exp_tar, exp_off0);

      __dtu_v_vpr_tvsta_w(va[5], out_tar, out_off0);
      va[5] = __dtu_m_mop_mul_f32_va(va5, vacc5);
      va5 = __dtu_l_lpr_tvldqa_f32_va(exp_tar, exp_off0);

      __dtu_v_vpr_tvsta_w(va[6], out_tar, out_off0);
      va[6] = __dtu_m_mop_mul_f32_va(va6, vacc6);
      va6 = __dtu_l_lpr_tvldqa_f32_va(exp_tar, exp_off0);

      __dtu_v_vpr_tvsta_w(va[7], out_tar, out_off1);
      va[7] = __dtu_m_mop_mul_f32_va(va7, vacc7);
      va7 = __dtu_l_lpr_tvldqa_f32_va(exp_tar, exp_off1);

      __dtu_c_movsr2vpr(0);
      lpr_cnt--;
      lpr_ind = __dtu_srli_a(lpr_cnt, 31);
      __dtu_c_movsr2lpr(lpr_ind);
    }

    exp_tar = __dtu_v_taradd(exp_tar, exp_off3);
    __dtu_v_tvsta_w(va[0], out_tar, out_off0);
    __dtu_v_tvsta_w(va[1], out_tar, out_off0);
    __dtu_v_tvsta_w(va[2], out_tar, out_off0);
    __dtu_v_tvsta_w(va[3], out_tar, out_off0);
    __dtu_v_tvsta_w(va[4], out_tar, out_off0);
    __dtu_v_tvsta_w(va[5], out_tar, out_off0);
    __dtu_v_tvsta_w(va[6], out_tar, out_off0);
    __dtu_v_tvsta_w(va[7], out_tar, out_off2);
  }
}

__device__
void call_softmax_qa(float* dst_ptr, float* in_ptr, float* p_max,
                    int dim1, int dim0, int left_ctx_len) {
  const int unroll_time = 16;
  const int bpe = 4;
  int out_max = reinterpret_cast<long>(p_max);
  int out_addr0 = reinterpret_cast<long>(dst_ptr);
  int in_addr0 = reinterpret_cast<long>(in_ptr);
  int out_addr1 = out_addr0 + unroll_time * dim0 * bpe;
  int in_addr1 = in_addr0 + unroll_time * dim0 * bpe;
  int qa_step = dim0 / 64;
  int qa_back_step = -dim0 * (unroll_time - 1) / 64 + 1;
  int da_step = dim0 / 32;
  int da_back_step = -dim0 * (unroll_time - 1) / 32 + 1;

  if (left_ctx_len < dim0) {
    auto vdefault = __dtu_l_movr2qa_u32(0xff800000);
    auto mask_in = __dtu_m_mid_m0_u32(0);
    auto mask_val = __dtu_l_movr2qa_u32(left_ctx_len & 0x7f);
    auto vmask = __dtu_m_mop_mslt_u32_qa(mask_in, mask_val);
    int tar_addr = (in_addr0 + dim0 * 4 - 512) >> 8;
    int stride = dim0 / 64;
    tar_t in_tar = __dtu_s_movsr2targ(TAR32(tar_addr, tar_addr + 1));
    tar_t in_off0 = __dtu_s_movsr2tari(TAR32(stride, stride), in_tar);

    tar_addr = (in_addr0 + dim0 * 4 - 512) >> 7;
    stride = dim0 / 32;
    tar_t out_tar = __dtu_s_movsr2targ(TAR32(tar_addr, tar_addr + 2));
    tar_t out_off0 = __dtu_s_movsr2tari(TAR32(1, 1), out_tar);
    tar_t out_off1 = __dtu_s_movsr2tari(TAR32(stride - 1, stride - 1), out_tar);

#pragma clang loop unroll(disable)
    for (int i = 0; i < dim1; i += 16) {
      auto qa0 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa1 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa2 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa3 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa4 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa5 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa6 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa7 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa8 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa9 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa10 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa11 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa12 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa13 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa14 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);
      auto qa15 = __dtu_l_tvldqa_f32_qa(in_tar, in_off0);

      auto out0 = __dtu_m_mop_merge_f32_qa(qa0, vdefault, vmask);
      auto out1 = __dtu_m_mop_merge_f32_qa(qa1, vdefault, vmask);
      auto out2 = __dtu_m_mop_merge_f32_qa(qa2, vdefault, vmask);
      auto out3 = __dtu_m_mop_merge_f32_qa(qa3, vdefault, vmask);
      auto out4 = __dtu_m_mop_merge_f32_qa(qa4, vdefault, vmask);
      auto out5 = __dtu_m_mop_merge_f32_qa(qa5, vdefault, vmask);
      auto out6 = __dtu_m_mop_merge_f32_qa(qa6, vdefault, vmask);
      auto out7 = __dtu_m_mop_merge_f32_qa(qa7, vdefault, vmask);
      auto out8 = __dtu_m_mop_merge_f32_qa(qa8, vdefault, vmask);
      auto out9 = __dtu_m_mop_merge_f32_qa(qa9, vdefault, vmask);
      auto out10 = __dtu_m_mop_merge_f32_qa(qa10, vdefault, vmask);
      auto out11 = __dtu_m_mop_merge_f32_qa(qa11, vdefault, vmask);
      auto out12 = __dtu_m_mop_merge_f32_qa(qa12, vdefault, vmask);
      auto out13 = __dtu_m_mop_merge_f32_qa(qa13, vdefault, vmask);
      auto out14 = __dtu_m_mop_merge_f32_qa(qa14, vdefault, vmask);
      auto out15 = __dtu_m_mop_merge_f32_qa(qa15, vdefault, vmask);

      auto da0 = __dtu_extractqa2da_f32(out0, 0);
      auto da1 = __dtu_extractqa2da_f32(out0, 1);
      auto da2 = __dtu_extractqa2da_f32(out1, 0);
      auto da3 = __dtu_extractqa2da_f32(out1, 1);
      auto da4 = __dtu_extractqa2da_f32(out2, 0);
      auto da5 = __dtu_extractqa2da_f32(out2, 1);
      auto da6 = __dtu_extractqa2da_f32(out3, 0);
      auto da7 = __dtu_extractqa2da_f32(out3, 1);
      auto da8 = __dtu_extractqa2da_f32(out4, 0);
      auto da9 = __dtu_extractqa2da_f32(out4, 1);
      auto da10 = __dtu_extractqa2da_f32(out5, 0);
      auto da11 = __dtu_extractqa2da_f32(out5, 1);
      auto da12 = __dtu_extractqa2da_f32(out6, 0);
      auto da13 = __dtu_extractqa2da_f32(out6, 1);
      auto da14 = __dtu_extractqa2da_f32(out7, 0);
      auto da15 = __dtu_extractqa2da_f32(out7, 1);

      auto da16 = __dtu_extractqa2da_f32(out8, 0);
      auto da17 = __dtu_extractqa2da_f32(out8, 1);
      auto da18 = __dtu_extractqa2da_f32(out9, 0);
      auto da19 = __dtu_extractqa2da_f32(out9, 1);
      auto da20 = __dtu_extractqa2da_f32(out10, 0);
      auto da21 = __dtu_extractqa2da_f32(out10, 1);
      auto da22 = __dtu_extractqa2da_f32(out11, 0);
      auto da23 = __dtu_extractqa2da_f32(out11, 1);
      auto da24 = __dtu_extractqa2da_f32(out12, 0);
      auto da25 = __dtu_extractqa2da_f32(out12, 1);
      auto da26 = __dtu_extractqa2da_f32(out13, 0);
      auto da27 = __dtu_extractqa2da_f32(out13, 1);
      auto da28 = __dtu_extractqa2da_f32(out14, 0);
      auto da29 = __dtu_extractqa2da_f32(out14, 1);
      auto da30 = __dtu_extractqa2da_f32(out15, 0);
      auto da31 = __dtu_extractqa2da_f32(out15, 1);

      __dtu_v_tvstda_f32_dual(da0, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da1, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da2, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da3, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da4, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da5, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da6, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da7, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da8, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da9, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da10, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da11, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da12, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da13, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da14, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da15, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da16, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da17, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da18, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da19, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da20, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da21, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da22, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da23, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da24, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da25, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da26, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da27, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da28, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da29, out_tar, out_off1);
      __dtu_v_tvstda_f32_dual(da30, out_tar, out_off0);
      __dtu_v_tvstda_f32_dual(da31, out_tar, out_off1);
    }
  }

  int loop_times = dim0 * 2 / TOPS_VECTOR_LENGTH;

  asm volatile (
    " l.ldi_hi.m1 r10, 0x3f80               \n"
    " m.smrclr smr0                         \n"
    " l.movr2da dacc2, r10                  \n"
    " m.ldsmr2.mode18.f.row smr0, dacc2, 0  \n"

    " m.ldsmr2.mode18.f.row smr1, dacc2, 0     \n"
    " m.ldsmr2.mode18.f.row smr1, dacc2, 1     \n"
    " m.ldsmr2.mode18.f.row smr1, dacc2, 2     \n"

    " m.ldsmr2.mode18.f.row smr1, dacc2, 3     \n"
    " l.movsr2targ.ext.t0 ta_g0, %0         \n"
    " m.ldsmr2.mode18.f.row smr1, dacc2, 4     \n"
    " l.movsr2targ.ext.t1 ta_g0, %1         \n"
    " m.ldsmr2.mode18.f.row smr1, dacc2, 5     \n"
    " l.movsr2tari.ext.t0 [ta_g0, 1], %2    \n"
    " m.ldsmr2.mode18.f.row smr1, dacc2, 6     \n"
    " l.movsr2tari.ext.t1 [ta_g0, 1], %2    \n"
    " m.ldsmr2.mode18.f.row smr1, dacc2, 7     \n"
    " l.movsr2tari.ext.t0 [ta_g0, 2], %3    \n"
    " m.ldsmr2.mode18.f.row smr1, dacc2, 8     \n"
    " l.movsr2tari.ext.t1 [ta_g0, 2], %3    \n"
    " m.ldsmr2.mode18.f.row smr1, dacc2, 9     \n"
    " l.movsr2tari.ext.t0 [ta_g0, 3], %4    \n"

    " m.ldsmr2.mode18.f.row smr1, dacc2, 10     \n"
    " l.movsr2tari.ext.t1 [ta_g0, 3], %4    \n"
    " m.ldsmr2.mode18.f.row smr1, dacc2, 11     \n"
    " l.ldi16.s r8, 1           \n"
    " m.ldsmr2.mode18.f.row smr1, dacc2, 12     \n"
    " l.ldi_hi r8, 1            \n"
    " m.ldsmr2.mode18.f.row smr1, dacc2, 13     \n"
    " m.ldsmr2.mode18.f.row smr1, dacc2, 14     \n"
    " m.ldsmr2.mode18.f.row smr1, dacc2, 15     \n"

    " c.movsr2spr NACCOVR, r8   \n"
    :
    : "r"(in_addr0 >> 8),
      "r"(in_addr1 >> 8),
      "r"(qa_step),
      "r"(qa_back_step),
      "r"(-qa_step)
    : "r8", "r10"
  );

  asm volatile (
    " l.movsr2targ.ext.t0 ta_g3, %0         \n"
    " l.movsr2targ.ext.t1 ta_g3, %1         \n"
    " l.movsr2tari.ext.t0 [ta_g3, 1], %2    \n"
    " l.movsr2tari.ext.t1 [ta_g3, 1], %2    \n"
    " l.movsr2tari.ext.t0 [ta_g3, 2], %3    \n"
    " l.movsr2tari.ext.t1 [ta_g3, 2], %3    \n"
    " l.movsr2tari.ext.t0 [ta_g3, 3], %4    \n"
    " l.movsr2tari.ext.t1 [ta_g3, 3], %4    \n"

    :
    : "r" (out_addr0 >> 7),
      "r" (out_addr1 >> 7),
      "r" (1),
      "r" (da_back_step),
      "r" (da_step - 1)
    :
    );

  asm volatile (
    " l.movsr2targ.ext.t0 ta_g2, %0         \n"
    " l.movsr2targ.ext.t1 ta_g2, %1         \n"
    " l.movsr2tari.ext.t0 [ta_g2, 1], %2    \n"
    " l.movsr2tari.ext.t1 [ta_g2, 1], %2    \n"
    " l.movsr2tari.ext.t0 [ta_g2, 2], %3    \n"
    " l.movsr2tari.ext.t1 [ta_g2, 2], %3    \n"

    :
    : "r" (out_max >> 6),
      "r" ((out_max >> 6) + 16),
      "r" (1),
      "r" (17)
    :
    );

  asm volatile (
      " l.ldi_hi r11, 0xff80  \n"
      " l.ldi_lo r11, 0x0000  \n"
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
      " m.tvldqa.w.qa qacc16, [ta_g0, 2]     \n"
      " l.movr2qa qacc43, r11                  \n"
      :
      :
      : "r11"
      );

#pragma clang loop unroll(disable)
    for (int i = 0; i < loop_times - 1; i++) {
      asm volatile (
        " m.mop.max.f32.qa qacc28, qacc28, qacc1        \n"

        " m.mop.max.f32.qa qacc29, qacc29, qacc2        \n"
        " l.tvldqa.w.qa qacc1, [ta_g0, 1]               \n"

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

        " l.tvldqa.w.qa qacc16, [ta_g0, 2]    \n"
      );
    }

    asm volatile (
    // reduce max to qa
    " s.taradd ta_g0, [ta_g0, 3]                    \n"
    " m.mop.max.f32.qa qacc28, qacc28, qacc1        \n"
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

    // reduce max to da
    "  m.mop.max.f32.da dacc56, dacc56, dacc57        \n"
    "  m.mop.max.f32.da dacc58, dacc58, dacc59        \n"
    "  m.mop.max.f32.da dacc60, dacc60, dacc61        \n"
    "  m.mop.max.f32.da dacc62, dacc62, dacc63        \n"

    "  m.mop.max.f32.da dacc64, dacc64, dacc65        \n"
    "  m.mop.max.f32.da dacc66, dacc66, dacc67        \n"
    "  m.mop.max.f32.da dacc68, dacc68, dacc69        \n"
    "  m.mop.max.f32.da dacc70, dacc70, dacc71        \n"

    "  m.mop.max.f32.da dacc72, dacc72, dacc73        \n"
    "  m.mop.max.f32.da dacc74, dacc74, dacc75        \n"
    "  m.mop.max.f32.da dacc76, dacc76, dacc77        \n"
    "  m.mop.max.f32.da dacc78, dacc78, dacc79        \n"

    "  m.mop.max.f32.da dacc80, dacc80, dacc81        \n"
    "  m.mop.max.f32.da dacc82, dacc82, dacc83        \n"
    "  m.mop.max.f32.da dacc84, dacc84, dacc85        \n"
    "  m.mop.max.f32.da dacc86, dacc86, dacc87        \n"

    "  m.mop.max.f32.va vacc112, vacc112, vacc113         \n"
    "  m.mop.max.f32.va vacc116, vacc116, vacc117         \n"
    "  m.mop.max.f32.va vacc120, vacc120, vacc121         \n"
    "  m.mop.max.f32.va vacc124, vacc124, vacc125         \n"

    "  m.mop.max.f32.va vacc128, vacc128, vacc129         \n"
    "  m.mop.max.f32.va vacc132, vacc132, vacc133         \n"
    "  m.mop.max.f32.va vacc136, vacc136, vacc137         \n"
    "  m.mop.max.f32.va vacc140, vacc140, vacc141         \n"

    "  m.mop.max.f32.va vacc144, vacc144, vacc145         \n"
    "  m.mop.max.f32.va vacc148, vacc148, vacc149         \n"
    "  m.mop.max.f32.va vacc152, vacc152, vacc153         \n"
    "  m.mop.max.f32.va vacc156, vacc156, vacc157         \n"

    "  m.mop.max.f32.va vacc160, vacc160, vacc161         \n"
    "  m.mop.max.f32.va vacc164, vacc164, vacc165         \n"
    "  m.mop.max.f32.va vacc168, vacc168, vacc169         \n"
    "  m.mop.max.f32.va vacc172, vacc172, vacc173         \n"

    " l.movva2vr.w vr0, vacc112                           \n"
    " l.movva2vr.w vr1, vacc116                           \n"
    " l.movva2vr.w vr2, vacc120                           \n"
    " l.movva2vr.w vr3, vacc124                           \n"
    " l.movva2vr.w vr4, vacc128                           \n"
    " l.movva2vr.w vr5, vacc132                           \n"
    " l.movva2vr.w vr6, vacc136                           \n"
    " l.movva2vr.w vr7, vacc140                           \n"
    " l.movva2vr.w vr8, vacc144                           \n"
    " l.movva2vr.w vr9, vacc148                           \n"
    " l.movva2vr.w vr10, vacc152                          \n"
    " l.movva2vr.w vr11, vacc156                          \n"
    " l.movva2vr.w vr12, vacc160                          \n"
    " l.movva2vr.w vr13, vacc164                          \n"
    " l.movva2vr.w vr14, vacc168                          \n"
    " l.movva2vr.w vr15, vacc172                          \n"

    // 第一次右移上半部分
    " l.movsfti.r.qw vr16, vr0, 2                         \n"
    " l.movsfti.r.qw vr17, vr1, 2                         \n"
    " l.movsfti.r.qw vr18, vr2, 2                         \n"
    " l.movsfti.r.qw vr19, vr3, 2                         \n"

    " l.movsfti.r.qw vr20, vr4, 2                         \n"
    " l.movsfti.r.qw vr21, vr5, 2                         \n"
    " l.movsfti.r.qw vr22, vr6, 2                         \n"
    " l.movsfti.r.qw vr23, vr7, 2                         \n"

    // 第一次右移下半部分 + 第一次max上半部分
    " v.vmaxa.f32 vr0, vr0, vr16                          \n"
    " l.movsfti.r.qw vr24, vr8, 2                         \n"

    " v.vmaxa.f32 vr1, vr1, vr17                          \n"
    " l.movsfti.r.qw vr25, vr9, 2                         \n"

    " v.vmaxa.f32 vr2, vr2, vr18                          \n"
    " l.movsfti.r.qw vr26, vr10, 2                         \n"

    " v.vmaxa.f32 vr3, vr3, vr19                          \n"
    " l.movsfti.r.qw vr27, vr11, 2                         \n"

    " v.vmaxa.f32 vr4, vr4, vr20                          \n"
    " l.movsfti.r.qw vr28, vr12, 2                         \n"

    " v.vmaxa.f32 vr5, vr5, vr21                          \n"
    " l.movsfti.r.qw vr29, vr13, 2                         \n"

    " v.vmaxa.f32 vr6, vr6, vr22                          \n"
    " l.movsfti.r.qw vr30, vr14, 2                         \n"

    " v.vmaxa.f32 vr7, vr7, vr23                          \n"
    " l.movsfti.r.qw vr31, vr15, 2                         \n"

    // 第二次右移上半部分 + 第一次max 下半部分
    " v.vmaxa.f32 vr8, vr8, vr24                          \n"
    " l.movsfti.r.qw vr16, vr0, 1                         \n"

    " v.vmaxa.f32 vr9, vr9, vr25                          \n"
    " l.movsfti.r.qw vr17, vr1, 1                         \n"

    " v.vmaxa.f32 vr10, vr10, vr26                          \n"
    " l.movsfti.r.qw vr18, vr2, 1                         \n"

    " v.vmaxa.f32 vr11, vr11, vr27                          \n"
    " l.movsfti.r.qw vr19, vr3, 1                         \n"

    " v.vmaxa.f32 vr12, vr12, vr28                          \n"
    " l.movsfti.r.qw vr20, vr4, 1                         \n"

    " v.vmaxa.f32 vr13, vr13, vr29                          \n"
    " l.movsfti.r.qw vr21, vr5, 1                         \n"

    " v.vmaxa.f32 vr14, vr14, vr30                          \n"
    " l.movsfti.r.qw vr22, vr6, 1                         \n"

    " v.vmaxa.f32 vr15, vr15, vr31                          \n"
    " l.movsfti.r.qw vr23, vr7, 1                         \n"

    // 第二次右移下半部分 + 第二次max 上半部分
    " v.vmaxa.f32 vr0, vr0, vr16                          \n"
    " l.movsfti.r.qw vr24, vr8, 1                         \n"

    " v.vmaxa.f32 vr1, vr1, vr17                          \n"
    " l.movsfti.r.qw vr25, vr9, 1                         \n"

    " v.vmaxa.f32 vr2, vr2, vr18                          \n"
    " l.movsfti.r.qw vr26, vr10, 1                         \n"

    " v.vmaxa.f32 vr3, vr3, vr19                          \n"
    " l.movsfti.r.qw vr27, vr11, 1                         \n"

    " v.vmaxa.f32 vr4, vr4, vr20                          \n"
    " l.movsfti.r.qw vr28, vr12, 1                         \n"

    " v.vmaxa.f32 vr5, vr5, vr21                          \n"
    " l.movsfti.r.qw vr29, vr13, 1                         \n"

    " v.vmaxa.f32 vr6, vr6, vr22                          \n"
    " l.movsfti.r.qw vr30, vr14, 1                         \n"

    " v.vmaxa.f32 vr7, vr7, vr23                          \n"
    " l.movsfti.r.qw vr31, vr15, 1                         \n"

    // 第三次右移上半部分 + 第二次max 下半部分
    " v.vmaxa.f32 vr8, vr8, vr24                          \n"
    " l.movsfti.r.b vr16, vr0, 8                       \n"

    " v.vmaxa.f32 vr9, vr9, vr25                          \n"
    " l.movsfti.r.b vr17, vr1, 8                       \n"

    " v.vmaxa.f32 vr10, vr10, vr26                          \n"
    " l.movsfti.r.b vr18, vr2, 8                       \n"

    " v.vmaxa.f32 vr11, vr11, vr27                          \n"
    " l.movsfti.r.b vr19, vr3, 8                       \n"

    " v.vmaxa.f32 vr12, vr12, vr28                          \n"
    " l.movsfti.r.b vr20, vr4, 8                       \n"

    " v.vmaxa.f32 vr13, vr13, vr29                          \n"
    " l.movsfti.r.b vr21, vr5, 8                       \n"

    " v.vmaxa.f32 vr14, vr14, vr30                          \n"
    " l.movsfti.r.b vr22, vr6, 8                       \n"

    " v.vmaxa.f32 vr15, vr15, vr31                          \n"
    " l.movsfti.r.b vr23, vr7, 8                       \n"

    // 第三次右移下半部分 + 第三次max 上半部分
    " v.vmaxa.f32 vr0, vr0, vr16                          \n"
    " l.movsfti.r.b vr24, vr8, 8                       \n"

    " v.vmaxa.f32 vr1, vr1, vr17                          \n"
    " l.movsfti.r.b vr25, vr9, 8                       \n"

    " v.vmaxa.f32 vr2, vr2, vr18                          \n"
    " l.movsfti.r.b vr26, vr10, 8                       \n"

    " v.vmaxa.f32 vr3, vr3, vr19                          \n"
    " l.movsfti.r.b vr27, vr11, 8                       \n"

    " v.vmaxa.f32 vr4, vr4, vr20                          \n"
    " l.movsfti.r.b vr28, vr12, 8                       \n"

    " v.vmaxa.f32 vr5, vr5, vr21                          \n"
    " l.movsfti.r.b vr29, vr13, 8                       \n"

    " v.vmaxa.f32 vr6, vr6, vr22                          \n"
    " l.movsfti.r.b vr30, vr14, 8                       \n"

    " v.vmaxa.f32 vr7, vr7, vr23                          \n"
    " l.movsfti.r.b vr31, vr15, 8                       \n"

    // 第四次右移上半部分 + 第三次max 下半部分
    " v.vmaxa.f32 vr8, vr8, vr24                          \n"
    " l.movsfti.r.b vr16, vr0, 4                       \n"

    " v.vmaxa.f32 vr9, vr9, vr25                          \n"
    " l.movsfti.r.b vr17, vr1, 4                       \n"

    " v.vmaxa.f32 vr10, vr10, vr26                          \n"
    " l.movsfti.r.b vr18, vr2, 4                       \n"

    " v.vmaxa.f32 vr11, vr11, vr27                          \n"
    " l.movsfti.r.b vr19, vr3, 4                       \n"

    " v.vmaxa.f32 vr12, vr12, vr28                          \n"
    " l.movsfti.r.b vr20, vr4, 4                       \n"

    " v.vmaxa.f32 vr13, vr13, vr29                          \n"
    " l.movsfti.r.b vr21, vr5, 4                       \n"

    " v.vmaxa.f32 vr14, vr14, vr30                          \n"
    " l.movsfti.r.b vr22, vr6, 4                       \n"

    " v.vmaxa.f32 vr15, vr15, vr31                          \n"
    " l.movsfti.r.b vr23, vr7, 4                       \n"

    // 第四次右移下半部分 + 第四次max 上半部分
    " v.vmaxa.f32 vr0, vr0, vr16                          \n"
    " l.movsfti.r.b vr24, vr8, 4                       \n"

    " v.vmaxa.f32 vr1, vr1, vr17                          \n"
    " l.movsfti.r.b vr25, vr9, 4                       \n"

    " v.vmaxa.f32 vr2, vr2, vr18                          \n"
    " l.movsfti.r.b vr26, vr10, 4                       \n"

    " v.vmaxa.f32 vr3, vr3, vr19                          \n"
    " l.movsfti.r.b vr27, vr11, 4                       \n"

    " v.vmaxa.f32 vr4, vr4, vr20                          \n"
    " l.movsfti.r.b vr28, vr12, 4                       \n"

    " v.vmaxa.f32 vr5, vr5, vr21                          \n"
    " l.movsfti.r.b vr29, vr13, 4                       \n"

    " v.vmaxa.f32 vr6, vr6, vr22                          \n"
    " l.movsfti.r.b vr30, vr14, 4                       \n"

    " v.vmaxa.f32 vr7, vr7, vr23                          \n"
    " l.movsfti.r.b vr31, vr15, 4                       \n"

    // 第五次右移上半部分 + 第四次max 下半部分
    " v.vmaxa.f32 vr8, vr8, vr24                          \n"
    " l.ldi16.s r8, 1               \n"
    " v.vmaxa.f32 vr9, vr9, vr25                          \n"
    " l.ldi16.s r9, 0               \n"
    " v.vmaxa.f32 vr10, vr10, vr26                        \n"
    " l.addia.s r10, %[cnt], -1     \n"
    " v.vmaxa.f32 vr11, vr11, vr27                        \n"

    " v.vmaxa.f32 vr12, vr12, vr28                        \n"
    " m.vmm2.mode18.f.nacc dacc2, vr0, smr0    \n"
    " l.vclr.qa qacc105              \n"

    " v.vmaxa.f32 vr13, vr13, vr29                        \n"
    " m.vmm2.mode18.f.nacc dacc4, vr1, smr0    \n"
    " l.vclr.qa qacc106              \n"

    " v.vmaxa.f32 vr14, vr14, vr30                        \n"
    " m.vmm2.mode18.f.nacc dacc6, vr2, smr0    \n"
    " l.vclr.qa qacc107              \n"

    " v.vmaxa.f32 vr15, vr15, vr31                        \n"
    " m.vmm2.mode18.f.nacc dacc8, vr3, smr0    \n"
    " l.vclr.qa qacc108              \n"

    " m.vmm2.mode18.f.nacc dacc10, vr4, smr0    \n"
    " l.vclr.qa qacc109              \n"
    " m.vmm2.mode18.f.nacc dacc12, vr5, smr0    \n"
    " l.vclr.qa qacc110              \n"
    " m.vmm2.mode18.f.nacc dacc14, vr6, smr0    \n"
    " l.vclr.qa qacc111              \n"
    " c.movsr2spr VPR, r8         \n"
    " m.vmm2.mode18.f.nacc dacc16, vr7, smr0    \n"
    " l.vclr.qa qacc112              \n"
    " c.movsr2spr LPR, r9         \n"

    " m.vmm2.mode18.f.nacc dacc18, vr8, smr0    \n"
    " l.vclr.qa qacc113              \n"
    " m.vmm2.mode18.f.nacc dacc20, vr9, smr0    \n"
    " l.vclr.qa qacc114              \n"
    " m.vmm2.mode18.f.nacc dacc22, vr10, smr0    \n"
    " l.vclr.qa qacc115              \n"
    " m.vmm2.mode18.f.nacc dacc24, vr11, smr0    \n"
    " l.vclr.qa qacc116              \n"
    " m.vmm2.mode18.f.nacc dacc26, vr12, smr0    \n"
    " l.vclr.qa qacc117              \n"
    " m.vmm2.mode18.f.nacc dacc28, vr13, smr0    \n"
    " l.vclr.qa qacc118              \n"
    " m.vmm2.mode18.f.nacc dacc30, vr14, smr0    \n"
    " l.vclr.qa qacc119              \n"

    " v.tvsta.w vacc4, [ta_g2, 1]  \n"
    " m.vmm2.mode18.f.nacc dacc32, vr15, smr0    \n"
    " l.vclr.qa qacc120              \n"

    " v.tvsta.w vacc8, [ta_g2, 1]    \n"
    " m.tvldqa.w.qa qacc18, [ta_g0, 1]   \n"  //  2
    " l.movda dacc3, dacc2    \n"  // qa1

    " v.tvsta.w vacc12, [ta_g2, 1]  \n"
    " m.tvldqa.w.qa qacc19, [ta_g0, 1]   \n"  //  3
    " l.movda dacc5, dacc4    \n"

    " v.tvsta.w vacc16, [ta_g2, 1]  \n"
    " m.tvldqa.w.qa qacc20, [ta_g0, 1]   \n"  //  0
    " l.movda dacc7, dacc6    \n"

    " v.tvsta.w vacc20, [ta_g2, 1]  \n"
    " m.tvldqa.w.qa qacc21, [ta_g0, 1]   \n"  //  1
    " l.movda dacc9, dacc8    \n"

    " v.tvsta.w vacc24, [ta_g2, 1]  \n"
    " m.tvldqa.w.qa qacc22, [ta_g0, 1]   \n"  //  2
    " l.movda dacc11, dacc10    \n"

    " v.tvsta.w vacc28, [ta_g2, 1]  \n"
    " m.tvldqa.w.qa qacc23, [ta_g0, 1]   \n"  //  3
    " l.movda dacc13, dacc12    \n"

    " v.tvsta.w vacc32, [ta_g2, 1]  \n"
    " m.tvldqa.w.qa qacc24, [ta_g0, 1]   \n"  //  0
    " l.movda dacc15, dacc14    \n"

    " v.tvsta.w vacc36, [ta_g2, 1]  \n"
    " m.tvldqa.w.qa qacc25, [ta_g0, 1]   \n"  //  1
    " l.movda dacc17, dacc16    \n"

    " v.tvsta.w vacc40, [ta_g2, 1]  \n"
    " m.tvldqa.w.qa qacc26, [ta_g0, 1]   \n"  //  2
    " l.movda dacc19, dacc18    \n"

    " v.tvsta.w vacc44, [ta_g2, 1]  \n"
    " m.tvldqa.w.qa qacc27, [ta_g0, 1]   \n"  //  3
    " l.movda dacc21, dacc20    \n"

    " v.tvsta.w vacc48, [ta_g2, 1]  \n"
    " m.tvldqa.w.qa qacc28, [ta_g0, 1]   \n"  //  0
    " l.movda dacc23, dacc22    \n"

    " v.tvsta.w vacc52, [ta_g2, 1]  \n"
    " m.tvldqa.w.qa qacc29, [ta_g0, 1]   \n"  //  1
    " l.movda dacc25, dacc24    \n"

    " v.tvsta.w vacc56, [ta_g2, 1]  \n"
    " m.tvldqa.w.qa qacc30, [ta_g0, 1]   \n"  //  2
    " l.movda dacc27, dacc26    \n"

    " v.tvsta.w vacc60, [ta_g2, 1]  \n"
    " m.tvldqa.w.qa qacc31, [ta_g0, 1]   \n"  //  3
    " l.movda dacc29, dacc28    \n"

    " v.tvsta.w vacc64, [ta_g2, 2]  \n"
    " m.tvldqa.w.qa qacc32, [ta_g0, 1]   \n"  //  0
    " l.movda dacc31, dacc30    \n"

    " m.tvldqa.w.qa qacc33, [ta_g0, 2]   \n"  //  1
    " l.movda dacc33, dacc32    \n"

    " l.vldi_hi.qa qacc37, 16312                          \n"
    " l.vldi_lo.qa qacc37, 43579                          \n"

    // " l.vldi_hi.qa qacc38, 16312                          \n"
    // " l.vldi_lo.qa qacc38, 43579                          \n"

    " l.vldi_hi.qa qacc39, 16312                          \n"
    " l.vldi_lo.qa qacc39, 43579                          \n"
    // " l.vldi_hi.qa qacc36, 16312     \n"
    // " l.vldi_lo.qa qacc36, 43579     \n"


    "11: \n"
      " if (vpr) v.tvstda.w.dual dacc144, [ta_g3, 1]   \n" //      0
      " m.mop.msub.f32.qa qacc18, qacc18, qacc1     \n"   //  2 1
      " l.addia.s r10, r10, -1 \n"
      " if (vpr) v.tvstda.w.dual dacc145, [ta_g3, 3]   \n" //      0
      " m.mop.msub.f32.qa qacc19, qacc19, qacc2     \n"   //  3 2

      " if (vpr) v.tvstda.w.dual dacc148, [ta_g3, 1]   \n" //      2
      " m.mop.msub.f32.qa qacc20, qacc20, qacc3     \n"   //  0 3
      " if (vpr) v.tvstda.w.dual dacc149, [ta_g3, 3]   \n" //      2
      " m.mop.msub.f32.qa qacc21, qacc21, qacc4     \n"   //  1 0

      " if (vpr) v.tvstda.w.dual dacc152, [ta_g3, 1]   \n" //      0
      " m.mop.msub.f32.qa qacc22, qacc22, qacc5     \n"   //  2 1
      " if (vpr) v.tvstda.w.dual dacc153, [ta_g3, 3]   \n" //      0
      " m.mop.msub.f32.qa qacc23, qacc23, qacc6     \n"   //  3 2

      " if (vpr) v.tvstda.w.dual dacc156, [ta_g3, 1]   \n" //      2
      " m.mop.msub.f32.qa qacc24, qacc24, qacc7     \n"   //  0 3
      " if (vpr) v.tvstda.w.dual dacc157, [ta_g3, 3]   \n" //      2
      " m.mop.msub.f32.qa qacc25, qacc25, qacc8     \n"   //  1 0

      " if (vpr) v.tvstda.w.dual dacc160, [ta_g3, 1]   \n"
      " m.mop.msub.f32.qa qacc26, qacc26, qacc9     \n"
      " if (vpr) v.tvstda.w.dual dacc161, [ta_g3, 3]   \n"
      " m.mop.msub.f32.qa qacc27, qacc27, qacc10     \n"

      " if (vpr) v.tvstda.w.dual dacc164, [ta_g3, 1]   \n"
      " m.mop.msub.f32.qa qacc28, qacc28, qacc11     \n"
      " if (vpr) v.tvstda.w.dual dacc165, [ta_g3, 3]   \n"
      " m.mop.msub.f32.qa qacc29, qacc29, qacc12     \n"

      " if (vpr) v.tvstda.w.dual dacc168, [ta_g3, 1]   \n"
      " m.mop.msub.f32.qa qacc30, qacc30, qacc13     \n"
      " if (vpr) v.tvstda.w.dual dacc169, [ta_g3, 3]   \n"
      " m.mop.msub.f32.qa qacc31, qacc31, qacc14     \n"

      " if (vpr) v.tvstda.w.dual dacc172, [ta_g3, 1]   \n"
      " m.mop.msub.f32.qa qacc32, qacc32, qacc15     \n"
      " if (vpr) v.tvstda.w.dual dacc173, [ta_g3, 3]   \n"
      " m.mop.msub.f32.qa qacc33, qacc33, qacc16     \n"
      " l.srlia r9, r10, 31   \n"

      " if (vpr) v.tvstda.w.dual dacc176, [ta_g3, 1]   \n"  //  0
      " m.mop.mdm.f32.qa qacc40, qacc18, qacc37     \n"  // 2  1
      " if (vpr) v.tvstda.w.dual dacc177, [ta_g3, 3]   \n"  //  0
      " m.mop.mdm.f32.qa qacc42, qacc19, qacc37     \n"  // 3  1

      " if (vpr) v.tvstda.w.dual dacc180, [ta_g3, 1]   \n"  //  2
      " m.mop.mdm.f32.qa qacc44, qacc20, qacc39     \n"  // 0  3
      " if (vpr) v.tvstda.w.dual dacc181, [ta_g3, 3]   \n"  //  2
      " m.mop.mdm.f32.qa qacc46, qacc21, qacc39     \n"  // 1  3

      " if (vpr) v.tvstda.w.dual dacc184, [ta_g3, 1]   \n"  //  0
      " m.mop.mdm.f32.qa qacc48, qacc22, qacc37     \n"  // 2  1
      " if (vpr) v.tvstda.w.dual dacc185, [ta_g3, 3]   \n"  //  0
      " m.mop.mdm.f32.qa qacc50, qacc23, qacc37     \n"  // 3  1

      " if (vpr) v.tvstda.w.dual dacc188, [ta_g3, 1]   \n"  //  2
      " m.mop.mdm.f32.qa qacc52, qacc24, qacc39     \n"  // 0  3
      " if (vpr) v.tvstda.w.dual dacc189, [ta_g3, 3]   \n"  //  2
      " m.mop.mdm.f32.qa qacc54, qacc25, qacc39     \n"  // 1  3

      " if (vpr) v.tvstda.w.dual dacc192, [ta_g3, 1]   \n"
      " m.mop.mdm.f32.qa qacc56, qacc26, qacc37     \n"  // 2  1
      " if (vpr) v.tvstda.w.dual dacc193, [ta_g3, 3]   \n"
      " m.mop.mdm.f32.qa qacc58, qacc27, qacc37     \n"  // 3  1

      " if (vpr) v.tvstda.w.dual dacc196, [ta_g3, 1]   \n"
      " m.mop.mdm.f32.qa qacc60, qacc28, qacc39     \n"  // 0  3
      " if (vpr) v.tvstda.w.dual dacc197, [ta_g3, 3]   \n"
      " m.mop.mdm.f32.qa qacc62, qacc29, qacc39     \n"  // 1  3

      " if (vpr) v.tvstda.w.dual dacc200, [ta_g3, 1]   \n"
      " m.mop.mdm.f32.qa qacc64, qacc30, qacc37     \n"  // 2  1
      " if (vpr) v.tvstda.w.dual dacc201, [ta_g3, 3]   \n"
      " m.mop.mdm.f32.qa qacc66, qacc31, qacc37     \n"  // 3  1

      " if (vpr) v.tvstda.w.dual dacc204, [ta_g3, 1]   \n"
      " m.mop.mdm.f32.qa qacc68, qacc32, qacc39     \n"  // 0  3
      " if (vpr) v.tvstda.w.dual dacc205, [ta_g3, 2]   \n"
      " m.mop.mdm.f32.qa qacc70, qacc33, qacc39     \n"  // 1  3
      " c.movsr2spr LPR, r9 \n"

      //  0
      " if (lpr) l.tvldqa.w.qa qacc18, [ta_g0, 1]  \n"
      " m.msf.mode0 vacc288, vacc160     \n"  // qa72
      " m.msf.mode0 vacc289, vacc161     \n"
      " m.msf.mode0 vacc290, vacc162     \n"
      " m.msf.mode0 vacc291, vacc163     \n"

      //  1
      " if (lpr) l.tvldqa.w.qa qacc19, [ta_g0, 1]  \n"
      " m.msf.mode0 vacc296, vacc168     \n"
      " m.msf.mode0 vacc297, vacc169     \n"
      " m.msf.mode0 vacc298, vacc170     \n"
      " m.msf.mode0 vacc299, vacc171     \n"

      //  2
      " if (lpr) l.tvldqa.w.qa qacc20, [ta_g0, 1]  \n"
      " m.msf.mode0 vacc304, vacc176     \n"
      " m.msf.mode0 vacc305, vacc177     \n"
      " m.msf.mode0 vacc306, vacc178     \n"
      " m.msf.mode0 vacc307, vacc179     \n"

      //  3
      " if (lpr) l.tvldqa.w.qa qacc21, [ta_g0, 1]  \n"
      " m.msf.mode0 vacc312, vacc184     \n"
      " m.msf.mode0 vacc313, vacc185     \n"
      " m.msf.mode0 vacc314, vacc186     \n"
      " m.msf.mode0 vacc315, vacc187     \n"

      //  4
      " if (lpr) l.tvldqa.w.qa qacc22, [ta_g0, 1]  \n"
      " m.msf.mode0 vacc320, vacc192     \n"
      " m.msf.mode0 vacc321, vacc193     \n"
      " m.msf.mode0 vacc322, vacc194     \n"
      " m.msf.mode0 vacc323, vacc195     \n"

      //  5
      " if (lpr) l.tvldqa.w.qa qacc23, [ta_g0, 1]  \n"
      " m.msf.mode0 vacc328, vacc200     \n"
      " m.msf.mode0 vacc329, vacc201     \n"
      " m.msf.mode0 vacc330, vacc202     \n"
      " m.msf.mode0 vacc331, vacc203     \n"

      //  6
      " if (lpr) l.tvldqa.w.qa qacc24, [ta_g0, 1]  \n"
      " m.msf.mode0 vacc336, vacc208     \n"
      " m.msf.mode0 vacc337, vacc209     \n"
      " m.msf.mode0 vacc338, vacc210     \n"
      " m.msf.mode0 vacc339, vacc211     \n"

      //  7
      " if (lpr) l.tvldqa.w.qa qacc25, [ta_g0, 1]  \n"
      " m.msf.mode0 vacc344, vacc216     \n"
      " m.msf.mode0 vacc345, vacc217     \n"
      " m.msf.mode0 vacc346, vacc218     \n"
      " m.msf.mode0 vacc347, vacc219     \n"

      //  8
      " if (lpr) l.tvldqa.w.qa qacc26, [ta_g0, 1]  \n"
      " m.msf.mode0 vacc352, vacc224     \n"  // qa88
      " m.msf.mode0 vacc353, vacc225     \n"
      " m.msf.mode0 vacc354, vacc226     \n"
      " m.msf.mode0 vacc355, vacc227     \n"

      //  9
      " if (lpr) l.tvldqa.w.qa qacc27, [ta_g0, 1]  \n"
      " m.msf.mode0 vacc360, vacc232     \n"
      " m.msf.mode0 vacc361, vacc233     \n"
      " m.msf.mode0 vacc362, vacc234     \n"
      " m.msf.mode0 vacc363, vacc235     \n"

      //  10
      " if (lpr) l.tvldqa.w.qa qacc28, [ta_g0, 1]  \n"
      " m.msf.mode0 vacc368, vacc240     \n"
      " m.msf.mode0 vacc369, vacc241     \n"
      " m.msf.mode0 vacc370, vacc242     \n"
      " m.msf.mode0 vacc371, vacc243     \n"

      //  11
      " if (lpr) l.tvldqa.w.qa qacc29, [ta_g0, 1]  \n"
      " m.msf.mode0 vacc376, vacc248     \n"
      " m.msf.mode0 vacc377, vacc249     \n"
      " m.msf.mode0 vacc378, vacc250     \n"
      " m.msf.mode0 vacc379, vacc251     \n"

      //  12
      " if (lpr) l.tvldqa.w.qa qacc30, [ta_g0, 1]  \n"
      " m.msf.mode0 vacc384, vacc256     \n"
      " m.msf.mode0 vacc385, vacc257     \n"
      " m.msf.mode0 vacc386, vacc258     \n"
      " m.msf.mode0 vacc387, vacc259     \n"

      //  13
      " if (lpr) l.tvldqa.w.qa qacc31, [ta_g0, 1]  \n"
      " m.msf.mode0 vacc392, vacc264     \n"
      " m.msf.mode0 vacc393, vacc265     \n"
      " m.msf.mode0 vacc394, vacc266     \n"
      " m.msf.mode0 vacc395, vacc267     \n"

      //  14
      " if (lpr) l.tvldqa.w.qa qacc32, [ta_g0, 1]  \n"
      " m.msf.mode0 vacc400, vacc272     \n"
      " m.msf.mode0 vacc401, vacc273     \n"
      " m.msf.mode0 vacc402, vacc274     \n"
      " m.msf.mode0 vacc403, vacc275     \n"

      //  15
      " if (lpr) l.tvldqa.w.qa qacc33, [ta_g0, 2]  \n"
      " m.msf.mode0 vacc408, vacc280     \n"
      " m.msf.mode0 vacc409, vacc281     \n"
      " m.msf.mode0 vacc410, vacc282     \n"
      " m.msf.mode0 vacc411, vacc283     \n"

      " l.ldi16.s r8, 0       \n"
      " c.movsr2spr VPR, r8   \n"

      " m.mop.madd.f32.qa qacc105, qacc105, qacc72     \n"
      " l.addia.s r11, r10, 1 \n"
      " m.mop.madd.f32.qa qacc106, qacc106, qacc74     \n"
      " m.mop.madd.f32.qa qacc107, qacc107, qacc76     \n"
      " m.mop.madd.f32.qa qacc108, qacc108, qacc78     \n"
      " m.mop.madd.f32.qa qacc109, qacc109, qacc80     \n"
      " m.mop.madd.f32.qa qacc110, qacc110, qacc82     \n"
      " m.mop.madd.f32.qa qacc111, qacc111, qacc84     \n"
      " m.mop.madd.f32.qa qacc112, qacc112, qacc86     \n"
      " m.mop.madd.f32.qa qacc113, qacc113, qacc88     \n"
      " m.mop.madd.f32.qa qacc114, qacc114, qacc90     \n"
      " m.mop.madd.f32.qa qacc115, qacc115, qacc92     \n"
      " m.mop.madd.f32.qa qacc116, qacc116, qacc94     \n"
      " m.mop.madd.f32.qa qacc117, qacc117, qacc96     \n"
      " m.mop.madd.f32.qa qacc118, qacc118, qacc98     \n"
      " m.mop.madd.f32.qa qacc119, qacc119, qacc100     \n"
      " m.mop.madd.f32.qa qacc120, qacc120, qacc102     \n"

      " l.bne r0, r11, 11b   \n"
      :
      : [ cnt ] "r" (loop_times)
      : "r8", "r9", "r10", "r11"
    );

    // add reduce
    asm volatile(
      " v.tvstda.w.dual dacc144, [ta_g3, 1]   \n" //      0
      " m.mop.madd.f32.da dacc210, dacc210, dacc211     \n"
      " v.tvstda.w.dual dacc145, [ta_g3, 3]   \n" //      0
      " m.mop.madd.f32.da dacc212, dacc212, dacc213     \n"

      " v.tvstda.w.dual dacc148, [ta_g3, 1]   \n" //      1
      " m.mop.madd.f32.da dacc214, dacc214, dacc215     \n"
      " v.tvstda.w.dual dacc149, [ta_g3, 3]   \n" //      1
      " m.mop.madd.f32.da dacc216, dacc216, dacc217     \n"

      " v.tvstda.w.dual dacc152, [ta_g3, 1]   \n" //      2
      " m.mop.madd.f32.da dacc218, dacc218, dacc219     \n"
      " v.tvstda.w.dual dacc153, [ta_g3, 3]   \n" //      2
      " m.mop.madd.f32.da dacc220, dacc220, dacc221     \n"

      " v.tvstda.w.dual dacc156, [ta_g3, 1]   \n" //      3
      " m.mop.madd.f32.da dacc222, dacc222, dacc223     \n"
      " v.tvstda.w.dual dacc157, [ta_g3, 3]   \n" //      3
      " m.mop.madd.f32.da dacc224, dacc224, dacc225     \n"

      " v.tvstda.w.dual dacc160, [ta_g3, 1]   \n" //      0
      " m.mop.madd.f32.da dacc226, dacc226, dacc227     \n"
      " v.tvstda.w.dual dacc161, [ta_g3, 3]   \n" //      0
      " m.mop.madd.f32.da dacc228, dacc228, dacc229     \n"

      " v.tvstda.w.dual dacc164, [ta_g3, 1]   \n" //      1
      " m.mop.madd.f32.da dacc230, dacc230, dacc231     \n"
      " v.tvstda.w.dual dacc165, [ta_g3, 3]   \n" //      1
      " m.mop.madd.f32.da dacc232, dacc232, dacc233     \n"

      " v.tvstda.w.dual dacc168, [ta_g3, 1]   \n" //      2
      " m.mop.madd.f32.da dacc234, dacc234, dacc235     \n"
      " v.tvstda.w.dual dacc169, [ta_g3, 3]   \n" //      2
      " m.mop.madd.f32.da dacc236, dacc236, dacc237     \n"

      " v.tvstda.w.dual dacc172, [ta_g3, 1]   \n" //      3
      " m.mop.madd.f32.da dacc238, dacc238, dacc239     \n"
      " v.tvstda.w.dual dacc173, [ta_g3, 3]   \n" //      3
      " m.mop.madd.f32.da dacc240, dacc240, dacc241     \n"

      " v.tvstda.w.dual dacc176, [ta_g3, 1]   \n" //      0
      " m.mop.madd.f32.va vacc420, vacc420, vacc421         \n"
      " v.tvstda.w.dual dacc177, [ta_g3, 3]   \n" //      0
      " m.mop.madd.f32.va vacc424, vacc424, vacc425         \n"

      " v.tvstda.w.dual dacc180, [ta_g3, 1]   \n" //      1
      " m.mop.madd.f32.va vacc428, vacc428, vacc429         \n"
      " v.tvstda.w.dual dacc181, [ta_g3, 3]   \n" //      1
      " m.mop.madd.f32.va vacc432, vacc432, vacc433         \n"

      " v.tvstda.w.dual dacc184, [ta_g3, 1]   \n" //      2
      " m.mop.madd.f32.va vacc436, vacc436, vacc437         \n"
      " v.tvstda.w.dual dacc185, [ta_g3, 3]   \n" //      2
      " m.mop.madd.f32.va vacc440, vacc440, vacc441         \n"

      " v.tvstda.w.dual dacc188, [ta_g3, 1]   \n" //      3
      " m.mop.madd.f32.va vacc444, vacc444, vacc445         \n"
      " v.tvstda.w.dual dacc189, [ta_g3, 3]   \n" //      3
      " m.mop.madd.f32.va vacc448, vacc448, vacc449         \n"

      " v.tvstda.w.dual dacc192, [ta_g3, 1]   \n" //      0
      " m.mop.madd.f32.va vacc452, vacc452, vacc453         \n"
      " v.tvstda.w.dual dacc193, [ta_g3, 3]   \n" //      0
      " m.mop.madd.f32.va vacc456, vacc456, vacc457         \n"

      " v.tvstda.w.dual dacc196, [ta_g3, 1]   \n" //      1
      " m.mop.madd.f32.va vacc460, vacc460, vacc461         \n"
      " v.tvstda.w.dual dacc197, [ta_g3, 3]   \n" //      1
      " m.mop.madd.f32.va vacc464, vacc464, vacc465         \n"

      " v.tvstda.w.dual dacc200, [ta_g3, 1]   \n" //      2
      " m.mop.madd.f32.va vacc468, vacc468, vacc469         \n"
      " v.tvstda.w.dual dacc201, [ta_g3, 3]   \n" //      2
      " m.mop.madd.f32.va vacc472, vacc472, vacc473         \n"

      " v.tvstda.w.dual dacc204, [ta_g3, 1]   \n" //      3
      " m.mop.madd.f32.va vacc476, vacc476, vacc477         \n"
      " v.tvstda.w.dual dacc205, [ta_g3, 3]   \n" //      3
      " m.mop.madd.f32.va vacc480, vacc480, vacc481         \n"

      " l.movva2vr.w vr0, vacc420                           \n"
      " l.movva2vr.w vr1, vacc424                           \n"
      " l.movva2vr.w vr2, vacc428                           \n"
      " l.movva2vr.w vr3, vacc432                           \n"
      " l.movva2vr.w vr4, vacc436                           \n"
      " l.movva2vr.w vr5, vacc440                           \n"
      " l.movva2vr.w vr6, vacc444                           \n"
      " l.movva2vr.w vr7, vacc448                           \n"
      " l.movva2vr.w vr8, vacc452                           \n"
      " l.movva2vr.w vr9, vacc456                           \n"
      " l.movva2vr.w vr10, vacc460                          \n"
      " l.movva2vr.w vr11, vacc464                          \n"
      " l.movva2vr.w vr12, vacc468                          \n"
      " l.movva2vr.w vr13, vacc472                          \n"
      " l.movva2vr.w vr14, vacc476                          \n"
      " l.movva2vr.w vr15, vacc480                          \n"

      // smr1
      " m.vmm2.mode18.f.nacc dacc1, vr0, smr1    \n"
      " m.vmm2.mode18.f.nacc dacc2, vr1, smr1    \n"
      " m.vmm2.mode18.f.nacc dacc3, vr2, smr1    \n"
      " m.vmm2.mode18.f.nacc dacc4, vr3, smr1    \n"
      " m.vmm2.mode18.f.nacc dacc5, vr4, smr1    \n"
      " m.vmm2.mode18.f.nacc dacc6, vr5, smr1    \n"
      " m.vmm2.mode18.f.nacc dacc7, vr6, smr1    \n"
      " m.vmm2.mode18.f.nacc dacc8, vr7, smr1    \n"

      " m.vmm2.mode18.f.nacc dacc9, vr8, smr1    \n"
      " m.vmm2.mode18.f.nacc dacc10, vr9, smr1    \n"
      " m.vmm2.mode18.f.nacc dacc11, vr10, smr1    \n"
      " m.vmm2.mode18.f.nacc dacc12, vr11, smr1    \n"
      " m.vmm2.mode18.f.nacc dacc13, vr12, smr1    \n"
      " m.vmm2.mode18.f.nacc dacc14, vr13, smr1    \n"
      " m.vmm2.mode18.f.nacc dacc15, vr14, smr1    \n"
      " m.vmm2.mode18.f.nacc dacc16, vr15, smr1    \n"

      " v.tvsta.w vacc2, [ta_g2, 1]  \n"
      " v.tvsta.w vacc4, [ta_g2, 1]  \n"
      " v.tvsta.w vacc6, [ta_g2, 1]  \n"
      " v.tvsta.w vacc8, [ta_g2, 1]  \n"
      " v.tvsta.w vacc10, [ta_g2, 1]  \n"
      " v.tvsta.w vacc12, [ta_g2, 1]  \n"
      " v.tvsta.w vacc14, [ta_g2, 1]  \n"
      " v.tvsta.w vacc16, [ta_g2, 1]  \n"
      " v.tvsta.w vacc18, [ta_g2, 1]  \n"
      " v.tvsta.w vacc20, [ta_g2, 1]  \n"
      " v.tvsta.w vacc22, [ta_g2, 1]  \n"
      " v.tvsta.w vacc24, [ta_g2, 1]  \n"
      " v.tvsta.w vacc26, [ta_g2, 1]  \n"
      " v.tvsta.w vacc28, [ta_g2, 1]  \n"
      " v.tvsta.w vacc30, [ta_g2, 1]  \n"
      " v.tvsta.w vacc32, [ta_g2, 1]  \n"
    );
}

// template <>
__device__ void __attribute__((dtu_maxinum_vacc(192))) call_k_dot_kn(
    float* out, float* lhs, tops::half* rhs,
    int M, int K, int N, int left_ctx_len, int f32_ind) {
  int out_addr = reinterpret_cast<long>(out);
  int lhs_addr = reinterpret_cast<long>(lhs);
  int rhs_addr = reinterpret_cast<long>(rhs);

  int k = left_ctx_len;
  auto vr_0 = __dtu_s_movr2vr_dup(0);
  if (K > k) {
    int loop_times = ((K - k) * N) >> 6;
    char* st_addr = reinterpret_cast<char*>(rhs + k * N);
    for (int i = 0; i < loop_times; i++) {
      __dtu_l_vstl(vr_0, st_addr + 128 * i, 0);
    }
  }

  int naccovr = 0x10001;

  auto k_unit = K >> 5;
  auto n_unit = N >> 6;

  lhs_addr = lhs_addr >> 7;
  rhs_addr = rhs_addr >> 7;
  out_addr = out_addr >> 7;

  tar_t lt_base = __dtu_c_movsr2targ(TAR32(lhs_addr, lhs_addr));
  int offset = TAR32(k_unit, k_unit);
  tar_t lt_off0 = __dtu_c_movsr2tari(offset, lt_base);
  offset = TAR32(1 - 15 * k_unit, 1 - 15 * k_unit);
  tar_t lt_off1 = __dtu_c_movsr2tari(offset, lt_base);

  tar_t rt_base = __dtu_c_movsr2targ(TAR32(rhs_addr, rhs_addr + 1));
  offset = TAR32(n_unit, n_unit);
  tar_t rt_off0 = __dtu_c_movsr2tari(offset, rt_base);
  offset = TAR32(2 - (K) * n_unit, 2 - (K) * n_unit);
  tar_t rt_off1 = __dtu_c_movsr2tari(offset, rt_base);

  int vpr = (M > 8) ? 0 : 1;
  __dtu_c_movsr2vpr(vpr);

  int vab_shift = 0;
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_d(0);
  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_lv_d(0);
  __dtu_c_movsr2vab_m_s2(0);

  smr_t smr0, smr1;
  va16f32x4 qacc0, qacc1, qacc2, qacc3, qacc4, qacc5, qacc6, qacc7,
      qacc8, qacc9, qacc10, qacc11, qacc12, qacc13, qacc14, qacc15;

#pragma clang loop unroll(disable)
  for (int n = 0; n < N; n += 128) {
    __dtu_c_movsr2naccovr(naccovr);

    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 0);
    auto da0 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 1);
    auto da1 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 2);
    auto da2 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 3);
    auto da3 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 4);
    auto da4 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 5);
    auto da5 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 6);
    auto da6 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 7);
    auto da7 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 8);
    auto da8 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 9);
    auto da9 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 10);
    auto da10 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 11);
    auto da11 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 12);
    auto da12 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 13);
    auto da13 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 14);
    auto da14 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 15);
    auto da15 = __dtu_l_tvldqa_f32_da(lt_base, lt_off1);

    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 16);
    auto vr0 = __dtu_l_movva2vr_cvt2fp16(da0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 17);
    auto vr1 = __dtu_l_movva2vr_cvt2fp16(da1);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 18);
    auto vr2 = __dtu_l_movva2vr_cvt2fp16(da2);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 19);
    auto vr3 = __dtu_l_movva2vr_cvt2fp16(da3);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 20);
    auto vr4 = __dtu_l_movva2vr_cvt2fp16(da4);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 21);
    auto vr5 = __dtu_l_movva2vr_cvt2fp16(da5);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 22);
    auto vr6 = __dtu_l_movva2vr_cvt2fp16(da6);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 23);
    auto vr7 = __dtu_l_movva2vr_cvt2fp16(da7);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 24);
    auto vr8 = __dtu_l_movva2vr_cvt2fp16(da8);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 25);
    auto vr9 = __dtu_l_movva2vr_cvt2fp16(da9);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 26);
    auto vr10 = __dtu_l_movva2vr_cvt2fp16(da10);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 27);
    auto vr11 = __dtu_l_movva2vr_cvt2fp16(da11);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 28);
    auto vr12 = __dtu_l_movva2vr_cvt2fp16(da12);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 29);
    auto vr13 = __dtu_l_movva2vr_cvt2fp16(da13);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 30);
    auto vr14 = __dtu_l_movva2vr_cvt2fp16(da14);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 31);
    auto vr15 = __dtu_l_movva2vr_cvt2fp16(da15);


    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 0);
    auto da16 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 1);
    auto da17 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 2);
    auto da18 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 3);
    auto da19 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 4);
    auto da20 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 5);
    auto da21 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 6);
    auto da22 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 7);
    auto da23 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 8);
    auto da24 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 9);
    auto da25 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 10);
    auto da26 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 11);
    auto da27 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 12);
    auto da28 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 13);
    auto da29 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 14);
    auto da30 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 15);
    auto da31 = __dtu_l_tvldqa_f32_da(lt_base, lt_off1);

    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 16);
    auto vr16 = __dtu_l_movva2vr_cvt2fp16(da16);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 17);
    auto vr17 = __dtu_l_movva2vr_cvt2fp16(da17);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 18);
    auto vr18 = __dtu_l_movva2vr_cvt2fp16(da18);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 19);
    auto vr19 = __dtu_l_movva2vr_cvt2fp16(da19);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 20);
    auto vr20 = __dtu_l_movva2vr_cvt2fp16(da20);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 21);
    auto vr21 = __dtu_l_movva2vr_cvt2fp16(da21);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 22);
    auto vr22 = __dtu_l_movva2vr_cvt2fp16(da22);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 23);
    auto vr23 = __dtu_l_movva2vr_cvt2fp16(da23);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 24);
    auto vr24 = __dtu_l_movva2vr_cvt2fp16(da24);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 25);
    auto vr25 = __dtu_l_movva2vr_cvt2fp16(da25);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 26);
    auto vr26 = __dtu_l_movva2vr_cvt2fp16(da26);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 27);
    auto vr27 = __dtu_l_movva2vr_cvt2fp16(da27);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 28);
    auto vr28 = __dtu_l_movva2vr_cvt2fp16(da28);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 29);
    auto vr29 = __dtu_l_movva2vr_cvt2fp16(da29);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 30);
    auto vr30 = __dtu_l_movva2vr_cvt2fp16(da30);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 31);
    auto vr31 = __dtu_l_movva2vr_cvt2fp16(da31);

#pragma clang loop unroll(disable)
    for (int k = 0; k < K - 64; k += 64) {
      qacc0 = __dtu_m_vmm2_mode17_f16_nacc(qacc0, vr0, smr0);
      da0 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc1 = __dtu_m_vmm2_mode17_f16_nacc(qacc1, vr1, smr0);
      da1 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc2 = __dtu_m_vmm2_mode17_f16_nacc(qacc2, vr2, smr0);
      da2 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc3 = __dtu_m_vmm2_mode17_f16_nacc(qacc3, vr3, smr0);
      da3 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc4 = __dtu_m_vmm2_mode17_f16_nacc(qacc4, vr4, smr0);
      da4 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc5 = __dtu_m_vmm2_mode17_f16_nacc(qacc5, vr5, smr0);
      da5 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc6 = __dtu_m_vmm2_mode17_f16_nacc(qacc6, vr6, smr0);
      da6 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc7 = __dtu_m_vmm2_mode17_f16_nacc(qacc7, vr7, smr0);
      da7 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc8 = __dtu_m_vmm2_mode17_f16_nacc(qacc8, vr8, smr0);
      da8 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc9 = __dtu_m_vmm2_mode17_f16_nacc(qacc9, vr9, smr0);
      da9 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc10 = __dtu_m_vmm2_mode17_f16_nacc(qacc10, vr10, smr0);
      da10 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc11 = __dtu_m_vmm2_mode17_f16_nacc(qacc11, vr11, smr0);
      da11 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc12 = __dtu_m_vmm2_mode17_f16_nacc(qacc12, vr12, smr0);
      da12 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc13 = __dtu_m_vmm2_mode17_f16_nacc(qacc13, vr13, smr0);
      da13 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc14 = __dtu_m_vmm2_mode17_f16_nacc(qacc14, vr14, smr0);
      da14 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc15 = __dtu_m_vmm2_mode17_f16_nacc(qacc15, vr15, smr0);
      da15 = __dtu_l_tvldqa_f32_da(lt_base, lt_off1);

      __dtu_c_movsr2naccovr(0x1);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 0);
      qacc0 = __dtu_m_vmm2_mode17_f16_nacc(qacc0, vr16, smr1);
      da16 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 1);
      qacc1 = __dtu_m_vmm2_mode17_f16_nacc(qacc1, vr17, smr1);
      da17 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 2);
      qacc2 = __dtu_m_vmm2_mode17_f16_nacc(qacc2, vr18, smr1);
      da18 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 3);
      qacc3 = __dtu_m_vmm2_mode17_f16_nacc(qacc3, vr19, smr1);
      da19 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 4);
      qacc4 = __dtu_m_vmm2_mode17_f16_nacc(qacc4, vr20, smr1);
      da20 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 5);
      qacc5 = __dtu_m_vmm2_mode17_f16_nacc(qacc5, vr21, smr1);
      da21 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 6);
      qacc6 = __dtu_m_vmm2_mode17_f16_nacc(qacc6, vr22, smr1);
      da22 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 7);
      qacc7 = __dtu_m_vmm2_mode17_f16_nacc(qacc7, vr23, smr1);
      da23 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 8);
      qacc8 = __dtu_m_vmm2_mode17_f16_nacc(qacc8, vr24, smr1);
      da24 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 9);
      qacc9 = __dtu_m_vmm2_mode17_f16_nacc(qacc9, vr25, smr1);
      da25 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 10);
      qacc10 = __dtu_m_vmm2_mode17_f16_nacc(qacc10, vr26, smr1);
      da26 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 11);
      qacc11 = __dtu_m_vmm2_mode17_f16_nacc(qacc11, vr27, smr1);
      da27 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 12);
      qacc12 = __dtu_m_vmm2_mode17_f16_nacc(qacc12, vr28, smr1);
      da28 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 13);
      qacc13 = __dtu_m_vmm2_mode17_f16_nacc(qacc13, vr29, smr1);
      da29 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 14);
      qacc14 = __dtu_m_vmm2_mode17_f16_nacc(qacc14, vr30, smr1);
      da30 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 15);
      qacc15 = __dtu_m_vmm2_mode17_f16_nacc(qacc15, vr31, smr1);
      da31 = __dtu_l_tvldqa_f32_da(lt_base, lt_off1);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 16);
      vr0 = __dtu_l_movva2vr_cvt2fp16(da0);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 17);
      vr1 = __dtu_l_movva2vr_cvt2fp16(da1);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 18);
      vr2 = __dtu_l_movva2vr_cvt2fp16(da2);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 19);
      vr3 = __dtu_l_movva2vr_cvt2fp16(da3);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 20);
      vr4 = __dtu_l_movva2vr_cvt2fp16(da4);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 21);
      vr5 = __dtu_l_movva2vr_cvt2fp16(da5);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 22);
      vr6 = __dtu_l_movva2vr_cvt2fp16(da6);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 23);
      vr7 = __dtu_l_movva2vr_cvt2fp16(da7);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 24);
      vr8 = __dtu_l_movva2vr_cvt2fp16(da8);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 25);
      vr9 = __dtu_l_movva2vr_cvt2fp16(da9);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 26);
      vr10 = __dtu_l_movva2vr_cvt2fp16(da10);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 27);
      vr11 = __dtu_l_movva2vr_cvt2fp16(da11);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 28);
      vr12 = __dtu_l_movva2vr_cvt2fp16(da12);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 29);
      vr13 = __dtu_l_movva2vr_cvt2fp16(da13);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 30);
      vr14 = __dtu_l_movva2vr_cvt2fp16(da14);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr0, rt_base, rt_off0, 31);
      vr15 = __dtu_l_movva2vr_cvt2fp16(da15);

      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 0);
      vr16 = __dtu_l_movva2vr_cvt2fp16(da16);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 1);
      vr17 = __dtu_l_movva2vr_cvt2fp16(da17);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 2);
      vr18 = __dtu_l_movva2vr_cvt2fp16(da18);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 3);
      vr19 = __dtu_l_movva2vr_cvt2fp16(da19);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 4);
      vr20 = __dtu_l_movva2vr_cvt2fp16(da20);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 5);
      vr21 = __dtu_l_movva2vr_cvt2fp16(da21);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 6);
      vr22 = __dtu_l_movva2vr_cvt2fp16(da22);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 7);
      vr23 = __dtu_l_movva2vr_cvt2fp16(da23);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 8);
      vr24 = __dtu_l_movva2vr_cvt2fp16(da24);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 9);
      vr25 = __dtu_l_movva2vr_cvt2fp16(da25);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 10);
      vr26 = __dtu_l_movva2vr_cvt2fp16(da26);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 11);
      vr27 = __dtu_l_movva2vr_cvt2fp16(da27);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 12);
      vr28 = __dtu_l_movva2vr_cvt2fp16(da28);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 13);
      vr29 = __dtu_l_movva2vr_cvt2fp16(da29);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 14);
      vr30 = __dtu_l_movva2vr_cvt2fp16(da30);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_f16_row(smr1, rt_base, rt_off0, 15);
      vr31 = __dtu_l_movva2vr_cvt2fp16(da31);

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
    qacc1 = __dtu_m_vmm2_mode17_f16_nacc(qacc1, vr1, smr0);
    qacc2 = __dtu_m_vmm2_mode17_f16_nacc(qacc2, vr2, smr0);
    qacc3 = __dtu_m_vmm2_mode17_f16_nacc(qacc3, vr3, smr0);
    qacc4 = __dtu_m_vmm2_mode17_f16_nacc(qacc4, vr4, smr0);
    qacc5 = __dtu_m_vmm2_mode17_f16_nacc(qacc5, vr5, smr0);
    qacc6 = __dtu_m_vmm2_mode17_f16_nacc(qacc6, vr6, smr0);
    qacc7 = __dtu_m_vmm2_mode17_f16_nacc(qacc7, vr7, smr0);
    qacc8 = __dtu_m_vmm2_mode17_f16_nacc(qacc8, vr8, smr0);
    qacc9 = __dtu_m_vmm2_mode17_f16_nacc(qacc9, vr9, smr0);
    qacc10 = __dtu_m_vmm2_mode17_f16_nacc(qacc10, vr10, smr0);
    qacc11 = __dtu_m_vmm2_mode17_f16_nacc(qacc11, vr11, smr0);
    qacc12 = __dtu_m_vmm2_mode17_f16_nacc(qacc12, vr12, smr0);
    qacc13 = __dtu_m_vmm2_mode17_f16_nacc(qacc13, vr13, smr0);
    qacc14 = __dtu_m_vmm2_mode17_f16_nacc(qacc14, vr14, smr0);
    qacc15 = __dtu_m_vmm2_mode17_f16_nacc(qacc15, vr15, smr0);

    __dtu_c_movsr2naccovr(0x1);
    qacc0 = __dtu_m_vmm2_mode17_f16_nacc(qacc0, vr16, smr1);
    qacc1 = __dtu_m_vmm2_mode17_f16_nacc(qacc1, vr17, smr1);
    qacc2 = __dtu_m_vmm2_mode17_f16_nacc(qacc2, vr18, smr1);
    qacc3 = __dtu_m_vmm2_mode17_f16_nacc(qacc3, vr19, smr1);
    qacc4 = __dtu_m_vmm2_mode17_f16_nacc(qacc4, vr20, smr1);
    qacc5 = __dtu_m_vmm2_mode17_f16_nacc(qacc5, vr21, smr1);
    qacc6 = __dtu_m_vmm2_mode17_f16_nacc(qacc6, vr22, smr1);
    qacc7 = __dtu_m_vmm2_mode17_f16_nacc(qacc7, vr23, smr1);
    qacc8 = __dtu_m_vmm2_mode17_f16_nacc(qacc8, vr24, smr1);
    qacc9 = __dtu_m_vmm2_mode17_f16_nacc(qacc9, vr25, smr1);
    qacc10 = __dtu_m_vmm2_mode17_f16_nacc(qacc10, vr26, smr1);
    qacc11 = __dtu_m_vmm2_mode17_f16_nacc(qacc11, vr27, smr1);
    qacc12 = __dtu_m_vmm2_mode17_f16_nacc(qacc12, vr28, smr1);
    qacc13 = __dtu_m_vmm2_mode17_f16_nacc(qacc13, vr29, smr1);
    qacc14 = __dtu_m_vmm2_mode17_f16_nacc(qacc14, vr30, smr1);
    qacc15 = __dtu_m_vmm2_mode17_f16_nacc(qacc15, vr31, smr1);

    vab_shift += 192;
    __dtu_c_movsr2vab_m_s1(vab_shift);
    __dtu_c_movsr2vab_m_d(vab_shift);

    lt_base = __dtu_v_taradd(lt_base, lt_off1);
    rt_base = __dtu_v_taradd(rt_base, rt_off1);
  }

  if (f32_ind == 1) {
    vab_shift = 0;
    __dtu_c_movsr2vab_lv_s(0);
    int m = (M > 8) ? ALIGN_16(M) : M;

    tar_t ot_base = __dtu_c_movsr2targ(TAR32(out_addr, out_addr + 2));
    offset = TAR32(1, 1);
    tar_t ot_off0 = __dtu_c_movsr2tari(offset, ot_base);
    offset = TAR32((N >> 5) - 1, (N >> 5) - 1);
    tar_t ot_off1 = __dtu_c_movsr2tari(offset, ot_base);
    offset = TAR32(2 - m * (N >> 5), 2 - m * (N >> 5));
    tar_t ot_off2 = __dtu_c_movsr2tari(offset, ot_base);

    if (M == 4) {
#pragma clang loop unroll(disable)
      for (int n = 0; n < N; n += 128) {
        auto dacc0 = __dtu_extractqa2da(qacc0, 0);
        auto dacc1 = __dtu_extractqa2da(qacc0, 1);
        auto dacc2 = __dtu_extractqa2da(qacc1, 0);
        auto dacc3 = __dtu_extractqa2da(qacc1, 1);
        auto dacc4 = __dtu_extractqa2da(qacc2, 0);
        auto dacc5 = __dtu_extractqa2da(qacc2, 1);
        auto dacc6 = __dtu_extractqa2da(qacc3, 0);
        auto dacc7 = __dtu_extractqa2da(qacc3, 1);

        __dtu_v_tvstda_f32_dual(dacc0, ot_base, ot_off0);
        __dtu_v_tvstda_f32_dual(dacc1, ot_base, ot_off1);
        __dtu_v_tvstda_f32_dual(dacc2, ot_base, ot_off0);
        __dtu_v_tvstda_f32_dual(dacc3, ot_base, ot_off1);
        __dtu_v_tvstda_f32_dual(dacc4, ot_base, ot_off0);
        __dtu_v_tvstda_f32_dual(dacc5, ot_base, ot_off1);
        __dtu_v_tvstda_f32_dual(dacc6, ot_base, ot_off0);
        __dtu_v_tvstda_f32_dual(dacc7, ot_base, ot_off1);

        ot_base = __dtu_v_taradd(ot_base, ot_off2);

        vab_shift += 192;
        __dtu_c_movsr2vab_lv_s(vab_shift);
      }
    } else {
#pragma clang loop unroll(disable)
      for (int n = 0; n < N; n += 128) {
        auto dacc0 = __dtu_extractqa2da(qacc0, 0);
        auto dacc1 = __dtu_extractqa2da(qacc0, 1);
        auto dacc2 = __dtu_extractqa2da(qacc1, 0);
        auto dacc3 = __dtu_extractqa2da(qacc1, 1);
        auto dacc4 = __dtu_extractqa2da(qacc2, 0);
        auto dacc5 = __dtu_extractqa2da(qacc2, 1);
        auto dacc6 = __dtu_extractqa2da(qacc3, 0);
        auto dacc7 = __dtu_extractqa2da(qacc3, 1);
        auto dacc8 = __dtu_extractqa2da(qacc4, 0);
        auto dacc9 = __dtu_extractqa2da(qacc4, 1);
        auto dacc10 = __dtu_extractqa2da(qacc5, 0);
        auto dacc11 = __dtu_extractqa2da(qacc5, 1);
        auto dacc12 = __dtu_extractqa2da(qacc6, 0);
        auto dacc13 = __dtu_extractqa2da(qacc6, 1);
        auto dacc14 = __dtu_extractqa2da(qacc7, 0);
        auto dacc15 = __dtu_extractqa2da(qacc7, 1);
        auto dacc16 = __dtu_extractqa2da(qacc8, 0);
        auto dacc17 = __dtu_extractqa2da(qacc8, 1);
        auto dacc18 = __dtu_extractqa2da(qacc9, 0);
        auto dacc19 = __dtu_extractqa2da(qacc9, 1);
        auto dacc20 = __dtu_extractqa2da(qacc10, 0);
        auto dacc21 = __dtu_extractqa2da(qacc10, 1);
        auto dacc22 = __dtu_extractqa2da(qacc11, 0);
        auto dacc23 = __dtu_extractqa2da(qacc11, 1);
        auto dacc24 = __dtu_extractqa2da(qacc12, 0);
        auto dacc25 = __dtu_extractqa2da(qacc12, 1);
        auto dacc26 = __dtu_extractqa2da(qacc13, 0);
        auto dacc27 = __dtu_extractqa2da(qacc13, 1);
        auto dacc28 = __dtu_extractqa2da(qacc14, 0);
        auto dacc29 = __dtu_extractqa2da(qacc14, 1);
        auto dacc30 = __dtu_extractqa2da(qacc15, 0);
        auto dacc31 = __dtu_extractqa2da(qacc15, 1);

        __dtu_v_tvstda_f32_dual(dacc0, ot_base, ot_off0);
        __dtu_v_tvstda_f32_dual(dacc1, ot_base, ot_off1);
        __dtu_v_tvstda_f32_dual(dacc2, ot_base, ot_off0);
        __dtu_v_tvstda_f32_dual(dacc3, ot_base, ot_off1);
        __dtu_v_tvstda_f32_dual(dacc4, ot_base, ot_off0);
        __dtu_v_tvstda_f32_dual(dacc5, ot_base, ot_off1);
        __dtu_v_tvstda_f32_dual(dacc6, ot_base, ot_off0);
        __dtu_v_tvstda_f32_dual(dacc7, ot_base, ot_off1);

        __dtu_v_tvstda_f32_dual(dacc8, ot_base, ot_off0);
        __dtu_v_tvstda_f32_dual(dacc9, ot_base, ot_off1);
        __dtu_v_tvstda_f32_dual(dacc10, ot_base, ot_off0);
        __dtu_v_tvstda_f32_dual(dacc11, ot_base, ot_off1);
        __dtu_v_tvstda_f32_dual(dacc12, ot_base, ot_off0);
        __dtu_v_tvstda_f32_dual(dacc13, ot_base, ot_off1);
        __dtu_v_tvstda_f32_dual(dacc14, ot_base, ot_off0);
        __dtu_v_tvstda_f32_dual(dacc15, ot_base, ot_off1);

        __dtu_v_vpr_tvstda_f32_dual(dacc16, ot_base, ot_off0);
        __dtu_v_vpr_tvstda_f32_dual(dacc17, ot_base, ot_off1);
        __dtu_v_vpr_tvstda_f32_dual(dacc18, ot_base, ot_off0);
        __dtu_v_vpr_tvstda_f32_dual(dacc19, ot_base, ot_off1);
        __dtu_v_vpr_tvstda_f32_dual(dacc20, ot_base, ot_off0);
        __dtu_v_vpr_tvstda_f32_dual(dacc21, ot_base, ot_off1);
        __dtu_v_vpr_tvstda_f32_dual(dacc22, ot_base, ot_off0);
        __dtu_v_vpr_tvstda_f32_dual(dacc23, ot_base, ot_off1);

        __dtu_v_vpr_tvstda_f32_dual(dacc24, ot_base, ot_off0);
        __dtu_v_vpr_tvstda_f32_dual(dacc25, ot_base, ot_off1);
        __dtu_v_vpr_tvstda_f32_dual(dacc26, ot_base, ot_off0);
        __dtu_v_vpr_tvstda_f32_dual(dacc27, ot_base, ot_off1);
        __dtu_v_vpr_tvstda_f32_dual(dacc28, ot_base, ot_off0);
        __dtu_v_vpr_tvstda_f32_dual(dacc29, ot_base, ot_off1);
        __dtu_v_vpr_tvstda_f32_dual(dacc30, ot_base, ot_off0);
        __dtu_v_vpr_tvstda_f32_dual(dacc31, ot_base, ot_off1);

        ot_base = __dtu_v_taradd(ot_base, ot_off2);

        vab_shift += 192;
        __dtu_c_movsr2vab_lv_s(vab_shift);
      }
    }
  } else {
    tar_t ot_base = __dtu_c_movsr2targ(TAR32(out_addr, out_addr + 1));
    int offset = TAR32(N >> 6, N >> 6);
    tar_t ot_off0 = __dtu_c_movsr2tari(offset, ot_base);
    offset = TAR32(2 - 15 * (N >> 6), 2 - 15 * (N >> 6));
    tar_t ot_off1 = __dtu_c_movsr2tari(offset, ot_base);

    vab_shift = 0;
    __dtu_c_movsr2vab_m_s1(0);
    __dtu_c_movsr2vab_m_d(0);
    __dtu_c_movsr2vab_lv_s(0);

#pragma clang loop unroll(disable)
    for (int n = 0; n < N; n += 128) {
      auto dacc0 = __dtu_m_mop_cvt_qa_rne_f32_f16(qacc0);
      auto dacc1 = __dtu_m_mop_cvt_qa_rne_f32_f16(qacc1);
      auto dacc2 = __dtu_m_mop_cvt_qa_rne_f32_f16(qacc2);
      auto dacc3 = __dtu_m_mop_cvt_qa_rne_f32_f16(qacc3);
      auto dacc4 = __dtu_m_mop_cvt_qa_rne_f32_f16(qacc4);
      auto dacc5 = __dtu_m_mop_cvt_qa_rne_f32_f16(qacc5);
      auto dacc6 = __dtu_m_mop_cvt_qa_rne_f32_f16(qacc6);
      auto dacc7 = __dtu_m_mop_cvt_qa_rne_f32_f16(qacc7);
      auto dacc8 = __dtu_m_mop_cvt_qa_rne_f32_f16(qacc8);
      auto dacc9 = __dtu_m_mop_cvt_qa_rne_f32_f16(qacc9);
      auto dacc10 = __dtu_m_mop_cvt_qa_rne_f32_f16(qacc10);
      auto dacc11 = __dtu_m_mop_cvt_qa_rne_f32_f16(qacc11);
      auto dacc12 = __dtu_m_mop_cvt_qa_rne_f32_f16(qacc12);
      auto dacc13 = __dtu_m_mop_cvt_qa_rne_f32_f16(qacc13);
      auto dacc14 = __dtu_m_mop_cvt_qa_rne_f32_f16(qacc14);
      auto dacc15 = __dtu_m_mop_cvt_qa_rne_f32_f16(qacc15);

      __dtu_v_tvstda_f16_dual(dacc0, ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(dacc1, ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(dacc2, ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(dacc3, ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(dacc4, ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(dacc5, ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(dacc6, ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(dacc7, ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(dacc8, ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(dacc9, ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(dacc10, ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(dacc11, ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(dacc12, ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(dacc13, ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(dacc14, ot_base, ot_off0);
      __dtu_v_tvstda_f16_dual(dacc15, ot_base, ot_off1);

      vab_shift += 192;
      __dtu_c_movsr2vab_m_s1(vab_shift);
    }
  }
}

// template <>
__device__ void __attribute__((dtu_maxinum_vacc(192))) call_k_dot_kn(
    float* out, float* lhs, tops::bfloat* rhs,
    int M, int K, int N, int left_ctx_len, int f32_ind) {
  int out_addr = reinterpret_cast<long>(out);
  int lhs_addr = reinterpret_cast<long>(lhs);
  int rhs_addr = reinterpret_cast<long>(rhs);

  int k = left_ctx_len;
  auto vr_0 = __dtu_s_movr2vr_dup(0);
  if (K > k) {
    int loop_times = ((K - k) * N) >> 6;
    char* st_addr = reinterpret_cast<char*>(rhs + k * N);
    for (int i = 0; i < loop_times; i++) {
      __dtu_l_vstl(vr_0, st_addr + 128 * i, 0);
    }
  }

  int naccovr = 0x10001;

  auto k_unit = K >> 5;
  auto n_unit = N >> 6;

  lhs_addr = lhs_addr >> 7;
  rhs_addr = rhs_addr >> 7;
  out_addr = out_addr >> 7;

  tar_t lt_base = __dtu_c_movsr2targ(TAR32(lhs_addr, lhs_addr));
  int offset = TAR32(k_unit, k_unit);
  tar_t lt_off0 = __dtu_c_movsr2tari(offset, lt_base);
  offset = TAR32(1 - 15 * k_unit, 1 - 15 * k_unit);
  tar_t lt_off1 = __dtu_c_movsr2tari(offset, lt_base);

  tar_t rt_base = __dtu_c_movsr2targ(TAR32(rhs_addr, rhs_addr + 1));
  offset = TAR32(n_unit, n_unit);
  tar_t rt_off0 = __dtu_c_movsr2tari(offset, rt_base);
  offset = TAR32(2 - (K) * n_unit, 2 - (K) * n_unit);
  tar_t rt_off1 = __dtu_c_movsr2tari(offset, rt_base);

  int vpr = (M > 8) ? 0 : 1;
  __dtu_c_movsr2vpr(vpr);

  int vab_shift = 0;
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_d(0);
  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_lv_d(0);
  __dtu_c_movsr2vab_m_s2(0);

  smr_t smr0, smr1;
  va16f32x4 qacc0, qacc1, qacc2, qacc3, qacc4, qacc5, qacc6, qacc7,
      qacc8, qacc9, qacc10, qacc11, qacc12, qacc13, qacc14, qacc15;

#pragma clang loop unroll(disable)
  for (int n = 0; n < N; n += 128) {
    __dtu_c_movsr2naccovr(naccovr);

    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 0);
    auto da0 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 1);
    auto da1 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 2);
    auto da2 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 3);
    auto da3 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 4);
    auto da4 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 5);
    auto da5 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 6);
    auto da6 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 7);
    auto da7 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 8);
    auto da8 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 9);
    auto da9 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 10);
    auto da10 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 11);
    auto da11 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 12);
    auto da12 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 13);
    auto da13 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 14);
    auto da14 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 15);
    auto da15 = __dtu_l_tvldqa_f32_da(lt_base, lt_off1);

    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 16);
    auto vr0 = __dtu_l_movva2vr_cvt2bf16(da0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 17);
    auto vr1 = __dtu_l_movva2vr_cvt2bf16(da1);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 18);
    auto vr2 = __dtu_l_movva2vr_cvt2bf16(da2);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 19);
    auto vr3 = __dtu_l_movva2vr_cvt2bf16(da3);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 20);
    auto vr4 = __dtu_l_movva2vr_cvt2bf16(da4);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 21);
    auto vr5 = __dtu_l_movva2vr_cvt2bf16(da5);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 22);
    auto vr6 = __dtu_l_movva2vr_cvt2bf16(da6);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 23);
    auto vr7 = __dtu_l_movva2vr_cvt2bf16(da7);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 24);
    auto vr8 = __dtu_l_movva2vr_cvt2bf16(da8);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 25);
    auto vr9 = __dtu_l_movva2vr_cvt2bf16(da9);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 26);
    auto vr10 = __dtu_l_movva2vr_cvt2bf16(da10);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 27);
    auto vr11 = __dtu_l_movva2vr_cvt2bf16(da11);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 28);
    auto vr12 = __dtu_l_movva2vr_cvt2bf16(da12);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 29);
    auto vr13 = __dtu_l_movva2vr_cvt2bf16(da13);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 30);
    auto vr14 = __dtu_l_movva2vr_cvt2bf16(da14);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 31);
    auto vr15 = __dtu_l_movva2vr_cvt2bf16(da15);


    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 0);
    auto da16 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 1);
    auto da17 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 2);
    auto da18 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 3);
    auto da19 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 4);
    auto da20 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 5);
    auto da21 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 6);
    auto da22 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 7);
    auto da23 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 8);
    auto da24 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 9);
    auto da25 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 10);
    auto da26 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 11);
    auto da27 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 12);
    auto da28 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 13);
    auto da29 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 14);
    auto da30 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 15);
    auto da31 = __dtu_l_tvldqa_f32_da(lt_base, lt_off1);

    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 16);
    auto vr16 = __dtu_l_movva2vr_cvt2bf16(da16);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 17);
    auto vr17 = __dtu_l_movva2vr_cvt2bf16(da17);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 18);
    auto vr18 = __dtu_l_movva2vr_cvt2bf16(da18);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 19);
    auto vr19 = __dtu_l_movva2vr_cvt2bf16(da19);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 20);
    auto vr20 = __dtu_l_movva2vr_cvt2bf16(da20);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 21);
    auto vr21 = __dtu_l_movva2vr_cvt2bf16(da21);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 22);
    auto vr22 = __dtu_l_movva2vr_cvt2bf16(da22);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 23);
    auto vr23 = __dtu_l_movva2vr_cvt2bf16(da23);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 24);
    auto vr24 = __dtu_l_movva2vr_cvt2bf16(da24);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 25);
    auto vr25 = __dtu_l_movva2vr_cvt2bf16(da25);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 26);
    auto vr26 = __dtu_l_movva2vr_cvt2bf16(da26);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 27);
    auto vr27 = __dtu_l_movva2vr_cvt2bf16(da27);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 28);
    auto vr28 = __dtu_l_movva2vr_cvt2bf16(da28);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 29);
    auto vr29 = __dtu_l_movva2vr_cvt2bf16(da29);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 30);
    auto vr30 = __dtu_l_movva2vr_cvt2bf16(da30);
    smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 31);
    auto vr31 = __dtu_l_movva2vr_cvt2bf16(da31);

#pragma clang loop unroll(disable)
    for (int k = 0; k < K - 64; k += 64) {
      qacc0 = __dtu_m_vmm2_mode17_bf16_nacc(qacc0, vr0, smr0);
      da0 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc1 = __dtu_m_vmm2_mode17_bf16_nacc(qacc1, vr1, smr0);
      da1 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc2 = __dtu_m_vmm2_mode17_bf16_nacc(qacc2, vr2, smr0);
      da2 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc3 = __dtu_m_vmm2_mode17_bf16_nacc(qacc3, vr3, smr0);
      da3 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc4 = __dtu_m_vmm2_mode17_bf16_nacc(qacc4, vr4, smr0);
      da4 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc5 = __dtu_m_vmm2_mode17_bf16_nacc(qacc5, vr5, smr0);
      da5 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc6 = __dtu_m_vmm2_mode17_bf16_nacc(qacc6, vr6, smr0);
      da6 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc7 = __dtu_m_vmm2_mode17_bf16_nacc(qacc7, vr7, smr0);
      da7 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc8 = __dtu_m_vmm2_mode17_bf16_nacc(qacc8, vr8, smr0);
      da8 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc9 = __dtu_m_vmm2_mode17_bf16_nacc(qacc9, vr9, smr0);
      da9 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc10 = __dtu_m_vmm2_mode17_bf16_nacc(qacc10, vr10, smr0);
      da10 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc11 = __dtu_m_vmm2_mode17_bf16_nacc(qacc11, vr11, smr0);
      da11 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc12 = __dtu_m_vmm2_mode17_bf16_nacc(qacc12, vr12, smr0);
      da12 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc13 = __dtu_m_vmm2_mode17_bf16_nacc(qacc13, vr13, smr0);
      da13 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc14 = __dtu_m_vmm2_mode17_bf16_nacc(qacc14, vr14, smr0);
      da14 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);
      qacc15 = __dtu_m_vmm2_mode17_bf16_nacc(qacc15, vr15, smr0);
      da15 = __dtu_l_tvldqa_f32_da(lt_base, lt_off1);

      __dtu_c_movsr2naccovr(0x1);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 0);
      qacc0 = __dtu_m_vmm2_mode17_bf16_nacc(qacc0, vr16, smr1);
      da16 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 1);
      qacc1 = __dtu_m_vmm2_mode17_bf16_nacc(qacc1, vr17, smr1);
      da17 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 2);
      qacc2 = __dtu_m_vmm2_mode17_bf16_nacc(qacc2, vr18, smr1);
      da18 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 3);
      qacc3 = __dtu_m_vmm2_mode17_bf16_nacc(qacc3, vr19, smr1);
      da19 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 4);
      qacc4 = __dtu_m_vmm2_mode17_bf16_nacc(qacc4, vr20, smr1);
      da20 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 5);
      qacc5 = __dtu_m_vmm2_mode17_bf16_nacc(qacc5, vr21, smr1);
      da21 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 6);
      qacc6 = __dtu_m_vmm2_mode17_bf16_nacc(qacc6, vr22, smr1);
      da22 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 7);
      qacc7 = __dtu_m_vmm2_mode17_bf16_nacc(qacc7, vr23, smr1);
      da23 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 8);
      qacc8 = __dtu_m_vmm2_mode17_bf16_nacc(qacc8, vr24, smr1);
      da24 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 9);
      qacc9 = __dtu_m_vmm2_mode17_bf16_nacc(qacc9, vr25, smr1);
      da25 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 10);
      qacc10 = __dtu_m_vmm2_mode17_bf16_nacc(qacc10, vr26, smr1);
      da26 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 11);
      qacc11 = __dtu_m_vmm2_mode17_bf16_nacc(qacc11, vr27, smr1);
      da27 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 12);
      qacc12 = __dtu_m_vmm2_mode17_bf16_nacc(qacc12, vr28, smr1);
      da28 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 13);
      qacc13 = __dtu_m_vmm2_mode17_bf16_nacc(qacc13, vr29, smr1);
      da29 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 14);
      qacc14 = __dtu_m_vmm2_mode17_bf16_nacc(qacc14, vr30, smr1);
      da30 = __dtu_l_tvldqa_f32_da(lt_base, lt_off0);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 15);
      qacc15 = __dtu_m_vmm2_mode17_bf16_nacc(qacc15, vr31, smr1);
      da31 = __dtu_l_tvldqa_f32_da(lt_base, lt_off1);

      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 16);
      vr0 = __dtu_l_movva2vr_cvt2bf16(da0);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 17);
      vr1 = __dtu_l_movva2vr_cvt2bf16(da1);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 18);
      vr2 = __dtu_l_movva2vr_cvt2bf16(da2);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 19);
      vr3 = __dtu_l_movva2vr_cvt2bf16(da3);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 20);
      vr4 = __dtu_l_movva2vr_cvt2bf16(da4);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 21);
      vr5 = __dtu_l_movva2vr_cvt2bf16(da5);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 22);
      vr6 = __dtu_l_movva2vr_cvt2bf16(da6);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 23);
      vr7 = __dtu_l_movva2vr_cvt2bf16(da7);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 24);
      vr8 = __dtu_l_movva2vr_cvt2bf16(da8);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 25);
      vr9 = __dtu_l_movva2vr_cvt2bf16(da9);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 26);
      vr10 = __dtu_l_movva2vr_cvt2bf16(da10);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 27);
      vr11 = __dtu_l_movva2vr_cvt2bf16(da11);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 28);
      vr12 = __dtu_l_movva2vr_cvt2bf16(da12);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 29);
      vr13 = __dtu_l_movva2vr_cvt2bf16(da13);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 30);
      vr14 = __dtu_l_movva2vr_cvt2bf16(da14);
      smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr0, rt_base, rt_off0, 31);
      vr15 = __dtu_l_movva2vr_cvt2bf16(da15);

      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 0);
      vr16 = __dtu_l_movva2vr_cvt2bf16(da16);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 1);
      vr17 = __dtu_l_movva2vr_cvt2bf16(da17);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 2);
      vr18 = __dtu_l_movva2vr_cvt2bf16(da18);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 3);
      vr19 = __dtu_l_movva2vr_cvt2bf16(da19);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 4);
      vr20 = __dtu_l_movva2vr_cvt2bf16(da20);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 5);
      vr21 = __dtu_l_movva2vr_cvt2bf16(da21);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 6);
      vr22 = __dtu_l_movva2vr_cvt2bf16(da22);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 7);
      vr23 = __dtu_l_movva2vr_cvt2bf16(da23);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 8);
      vr24 = __dtu_l_movva2vr_cvt2bf16(da24);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 9);
      vr25 = __dtu_l_movva2vr_cvt2bf16(da25);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 10);
      vr26 = __dtu_l_movva2vr_cvt2bf16(da26);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 11);
      vr27 = __dtu_l_movva2vr_cvt2bf16(da27);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 12);
      vr28 = __dtu_l_movva2vr_cvt2bf16(da28);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 13);
      vr29 = __dtu_l_movva2vr_cvt2bf16(da29);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 14);
      vr30 = __dtu_l_movva2vr_cvt2bf16(da30);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_row(smr1, rt_base, rt_off0, 15);
      vr31 = __dtu_l_movva2vr_cvt2bf16(da31);

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
    qacc1 = __dtu_m_vmm2_mode17_bf16(qacc1, vr1, smr0);
    qacc2 = __dtu_m_vmm2_mode17_bf16(qacc2, vr2, smr0);
    qacc3 = __dtu_m_vmm2_mode17_bf16(qacc3, vr3, smr0);
    qacc4 = __dtu_m_vmm2_mode17_bf16(qacc4, vr4, smr0);
    qacc5 = __dtu_m_vmm2_mode17_bf16(qacc5, vr5, smr0);
    qacc6 = __dtu_m_vmm2_mode17_bf16(qacc6, vr6, smr0);
    qacc7 = __dtu_m_vmm2_mode17_bf16(qacc7, vr7, smr0);
    qacc8 = __dtu_m_vmm2_mode17_bf16(qacc8, vr8, smr0);
    qacc9 = __dtu_m_vmm2_mode17_bf16(qacc9, vr9, smr0);
    qacc10 = __dtu_m_vmm2_mode17_bf16(qacc10, vr10, smr0);
    qacc11 = __dtu_m_vmm2_mode17_bf16(qacc11, vr11, smr0);
    qacc12 = __dtu_m_vmm2_mode17_bf16(qacc12, vr12, smr0);
    qacc13 = __dtu_m_vmm2_mode17_bf16(qacc13, vr13, smr0);
    qacc14 = __dtu_m_vmm2_mode17_bf16(qacc14, vr14, smr0);
    qacc15 = __dtu_m_vmm2_mode17_bf16(qacc15, vr15, smr0);

    __dtu_c_movsr2naccovr(0x1);
    qacc0 = __dtu_m_vmm2_mode17_bf16(qacc0, vr16, smr1);
    qacc1 = __dtu_m_vmm2_mode17_bf16(qacc1, vr17, smr1);
    qacc2 = __dtu_m_vmm2_mode17_bf16(qacc2, vr18, smr1);
    qacc3 = __dtu_m_vmm2_mode17_bf16(qacc3, vr19, smr1);
    qacc4 = __dtu_m_vmm2_mode17_bf16(qacc4, vr20, smr1);
    qacc5 = __dtu_m_vmm2_mode17_bf16(qacc5, vr21, smr1);
    qacc6 = __dtu_m_vmm2_mode17_bf16(qacc6, vr22, smr1);
    qacc7 = __dtu_m_vmm2_mode17_bf16(qacc7, vr23, smr1);
    qacc8 = __dtu_m_vmm2_mode17_bf16(qacc8, vr24, smr1);
    qacc9 = __dtu_m_vmm2_mode17_bf16(qacc9, vr25, smr1);
    qacc10 = __dtu_m_vmm2_mode17_bf16(qacc10, vr26, smr1);
    qacc11 = __dtu_m_vmm2_mode17_bf16(qacc11, vr27, smr1);
    qacc12 = __dtu_m_vmm2_mode17_bf16(qacc12, vr28, smr1);
    qacc13 = __dtu_m_vmm2_mode17_bf16(qacc13, vr29, smr1);
    qacc14 = __dtu_m_vmm2_mode17_bf16(qacc14, vr30, smr1);
    qacc15 = __dtu_m_vmm2_mode17_bf16(qacc15, vr31, smr1);

    vab_shift += 192;
    __dtu_c_movsr2vab_m_s1(vab_shift);
    __dtu_c_movsr2vab_m_d(vab_shift);

    lt_base = __dtu_v_taradd(lt_base, lt_off1);
    rt_base = __dtu_v_taradd(rt_base, rt_off1);
  }

  if (f32_ind == 1) {
    vab_shift = 0;
    __dtu_c_movsr2vab_lv_s(0);
    int m = (M > 8) ? ALIGN_16(M) : M;

    tar_t ot_base = __dtu_c_movsr2targ(TAR32(out_addr, out_addr + 2));
    offset = TAR32(1, 1);
    tar_t ot_off0 = __dtu_c_movsr2tari(offset, ot_base);
    offset = TAR32((N >> 5) - 1, (N >> 5) - 1);
    tar_t ot_off1 = __dtu_c_movsr2tari(offset, ot_base);
    offset = TAR32(2 - m * (N >> 5), 2 - m * (N >> 5));
    tar_t ot_off2 = __dtu_c_movsr2tari(offset, ot_base);

    if (M == 4) {
#pragma clang loop unroll(disable)
      for (int n = 0; n < N; n += 128) {
        auto dacc0 = __dtu_extractqa2da(qacc0, 0);
        auto dacc1 = __dtu_extractqa2da(qacc0, 1);
        auto dacc2 = __dtu_extractqa2da(qacc1, 0);
        auto dacc3 = __dtu_extractqa2da(qacc1, 1);
        auto dacc4 = __dtu_extractqa2da(qacc2, 0);
        auto dacc5 = __dtu_extractqa2da(qacc2, 1);
        auto dacc6 = __dtu_extractqa2da(qacc3, 0);
        auto dacc7 = __dtu_extractqa2da(qacc3, 1);

        __dtu_v_tvstda_f32_dual(dacc0, ot_base, ot_off0);
        __dtu_v_tvstda_f32_dual(dacc1, ot_base, ot_off1);
        __dtu_v_tvstda_f32_dual(dacc2, ot_base, ot_off0);
        __dtu_v_tvstda_f32_dual(dacc3, ot_base, ot_off1);
        __dtu_v_tvstda_f32_dual(dacc4, ot_base, ot_off0);
        __dtu_v_tvstda_f32_dual(dacc5, ot_base, ot_off1);
        __dtu_v_tvstda_f32_dual(dacc6, ot_base, ot_off0);
        __dtu_v_tvstda_f32_dual(dacc7, ot_base, ot_off1);

        ot_base = __dtu_v_taradd(ot_base, ot_off2);

        vab_shift += 192;
        __dtu_c_movsr2vab_lv_s(vab_shift);
      }
    } else {
#pragma clang loop unroll(disable)
      for (int n = 0; n < N; n += 128) {
        auto dacc0 = __dtu_extractqa2da(qacc0, 0);
        auto dacc1 = __dtu_extractqa2da(qacc0, 1);
        auto dacc2 = __dtu_extractqa2da(qacc1, 0);
        auto dacc3 = __dtu_extractqa2da(qacc1, 1);
        auto dacc4 = __dtu_extractqa2da(qacc2, 0);
        auto dacc5 = __dtu_extractqa2da(qacc2, 1);
        auto dacc6 = __dtu_extractqa2da(qacc3, 0);
        auto dacc7 = __dtu_extractqa2da(qacc3, 1);
        auto dacc8 = __dtu_extractqa2da(qacc4, 0);
        auto dacc9 = __dtu_extractqa2da(qacc4, 1);
        auto dacc10 = __dtu_extractqa2da(qacc5, 0);
        auto dacc11 = __dtu_extractqa2da(qacc5, 1);
        auto dacc12 = __dtu_extractqa2da(qacc6, 0);
        auto dacc13 = __dtu_extractqa2da(qacc6, 1);
        auto dacc14 = __dtu_extractqa2da(qacc7, 0);
        auto dacc15 = __dtu_extractqa2da(qacc7, 1);
        auto dacc16 = __dtu_extractqa2da(qacc8, 0);
        auto dacc17 = __dtu_extractqa2da(qacc8, 1);
        auto dacc18 = __dtu_extractqa2da(qacc9, 0);
        auto dacc19 = __dtu_extractqa2da(qacc9, 1);
        auto dacc20 = __dtu_extractqa2da(qacc10, 0);
        auto dacc21 = __dtu_extractqa2da(qacc10, 1);
        auto dacc22 = __dtu_extractqa2da(qacc11, 0);
        auto dacc23 = __dtu_extractqa2da(qacc11, 1);
        auto dacc24 = __dtu_extractqa2da(qacc12, 0);
        auto dacc25 = __dtu_extractqa2da(qacc12, 1);
        auto dacc26 = __dtu_extractqa2da(qacc13, 0);
        auto dacc27 = __dtu_extractqa2da(qacc13, 1);
        auto dacc28 = __dtu_extractqa2da(qacc14, 0);
        auto dacc29 = __dtu_extractqa2da(qacc14, 1);
        auto dacc30 = __dtu_extractqa2da(qacc15, 0);
        auto dacc31 = __dtu_extractqa2da(qacc15, 1);

        __dtu_v_tvstda_f32_dual(dacc0, ot_base, ot_off0);
        __dtu_v_tvstda_f32_dual(dacc1, ot_base, ot_off1);
        __dtu_v_tvstda_f32_dual(dacc2, ot_base, ot_off0);
        __dtu_v_tvstda_f32_dual(dacc3, ot_base, ot_off1);
        __dtu_v_tvstda_f32_dual(dacc4, ot_base, ot_off0);
        __dtu_v_tvstda_f32_dual(dacc5, ot_base, ot_off1);
        __dtu_v_tvstda_f32_dual(dacc6, ot_base, ot_off0);
        __dtu_v_tvstda_f32_dual(dacc7, ot_base, ot_off1);

        __dtu_v_tvstda_f32_dual(dacc8, ot_base, ot_off0);
        __dtu_v_tvstda_f32_dual(dacc9, ot_base, ot_off1);
        __dtu_v_tvstda_f32_dual(dacc10, ot_base, ot_off0);
        __dtu_v_tvstda_f32_dual(dacc11, ot_base, ot_off1);
        __dtu_v_tvstda_f32_dual(dacc12, ot_base, ot_off0);
        __dtu_v_tvstda_f32_dual(dacc13, ot_base, ot_off1);
        __dtu_v_tvstda_f32_dual(dacc14, ot_base, ot_off0);
        __dtu_v_tvstda_f32_dual(dacc15, ot_base, ot_off1);

        __dtu_v_vpr_tvstda_f32_dual(dacc16, ot_base, ot_off0);
        __dtu_v_vpr_tvstda_f32_dual(dacc17, ot_base, ot_off1);
        __dtu_v_vpr_tvstda_f32_dual(dacc18, ot_base, ot_off0);
        __dtu_v_vpr_tvstda_f32_dual(dacc19, ot_base, ot_off1);
        __dtu_v_vpr_tvstda_f32_dual(dacc20, ot_base, ot_off0);
        __dtu_v_vpr_tvstda_f32_dual(dacc21, ot_base, ot_off1);
        __dtu_v_vpr_tvstda_f32_dual(dacc22, ot_base, ot_off0);
        __dtu_v_vpr_tvstda_f32_dual(dacc23, ot_base, ot_off1);

        __dtu_v_vpr_tvstda_f32_dual(dacc24, ot_base, ot_off0);
        __dtu_v_vpr_tvstda_f32_dual(dacc25, ot_base, ot_off1);
        __dtu_v_vpr_tvstda_f32_dual(dacc26, ot_base, ot_off0);
        __dtu_v_vpr_tvstda_f32_dual(dacc27, ot_base, ot_off1);
        __dtu_v_vpr_tvstda_f32_dual(dacc28, ot_base, ot_off0);
        __dtu_v_vpr_tvstda_f32_dual(dacc29, ot_base, ot_off1);
        __dtu_v_vpr_tvstda_f32_dual(dacc30, ot_base, ot_off0);
        __dtu_v_vpr_tvstda_f32_dual(dacc31, ot_base, ot_off1);

        ot_base = __dtu_v_taradd(ot_base, ot_off2);

        vab_shift += 192;
        __dtu_c_movsr2vab_lv_s(vab_shift);
      }
    }
  } else {
    tar_t ot_base = __dtu_c_movsr2targ(TAR32(out_addr, out_addr + 1));
    int offset = TAR32(N >> 6, N >> 6);
    tar_t ot_off0 = __dtu_c_movsr2tari(offset, ot_base);
    offset = TAR32(2 - 15 * (N >> 6), 2 - 15 * (N >> 6));
    tar_t ot_off1 = __dtu_c_movsr2tari(offset, ot_base);

    vab_shift = 0;
    __dtu_c_movsr2vab_m_s1(0);
    __dtu_c_movsr2vab_m_d(0);
    __dtu_c_movsr2vab_lv_s(0);

#pragma clang loop unroll(disable)
    for (int n = 0; n < N; n += 128) {
      auto dacc0 = __dtu_m_mop_cvt_qa_rne_f32_bf16(qacc0);
      auto dacc1 = __dtu_m_mop_cvt_qa_rne_f32_bf16(qacc1);
      auto dacc2 = __dtu_m_mop_cvt_qa_rne_f32_bf16(qacc2);
      auto dacc3 = __dtu_m_mop_cvt_qa_rne_f32_bf16(qacc3);
      auto dacc4 = __dtu_m_mop_cvt_qa_rne_f32_bf16(qacc4);
      auto dacc5 = __dtu_m_mop_cvt_qa_rne_f32_bf16(qacc5);
      auto dacc6 = __dtu_m_mop_cvt_qa_rne_f32_bf16(qacc6);
      auto dacc7 = __dtu_m_mop_cvt_qa_rne_f32_bf16(qacc7);
      auto dacc8 = __dtu_m_mop_cvt_qa_rne_f32_bf16(qacc8);
      auto dacc9 = __dtu_m_mop_cvt_qa_rne_f32_bf16(qacc9);
      auto dacc10 = __dtu_m_mop_cvt_qa_rne_f32_bf16(qacc10);
      auto dacc11 = __dtu_m_mop_cvt_qa_rne_f32_bf16(qacc11);
      auto dacc12 = __dtu_m_mop_cvt_qa_rne_f32_bf16(qacc12);
      auto dacc13 = __dtu_m_mop_cvt_qa_rne_f32_bf16(qacc13);
      auto dacc14 = __dtu_m_mop_cvt_qa_rne_f32_bf16(qacc14);
      auto dacc15 = __dtu_m_mop_cvt_qa_rne_f32_bf16(qacc15);

      __dtu_v_tvstda_bf16_dual(dacc0, ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(dacc1, ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(dacc2, ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(dacc3, ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(dacc4, ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(dacc5, ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(dacc6, ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(dacc7, ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(dacc8, ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(dacc9, ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(dacc10, ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(dacc11, ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(dacc12, ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(dacc13, ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(dacc14, ot_base, ot_off0);
      __dtu_v_tvstda_bf16_dual(dacc15, ot_base, ot_off1);

      vab_shift += 192;
      __dtu_c_movsr2vab_m_s1(vab_shift);
    }
  }
}

// template <>
__device__ void __attribute__((dtu_maxinum_vacc(64))) call_k_dot_nk(
    float* out, tops::half* lhs, tops::half* rhs,
    int M, int K, int N, float scale) {
  int out_addr = reinterpret_cast<long>(out);
  int lhs_addr = reinterpret_cast<long>(lhs);
  int rhs_addr = reinterpret_cast<long>(rhs);

  int m = (M > 8) ? ALIGN_16(M) : M;
  auto k_unit = K >> 6;
  auto n_unit = N >> 5;
  lhs_addr = lhs_addr >> 7;
  rhs_addr = rhs_addr >> 7;
  out_addr = out_addr >> 7;

  tar_t lt_base =
      __dtu_c_movsr2targ(TAR32(lhs_addr, lhs_addr));
  int offset = TAR32(k_unit, k_unit);
  tar_t lt_off0 = __dtu_c_movsr2tari(offset, lt_base);
  offset = TAR32(1 - 15 * k_unit, 1 - 15 * k_unit);
  tar_t lt_off1 = __dtu_c_movsr2tari(offset, lt_base);

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
  offset = TAR32(n_unit, n_unit);
  tar_t ot_off0 = __dtu_c_movsr2tari(offset, ot_base);
  offset = TAR32(2 - m * n_unit, 2 - m * n_unit);
  tar_t ot_off1 = __dtu_c_movsr2tari(offset, ot_base);

  int vpr = (M > 8) ? 0 : 1;
  __dtu_c_movsr2vpr(vpr);

  smr_t smr0, smr1;

  int vab_shift = 0;
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_d(0);
  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_lv_d(0);
  __dtu_c_movsr2vab_m_s2(0);

  va16f32x2 dacc21, dacc22, dacc23, dacc24, dacc25, dacc26, dacc27, dacc28,
      dacc29, dacc30, dacc31, dacc32, dacc33, dacc34, dacc35, dacc36;

  using vtype = typename scalar_to_vector<float, TOPS_VECTOR_LENGTH / 2>::type;
  vtype v_scale = vbroadcast<vtype>(scale);
  __dtu_c_movsr2naccovr(0x10001);

#pragma clang loop unroll(disable)
  for (int k = 0; k < K; k += 64) {
    __dtu_c_movsr2vab_m_s1(0);
    __dtu_c_movsr2vab_m_d(0);
    vab_shift = 0;

    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 0);
    auto vr0 = __dtu_s_tivldd_itar_f32(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 1);
    auto vr1 = __dtu_s_tivldd_itar_f32(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 2);
    auto vr2 = __dtu_s_tivldd_itar_f32(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 3);
    auto vr3 = __dtu_s_tivldd_itar_f32(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 4);
    auto vr4 = __dtu_s_tivldd_itar_f32(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 5);
    auto vr5 = __dtu_s_tivldd_itar_f32(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 6);
    auto vr6 = __dtu_s_tivldd_itar_f32(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 7);
    auto vr7 = __dtu_s_tivldd_itar_f32(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 8);
    auto vr8 = __dtu_s_tivldd_itar_f32(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 9);
    auto vr9 = __dtu_s_tivldd_itar_f32(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 10);
    auto vr10 = __dtu_s_tivldd_itar_f32(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 11);
    auto vr11 = __dtu_s_tivldd_itar_f32(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 12);
    auto vr12 = __dtu_s_tivldd_itar_f32(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 13);
    auto vr13 = __dtu_s_tivldd_itar_f32(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 14);
    auto vr14 = __dtu_s_tivldd_itar_f32(lt_base, lt_off0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode16_f16_col(smr0, rt_base, rt_off0, 15);
    auto vr15 = __dtu_s_tivldd_itar_f32(lt_base, lt_off1);
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
    // rt_base = __dtu_v_taradd(rt_base, rt_off1);

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
      // rt_base = __dtu_v_taradd(rt_base, rt_off1);

      dacc21 = __dtu_m_vmm2_mode16_f16_nacc(dacc21, vr0, smr0);
      dacc22 = __dtu_m_vmm2_mode16_f16_nacc(dacc22, vr1, smr0);
      dacc23 = __dtu_m_vmm2_mode16_f16_nacc(dacc23, vr2, smr0);
      dacc24 = __dtu_m_vmm2_mode16_f16_nacc(dacc24, vr3, smr0);
      dacc25 = __dtu_m_vmm2_mode16_f16_nacc(dacc25, vr4, smr0);
      dacc26 = __dtu_m_vmm2_mode16_f16_nacc(dacc26, vr5, smr0);
      dacc27 = __dtu_m_vmm2_mode16_f16_nacc(dacc27, vr6, smr0);
      dacc28 = __dtu_m_vmm2_mode16_f16_nacc(dacc28, vr7, smr0);
      dacc29 = __dtu_m_vmm2_mode16_f16_nacc(dacc29, vr8, smr0);
      dacc30 = __dtu_m_vmm2_mode16_f16_nacc(dacc30, vr9, smr0);
      dacc31 = __dtu_m_vmm2_mode16_f16_nacc(dacc31, vr10, smr0);
      dacc32 = __dtu_m_vmm2_mode16_f16_nacc(dacc32, vr11, smr0);
      dacc33 = __dtu_m_vmm2_mode16_f16_nacc(dacc33, vr12, smr0);
      dacc34 = __dtu_m_vmm2_mode16_f16_nacc(dacc34, vr13, smr0);
      dacc35 = __dtu_m_vmm2_mode16_f16_nacc(dacc35, vr14, smr0);
      dacc36 = __dtu_m_vmm2_mode16_f16_nacc(dacc36, vr15, smr0);

      __dtu_v_swap_smr(smr0, smr1);

      vab_shift += 64;
      __dtu_c_movsr2vab_m_s1(vab_shift);
      __dtu_c_movsr2vab_m_d(vab_shift);
    }

    dacc21 = __dtu_m_vmm2_mode16_f16_nacc(dacc21, vr0, smr0);
    dacc22 = __dtu_m_vmm2_mode16_f16_nacc(dacc22, vr1, smr0);
    dacc23 = __dtu_m_vmm2_mode16_f16_nacc(dacc23, vr2, smr0);
    dacc24 = __dtu_m_vmm2_mode16_f16_nacc(dacc24, vr3, smr0);
    dacc25 = __dtu_m_vmm2_mode16_f16_nacc(dacc25, vr4, smr0);
    dacc26 = __dtu_m_vmm2_mode16_f16_nacc(dacc26, vr5, smr0);
    dacc27 = __dtu_m_vmm2_mode16_f16_nacc(dacc27, vr6, smr0);
    dacc28 = __dtu_m_vmm2_mode16_f16_nacc(dacc28, vr7, smr0);
    dacc29 = __dtu_m_vmm2_mode16_f16_nacc(dacc29, vr8, smr0);
    dacc30 = __dtu_m_vmm2_mode16_f16_nacc(dacc30, vr9, smr0);
    dacc31 = __dtu_m_vmm2_mode16_f16_nacc(dacc31, vr10, smr0);
    dacc32 = __dtu_m_vmm2_mode16_f16_nacc(dacc32, vr11, smr0);
    dacc33 = __dtu_m_vmm2_mode16_f16_nacc(dacc33, vr12, smr0);
    dacc34 = __dtu_m_vmm2_mode16_f16_nacc(dacc34, vr13, smr0);
    dacc35 = __dtu_m_vmm2_mode16_f16_nacc(dacc35, vr14, smr0);
    dacc36 = __dtu_m_vmm2_mode16_f16_nacc(dacc36, vr15, smr0);

    __dtu_c_movsr2naccovr(0x1);
    rt_base = __dtu_v_taradd(rt_base, rt_off2);
  }

  vab_shift = 0;
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_s2(0);
  __dtu_c_movsr2vab_m_d(0);
  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_lv_d(0);

  if (M == 4) {
#pragma clang loop unroll(disable)
    for (int n = 0; n < N; n += 64) {
      dacc21 = __dtu_m_mop_mul_f32_da(dacc21, v_scale);
      dacc22 = __dtu_m_mop_mul_f32_da(dacc22, v_scale);
      dacc23 = __dtu_m_mop_mul_f32_da(dacc23, v_scale);
      dacc24 = __dtu_m_mop_mul_f32_da(dacc24, v_scale);

      __dtu_v_tvstda_f32_dual(dacc21, ot_base, ot_off0);
      __dtu_v_tvstda_f32_dual(dacc22, ot_base, ot_off0);
      __dtu_v_tvstda_f32_dual(dacc23, ot_base, ot_off0);
      __dtu_v_tvstda_f32_dual(dacc24, ot_base, ot_off0);

      ot_base = __dtu_v_taradd(ot_base, ot_off1);

      vab_shift += 64;
      __dtu_c_movsr2vab_lv_s(vab_shift);
      __dtu_c_movsr2vab_m_s1(vab_shift);
      __dtu_c_movsr2vab_m_d(vab_shift);
    }
  } else {
#pragma clang loop unroll(disable)
    for (int n = 0; n < N; n += 64) {
      dacc21 = __dtu_m_mop_mul_f32_da(dacc21, v_scale);
      dacc22 = __dtu_m_mop_mul_f32_da(dacc22, v_scale);
      dacc23 = __dtu_m_mop_mul_f32_da(dacc23, v_scale);
      dacc24 = __dtu_m_mop_mul_f32_da(dacc24, v_scale);
      dacc25 = __dtu_m_mop_mul_f32_da(dacc25, v_scale);
      dacc26 = __dtu_m_mop_mul_f32_da(dacc26, v_scale);
      dacc27 = __dtu_m_mop_mul_f32_da(dacc27, v_scale);
      dacc28 = __dtu_m_mop_mul_f32_da(dacc28, v_scale);
      dacc29 = __dtu_m_mop_mul_f32_da(dacc29, v_scale);
      dacc30 = __dtu_m_mop_mul_f32_da(dacc30, v_scale);
      dacc31 = __dtu_m_mop_mul_f32_da(dacc31, v_scale);
      dacc32 = __dtu_m_mop_mul_f32_da(dacc32, v_scale);
      dacc33 = __dtu_m_mop_mul_f32_da(dacc33, v_scale);
      dacc34 = __dtu_m_mop_mul_f32_da(dacc34, v_scale);
      dacc35 = __dtu_m_mop_mul_f32_da(dacc35, v_scale);
      dacc36 = __dtu_m_mop_mul_f32_da(dacc36, v_scale);

      __dtu_v_tvstda_f32_dual(dacc21, ot_base, ot_off0);
      __dtu_v_tvstda_f32_dual(dacc22, ot_base, ot_off0);
      __dtu_v_tvstda_f32_dual(dacc23, ot_base, ot_off0);
      __dtu_v_tvstda_f32_dual(dacc24, ot_base, ot_off0);
      __dtu_v_tvstda_f32_dual(dacc25, ot_base, ot_off0);
      __dtu_v_tvstda_f32_dual(dacc26, ot_base, ot_off0);
      __dtu_v_tvstda_f32_dual(dacc27, ot_base, ot_off0);
      __dtu_v_tvstda_f32_dual(dacc28, ot_base, ot_off0);

      __dtu_v_vpr_tvstda_f32_dual(dacc29, ot_base, ot_off0);
      __dtu_v_vpr_tvstda_f32_dual(dacc30, ot_base, ot_off0);
      __dtu_v_vpr_tvstda_f32_dual(dacc31, ot_base, ot_off0);
      __dtu_v_vpr_tvstda_f32_dual(dacc32, ot_base, ot_off0);
      __dtu_v_vpr_tvstda_f32_dual(dacc33, ot_base, ot_off0);
      __dtu_v_vpr_tvstda_f32_dual(dacc34, ot_base, ot_off0);
      __dtu_v_vpr_tvstda_f32_dual(dacc35, ot_base, ot_off0);
      __dtu_v_vpr_tvstda_f32_dual(dacc36, ot_base, ot_off0);

      ot_base = __dtu_v_taradd(ot_base, ot_off1);

      vab_shift += 64;
      __dtu_c_movsr2vab_lv_s(vab_shift);
      __dtu_c_movsr2vab_m_s1(vab_shift);
      __dtu_c_movsr2vab_m_d(vab_shift);
    }
  }
}

// template <>
__device__ void __attribute__((dtu_maxinum_vacc(128))) call_k_dot_nk(
    float* out, tops::bfloat* lhs, tops::bfloat* rhs,
    int M, int K, int N, float scale) {
  int out_addr = reinterpret_cast<long>(out);
  int lhs_addr = reinterpret_cast<long>(lhs);
  int rhs_addr = reinterpret_cast<long>(rhs);

  auto k_unit = K >> 5;
  auto n_unit = N >> 5;
  lhs_addr = lhs_addr >> 6;
  rhs_addr = rhs_addr >> 6;
  out_addr = out_addr >> 7;

  tar_t lt_base =
      __dtu_c_movsr2targ(TAR32(lhs_addr, lhs_addr));
  int offset = TAR32(k_unit, k_unit);
  tar_t lt_off0 = __dtu_c_movsr2tari(offset, lt_base);
  offset = TAR32(1 - 15 * k_unit, 1 - 15 * k_unit);
  tar_t lt_off1 = __dtu_c_movsr2tari(offset, lt_base);

  auto next_stride = 1 - k_unit * N;
  tar_t rt_base =
      __dtu_c_movsr2targ(TAR32(rhs_addr, rhs_addr + 64 * k_unit));
  offset = TAR32(k_unit, k_unit);
  tar_t rt_off0 = __dtu_c_movsr2tari(offset, rt_base);
  offset = TAR32(k_unit * 64, k_unit * 64);
  tar_t rt_off1 = __dtu_c_movsr2tari(offset, rt_base);
  offset = TAR32(next_stride, next_stride);
  tar_t rt_off2 = __dtu_c_movsr2tari(offset, rt_base);

  tar_t ot_base = __dtu_c_movsr2targ(TAR32(out_addr, out_addr + 2));
  offset = TAR32(1, 1);
  tar_t ot_off0 = __dtu_c_movsr2tari(offset, ot_base);
  offset = TAR32(4 - M * n_unit, 4 - M * n_unit);
  tar_t ot_off1 = __dtu_c_movsr2tari(offset, ot_base);
  offset = TAR32(n_unit - 1, n_unit - 1);
  tar_t ot_off2 = __dtu_c_movsr2tari(offset, ot_base);

  int vpr = (M > 8) ? 0 : 1;
  __dtu_c_movsr2vpr(vpr);

  smr_t smr0, smr1;

  int vab_shift = 0;
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_d(0);
  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_lv_d(0);
  __dtu_c_movsr2vab_m_s2(0);

  va16f32x4 qacc21, qacc22, qacc23, qacc24, qacc25, qacc26, qacc27, qacc28,
      qacc29, qacc30, qacc31, qacc32, qacc33, qacc34, qacc35, qacc36;
  // va16f32x2 v_mid, v_ctx_len;
  using vtype = typename scalar_to_vector<float, TOPS_VECTOR_LENGTH>::type;
  vtype v_scale = vbroadcast<vtype>(scale);
  // vtype v_slope = vbroadcast<vtype>(slope);
  // auto da_mid = __dtu_l_vldqa_s32_da(p_mid);
  __dtu_c_movsr2naccovr(0x10001);

#pragma clang loop unroll(disable)
  for (int k = 0; k < K; k += 32) {
    __dtu_c_movsr2vab_m_s1(0);
    __dtu_c_movsr2vab_m_d(0);
    vab_shift = 0;

    auto vr0 = __dtu_s_tvld_itar(lt_base, lt_off0);
    auto vr1 = __dtu_s_tvld_itar(lt_base, lt_off0);
    auto vr2 = __dtu_s_tvld_itar(lt_base, lt_off0);
    auto vr3 = __dtu_s_tvld_itar(lt_base, lt_off0);
    auto vr4 = __dtu_s_tvld_itar(lt_base, lt_off0);
    auto vr5 = __dtu_s_tvld_itar(lt_base, lt_off0);
    auto vr6 = __dtu_s_tvld_itar(lt_base, lt_off0);
    auto vr7 = __dtu_s_tvld_itar(lt_base, lt_off0);
    auto vr8 = __dtu_s_tvld_itar(lt_base, lt_off0);
    auto vr9 = __dtu_s_tvld_itar(lt_base, lt_off0);
    auto vr10 = __dtu_s_tvld_itar(lt_base, lt_off0);
    auto vr11 = __dtu_s_tvld_itar(lt_base, lt_off0);
    auto vr12 = __dtu_s_tvld_itar(lt_base, lt_off0);
    auto vr13 = __dtu_s_tvld_itar(lt_base, lt_off0);
    auto vr14 = __dtu_s_tvld_itar(lt_base, lt_off0);
    auto vr15 = __dtu_s_tvld_itar(lt_base, lt_off1);

    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 0);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 1);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 2);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 3);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 4);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 5);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 6);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 7);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 8);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 9);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 10);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 11);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 12);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 13);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 14);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 15);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 16);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 17);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 18);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 19);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 20);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 21);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 22);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 23);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 24);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 25);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 26);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 27);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 28);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 29);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 30);
    smr0 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr0, rt_base, rt_off0, 31);
    rt_base = __dtu_v_taradd(rt_base, rt_off1);

#pragma clang loop unroll(disable)
    for (int n = 0; n < N - 128; n += 128) {
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 0);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 1);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 2);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 3);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 4);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 5);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 6);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 7);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 8);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 9);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 10);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 11);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 12);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 13);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 14);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 15);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 16);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 17);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 18);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 19);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 20);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 21);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 22);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 23);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 24);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 25);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 26);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 27);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 28);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 29);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 30);
      smr1 = __dtu_v_ldsmr2_mem_v_mode17_bf16_col(smr1, rt_base, rt_off0, 31);
      rt_base = __dtu_v_taradd(rt_base, rt_off1);

      qacc21 = __dtu_m_vmm2_mode17_bf16_nacc(qacc21, vr0, smr0);
      qacc22 = __dtu_m_vmm2_mode17_bf16_nacc(qacc22, vr1, smr0);
      qacc23 = __dtu_m_vmm2_mode17_bf16_nacc(qacc23, vr2, smr0);
      qacc24 = __dtu_m_vmm2_mode17_bf16_nacc(qacc24, vr3, smr0);
      qacc25 = __dtu_m_vmm2_mode17_bf16_nacc(qacc25, vr4, smr0);
      qacc26 = __dtu_m_vmm2_mode17_bf16_nacc(qacc26, vr5, smr0);
      qacc27 = __dtu_m_vmm2_mode17_bf16_nacc(qacc27, vr6, smr0);
      qacc28 = __dtu_m_vmm2_mode17_bf16_nacc(qacc28, vr7, smr0);
      qacc29 = __dtu_m_vmm2_mode17_bf16_nacc(qacc29, vr8, smr0);
      qacc30 = __dtu_m_vmm2_mode17_bf16_nacc(qacc30, vr9, smr0);
      qacc31 = __dtu_m_vmm2_mode17_bf16_nacc(qacc31, vr10, smr0);
      qacc32 = __dtu_m_vmm2_mode17_bf16_nacc(qacc32, vr11, smr0);
      qacc33 = __dtu_m_vmm2_mode17_bf16_nacc(qacc33, vr12, smr0);
      qacc34 = __dtu_m_vmm2_mode17_bf16_nacc(qacc34, vr13, smr0);
      qacc35 = __dtu_m_vmm2_mode17_bf16_nacc(qacc35, vr14, smr0);
      qacc36 = __dtu_m_vmm2_mode17_bf16_nacc(qacc36, vr15, smr0);

      __dtu_v_swap_smr(smr0, smr1);

      vab_shift += 128;
      __dtu_c_movsr2vab_m_s1(vab_shift);
      __dtu_c_movsr2vab_m_d(vab_shift);
    }

    qacc21 = __dtu_m_vmm2_mode17_bf16_nacc(qacc21, vr0, smr0);
    qacc22 = __dtu_m_vmm2_mode17_bf16_nacc(qacc22, vr1, smr0);
    qacc23 = __dtu_m_vmm2_mode17_bf16_nacc(qacc23, vr2, smr0);
    qacc24 = __dtu_m_vmm2_mode17_bf16_nacc(qacc24, vr3, smr0);
    qacc25 = __dtu_m_vmm2_mode17_bf16_nacc(qacc25, vr4, smr0);
    qacc26 = __dtu_m_vmm2_mode17_bf16_nacc(qacc26, vr5, smr0);
    qacc27 = __dtu_m_vmm2_mode17_bf16_nacc(qacc27, vr6, smr0);
    qacc28 = __dtu_m_vmm2_mode17_bf16_nacc(qacc28, vr7, smr0);
    qacc29 = __dtu_m_vmm2_mode17_bf16_nacc(qacc29, vr8, smr0);
    qacc30 = __dtu_m_vmm2_mode17_bf16_nacc(qacc30, vr9, smr0);
    qacc31 = __dtu_m_vmm2_mode17_bf16_nacc(qacc31, vr10, smr0);
    qacc32 = __dtu_m_vmm2_mode17_bf16_nacc(qacc32, vr11, smr0);
    qacc33 = __dtu_m_vmm2_mode17_bf16_nacc(qacc33, vr12, smr0);
    qacc34 = __dtu_m_vmm2_mode17_bf16_nacc(qacc34, vr13, smr0);
    qacc35 = __dtu_m_vmm2_mode17_bf16_nacc(qacc35, vr14, smr0);
    qacc36 = __dtu_m_vmm2_mode17_bf16_nacc(qacc36, vr15, smr0);

    __dtu_c_movsr2naccovr(0x1);
    rt_base = __dtu_v_taradd(rt_base, rt_off2);
  }

  vab_shift = 0;
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_s2(0);
  __dtu_c_movsr2vab_m_d(0);
  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_lv_d(0);

  if (M == 4) {
#pragma clang loop unroll(disable)
    for (int n = 0; n < N; n += 128) {
      qacc21 = __dtu_m_mop_mul_f32_qa(qacc21, v_scale);
      qacc22 = __dtu_m_mop_mul_f32_qa(qacc22, v_scale);
      qacc23 = __dtu_m_mop_mul_f32_qa(qacc23, v_scale);
      qacc24 = __dtu_m_mop_mul_f32_qa(qacc24, v_scale);

      auto da0 = __dtu_extractqa2da(qacc21, 0);
      auto da1 = __dtu_extractqa2da(qacc21, 1);
      auto da2 = __dtu_extractqa2da(qacc22, 0);
      auto da3 = __dtu_extractqa2da(qacc22, 1);
      auto da4 = __dtu_extractqa2da(qacc23, 0);
      auto da5 = __dtu_extractqa2da(qacc23, 1);
      auto da6 = __dtu_extractqa2da(qacc24, 0);
      auto da7 = __dtu_extractqa2da(qacc24, 1);

      __dtu_v_tvstda_f32_dual(da0, ot_base, ot_off0);
      __dtu_v_tvstda_f32_dual(da1, ot_base, ot_off2);
      __dtu_v_tvstda_f32_dual(da2, ot_base, ot_off0);
      __dtu_v_tvstda_f32_dual(da3, ot_base, ot_off2);
      __dtu_v_tvstda_f32_dual(da4, ot_base, ot_off0);
      __dtu_v_tvstda_f32_dual(da5, ot_base, ot_off2);
      __dtu_v_tvstda_f32_dual(da6, ot_base, ot_off0);
      __dtu_v_tvstda_f32_dual(da7, ot_base, ot_off2);

      ot_base = __dtu_v_taradd(ot_base, ot_off1);

      vab_shift += 128;
      __dtu_c_movsr2vab_lv_s(vab_shift);
      __dtu_c_movsr2vab_m_s1(vab_shift);
      __dtu_c_movsr2vab_m_d(vab_shift);
    }
  } else {
#pragma clang loop unroll(disable)
    for (int n = 0; n < N; n += 128) {
      qacc21 = __dtu_m_mop_mul_f32_qa(qacc21, v_scale);
      qacc22 = __dtu_m_mop_mul_f32_qa(qacc22, v_scale);
      qacc23 = __dtu_m_mop_mul_f32_qa(qacc23, v_scale);
      qacc24 = __dtu_m_mop_mul_f32_qa(qacc24, v_scale);
      qacc25 = __dtu_m_mop_mul_f32_qa(qacc25, v_scale);
      qacc26 = __dtu_m_mop_mul_f32_qa(qacc26, v_scale);
      qacc27 = __dtu_m_mop_mul_f32_qa(qacc27, v_scale);
      qacc28 = __dtu_m_mop_mul_f32_qa(qacc28, v_scale);
      qacc29 = __dtu_m_mop_mul_f32_qa(qacc29, v_scale);
      qacc30 = __dtu_m_mop_mul_f32_qa(qacc30, v_scale);
      qacc31 = __dtu_m_mop_mul_f32_qa(qacc31, v_scale);
      qacc32 = __dtu_m_mop_mul_f32_qa(qacc32, v_scale);
      qacc33 = __dtu_m_mop_mul_f32_qa(qacc33, v_scale);
      qacc34 = __dtu_m_mop_mul_f32_qa(qacc34, v_scale);
      qacc35 = __dtu_m_mop_mul_f32_qa(qacc35, v_scale);
      qacc36 = __dtu_m_mop_mul_f32_qa(qacc36, v_scale);

      auto da0 = __dtu_extractqa2da(qacc21, 0);
      auto da1 = __dtu_extractqa2da(qacc21, 1);
      auto da2 = __dtu_extractqa2da(qacc22, 0);
      auto da3 = __dtu_extractqa2da(qacc22, 1);
      auto da4 = __dtu_extractqa2da(qacc23, 0);
      auto da5 = __dtu_extractqa2da(qacc23, 1);
      auto da6 = __dtu_extractqa2da(qacc24, 0);
      auto da7 = __dtu_extractqa2da(qacc24, 1);

      auto da8 = __dtu_extractqa2da(qacc25, 0);
      auto da9 = __dtu_extractqa2da(qacc25, 1);
      auto da10 = __dtu_extractqa2da(qacc26, 0);
      auto da11 = __dtu_extractqa2da(qacc26, 1);
      auto da12 = __dtu_extractqa2da(qacc27, 0);
      auto da13 = __dtu_extractqa2da(qacc27, 1);
      auto da14 = __dtu_extractqa2da(qacc28, 0);
      auto da15 = __dtu_extractqa2da(qacc28, 1);

      auto da16 = __dtu_extractqa2da(qacc29, 0);
      auto da17 = __dtu_extractqa2da(qacc29, 1);
      auto da18 = __dtu_extractqa2da(qacc30, 0);
      auto da19 = __dtu_extractqa2da(qacc30, 1);
      auto da20 = __dtu_extractqa2da(qacc31, 0);
      auto da21 = __dtu_extractqa2da(qacc31, 1);
      auto da22 = __dtu_extractqa2da(qacc32, 0);
      auto da23 = __dtu_extractqa2da(qacc32, 1);

      auto da24 = __dtu_extractqa2da(qacc33, 0);
      auto da25 = __dtu_extractqa2da(qacc33, 1);
      auto da26 = __dtu_extractqa2da(qacc34, 0);
      auto da27 = __dtu_extractqa2da(qacc34, 1);
      auto da28 = __dtu_extractqa2da(qacc35, 0);
      auto da29 = __dtu_extractqa2da(qacc35, 1);
      auto da30 = __dtu_extractqa2da(qacc36, 0);
      auto da31 = __dtu_extractqa2da(qacc36, 1);

      __dtu_v_tvstda_f32_dual(da0, ot_base, ot_off0);
      __dtu_v_tvstda_f32_dual(da1, ot_base, ot_off2);
      __dtu_v_tvstda_f32_dual(da2, ot_base, ot_off0);
      __dtu_v_tvstda_f32_dual(da3, ot_base, ot_off2);
      __dtu_v_tvstda_f32_dual(da4, ot_base, ot_off0);
      __dtu_v_tvstda_f32_dual(da5, ot_base, ot_off2);
      __dtu_v_tvstda_f32_dual(da6, ot_base, ot_off0);
      __dtu_v_tvstda_f32_dual(da7, ot_base, ot_off2);

      __dtu_v_tvstda_f32_dual(da8, ot_base, ot_off0);
      __dtu_v_tvstda_f32_dual(da9, ot_base, ot_off2);
      __dtu_v_tvstda_f32_dual(da10, ot_base, ot_off0);
      __dtu_v_tvstda_f32_dual(da11, ot_base, ot_off2);
      __dtu_v_tvstda_f32_dual(da12, ot_base, ot_off0);
      __dtu_v_tvstda_f32_dual(da13, ot_base, ot_off2);
      __dtu_v_tvstda_f32_dual(da14, ot_base, ot_off0);
      __dtu_v_tvstda_f32_dual(da15, ot_base, ot_off2);

      __dtu_v_vpr_tvstda_f32_dual(da16, ot_base, ot_off0);
      __dtu_v_vpr_tvstda_f32_dual(da17, ot_base, ot_off2);
      __dtu_v_vpr_tvstda_f32_dual(da18, ot_base, ot_off0);
      __dtu_v_vpr_tvstda_f32_dual(da19, ot_base, ot_off2);
      __dtu_v_vpr_tvstda_f32_dual(da20, ot_base, ot_off0);
      __dtu_v_vpr_tvstda_f32_dual(da21, ot_base, ot_off2);
      __dtu_v_vpr_tvstda_f32_dual(da22, ot_base, ot_off0);
      __dtu_v_vpr_tvstda_f32_dual(da23, ot_base, ot_off2);

      __dtu_v_vpr_tvstda_f32_dual(da24, ot_base, ot_off0);
      __dtu_v_vpr_tvstda_f32_dual(da25, ot_base, ot_off2);
      __dtu_v_vpr_tvstda_f32_dual(da26, ot_base, ot_off0);
      __dtu_v_vpr_tvstda_f32_dual(da27, ot_base, ot_off2);
      __dtu_v_vpr_tvstda_f32_dual(da28, ot_base, ot_off0);
      __dtu_v_vpr_tvstda_f32_dual(da29, ot_base, ot_off2);
      __dtu_v_vpr_tvstda_f32_dual(da30, ot_base, ot_off0);
      __dtu_v_vpr_tvstda_f32_dual(da31, ot_base, ot_off2);

      ot_base = __dtu_v_taradd(ot_base, ot_off1);

      vab_shift += 128;
      __dtu_c_movsr2vab_lv_s(vab_shift);
      __dtu_c_movsr2vab_m_s1(vab_shift);
      __dtu_c_movsr2vab_m_d(vab_shift);
    }
  }
}

__device__
void call_reduce_pro(tops::half* dst_ptr, float* in_ptr, int dim0,
    int dim1) {
  int out_addr = reinterpret_cast<long>(dst_ptr);
  int in_addr = reinterpret_cast<long>(in_ptr);

  in_addr = in_addr >> 8;
  in_addr = TAR32(in_addr, in_addr + 1);
  tar_t in_tar = __dtu_s_movsr2targ(in_addr);
  tar_t in_offset = __dtu_s_movsr2tari(TAR32(2, 2), in_tar);

  out_addr = out_addr >> 6;
  out_addr = TAR32(out_addr, out_addr + 2);
  tar_t out_tar = __dtu_s_movsr2targ(out_addr);
  tar_t out_offset = __dtu_s_movsr2tari(TAR32(1, 1), out_tar);
  tar_t out_offset2 = __dtu_s_movsr2tari(TAR32(3, 3), out_tar);

  auto qa0 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa1 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa2 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa3 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa4 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa5 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa6 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa7 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa8 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa9 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa10 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa11 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa12 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa13 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa14 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa15 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa16 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa17 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa18 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa19 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa20 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa21 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa22 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa23 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa24 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa25 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa26 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa27 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa28 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa29 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa30 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa31 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

  va16f32x4 qa[32];
  __dtu_c_movsr2mpr(1);

#pragma clang loop unroll(disable)
  for (int i = 0; i < dim0 - 1; i++) {
    qa0 = __dtu_m_mpr_mop_add_f32_qa(qa0, qa[0]);
    qa[0] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa1 = __dtu_m_mpr_mop_add_f32_qa(qa1, qa[1]);
    qa[1] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa2 = __dtu_m_mpr_mop_add_f32_qa(qa2, qa[2]);
    qa[2] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa3 = __dtu_m_mpr_mop_add_f32_qa(qa3, qa[3]);
    qa[3] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa4 = __dtu_m_mpr_mop_add_f32_qa(qa4, qa[4]);
    qa[4] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa5 = __dtu_m_mpr_mop_add_f32_qa(qa5, qa[5]);
    qa[5] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa6 = __dtu_m_mpr_mop_add_f32_qa(qa6, qa[6]);
    qa[6] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa7 = __dtu_m_mpr_mop_add_f32_qa(qa7, qa[7]);
    qa[7] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa8 = __dtu_m_mpr_mop_add_f32_qa(qa8, qa[8]);
    qa[8] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa9 = __dtu_m_mpr_mop_add_f32_qa(qa9, qa[9]);
    qa[9] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa10 = __dtu_m_mpr_mop_add_f32_qa(qa10, qa[10]);
    qa[10] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa11 = __dtu_m_mpr_mop_add_f32_qa(qa11, qa[11]);
    qa[11] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa12 = __dtu_m_mpr_mop_add_f32_qa(qa12, qa[12]);
    qa[12] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa13 = __dtu_m_mpr_mop_add_f32_qa(qa13, qa[13]);
    qa[13] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa14 = __dtu_m_mpr_mop_add_f32_qa(qa14, qa[14]);
    qa[14] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa15 = __dtu_m_mpr_mop_add_f32_qa(qa15, qa[15]);
    qa[15] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa16 = __dtu_m_mpr_mop_add_f32_qa(qa16, qa[16]);
    qa[16] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa17 = __dtu_m_mpr_mop_add_f32_qa(qa17, qa[17]);
    qa[17] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa18 = __dtu_m_mpr_mop_add_f32_qa(qa18, qa[18]);
    qa[18] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa19 = __dtu_m_mpr_mop_add_f32_qa(qa19, qa[19]);
    qa[19] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa20 = __dtu_m_mpr_mop_add_f32_qa(qa20, qa[20]);
    qa[20] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa21 = __dtu_m_mpr_mop_add_f32_qa(qa21, qa[21]);
    qa[21] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa22 = __dtu_m_mpr_mop_add_f32_qa(qa22, qa[22]);
    qa[22] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa23 = __dtu_m_mpr_mop_add_f32_qa(qa23, qa[23]);
    qa[23] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa24 = __dtu_m_mpr_mop_add_f32_qa(qa24, qa[24]);
    qa[24] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa25 = __dtu_m_mpr_mop_add_f32_qa(qa25, qa[25]);
    qa[25] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa26 = __dtu_m_mpr_mop_add_f32_qa(qa26, qa[26]);
    qa[26] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa27 = __dtu_m_mpr_mop_add_f32_qa(qa27, qa[27]);
    qa[27] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa28 = __dtu_m_mpr_mop_add_f32_qa(qa28, qa[28]);
    qa[28] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa29 = __dtu_m_mpr_mop_add_f32_qa(qa29, qa[29]);
    qa[29] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa30 = __dtu_m_mpr_mop_add_f32_qa(qa30, qa[30]);
    qa[30] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa31 = __dtu_m_mpr_mop_add_f32_qa(qa31, qa[31]);
    qa[31] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
    __dtu_c_movsr2mpr(0);
  }

  qa0 = __dtu_m_mpr_mop_add_f32_qa(qa0, qa[0]);
  qa1 = __dtu_m_mpr_mop_add_f32_qa(qa1, qa[1]);
  qa2 = __dtu_m_mpr_mop_add_f32_qa(qa2, qa[2]);
  qa3 = __dtu_m_mpr_mop_add_f32_qa(qa3, qa[3]);
  qa4 = __dtu_m_mpr_mop_add_f32_qa(qa4, qa[4]);
  qa5 = __dtu_m_mpr_mop_add_f32_qa(qa5, qa[5]);
  qa6 = __dtu_m_mpr_mop_add_f32_qa(qa6, qa[6]);
  qa7 = __dtu_m_mpr_mop_add_f32_qa(qa7, qa[7]);
  qa8 = __dtu_m_mpr_mop_add_f32_qa(qa8, qa[8]);
  qa9 = __dtu_m_mpr_mop_add_f32_qa(qa9, qa[9]);
  qa10 = __dtu_m_mpr_mop_add_f32_qa(qa10, qa[10]);
  qa11 = __dtu_m_mpr_mop_add_f32_qa(qa11, qa[11]);
  qa12 = __dtu_m_mpr_mop_add_f32_qa(qa12, qa[12]);
  qa13 = __dtu_m_mpr_mop_add_f32_qa(qa13, qa[13]);
  qa14 = __dtu_m_mpr_mop_add_f32_qa(qa14, qa[14]);
  qa15 = __dtu_m_mpr_mop_add_f32_qa(qa15, qa[15]);
  qa16 = __dtu_m_mpr_mop_add_f32_qa(qa16, qa[16]);
  qa17 = __dtu_m_mpr_mop_add_f32_qa(qa17, qa[17]);
  qa18 = __dtu_m_mpr_mop_add_f32_qa(qa18, qa[18]);
  qa19 = __dtu_m_mpr_mop_add_f32_qa(qa19, qa[19]);
  qa20 = __dtu_m_mpr_mop_add_f32_qa(qa20, qa[20]);
  qa21 = __dtu_m_mpr_mop_add_f32_qa(qa21, qa[21]);
  qa22 = __dtu_m_mpr_mop_add_f32_qa(qa22, qa[22]);
  qa23 = __dtu_m_mpr_mop_add_f32_qa(qa23, qa[23]);
  qa24 = __dtu_m_mpr_mop_add_f32_qa(qa24, qa[24]);
  qa25 = __dtu_m_mpr_mop_add_f32_qa(qa25, qa[25]);
  qa26 = __dtu_m_mpr_mop_add_f32_qa(qa26, qa[26]);
  qa27 = __dtu_m_mpr_mop_add_f32_qa(qa27, qa[27]);
  qa28 = __dtu_m_mpr_mop_add_f32_qa(qa28, qa[28]);
  qa29 = __dtu_m_mpr_mop_add_f32_qa(qa29, qa[29]);
  qa30 = __dtu_m_mpr_mop_add_f32_qa(qa30, qa[30]);
  qa31 = __dtu_m_mpr_mop_add_f32_qa(qa31, qa[31]);

  auto rst_da0 = __dtu_extractqa2da(qa0, 0);
  auto rst_da1 = __dtu_extractqa2da(qa0, 1);
  auto rst_da2 = __dtu_extractqa2da(qa1, 0);
  auto rst_da3 = __dtu_extractqa2da(qa1, 1);
  auto rst_da4 = __dtu_extractqa2da(qa2, 0);
  auto rst_da5 = __dtu_extractqa2da(qa2, 1);
  auto rst_da6 = __dtu_extractqa2da(qa3, 0);
  auto rst_da7 = __dtu_extractqa2da(qa3, 1);
  auto rst_da8 = __dtu_extractqa2da(qa4, 0);
  auto rst_da9 = __dtu_extractqa2da(qa4, 1);
  auto rst_da10 = __dtu_extractqa2da(qa5, 0);
  auto rst_da11 = __dtu_extractqa2da(qa5, 1);
  auto rst_da12 = __dtu_extractqa2da(qa6, 0);
  auto rst_da13 = __dtu_extractqa2da(qa6, 1);
  auto rst_da14 = __dtu_extractqa2da(qa7, 0);
  auto rst_da15 = __dtu_extractqa2da(qa7, 1);

  auto rst_da16 = __dtu_extractqa2da(qa8, 0);
  auto rst_da17 = __dtu_extractqa2da(qa8, 1);
  auto rst_da18 = __dtu_extractqa2da(qa9, 0);
  auto rst_da19 = __dtu_extractqa2da(qa9, 1);
  auto rst_da20 = __dtu_extractqa2da(qa10, 0);
  auto rst_da21 = __dtu_extractqa2da(qa10, 1);
  auto rst_da22 = __dtu_extractqa2da(qa11, 0);
  auto rst_da23 = __dtu_extractqa2da(qa11, 1);
  auto rst_da24 = __dtu_extractqa2da(qa12, 0);
  auto rst_da25 = __dtu_extractqa2da(qa12, 1);
  auto rst_da26 = __dtu_extractqa2da(qa13, 0);
  auto rst_da27 = __dtu_extractqa2da(qa13, 1);
  auto rst_da28 = __dtu_extractqa2da(qa14, 0);
  auto rst_da29 = __dtu_extractqa2da(qa14, 1);
  auto rst_da30 = __dtu_extractqa2da(qa15, 0);
  auto rst_da31 = __dtu_extractqa2da(qa15, 1);

  auto rst_da32 = __dtu_extractqa2da(qa16, 0);
  auto rst_da33 = __dtu_extractqa2da(qa16, 1);
  auto rst_da34 = __dtu_extractqa2da(qa17, 0);
  auto rst_da35 = __dtu_extractqa2da(qa17, 1);
  auto rst_da36 = __dtu_extractqa2da(qa18, 0);
  auto rst_da37 = __dtu_extractqa2da(qa18, 1);
  auto rst_da38 = __dtu_extractqa2da(qa19, 0);
  auto rst_da39 = __dtu_extractqa2da(qa19, 1);
  auto rst_da40 = __dtu_extractqa2da(qa20, 0);
  auto rst_da41 = __dtu_extractqa2da(qa20, 1);
  auto rst_da42 = __dtu_extractqa2da(qa21, 0);
  auto rst_da43 = __dtu_extractqa2da(qa21, 1);
  auto rst_da44 = __dtu_extractqa2da(qa22, 0);
  auto rst_da45 = __dtu_extractqa2da(qa22, 1);
  auto rst_da46 = __dtu_extractqa2da(qa23, 0);
  auto rst_da47 = __dtu_extractqa2da(qa23, 1);

  auto rst_da48 = __dtu_extractqa2da(qa24, 0);
  auto rst_da49 = __dtu_extractqa2da(qa24, 1);
  auto rst_da50 = __dtu_extractqa2da(qa25, 0);
  auto rst_da51 = __dtu_extractqa2da(qa25, 1);
  auto rst_da52 = __dtu_extractqa2da(qa26, 0);
  auto rst_da53 = __dtu_extractqa2da(qa26, 1);
  auto rst_da54 = __dtu_extractqa2da(qa27, 0);
  auto rst_da55 = __dtu_extractqa2da(qa27, 1);
  auto rst_da56 = __dtu_extractqa2da(qa28, 0);
  auto rst_da57 = __dtu_extractqa2da(qa28, 1);
  auto rst_da58 = __dtu_extractqa2da(qa29, 0);
  auto rst_da59 = __dtu_extractqa2da(qa29, 1);
  auto rst_da60 = __dtu_extractqa2da(qa30, 0);
  auto rst_da61 = __dtu_extractqa2da(qa30, 1);
  auto rst_da62 = __dtu_extractqa2da(qa31, 0);
  auto rst_da63 = __dtu_extractqa2da(qa31, 1);

  __dtu_l_tvsta_cvt2fp16(rst_da0, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da1, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2fp16(rst_da2, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da3, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2fp16(rst_da4, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da5, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2fp16(rst_da6, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da7, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2fp16(rst_da8, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da9, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2fp16(rst_da10, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da11, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2fp16(rst_da12, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da13, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2fp16(rst_da14, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da15, out_tar, out_offset2);

  __dtu_l_tvsta_cvt2fp16(rst_da16, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da17, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2fp16(rst_da18, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da19, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2fp16(rst_da20, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da21, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2fp16(rst_da22, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da23, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2fp16(rst_da24, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da25, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2fp16(rst_da26, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da27, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2fp16(rst_da28, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da29, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2fp16(rst_da30, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da31, out_tar, out_offset2);

  __dtu_l_tvsta_cvt2fp16(rst_da32, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da33, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2fp16(rst_da34, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da35, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2fp16(rst_da36, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da37, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2fp16(rst_da38, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da39, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2fp16(rst_da40, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da41, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2fp16(rst_da42, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da43, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2fp16(rst_da44, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da45, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2fp16(rst_da46, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da47, out_tar, out_offset2);

  __dtu_l_tvsta_cvt2fp16(rst_da48, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da49, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2fp16(rst_da50, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da51, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2fp16(rst_da52, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da53, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2fp16(rst_da54, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da55, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2fp16(rst_da56, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da57, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2fp16(rst_da58, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da59, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2fp16(rst_da60, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da61, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2fp16(rst_da62, out_tar, out_offset);
  __dtu_l_tvsta_cvt2fp16(rst_da63, out_tar, out_offset2);
}

__device__
void call_reduce_pro(tops::bfloat* dst_ptr, float* in_ptr, int dim0,
    int dim1) {
  int out_addr = reinterpret_cast<long>(dst_ptr);
  int in_addr = reinterpret_cast<long>(in_ptr);

  in_addr = in_addr >> 8;
  in_addr = TAR32(in_addr, in_addr + 1);
  tar_t in_tar = __dtu_s_movsr2targ(in_addr);
  tar_t in_offset = __dtu_s_movsr2tari(TAR32(2, 2), in_tar);

  out_addr = out_addr >> 6;
  out_addr = TAR32(out_addr, out_addr + 2);
  tar_t out_tar = __dtu_s_movsr2targ(out_addr);
  tar_t out_offset = __dtu_s_movsr2tari(TAR32(1, 1), out_tar);
  tar_t out_offset2 = __dtu_s_movsr2tari(TAR32(3, 3), out_tar);

  auto qa0 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa1 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa2 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa3 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa4 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa5 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa6 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa7 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa8 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa9 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa10 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa11 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa12 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa13 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa14 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa15 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa16 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa17 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa18 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa19 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa20 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa21 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa22 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa23 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa24 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa25 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa26 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa27 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa28 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa29 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa30 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
  auto qa31 = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

  va16f32x4 qa[32];
  __dtu_c_movsr2mpr(1);

#pragma clang loop unroll(disable)
  for (int i = 0; i < dim0 - 1; i++) {
    qa0 = __dtu_m_mpr_mop_add_f32_qa(qa0, qa[0]);
    qa[0] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa1 = __dtu_m_mpr_mop_add_f32_qa(qa1, qa[1]);
    qa[1] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa2 = __dtu_m_mpr_mop_add_f32_qa(qa2, qa[2]);
    qa[2] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa3 = __dtu_m_mpr_mop_add_f32_qa(qa3, qa[3]);
    qa[3] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa4 = __dtu_m_mpr_mop_add_f32_qa(qa4, qa[4]);
    qa[4] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa5 = __dtu_m_mpr_mop_add_f32_qa(qa5, qa[5]);
    qa[5] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa6 = __dtu_m_mpr_mop_add_f32_qa(qa6, qa[6]);
    qa[6] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa7 = __dtu_m_mpr_mop_add_f32_qa(qa7, qa[7]);
    qa[7] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa8 = __dtu_m_mpr_mop_add_f32_qa(qa8, qa[8]);
    qa[8] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa9 = __dtu_m_mpr_mop_add_f32_qa(qa9, qa[9]);
    qa[9] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa10 = __dtu_m_mpr_mop_add_f32_qa(qa10, qa[10]);
    qa[10] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa11 = __dtu_m_mpr_mop_add_f32_qa(qa11, qa[11]);
    qa[11] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa12 = __dtu_m_mpr_mop_add_f32_qa(qa12, qa[12]);
    qa[12] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa13 = __dtu_m_mpr_mop_add_f32_qa(qa13, qa[13]);
    qa[13] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa14 = __dtu_m_mpr_mop_add_f32_qa(qa14, qa[14]);
    qa[14] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa15 = __dtu_m_mpr_mop_add_f32_qa(qa15, qa[15]);
    qa[15] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa16 = __dtu_m_mpr_mop_add_f32_qa(qa16, qa[16]);
    qa[16] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa17 = __dtu_m_mpr_mop_add_f32_qa(qa17, qa[17]);
    qa[17] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa18 = __dtu_m_mpr_mop_add_f32_qa(qa18, qa[18]);
    qa[18] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa19 = __dtu_m_mpr_mop_add_f32_qa(qa19, qa[19]);
    qa[19] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa20 = __dtu_m_mpr_mop_add_f32_qa(qa20, qa[20]);
    qa[20] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa21 = __dtu_m_mpr_mop_add_f32_qa(qa21, qa[21]);
    qa[21] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa22 = __dtu_m_mpr_mop_add_f32_qa(qa22, qa[22]);
    qa[22] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa23 = __dtu_m_mpr_mop_add_f32_qa(qa23, qa[23]);
    qa[23] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa24 = __dtu_m_mpr_mop_add_f32_qa(qa24, qa[24]);
    qa[24] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa25 = __dtu_m_mpr_mop_add_f32_qa(qa25, qa[25]);
    qa[25] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa26 = __dtu_m_mpr_mop_add_f32_qa(qa26, qa[26]);
    qa[26] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa27 = __dtu_m_mpr_mop_add_f32_qa(qa27, qa[27]);
    qa[27] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa28 = __dtu_m_mpr_mop_add_f32_qa(qa28, qa[28]);
    qa[28] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa29 = __dtu_m_mpr_mop_add_f32_qa(qa29, qa[29]);
    qa[29] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa30 = __dtu_m_mpr_mop_add_f32_qa(qa30, qa[30]);
    qa[30] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);

    qa31 = __dtu_m_mpr_mop_add_f32_qa(qa31, qa[31]);
    qa[31] = __dtu_l_tvldqa_f32_qa(in_tar, in_offset);
    __dtu_c_movsr2mpr(0);
  }

  qa0 = __dtu_m_mpr_mop_add_f32_qa(qa0, qa[0]);
  qa1 = __dtu_m_mpr_mop_add_f32_qa(qa1, qa[1]);
  qa2 = __dtu_m_mpr_mop_add_f32_qa(qa2, qa[2]);
  qa3 = __dtu_m_mpr_mop_add_f32_qa(qa3, qa[3]);
  qa4 = __dtu_m_mpr_mop_add_f32_qa(qa4, qa[4]);
  qa5 = __dtu_m_mpr_mop_add_f32_qa(qa5, qa[5]);
  qa6 = __dtu_m_mpr_mop_add_f32_qa(qa6, qa[6]);
  qa7 = __dtu_m_mpr_mop_add_f32_qa(qa7, qa[7]);
  qa8 = __dtu_m_mpr_mop_add_f32_qa(qa8, qa[8]);
  qa9 = __dtu_m_mpr_mop_add_f32_qa(qa9, qa[9]);
  qa10 = __dtu_m_mpr_mop_add_f32_qa(qa10, qa[10]);
  qa11 = __dtu_m_mpr_mop_add_f32_qa(qa11, qa[11]);
  qa12 = __dtu_m_mpr_mop_add_f32_qa(qa12, qa[12]);
  qa13 = __dtu_m_mpr_mop_add_f32_qa(qa13, qa[13]);
  qa14 = __dtu_m_mpr_mop_add_f32_qa(qa14, qa[14]);
  qa15 = __dtu_m_mpr_mop_add_f32_qa(qa15, qa[15]);
  qa16 = __dtu_m_mpr_mop_add_f32_qa(qa16, qa[16]);
  qa17 = __dtu_m_mpr_mop_add_f32_qa(qa17, qa[17]);
  qa18 = __dtu_m_mpr_mop_add_f32_qa(qa18, qa[18]);
  qa19 = __dtu_m_mpr_mop_add_f32_qa(qa19, qa[19]);
  qa20 = __dtu_m_mpr_mop_add_f32_qa(qa20, qa[20]);
  qa21 = __dtu_m_mpr_mop_add_f32_qa(qa21, qa[21]);
  qa22 = __dtu_m_mpr_mop_add_f32_qa(qa22, qa[22]);
  qa23 = __dtu_m_mpr_mop_add_f32_qa(qa23, qa[23]);
  qa24 = __dtu_m_mpr_mop_add_f32_qa(qa24, qa[24]);
  qa25 = __dtu_m_mpr_mop_add_f32_qa(qa25, qa[25]);
  qa26 = __dtu_m_mpr_mop_add_f32_qa(qa26, qa[26]);
  qa27 = __dtu_m_mpr_mop_add_f32_qa(qa27, qa[27]);
  qa28 = __dtu_m_mpr_mop_add_f32_qa(qa28, qa[28]);
  qa29 = __dtu_m_mpr_mop_add_f32_qa(qa29, qa[29]);
  qa30 = __dtu_m_mpr_mop_add_f32_qa(qa30, qa[30]);
  qa31 = __dtu_m_mpr_mop_add_f32_qa(qa31, qa[31]);

  auto rst_da0 = __dtu_extractqa2da(qa0, 0);
  auto rst_da1 = __dtu_extractqa2da(qa0, 1);
  auto rst_da2 = __dtu_extractqa2da(qa1, 0);
  auto rst_da3 = __dtu_extractqa2da(qa1, 1);
  auto rst_da4 = __dtu_extractqa2da(qa2, 0);
  auto rst_da5 = __dtu_extractqa2da(qa2, 1);
  auto rst_da6 = __dtu_extractqa2da(qa3, 0);
  auto rst_da7 = __dtu_extractqa2da(qa3, 1);
  auto rst_da8 = __dtu_extractqa2da(qa4, 0);
  auto rst_da9 = __dtu_extractqa2da(qa4, 1);
  auto rst_da10 = __dtu_extractqa2da(qa5, 0);
  auto rst_da11 = __dtu_extractqa2da(qa5, 1);
  auto rst_da12 = __dtu_extractqa2da(qa6, 0);
  auto rst_da13 = __dtu_extractqa2da(qa6, 1);
  auto rst_da14 = __dtu_extractqa2da(qa7, 0);
  auto rst_da15 = __dtu_extractqa2da(qa7, 1);

  auto rst_da16 = __dtu_extractqa2da(qa8, 0);
  auto rst_da17 = __dtu_extractqa2da(qa8, 1);
  auto rst_da18 = __dtu_extractqa2da(qa9, 0);
  auto rst_da19 = __dtu_extractqa2da(qa9, 1);
  auto rst_da20 = __dtu_extractqa2da(qa10, 0);
  auto rst_da21 = __dtu_extractqa2da(qa10, 1);
  auto rst_da22 = __dtu_extractqa2da(qa11, 0);
  auto rst_da23 = __dtu_extractqa2da(qa11, 1);
  auto rst_da24 = __dtu_extractqa2da(qa12, 0);
  auto rst_da25 = __dtu_extractqa2da(qa12, 1);
  auto rst_da26 = __dtu_extractqa2da(qa13, 0);
  auto rst_da27 = __dtu_extractqa2da(qa13, 1);
  auto rst_da28 = __dtu_extractqa2da(qa14, 0);
  auto rst_da29 = __dtu_extractqa2da(qa14, 1);
  auto rst_da30 = __dtu_extractqa2da(qa15, 0);
  auto rst_da31 = __dtu_extractqa2da(qa15, 1);

  auto rst_da32 = __dtu_extractqa2da(qa16, 0);
  auto rst_da33 = __dtu_extractqa2da(qa16, 1);
  auto rst_da34 = __dtu_extractqa2da(qa17, 0);
  auto rst_da35 = __dtu_extractqa2da(qa17, 1);
  auto rst_da36 = __dtu_extractqa2da(qa18, 0);
  auto rst_da37 = __dtu_extractqa2da(qa18, 1);
  auto rst_da38 = __dtu_extractqa2da(qa19, 0);
  auto rst_da39 = __dtu_extractqa2da(qa19, 1);
  auto rst_da40 = __dtu_extractqa2da(qa20, 0);
  auto rst_da41 = __dtu_extractqa2da(qa20, 1);
  auto rst_da42 = __dtu_extractqa2da(qa21, 0);
  auto rst_da43 = __dtu_extractqa2da(qa21, 1);
  auto rst_da44 = __dtu_extractqa2da(qa22, 0);
  auto rst_da45 = __dtu_extractqa2da(qa22, 1);
  auto rst_da46 = __dtu_extractqa2da(qa23, 0);
  auto rst_da47 = __dtu_extractqa2da(qa23, 1);

  auto rst_da48 = __dtu_extractqa2da(qa24, 0);
  auto rst_da49 = __dtu_extractqa2da(qa24, 1);
  auto rst_da50 = __dtu_extractqa2da(qa25, 0);
  auto rst_da51 = __dtu_extractqa2da(qa25, 1);
  auto rst_da52 = __dtu_extractqa2da(qa26, 0);
  auto rst_da53 = __dtu_extractqa2da(qa26, 1);
  auto rst_da54 = __dtu_extractqa2da(qa27, 0);
  auto rst_da55 = __dtu_extractqa2da(qa27, 1);
  auto rst_da56 = __dtu_extractqa2da(qa28, 0);
  auto rst_da57 = __dtu_extractqa2da(qa28, 1);
  auto rst_da58 = __dtu_extractqa2da(qa29, 0);
  auto rst_da59 = __dtu_extractqa2da(qa29, 1);
  auto rst_da60 = __dtu_extractqa2da(qa30, 0);
  auto rst_da61 = __dtu_extractqa2da(qa30, 1);
  auto rst_da62 = __dtu_extractqa2da(qa31, 0);
  auto rst_da63 = __dtu_extractqa2da(qa31, 1);

  __dtu_l_tvsta_cvt2bf16(rst_da0, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da1, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2bf16(rst_da2, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da3, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2bf16(rst_da4, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da5, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2bf16(rst_da6, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da7, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2bf16(rst_da8, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da9, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2bf16(rst_da10, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da11, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2bf16(rst_da12, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da13, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2bf16(rst_da14, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da15, out_tar, out_offset2);

  __dtu_l_tvsta_cvt2bf16(rst_da16, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da17, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2bf16(rst_da18, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da19, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2bf16(rst_da20, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da21, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2bf16(rst_da22, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da23, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2bf16(rst_da24, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da25, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2bf16(rst_da26, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da27, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2bf16(rst_da28, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da29, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2bf16(rst_da30, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da31, out_tar, out_offset2);

  __dtu_l_tvsta_cvt2bf16(rst_da32, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da33, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2bf16(rst_da34, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da35, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2bf16(rst_da36, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da37, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2bf16(rst_da38, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da39, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2bf16(rst_da40, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da41, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2bf16(rst_da42, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da43, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2bf16(rst_da44, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da45, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2bf16(rst_da46, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da47, out_tar, out_offset2);

  __dtu_l_tvsta_cvt2bf16(rst_da48, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da49, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2bf16(rst_da50, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da51, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2bf16(rst_da52, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da53, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2bf16(rst_da54, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da55, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2bf16(rst_da56, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da57, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2bf16(rst_da58, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da59, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2bf16(rst_da60, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da61, out_tar, out_offset2);
  __dtu_l_tvsta_cvt2bf16(rst_da62, out_tar, out_offset);
  __dtu_l_tvsta_cvt2bf16(rst_da63, out_tar, out_offset2);
}

// template <typename T>
// __device__ constexpr int padding_inf() { return 0xfc00; }
// template <>
// __device__ constexpr int padding_inf<tops::bfloat>() { return 0xff80; }

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
__device__ void paged_attention_gqa_kernel(
    T *output,
    T *query, T *key_cache, T *value_cache,
    // int *head_mapping,
    float scale,
    int *block_tables, int *context_lens,
    int block_size, int max_context_len,
    float *alibi_slopes,
    int num_seqs, int num_heads, int head_size,
    int num_blocks, int num_kv_heads,
    int max_num_blocks_per_seq, int stride, int alibi_enable, float softcapping, char* buffer_sip) {
// krt_reset_clock();
  int block_id = GetBlockIdx();
  int block_num = GetBlockNum();
  int thread_id = GetThreadIdxInBlock();
  int thread_num = GetThreadNumEachBlock();

  int q_head_num = num_heads;
  int head_num_pro = q_head_num;
  int seq_start = block_id;
  int seq_step = block_num;
  if (num_seqs == 1) {
    seq_start = 0;
    seq_step = 1;
    q_head_num = (num_heads + block_num - 1) / block_num;
    head_num_pro = q_head_num;
    if ((block_id + 1) == block_num) {
      head_num_pro = num_heads - block_id * q_head_num;
    }
  }

  int bpe = sizeof(T);
  int f32_bpe = sizeof(float);
  int sip_offset = 0;
  T *p_q = reinterpret_cast<T*>(buffer_sip);
  sip_offset += ALIGN_16(q_head_num) * head_size * bpe;

  int *p_block_tables = reinterpret_cast<int*>(buffer_sip + sip_offset);
  sip_offset += (max_num_blocks_per_seq) * sizeof(int);

  int *p_context_len = reinterpret_cast<int*>(buffer_sip + sip_offset);
  sip_offset += (num_seqs) * sizeof(int);

  float *p_alibi_slopes = reinterpret_cast<float*>(buffer_sip + sip_offset);
  sip_offset += ALIGN_16(num_heads) * sizeof(float);
  sip_offset = ALIGN_256(sip_offset);

  const int min_kv_space = 256 * 1024;
  T *p_key = reinterpret_cast<T*>(buffer_sip + sip_offset);
  T *p_val = (p_key);
  float *p_reduce = reinterpret_cast<float*>(p_key);

  int block_size_per_sip = (max_context_len + thread_num - 1) / thread_num;
  block_size_per_sip = ALIGN_128(block_size_per_sip);
  int kv_space = head_size * block_size_per_sip * bpe;
  sip_offset += (kv_space < min_kv_space) ? min_kv_space : kv_space;

  float *p_attn = reinterpret_cast<float*>(buffer_sip + sip_offset);
  int small_off = sip_offset + ALIGN_16(q_head_num) * SMALL_SIZE * f32_bpe;
  sip_offset += ALIGN_32(q_head_num) * block_size_per_sip * f32_bpe;

  T *p_out = reinterpret_cast<T*>(buffer_sip + sip_offset);
  float *p_out32 = reinterpret_cast<float*>(buffer_sip + sip_offset);
  sip_offset += ALIGN_32(q_head_num) * head_size * sizeof(float);

  T *p_out_small = reinterpret_cast<T*>(buffer_sip + small_off);
  float *p_out32_small = reinterpret_cast<float*>(buffer_sip + small_off);

  float *p_max = reinterpret_cast<float*>(buffer_sip + sip_offset);
  sip_offset += ALIGN_32(q_head_num) * 64 * 2;

  tops_dte_ctx_t ctx_s2c;
  // tops::dte_scope s_s2c(ctx_s2c);

  extern  __shared__ __valigned__ char buf_l2[];
  float *p_l2_out = reinterpret_cast<float*>(buf_l2);

  //  Q
  tops_dte_ctx_t ctxs_q;
  // tops::dte_scope s_q(ctxs_q);
  int32_t query_shape[] = {1, head_num_pro, head_size};
  tops::mdspan l1_query(tops::Private, p_q, query_shape);

  // context len
  tops::mdspan l3_cl(tops::Global, context_lens, num_seqs);
  tops::mdspan l1_cl(tops::Private, p_context_len, num_seqs);
  tops::memcpy(ctxs_q, l1_cl, l3_cl);

  if (alibi_enable == 1) {
    int alibi_off = 0;
    if (num_seqs == 1) {
      alibi_off = block_id * q_head_num;
    }
    tops::mdspan l3_slopes(tops::Global, alibi_slopes + alibi_off,
                           head_num_pro);
    tops::mdspan l1_slopes(tops::Private, p_alibi_slopes, head_num_pro);
    tops::memcpy(ctxs_q, l1_slopes, l3_slopes);
  }

  // Key
  tops_dte_ctx_t ctxs_k[MAX_WAIT_NUM];
  tops::event evs_k[MAX_WAIT_NUM];
  // for (int i = 0; i < MAX_WAIT_NUM; i++) {
  //   ctxs_k[i].init();
  // }

  int32_t kv_cache_shape[] = {1, 1, block_size, head_size};
  tops::mdspan l3_key(tops::Global, key_cache, kv_cache_shape);
  tops::mdspan l1_key(tops::Private, p_key, kv_cache_shape);

  tops::mdspan l3_val(tops::Global, value_cache, kv_cache_shape);
  tops::mdspan l1_val(tops::Private, p_val, kv_cache_shape);

  for (int i = 0; i < MAX_WAIT_NUM; i++) {
    ctxs_k[i].config_memcpy(l1_key, l3_key);
  }

  // small size
  int start_idx = block_num * thread_id + block_id;
  int step = block_num * thread_num;
  if (num_seqs == 1) {
    start_idx = 0;
    step = 1;
  }
  for (int seq_idx = start_idx; seq_idx < num_seqs; seq_idx += step) {
    int ctx_len = p_context_len[seq_idx];
    if ((ctx_len <= 0) || (ctx_len > SMALL_SIZE)) continue;
    ctx_len = (ctx_len > max_context_len) ? max_context_len : ctx_len;
    int num_blocks_ = (ctx_len + block_size - 1) / block_size;

    int kv_head_per_block = num_kv_heads;
    int kv_block_offset = 0;
    int head_per_kv = num_heads / kv_head_per_block;
    if (num_seqs == 1) {
      kv_head_per_block = num_kv_heads / block_num;
      kv_block_offset = block_id * kv_head_per_block * head_size * block_size;
      head_per_kv = head_num_pro / kv_head_per_block;
    }

    auto p_query = query + seq_idx * stride;
    int q_offset = 0;
    if (num_seqs == 1) {
      q_offset = block_id * q_head_num * head_size;
    }
    tops::mdspan l3_query(tops::Global, p_query + q_offset, query_shape);
    tops::memcpy(ctxs_q, l1_query, l3_query);

    // block table
    int32_t block_tables_shape[] = {1, max_num_blocks_per_seq};
    tops::mdspan l3_bt(tops::Global,
        block_tables + seq_idx * max_num_blocks_per_seq, block_tables_shape);
    tops::mdspan l1_bt(tops::Private, p_block_tables, block_tables_shape);
    tops::memcpy(ctxs_q, l1_bt, l3_bt);

    // int M = num_heads;
    int N = num_blocks_ * block_size;
    N = ALIGN_128(N);

    for (int j = 0; j < kv_head_per_block; j++) {
      auto p_key_dst = p_key;
      int wait_times = 0;

      for (int i = 0; i < num_blocks_; i++) {
        int kv_block_idx = p_block_tables[i];
        auto p_key_src =
            key_cache + kv_block_idx * num_kv_heads * head_size * block_size;
        p_key_src += kv_block_offset;
        p_key_src += j * head_size * block_size;

        ctxs_k[wait_times].set_src_addr(p_key_src);
        ctxs_k[wait_times].set_dst_addr(p_key_dst);
        evs_k[wait_times] = ctxs_k[wait_times].trigger();
        wait_times++;
        p_key_dst += head_size * block_size;
        if (wait_times >= MAX_WAIT_NUM) {
          for (int i = 0; i < MAX_WAIT_NUM; i++) { evs_k[i].wait(); }
          wait_times = 0;
        }
      }

      for (int i = 0; i < wait_times; i++) {
        evs_k[i].wait();
      }

      // attn = q @ key
      call_k_dot_nk(p_attn + j * head_per_kv * N,
                    p_q + j * head_per_kv * head_size, p_key, head_per_kv,
                    ALIGN_32(head_size), (N), scale);
    }

    // softmax
    if (alibi_enable == 1) {
      call_alibi(p_attn, p_attn,
                 p_alibi_slopes, head_num_pro, N, ctx_len);
    }
    call_softmax_vr(p_attn, p_attn, head_num_pro, N, ctx_len);

    // softmax @ val
    for (int j = 0; j < kv_head_per_block; j++) {
      auto p_val_dst = p_val;
      int wait_times = 0;

      for (int i = 0; i < num_blocks_; i++) {
        int kv_block_idx = p_block_tables[i];
        auto p_val_src = value_cache +
            kv_block_idx * num_kv_heads * head_size * block_size;
        p_val_src += kv_block_offset;
        p_val_src += j * head_size * block_size;

        ctxs_k[wait_times].set_src_addr(p_val_src);
        ctxs_k[wait_times].set_dst_addr(p_val_dst);
        evs_k[wait_times] = ctxs_k[wait_times].trigger();
        wait_times++;
        p_val_dst += head_size * block_size;
        if (wait_times >= MAX_WAIT_NUM) {
          for (int i = 0; i < MAX_WAIT_NUM; i++) { evs_k[i].wait(); }
          wait_times = 0;
        }
      }

      for (int i = 0; i < wait_times; i++) {
        evs_k[i].wait();
      }

      int M = head_per_kv;
      int K = num_blocks_ * block_size;
      K = ALIGN_128(K);

      call_k_dot_kn(p_out32_small + j * head_per_kv * head_size / 2,
                    p_attn + j * head_per_kv * K, p_val,
                    M, K, ALIGN_128(head_size),
                    ctx_len, 0);
    }

    // s2d
    auto p_output = output + seq_idx * num_heads * head_size;
    int out_offset = 0;
    if (num_seqs == 1) {
      out_offset = block_id * q_head_num * head_size;
    }
    tops::mdspan l3_out(
        tops::Global, p_output + out_offset, head_num_pro * head_size);
    tops::mdspan l1_out(tops::Private, p_out_small, head_num_pro * head_size);
    tops::memcpy(ctx_s2c, l3_out, l1_out);
  }

  // large size
  for (int seq_idx = seq_start; seq_idx < num_seqs; seq_idx += seq_step) {
    int ctx_len = p_context_len[seq_idx];
    if (ctx_len <= SMALL_SIZE) continue;
    ctx_len = (ctx_len > max_context_len) ? max_context_len : ctx_len;
    int num_blocks_ = (ctx_len + block_size - 1) / block_size;

    // int32_t l2_shape[] = {q_head_num, num_blocks_ * block_size};
    // tops::mdspan l2_attn(tops::Shared, p_l2, l2_shape);

    int block_per_sip = (num_blocks_ + thread_num - 1) / thread_num;
    // to make N align to 128 ele
    block_per_sip = ALIGN_128(block_per_sip * block_size) / block_size;
    int block_start = block_per_sip * thread_id;
    int block_end = block_start + block_per_sip;
    int kv_head_per_block = num_kv_heads;
    int kv_block_offset = 0;
    int head_per_kv = head_num_pro / kv_head_per_block;
    if (num_seqs == 1) {
      kv_head_per_block = num_kv_heads / block_num;
      kv_block_offset = block_id * kv_head_per_block * head_size * block_size;
      head_per_kv = head_num_pro / kv_head_per_block;
    }

    if (block_start < num_blocks_) {
      auto p_query = query + seq_idx * stride;
      int q_offset = 0;
      if (num_seqs == 1) {
        q_offset = block_id * q_head_num * head_size;
      }
      tops::mdspan l3_query(tops::Global, p_query + q_offset, query_shape);
      tops::memcpy(ctxs_q, l1_query, l3_query);

      // block table
      int32_t block_tables_shape[] = {1, max_num_blocks_per_seq};
      tops::mdspan l3_bt(tops::Global,
          block_tables + seq_idx * max_num_blocks_per_seq, block_tables_shape);
      tops::mdspan l1_bt(tops::Private, p_block_tables, block_tables_shape);
      tops::memcpy(ctxs_q, l1_bt, l3_bt);

      block_end = (block_end > num_blocks_) ? num_blocks_ : block_end;
      int N = (block_end - block_start) * block_size;
      N = ALIGN_128(N);
      int left_ctx_len = ctx_len - block_start * block_size;

      for (int j = 0; j < kv_head_per_block; j++) {
        auto p_key_dst = p_key;
        int wait_times = 0;

        for (int i = block_start; i < block_end; i++) {
          int kv_block_idx = p_block_tables[i];
          auto p_key_src =
              key_cache + kv_block_idx * num_kv_heads * head_size * block_size;
          p_key_src += kv_block_offset;
          p_key_src += j * head_size * block_size;

          ctxs_k[wait_times].set_src_addr(p_key_src);
          ctxs_k[wait_times].set_dst_addr(p_key_dst);
          evs_k[wait_times] = ctxs_k[wait_times].trigger();
          wait_times++;
          p_key_dst += head_size * block_size;
          if (wait_times >= MAX_WAIT_NUM) {
            for (int i = 0; i < MAX_WAIT_NUM; i++) { evs_k[i].wait(); }
            wait_times = 0;
          }
        }

        for (int i = 0; i < wait_times; i++) {
          evs_k[i].wait();
        }

        // attn = q @ key
        int M = head_per_kv;
        call_k_dot_nk(p_attn + j * head_per_kv * N,
            p_q + j * head_per_kv * head_size,
            p_key, M,
            ALIGN_64(head_size), N, scale);
      }

      // softmax
      if (alibi_enable == 1) {
        call_alibi(p_attn, p_attn,
                   p_alibi_slopes, num_heads, N, left_ctx_len);
      }
      call_softmax_qa(p_attn, p_attn, p_max,
                      ALIGN_32(num_heads), N, left_ctx_len);

      int K = (block_end - block_start) * block_size;
      for (int j = 0; j < kv_head_per_block; j++) {
        auto p_val_dst = p_val;
        int wait_times = 0;

        for (int i = block_start; i < block_end; i++) {
          int kv_block_idx = p_block_tables[i];
          auto p_val_src = value_cache +
              kv_block_idx * num_kv_heads * head_size * block_size;
          p_val_src += kv_block_offset;
          p_val_src += j * head_size * block_size;

          ctxs_k[wait_times].set_src_addr(p_val_src);
          ctxs_k[wait_times].set_dst_addr(p_val_dst);
          evs_k[wait_times] = ctxs_k[wait_times].trigger();
          wait_times++;
          p_val_dst += head_size * block_size;
          if (wait_times >= MAX_WAIT_NUM) {
            for (int i = 0; i < MAX_WAIT_NUM; i++) { evs_k[i].wait(); }
            wait_times = 0;
          }
        }

        for (int i = 0; i < wait_times; i++) {
          evs_k[i].wait();
        }

        call_k_dot_kn(p_out32 + j * head_per_kv * head_size,
            p_attn + j * head_per_kv * ALIGN_128(K),
            p_val,
            head_per_kv, ALIGN_128(K), head_size,
            left_ctx_len, 1);
      }

      // s2c
      auto offset = thread_id * ALIGN_32(num_heads) * 2;
      int out_shape = ALIGN_32(num_heads) * 2;
      tops::mdspan l2_out(
          tops::Shared, p_l2_out + offset, out_shape, 1);
      tops::mdspan l1_out(tops::Private, p_max, out_shape, 16);
      tops::deslice(ctx_s2c, l2_out, l1_out, {0, 0});
    }

    __syncthreads();
    // max
    if (block_start < num_blocks_) {
      int thread_used = (num_blocks_ + block_per_sip - 1) / block_per_sip;
      int shape = thread_used * ALIGN_32(num_heads) * 2;
      tops::mdspan l2_reduce(tops::Shared, p_l2_out, shape);
      tops::mdspan l1_reduce(tops::Private, p_reduce, shape);
      tops::memcpy(ctx_s2c, l1_reduce, l2_reduce);

      auto sum_offset = thread_id * ALIGN_32(num_heads) * 2 +
                        ALIGN_32(num_heads);
      call_get_max(p_reduce, p_reduce, thread_used, ALIGN_32(num_heads));

      call_update_softmax(p_reduce + sum_offset,
          p_out32, ALIGN_16(q_head_num), head_size);

      // s2c
      int offset = thread_id * ALIGN_32(num_heads) * head_size;
      int out_shape = q_head_num * head_size;
      tops::mdspan l2_out(
          tops::Shared, p_l2_out + offset, out_shape);
      tops::mdspan l1_out(tops::Private, p_out32, out_shape);
      tops::memcpy(ctx_s2c, l2_out, l1_out);
    }

    __syncthreads();
    // reduce
    if (thread_id == 0) {
      // output
      auto p_output = output + seq_idx * num_heads * head_size;
      int out_offset = 0;
      if (num_seqs == 1) {
        out_offset = block_id * q_head_num * head_size;
      }
      tops::mdspan l3_out(
          tops::Global, p_output + out_offset, head_num_pro * head_size);

      int thread_used = (num_blocks_ + block_per_sip - 1) / block_per_sip;
      int shape = thread_used * ALIGN_32(num_heads) * head_size;
      tops::mdspan l2_reduce(tops::Shared, p_l2_out, shape);
      tops::mdspan l1_reduce(tops::Private, p_reduce, shape);
      tops::memcpy(ctx_s2c, l1_reduce, l2_reduce);

      call_reduce_pro(p_out, p_reduce,
          thread_used, ALIGN_32(num_heads) * head_size);

      tops::mdspan l1_out(tops::Private, p_out, head_num_pro * head_size);
      tops::memcpy(ctx_s2c, l3_out, l1_out);
    }
  }
}

