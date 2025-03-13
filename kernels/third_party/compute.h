/**
 * Copyright 2023 Enflame. All Rights Reserved.
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
#ifndef CC_KERNEL_TOPK_COMPUTE_H_
#define CC_KERNEL_TOPK_COMPUTE_H_

#include <tops.h>
#include <tops/tops_runtime.h>
#include <tops/bfloat.h>
#include <tops/half.h>

#include <algorithm>
#include <limits>
#include <type_traits>

#include "tcle.h"
#include "bit_cast.h"

using namespace tops;

template <int VECTOR_LENGTH>
struct scalar_to_vector<half, VECTOR_LENGTH> {
  using type = typename scalar_to_vector<__fp16, VECTOR_LENGTH>::type;
};

template <int VECTOR_LENGTH>
struct scalar_to_vector<bfloat, VECTOR_LENGTH> {
  using type = typename scalar_to_vector<__bf16, VECTOR_LENGTH>::type;
};

template <int VECTOR_LENGTH>
struct tcle::altivector<half, VECTOR_LENGTH> {
  // using in_ctype =
  //     typename tops::scalar_to_cpp_type<tops::ScalarType::kFloat16>::type;
  using VT = typename altivector<__fp16, VECTOR_LENGTH>::VT;
};

template <int VECTOR_LENGTH>
struct tcle::altivector<bfloat, VECTOR_LENGTH> {
  // using in_ctype =
  //     typename tops::scalar_to_cpp_type<tops::ScalarType::kBfloat16>::type;
  using VT = typename altivector<__bf16, VECTOR_LENGTH>::VT;
};

template <int VECTOR_LENGTH>
struct tcle::altivector<int8_t, VECTOR_LENGTH> {
  // using in_ctype =
  //     typename tops::scalar_to_cpp_type<tops::ScalarType::kInt8>::type;
  using VT = typename altivector<char, VECTOR_LENGTH>::VT;
};

template <typename T>
struct dtype_mapping {
  using type = T;
};

template <>
struct dtype_mapping<tops::half> {
  using type = __fp16;
};

template <>
struct dtype_mapping<tops::bfloat> {
  using type = __bf16;
};

template <bool DESCENDING, typename VType, typename MT>
__device__ __forceinline__ MT kernel_cmp0(VType vlhs, VType vrhs);

template <bool DESCENDING, typename VType, typename MT>
__device__ __forceinline__ MT kernel_cmp1(VType vlhs, VType vrhs);

template <bool DESCENDING, typename VType>
__device__ __forceinline__ VType kernel_getV0(VType vlhs, VType vrhs);

template <bool DESCENDING, typename VType>
__device__ __forceinline__ VType kernel_getV1(VType vlhs, VType vrhs);

template <typename VType>
__device__ __forceinline__ smr_t load_smr_va(VType src);

template <bool DESCENDING>
__device__ __forceinline__ va16u32x4 vcmpac_f32(v16u32 vr_s, v16f32 lhs_vr,
                                                smr_t smr0);


/*  =========================================================================
                                gcu_arch 300 only
    ========================================================================= */
#if (__GCU_ARCH__ >= 300 && __GCU_ARCH__ < 400)
template <bool DESCENDING, typename VType, typename MT>
__device__ __forceinline__ MT kernel_cmp0(VType vlhs, VType vrhs) {
  if (DESCENDING) {
    return tops::vgt<MT>(vlhs, vrhs);
  } else {
    return tops::vlt<MT>(vlhs, vrhs);
  }
}

template <bool DESCENDING, typename VType, typename MT>
__device__ __forceinline__ MT kernel_cmp1(VType vlhs, VType vrhs) {
  if (DESCENDING) {
    return tops::vlt<MT>(vlhs, vrhs);
  } else {
    return tops::vgt<MT>(vlhs, vrhs);
  }
}

template <bool DESCENDING, typename VType>
__device__ __forceinline__ VType kernel_getV0(VType vlhs, VType vrhs) {
  if (DESCENDING) {
    return tops::vmax<VType>(vlhs, vrhs);
  } else {
    return tops::vmin<VType>(vlhs, vrhs);
  }
}

template <bool DESCENDING, typename VType>
__device__ __forceinline__ VType kernel_getV1(VType vlhs, VType vrhs) {
  if (DESCENDING) {
    return tops::vmin<VType>(vlhs, vrhs);
  } else {
    return tops::vmax<VType>(vlhs, vrhs);
  }
}

template <typename VType>
__device__ __forceinline__ smr_t load_smr_va(VType src) {
  smr_t smr0 = __dtu_v_clrsmr();
  return smr0;
}

template <>
__device__ __forceinline__ smr_t load_smr_va(va16f32x4 src) {
  smr_t smr0 = __dtu_v_clrsmr();
  auto vr0 = __dtu_extractqa2vr(src, 0);
  smr0 = __dtu_m_ldsmr_mode8_f_row(smr0, vr0, 0);
  auto vr1 = __dtu_l_extractqa2vr(src, 1);
  smr0 = __dtu_m_ldsmr_mode8_f_row(smr0, vr1, 4);
  auto vr2 = __dtu_l_extractqa2vr(src, 2);
  smr0 = __dtu_m_ldsmr_mode8_f_row(smr0, vr2, 8);
  auto vr3 = __dtu_l_extractqa2vr(src, 3);
  smr0 = __dtu_m_ldsmr_mode8_f_row(smr0, vr3, 12);
  return smr0;
}

template <>
__device__ __forceinline__ smr_t load_smr_va(va32f16x2 src) {
  smr_t smr0 = __dtu_v_clrsmr();
  v32f16x2 da = __dtu_v_movda2vr_f16_dual(src);
  smr0 = __dtu_m_ldsmr2_mode17_f16_row_vr(smr0, da, 0);
  return smr0;
}

template <>
__device__ __forceinline__ smr_t load_smr_va(va32bf16x2 src) {
  smr_t smr0 = __dtu_v_clrsmr();
  v32bf16x2 da = __dtu_v_movda2vr_bf16_dual(src);
  smr0 = __dtu_m_ldsmr2_mode17_bf16_row_vr(smr0, da, 0);
  return smr0;
}


template <bool DESCENDING, int ID>
__device__ __forceinline__ void dtu_vcmpac_f32(va16u32x4& ov_qa, v16u32 vr_s,
                                               v16f32 lhs_vr, smr_t smr0) {
  if (DESCENDING) {
    ov_qa = __dtu_m_vcmpac_f32(ov_qa, vr_s, lhs_vr, ID, smr0);
  } else {
    ov_qa = __dtu_m_vcmpac_f32_ascend(ov_qa, vr_s, lhs_vr, ID, smr0);
  }
}

template <bool DESCENDING, int ID>
__device__ __forceinline__ void dtu_vcmpac_f16(va16u32x4& ov_qa, v16u32 vr_s,
                                               v32f16 lhs_vr, smr_t smr0) {
  if (DESCENDING) {
    ov_qa = __dtu_m_vcmpac_f16(ov_qa, vr_s, lhs_vr, ID, smr0);
  } else {
    ov_qa = __dtu_m_vcmpac_f16_ascend(ov_qa, vr_s, lhs_vr, ID, smr0);
  }
}

template <bool DESCENDING, int ID>
__device__ __forceinline__ void dtu_vcmpac_bf16(va16u32x4& ov_qa, v16u32 vr_s,
                                                v32bf16 lhs_vr, smr_t smr0) {
  if (DESCENDING) {
    ov_qa = __dtu_m_vcmpac_bf16(ov_qa, vr_s, lhs_vr, ID, smr0);
  } else {
    ov_qa = __dtu_m_vcmpac_bf16_ascend(ov_qa, vr_s, lhs_vr, ID, smr0);
  }
}

template <bool DESCENDING>
__device__ __forceinline__ va16u32x4 vcmpac_f32(v16u32 vr_s, v16f32 lhs_vr,
                                                smr_t smr0) {
  va16u32x4 ov_qa = __dtu_l_vclr_qa();
  dtu_vcmpac_f32<DESCENDING, 0>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f32<DESCENDING, 1>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f32<DESCENDING, 2>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f32<DESCENDING, 3>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f32<DESCENDING, 4>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f32<DESCENDING, 5>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f32<DESCENDING, 6>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f32<DESCENDING, 7>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f32<DESCENDING, 8>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f32<DESCENDING, 9>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f32<DESCENDING, 10>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f32<DESCENDING, 11>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f32<DESCENDING, 12>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f32<DESCENDING, 13>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f32<DESCENDING, 14>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f32<DESCENDING, 15>(ov_qa, vr_s, lhs_vr, smr0);

  return ov_qa;
}

template <bool DESCENDING>
__device__ __forceinline__ va16u32x4 vcmpac_f16(v16u32 vr_s, v32f16 lhs_vr,
                                                smr_t smr0) {
  va16u32x4 ov_qa = __dtu_l_vclr_qa();
  dtu_vcmpac_f16<DESCENDING, 0>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 1>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 2>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 3>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 4>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 5>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 6>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 7>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 8>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 9>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 10>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 11>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 12>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 13>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 14>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 15>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 16>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 17>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 18>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 19>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 20>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 21>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 22>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 23>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 24>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 25>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 26>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 27>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 28>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 29>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 30>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_f16<DESCENDING, 31>(ov_qa, vr_s, lhs_vr, smr0);

  return ov_qa;
}

template <bool DESCENDING>
__device__ __forceinline__ va16u32x4 vcmpac_bf16(v16u32 vr_s, v32bf16 lhs_vr,
                                                 smr_t smr0) {
  va16u32x4 ov_qa = __dtu_l_vclr_qa();
  dtu_vcmpac_bf16<DESCENDING, 0>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 1>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 2>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 3>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 4>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 5>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 6>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 7>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 8>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 9>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 10>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 11>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 12>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 13>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 14>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 15>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 16>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 17>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 18>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 19>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 20>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 21>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 22>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 23>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 24>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 25>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 26>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 27>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 28>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 29>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 30>(ov_qa, vr_s, lhs_vr, smr0);
  dtu_vcmpac_bf16<DESCENDING, 31>(ov_qa, vr_s, lhs_vr, smr0);

  return ov_qa;
}

template <
    typename T, typename VType, bool DESCENDING,
    typename std::enable_if<std::is_same<T, float>::value, bool>::type = true>
__device__ __forceinline__ va16u32x4
get_ov_from_smr_qa(VType lhs, v16u32 vr_s_ori, int lhs_base, smr_t smr0) {
  va16u32x4 ov_qa = __dtu_l_vclr_qa();
  v16f32 lhs_vr[4];
  v16u32 vr_s[4];
  vr_s[0] = vr_s_ori;
  vr_s[1] = vr_s_ori;
  vr_s[2] = vr_s_ori;
  vr_s[3] = vr_s_ori;
  lhs_vr[0] = __dtu_l_extractqa2vr(lhs, 0);
  lhs_vr[1] = __dtu_l_extractqa2vr(lhs, 1);
  lhs_vr[2] = __dtu_l_extractqa2vr(lhs, 2);
  lhs_vr[3] = __dtu_l_extractqa2vr(lhs, 3);
  vr_s[0] = __dtu_v_movr2vr_lsb(vr_s[0], lhs_base);
  vr_s[1] = __dtu_v_movr2vr_lsb(vr_s[1], lhs_base + 16);
  vr_s[2] = __dtu_v_movr2vr_lsb(vr_s[2], lhs_base + 32);
  vr_s[3] = __dtu_v_movr2vr_lsb(vr_s[3], lhs_base + 48);

  va16u32x4 ov_qa_0 = vcmpac_f32<DESCENDING>(vr_s[0], lhs_vr[0], smr0);
  va16u32x4 ov_qa_1 = vcmpac_f32<DESCENDING>(vr_s[1], lhs_vr[1], smr0);
  va16u32x4 ov_qa_2 = vcmpac_f32<DESCENDING>(vr_s[2], lhs_vr[2], smr0);
  va16u32x4 ov_qa_3 = vcmpac_f32<DESCENDING>(vr_s[3], lhs_vr[3], smr0);

  va16u32x4 ov_qa_tmp0 = __dtu_m_mop_add_u32_qa(ov_qa_0, ov_qa_1);
  va16u32x4 ov_qa_tmp1 = __dtu_m_mop_add_u32_qa(ov_qa_2, ov_qa_3);
  ov_qa = __dtu_m_mop_add_u32_qa(ov_qa_tmp0, ov_qa_tmp1);
  return ov_qa;
}

template <
    typename T, typename VType, bool DESCENDING,
    typename std::enable_if<std::is_same<T, half>::value, bool>::type = true>
__device__ __forceinline__ va16u32x4
get_ov_from_smr_qa(VType lhs, v16u32 vr_s_ori, int lhs_base, smr_t smr0) {
  va16u32x4 ov_qa = __dtu_l_vclr_qa();
  v32f16 lhs_vr[2];
  v16u32 vr_s[2];
  vr_s[0] = vr_s_ori;
  vr_s[1] = vr_s_ori;
  auto va0 = __dtu_extractda2va_f16(lhs, 0);
  auto va1 = __dtu_extractda2va_f16(lhs, 1);
  lhs_vr[0] = __dtu_v_movda2vr_f16(va0);
  lhs_vr[1] = __dtu_v_movda2vr_f16(va1);
  vr_s[0] = __dtu_v_movr2vr_lsb(vr_s[0], lhs_base);
  vr_s[1] = __dtu_v_movr2vr_lsb(vr_s[1], lhs_base + 32);

  va16u32x4 ov_qa_0 = vcmpac_f16<DESCENDING>(vr_s[0], lhs_vr[0], smr0);
  va16u32x4 ov_qa_1 = vcmpac_f16<DESCENDING>(vr_s[1], lhs_vr[1], smr0);

  ov_qa = __dtu_m_mop_add_u32_qa(ov_qa_0, ov_qa_1);
  return ov_qa;
}

template <
    typename T, typename VType, bool DESCENDING,
    typename std::enable_if<std::is_same<T, bfloat>::value, bool>::type = true>
__device__ __forceinline__ va16u32x4
get_ov_from_smr_qa(VType lhs, v16u32 vr_s_ori, int lhs_base, smr_t smr0) {
  va16u32x4 ov_qa = __dtu_l_vclr_qa();
  v32bf16 lhs_vr[2];
  v16u32 vr_s[2];
  vr_s[0] = vr_s_ori;
  vr_s[1] = vr_s_ori;
  auto va0 = __dtu_extractda2va_bf16(lhs, 0);
  auto va1 = __dtu_extractda2va_bf16(lhs, 1);
  lhs_vr[0] = __dtu_v_movda2vr_bf16(va0);
  lhs_vr[1] = __dtu_v_movda2vr_bf16(va1);
  vr_s[0] = __dtu_v_movr2vr_lsb(vr_s[0], lhs_base);
  vr_s[1] = __dtu_v_movr2vr_lsb(vr_s[1], lhs_base + 32);

  va16u32x4 ov_qa_0 = vcmpac_bf16<DESCENDING>(vr_s[0], lhs_vr[0], smr0);
  va16u32x4 ov_qa_1 = vcmpac_bf16<DESCENDING>(vr_s[1], lhs_vr[1], smr0);

  ov_qa = __dtu_m_mop_add_u32_qa(ov_qa_0, ov_qa_1);
  return ov_qa;
}

template <
    typename T, typename VType, bool DESCENDING,
    typename std::enable_if<std::is_same<T, int>::value, bool>::type = true>
__device__ __forceinline__ va16u32x4
get_ov_from_smr_qa(VType lhs, v16u32 vr_s_ori, int lhs_base, smr_t smr0) {
  va16u32x4 ov_qa = __dtu_l_vclr_qa();
  return ov_qa;
}

template <
    typename T, typename VType, bool DESCENDING,
    typename std::enable_if<std::is_same<T, int8_t>::value, bool>::type = true>
__device__ __forceinline__ va16u32x4
get_ov_from_smr_qa(VType lhs, v16u32 vr_s_ori, int lhs_base, smr_t smr0) {
  va16u32x4 ov_qa = __dtu_l_vclr_qa();
  return ov_qa;
}

template <typename T, typename VType, bool DESCENDING,
          typename std::enable_if<std::is_same<T, uint32_t>::value,
                                  bool>::type = true>
__device__ __forceinline__ va16u32x4
get_ov_from_smr_qa(VType lhs, v16u32 vr_s_ori, int lhs_base, smr_t smr0) {
  va16u32x4 ov_qa = __dtu_l_vclr_qa();
  return ov_qa;
}

template <typename T, typename VType, bool DESCENDING,
          typename std::enable_if<std::is_same<T, unsigned char>::value,
                                  bool>::type = true>
__device__ __forceinline__ va16u32x4
get_ov_from_smr_qa(VType lhs, v16u32 vr_s_ori, int lhs_base, smr_t smr0) {
  va16u32x4 ov_qa = __dtu_l_vclr_qa();
  return ov_qa;
}

#endif

template <bool DESCENDING, typename TCLE_VType>
__device__ __forceinline__ TCLE_VType kernel_tcle_get_value(TCLE_VType vlhs,
                                                            TCLE_VType vrhs) {
  if (DESCENDING) {
    return tcle::max(vlhs, vrhs);
  } else {
    return tcle::min(vlhs, vrhs);
  }
}

template <bool DESCENDING, typename TCLE_VType, typename TCLE_MT>
__device__ __forceinline__ TCLE_MT kernel_tcle_cmp_value(TCLE_VType vlhs,
                                                         TCLE_VType vrhs) {
  if (DESCENDING) {
    return vlhs > vrhs;
  } else {
    return vlhs < vrhs;
  }
}

#endif
