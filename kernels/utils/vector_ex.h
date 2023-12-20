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
#ifndef CC_KERNEL_VECTOR_EX_H
#define CC_KERNEL_VECTOR_EX_H
#include <tops.h>

namespace tops {
/* ==========================================================================
                         bitcast between vbool types
   ========================================================================== */
template <typename TO_TYPE, typename FROM_TYPE>
inline KRT_API TO_TYPE mask_bitcast(const FROM_TYPE& v);

#if (__KRT_ARCH__ >= 300)
template <>
inline KRT_API vbool16_t mask_bitcast(const vbool16_t& v) {
  return v;
}

template <>
inline KRT_API vbool32_t mask_bitcast(const vbool32_t& v) {
  return v;
}

template <>
inline KRT_API vbool64_t mask_bitcast(const vbool64_t& v) {
  return v;
}

template <>
inline KRT_API vbool128_t mask_bitcast(const vbool128_t& v) {
  return v;
}

template <>
inline KRT_API vbool256_t mask_bitcast(const vbool256_t& v) {
  return v;
}

#define BITCAST_BETWEEN_BOOL_VECTORS(FROM_NUM, TO_NUM) \
  template <>                                          \
  inline KRT_API vbool##TO_NUM##_t mask_bitcast(       \
      const vbool##FROM_NUM##_t& v) {                  \
    return __dtu_bitcast_m##FROM_NUM##_m##TO_NUM(v);   \
  }

BITCAST_BETWEEN_BOOL_VECTORS(16, 32);
BITCAST_BETWEEN_BOOL_VECTORS(16, 64);
BITCAST_BETWEEN_BOOL_VECTORS(16, 128);
BITCAST_BETWEEN_BOOL_VECTORS(16, 256);
BITCAST_BETWEEN_BOOL_VECTORS(32, 16);
BITCAST_BETWEEN_BOOL_VECTORS(32, 64);
BITCAST_BETWEEN_BOOL_VECTORS(32, 128);
BITCAST_BETWEEN_BOOL_VECTORS(32, 256);
BITCAST_BETWEEN_BOOL_VECTORS(64, 16);
BITCAST_BETWEEN_BOOL_VECTORS(64, 32);
BITCAST_BETWEEN_BOOL_VECTORS(64, 128);
BITCAST_BETWEEN_BOOL_VECTORS(64, 256);
BITCAST_BETWEEN_BOOL_VECTORS(128, 16);
BITCAST_BETWEEN_BOOL_VECTORS(128, 32);
BITCAST_BETWEEN_BOOL_VECTORS(128, 64);
BITCAST_BETWEEN_BOOL_VECTORS(128, 256);
BITCAST_BETWEEN_BOOL_VECTORS(256, 16);
BITCAST_BETWEEN_BOOL_VECTORS(256, 32);
BITCAST_BETWEEN_BOOL_VECTORS(256, 64);
BITCAST_BETWEEN_BOOL_VECTORS(256, 128);
#undef BITCAST_BETWEEN_BOOL_VECTORS
#endif  // __KRT_ARCH__ >= 300

/* ==========================================================================
                      bitcast between vbool and other types
   ========================================================================== */
#if (__KRT_ARCH__ >= 300)
#define BITCAST_BETWEEN_BOOL_AND_OTHER(BOOL_TYPE, BOOL_SUFFIX, OTHER_TYPE, \
                                       OTHER_SUFFIX, SUFFIX)               \
  template <>                                                              \
  inline KRT_API BOOL_TYPE mask_bitcast(const OTHER_TYPE& v) {             \
    return __dtu_m_remapva_##OTHER_SUFFIX##_##SUFFIX##_##BOOL_SUFFIX(v);   \
  }                                                                        \
  template <>                                                              \
  inline KRT_API OTHER_TYPE mask_bitcast(const BOOL_TYPE& v) {             \
    return __dtu_m_remapva_##OTHER_SUFFIX##_##BOOL_SUFFIX##_##SUFFIX(v);   \
  }

BITCAST_BETWEEN_BOOL_AND_OTHER(vbool16_t, m16, va64i8x4, s8, b);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool16_t, m16, va64u8x4, u8, b);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool16_t, m16, va32i16x4, s16, h);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool16_t, m16, va32u16x4, u16, h);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool16_t, m16, va32f16x4, f16, h);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool16_t, m16, va32bf16x4, bf16, h);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool16_t, m16, va16i32x4, s32, w);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool16_t, m16, va16u32x4, u32, w);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool16_t, m16, va16f32x4, f32, w);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool32_t, m32, va64i8x4, s8, b);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool32_t, m32, va64u8x4, u8, b);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool32_t, m32, va32i16x4, s16, h);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool32_t, m32, va32u16x4, u16, h);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool32_t, m32, va32f16x4, f16, h);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool32_t, m32, va32bf16x4, bf16, h);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool32_t, m32, va16i32x4, s32, w);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool32_t, m32, va16u32x4, u32, w);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool32_t, m32, va16f32x4, f32, w);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool64_t, m64, va64i8x4, s8, b);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool64_t, m64, va64u8x4, u8, b);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool64_t, m64, va32i16x4, s16, h);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool64_t, m64, va32u16x4, u16, h);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool64_t, m64, va32f16x4, f16, h);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool64_t, m64, va32bf16x4, bf16, h);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool64_t, m64, va16i32x4, s32, w);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool64_t, m64, va16u32x4, u32, w);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool64_t, m64, va16f32x4, f32, w);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool128_t, m128, va64i8x4, s8, b);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool128_t, m128, va64u8x4, u8, b);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool128_t, m128, va32i16x4, s16, h);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool128_t, m128, va32u16x4, u16, h);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool128_t, m128, va32f16x4, f16, h);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool128_t, m128, va32bf16x4, bf16, h);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool128_t, m128, va16i32x4, s32, w);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool128_t, m128, va16u32x4, u32, w);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool128_t, m128, va16f32x4, f32, w);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool256_t, m256, va64i8x4, s8, b);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool256_t, m256, va64u8x4, u8, b);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool256_t, m256, va32i16x4, s16, h);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool256_t, m256, va32u16x4, u16, h);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool256_t, m256, va32f16x4, f16, h);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool256_t, m256, va32bf16x4, bf16, h);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool256_t, m256, va16i32x4, s32, w);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool256_t, m256, va16u32x4, u32, w);
BITCAST_BETWEEN_BOOL_AND_OTHER(vbool256_t, m256, va16f32x4, f32, w);
#undef BITCAST_BETWEEN_BOOL_AND_OTHER
#endif  // __KRT_ARCH__ >= 300

/* ==========================================================================
                         vector length for vbool types
   ========================================================================== */
#if (__KRT_ARCH__ >= 300)
template <>
struct vector_length<vbool16_t> {
  constexpr static int value = 32;
};

template <>
struct vector_length<vbool32_t> {
  constexpr static int value = 64;
};

template <>
struct vector_length<vbool64_t> {
  constexpr static int value = 128;
};

template <>
struct vector_length<vbool128_t> {
  constexpr static int value = 256;
};

template <>
struct vector_length<vbool256_t> {
  constexpr static int value = 512;
};
#endif  // __KRT_ARCH__ >= 300

/* ==========================================================================
                 vector_length and vector_to_scalar for arrays
   ========================================================================== */
template <typename VEC_TYPE, int N>
struct vector_length<VEC_TYPE[N]> {
  constexpr static int value = vector_length<VEC_TYPE>::value * N;
};

template <typename VEC_TYPE, int N>
struct vector_to_scalar<VEC_TYPE[N]> {
  using type = typename vector_to_scalar<VEC_TYPE>::type;
};

/* ==========================================================================
                           vgather without maskbit
   ========================================================================== */
template <typename TO_TYPE, typename OFFSET_TYPE>
inline KRT_API TO_TYPE vgather(void* src, const OFFSET_TYPE& offset);

template <typename TO_TYPE, typename OFFSET_TYPE>
inline KRT_API TO_TYPE vgather(void* src, const OFFSET_TYPE& offset) {
  static_assert(
      std::is_integral<typename vector_to_scalar<OFFSET_TYPE>::type>::value &&
          sizeof(typename vector_to_scalar<OFFSET_TYPE>::type) == 4,
      "OFFSET_TYPE must be u32/i32 type");
  static_assert(
      vector_length<TO_TYPE>::value == vector_length<OFFSET_TYPE>::value,
      "vector length of TO_TYPE and OFFSET_TYPE must be equal");
}

#if (__KRT_ARCH__ >= 300)
#define VGATHER_DIRECT_IMPL(TO_TYPE, TO_SUFFIX, OFFSET_TYPE)                 \
  template <>                                                                \
  inline KRT_API TO_TYPE vgather(void* src, const OFFSET_TYPE& offset) {     \
    return __dtu_m_vldxda_##TO_SUFFIX((__DTU_INTRIN_AS__ char*)src, offset); \
  }                                                                          \
  template <>                                                                \
  inline KRT_API TO_TYPE vgather(void* src, const OFFSET_TYPE(&offset)[1]) { \
    return vgather<TO_TYPE>(src, offset[0]);                                 \
  }

VGATHER_DIRECT_IMPL(va16f32, f32, va16u32);
VGATHER_DIRECT_IMPL(va16f32, f32, va16i32);
VGATHER_DIRECT_IMPL(va16u32, u32, va16u32);
VGATHER_DIRECT_IMPL(va16u32, u32, va16i32);
VGATHER_DIRECT_IMPL(va16i32, s32, va16u32);
VGATHER_DIRECT_IMPL(va16i32, s32, va16i32);
VGATHER_DIRECT_IMPL(va16f32x2, dual_f32, va16u32x2);
VGATHER_DIRECT_IMPL(va16f32x2, dual_f32, va16i32x2);
VGATHER_DIRECT_IMPL(va16u32x2, dual_u32, va16u32x2);
VGATHER_DIRECT_IMPL(va16u32x2, dual_u32, va16i32x2);
VGATHER_DIRECT_IMPL(va16i32x2, dual_s32, va16u32x2);
VGATHER_DIRECT_IMPL(va16i32x2, dual_s32, va16i32x2);
VGATHER_DIRECT_IMPL(va32f16, f16, va16u32x2);
VGATHER_DIRECT_IMPL(va32f16, f16, va16i32x2);
VGATHER_DIRECT_IMPL(va32bf16, bf16, va16i32x2);
VGATHER_DIRECT_IMPL(va32bf16, bf16, va16u32x2);
VGATHER_DIRECT_IMPL(va32u16, u16, va16u32x2);
VGATHER_DIRECT_IMPL(va32u16, u16, va16i32x2);
VGATHER_DIRECT_IMPL(va32i16, s16, va16u32x2);
VGATHER_DIRECT_IMPL(va32i16, s16, va16i32x2);
VGATHER_DIRECT_IMPL(va64u8, u8, va16u32x4);
VGATHER_DIRECT_IMPL(va64u8, u8, va16i32x4);
VGATHER_DIRECT_IMPL(va64i8, s8, va16u32x4);
VGATHER_DIRECT_IMPL(va64i8, s8, va16i32x4);
#undef VGATHER_DIRECT_IMPL

#define VGATHER_DUAL_IMPL(TO_TYPE, OFFSET_TYPE)                              \
  template <>                                                                \
  inline KRT_API TO_TYPE vgather(void* src, const OFFSET_TYPE& offset) {     \
    using ValueType = vector_to_scalar<TO_TYPE>::type;                       \
    using IntType = vector_to_scalar<OFFSET_TYPE>::type;                     \
    constexpr int VEC_LENGTH = vector_length<TO_TYPE>::value;                \
    using HalfVecType = scalar_to_vector<ValueType, VEC_LENGTH / 2>::type;   \
    using HalfIntType = scalar_to_vector<IntType, VEC_LENGTH / 2>::type;     \
    HalfIntType offsets[2];                                                  \
    offsets[0] = vunpack0<HalfIntType>(offset);                              \
    offsets[1] = vunpack1<HalfIntType>(offset);                              \
    HalfVecType values[2];                                                   \
    values[0] = vgather<HalfVecType>(src, offsets[0]);                       \
    values[1] = vgather<HalfVecType>(src, offsets[1]);                       \
    return vpack2<TO_TYPE>(values[0], values[1]);                            \
  }                                                                          \
  template <>                                                                \
  inline KRT_API TO_TYPE vgather(void* src, const OFFSET_TYPE(&offset)[1]) { \
    return vgather<TO_TYPE>(src, offset[0]);                                 \
  }

VGATHER_DUAL_IMPL(va16f32x4, va16u32x4);
VGATHER_DUAL_IMPL(va16f32x4, va16i32x4);
VGATHER_DUAL_IMPL(va16u32x4, va16u32x4);
VGATHER_DUAL_IMPL(va16u32x4, va16i32x4);
VGATHER_DUAL_IMPL(va16i32x4, va16u32x4);
VGATHER_DUAL_IMPL(va16i32x4, va16i32x4);
VGATHER_DUAL_IMPL(va32f16x2, va16u32x4);
VGATHER_DUAL_IMPL(va32f16x2, va16i32x4);
VGATHER_DUAL_IMPL(va32bf16x2, va16u32x4);
VGATHER_DUAL_IMPL(va32bf16x2, va16i32x4);
VGATHER_DUAL_IMPL(va32u16x2, va16u32x4);
VGATHER_DUAL_IMPL(va32u16x2, va16i32x4);
VGATHER_DUAL_IMPL(va32i16x2, va16u32x4);
VGATHER_DUAL_IMPL(va32i16x2, va16i32x4);
#undef VGATHER_DUAL_IMPL

#define VGATHER_DUAL_ARRAY_IMPL(TO_TYPE, OFFSET_TYPE)                         \
  template <>                                                                 \
  inline KRT_API TO_TYPE vgather(void* src, const OFFSET_TYPE(&offsets)[2]) { \
    using ValueType = vector_to_scalar<TO_TYPE>::type;                        \
    constexpr int VEC_LENGTH = vector_length<TO_TYPE>::value;                 \
    using HalfVecType = scalar_to_vector<ValueType, VEC_LENGTH / 2>::type;    \
    HalfVecType values[2];                                                    \
    values[0] = vgather<HalfVecType>(src, offsets[0]);                        \
    values[1] = vgather<HalfVecType>(src, offsets[1]);                        \
    return vpack2<TO_TYPE>(values[0], values[1]);                             \
  }

VGATHER_DUAL_ARRAY_IMPL(va32f16x4, va16u32x4);
VGATHER_DUAL_ARRAY_IMPL(va32f16x4, va16i32x4);
VGATHER_DUAL_ARRAY_IMPL(va32bf16x4, va16u32x4);
VGATHER_DUAL_ARRAY_IMPL(va32bf16x4, va16i32x4);
VGATHER_DUAL_ARRAY_IMPL(va32u16x4, va16u32x4);
VGATHER_DUAL_ARRAY_IMPL(va32u16x4, va16i32x4);
VGATHER_DUAL_ARRAY_IMPL(va32i16x4, va16u32x4);
VGATHER_DUAL_ARRAY_IMPL(va32i16x4, va16i32x4);
VGATHER_DUAL_ARRAY_IMPL(va64u8x2, va16u32x4);
VGATHER_DUAL_ARRAY_IMPL(va64u8x2, va16i32x4);
VGATHER_DUAL_ARRAY_IMPL(va64i8x2, va16u32x4);
VGATHER_DUAL_ARRAY_IMPL(va64i8x2, va16i32x4);
#undef VGATHER_DUAL_ARRAY_IMPL

#define VGATHER_QUAD_ARRAY_IMPL(TO_TYPE, OFFSET_TYPE)                         \
  template <>                                                                 \
  inline KRT_API TO_TYPE vgather(void* src, const OFFSET_TYPE(&offsets)[4]) { \
    using ValueType = vector_to_scalar<TO_TYPE>::type;                        \
    constexpr int VEC_LENGTH = vector_length<TO_TYPE>::value;                 \
    using QuarterVecType = scalar_to_vector<ValueType, VEC_LENGTH / 4>::type; \
    QuarterVecType values[4];                                                 \
    values[0] = vgather<QuarterVecType>(src, offsets[0]);                     \
    values[1] = vgather<QuarterVecType>(src, offsets[1]);                     \
    values[2] = vgather<QuarterVecType>(src, offsets[2]);                     \
    values[3] = vgather<QuarterVecType>(src, offsets[3]);                     \
    return vpack4<TO_TYPE>(values[0], values[1], values[2], values[3]);       \
  }

VGATHER_QUAD_ARRAY_IMPL(va64u8x4, va16u32x4);
VGATHER_QUAD_ARRAY_IMPL(va64u8x4, va16i32x4);
VGATHER_QUAD_ARRAY_IMPL(va64i8x4, va16u32x4);
VGATHER_QUAD_ARRAY_IMPL(va64i8x4, va16i32x4);
#undef VGATHER_QUAD_ARRAY_IMPL
#endif  // __KRT_ARCH__ >= 300

/* ==========================================================================
                             vgather with maskbit
   ========================================================================== */
template <typename TO_TYPE, typename MB_TYPE, typename OFFSET_TYPE>
inline KRT_API TO_TYPE vgather_t(const MB_TYPE& mb, void* src,
                                 const OFFSET_TYPE& offset,
                                 const TO_TYPE& remain);

template <typename TO_TYPE, typename MB_TYPE, typename OFFSET_TYPE>
inline KRT_API TO_TYPE vgather_f(const MB_TYPE& mb, void* src,
                                 const OFFSET_TYPE& offset,
                                 const TO_TYPE& remain) {
  auto mask = mask_not(mb);
  return vgather_t<TO_TYPE>(mask, src, offset, remain);
}

template <typename TO_TYPE, typename MB_TYPE, typename OFFSET_TYPE>
inline KRT_API TO_TYPE vgather_t(const MB_TYPE& mb, void* src,
                                 const OFFSET_TYPE& offset,
                                 const TO_TYPE& remain) {
  static_assert(
      std::is_integral<typename vector_to_scalar<OFFSET_TYPE>::type>::value &&
          sizeof(typename vector_to_scalar<OFFSET_TYPE>::type) == 4,
      "OFFSET_TYPE must be u32/i32 type");
  static_assert(
      vector_length<TO_TYPE>::value == vector_length<OFFSET_TYPE>::value &&
          vector_length<TO_TYPE>::value == vector_length<MB_TYPE>::value,
      "vector length of TO_TYPE, OFFSET_TYPE and MB_TYPE must be equal");
}

#if (__KRT_ARCH__ >= 300)
#define VGATHER_T_DIRECT_IMPL(TO_TYPE, TO_SUFFIX, MB_TYPE, OFFSET_TYPE)      \
  template <>                                                                \
  inline KRT_API TO_TYPE vgather_t(const MB_TYPE& mb, void* src,             \
                                   const OFFSET_TYPE& offset,                \
                                   const TO_TYPE& remain) {                  \
    using ValueType = vector_to_scalar<TO_TYPE>::type;                       \
    using QaType = scalar_to_vector<ValueType, TOPS_VECTOR_LENGTH * 4 /      \
                                                   sizeof(ValueType)>::type; \
    using MaskType = vector_to_mask<QaType>::type;                           \
    auto mask = mask_bitcast<MaskType>(mb);                                  \
    return __dtu_m_vldxda_##TO_SUFFIX##_vs0_vm((__DTU_INTRIN_AS__ char*)src, \
                                               offset, remain, mask);        \
  }                                                                          \
  template <>                                                                \
  inline KRT_API TO_TYPE vgather_t(const MB_TYPE& mb, void* src,             \
                                   const OFFSET_TYPE(&offset)[1],            \
                                   const TO_TYPE& remain) {                  \
    return vgather_t(mb, src, offset[0], remain);                            \
  }

VGATHER_T_DIRECT_IMPL(va16f32, f32, vbool16_t, va16u32);
VGATHER_T_DIRECT_IMPL(va16f32, f32, vbool16_t, va16i32);
VGATHER_T_DIRECT_IMPL(va16u32, u32, vbool16_t, va16u32);
VGATHER_T_DIRECT_IMPL(va16u32, u32, vbool16_t, va16i32);
VGATHER_T_DIRECT_IMPL(va16i32, s32, vbool16_t, va16u32);
VGATHER_T_DIRECT_IMPL(va16i32, s32, vbool16_t, va16i32);
VGATHER_T_DIRECT_IMPL(va16f32x2, dual_f32, vbool32_t, va16u32x2);
VGATHER_T_DIRECT_IMPL(va16f32x2, dual_f32, vbool32_t, va16i32x2);
VGATHER_T_DIRECT_IMPL(va16u32x2, dual_u32, vbool32_t, va16u32x2);
VGATHER_T_DIRECT_IMPL(va16u32x2, dual_u32, vbool32_t, va16i32x2);
VGATHER_T_DIRECT_IMPL(va16i32x2, dual_s32, vbool32_t, va16u32x2);
VGATHER_T_DIRECT_IMPL(va16i32x2, dual_s32, vbool32_t, va16i32x2);
VGATHER_T_DIRECT_IMPL(va32f16, f16, vbool32_t, va16u32x2);
VGATHER_T_DIRECT_IMPL(va32f16, f16, vbool32_t, va16i32x2);
VGATHER_T_DIRECT_IMPL(va32bf16, bf16, vbool32_t, va16u32x2);
VGATHER_T_DIRECT_IMPL(va32bf16, bf16, vbool32_t, va16i32x2);
VGATHER_T_DIRECT_IMPL(va32u16, u16, vbool32_t, va16u32x2);
VGATHER_T_DIRECT_IMPL(va32u16, u16, vbool32_t, va16i32x2);
VGATHER_T_DIRECT_IMPL(va32i16, s16, vbool32_t, va16u32x2);
VGATHER_T_DIRECT_IMPL(va32i16, s16, vbool32_t, va16i32x2);
VGATHER_T_DIRECT_IMPL(va64u8, u8, vbool64_t, va16u32x4);
VGATHER_T_DIRECT_IMPL(va64u8, u8, vbool64_t, va16i32x4);
VGATHER_T_DIRECT_IMPL(va64i8, s8, vbool64_t, va16u32x4);
VGATHER_T_DIRECT_IMPL(va64i8, s8, vbool64_t, va16i32x4);
#undef VGATHER_T_DIRECT_IMPL

#define VGATHER_T_DUAL_IMPL(TO_TYPE, TO_SUFFIX, MB_TYPE, OFFSET_TYPE)        \
  template <>                                                                \
  inline KRT_API TO_TYPE vgather_t(const MB_TYPE& mb, void* src,             \
                                   const OFFSET_TYPE& offset,                \
                                   const TO_TYPE& remain) {                  \
    using ValueType = vector_to_scalar<TO_TYPE>::type;                       \
    using IntType = vector_to_scalar<OFFSET_TYPE>::type;                     \
    constexpr int VEC_LENGTH = vector_length<TO_TYPE>::value;                \
    using HalfVecType = scalar_to_vector<ValueType, VEC_LENGTH / 2>::type;   \
    using HalfIntType = scalar_to_vector<IntType, VEC_LENGTH / 2>::type;     \
    using QaType = scalar_to_vector<ValueType, TOPS_VECTOR_LENGTH * 4 /      \
                                                   sizeof(ValueType)>::type; \
    using MaskType = vector_to_mask<QaType>::type;                           \
    auto mask = mask_bitcast<MaskType>(mb);                                  \
    HalfIntType offsets[2];                                                  \
    offsets[0] = vunpack0<HalfIntType>(offset);                              \
    offsets[1] = vunpack1<HalfIntType>(offset);                              \
    HalfVecType remains[2];                                                  \
    remains[0] = vunpack0<HalfVecType>(remain);                              \
    remains[1] = vunpack1<HalfVecType>(remain);                              \
    HalfVecType values[2];                                                   \
    values[0] = __dtu_m_vldxda_##TO_SUFFIX##_vs0_vm(                         \
        (__DTU_INTRIN_AS__ char*)src, offsets[0], remains[0], mask);         \
    values[1] = __dtu_m_vldxda_##TO_SUFFIX##_vs1_vm(                         \
        (__DTU_INTRIN_AS__ char*)src, offsets[1], remains[1], mask);         \
    return vpack2<TO_TYPE>(values[0], values[1]);                            \
  }                                                                          \
                                                                             \
  template <>                                                                \
  inline KRT_API TO_TYPE vgather_t(const MB_TYPE& mb, void* src,             \
                                   const OFFSET_TYPE(&offset)[1],            \
                                   const TO_TYPE& remain) {                  \
    return vgather_t(mb, src, offset[0], remain);                            \
  }

VGATHER_T_DUAL_IMPL(va16f32x4, dual_f32, vbool64_t, va16u32x4);
VGATHER_T_DUAL_IMPL(va16f32x4, dual_f32, vbool64_t, va16i32x4);
VGATHER_T_DUAL_IMPL(va16u32x4, dual_u32, vbool64_t, va16u32x4);
VGATHER_T_DUAL_IMPL(va16u32x4, dual_u32, vbool64_t, va16i32x4);
VGATHER_T_DUAL_IMPL(va16i32x4, dual_s32, vbool64_t, va16u32x4);
VGATHER_T_DUAL_IMPL(va16i32x4, dual_s32, vbool64_t, va16i32x4);
VGATHER_T_DUAL_IMPL(va32f16x2, f16, vbool64_t, va16u32x4);
VGATHER_T_DUAL_IMPL(va32f16x2, f16, vbool64_t, va16i32x4);
VGATHER_T_DUAL_IMPL(va32bf16x2, bf16, vbool64_t, va16u32x4);
VGATHER_T_DUAL_IMPL(va32bf16x2, bf16, vbool64_t, va16i32x4);
VGATHER_T_DUAL_IMPL(va32u16x2, u16, vbool64_t, va16u32x4);
VGATHER_T_DUAL_IMPL(va32u16x2, u16, vbool64_t, va16i32x4);
VGATHER_T_DUAL_IMPL(va32i16x2, s16, vbool64_t, va16u32x4);
VGATHER_T_DUAL_IMPL(va32i16x2, s16, vbool64_t, va16i32x4);
#undef VGATHER_T_DUAL_IMPL

#define VGATHER_T_DUAL_ARRAY_IMPL(TO_TYPE, TO_SUFFIX, MB_TYPE, OFFSET_TYPE)  \
  template <>                                                                \
  inline KRT_API TO_TYPE vgather_t(const MB_TYPE& mb, void* src,             \
                                   const OFFSET_TYPE(&offsets)[2],           \
                                   const TO_TYPE& remain) {                  \
    using ValueType = vector_to_scalar<TO_TYPE>::type;                       \
    constexpr int VEC_LENGTH = vector_length<TO_TYPE>::value;                \
    using HalfVecType = scalar_to_vector<ValueType, VEC_LENGTH / 2>::type;   \
    using QaType = scalar_to_vector<ValueType, TOPS_VECTOR_LENGTH * 4 /      \
                                                   sizeof(ValueType)>::type; \
    using MaskType = vector_to_mask<QaType>::type;                           \
    auto mask = mask_bitcast<MaskType>(mb);                                  \
    HalfVecType remains[2];                                                  \
    remains[0] = vunpack0<HalfVecType>(remain);                              \
    remains[1] = vunpack1<HalfVecType>(remain);                              \
    HalfVecType values[2];                                                   \
    values[0] = __dtu_m_vldxda_##TO_SUFFIX##_vs0_vm(                         \
        (__DTU_INTRIN_AS__ char*)src, offsets[0], remains[0], mask);         \
    values[1] = __dtu_m_vldxda_##TO_SUFFIX##_vs1_vm(                         \
        (__DTU_INTRIN_AS__ char*)src, offsets[1], remains[1], mask);         \
    return vpack2<TO_TYPE>(values[0], values[1]);                            \
  }

VGATHER_T_DUAL_ARRAY_IMPL(va64u8x2, u8, vbool128_t, va16u32x4);
VGATHER_T_DUAL_ARRAY_IMPL(va64u8x2, u8, vbool128_t, va16i32x4);
VGATHER_T_DUAL_ARRAY_IMPL(va64i8x2, s8, vbool128_t, va16u32x4);
VGATHER_T_DUAL_ARRAY_IMPL(va64i8x2, s8, vbool128_t, va16i32x4);
#undef VGATHER_T_DUAL_ARRAY_IMPL

#define VGATHER_T_QUAD_ARRAY2_IMPL(TO_TYPE, TO_SUFFIX, MB_TYPE, OFFSET_TYPE)  \
  template <>                                                                 \
  inline KRT_API TO_TYPE vgather_t(const MB_TYPE& mb, void* src,              \
                                   const OFFSET_TYPE(&offset)[2],             \
                                   const TO_TYPE& remain) {                   \
    using ValueType = vector_to_scalar<TO_TYPE>::type;                        \
    using IntType = vector_to_scalar<OFFSET_TYPE>::type;                      \
    constexpr int VEC_LENGTH = vector_length<TO_TYPE>::value;                 \
    constexpr int OFF_LENGTH = vector_length<OFFSET_TYPE>::value;             \
    using QuarterVecType = scalar_to_vector<ValueType, VEC_LENGTH / 4>::type; \
    using HalfIntType = scalar_to_vector<IntType, OFF_LENGTH / 2>::type;      \
    HalfIntType offsets[4];                                                   \
    offsets[0] = vunpack0<HalfIntType>(offset[0]);                            \
    offsets[1] = vunpack1<HalfIntType>(offset[0]);                            \
    offsets[2] = vunpack0<HalfIntType>(offset[1]);                            \
    offsets[3] = vunpack1<HalfIntType>(offset[1]);                            \
    QuarterVecType remains[4];                                                \
    remains[0] = vunpack0<QuarterVecType>(remain);                            \
    remains[1] = vunpack1<QuarterVecType>(remain);                            \
    remains[2] = vunpack2<QuarterVecType>(remain);                            \
    remains[3] = vunpack3<QuarterVecType>(remain);                            \
    QuarterVecType values[4];                                                 \
    values[0] = __dtu_m_vldxda_##TO_SUFFIX##_vs0_vm(                          \
        (__DTU_INTRIN_AS__ char*)src, offsets[0], remains[0], mb);            \
    values[1] = __dtu_m_vldxda_##TO_SUFFIX##_vs1_vm(                          \
        (__DTU_INTRIN_AS__ char*)src, offsets[1], remains[1], mb);            \
    values[2] = __dtu_m_vldxda_##TO_SUFFIX##_vs2_vm(                          \
        (__DTU_INTRIN_AS__ char*)src, offsets[2], remains[2], mb);            \
    values[3] = __dtu_m_vldxda_##TO_SUFFIX##_vs3_vm(                          \
        (__DTU_INTRIN_AS__ char*)src, offsets[3], remains[3], mb);            \
    return vpack4<TO_TYPE>(values[0], values[1], values[2], values[3]);       \
  }

VGATHER_T_QUAD_ARRAY2_IMPL(va32f16x4, f16, vbool128_t, va16u32x4);
VGATHER_T_QUAD_ARRAY2_IMPL(va32f16x4, f16, vbool128_t, va16i32x4);
VGATHER_T_QUAD_ARRAY2_IMPL(va32bf16x4, bf16, vbool128_t, va16u32x4);
VGATHER_T_QUAD_ARRAY2_IMPL(va32bf16x4, bf16, vbool128_t, va16i32x4);
VGATHER_T_QUAD_ARRAY2_IMPL(va32u16x4, u16, vbool128_t, va16u32x4);
VGATHER_T_QUAD_ARRAY2_IMPL(va32u16x4, u16, vbool128_t, va16i32x4);
VGATHER_T_QUAD_ARRAY2_IMPL(va32i16x4, s16, vbool128_t, va16u32x4);
VGATHER_T_QUAD_ARRAY2_IMPL(va32i16x4, s16, vbool128_t, va16i32x4);
#undef VGATHER_T_QUAD_ARRAY2_IMPL

#define VGATHER_T_QUAD_ARRAY4_IMPL(TO_TYPE, TO_SUFFIX, MB_TYPE, OFFSET_TYPE)  \
  template <>                                                                 \
  inline KRT_API TO_TYPE vgather_t(const MB_TYPE& mb, void* src,              \
                                   const OFFSET_TYPE(&offsets)[4],            \
                                   const TO_TYPE& remain) {                   \
    using ValueType = vector_to_scalar<TO_TYPE>::type;                        \
    constexpr int VEC_LENGTH = vector_length<TO_TYPE>::value;                 \
    using QuarterVecType = scalar_to_vector<ValueType, VEC_LENGTH / 4>::type; \
    QuarterVecType remains[4];                                                \
    remains[0] = vunpack0<QuarterVecType>(remain);                            \
    remains[1] = vunpack1<QuarterVecType>(remain);                            \
    remains[2] = vunpack2<QuarterVecType>(remain);                            \
    remains[3] = vunpack3<QuarterVecType>(remain);                            \
    QuarterVecType values[4];                                                 \
    values[0] = __dtu_m_vldxda_##TO_SUFFIX##_vs0_vm(                          \
        (__DTU_INTRIN_AS__ char*)src, offsets[0], remains[0], mb);            \
    values[1] = __dtu_m_vldxda_##TO_SUFFIX##_vs1_vm(                          \
        (__DTU_INTRIN_AS__ char*)src, offsets[1], remains[1], mb);            \
    values[2] = __dtu_m_vldxda_##TO_SUFFIX##_vs2_vm(                          \
        (__DTU_INTRIN_AS__ char*)src, offsets[2], remains[2], mb);            \
    values[3] = __dtu_m_vldxda_##TO_SUFFIX##_vs3_vm(                          \
        (__DTU_INTRIN_AS__ char*)src, offsets[3], remains[3], mb);            \
    return vpack4<TO_TYPE>(values[0], values[1], values[2], values[3]);       \
  }

VGATHER_T_QUAD_ARRAY4_IMPL(va64u8x4, u8, vbool256_t, va16u32x4);
VGATHER_T_QUAD_ARRAY4_IMPL(va64u8x4, u8, vbool256_t, va16i32x4);
VGATHER_T_QUAD_ARRAY4_IMPL(va64i8x4, s8, vbool256_t, va16u32x4);
VGATHER_T_QUAD_ARRAY4_IMPL(va64i8x4, s8, vbool256_t, va16i32x4);
#undef VGATHER_T_QUAD_ARRAY4_IMPL
#endif  // __KRT_ARCH__ >= 300

/* ==========================================================================
                           vscatter without maskbit
   ========================================================================== */
template <typename FROM_TYPE, typename OFFSET_TYPE>
inline KRT_API void vscatter(void* dst, const FROM_TYPE& value,
                             const OFFSET_TYPE& offset);

template <typename FROM_TYPE, typename OFFSET_TYPE>
inline KRT_API void vscatter(void* dst, const FROM_TYPE& value,
                             const OFFSET_TYPE& offset) {
  static_assert(
      std::is_integral<typename vector_to_scalar<OFFSET_TYPE>::type>::value &&
          sizeof(typename vector_to_scalar<OFFSET_TYPE>::type) == 4,
      "OFFSET_TYPE must be u32/i32 type");
  static_assert(
      vector_length<FROM_TYPE>::value == vector_length<OFFSET_TYPE>::value,
      "vector length of FROM_TYPE and OFFSET_TYPE must be equal");
}

#if (__KRT_ARCH__ >= 300)
#define VSCATTER_DIRECT_IMPL(FROM_TYPE, FROM_SUFFIX, OFFSET_TYPE)              \
  template <>                                                                  \
  inline KRT_API void vscatter(void* dst, const FROM_TYPE& value,              \
                               const OFFSET_TYPE& offset) {                    \
    __dtu_m_vstxda_##FROM_SUFFIX(value, (__DTU_INTRIN_AS__ char*)dst, offset); \
  }                                                                            \
  template <>                                                                  \
  inline KRT_API void vscatter(void* dst, const FROM_TYPE& value,              \
                               const OFFSET_TYPE(&offset)[1]) {                \
    vscatter(dst, value, offset[0]);                                           \
  }

VSCATTER_DIRECT_IMPL(va16f32, f32, va16u32);
VSCATTER_DIRECT_IMPL(va16f32, f32, va16i32);
VSCATTER_DIRECT_IMPL(va16u32, u32, va16u32);
VSCATTER_DIRECT_IMPL(va16u32, u32, va16i32);
VSCATTER_DIRECT_IMPL(va16i32, s32, va16u32);
VSCATTER_DIRECT_IMPL(va16i32, s32, va16i32);
VSCATTER_DIRECT_IMPL(va16f32x2, dual_f32, va16u32x2);
VSCATTER_DIRECT_IMPL(va16f32x2, dual_f32, va16i32x2);
VSCATTER_DIRECT_IMPL(va16u32x2, dual_u32, va16u32x2);
VSCATTER_DIRECT_IMPL(va16u32x2, dual_u32, va16i32x2);
VSCATTER_DIRECT_IMPL(va16i32x2, dual_s32, va16u32x2);
VSCATTER_DIRECT_IMPL(va16i32x2, dual_s32, va16i32x2);
VSCATTER_DIRECT_IMPL(va32f16, f16, va16u32x2);
VSCATTER_DIRECT_IMPL(va32f16, f16, va16i32x2);
VSCATTER_DIRECT_IMPL(va32bf16, bf16, va16i32x2);
VSCATTER_DIRECT_IMPL(va32bf16, bf16, va16u32x2);
VSCATTER_DIRECT_IMPL(va32u16, u16, va16u32x2);
VSCATTER_DIRECT_IMPL(va32u16, u16, va16i32x2);
VSCATTER_DIRECT_IMPL(va32i16, s16, va16u32x2);
VSCATTER_DIRECT_IMPL(va32i16, s16, va16i32x2);
#undef VSCATTER_DIRECT_IMPL

#define VSCATTER_BPE1_IMPL(FROM_TYPE, FROM_SUFFIX, OFFSET_TYPE)        \
  template <>                                                          \
  inline KRT_API void vscatter(void* dst, const FROM_TYPE& value,      \
                               const OFFSET_TYPE& offset) {            \
    using IntType = vector_to_scalar<OFFSET_TYPE>::type;               \
    constexpr int OFF_LENGTH = vector_length<OFFSET_TYPE>::value;      \
    using DaIntType = scalar_to_vector<IntType, OFF_LENGTH / 2>::type; \
    DaIntType offsets[2];                                              \
    offsets[0] = vunpack0<DaIntType>(offset);                          \
    offsets[1] = vunpack1<DaIntType>(offset);                          \
    __dtu_m_vstxda_##FROM_SUFFIX(value, (__DTU_INTRIN_AS__ char*)dst,  \
                                 offsets[0], 0);                       \
    __dtu_m_vstxda_##FROM_SUFFIX(value, (__DTU_INTRIN_AS__ char*)dst,  \
                                 offsets[1], 1);                       \
  }                                                                    \
  template <>                                                          \
  inline KRT_API void vscatter(void* dst, const FROM_TYPE& value,      \
                               const OFFSET_TYPE(&offset)[1]) {        \
    vscatter(dst, value, offset[0]);                                   \
  }

VSCATTER_BPE1_IMPL(va64u8, u8, va16u32x4);
VSCATTER_BPE1_IMPL(va64u8, u8, va16i32x4);
VSCATTER_BPE1_IMPL(va64i8, s8, va16u32x4);
VSCATTER_BPE1_IMPL(va64i8, s8, va16i32x4);
#undef VSCATTER_BPE1_IMPL

#define VSCATTER_DUAL_IMPL(FROM_TYPE, OFFSET_TYPE)                         \
  template <>                                                              \
  inline KRT_API void vscatter(void* dst, const FROM_TYPE& value,          \
                               const OFFSET_TYPE& offset) {                \
    using ValueType = vector_to_scalar<FROM_TYPE>::type;                   \
    using IntType = vector_to_scalar<OFFSET_TYPE>::type;                   \
    constexpr int VEC_LENGTH = vector_length<FROM_TYPE>::value;            \
    constexpr int OFF_LENGTH = vector_length<OFFSET_TYPE>::value;          \
    using HalfVecType = scalar_to_vector<ValueType, VEC_LENGTH / 2>::type; \
    using HalfIntType = scalar_to_vector<IntType, OFF_LENGTH / 2>::type;   \
    HalfIntType offsets[2];                                                \
    offsets[0] = vunpack0<HalfIntType>(offset);                            \
    offsets[1] = vunpack1<HalfIntType>(offset);                            \
    HalfVecType values[2];                                                 \
    values[0] = vunpack0<HalfVecType>(value);                              \
    values[1] = vunpack1<HalfVecType>(value);                              \
    vscatter(dst, values[0], offsets[0]);                                  \
    vscatter(dst, values[1], offsets[1]);                                  \
  }                                                                        \
  template <>                                                              \
  inline KRT_API void vscatter(void* dst, const FROM_TYPE& value,          \
                               const OFFSET_TYPE(&offset)[1]) {            \
    vscatter(dst, value, offset[0]);                                       \
  }

VSCATTER_DUAL_IMPL(va16f32x4, va16u32x4);
VSCATTER_DUAL_IMPL(va16f32x4, va16i32x4);
VSCATTER_DUAL_IMPL(va16u32x4, va16u32x4);
VSCATTER_DUAL_IMPL(va16u32x4, va16i32x4);
VSCATTER_DUAL_IMPL(va16i32x4, va16u32x4);
VSCATTER_DUAL_IMPL(va16i32x4, va16i32x4);
VSCATTER_DUAL_IMPL(va32f16x2, va16u32x4);
VSCATTER_DUAL_IMPL(va32f16x2, va16i32x4);
VSCATTER_DUAL_IMPL(va32bf16x2, va16u32x4);
VSCATTER_DUAL_IMPL(va32bf16x2, va16i32x4);
VSCATTER_DUAL_IMPL(va32u16x2, va16u32x4);
VSCATTER_DUAL_IMPL(va32u16x2, va16i32x4);
VSCATTER_DUAL_IMPL(va32i16x2, va16u32x4);
VSCATTER_DUAL_IMPL(va32i16x2, va16i32x4);
#undef VSCATTER_DUAL_IMPL

#define VSCATTER_DUAL_ARRAY_IMPL(FROM_TYPE, OFFSET_TYPE)                   \
  template <>                                                              \
  inline KRT_API void vscatter(void* dst, const FROM_TYPE& value,          \
                               const OFFSET_TYPE(&offsets)[2]) {           \
    using ValueType = vector_to_scalar<FROM_TYPE>::type;                   \
    constexpr int VEC_LENGTH = vector_length<FROM_TYPE>::value;            \
    using HalfVecType = scalar_to_vector<ValueType, VEC_LENGTH / 2>::type; \
    HalfVecType values[2];                                                 \
    values[0] = vunpack0<HalfVecType>(value);                              \
    values[1] = vunpack1<HalfVecType>(value);                              \
    vscatter(dst, values[0], offsets[0]);                                  \
    vscatter(dst, values[1], offsets[1]);                                  \
  }

VSCATTER_DUAL_ARRAY_IMPL(va32f16x4, va16u32x4);
VSCATTER_DUAL_ARRAY_IMPL(va32f16x4, va16i32x4);
VSCATTER_DUAL_ARRAY_IMPL(va32bf16x4, va16u32x4);
VSCATTER_DUAL_ARRAY_IMPL(va32bf16x4, va16i32x4);
VSCATTER_DUAL_ARRAY_IMPL(va32u16x4, va16u32x4);
VSCATTER_DUAL_ARRAY_IMPL(va32u16x4, va16i32x4);
VSCATTER_DUAL_ARRAY_IMPL(va32i16x4, va16u32x4);
VSCATTER_DUAL_ARRAY_IMPL(va32i16x4, va16i32x4);
VSCATTER_DUAL_ARRAY_IMPL(va64u8x2, va16u32x4);
VSCATTER_DUAL_ARRAY_IMPL(va64u8x2, va16i32x4);
VSCATTER_DUAL_ARRAY_IMPL(va64i8x2, va16u32x4);
VSCATTER_DUAL_ARRAY_IMPL(va64i8x2, va16i32x4);
#undef VSCATTER_DUAL_ARRAY_IMPL

#define VSCATTER_QUAD_ARRAY_IMPL(FROM_TYPE, OFFSET_TYPE)                      \
  template <>                                                                 \
  inline KRT_API void vscatter(void* dst, const FROM_TYPE& value,             \
                               const OFFSET_TYPE(&offsets)[4]) {              \
    using ValueType = vector_to_scalar<FROM_TYPE>::type;                      \
    constexpr int VEC_LENGTH = vector_length<FROM_TYPE>::value;               \
    using QuarterVecType = scalar_to_vector<ValueType, VEC_LENGTH / 4>::type; \
    QuarterVecType values[4];                                                 \
    values[0] = vunpack0<QuarterVecType>(value);                              \
    values[1] = vunpack1<QuarterVecType>(value);                              \
    values[2] = vunpack2<QuarterVecType>(value);                              \
    values[3] = vunpack3<QuarterVecType>(value);                              \
    vscatter(dst, values[0], offsets[0]);                                     \
    vscatter(dst, values[1], offsets[1]);                                     \
    vscatter(dst, values[2], offsets[2]);                                     \
    vscatter(dst, values[3], offsets[3]);                                     \
  }

VSCATTER_QUAD_ARRAY_IMPL(va64u8x4, va16u32x4);
VSCATTER_QUAD_ARRAY_IMPL(va64u8x4, va16i32x4);
VSCATTER_QUAD_ARRAY_IMPL(va64i8x4, va16u32x4);
VSCATTER_QUAD_ARRAY_IMPL(va64i8x4, va16i32x4);
#undef VSCATTER_QUAD_ARRAY_IMPL
#endif  // __KRT_ARCH__ >= 300

/* ==========================================================================
                             vscatter with maskbit
   ========================================================================== */
template <typename MB_TYPE, typename FROM_TYPE, typename OFFSET_TYPE>
inline KRT_API void vscatter_t(const MB_TYPE& mb, void* dst,
                               const FROM_TYPE& value,
                               const OFFSET_TYPE& offset);

template <typename MB_TYPE, typename FROM_TYPE, typename OFFSET_TYPE>
inline KRT_API void vscatter_f(const MB_TYPE& mb, void* dst,
                               const FROM_TYPE& value,
                               const OFFSET_TYPE& offset) {
  auto mask = mask_not(mb);
  return vscatter_t(mask, dst, value, offset);
}

template <typename MB_TYPE, typename FROM_TYPE, typename OFFSET_TYPE>
inline KRT_API void vscatter_t(const MB_TYPE& mb, void* dst,
                               const FROM_TYPE& value,
                               const OFFSET_TYPE& offset) {
  static_assert(
      std::is_integral<typename vector_to_scalar<OFFSET_TYPE>::type>::value &&
          sizeof(typename vector_to_scalar<OFFSET_TYPE>::type) == 4,
      "OFFSET_TYPE must be u32/i32 type");
  static_assert(
      vector_length<FROM_TYPE>::value == vector_length<OFFSET_TYPE>::value &&
          vector_length<FROM_TYPE>::value == vector_length<MB_TYPE>::value,
      "vector length of FROM_TYPE, OFFSET_TYPE and MB_TYPE must be equal");
}

#if (__KRT_ARCH__ >= 300)
#define VSCATTER_T_DIRECT_IMPL(FROM_TYPE, FROM_SUFFIX, MB_TYPE, OFFSET_TYPE)   \
  template <>                                                                  \
  inline KRT_API void vscatter_t(const MB_TYPE& mb, void* dst,                 \
                                 const FROM_TYPE& value,                       \
                                 const OFFSET_TYPE& offset) {                  \
    using ValueType = vector_to_scalar<FROM_TYPE>::type;                       \
    using QaType = scalar_to_vector<ValueType, TOPS_VECTOR_LENGTH * 4 /        \
                                                   sizeof(ValueType)>::type;   \
    using MaskType = vector_to_mask<QaType>::type;                             \
    auto mask = mask_bitcast<MaskType>(mb);                                    \
    __dtu_m_vstxda_##FROM_SUFFIX##_vs0_vm(value, (__DTU_INTRIN_AS__ char*)dst, \
                                          offset, mask);                       \
  }                                                                            \
  template <>                                                                  \
  inline KRT_API void vscatter_t(const MB_TYPE& mb, void* dst,                 \
                                 const FROM_TYPE& value,                       \
                                 const OFFSET_TYPE(&offset)[1]) {              \
    vscatter_t(mb, dst, value, offset[0]);                                     \
  }

VSCATTER_T_DIRECT_IMPL(va16f32, f32, vbool16_t, va16u32);
VSCATTER_T_DIRECT_IMPL(va16f32, f32, vbool16_t, va16i32);
VSCATTER_T_DIRECT_IMPL(va16u32, u32, vbool16_t, va16u32);
VSCATTER_T_DIRECT_IMPL(va16u32, u32, vbool16_t, va16i32);
VSCATTER_T_DIRECT_IMPL(va16i32, s32, vbool16_t, va16u32);
VSCATTER_T_DIRECT_IMPL(va16i32, s32, vbool16_t, va16i32);
VSCATTER_T_DIRECT_IMPL(va16f32x2, dual_f32, vbool32_t, va16u32x2);
VSCATTER_T_DIRECT_IMPL(va16f32x2, dual_f32, vbool32_t, va16i32x2);
VSCATTER_T_DIRECT_IMPL(va16u32x2, dual_u32, vbool32_t, va16u32x2);
VSCATTER_T_DIRECT_IMPL(va16u32x2, dual_u32, vbool32_t, va16i32x2);
VSCATTER_T_DIRECT_IMPL(va16i32x2, dual_s32, vbool32_t, va16u32x2);
VSCATTER_T_DIRECT_IMPL(va16i32x2, dual_s32, vbool32_t, va16i32x2);
VSCATTER_T_DIRECT_IMPL(va32f16, f16, vbool32_t, va16u32x2);
VSCATTER_T_DIRECT_IMPL(va32f16, f16, vbool32_t, va16i32x2);
VSCATTER_T_DIRECT_IMPL(va32bf16, bf16, vbool32_t, va16u32x2);
VSCATTER_T_DIRECT_IMPL(va32bf16, bf16, vbool32_t, va16i32x2);
VSCATTER_T_DIRECT_IMPL(va32u16, u16, vbool32_t, va16u32x2);
VSCATTER_T_DIRECT_IMPL(va32u16, u16, vbool32_t, va16i32x2);
VSCATTER_T_DIRECT_IMPL(va32i16, s16, vbool32_t, va16u32x2);
VSCATTER_T_DIRECT_IMPL(va32i16, s16, vbool32_t, va16i32x2);
#undef VSCATTER_T_DIRECT_IMPL

#define VSCATTER_T_BPE1_IMPL(VA_TYPE, FROM_SUFFIX, OFFSET_TYPE)                \
  template <>                                                                  \
  inline KRT_API void vscatter_t(const vbool64_t& mb, void* dst,               \
                                 const VA_TYPE& value,                         \
                                 const OFFSET_TYPE& offset) {                  \
    using FROM_TYPE = VA_TYPE;                                                 \
    using ValueType = vector_to_scalar<FROM_TYPE>::type;                       \
    using IntType = vector_to_scalar<OFFSET_TYPE>::type;                       \
    constexpr int OFF_LENGTH = vector_length<OFFSET_TYPE>::value;              \
    using HalfIntType = scalar_to_vector<IntType, OFF_LENGTH / 2>::type;       \
    using QaType = scalar_to_vector<ValueType, TOPS_VECTOR_LENGTH * 4 /        \
                                                   sizeof(ValueType)>::type;   \
    using MaskType = vector_to_mask<QaType>::type;                             \
    auto mask = mask_bitcast<MaskType>(mb);                                    \
    HalfIntType offsets[2];                                                    \
    offsets[0] = vunpack0<HalfIntType>(offset);                                \
    offsets[1] = vunpack1<HalfIntType>(offset);                                \
    __dtu_m_vstxda_##FROM_SUFFIX##_vs0_vm(value, (__DTU_INTRIN_AS__ char*)dst, \
                                          offsets[0], 0, mask);                \
    __dtu_m_vstxda_##FROM_SUFFIX##_vs0_vm(value, (__DTU_INTRIN_AS__ char*)dst, \
                                          offsets[1], 1, mask);                \
  }                                                                            \
  template <>                                                                  \
  inline KRT_API void vscatter_t(const vbool64_t& mb, void* dst,               \
                                 const VA_TYPE& value,                         \
                                 const OFFSET_TYPE(&offset)[1]) {              \
    vscatter_t(mb, dst, value, offset[0]);                                     \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  inline KRT_API void vscatter_t(const vbool128_t& mb, void* dst,              \
                                 const VA_TYPE##x2& value,                     \
                                 const OFFSET_TYPE(&offset)[2]) {              \
    using FROM_TYPE = VA_TYPE##x2;                                             \
    using ValueType = vector_to_scalar<FROM_TYPE>::type;                       \
    using IntType = vector_to_scalar<OFFSET_TYPE>::type;                       \
    constexpr int VEC_LENGTH = vector_length<FROM_TYPE>::value;                \
    constexpr int OFF_LENGTH = vector_length<OFFSET_TYPE>::value;              \
    using HalfVecType = scalar_to_vector<ValueType, VEC_LENGTH / 2>::type;     \
    using HalfIntType = scalar_to_vector<IntType, OFF_LENGTH / 2>::type;       \
    using QaType = scalar_to_vector<ValueType, TOPS_VECTOR_LENGTH * 4 /        \
                                                   sizeof(ValueType)>::type;   \
    using MaskType = vector_to_mask<QaType>::type;                             \
    auto mask = mask_bitcast<MaskType>(mb);                                    \
    HalfVecType values[2];                                                     \
    values[0] = vunpack0<HalfVecType>(value);                                  \
    values[1] = vunpack1<HalfVecType>(value);                                  \
    HalfIntType offsets[4];                                                    \
    offsets[0] = vunpack0<HalfIntType>(offset[0]);                             \
    offsets[1] = vunpack1<HalfIntType>(offset[0]);                             \
    offsets[2] = vunpack0<HalfIntType>(offset[1]);                             \
    offsets[3] = vunpack1<HalfIntType>(offset[1]);                             \
    __dtu_m_vstxda_##FROM_SUFFIX##_vs0_vm(                                     \
        values[0], (__DTU_INTRIN_AS__ char*)dst, offsets[0], 0, mask);         \
    __dtu_m_vstxda_##FROM_SUFFIX##_vs0_vm(                                     \
        values[0], (__DTU_INTRIN_AS__ char*)dst, offsets[1], 1, mask);         \
    __dtu_m_vstxda_##FROM_SUFFIX##_vs1_vm(                                     \
        values[1], (__DTU_INTRIN_AS__ char*)dst, offsets[2], 0, mask);         \
    __dtu_m_vstxda_##FROM_SUFFIX##_vs1_vm(                                     \
        values[1], (__DTU_INTRIN_AS__ char*)dst, offsets[3], 1, mask);         \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  inline KRT_API void vscatter_t(const vbool256_t& mb, void* dst,              \
                                 const VA_TYPE##x4& value,                     \
                                 const OFFSET_TYPE(&offset)[4]) {              \
    using FROM_TYPE = VA_TYPE##x4;                                             \
    using ValueType = vector_to_scalar<FROM_TYPE>::type;                       \
    using IntType = vector_to_scalar<OFFSET_TYPE>::type;                       \
    constexpr int VEC_LENGTH = vector_length<FROM_TYPE>::value;                \
    constexpr int OFF_LENGTH = vector_length<OFFSET_TYPE>::value;              \
    using QuarterVecType = scalar_to_vector<ValueType, VEC_LENGTH / 4>::type;  \
    using HalfIntType = scalar_to_vector<IntType, OFF_LENGTH / 2>::type;       \
    QuarterVecType values[4];                                                  \
    values[0] = vunpack0<QuarterVecType>(value);                               \
    values[1] = vunpack1<QuarterVecType>(value);                               \
    values[2] = vunpack2<QuarterVecType>(value);                               \
    values[3] = vunpack3<QuarterVecType>(value);                               \
    HalfIntType offsets[8];                                                    \
    offsets[0] = vunpack0<HalfIntType>(offset[0]);                             \
    offsets[1] = vunpack1<HalfIntType>(offset[0]);                             \
    offsets[2] = vunpack0<HalfIntType>(offset[1]);                             \
    offsets[3] = vunpack1<HalfIntType>(offset[1]);                             \
    offsets[4] = vunpack0<HalfIntType>(offset[2]);                             \
    offsets[5] = vunpack1<HalfIntType>(offset[2]);                             \
    offsets[6] = vunpack0<HalfIntType>(offset[3]);                             \
    offsets[7] = vunpack1<HalfIntType>(offset[3]);                             \
    __dtu_m_vstxda_##FROM_SUFFIX##_vs0_vm(                                     \
        values[0], (__DTU_INTRIN_AS__ char*)dst, offsets[0], 0, mb);           \
    __dtu_m_vstxda_##FROM_SUFFIX##_vs0_vm(                                     \
        values[0], (__DTU_INTRIN_AS__ char*)dst, offsets[1], 1, mb);           \
    __dtu_m_vstxda_##FROM_SUFFIX##_vs1_vm(                                     \
        values[1], (__DTU_INTRIN_AS__ char*)dst, offsets[2], 0, mb);           \
    __dtu_m_vstxda_##FROM_SUFFIX##_vs1_vm(                                     \
        values[1], (__DTU_INTRIN_AS__ char*)dst, offsets[3], 1, mb);           \
    __dtu_m_vstxda_##FROM_SUFFIX##_vs2_vm(                                     \
        values[2], (__DTU_INTRIN_AS__ char*)dst, offsets[4], 0, mb);           \
    __dtu_m_vstxda_##FROM_SUFFIX##_vs2_vm(                                     \
        values[2], (__DTU_INTRIN_AS__ char*)dst, offsets[5], 1, mb);           \
    __dtu_m_vstxda_##FROM_SUFFIX##_vs3_vm(                                     \
        values[3], (__DTU_INTRIN_AS__ char*)dst, offsets[6], 0, mb);           \
    __dtu_m_vstxda_##FROM_SUFFIX##_vs3_vm(                                     \
        values[3], (__DTU_INTRIN_AS__ char*)dst, offsets[7], 1, mb);           \
  }

VSCATTER_T_BPE1_IMPL(va64u8, u8, va16u32x4);
VSCATTER_T_BPE1_IMPL(va64u8, u8, va16i32x4);
VSCATTER_T_BPE1_IMPL(va64i8, s8, va16u32x4);
VSCATTER_T_BPE1_IMPL(va64i8, s8, va16i32x4);
#undef VSCATTER_T_BPE1_IMPL

#define VSCATTER_T_DUAL_IMPL(FROM_TYPE, FROM_SUFFIX, MB_TYPE, OFFSET_TYPE)   \
  template <>                                                                \
  inline KRT_API void vscatter_t(const MB_TYPE& mb, void* dst,               \
                                 const FROM_TYPE& value,                     \
                                 const OFFSET_TYPE& offset) {                \
    using ValueType = vector_to_scalar<FROM_TYPE>::type;                     \
    using IntType = vector_to_scalar<OFFSET_TYPE>::type;                     \
    constexpr int VEC_LENGTH = vector_length<FROM_TYPE>::value;              \
    using HalfVecType = scalar_to_vector<ValueType, VEC_LENGTH / 2>::type;   \
    using HalfIntType = scalar_to_vector<IntType, VEC_LENGTH / 2>::type;     \
    using QaType = scalar_to_vector<ValueType, TOPS_VECTOR_LENGTH * 4 /      \
                                                   sizeof(ValueType)>::type; \
    using MaskType = vector_to_mask<QaType>::type;                           \
    auto mask = mask_bitcast<MaskType>(mb);                                  \
    HalfIntType offsets[2];                                                  \
    offsets[0] = vunpack0<HalfIntType>(offset);                              \
    offsets[1] = vunpack1<HalfIntType>(offset);                              \
    HalfVecType values[2];                                                   \
    values[0] = vunpack0<HalfVecType>(value);                                \
    values[1] = vunpack1<HalfVecType>(value);                                \
    __dtu_m_vstxda_##FROM_SUFFIX##_vs0_vm(                                   \
        values[0], (__DTU_INTRIN_AS__ char*)dst, offsets[0], mask);          \
    __dtu_m_vstxda_##FROM_SUFFIX##_vs1_vm(                                   \
        values[1], (__DTU_INTRIN_AS__ char*)dst, offsets[1], mask);          \
  }                                                                          \
  template <>                                                                \
  inline KRT_API void vscatter_t(const MB_TYPE& mb, void* dst,               \
                                 const FROM_TYPE& value,                     \
                                 const OFFSET_TYPE(&offset)[1]) {            \
    vscatter_t(mb, dst, value, offset[0]);                                   \
  }

VSCATTER_T_DUAL_IMPL(va16f32x4, dual_f32, vbool64_t, va16u32x4);
VSCATTER_T_DUAL_IMPL(va16f32x4, dual_f32, vbool64_t, va16i32x4);
VSCATTER_T_DUAL_IMPL(va16u32x4, dual_u32, vbool64_t, va16u32x4);
VSCATTER_T_DUAL_IMPL(va16u32x4, dual_u32, vbool64_t, va16i32x4);
VSCATTER_T_DUAL_IMPL(va16i32x4, dual_s32, vbool64_t, va16u32x4);
VSCATTER_T_DUAL_IMPL(va16i32x4, dual_s32, vbool64_t, va16i32x4);
VSCATTER_T_DUAL_IMPL(va32f16x2, f16, vbool64_t, va16u32x4);
VSCATTER_T_DUAL_IMPL(va32f16x2, f16, vbool64_t, va16i32x4);
VSCATTER_T_DUAL_IMPL(va32bf16x2, bf16, vbool64_t, va16u32x4);
VSCATTER_T_DUAL_IMPL(va32bf16x2, bf16, vbool64_t, va16i32x4);
VSCATTER_T_DUAL_IMPL(va32u16x2, u16, vbool64_t, va16u32x4);
VSCATTER_T_DUAL_IMPL(va32u16x2, u16, vbool64_t, va16i32x4);
VSCATTER_T_DUAL_IMPL(va32i16x2, s16, vbool64_t, va16u32x4);
VSCATTER_T_DUAL_IMPL(va32i16x2, s16, vbool64_t, va16i32x4);
#undef VSCATTER_T_DUAL_IMPL

#define VSCATTER_T_QUAD_ARRAY2_IMPL(FROM_TYPE, FROM_SUFFIX, MB_TYPE,          \
                                    OFFSET_TYPE)                              \
  template <>                                                                 \
  inline KRT_API void vscatter_t(const MB_TYPE& mb, void* dst,                \
                                 const FROM_TYPE& value,                      \
                                 const OFFSET_TYPE(&offset)[2]) {             \
    using ValueType = vector_to_scalar<FROM_TYPE>::type;                      \
    using IntType = vector_to_scalar<OFFSET_TYPE>::type;                      \
    constexpr int VEC_LENGTH = vector_length<FROM_TYPE>::value;               \
    constexpr int OFF_LENGTH = vector_length<OFFSET_TYPE>::value;             \
    using QuarterVecType = scalar_to_vector<ValueType, VEC_LENGTH / 4>::type; \
    using HalfIntType = scalar_to_vector<IntType, OFF_LENGTH / 2>::type;      \
    HalfIntType offsets[4];                                                   \
    offsets[0] = vunpack0<HalfIntType>(offset[0]);                            \
    offsets[1] = vunpack1<HalfIntType>(offset[0]);                            \
    offsets[2] = vunpack0<HalfIntType>(offset[1]);                            \
    offsets[3] = vunpack1<HalfIntType>(offset[1]);                            \
    QuarterVecType values[4];                                                 \
    values[0] = vunpack0<QuarterVecType>(value);                              \
    values[1] = vunpack1<QuarterVecType>(value);                              \
    values[2] = vunpack2<QuarterVecType>(value);                              \
    values[3] = vunpack3<QuarterVecType>(value);                              \
    __dtu_m_vstxda_##FROM_SUFFIX##_vs0_vm(                                    \
        values[0], (__DTU_INTRIN_AS__ char*)dst, offsets[0], mb);             \
    __dtu_m_vstxda_##FROM_SUFFIX##_vs1_vm(                                    \
        values[1], (__DTU_INTRIN_AS__ char*)dst, offsets[1], mb);             \
    __dtu_m_vstxda_##FROM_SUFFIX##_vs2_vm(                                    \
        values[2], (__DTU_INTRIN_AS__ char*)dst, offsets[2], mb);             \
    __dtu_m_vstxda_##FROM_SUFFIX##_vs3_vm(                                    \
        values[3], (__DTU_INTRIN_AS__ char*)dst, offsets[3], mb);             \
  }

VSCATTER_T_QUAD_ARRAY2_IMPL(va32f16x4, f16, vbool128_t, va16u32x4);
VSCATTER_T_QUAD_ARRAY2_IMPL(va32f16x4, f16, vbool128_t, va16i32x4);
VSCATTER_T_QUAD_ARRAY2_IMPL(va32bf16x4, bf16, vbool128_t, va16u32x4);
VSCATTER_T_QUAD_ARRAY2_IMPL(va32bf16x4, bf16, vbool128_t, va16i32x4);
VSCATTER_T_QUAD_ARRAY2_IMPL(va32u16x4, u16, vbool128_t, va16u32x4);
VSCATTER_T_QUAD_ARRAY2_IMPL(va32u16x4, u16, vbool128_t, va16i32x4);
VSCATTER_T_QUAD_ARRAY2_IMPL(va32i16x4, s16, vbool128_t, va16u32x4);
VSCATTER_T_QUAD_ARRAY2_IMPL(va32i16x4, s16, vbool128_t, va16i32x4);
#undef VSCATTER_T_QUAD_ARRAY2_IMPL
#endif  // __KRT_ARCH__ >= 300

}  // namespace tops

#endif  // CC_KERNEL_VECTOR_EX_H
