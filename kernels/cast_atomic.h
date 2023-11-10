
#include <tops.h>
#include <tops/topsrtc.h>
#include <tops/half.h>
#include <tops/bfloat.h>
#if __GCU_ARCH__ >= 300
#include "sip30intrin.h"
#include "include/common/atomic_op.h"
// #include "/home/kernel/atomicop_pkg/src/include/common/atomic_op.h"
#endif

namespace {
#include <krt/vector_mask.h>
#include <krt/dispatch.h>
#include <krt/leaptr.h>
}

#include <tops/tops_runtime.h>
using namespace tops;

template <typename FROM_TYPE = char, typename TO_TYPE = char,
          RoundingMode_t RM_MODE = RoundingMode_t::RM_RZ>
__device__ void __attribute__((always_inline))
cast_atomic(TO_TYPE* dst_addr, FROM_TYPE* src_addr, unsigned int size) {
  // ...
}

#define DISPATCH_L1_CONVERT_OP(FROM_TYPE, TO_TYPE, RM_MODE, op_name,           \
                               dataflow_func, ...)                             \
  template <>                                                                  \
  __device__ void __attribute__((always_inline))                                          \
      cast_atomic<FROM_TYPE, TO_TYPE, RoundingMode_t::RM_MODE>(                    \
          TO_TYPE * dst_addr, FROM_TYPE * src_addr, unsigned int size) {       \
    constexpr int from_bpe = sizeof(FROM_TYPE);                                \
    constexpr int from_vec_elems =                                             \
        sizeof(typename unified_scalar<FROM_TYPE>::type) *                     \
        TOPS_VECTOR_LENGTH / from_bpe;                                         \
    constexpr int to_bpe = sizeof(TO_TYPE);                                    \
    constexpr int to_vec_elems =                                               \
        sizeof(typename unified_scalar<TO_TYPE>::type) * TOPS_VECTOR_LENGTH /  \
        to_bpe;                                                                \
    constexpr int real_from_vec_elems =                                        \
        (from_bpe > to_bpe) ? from_vec_elems                                   \
                            : from_vec_elems * from_bpe / to_bpe;              \
    constexpr int real_to_vec_elems =                                          \
        (from_bpe > to_bpe) ? to_vec_elems * to_bpe / from_bpe : to_vec_elems; \
    using from_vtype =                                                         \
        typename scalar_to_vector<FROM_TYPE, real_from_vec_elems>::type;       \
    using to_vtype =                                                           \
        typename scalar_to_vector<TO_TYPE, real_to_vec_elems>::type;           \
    auto kernel_level0 = [&](from_vtype vsrc) {                                \
      to_vtype vdst = vcast<to_vtype>(vsrc);                                   \
      return vdst;                                                             \
    };                                                                         \
    dataflow_func(dst_addr, src_addr, size, kernel_level0);                    \
  }

#define ALL_CONVERT_OP_TYPES(_, ...) _(convert, __VA_ARGS__)

#define bfloat __bf16 

#define CONVERT_FUNC_TYPES(_, ...)                                \
  _(char, short, RM_RZ, __VA_ARGS__)                           \
  _(char, short, RM_RZ_CLAMP, __VA_ARGS__)                     \
  _(char, int, RM_RZ, __VA_ARGS__)                             \
  _(char, int, RM_RZ_CLAMP, __VA_ARGS__)                       \
  _(char, char, RM_RZ, __VA_ARGS__)                            \
  _(char, char, RM_RZ_CLAMP, __VA_ARGS__)                      \
  _(char, __half, RM_RZ, __VA_ARGS__)                          \
  _(char, __half, RM_RN, __VA_ARGS__)                          \
  _(char, __half, RM_RZ_CLAMP, __VA_ARGS__)                    \
  _(char, __half, RM_RN_CLAMP, __VA_ARGS__)                    \
  _(char, float, RM_RZ, __VA_ARGS__)                           \
  _(char, float, RM_RN, __VA_ARGS__)                           \
  _(char, float, RM_RZ_CLAMP, __VA_ARGS__)                     \
  _(char, float, RM_RN_CLAMP, __VA_ARGS__)                     \
  _(short, char, RM_RZ, __VA_ARGS__)                           \
  _(short, char, RM_RZ_CLAMP, __VA_ARGS__)                     \
  _(short, int, RM_RZ, __VA_ARGS__)                           \
  _(short, int, RM_RZ_CLAMP, __VA_ARGS__)                     \
  _(short, short, RM_RZ, __VA_ARGS__)                         \
  _(short, short, RM_RZ_CLAMP, __VA_ARGS__)                   \
  _(short, __half, RM_RZ, __VA_ARGS__)                        \
  _(short, __half, RM_RN, __VA_ARGS__)                        \
  _(short, __half, RM_RZ_CLAMP, __VA_ARGS__)                  \
  _(short, __half, RM_RN_CLAMP, __VA_ARGS__)                  \
  _(short, float, RM_RZ, __VA_ARGS__)                         \
  _(short, float, RM_RN, __VA_ARGS__)                         \
  _(short, float, RM_RZ_CLAMP, __VA_ARGS__)                   \
  _(short, float, RM_RN_CLAMP, __VA_ARGS__)                   \
  _(int, char, RM_RZ, __VA_ARGS__)                            \
  _(int, char, RM_RZ_CLAMP, __VA_ARGS__)                      \
  _(int, short, RM_RZ, __VA_ARGS__)                           \
  _(int, short, RM_RZ_CLAMP, __VA_ARGS__)                     \
  _(int, int, RM_RZ, __VA_ARGS__)                             \
  _(int, int, RM_RZ_CLAMP, __VA_ARGS__)                       \
  _(int, __half, RM_RZ, __VA_ARGS__)                          \
  _(int, __half, RM_RN, __VA_ARGS__)                          \
  _(int, __half, RM_RZ_CLAMP, __VA_ARGS__)                    \
  _(int, __half, RM_RN_CLAMP, __VA_ARGS__)                    \
  _(int, float, RM_RZ, __VA_ARGS__)                           \
  _(int, float, RM_RN, __VA_ARGS__)                           \
  _(int, float, RM_RZ_CLAMP, __VA_ARGS__)                     \
  _(int, float, RM_RN_CLAMP, __VA_ARGS__)                     \
  _(unsigned char, unsigned char, RM_RZ, __VA_ARGS__)         \
  _(unsigned char, unsigned char, RM_RZ_CLAMP, __VA_ARGS__)   \
  _(unsigned char, unsigned short, RM_RZ, __VA_ARGS__)        \
  _(unsigned char, unsigned short, RM_RZ_CLAMP, __VA_ARGS__)  \
  _(unsigned char, unsigned int, RM_RZ, __VA_ARGS__)          \
  _(unsigned char, unsigned int, RM_RZ_CLAMP, __VA_ARGS__)    \
  _(unsigned char, __half, RM_RZ, __VA_ARGS__)                \
  _(unsigned char, __half, RM_RN, __VA_ARGS__)                \
  _(unsigned char, __half, RM_RZ_CLAMP, __VA_ARGS__)          \
  _(unsigned char, __half, RM_RN_CLAMP, __VA_ARGS__)          \
  _(unsigned char, float, RM_RZ, __VA_ARGS__)                 \
  _(unsigned char, float, RM_RN, __VA_ARGS__)                 \
  _(unsigned char, float, RM_RZ_CLAMP, __VA_ARGS__)           \
  _(unsigned char, float, RM_RN_CLAMP, __VA_ARGS__)           \
  _(unsigned short, unsigned char, RM_RZ, __VA_ARGS__)        \
  _(unsigned short, unsigned char, RM_RZ_CLAMP, __VA_ARGS__)  \
  _(unsigned short, unsigned short, RM_RZ, __VA_ARGS__)       \
  _(unsigned short, unsigned short, RM_RZ_CLAMP, __VA_ARGS__) \
  _(unsigned short, unsigned int, RM_RZ, __VA_ARGS__)         \
  _(unsigned short, unsigned int, RM_RZ_CLAMP, __VA_ARGS__)   \
  _(unsigned short, __half, RM_RZ, __VA_ARGS__)               \
  _(unsigned short, __half, RM_RN, __VA_ARGS__)               \
  _(unsigned short, __half, RM_RZ_CLAMP, __VA_ARGS__)         \
  _(unsigned short, __half, RM_RN_CLAMP, __VA_ARGS__)         \
  _(unsigned short, float, RM_RZ, __VA_ARGS__)                \
  _(unsigned short, float, RM_RN, __VA_ARGS__)                \
  _(unsigned short, float, RM_RZ_CLAMP, __VA_ARGS__)          \
  _(unsigned short, float, RM_RN_CLAMP, __VA_ARGS__)          \
  _(unsigned int, unsigned char, RM_RZ, __VA_ARGS__)          \
  _(unsigned int, unsigned char, RM_RZ_CLAMP, __VA_ARGS__)    \
  _(unsigned int, unsigned short, RM_RZ, __VA_ARGS__)         \
  _(unsigned int, unsigned short, RM_RZ_CLAMP, __VA_ARGS__)   \
  _(unsigned int, unsigned int, RM_RZ, __VA_ARGS__)           \
  _(unsigned int, unsigned int, RM_RZ_CLAMP, __VA_ARGS__)     \
  _(unsigned int, __half, RM_RZ, __VA_ARGS__)                 \
  _(unsigned int, __half, RM_RN, __VA_ARGS__)                 \
  _(unsigned int, __half, RM_RZ_CLAMP, __VA_ARGS__)           \
  _(unsigned int, __half, RM_RN_CLAMP, __VA_ARGS__)           \
  _(unsigned int, float, RM_RZ, __VA_ARGS__)                  \
  _(unsigned int, float, RM_RN, __VA_ARGS__)                  \
  _(unsigned int, float, RM_RZ_CLAMP, __VA_ARGS__)            \
  _(unsigned int, float, RM_RN_CLAMP, __VA_ARGS__)            \
  _(__half, char, RM_RZ, __VA_ARGS__)                         \
  _(__half, char, RM_RZ_CLAMP, __VA_ARGS__)                   \
  _(__half, char, RM_RN, __VA_ARGS__)                         \
  _(__half, char, RM_RN_CLAMP, __VA_ARGS__)                   \
  _(__half, short, RM_RZ, __VA_ARGS__)                        \
  _(__half, short, RM_RZ_CLAMP, __VA_ARGS__)                  \
  _(__half, short, RM_RN, __VA_ARGS__)                        \
  _(__half, short, RM_RN_CLAMP, __VA_ARGS__)                  \
  _(__half, int, RM_RZ, __VA_ARGS__)                          \
  _(__half, int, RM_RZ_CLAMP, __VA_ARGS__)                    \
  _(__half, int, RM_RN, __VA_ARGS__)                          \
  _(__half, int, RM_RN_CLAMP, __VA_ARGS__)                    \
  _(__half, unsigned char, RM_RZ, __VA_ARGS__)                \
  _(__half, unsigned char, RM_RZ_CLAMP, __VA_ARGS__)          \
  _(__half, unsigned char, RM_RN, __VA_ARGS__)                \
  _(__half, unsigned char, RM_RN_CLAMP, __VA_ARGS__)          \
  _(__half, float, RM_RZ, __VA_ARGS__)                        \
  _(__half, float, RM_RN, __VA_ARGS__)                        \
  _(__half, float, RM_RZ_CLAMP, __VA_ARGS__)                  \
  _(__half, float, RM_RN_CLAMP, __VA_ARGS__)                  \
  _(float, char, RM_RZ, __VA_ARGS__)                          \
  _(float, char, RM_RZ_CLAMP, __VA_ARGS__)                    \
  _(float, char, RM_RN, __VA_ARGS__)                          \
  _(float, char, RM_RN_CLAMP, __VA_ARGS__)                    \
  _(float, short, RM_RZ, __VA_ARGS__)                         \
  _(float, short, RM_RZ_CLAMP, __VA_ARGS__)                   \
  _(float, short, RM_RN, __VA_ARGS__)                         \
  _(float, short, RM_RN_CLAMP, __VA_ARGS__)                   \
  _(float, int, RM_RZ, __VA_ARGS__)                           \
  _(float, int, RM_RZ_CLAMP, __VA_ARGS__)                     \
  _(float, int, RM_RN, __VA_ARGS__)                           \
  _(float, int, RM_RN_CLAMP, __VA_ARGS__)                     \
  _(float, unsigned char, RM_RZ, __VA_ARGS__)                 \
  _(float, unsigned char, RM_RZ_CLAMP, __VA_ARGS__)           \
  _(float, unsigned char, RM_RN, __VA_ARGS__)                 \
  _(float, unsigned char, RM_RN_CLAMP, __VA_ARGS__)           \
  _(float, __half, RM_RZ, __VA_ARGS__)                        \
  _(float, __half, RM_RN, __VA_ARGS__)                        \
  _(float, __half, RM_RZ_CLAMP, __VA_ARGS__)                  \
  _(float, __half, RM_RN_CLAMP, __VA_ARGS__)
//krt error in compile bf16
//   _(char, __bf16, RM_RZ, __VA_ARGS__)                          \
//   _(char, __bf16, RM_RN, __VA_ARGS__)                          \
//   _(char, __bf16, RM_RZ_CLAMP, __VA_ARGS__)                    \
//   _(char, __bf16, RM_RN_CLAMP, __VA_ARGS__)                    \
//   _(float, __bf16, RM_RZ, __VA_ARGS__)                        \
//   _(float, __bf16, RM_RN, __VA_ARGS__)                        \
//   _(float, __bf16, RM_RZ_CLAMP, __VA_ARGS__)                  \
//   _(float, __bf16, RM_RN_CLAMP, __VA_ARGS__)                  \
//   _(__half, __bf16, RM_RZ, __VA_ARGS__)                       \
//   _(__half, __bf16, RM_RN, __VA_ARGS__)                       \
//   _(__half, __bf16, RM_RZ_CLAMP, __VA_ARGS__)                 \
//   _(__half, __bf16, RM_RN_CLAMP, __VA_ARGS__)                 \
//   _(__bf16, char, RM_RZ, __VA_ARGS__)                         \
//   _(__bf16, char, RM_RZ_CLAMP, __VA_ARGS__)                   \
//   _(__bf16, char, RM_RN, __VA_ARGS__)                         \
//   _(__bf16, char, RM_RN_CLAMP, __VA_ARGS__)                   \
//   _(__bf16, short, RM_RZ, __VA_ARGS__)                        \
//   _(__bf16, short, RM_RZ_CLAMP, __VA_ARGS__)                  \
//   _(__bf16, short, RM_RN, __VA_ARGS__)                        \
//   _(__bf16, short, RM_RN_CLAMP, __VA_ARGS__)                  \
//   _(__bf16, int, RM_RZ, __VA_ARGS__)                          \
//   _(__bf16, int, RM_RZ_CLAMP, __VA_ARGS__)                    \
//   _(__bf16, int, RM_RN, __VA_ARGS__)                          \
//   _(__bf16, int, RM_RN_CLAMP, __VA_ARGS__)                    \
//   _(__bf16, unsigned char, RM_RZ, __VA_ARGS__)                \
//   _(__bf16, unsigned char, RM_RZ_CLAMP, __VA_ARGS__)          \
//   _(__bf16, unsigned char, RM_RN, __VA_ARGS__)                \
//   _(__bf16, unsigned char, RM_RN_CLAMP, __VA_ARGS__)          \
//   _(__bf16, __half, RM_RZ, __VA_ARGS__)                       \
//   _(__bf16, __half, RM_RN, __VA_ARGS__)                       \
//   _(__bf16, __half, RM_RZ_CLAMP, __VA_ARGS__)                 \
//   _(__bf16, __half, RM_RN_CLAMP, __VA_ARGS__)                 \
//   _(__bf16, float, RM_RZ, __VA_ARGS__)                        \
//   _(__bf16, float, RM_RN, __VA_ARGS__)                        \
//   _(__bf16, float, RM_RZ_CLAMP, __VA_ARGS__)                  \
//   _(__bf16, float, RM_RN_CLAMP, __VA_ARGS__)                  \
//   _(unsigned int, __bf16, RM_RZ, __VA_ARGS__)                 \
//   _(unsigned int, __bf16, RM_RN, __VA_ARGS__)                 \
//   _(unsigned int, __bf16, RM_RZ_CLAMP, __VA_ARGS__)           \
//   _(unsigned int, __bf16, RM_RN_CLAMP, __VA_ARGS__)           \
//   _(unsigned short, __bf16, RM_RZ, __VA_ARGS__)               \
//   _(unsigned short, __bf16, RM_RN, __VA_ARGS__)               \
//   _(unsigned short, __bf16, RM_RZ_CLAMP, __VA_ARGS__)         \
//   _(unsigned short, __bf16, RM_RN_CLAMP, __VA_ARGS__)         \
//   _(short, __bf16, RM_RZ, __VA_ARGS__)                        \
//   _(short, __bf16, RM_RN, __VA_ARGS__)                        \
//   _(short, __bf16, RM_RZ_CLAMP, __VA_ARGS__)                  \
//   _(short, __bf16, RM_RN_CLAMP, __VA_ARGS__)                  \
//   _(int, __bf16, RM_RZ, __VA_ARGS__)                          \
//   _(int, __bf16, RM_RN, __VA_ARGS__)                          \
//   _(int, __bf16, RM_RZ_CLAMP, __VA_ARGS__)                    \
//   _(int, __bf16, RM_RN_CLAMP, __VA_ARGS__)                    \
//   _(unsigned char, __bf16, RM_RZ, __VA_ARGS__)                \
//   _(unsigned char, __bf16, RM_RN, __VA_ARGS__)                \
//   _(unsigned char, __bf16, RM_RZ_CLAMP, __VA_ARGS__)          \
//   _(unsigned char, __bf16, RM_RN_CLAMP, __VA_ARGS__)          \
  
#define DISPATCH_L1_CONVERT_OP_ALL_TYPES(op_name, dataflow_func, ...) \
  CONVERT_FUNC_TYPES(DISPATCH_L1_CONVERT_OP, op_name, dataflow_func,  \
                     __VA_ARGS__)

template <typename FROM_TYPE, typename TO_TYPE, class CalcFunc>
__device__ void Level1DataFlowConvert(TO_TYPE* dst_ptr, FROM_TYPE* src_ptr,
                           unsigned int size, CalcFunc calc_func_level0) {
  generic_ptr src_addr = reinterpret_cast<generic_ptr>(src_ptr);
  generic_ptr dst_addr = reinterpret_cast<generic_ptr>(dst_ptr);

  constexpr int from_bpe = sizeof(FROM_TYPE);
  constexpr int from_vec_elems =
      sizeof(typename unified_scalar<FROM_TYPE>::type) * TOPS_VECTOR_LENGTH /
      from_bpe;

  constexpr int to_bpe = sizeof(TO_TYPE);
  constexpr int to_vec_elems = sizeof(typename unified_scalar<TO_TYPE>::type) *
                               TOPS_VECTOR_LENGTH / to_bpe;

  constexpr int real_from_vec_elems =
      (from_bpe > to_bpe) ? from_vec_elems : from_vec_elems * from_bpe / to_bpe;
  constexpr int real_to_vec_elems =
      (from_bpe > to_bpe) ? to_vec_elems * to_bpe / from_bpe : to_vec_elems;

  using from_vtype =
      typename scalar_to_vector<FROM_TYPE, real_from_vec_elems>::type;
  using to_vtype = typename scalar_to_vector<TO_TYPE, real_to_vec_elems>::type;

  int from_group_num = (size + real_from_vec_elems - 1) / real_from_vec_elems;
  // int to_group_num = (size + real_to_vec_elems - 1) / real_to_vec_elems;

  from_vtype src;
  to_vtype dst;

  auto src_leaptr = simple_leaptr<from_vtype>(src_addr);
  auto dst_leaptr = simple_leaptr<to_vtype>(dst_addr);

#pragma clang loop unroll_count(16)
  for (int i = 0; i < from_group_num; i++) {
    src = src_leaptr.load();
    dst = calc_func_level0(src);
    dst_leaptr.store(dst);
  }
}
ALL_CONVERT_OP_TYPES(DISPATCH_L1_CONVERT_OP_ALL_TYPES, Level1DataFlowConvert);
