/**
 * Minimal topsop cc_kernel / launch-check compatibility for ubridge MoE kernels.
 */
#ifndef MOE_KERNEL_COMPAT_H_
#define MOE_KERNEL_COMPAT_H_

#include <tops.h>

#ifndef CC_KERNEL_LAUNCH_CHECK
#define CC_KERNEL_LAUNCH_CHECK()
#endif

#if defined __cplusplus
#define __OP_ASSERT_NO_CAST static_cast<void>
#else
#define __OP_ASSERT_NO_CAST (void)
#endif

#define __op_assert_fail(assertion, file_name, line, fmt, ...)                 \
  __builtin_printf("%s %d : op_assertion %s failed. " fmt "\r\n",              \
                   file_name, line, assertion, ##__VA_ARGS__),                   \
      __builtin_trap()

#define op_assert(val, fmt, ...)                                               \
  if (!(val)) {                                                                \
    __op_assert_fail(#val, __FILE_NAME__, __LINE__, fmt, ##__VA_ARGS__);       \
  }

namespace cc_kernel {

template <typename T>
struct UnderlyingType {
  using type = T;
};
template <>
struct UnderlyingType<float> {
  using type = float;
};
template <>
struct UnderlyingType<tops::half> {
  using type = __fp16;
};
template <>
struct UnderlyingType<tops::bfloat16> {
  using type = __bf16;
};

}  // namespace cc_kernel

/** Host-only align (topsop cc_kernel math_utils exposes AlignUp as HD; utils.h is device-only). */
__host__ __forceinline__ int32_t moe_host_align_up(int32_t x, int32_t y) {
  return (x + y - 1) / y * y;
}

#endif  // MOE_KERNEL_COMPAT_H_
