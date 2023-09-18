#include <stdio.h>
#include <tops.h>
#include <tops/tops_runtime.h>
#include <tops/topsrtc.h>
#include <tops/half.h>
#include <tops/bfloat.h>
#include <krt/scalar.h>

namespace tops {
template <typename T>
__device__ __host__ __forceinline__ constexpr int hvlength() {
  return 128 / sizeof(T);
}

} // namespace tops

__device__ __forceinline__
auto get_index() {
    std::size_t blockIndex = blockIdx.z*(gridDim.x*gridDim.y)
        + blockIdx.y*gridDim.x + blockIdx.x;
    std::size_t threadIndex = threadIdx.z*(blockDim.x*blockDim.y)
        + threadIdx.y*blockDim.x + threadIdx.x;
    return blockIndex*(blockDim.x*blockDim.y*blockDim.z) + threadIndex;
}

#define UNARY_OP(TYPE, VT, FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    TYPE *inp, \
    TYPE *out) \
{ \
    tops_dte_ctx_t ctx; \
    tops::dte_scope s(ctx); \
    std::size_t idx = get_index(); \
    constexpr std::size_t num_len = tops::hvlength<VT>(); \
    __valigned__ TYPE buffer1[num_len]; \
    tops::mdspan buf1(tops::Private, &buffer1, num_len); \
    tops::mdspan src1(tops::Global, inp + idx * num_len, num_len); \
    tops::memcpy(ctx, buf1, src1); \
    const auto &x = tops::vload<VT>(buffer1);  \
    tops::mdspan dst(tops::Global, out + idx *num_len, num_len); \
    tops::vstore(FUNC, buffer1);  \
    tops::memcpy(ctx, dst, buf1); \
} \


#define UNARY_COPY_OP(TYPE, VT, FN_NAME)  \
extern "C" __global__ void FN_NAME(  \
    const size_t numel,  \
    const size_t num_dims,  \
    const size_t *info,  \
    TYPE *inp,  \
    TYPE *out) \
{  \
    tops_dte_ctx_t ctx; \
    tops::dte_scope s(ctx); \
    std::size_t idx = get_index(); \
    constexpr std::size_t num_len = tops::hvlength<VT>(); \
    __valigned__ TYPE buffer1[num_len]; \
    tops::mdspan buf1(tops::Private, &buffer1, num_len); \
    tops::mdspan src1(tops::Global, inp + idx * num_len, num_len); \
    tops::memcpy(ctx, buf1, src1); \
    tops::mdspan dst(tops::Global, out + idx *num_len, num_len);  \
    tops::memcpy(ctx, dst, buf1); \
}  \

// template<typename T>
// __device__ __forceinline__ T elu_fwd(T x, T alpha) {
//   if (x > static_cast<T>(0)) {
//     return x;
//   }
//   return alpha * (tops::exp<T>(x) - static_cast<T>(1));
// }



#define UNARY_OP1(TYPE, VT, FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    TYPE param, \
    TYPE *inp, \
    TYPE *out) \
{ \
    tops_dte_ctx_t ctx; \
    tops::dte_scope s(ctx); \
    std::size_t idx = get_index(); \
    constexpr std::size_t num_len = tops::hvlength<VT>(); \
    __valigned__ TYPE buffer1[num_len]; \
    tops::mdspan buf1(tops::Private, &buffer1, num_len); \
    __valigned__ TYPE buffer2[num_len]; \
    tops::mdspan buf2(tops::Private, &buffer2, num_len); \
    tops::mdspan src1(tops::Global, inp + idx * num_len, num_len); \
    tops::memcpy(ctx, buf1, src1); \
    const auto &x = tops::vload<VT>(buffer1);  \
    tops::mdspan dst(tops::Global, out + idx *num_len, num_len); \
    for (int i = 0; i < num_len; i++) { \
        buffer2[i] = FUNC; \
    } \
    tops::memcpy(ctx, dst, buf2); \
} \



UNARY_COPY_OP(tops::bfloat, vbfloat, ucopy_bf16)
UNARY_OP(tops::bfloat, vbfloat, uneg_bf16, tops::vneg<vbfloat>(x))
UNARY_OP(tops::bfloat, vbfloat, uexp_bf16, tops::vexp<vbfloat>(x))
UNARY_OP(tops::bfloat, vbfloat, ulog_bf16, tops::vlog<vbfloat>(x))
UNARY_OP(tops::bfloat, vbfloat, usin_bf16, tops::vsin<vbfloat>(x))
UNARY_OP(tops::bfloat, vbfloat, ucos_bf16, tops::vcos<vbfloat>(x))
UNARY_OP(tops::bfloat, vbfloat, uabs_bf16, tops::vabs<vbfloat>(x))
UNARY_OP(tops::bfloat, vbfloat, usqr_bf16, tops::vmul<vbfloat>(x, x))
UNARY_OP(tops::bfloat, vbfloat, usqrt_bf16, tops::vsqrt<vbfloat>(x))
UNARY_OP(tops::bfloat, vbfloat, ugelu_bf16, tops::vgelu<vbfloat>(x))
UNARY_OP(tops::bfloat, vbfloat, urelu_bf16, tops::vmax<vbfloat>(x, tops::vzero<vbfloat>())) 
// UNARY_OP1(tops::bfloat, vbfloat, uelu_bf16, elu_fwd(x[i], param))



UNARY_COPY_OP(tops::half, vhalf, ucopy_f16)
UNARY_OP(tops::half, vhalf, uneg_f16, tops::vneg<vhalf>(x))
UNARY_OP(tops::half, vhalf, uexp_f16, tops::vexp<vhalf>(x))
UNARY_OP(tops::half, vhalf, ulog_f16, tops::vlog<vhalf>(x))
UNARY_OP(tops::half, vhalf, usin_f16, tops::vsin<vhalf>(x))
UNARY_OP(tops::half, vhalf, ucos_f16, tops::vcos<vhalf>(x))
UNARY_OP(tops::half, vhalf, uabs_f16, tops::vabs<vhalf>(x))
UNARY_OP(tops::half, vhalf, usqr_f16, tops::vmul<vhalf>(x, x))
UNARY_OP(tops::half, vhalf, usqrt_f16, tops::vsqrt<vhalf>(x))
UNARY_OP(tops::half, vhalf, ugelu_f16, tops::vgelu<vhalf>(x))
UNARY_OP(tops::half, vhalf, urelu_f16, tops::vmax<vhalf>(x, tops::vzero<vhalf>()))
// UNARY_OP1(tops::half, vhalf, uelu_f16, elu_fwd(x[i], param))

UNARY_COPY_OP(int8_t, vchar, ucopy_i8)
UNARY_COPY_OP(uint8_t, vuchar, ucopy_u8)
UNARY_COPY_OP(int32_t, vint, ucopy_i32)
UNARY_COPY_OP(uint32_t, vuint, ucopy_u32)

UNARY_COPY_OP(float, vfloat, ucopy_f32)

UNARY_OP(float, vfloat, uneg_f32, tops::vneg<vfloat>(x))
UNARY_OP(float, vfloat, uexp_f32, tops::vexp<vfloat>(x))
UNARY_OP(float, vfloat, ulog_f32, tops::vlog<vfloat>(x))
UNARY_OP(float, vfloat, usin_f32, tops::vsin<vfloat>(x))
UNARY_OP(float, vfloat, ucos_f32, tops::vcos<vfloat>(x))
UNARY_OP(float, vfloat, uabs_f32, tops::vabs<vfloat>(x))
UNARY_OP(float, vfloat, usqr_f32, tops::vmul<vfloat>(x, x))
UNARY_OP(float, vfloat, usqrt_f32, tops::vsqrt<vfloat>(x))
UNARY_OP(float, vfloat, ugelu_f32, tops::vgelu<vfloat>(x))
UNARY_OP(float, vfloat, urelu_f32, tops::vmax<vfloat>(x, tops::vzero<vfloat>()))

// UNARY_OP1(float, vfloat, uelu_f32, elu_fwd(x[i], param))

int main() {
    return 0;
}