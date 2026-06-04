/**
 * Copyright 2025 Enflame. All Rights Reserved.
 *
 * L2 norm last dimension for GCU:
 *   output[row, :] = input[row, :] / sqrt(sum(input[row, :]^2) + eps)
 */

#include <acore_op.h>
#include "utils/utils.h"

#include <tops.h>
#include <tops/half.h>
#include <tops/bfloat.h>
#include <tops/topscc_types.h>
#include <tops/tops_runtime.h>
#include <type_traits>

using namespace tops;

#if defined(__GCU_ARCH__)
using tcle::FenceType;
#endif

template <typename T> struct vpu_type;
template <> struct vpu_type<tops::bfloat> { using type = __bf16; };
template <> struct vpu_type<tops::half>   { using type = __fp16; };
template <> struct vpu_type<float>        { using type = float; };

namespace cc_kernel {
template <typename T> struct accvector;
template <> struct accvector<float>        { using VT = __vector float; };
template <> struct accvector<tops::bfloat> { using VT = __vector2 float; };
template <> struct accvector<__bf16>       { using VT = __vector2 float; };
template <> struct accvector<tops::half>   { using VT = __vector2 float; };
template <> struct accvector<__fp16>       { using VT = __vector2 float; };
}

template <typename T>
__global__ void l2_norm_last_dim_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int rows,
    int dim,
    float eps) {
  const int thread_num = GetThreadNum();
  const int thread_id  = GetThreadIdx();

  constexpr int TILE_DIM = 256;
  __local__ __valigned__ T l_in[TILE_DIM];
  __local__ __valigned__ T l_out[TILE_DIM];
  __local__ __valigned__ float l_f32[TILE_DIM];
  __local__ __valigned__ float l_sumsq[1];

  tops::private_dte ctx;
  ctx.init();

  for (int row = thread_id; row < rows; row += thread_num) {
    const T* in_row  = input + row * dim;
    T*       out_row = output + row * dim;

    float total_sumsq = 0.0f;
    const int num_tiles = CeilDiv(dim, TILE_DIM);

    for (int ti = 0; ti < num_tiles; ti++) {
      const int offset = ti * TILE_DIM;
      const int count = (offset + TILE_DIM <= dim) ? TILE_DIM : (dim - offset);

      tops::mdspan g_in(tops::Global, const_cast<T*>(in_row) + offset, count);
      tops::mdspan l_in_s(tops::Private, l_in, count);
      tops::memcpy(ctx, l_in_s, g_in);
      tcle::fence<FenceType::L1_SDMEM>();
      tcle::fence<FenceType::L1_VDMEM>();

      using U = typename vpu_type<T>::type;
      constexpr int VE = TCLE_SINGLE_VEC_BYTES / sizeof(U);

      if constexpr (std::is_same_v<T, float>) {
        using VT = typename tcle::altivector<float, VE>::VT;
        tcle::leaptr<VT> pin  = tcle::simple_leaptr<VT>(l_in);
        tcle::leaptr<VT> pacc = tcle::simple_leaptr<VT>(l_f32);
        for (int i = 0; i < TILE_DIM; i += VE) {
          VT v = pin.load();
          pacc.store(v * v);
        }
      } else {
        using VT   = typename tcle::altivector<U, VE>::VT;
        using VT_F = typename cc_kernel::accvector<U>::VT;
        tcle::leaptr<VT>   pin  = tcle::simple_leaptr<VT>(reinterpret_cast<U*>(l_in));
        tcle::leaptr<VT_F> pacc = tcle::simple_leaptr<VT_F>(l_f32);
        for (int i = 0; i < TILE_DIM; i += VE) {
          VT v = pin.load();
          VT_F vf = tcle::cvt<VT_F>(v);
          pacc.store(vf * vf);
        }
      }

      tcle::fence<FenceType::L1_VDMEM>();

      for (int i = 0; i < count; i++) {
        total_sumsq += l_f32[i];
      }
    }

    float val = total_sumsq + eps;
    float l_rsqrt_buf[4] __attribute__((aligned(64)));
    l_rsqrt_buf[0] = val;
    {
      using VT_F = typename tcle::altivector<float, 4>::VT;
      tcle::leaptr<VT_F> pv = tcle::simple_leaptr<VT_F>(l_rsqrt_buf);
      VT_F vv = pv.load();
      pv.store(tcle::rsqrt(vv));
    }
    tcle::fence<FenceType::L1_VDMEM>();
    float inv_norm = l_rsqrt_buf[0];

    for (int ti = 0; ti < num_tiles; ti++) {
      const int offset = ti * TILE_DIM;
      const int count = (offset + TILE_DIM <= dim) ? TILE_DIM : (dim - offset);

      tops::mdspan g_in(tops::Global, const_cast<T*>(in_row) + offset, count);
      tops::mdspan l_in_s(tops::Private, l_in, count);
      tops::memcpy(ctx, l_in_s, g_in);
      tcle::fence<FenceType::L1_SDMEM>();
      tcle::fence<FenceType::L1_VDMEM>();

      using U = typename vpu_type<T>::type;
      constexpr int VE = TCLE_SINGLE_VEC_BYTES / sizeof(U);

      if constexpr (std::is_same_v<T, float>) {
        using VT = typename tcle::altivector<float, VE>::VT;
        tcle::leaptr<VT> pin  = tcle::simple_leaptr<VT>(l_in);
        tcle::leaptr<VT> pout = tcle::simple_leaptr<VT>(l_out);
        VT vinv = VT{inv_norm};
        for (int i = 0; i < TILE_DIM; i += VE) {
          VT v = pin.load();
          pout.store(v * vinv);
        }
      } else {
        using VT   = typename tcle::altivector<U, VE>::VT;
        using VT_F = typename cc_kernel::accvector<U>::VT;
        tcle::leaptr<VT> pin  = tcle::simple_leaptr<VT>(reinterpret_cast<U*>(l_in));
        tcle::leaptr<VT> pout = tcle::simple_leaptr<VT>(reinterpret_cast<U*>(l_out));
        VT_F vinv = VT_F{inv_norm};
        for (int i = 0; i < TILE_DIM; i += VE) {
          VT v = pin.load();
          VT_F vf = tcle::cvt<VT_F>(v);
          pout.store(tcle::cvt<VT>(vf * vinv));
        }
      }

      tcle::fence<FenceType::L1_VDMEM>();

      tops::mdspan g_o(tops::Global, out_row + offset, count);
      tops::mdspan l_o(tops::Private, l_out, count);
      tops::memcpy(ctx, g_o, l_o);
    }
  }
}

template __global__ void l2_norm_last_dim_kernel<float>(
    const float*, float*, int, int, float);
template __global__ void l2_norm_last_dim_kernel<tops::half>(
    const tops::half*, tops::half*, int, int, float);
template __global__ void l2_norm_last_dim_kernel<tops::bfloat>(
    const tops::bfloat*, tops::bfloat*, int, int, float);

extern "C" void gdn_l2_norm_f32(
    const float* input, float* output, int rows, int dim, float eps,
    unsigned int num_blocks, unsigned int dim_blocks, void* stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  l2_norm_last_dim_kernel<float><<<dim3(num_blocks, 1, 1),
      dim3(dim_blocks, 1, 1), 0, stream>>>(input, output, rows, dim, eps);
}

extern "C" void gdn_l2_norm_f16(
    const __fp16* input, __fp16* output, int rows, int dim, float eps,
    unsigned int num_blocks, unsigned int dim_blocks, void* stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  l2_norm_last_dim_kernel<tops::half><<<dim3(num_blocks, 1, 1),
      dim3(dim_blocks, 1, 1), 0, stream>>>(
      reinterpret_cast<const tops::half*>(input),
      reinterpret_cast<tops::half*>(output), rows, dim, eps);
}

extern "C" void gdn_l2_norm_bf16(
    const __bf16* input, __bf16* output, int rows, int dim, float eps,
    unsigned int num_blocks, unsigned int dim_blocks, void* stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  l2_norm_last_dim_kernel<tops::bfloat><<<dim3(num_blocks, 1, 1),
      dim3(dim_blocks, 1, 1), 0, stream>>>(
      reinterpret_cast<const tops::bfloat*>(input),
      reinterpret_cast<tops::bfloat*>(output), rows, dim, eps);
}
