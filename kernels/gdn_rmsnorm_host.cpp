/**
 * Copyright 2025 Enflame. All Rights Reserved.
 *
 * Fused Gated RMSNorm + SiLU + Mul for GCU (GDN output gating):
 *   out = RMSNorm(x, gamma, bias, eps) * SiLU(z)
 *
 * Per-group RMSNorm: x is [rows, value_dim], groups of size group_size.
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
__global__ void gated_rmsnorm_silu_mul_kernel(
    const T* __restrict__ x,
    const T* __restrict__ z,
    const T* __restrict__ gamma,
    const T* __restrict__ bias,
    T* __restrict__ out,
    int rows,
    int value_dim,
    int group_size,
    float eps,
    int per_group_weights,
    int has_bias) {
  const int thread_num = GetThreadNum();
  const int thread_id  = GetThreadIdx();
  const int num_groups = value_dim / group_size;
  const int total_work = rows * num_groups;

  constexpr int TILE = 256;
  __local__ __valigned__ T l_x[TILE];
  __local__ __valigned__ T l_z[TILE];
  __local__ __valigned__ T l_gamma[TILE];
  __local__ __valigned__ T l_bias[TILE];
  __local__ __valigned__ T l_out[TILE];
  __local__ __valigned__ float l_f32[TILE];

  tops::private_dte ctx;
  ctx.init();

  for (int work_id = thread_id; work_id < total_work; work_id += thread_num) {
    const int row   = work_id / num_groups;
    const int group = work_id % num_groups;
    const int group_offset = row * value_dim + group * group_size;

    float sumsq = 0.0f;
    const int num_tiles = CeilDiv(group_size, TILE);

    for (int ti = 0; ti < num_tiles; ti++) {
      const int offset = ti * TILE;
      const int count = (offset + TILE <= group_size) ? TILE : (group_size - offset);

      tops::mdspan g_x(tops::Global, const_cast<T*>(x) + group_offset + offset, count);
      tops::mdspan l_x_s(tops::Private, l_x, count);
      tops::memcpy(ctx, l_x_s, g_x);
      tcle::fence<FenceType::L1_SDMEM>();
      tcle::fence<FenceType::L1_VDMEM>();

      using U = typename vpu_type<T>::type;
      constexpr int VE = TCLE_SINGLE_VEC_BYTES / sizeof(U);

      if constexpr (std::is_same_v<T, float>) {
        using VT = typename tcle::altivector<float, VE>::VT;
        tcle::leaptr<VT> px  = tcle::simple_leaptr<VT>(l_x);
        tcle::leaptr<VT> pf  = tcle::simple_leaptr<VT>(l_f32);
        for (int i = 0; i < TILE; i += VE) {
          VT v = px.load();
          pf.store(v * v);
        }
      } else {
        using VT   = typename tcle::altivector<U, VE>::VT;
        using VT_F = typename cc_kernel::accvector<U>::VT;
        tcle::leaptr<VT>   px = tcle::simple_leaptr<VT>(reinterpret_cast<U*>(l_x));
        tcle::leaptr<VT_F> pf = tcle::simple_leaptr<VT_F>(l_f32);
        for (int i = 0; i < TILE; i += VE) {
          VT v = px.load();
          VT_F vf = tcle::cvt<VT_F>(v);
          pf.store(vf * vf);
        }
      }
      tcle::fence<FenceType::L1_VDMEM>();

      for (int i = 0; i < count; i++) {
        sumsq += l_f32[i];
      }
    }

    float rms_val = sumsq / (float)group_size + eps;
    float l_rsqrt_buf[4] __attribute__((aligned(64)));
    l_rsqrt_buf[0] = rms_val;
    {
      using VT_F = typename tcle::altivector<float, 4>::VT;
      tcle::leaptr<VT_F> pv = tcle::simple_leaptr<VT_F>(l_rsqrt_buf);
      VT_F vv = pv.load();
      pv.store(tcle::rsqrt(vv));
    }
    tcle::fence<FenceType::L1_VDMEM>();
    float inv_rms = l_rsqrt_buf[0];

    for (int ti = 0; ti < num_tiles; ti++) {
      const int offset = ti * TILE;
      const int count = (offset + TILE <= group_size) ? TILE : (group_size - offset);

      tops::mdspan gx(tops::Global, const_cast<T*>(x) + group_offset + offset, count);
      tops::mdspan lx(tops::Private, l_x, count);
      tops::memcpy(ctx, lx, gx);

      tops::mdspan gz(tops::Global, const_cast<T*>(z) + group_offset + offset, count);
      tops::mdspan lz(tops::Private, l_z, count);
      tops::memcpy(ctx, lz, gz);

      const int wb_offset = per_group_weights ? offset : (group * group_size + offset);
      tops::mdspan gg(tops::Global, const_cast<T*>(gamma) + wb_offset, count);
      tops::mdspan lg(tops::Private, l_gamma, count);
      tops::memcpy(ctx, lg, gg);

      if (has_bias) {
        tops::mdspan gb(tops::Global, const_cast<T*>(bias) + wb_offset, count);
        tops::mdspan lb(tops::Private, l_bias, count);
        tops::memcpy(ctx, lb, gb);
      }

      tcle::fence<FenceType::L1_SDMEM>();
      tcle::fence<FenceType::L1_VDMEM>();

      using U = typename vpu_type<T>::type;
      constexpr int VE = TCLE_SINGLE_VEC_BYTES / sizeof(U);

      if constexpr (std::is_same_v<T, float>) {
        using VT = typename tcle::altivector<float, VE>::VT;
        VT v_inv_rms = VT{inv_rms};
        tcle::leaptr<VT> px = tcle::simple_leaptr<VT>(l_x);
        tcle::leaptr<VT> pz = tcle::simple_leaptr<VT>(l_z);
        tcle::leaptr<VT> pw = tcle::simple_leaptr<VT>(l_gamma);
        tcle::leaptr<VT> po = tcle::simple_leaptr<VT>(l_out);
        if (has_bias) {
          tcle::leaptr<VT> pb = tcle::simple_leaptr<VT>(l_bias);
          for (int i = 0; i < TILE; i += VE) {
            VT vx = px.load();
            VT vz = pz.load();
            VT vw = pw.load();
            VT vb = pb.load();
            VT normed = vx * v_inv_rms * vw + vb;
            VT gate = vz * tcle::sigmoid(vz);
            po.store(normed * gate);
          }
        } else {
          for (int i = 0; i < TILE; i += VE) {
            VT vx = px.load();
            VT vz = pz.load();
            VT vw = pw.load();
            VT normed = vx * v_inv_rms * vw;
            VT gate = vz * tcle::sigmoid(vz);
            po.store(normed * gate);
          }
        }
      } else {
        using VT   = typename tcle::altivector<U, VE>::VT;
        using VT_F = typename cc_kernel::accvector<U>::VT;
        VT_F v_inv_rms = VT_F{inv_rms};
        tcle::leaptr<VT> px = tcle::simple_leaptr<VT>(reinterpret_cast<U*>(l_x));
        tcle::leaptr<VT> pz = tcle::simple_leaptr<VT>(reinterpret_cast<U*>(l_z));
        tcle::leaptr<VT> pw = tcle::simple_leaptr<VT>(reinterpret_cast<U*>(l_gamma));
        tcle::leaptr<VT> po = tcle::simple_leaptr<VT>(reinterpret_cast<U*>(l_out));
        if (has_bias) {
          tcle::leaptr<VT> pb = tcle::simple_leaptr<VT>(reinterpret_cast<U*>(l_bias));
          for (int i = 0; i < TILE; i += VE) {
            VT vx = px.load(); VT vz = pz.load();
            VT vw = pw.load(); VT vb = pb.load();
            VT_F normed = tcle::cvt<VT_F>(vx) * v_inv_rms * tcle::cvt<VT_F>(vw) + tcle::cvt<VT_F>(vb);
            VT_F gate = tcle::cvt<VT_F>(vz) * tcle::sigmoid(tcle::cvt<VT_F>(vz));
            po.store(tcle::cvt<VT>(normed * gate));
          }
        } else {
          for (int i = 0; i < TILE; i += VE) {
            VT vx = px.load(); VT vz = pz.load(); VT vw = pw.load();
            VT_F normed = tcle::cvt<VT_F>(vx) * v_inv_rms * tcle::cvt<VT_F>(vw);
            VT_F gate = tcle::cvt<VT_F>(vz) * tcle::sigmoid(tcle::cvt<VT_F>(vz));
            po.store(tcle::cvt<VT>(normed * gate));
          }
        }
      }
      tcle::fence<FenceType::L1_VDMEM>();

      tops::mdspan go(tops::Global, out + group_offset + offset, count);
      tops::mdspan lo(tops::Private, l_out, count);
      tops::memcpy(ctx, go, lo);
    }
  }
}

template __global__ void gated_rmsnorm_silu_mul_kernel<float>(
    const float*, const float*, const float*, const float*, float*,
    int, int, int, float, int, int);
template __global__ void gated_rmsnorm_silu_mul_kernel<tops::half>(
    const tops::half*, const tops::half*, const tops::half*, const tops::half*, tops::half*,
    int, int, int, float, int, int);
template __global__ void gated_rmsnorm_silu_mul_kernel<tops::bfloat>(
    const tops::bfloat*, const tops::bfloat*, const tops::bfloat*, const tops::bfloat*, tops::bfloat*,
    int, int, int, float, int, int);

extern "C" void gdn_gated_rmsnorm_f32(
    const float* x, const float* z, const float* gamma, const float* bias,
    float* out, int rows, int value_dim, int group_size, float eps,
    int per_group_weights, int has_bias,
    unsigned int num_blocks, unsigned int dim_blocks, void* stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  gated_rmsnorm_silu_mul_kernel<float><<<dim3(num_blocks, 1, 1),
      dim3(dim_blocks, 1, 1), 0, stream>>>(
      x, z, gamma, bias, out, rows, value_dim, group_size, eps,
      per_group_weights, has_bias);
}

extern "C" void gdn_gated_rmsnorm_f16(
    const __fp16* x, const __fp16* z, const __fp16* gamma, const __fp16* bias,
    __fp16* out, int rows, int value_dim, int group_size, float eps,
    int per_group_weights, int has_bias,
    unsigned int num_blocks, unsigned int dim_blocks, void* stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  gated_rmsnorm_silu_mul_kernel<tops::half><<<dim3(num_blocks, 1, 1),
      dim3(dim_blocks, 1, 1), 0, stream>>>(
      reinterpret_cast<const tops::half*>(x),
      reinterpret_cast<const tops::half*>(z),
      reinterpret_cast<const tops::half*>(gamma),
      reinterpret_cast<const tops::half*>(bias),
      reinterpret_cast<tops::half*>(out),
      rows, value_dim, group_size, eps, per_group_weights, has_bias);
}

extern "C" void gdn_gated_rmsnorm_bf16(
    const __bf16* x, const __bf16* z, const __bf16* gamma, const __bf16* bias,
    __bf16* out, int rows, int value_dim, int group_size, float eps,
    int per_group_weights, int has_bias,
    unsigned int num_blocks, unsigned int dim_blocks, void* stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  gated_rmsnorm_silu_mul_kernel<tops::bfloat><<<dim3(num_blocks, 1, 1),
      dim3(dim_blocks, 1, 1), 0, stream>>>(
      reinterpret_cast<const tops::bfloat*>(x),
      reinterpret_cast<const tops::bfloat*>(z),
      reinterpret_cast<const tops::bfloat*>(gamma),
      reinterpret_cast<const tops::bfloat*>(bias),
      reinterpret_cast<tops::bfloat*>(out),
      rows, value_dim, group_size, eps, per_group_weights, has_bias);
}
