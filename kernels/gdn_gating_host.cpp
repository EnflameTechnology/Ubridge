/**
 * Copyright 2025 Enflame. All Rights Reserved.
 *
 * Fused GDN gating kernel for GCU:
 *   g    = -exp(A_log) * softplus(a + dt_bias)
 *   beta = sigmoid(b)
 *
 * A_log and dt_bias are per-head (broadcast over batch*seq).
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
__global__ void fused_gdn_gating_kernel(
    const float* __restrict__ a_log,
    const T* __restrict__ a,
    const T* __restrict__ b,
    const float* __restrict__ dt_bias,
    T* __restrict__ g_out,
    T* __restrict__ beta_out,
    int total_elements,
    int num_heads) {
  const int thread_num = GetThreadNum();
  const int thread_id  = GetThreadIdx();

  constexpr int TILE = 256;
  constexpr int MAX_HEADS = 256;
  __local__ __valigned__ T l_a[TILE];
  __local__ __valigned__ T l_b[TILE];
  __local__ __valigned__ float l_alog[TILE];
  __local__ __valigned__ float l_dt[TILE];
  __local__ __valigned__ T l_g[TILE];
  __local__ __valigned__ T l_beta[TILE];

  __local__ __valigned__ float l_head_alog[MAX_HEADS];
  __local__ __valigned__ float l_head_dt[MAX_HEADS];

  tops::private_dte ctx;
  ctx.init();

  {
    int nh = num_heads < MAX_HEADS ? num_heads : MAX_HEADS;
    tops::mdspan g_alog_h(tops::Global, const_cast<float*>(a_log), nh);
    tops::mdspan l_alog_h(tops::Private, l_head_alog, nh);
    tops::memcpy(ctx, l_alog_h, g_alog_h);

    tops::mdspan g_dt_h(tops::Global, const_cast<float*>(dt_bias), nh);
    tops::mdspan l_dt_h(tops::Private, l_head_dt, nh);
    tops::memcpy(ctx, l_dt_h, g_dt_h);

    tcle::fence<FenceType::L1_SDMEM>();
    tcle::fence<FenceType::L1_VDMEM>();
  }

  const int tiles_per_thread = CeilDiv(total_elements, TILE * thread_num);

  for (int tile_idx = 0; tile_idx < tiles_per_thread; tile_idx++) {
    const int base = (thread_id * tiles_per_thread + tile_idx) * TILE;
    if (base >= total_elements) break;
    const int count = (base + TILE <= total_elements) ? TILE : (total_elements - base);

    tops::mdspan g_a(tops::Global, const_cast<T*>(a) + base, count);
    tops::mdspan l_a_s(tops::Private, l_a, count);
    tops::memcpy(ctx, l_a_s, g_a);

    tops::mdspan g_b_s(tops::Global, const_cast<T*>(b) + base, count);
    tops::mdspan l_b_s(tops::Private, l_b, count);
    tops::memcpy(ctx, l_b_s, g_b_s);

    for (int i = 0; i < count; i++) {
      int h_idx = (base + i) % num_heads;
      l_alog[i] = l_head_alog[h_idx];
      l_dt[i]   = l_head_dt[h_idx];
    }

    tcle::fence<FenceType::L1_SDMEM>();
    tcle::fence<FenceType::L1_VDMEM>();

    using U = typename vpu_type<T>::type;
    constexpr int VE = TCLE_SINGLE_VEC_BYTES / sizeof(U);

    if constexpr (std::is_same_v<T, float>) {
      using VT = typename tcle::altivector<float, VE>::VT;
      tcle::leaptr<VT> pa    = tcle::simple_leaptr<VT>(l_a);
      tcle::leaptr<VT> pdt   = tcle::simple_leaptr<VT>(l_dt);
      tcle::leaptr<VT> palog = tcle::simple_leaptr<VT>(l_alog);
      tcle::leaptr<VT> pb    = tcle::simple_leaptr<VT>(l_b);
      tcle::leaptr<VT> pg    = tcle::simple_leaptr<VT>(l_g);
      tcle::leaptr<VT> pbeta = tcle::simple_leaptr<VT>(l_beta);

      for (int i = 0; i < count; i += VE) {
        VT va    = pa.load();
        VT vdt   = pdt.load();
        VT valog = palog.load();
        VT vb    = pb.load();

        VT x = va + vdt;
        VT sp = tcle::ln(tcle::exp(x) + VT{1.0f});
        VT vg = -(tcle::exp(valog)) * sp;
        VT vbeta = tcle::sigmoid(vb);

        pg.store(vg);
        pbeta.store(vbeta);
      }
    } else {
      using VT    = typename tcle::altivector<U, VE>::VT;
      using VT_F  = typename cc_kernel::accvector<U>::VT;

      tcle::leaptr<VT> pa    = tcle::simple_leaptr<VT>(reinterpret_cast<U*>(l_a));
      tcle::leaptr<VT_F> pdt   = tcle::simple_leaptr<VT_F>(l_dt);
      tcle::leaptr<VT_F> palog = tcle::simple_leaptr<VT_F>(l_alog);
      tcle::leaptr<VT> pb    = tcle::simple_leaptr<VT>(reinterpret_cast<U*>(l_b));
      tcle::leaptr<VT> pg    = tcle::simple_leaptr<VT>(reinterpret_cast<U*>(l_g));
      tcle::leaptr<VT> pbeta = tcle::simple_leaptr<VT>(reinterpret_cast<U*>(l_beta));

      for (int i = 0; i < count; i += VE) {
        VT va    = pa.load();
        VT_F vdt   = pdt.load();
        VT_F valog = palog.load();
        VT vb    = pb.load();

        VT_F va_f    = tcle::cvt<VT_F>(va);
        VT_F vb_f    = tcle::cvt<VT_F>(vb);

        VT_F x = va_f + vdt;
        VT_F sp = tcle::ln(tcle::exp(x) + VT_F{1.0f});
        VT_F vg_f = -(tcle::exp(valog)) * sp;
        VT_F vbeta_f = tcle::sigmoid(vb_f);

        pg.store(tcle::cvt<VT>(vg_f));
        pbeta.store(tcle::cvt<VT>(vbeta_f));
      }
    }

    tcle::fence<FenceType::L1_VDMEM>();

    tops::mdspan g_go(tops::Global, g_out + base, count);
    tops::mdspan l_go(tops::Private, l_g, count);
    tops::memcpy(ctx, g_go, l_go);

    tops::mdspan g_bo(tops::Global, beta_out + base, count);
    tops::mdspan l_bo(tops::Private, l_beta, count);
    tops::memcpy(ctx, g_bo, l_bo);
  }
}

template __global__ void fused_gdn_gating_kernel<float>(
    const float*, const float*, const float*, const float*,
    float*, float*, int, int);
template __global__ void fused_gdn_gating_kernel<tops::half>(
    const float*, const tops::half*, const tops::half*, const float*,
    tops::half*, tops::half*, int, int);
template __global__ void fused_gdn_gating_kernel<tops::bfloat>(
    const float*, const tops::bfloat*, const tops::bfloat*, const float*,
    tops::bfloat*, tops::bfloat*, int, int);

extern "C" void gdn_fused_gating_f32(
    const float* a_log, const float* a, const float* b, const float* dt_bias,
    float* g, float* beta, int total, int num_heads,
    unsigned int num_blocks, unsigned int dim_blocks, void* stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  fused_gdn_gating_kernel<float><<<dim3(num_blocks, 1, 1),
      dim3(dim_blocks, 1, 1), 0, stream>>>(
      a_log, a, b, dt_bias, g, beta, total, num_heads);
}

extern "C" void gdn_fused_gating_f16(
    const float* a_log, const __fp16* a, const __fp16* b, const float* dt_bias,
    __fp16* g, __fp16* beta, int total, int num_heads,
    unsigned int num_blocks, unsigned int dim_blocks, void* stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  fused_gdn_gating_kernel<tops::half><<<dim3(num_blocks, 1, 1),
      dim3(dim_blocks, 1, 1), 0, stream>>>(
      a_log,
      reinterpret_cast<const tops::half*>(a),
      reinterpret_cast<const tops::half*>(b),
      dt_bias,
      reinterpret_cast<tops::half*>(g),
      reinterpret_cast<tops::half*>(beta),
      total, num_heads);
}

extern "C" void gdn_fused_gating_bf16(
    const float* a_log, const __bf16* a, const __bf16* b, const float* dt_bias,
    __bf16* g, __bf16* beta, int total, int num_heads,
    unsigned int num_blocks, unsigned int dim_blocks, void* stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  fused_gdn_gating_kernel<tops::bfloat><<<dim3(num_blocks, 1, 1),
      dim3(dim_blocks, 1, 1), 0, stream>>>(
      a_log,
      reinterpret_cast<const tops::bfloat*>(a),
      reinterpret_cast<const tops::bfloat*>(b),
      dt_bias,
      reinterpret_cast<tops::bfloat*>(g),
      reinterpret_cast<tops::bfloat*>(beta),
      total, num_heads);
}
