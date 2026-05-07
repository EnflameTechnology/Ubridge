/**
 * Copyright 2025 Enflame. All Rights Reserved.
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

#include <acore_op.h>
#include "utils/utils.h"

#include <tops.h>
#include <tops/half.h>
#include <tops/bfloat.h>
#include <tops/topscc_types.h>
#include <type_traits>

#include <tops/tops_runtime.h>

struct CausalConv1dParams {
  int dim;
  int batch;
  int num_cache_lines;
  int kernel_width;
  int state_len;
  int stride_x_token;
  int stride_w_dim;
  int stride_istate_seq;
  int stride_istate_token;
  int pad_slot_id;
  int has_bias;
  int silu_activation;
  int block_n;
};

namespace cc_kernel {
template <typename T> struct accvector;
template <> struct accvector<float> { using VT = __vector float; };
template <> struct accvector<tops::bfloat> { using VT = __vector2 float; };
template <> struct accvector<__bf16> { using VT = __vector2 float; };
template <> struct accvector<tops::half> { using VT = __vector2 float; };
template <> struct accvector<__fp16> { using VT = __vector2 float; };
}  // namespace cc_kernel

using namespace tops;
using namespace cc_kernel;

#if defined(__GCU_ARCH__)
using tcle::FenceType;
#endif

template <typename T> struct vpu_type;
template <> struct vpu_type<tops::bfloat> { using type = __bf16; };
template <> struct vpu_type<tops::half> { using type = __fp16; };
template <> struct vpu_type<float> { using type = float; };

// VPU local-to-local copy
template <typename T>
__device__ __forceinline__ void vec_copy(T* dst, T* src, int n) {
  using U = typename vpu_type<T>::type;
  constexpr int VE = TCLE_SINGLE_VEC_BYTES / sizeof(U);
  using VT = typename tcle::altivector<U, VE>::VT;
  tcle::leaptr<VT> d = tcle::simple_leaptr<VT>(reinterpret_cast<U*>(dst));
  tcle::leaptr<VT> s = tcle::simple_leaptr<VT>(reinterpret_cast<U*>(src));
  for (int i = 0; i < n; i += VE) {
    d.store(s.load());
  }
}

// --- FP32-accumulation convolution helpers ---
// All multiply-accumulate is done in fp32 for bf16/fp16 via tcle::cvt.
// For T=float, operates natively without conversion.

template <typename T>
__device__ __forceinline__ void vec_mul_f32(float* dst, T* lhs, T* rhs, int n) {
  if constexpr (std::is_same_v<T, float>) {
    constexpr int VE = TCLE_SINGLE_VEC_BYTES / sizeof(float);
    using VT = typename tcle::altivector<float, VE>::VT;
    tcle::leaptr<VT> d = tcle::simple_leaptr<VT>(dst);
    tcle::leaptr<VT> l = tcle::simple_leaptr<VT>(lhs);
    tcle::leaptr<VT> r = tcle::simple_leaptr<VT>(rhs);
    for (int i = 0; i < n; i += VE) {
      VT vl = l.load();
      VT vr = r.load();
      d.store(vl * vr);
    }
  } else {
    using U = typename vpu_type<T>::type;
    constexpr int VE = TCLE_SINGLE_VEC_BYTES / sizeof(U);
    using VT = typename tcle::altivector<U, VE>::VT;
    using VT_F32 = typename cc_kernel::accvector<U>::VT;
    tcle::leaptr<VT_F32> d = tcle::simple_leaptr<VT_F32>(dst);
    tcle::leaptr<VT> l = tcle::simple_leaptr<VT>(reinterpret_cast<U*>(lhs));
    tcle::leaptr<VT> r = tcle::simple_leaptr<VT>(reinterpret_cast<U*>(rhs));
    for (int i = 0; i < n; i += VE) {
      VT vl = l.load();
      VT vr = r.load();
      VT_F32 vl_f = tcle::cvt<VT_F32>(vl);
      VT_F32 vr_f = tcle::cvt<VT_F32>(vr);
      d.store(vl_f * vr_f);
    }
  }
}

template <typename T>
__device__ __forceinline__ void vec_mac_f32(float* acc, T* lhs, T* rhs, int n) {
  if constexpr (std::is_same_v<T, float>) {
    constexpr int VE = TCLE_SINGLE_VEC_BYTES / sizeof(float);
    using VT = typename tcle::altivector<float, VE>::VT;
    tcle::leaptr<VT> a_rd = tcle::simple_leaptr<VT>(acc);
    tcle::leaptr<VT> a_wr = tcle::simple_leaptr<VT>(acc);
    tcle::leaptr<VT> l = tcle::simple_leaptr<VT>(lhs);
    tcle::leaptr<VT> r = tcle::simple_leaptr<VT>(rhs);
    for (int i = 0; i < n; i += VE) {
      VT va = a_rd.load();
      VT vl = l.load();
      VT vr = r.load();
      a_wr.store(va + vl * vr);
    }
  } else {
    using U = typename vpu_type<T>::type;
    constexpr int VE = TCLE_SINGLE_VEC_BYTES / sizeof(U);
    using VT = typename tcle::altivector<U, VE>::VT;
    using VT_F32 = typename cc_kernel::accvector<U>::VT;
    tcle::leaptr<VT_F32> a_rd = tcle::simple_leaptr<VT_F32>(acc);
    tcle::leaptr<VT_F32> a_wr = tcle::simple_leaptr<VT_F32>(acc);
    tcle::leaptr<VT> l = tcle::simple_leaptr<VT>(reinterpret_cast<U*>(lhs));
    tcle::leaptr<VT> r = tcle::simple_leaptr<VT>(reinterpret_cast<U*>(rhs));
    for (int i = 0; i < n; i += VE) {
      VT_F32 va = a_rd.load();
      VT vl = l.load();
      VT vr = r.load();
      VT_F32 vl_f = tcle::cvt<VT_F32>(vl);
      VT_F32 vr_f = tcle::cvt<VT_F32>(vr);
      a_wr.store(va + vl_f * vr_f);
    }
  }
}

template <typename T>
__device__ __forceinline__ void vec_add_f32_bias(float* acc, T* bias, int n) {
  if constexpr (std::is_same_v<T, float>) {
    constexpr int VE = TCLE_SINGLE_VEC_BYTES / sizeof(float);
    using VT = typename tcle::altivector<float, VE>::VT;
    tcle::leaptr<VT> a_rd = tcle::simple_leaptr<VT>(acc);
    tcle::leaptr<VT> a_wr = tcle::simple_leaptr<VT>(acc);
    tcle::leaptr<VT> b = tcle::simple_leaptr<VT>(bias);
    for (int i = 0; i < n; i += VE) {
      VT va = a_rd.load();
      VT vb = b.load();
      a_wr.store(va + vb);
    }
  } else {
    using U = typename vpu_type<T>::type;
    constexpr int VE = TCLE_SINGLE_VEC_BYTES / sizeof(U);
    using VT = typename tcle::altivector<U, VE>::VT;
    using VT_F32 = typename cc_kernel::accvector<U>::VT;
    tcle::leaptr<VT_F32> a_rd = tcle::simple_leaptr<VT_F32>(acc);
    tcle::leaptr<VT_F32> a_wr = tcle::simple_leaptr<VT_F32>(acc);
    tcle::leaptr<VT> b = tcle::simple_leaptr<VT>(reinterpret_cast<U*>(bias));
    for (int i = 0; i < n; i += VE) {
      VT_F32 va = a_rd.load();
      VT vb = b.load();
      VT_F32 vb_f = tcle::cvt<VT_F32>(vb);
      a_wr.store(va + vb_f);
    }
  }
}

__device__ __forceinline__ void vpu_swish_f32(float* dst, float* src, unsigned int n) {
  constexpr int VE = TCLE_SINGLE_VEC_BYTES / sizeof(float);
  using VT = typename tcle::altivector<float, VE>::VT;
  tcle::leaptr<VT> d = tcle::simple_leaptr<VT>(dst);
  tcle::leaptr<VT> s = tcle::simple_leaptr<VT>(src);
  for (unsigned int i = 0; i < n; i += VE) {
    VT v = s.load();
    d.store(v * tcle::sigmoid(v));
  }
}

template <typename T>
__device__ __forceinline__ void vec_f32_to_native(T* dst, float* src, int n) {
  if constexpr (std::is_same_v<T, float>) {
    vec_copy(dst, src, n);
  } else {
    using U = typename vpu_type<T>::type;
    constexpr int VE = TCLE_SINGLE_VEC_BYTES / sizeof(U);
    using VT = typename tcle::altivector<U, VE>::VT;
    using VT_F32 = typename cc_kernel::accvector<U>::VT;
    tcle::leaptr<VT> d = tcle::simple_leaptr<VT>(reinterpret_cast<U*>(dst));
    tcle::leaptr<VT_F32> s = tcle::simple_leaptr<VT_F32>(src);
    for (int i = 0; i < n; i += VE) {
      VT_F32 v = s.load();
      d.store(tcle::cvt<VT>(v));
    }
  }
}

// DTE helpers using 2D slice/deslice
template <typename T>
__device__ __forceinline__ void dte_load(tops::private_dte& ctx,
                                         T* l_ptr, T* g_ptr, int n) {
  tops::mdspan l_2d(tops::Private, l_ptr, 1, n);
  tops::mdspan g_2d(tops::Global, g_ptr, 1, n);
  int offsets[] = {0, 0};
  tops::slice(ctx, l_2d, g_2d, offsets);
}

template <typename T>
__device__ __forceinline__ void dte_store(tops::private_dte& ctx,
                                          T* g_ptr, T* l_ptr, int n) {
  tops::mdspan l_2d(tops::Private, l_ptr, 1, n);
  tops::mdspan g_2d(tops::Global, g_ptr, 1, n);
  int offsets[] = {0, 0};
  tops::deslice(ctx, g_2d, l_2d, offsets);
}

template <typename T>
__device__ __forceinline__ void dte_zero(tops::private_dte& ctx,
                                         T* l_ptr, int n) {
  tops::mdspan l_2d(tops::Private, l_ptr, 1, n);
  tops::memset(ctx, l_2d, T(0));
}

template <typename T>
__device__ __forceinline__ void vpu_zero_aligned(T* ptr, int n) {
  using U = typename vpu_type<T>::type;
  constexpr int VE = TCLE_SINGLE_VEC_BYTES / sizeof(U);
  using VT = typename tcle::altivector<U, VE>::VT;
  VT z = {};
  tcle::leaptr<VT> d = tcle::simple_leaptr<VT>(reinterpret_cast<U*>(ptr));
  for (int i = 0; i < n; i += VE) {
    d.store(z);
  }
}


template <typename T>
__attribute__((no_mem_alias_in_vldst, no_mem_alias_in_tar,
              loop_iterator_less_than_1024, enable_software_pipeliner,
              enable_bc_resolver))
__global__ void causal_conv1d_fwd_kernel(
    T* __restrict__ x_ptr,
    T* __restrict__ w_ptr,
    T* __restrict__ bias_ptr,
    T* __restrict__ conv_states_ptr,
    int32_t* __restrict__ cache_indices_ptr,
    int8_t* __restrict__ has_initial_states_ptr,
    int32_t* __restrict__ query_start_loc_ptr,
    T* __restrict__ o_ptr,
    CausalConv1dParams params) {

  const int thread_num = GetThreadNum();
  const int thread_id = GetThreadIdx();

  const int dim = params.dim;
  const int batch = params.batch;
  const int kernel_width = params.kernel_width;
  const int state_len = kernel_width - 1;
  const int stride_x_token = params.stride_x_token;
  const int stride_w_dim = params.stride_w_dim;
  const int stride_istate_seq = params.stride_istate_seq;
  const int stride_istate_token = params.stride_istate_token;
  const int pad_slot_id = params.pad_slot_id;
  const int silu_activation = params.silu_activation;
  const int has_bias = params.has_bias;
  const int BLOCK_N = params.block_n;
  const int num_channel_blocks = CeilDiv(dim, BLOCK_N);
  const int total_work = batch * num_channel_blocks;

  __builtin_assume(dim > 0);
  __builtin_assume(batch > 0);
  __builtin_assume(kernel_width > 0);
  __builtin_assume(kernel_width <= 5);
  __builtin_assume(state_len > 0);
  __builtin_assume(state_len <= 4);

  constexpr int MAX_BATCH = 2048;
  constexpr int MAX_KW = 5;
  constexpr int MAX_SL = 4;
  constexpr int LOCAL_BLOCK_N = 256;
  constexpr int TILE_T = 256;
  constexpr int PP = 2;

  __local__ __valigned__ int32_t l_ci[MAX_BATCH];
  __local__ __valigned__ int32_t l_qsl[MAX_BATCH + 1];
  __local__ __valigned__ int8_t l_his[MAX_BATCH];

  __local__ __valigned__ T w_t[MAX_KW * LOCAL_BLOCK_N];
  __local__ __valigned__ T col_slots[MAX_SL * LOCAL_BLOCK_N];
  __local__ __valigned__ T bias_t[LOCAL_BLOCK_N];
  __local__ __valigned__ float f32_acc[LOCAL_BLOCK_N];

  __local__ __valigned__ T x_tile_pp[PP][TILE_T * LOCAL_BLOCK_N];
  __local__ __valigned__ T o_tile[PP][TILE_T * LOCAL_BLOCK_N];
  __local__ __valigned__ T t_buf[MAX_KW * LOCAL_BLOCK_N];

  tops::private_dte ctx_scalar;
  ctx_scalar.init();
  tops::private_dte dte_x_0;
  dte_x_0.init();
  tops::private_dte dte_x_1;
  dte_x_1.init();
  tops::private_dte dte_o;
  tops::event evt_x_0;
  tops::event evt_x_1;
  tops::event evt_o;
  bool store_pending = false;

  ctx_scalar.init();
  dte_x_0.init();
  dte_x_1.init();
  dte_o.init();

  {
    int ci_n = (batch < MAX_BATCH) ? batch : MAX_BATCH;
    tops::mdspan g_ci(tops::Global, cache_indices_ptr, ci_n);
    tops::mdspan l_ci_s(tops::Private, l_ci, ci_n);
    tops::memcpy(ctx_scalar, l_ci_s, g_ci);

    int qsl_n = (batch + 1 < MAX_BATCH + 1) ? batch + 1 : MAX_BATCH + 1;
    tops::mdspan g_qsl(tops::Global, query_start_loc_ptr, qsl_n);
    tops::mdspan l_qsl_s(tops::Private, l_qsl, qsl_n);
    tops::memcpy(ctx_scalar, l_qsl_s, g_qsl);

    int his_n = (batch < MAX_BATCH) ? batch : MAX_BATCH;
    tops::mdspan g_his(tops::Global, has_initial_states_ptr, his_n);
    tops::mdspan l_his_s(tops::Private, l_his, his_n);
    tops::memcpy(ctx_scalar, l_his_s, g_his);
  }

  for (int work_id = thread_id; work_id < total_work; work_id += thread_num) {
    const int seq_idx = work_id / num_channel_blocks;
    const int cb = work_id % num_channel_blocks;
    const int feat_start = cb * BLOCK_N;
    const int actual_n = ((feat_start + BLOCK_N) < dim)
                             ? BLOCK_N
                             : (dim - feat_start);

    __builtin_assume(actual_n > 0);
    __builtin_assume(actual_n <= LOCAL_BLOCK_N);

    const int32_t cidx = l_ci[seq_idx];
    if (cidx == pad_slot_id) continue;

    const int32_t seq_start = l_qsl[seq_idx];
    const int seqlen = l_qsl[seq_idx + 1] - seq_start;
    if (seqlen == 0) continue;

    __builtin_assume(seqlen > 0);
    __builtin_assume(seqlen <= 65536);

    const int8_t has_init = l_his[seq_idx];

    for (int k = 0; k < kernel_width; k++)
      dte_zero<T>(ctx_scalar, w_t + k * LOCAL_BLOCK_N, LOCAL_BLOCK_N);
    for (int k = 0; k < kernel_width; k++) {
      T* w_src = w_ptr + static_cast<int64_t>(k) * stride_w_dim + feat_start;
      dte_load<T>(ctx_scalar, w_t + k * LOCAL_BLOCK_N, w_src, actual_n);
    }

    int col_base = 0;

    for (int k = 0; k < state_len; k++)
      dte_zero<T>(ctx_scalar, col_slots + k * LOCAL_BLOCK_N, LOCAL_BLOCK_N);
    if (has_init) {
      T* cs_2d = conv_states_ptr +
                 static_cast<int64_t>(cidx) * stride_istate_seq;
      for (int k = 0; k < state_len; k++) {
        T* g_row = cs_2d + static_cast<int64_t>(k) * stride_istate_token +
                   feat_start;
        dte_load<T>(ctx_scalar, col_slots + k * LOCAL_BLOCK_N, g_row, actual_n);
      }
    }

    dte_zero<T>(ctx_scalar, bias_t, LOCAL_BLOCK_N);
    if (has_bias) {
      dte_load<T>(ctx_scalar, bias_t, bias_ptr + feat_start, actual_n);
    }

    T* w_ptrs[MAX_KW + 1];
    for (int k = 0; k <= state_len; k++) {
      w_ptrs[k] = w_t + k * LOCAL_BLOCK_N;
    }

    T* slot_ptrs[MAX_SL];
    for (int k = 0; k < state_len; k++) {
      slot_ptrs[k] = col_slots + k * LOCAL_BLOCK_N;
    }

    tcle::fence<FenceType::L1_SDMEM>();
    tcle::fence<FenceType::L1_VDMEM>();

    constexpr int DTE_ALIGN = 128 / sizeof(T);
    const int padded_n = ((actual_n + DTE_ALIGN - 1) / DTE_ALIGN) * DTE_ALIGN;

    const int eff_tile = (TILE_T * LOCAL_BLOCK_N) / padded_n;
    const int num_tiles = CeilDiv(seqlen, eff_tile);
    const int total_tok = l_qsl[batch];
    store_pending = false;

    __builtin_assume(eff_tile > 0);
    __builtin_assume(num_tiles > 0);
    __builtin_assume(total_tok > 0);

    // Async pre-load first tile (padded local stride, zero-fill padding)
    const bool need_dte_pad = (padded_n > actual_n);
    {
      const int tile0_t = (seqlen < eff_tile) ? seqlen : eff_tile;
      tops::mdspan g_x(tops::Global, x_ptr, total_tok, stride_x_token);
      tops::mdspan l_x(tops::Private, x_tile_pp[0], tile0_t, padded_n);
      int offsets[] = {seq_start, feat_start};
      if (need_dte_pad) {
        dte_x_0.config_slice(l_x, g_x, offsets, 0);
      } else {
        dte_x_0.config_slice(l_x, g_x, offsets);
      }
      evt_x_0 = dte_x_0.trigger();
    }

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
      const int pp = tile_idx & 1;
      const int next_pp = 1 - pp;
      const int t_start = tile_idx * eff_tile;
      const int tile_t = ((t_start + eff_tile) <= seqlen)
                             ? eff_tile
                             : (seqlen - t_start);

      // [1] Wait current tile load → DTE data visible to VPU
      if (pp == 0) evt_x_0.wait();
      else evt_x_1.wait();
      tcle::fence<FenceType::L1_SDMEM>();
      tcle::fence<FenceType::L1_VDMEM>();

      // [2] Async prefetch next tile (overlaps with compute below)
      if (tile_idx + 1 < num_tiles) {
        const int next_t_start = (tile_idx + 1) * eff_tile;
        const int next_tile_t = ((next_t_start + eff_tile) <= seqlen)
                                    ? eff_tile
                                    : (seqlen - next_t_start);
        tops::mdspan g_x(tops::Global, x_ptr, total_tok, stride_x_token);
        tops::mdspan l_x(tops::Private, x_tile_pp[next_pp],
                         next_tile_t, padded_n);
        int offsets[] = {seq_start + next_t_start, feat_start};
        if (next_pp == 0) {
          if (need_dte_pad) dte_x_0.config_slice(l_x, g_x, offsets, 0);
          else dte_x_0.config_slice(l_x, g_x, offsets);
          evt_x_0 = dte_x_0.trigger();
        } else {
          if (need_dte_pad) dte_x_1.config_slice(l_x, g_x, offsets, 0);
          else dte_x_1.config_slice(l_x, g_x, offsets);
          evt_x_1 = dte_x_1.trigger();
        }
      }

      // [3] Pure VPU compute → output directly to o_tile[pp]
      T* x_base = x_tile_pp[pp];
      T* x_row = x_base;
      T* o_row = o_tile[pp];

      // Phase 1: first min(tile_t, state_len) tokens use ring buffer
      {
        const int phase1 = (tile_t < state_len) ? tile_t : state_len;
        for (int t_off = 0; t_off < phase1; t_off++) {
          int s0 = col_base;
          vec_mul_f32<T>(f32_acc, slot_ptrs[s0], w_ptrs[0], LOCAL_BLOCK_N);
          for (int k = 1; k < state_len; k++) {
            int sk = s0 + k;
            if (sk >= state_len) sk -= state_len;
            vec_mac_f32<T>(f32_acc, slot_ptrs[sk], w_ptrs[k], LOCAL_BLOCK_N);
          }
          vec_mac_f32<T>(f32_acc, x_row, w_ptrs[state_len], LOCAL_BLOCK_N);
          if (has_bias) {
            vec_add_f32_bias<T>(f32_acc, bias_t, LOCAL_BLOCK_N);
            if (!std::is_same_v<T, float>)
              tcle::fence<FenceType::L1_VDMEM>();
          }
          if (silu_activation) vpu_swish_f32(f32_acc, f32_acc, LOCAL_BLOCK_N);
          vec_f32_to_native<T>(o_row, f32_acc, padded_n);
          vec_copy<T>(slot_ptrs[s0], x_row, LOCAL_BLOCK_N);
          col_base = s0 + 1;
          if (col_base >= state_len) col_base -= state_len;
          x_row += padded_n;
          o_row += padded_n;
        }
      }

      // Phase 2: remaining tokens read history directly from x_tile_pp
      if (tile_t > state_len) {
        T* h = x_base;
        for (int t_off = state_len; t_off < tile_t; t_off++) {
          vec_mul_f32<T>(f32_acc, h, w_ptrs[0], LOCAL_BLOCK_N);
          for (int k = 1; k < state_len; k++) {
            vec_mac_f32<T>(f32_acc, h + k * padded_n, w_ptrs[k],
                           LOCAL_BLOCK_N);
          }
          vec_mac_f32<T>(f32_acc, x_row, w_ptrs[state_len], LOCAL_BLOCK_N);
          if (has_bias) {
            vec_add_f32_bias<T>(f32_acc, bias_t, LOCAL_BLOCK_N);
            if (!std::is_same_v<T, float>)
              tcle::fence<FenceType::L1_VDMEM>();
          }
          if (silu_activation) vpu_swish_f32(f32_acc, f32_acc, LOCAL_BLOCK_N);
          vec_f32_to_native<T>(o_row, f32_acc, padded_n);
          h += padded_n;
          x_row += padded_n;
          o_row += padded_n;
        }
      }

      // Refresh ring buffer with last state_len tokens for next tile
      if (tile_t >= state_len) {
        for (int k = 0; k < state_len; k++) {
          vec_copy<T>(slot_ptrs[k],
                      x_base + (tile_t - state_len + k) * padded_n,
                      LOCAL_BLOCK_N);
        }
        col_base = 0;
      }

      // [4] Wait previous async store before reconfiguring dte_o
      if (store_pending) {
        evt_o.wait();
      }

      // [5] Flush VPU → DTE, async trigger output store
      tcle::fence<FenceType::L1_VDMEM>();
      {
        tops::mdspan g_o(tops::Global, o_ptr, total_tok, stride_x_token);
        tops::mdspan l_o(tops::Private, o_tile[pp], tile_t, padded_n);
        int offsets[] = {seq_start + t_start, feat_start};
        dte_o.config_deslice(g_o, l_o, offsets);
        evt_o = dte_o.trigger();
        store_pending = true;
      }
    }

    // Drain final async store
    if (store_pending) {
      evt_o.wait();
    }

    // Write back conv_states
    T* cs_out_2d = conv_states_ptr +
                   static_cast<int64_t>(cidx) * stride_istate_seq;

    if (state_len <= seqlen) {
      for (int k = 0; k < state_len; k++) {
        T* x_src = x_ptr +
                   static_cast<int64_t>(seq_start + seqlen - state_len + k) *
                       stride_x_token +
                   feat_start;
        dte_load<T>(ctx_scalar, t_buf + k * actual_n, x_src, actual_n);
      }
      for (int k = 0; k < state_len; k++) {
        T* g_row = cs_out_2d +
                   static_cast<int64_t>(k) * stride_istate_token + feat_start;
        T* l_row = t_buf + k * actual_n;
        dte_store<T>(ctx_scalar, g_row, l_row, actual_n);
      }
    } else {
      if (has_init) {
        T* cs_in_2d = conv_states_ptr +
                      static_cast<int64_t>(cidx) * stride_istate_seq;
        for (int k = 0; k < state_len; k++) {
          if (k < state_len - seqlen) {
            T* cs_src = cs_in_2d +
                        static_cast<int64_t>(k + seqlen) *
                            stride_istate_token +
                        feat_start;
            dte_load<T>(ctx_scalar, t_buf + k * actual_n, cs_src, actual_n);
          } else {
            int x_idx = k - (state_len - seqlen);
            T* x_src = x_ptr +
                       static_cast<int64_t>(seq_start + x_idx) *
                           stride_x_token +
                       feat_start;
            dte_load<T>(ctx_scalar, t_buf + k * actual_n, x_src, actual_n);
          }
        }
      } else {
        for (int k = 0; k < state_len; k++) {
          if (k < state_len - seqlen) {
            dte_zero<T>(ctx_scalar, t_buf + k * actual_n, actual_n);
          } else {
            int x_idx = k - (state_len - seqlen);
            T* x_src = x_ptr +
                       static_cast<int64_t>(seq_start + x_idx) *
                           stride_x_token +
                       feat_start;
            dte_load<T>(ctx_scalar, t_buf + k * actual_n, x_src, actual_n);
          }
        }
      }
      for (int k = 0; k < state_len; k++) {
        T* g_row = cs_out_2d +
                   static_cast<int64_t>(k) * stride_istate_token + feat_start;
        T* l_row = t_buf + k * actual_n;
        dte_store<T>(ctx_scalar, g_row, l_row, actual_n);
      }
    }
  }
}

template __global__ void causal_conv1d_fwd_kernel<float>(
    float*, float*, float*, float*, int32_t*, int8_t*, int32_t*, float*,
    CausalConv1dParams);
template __global__ void causal_conv1d_fwd_kernel<tops::half>(
    tops::half*, tops::half*, tops::half*, tops::half*, int32_t*, int8_t*,
    int32_t*, tops::half*, CausalConv1dParams);
template __global__ void causal_conv1d_fwd_kernel<tops::bfloat>(
    tops::bfloat*, tops::bfloat*, tops::bfloat*, tops::bfloat*, int32_t*,
    int8_t*, int32_t*, tops::bfloat*, CausalConv1dParams);

extern "C" void causal_conv1d_fwd_f32(
    float* x_ptr, float* w_ptr, float* bias_ptr, float* conv_states_ptr,
    int32_t* cache_indices_ptr, int8_t* has_initial_states_ptr,
    int32_t* query_start_loc_ptr, float* o_ptr, CausalConv1dParams* params,
    unsigned int num_blocks, unsigned int dim_blocks, void* stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  causal_conv1d_fwd_kernel<float><<<dim3(num_blocks, 1, 1),
                                         dim3(dim_blocks, 1, 1), 0, stream>>>(
      x_ptr, w_ptr, bias_ptr, conv_states_ptr, cache_indices_ptr,
      has_initial_states_ptr, query_start_loc_ptr, o_ptr, *params);
}

extern "C" void causal_conv1d_fwd_f16(
    __fp16* x_ptr, __fp16* w_ptr, __fp16* bias_ptr, __fp16* conv_states_ptr,
    int32_t* cache_indices_ptr, int8_t* has_initial_states_ptr,
    int32_t* query_start_loc_ptr, __fp16* o_ptr, CausalConv1dParams* params,
    unsigned int num_blocks, unsigned int dim_blocks, void* stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  causal_conv1d_fwd_kernel<tops::half><<<dim3(num_blocks, 1, 1),
                                              dim3(dim_blocks, 1, 1), 0,
                                              stream>>>(
      reinterpret_cast<tops::half*>(x_ptr), reinterpret_cast<tops::half*>(w_ptr),
      reinterpret_cast<tops::half*>(bias_ptr),
      reinterpret_cast<tops::half*>(conv_states_ptr), cache_indices_ptr,
      has_initial_states_ptr, query_start_loc_ptr,
      reinterpret_cast<tops::half*>(o_ptr), *params);
}

extern "C" void causal_conv1d_fwd_bf16(
    __bf16* x_ptr, __bf16* w_ptr, __bf16* bias_ptr, __bf16* conv_states_ptr,
    int32_t* cache_indices_ptr, int8_t* has_initial_states_ptr,
    int32_t* query_start_loc_ptr, __bf16* o_ptr, CausalConv1dParams* params,
    unsigned int num_blocks, unsigned int dim_blocks, void* stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  causal_conv1d_fwd_kernel<tops::bfloat><<<dim3(num_blocks, 1, 1),
                                                dim3(dim_blocks, 1, 1), 0,
                                                stream>>>(
      reinterpret_cast<tops::bfloat*>(x_ptr),
      reinterpret_cast<tops::bfloat*>(w_ptr),
      reinterpret_cast<tops::bfloat*>(bias_ptr),
      reinterpret_cast<tops::bfloat*>(conv_states_ptr), cache_indices_ptr,
      has_initial_states_ptr, query_start_loc_ptr,
      reinterpret_cast<tops::bfloat*>(o_ptr), *params);
}
