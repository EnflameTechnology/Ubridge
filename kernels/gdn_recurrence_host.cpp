/**
 * Copyright 2025 Enflame. All Rights Reserved.
 *
 * Gated Delta Rule Recurrence kernels for GCU.
 *
 * Three variants:
 * 1) gdn_recurrence       - prefill: [BH, seq_len, dim]
 * 2) gdn_decode_slots     - single-token decode with slot indexing
 * 3) gdn_recurrence_varlen - batched variable-length prefill with cu_seqlens
 *
 * State layout: [max_batch, num_heads, v_dim, k_dim] (F32 accumulator)
 *
 * The delta rule per timestep t, per value index v_idx:
 *   state[v_idx, :] *= decay
 *   kv_mem = dot(state[v_idx, :], k_t)
 *   delta  = (v_t[v_idx] - kv_mem) * beta_t
 *   state[v_idx, :] += k_t * delta
 *   out_t[v_idx] = dot(state[v_idx, :], q_t)
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

constexpr int MAX_K_DIM = 256;
constexpr int BV_TILE = 64;

template <typename T>
__global__ void gdn_recurrence_kernel(
    const T* __restrict__ q,       // [BH, S, K]
    const T* __restrict__ k,       // [BH, S, K]
    const T* __restrict__ v,       // [BH, S, V]
    const float* __restrict__ g,   // [BH, S] decay = exp(raw_g)
    const float* __restrict__ beta, // [BH, S]
    float* __restrict__ state,     // [BH, V, K]
    float* __restrict__ out,       // [BH, S, V]
    int bh,
    int seq_len,
    int k_dim,
    int v_dim) {
  const int thread_num = GetThreadNum();
  const int thread_id  = GetThreadIdx();

  __local__ __valigned__ float l_k[MAX_K_DIM];
  __local__ __valigned__ float l_q[MAX_K_DIM];
  __local__ __valigned__ float l_state[MAX_K_DIM];
  __local__ __valigned__ T l_v_tile[BV_TILE];
  __local__ __valigned__ float l_out_tile[BV_TILE];

  tops::private_dte ctx;
  ctx.init();

  const int total_v_tiles = CeilDiv(v_dim, BV_TILE);
  const int total_work = bh * total_v_tiles;

  for (int work_id = thread_id; work_id < total_work; work_id += thread_num) {
    const int bh_idx = work_id / total_v_tiles;
    const int v_tile = work_id % total_v_tiles;
    const int v_start = v_tile * BV_TILE;
    const int v_count = (v_start + BV_TILE <= v_dim) ? BV_TILE : (v_dim - v_start);

    const T* q_bh = q + bh_idx * seq_len * k_dim;
    const T* k_bh = k + bh_idx * seq_len * k_dim;
    const T* v_bh = v + bh_idx * seq_len * v_dim;
    const float* g_bh = g + bh_idx * seq_len;
    const float* beta_bh = beta + bh_idx * seq_len;
    float* out_bh = out + bh_idx * seq_len * v_dim;

    for (int vi = 0; vi < v_count; vi++) {
      const int v_idx = v_start + vi;
      float* state_row = state + bh_idx * k_dim * v_dim + v_idx * k_dim;

      {
        tops::mdspan g_s(tops::Global, state_row, k_dim);
        tops::mdspan l_s(tops::Private, l_state, k_dim);
        tops::memcpy(ctx, l_s, g_s);
      }

      for (int t = 0; t < seq_len; t++) {
        {
          tops::mdspan g_k(tops::Global, const_cast<T*>(k_bh) + t * k_dim, k_dim);
          tops::mdspan l_k_s(tops::Private, reinterpret_cast<T*>(l_k), k_dim);
          tops::memcpy(ctx, l_k_s, g_k);
        }
        tcle::fence<FenceType::L1_SDMEM>();
        tcle::fence<FenceType::L1_VDMEM>();

        float k_f32[MAX_K_DIM];
        if constexpr (std::is_same_v<T, float>) {
          for (int j = 0; j < k_dim; j++) k_f32[j] = l_k[j];
        } else {
          for (int j = 0; j < k_dim; j++) {
            k_f32[j] = static_cast<float>(reinterpret_cast<T*>(l_k)[j]);
          }
        }

        float decay = g_bh[t];
        float beta_t = beta_bh[t];

        T v_elem;
        {
          tops::mdspan g_ve(tops::Global, const_cast<T*>(v_bh) + t * v_dim + v_idx, 1);
          tops::mdspan l_ve(tops::Private, l_v_tile, 1);
          tops::memcpy(ctx, l_ve, g_ve);
        }
        tcle::fence<FenceType::L1_SDMEM>();
        float v_t = static_cast<float>(l_v_tile[0]);

        float kv_mem = 0.0f;
        for (int j = 0; j < k_dim; j++) {
          l_state[j] *= decay;
          kv_mem += l_state[j] * k_f32[j];
        }

        float delta = (v_t - kv_mem) * beta_t;

        {
          tops::mdspan g_q(tops::Global, const_cast<T*>(q_bh) + t * k_dim, k_dim);
          tops::mdspan l_q_s(tops::Private, reinterpret_cast<T*>(l_q), k_dim);
          tops::memcpy(ctx, l_q_s, g_q);
        }
        tcle::fence<FenceType::L1_SDMEM>();
        tcle::fence<FenceType::L1_VDMEM>();

        float q_f32[MAX_K_DIM];
        if constexpr (std::is_same_v<T, float>) {
          for (int j = 0; j < k_dim; j++) q_f32[j] = l_q[j];
        } else {
          for (int j = 0; j < k_dim; j++) {
            q_f32[j] = static_cast<float>(reinterpret_cast<T*>(l_q)[j]);
          }
        }

        float y_t = 0.0f;
        for (int j = 0; j < k_dim; j++) {
          l_state[j] += k_f32[j] * delta;
          y_t += l_state[j] * q_f32[j];
        }

        out_bh[t * v_dim + v_idx] = y_t;
      }

      {
        tops::mdspan g_s(tops::Global, state_row, k_dim);
        tops::mdspan l_s(tops::Private, l_state, k_dim);
        tops::memcpy(ctx, g_s, l_s);
      }
    }
  }
}

template <typename T>
__global__ void gdn_decode_slots_kernel(
    const T* __restrict__ q,          // [batch, heads, k_dim]
    const T* __restrict__ k,          // [batch, heads, k_dim]
    const T* __restrict__ v,          // [batch, heads, v_dim]
    const T* __restrict__ g,          // [batch, heads] decay = exp(raw_g)
    const T* __restrict__ beta,       // [batch, heads]
    float* __restrict__ state,        // [max_batch, heads, v_dim, k_dim]
    const int64_t* __restrict__ slots, // [batch]
    T* __restrict__ out,              // [batch, heads, v_dim]
    int batch,
    int heads,
    int k_dim,
    int v_dim) {
  const int thread_num = GetThreadNum();
  const int thread_id  = GetThreadIdx();

  __local__ __valigned__ float l_k[MAX_K_DIM];
  __local__ __valigned__ float l_q[MAX_K_DIM];
  __local__ __valigned__ float l_state[MAX_K_DIM];
  __local__ __valigned__ int64_t l_slots[1];

  tops::private_dte ctx;
  ctx.init();

  const int total_v_tiles = CeilDiv(v_dim, BV_TILE);
  const int total_work = batch * heads * total_v_tiles;

  for (int work_id = thread_id; work_id < total_work; work_id += thread_num) {
    const int bh_vt = work_id;
    const int v_tile_idx = bh_vt % total_v_tiles;
    const int bh_idx = bh_vt / total_v_tiles;
    const int b = bh_idx / heads;
    const int h = bh_idx % heads;
    const int v_start = v_tile_idx * BV_TILE;
    const int v_count = (v_start + BV_TILE <= v_dim) ? BV_TILE : (v_dim - v_start);

    {
      tops::mdspan g_sl(tops::Global, const_cast<int64_t*>(slots) + b, 1);
      tops::mdspan l_sl(tops::Private, l_slots, 1);
      tops::memcpy(ctx, l_sl, g_sl);
    }
    tcle::fence<FenceType::L1_SDMEM>();
    int64_t slot = l_slots[0];
    if (slot < 0) continue;

    float decay = static_cast<float>(g[bh_idx]);
    float beta_t = static_cast<float>(beta[bh_idx]);

    {
      tops::mdspan g_k(tops::Global, const_cast<T*>(k) + bh_idx * k_dim, k_dim);
      tops::mdspan l_k_s(tops::Private, reinterpret_cast<T*>(l_k), k_dim);
      tops::memcpy(ctx, l_k_s, g_k);
    }
    {
      tops::mdspan g_q(tops::Global, const_cast<T*>(q) + bh_idx * k_dim, k_dim);
      tops::mdspan l_q_s(tops::Private, reinterpret_cast<T*>(l_q), k_dim);
      tops::memcpy(ctx, l_q_s, g_q);
    }
    tcle::fence<FenceType::L1_SDMEM>();
    tcle::fence<FenceType::L1_VDMEM>();

    float k_f32[MAX_K_DIM];
    float q_f32[MAX_K_DIM];
    if constexpr (std::is_same_v<T, float>) {
      for (int j = 0; j < k_dim; j++) { k_f32[j] = l_k[j]; q_f32[j] = l_q[j]; }
    } else {
      for (int j = 0; j < k_dim; j++) {
        k_f32[j] = static_cast<float>(reinterpret_cast<T*>(l_k)[j]);
        q_f32[j] = static_cast<float>(reinterpret_cast<T*>(l_q)[j]);
      }
    }

    for (int vi = 0; vi < v_count; vi++) {
      const int v_idx = v_start + vi;
      float* state_row = state + ((slot * heads + h) * v_dim + v_idx) * k_dim;

      {
        tops::mdspan g_s(tops::Global, state_row, k_dim);
        tops::mdspan l_s(tops::Private, l_state, k_dim);
        tops::memcpy(ctx, l_s, g_s);
      }
      tcle::fence<FenceType::L1_SDMEM>();

      float v_t = static_cast<float>(v[bh_idx * v_dim + v_idx]);

      float kv_mem = 0.0f;
      for (int j = 0; j < k_dim; j++) {
        l_state[j] *= decay;
        kv_mem += l_state[j] * k_f32[j];
      }

      float delta = (v_t - kv_mem) * beta_t;

      float y = 0.0f;
      for (int j = 0; j < k_dim; j++) {
        l_state[j] += k_f32[j] * delta;
        y += l_state[j] * q_f32[j];
      }

      {
        tops::mdspan g_s(tops::Global, state_row, k_dim);
        tops::mdspan l_s(tops::Private, l_state, k_dim);
        tops::memcpy(ctx, g_s, l_s);
      }

      out[bh_idx * v_dim + v_idx] = static_cast<T>(y);
    }
  }
}

template <typename T>
__global__ void gdn_recurrence_varlen_kernel(
    const T* __restrict__ q,           // [total_tokens, num_heads, k_dim]
    const T* __restrict__ k,           // [total_tokens, num_heads, k_dim]
    const T* __restrict__ v,           // [total_tokens, num_heads, v_dim]
    const T* __restrict__ g,           // [total_tokens, num_heads]
    const T* __restrict__ beta,        // [total_tokens, num_heads]
    float* __restrict__ state,         // [max_batch, num_heads, v_dim, k_dim]
    const int64_t* __restrict__ slots, // [batch]
    T* __restrict__ out,               // [total_tokens, num_heads, v_dim]
    const uint32_t* __restrict__ cu_seqlens, // [batch + 1]
    int batch,
    int num_heads,
    int k_dim,
    int v_dim) {
  const int thread_num = GetThreadNum();
  const int thread_id  = GetThreadIdx();

  __local__ __valigned__ float l_k[MAX_K_DIM];
  __local__ __valigned__ float l_q[MAX_K_DIM];
  __local__ __valigned__ float l_state[MAX_K_DIM];
  __local__ __valigned__ uint32_t l_cu[2];
  __local__ __valigned__ int64_t l_slot[1];

  tops::private_dte ctx;
  ctx.init();

  const int total_v_tiles = CeilDiv(v_dim, BV_TILE);
  const int total_work = batch * num_heads * total_v_tiles;

  for (int work_id = thread_id; work_id < total_work; work_id += thread_num) {
    const int v_tile_idx = work_id % total_v_tiles;
    const int sh_idx = work_id / total_v_tiles;
    const int seq_idx = sh_idx / num_heads;
    const int head_idx = sh_idx % num_heads;
    const int v_start = v_tile_idx * BV_TILE;
    const int v_count = (v_start + BV_TILE <= v_dim) ? BV_TILE : (v_dim - v_start);

    {
      tops::mdspan g_cu(tops::Global, const_cast<uint32_t*>(cu_seqlens) + seq_idx, 2);
      tops::mdspan l_cu_s(tops::Private, l_cu, 2);
      tops::memcpy(ctx, l_cu_s, g_cu);
    }
    {
      tops::mdspan g_sl(tops::Global, const_cast<int64_t*>(slots) + seq_idx, 1);
      tops::mdspan l_sl(tops::Private, l_slot, 1);
      tops::memcpy(ctx, l_sl, g_sl);
    }
    tcle::fence<FenceType::L1_SDMEM>();

    const int start = static_cast<int>(l_cu[0]);
    const int end   = static_cast<int>(l_cu[1]);
    const int seq_len = end - start;
    const int64_t slot = l_slot[0];
    if (slot < 0 || seq_len <= 0) continue;

    const int token_stride_k = num_heads * k_dim;
    const int token_stride_v = num_heads * v_dim;
    const int token_stride_g = num_heads;

    for (int vi = 0; vi < v_count; vi++) {
      const int v_idx = v_start + vi;
      float* state_row = state + ((slot * num_heads + head_idx) * v_dim + v_idx) * k_dim;

      {
        tops::mdspan g_s(tops::Global, state_row, k_dim);
        tops::mdspan l_s(tops::Private, l_state, k_dim);
        tops::memcpy(ctx, l_s, g_s);
      }
      tcle::fence<FenceType::L1_SDMEM>();

      for (int t = 0; t < seq_len; t++) {
        const int token_idx = start + t;

        {
          tops::mdspan g_k(tops::Global,
              const_cast<T*>(k) + token_idx * token_stride_k + head_idx * k_dim, k_dim);
          tops::mdspan l_k_s(tops::Private, reinterpret_cast<T*>(l_k), k_dim);
          tops::memcpy(ctx, l_k_s, g_k);
        }
        tcle::fence<FenceType::L1_SDMEM>();
        tcle::fence<FenceType::L1_VDMEM>();

        float k_f32[MAX_K_DIM];
        if constexpr (std::is_same_v<T, float>) {
          for (int j = 0; j < k_dim; j++) k_f32[j] = l_k[j];
        } else {
          for (int j = 0; j < k_dim; j++)
            k_f32[j] = static_cast<float>(reinterpret_cast<T*>(l_k)[j]);
        }

        float decay = static_cast<float>(g[token_idx * token_stride_g + head_idx]);
        float beta_t = static_cast<float>(beta[token_idx * token_stride_g + head_idx]);
        float v_t = static_cast<float>(v[token_idx * token_stride_v + head_idx * v_dim + v_idx]);

        float kv_mem = 0.0f;
        for (int j = 0; j < k_dim; j++) {
          l_state[j] *= decay;
          kv_mem += l_state[j] * k_f32[j];
        }

        float delta = (v_t - kv_mem) * beta_t;

        {
          tops::mdspan g_q(tops::Global,
              const_cast<T*>(q) + token_idx * token_stride_k + head_idx * k_dim, k_dim);
          tops::mdspan l_q_s(tops::Private, reinterpret_cast<T*>(l_q), k_dim);
          tops::memcpy(ctx, l_q_s, g_q);
        }
        tcle::fence<FenceType::L1_SDMEM>();
        tcle::fence<FenceType::L1_VDMEM>();

        float q_f32[MAX_K_DIM];
        if constexpr (std::is_same_v<T, float>) {
          for (int j = 0; j < k_dim; j++) q_f32[j] = l_q[j];
        } else {
          for (int j = 0; j < k_dim; j++)
            q_f32[j] = static_cast<float>(reinterpret_cast<T*>(l_q)[j]);
        }

        float y_t = 0.0f;
        for (int j = 0; j < k_dim; j++) {
          l_state[j] += k_f32[j] * delta;
          y_t += l_state[j] * q_f32[j];
        }

        out[token_idx * token_stride_v + head_idx * v_dim + v_idx] = static_cast<T>(y_t);
      }

      {
        tops::mdspan g_s(tops::Global, state_row, k_dim);
        tops::mdspan l_s(tops::Private, l_state, k_dim);
        tops::memcpy(ctx, g_s, l_s);
      }
    }
  }
}

template __global__ void gdn_recurrence_kernel<float>(
    const float*, const float*, const float*, const float*, const float*,
    float*, float*, int, int, int, int);
template __global__ void gdn_recurrence_kernel<tops::half>(
    const tops::half*, const tops::half*, const tops::half*, const float*, const float*,
    float*, float*, int, int, int, int);
template __global__ void gdn_recurrence_kernel<tops::bfloat>(
    const tops::bfloat*, const tops::bfloat*, const tops::bfloat*, const float*, const float*,
    float*, float*, int, int, int, int);

template __global__ void gdn_decode_slots_kernel<float>(
    const float*, const float*, const float*, const float*, const float*,
    float*, const int64_t*, float*, int, int, int, int);
template __global__ void gdn_decode_slots_kernel<tops::half>(
    const tops::half*, const tops::half*, const tops::half*, const tops::half*, const tops::half*,
    float*, const int64_t*, tops::half*, int, int, int, int);
template __global__ void gdn_decode_slots_kernel<tops::bfloat>(
    const tops::bfloat*, const tops::bfloat*, const tops::bfloat*, const tops::bfloat*, const tops::bfloat*,
    float*, const int64_t*, tops::bfloat*, int, int, int, int);

template __global__ void gdn_recurrence_varlen_kernel<float>(
    const float*, const float*, const float*, const float*, const float*,
    float*, const int64_t*, float*, const uint32_t*, int, int, int, int);
template __global__ void gdn_recurrence_varlen_kernel<tops::half>(
    const tops::half*, const tops::half*, const tops::half*, const tops::half*, const tops::half*,
    float*, const int64_t*, tops::half*, const uint32_t*, int, int, int, int);
template __global__ void gdn_recurrence_varlen_kernel<tops::bfloat>(
    const tops::bfloat*, const tops::bfloat*, const tops::bfloat*, const tops::bfloat*, const tops::bfloat*,
    float*, const int64_t*, tops::bfloat*, const uint32_t*, int, int, int, int);

// --- Recurrence extern "C" ---

extern "C" void gdn_recurrence_f32(
    const float* q, const float* k, const float* v,
    const float* g, const float* beta, float* state, float* out,
    int bh, int seq_len, int k_dim, int v_dim,
    unsigned int num_blocks, unsigned int dim_blocks, void* stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  gdn_recurrence_kernel<float><<<dim3(num_blocks,1,1), dim3(dim_blocks,1,1), 0, stream>>>(
      q, k, v, g, beta, state, out, bh, seq_len, k_dim, v_dim);
}

extern "C" void gdn_recurrence_f16(
    const __fp16* q, const __fp16* k, const __fp16* v,
    const float* g, const float* beta, float* state, float* out,
    int bh, int seq_len, int k_dim, int v_dim,
    unsigned int num_blocks, unsigned int dim_blocks, void* stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  gdn_recurrence_kernel<tops::half><<<dim3(num_blocks,1,1), dim3(dim_blocks,1,1), 0, stream>>>(
      reinterpret_cast<const tops::half*>(q),
      reinterpret_cast<const tops::half*>(k),
      reinterpret_cast<const tops::half*>(v),
      g, beta, state, out, bh, seq_len, k_dim, v_dim);
}

extern "C" void gdn_recurrence_bf16(
    const __bf16* q, const __bf16* k, const __bf16* v,
    const float* g, const float* beta, float* state, float* out,
    int bh, int seq_len, int k_dim, int v_dim,
    unsigned int num_blocks, unsigned int dim_blocks, void* stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  gdn_recurrence_kernel<tops::bfloat><<<dim3(num_blocks,1,1), dim3(dim_blocks,1,1), 0, stream>>>(
      reinterpret_cast<const tops::bfloat*>(q),
      reinterpret_cast<const tops::bfloat*>(k),
      reinterpret_cast<const tops::bfloat*>(v),
      g, beta, state, out, bh, seq_len, k_dim, v_dim);
}

// --- Decode slots extern "C" ---

extern "C" void gdn_decode_slots_f32(
    const float* q, const float* k, const float* v,
    const float* g, const float* beta,
    float* state, const int64_t* slots, float* out,
    int batch, int heads, int k_dim, int v_dim,
    unsigned int num_blocks, unsigned int dim_blocks, void* stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  gdn_decode_slots_kernel<float><<<dim3(num_blocks,1,1), dim3(dim_blocks,1,1), 0, stream>>>(
      q, k, v, g, beta, state, slots, out, batch, heads, k_dim, v_dim);
}

extern "C" void gdn_decode_slots_f16(
    const __fp16* q, const __fp16* k, const __fp16* v,
    const __fp16* g, const __fp16* beta,
    float* state, const int64_t* slots, __fp16* out,
    int batch, int heads, int k_dim, int v_dim,
    unsigned int num_blocks, unsigned int dim_blocks, void* stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  gdn_decode_slots_kernel<tops::half><<<dim3(num_blocks,1,1), dim3(dim_blocks,1,1), 0, stream>>>(
      reinterpret_cast<const tops::half*>(q),
      reinterpret_cast<const tops::half*>(k),
      reinterpret_cast<const tops::half*>(v),
      reinterpret_cast<const tops::half*>(g),
      reinterpret_cast<const tops::half*>(beta),
      state, slots,
      reinterpret_cast<tops::half*>(out),
      batch, heads, k_dim, v_dim);
}

extern "C" void gdn_decode_slots_bf16(
    const __bf16* q, const __bf16* k, const __bf16* v,
    const __bf16* g, const __bf16* beta,
    float* state, const int64_t* slots, __bf16* out,
    int batch, int heads, int k_dim, int v_dim,
    unsigned int num_blocks, unsigned int dim_blocks, void* stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  gdn_decode_slots_kernel<tops::bfloat><<<dim3(num_blocks,1,1), dim3(dim_blocks,1,1), 0, stream>>>(
      reinterpret_cast<const tops::bfloat*>(q),
      reinterpret_cast<const tops::bfloat*>(k),
      reinterpret_cast<const tops::bfloat*>(v),
      reinterpret_cast<const tops::bfloat*>(g),
      reinterpret_cast<const tops::bfloat*>(beta),
      state, slots,
      reinterpret_cast<tops::bfloat*>(out),
      batch, heads, k_dim, v_dim);
}

// --- Varlen recurrence extern "C" ---

extern "C" void gdn_recurrence_varlen_f32(
    const float* q, const float* k, const float* v,
    const float* g, const float* beta,
    float* state, const int64_t* slots, float* out,
    const uint32_t* cu_seqlens,
    int batch, int num_heads, int k_dim, int v_dim,
    unsigned int num_blocks, unsigned int dim_blocks, void* stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  gdn_recurrence_varlen_kernel<float><<<dim3(num_blocks,1,1), dim3(dim_blocks,1,1), 0, stream>>>(
      q, k, v, g, beta, state, slots, out, cu_seqlens,
      batch, num_heads, k_dim, v_dim);
}

extern "C" void gdn_recurrence_varlen_f16(
    const __fp16* q, const __fp16* k, const __fp16* v,
    const __fp16* g, const __fp16* beta,
    float* state, const int64_t* slots, __fp16* out,
    const uint32_t* cu_seqlens,
    int batch, int num_heads, int k_dim, int v_dim,
    unsigned int num_blocks, unsigned int dim_blocks, void* stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  gdn_recurrence_varlen_kernel<tops::half><<<dim3(num_blocks,1,1), dim3(dim_blocks,1,1), 0, stream>>>(
      reinterpret_cast<const tops::half*>(q),
      reinterpret_cast<const tops::half*>(k),
      reinterpret_cast<const tops::half*>(v),
      reinterpret_cast<const tops::half*>(g),
      reinterpret_cast<const tops::half*>(beta),
      state, slots,
      reinterpret_cast<tops::half*>(out),
      cu_seqlens, batch, num_heads, k_dim, v_dim);
}

extern "C" void gdn_recurrence_varlen_bf16(
    const __bf16* q, const __bf16* k, const __bf16* v,
    const __bf16* g, const __bf16* beta,
    float* state, const int64_t* slots, __bf16* out,
    const uint32_t* cu_seqlens,
    int batch, int num_heads, int k_dim, int v_dim,
    unsigned int num_blocks, unsigned int dim_blocks, void* stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  gdn_recurrence_varlen_kernel<tops::bfloat><<<dim3(num_blocks,1,1), dim3(dim_blocks,1,1), 0, stream>>>(
      reinterpret_cast<const tops::bfloat*>(q),
      reinterpret_cast<const tops::bfloat*>(k),
      reinterpret_cast<const tops::bfloat*>(v),
      reinterpret_cast<const tops::bfloat*>(g),
      reinterpret_cast<const tops::bfloat*>(beta),
      state, slots,
      reinterpret_cast<tops::bfloat*>(out),
      cu_seqlens, batch, num_heads, k_dim, v_dim);
}
