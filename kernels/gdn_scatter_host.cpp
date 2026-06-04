/**
 * Copyright 2025 Enflame. All Rights Reserved.
 *
 * Mamba scatter rows kernel for GCU:
 *   For each row i: dst[slots[i], :] = src[i, :]
 *
 * Used to scatter batch-local state back into global slot-indexed cache.
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

template <typename T>
__global__ void mamba_scatter_rows_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    const int64_t* __restrict__ slots,
    int num_rows,
    int row_elems,
    int src_row_stride,
    int dst_row_stride) {
  const int thread_num = GetThreadNum();
  const int thread_id  = GetThreadIdx();

  constexpr int TILE = 256;
  __local__ __valigned__ T l_buf[TILE];
  __local__ __valigned__ int64_t l_slot[1];

  tops::private_dte ctx;
  ctx.init();

  for (int row = thread_id; row < num_rows; row += thread_num) {
    {
      tops::mdspan g_sl(tops::Global, const_cast<int64_t*>(slots) + row, 1);
      tops::mdspan l_sl(tops::Private, l_slot, 1);
      tops::memcpy(ctx, l_sl, g_sl);
    }
    tcle::fence<FenceType::L1_SDMEM>();
    int64_t slot = l_slot[0];
    if (slot < 0) continue;

    const T* src_row = src + row * src_row_stride;
    T*       dst_row = dst + slot * dst_row_stride;

    const int num_tiles = CeilDiv(row_elems, TILE);
    for (int ti = 0; ti < num_tiles; ti++) {
      const int offset = ti * TILE;
      const int count = (offset + TILE <= row_elems) ? TILE : (row_elems - offset);

      tops::mdspan g_s(tops::Global, const_cast<T*>(src_row) + offset, count);
      tops::mdspan l_s(tops::Private, l_buf, count);
      tops::memcpy(ctx, l_s, g_s);

      tops::mdspan g_d(tops::Global, dst_row + offset, count);
      tops::mdspan l_d(tops::Private, l_buf, count);
      tops::memcpy(ctx, g_d, l_d);
    }
  }
}

template __global__ void mamba_scatter_rows_kernel<float>(
    const float*, float*, const int64_t*, int, int, int, int);
template __global__ void mamba_scatter_rows_kernel<tops::half>(
    const tops::half*, tops::half*, const int64_t*, int, int, int, int);
template __global__ void mamba_scatter_rows_kernel<tops::bfloat>(
    const tops::bfloat*, tops::bfloat*, const int64_t*, int, int, int, int);

extern "C" void gdn_mamba_scatter_f32(
    const float* src, float* dst, const int64_t* slots,
    int num_rows, int row_elems, int src_row_stride, int dst_row_stride,
    unsigned int num_blocks, unsigned int dim_blocks, void* stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  mamba_scatter_rows_kernel<float><<<dim3(num_blocks,1,1), dim3(dim_blocks,1,1), 0, stream>>>(
      src, dst, slots, num_rows, row_elems, src_row_stride, dst_row_stride);
}

extern "C" void gdn_mamba_scatter_f16(
    const __fp16* src, __fp16* dst, const int64_t* slots,
    int num_rows, int row_elems, int src_row_stride, int dst_row_stride,
    unsigned int num_blocks, unsigned int dim_blocks, void* stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  mamba_scatter_rows_kernel<tops::half><<<dim3(num_blocks,1,1), dim3(dim_blocks,1,1), 0, stream>>>(
      reinterpret_cast<const tops::half*>(src),
      reinterpret_cast<tops::half*>(dst),
      slots, num_rows, row_elems, src_row_stride, dst_row_stride);
}

extern "C" void gdn_mamba_scatter_bf16(
    const __bf16* src, __bf16* dst, const int64_t* slots,
    int num_rows, int row_elems, int src_row_stride, int dst_row_stride,
    unsigned int num_blocks, unsigned int dim_blocks, void* stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  mamba_scatter_rows_kernel<tops::bfloat><<<dim3(num_blocks,1,1), dim3(dim_blocks,1,1), 0, stream>>>(
      reinterpret_cast<const tops::bfloat*>(src),
      reinterpret_cast<tops::bfloat*>(dst),
      slots, num_rows, row_elems, src_row_stride, dst_row_stride);
}
