/*
 * Copyright 2021-2024 Enflame. All Rights Reserved.

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
 *
 * @file    copy.cpp
 * @brief
 *
 * @author  Guoqing Bao
 * @date    2023-11-07 to 2024-03-04
 * @version V0.1
 * @par     Copyright (c) Enflame Tech Company.
 * @par     History: optimizition of general copy kernel with L1 and L2 buffer, standalone transpose kernel
 * @par     Comments: gcu kernel for general copy.
 */

#include <stdio.h>
#include <tops.h>
#include <tops/topsrtc.h>
#include <tops/half.h>
#include <tops/bfloat.h>
#include <string>
#include <tops/tops_runtime.h>
#include <tops/topsrtc.h>
#include <krt/mmu.h>
#include "utils/vector_ex.h"
#include <acore_op.h>
#include "utils/utils.h"
using namespace std;
using namespace tops;
#define PING_PONG_SIZE 2
#define TILE_SIZE AlignDown(((VDMEM_VALID_SIZE) / 32), 256)
#define GATHER_COPY
const int COPY_TILESIZE = 256 * 1024;
const int COPY_L1SIZE = VDMEM_VALID_SIZE - COPY_TILESIZE;


// #define PRINTHELPER(ARRAY, SZ, MSG) \
//   printf(MSG); \
//   for (int i=0; i< SZ; i++) \
//     printf("%d, ", (int)ARRAY[i]); \
//   printf("\n") \


__device__ __forceinline__ unsigned int get_strided_index(
    unsigned int idx,
    const int num_dims,
    const int *dims,
    const int *strides
) {
    unsigned int strided_i = 0;
    for (unsigned int d = 0; d < num_dims; d++) {
        unsigned int dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}

template <typename T, int RANK, int BPE>
__device__ void ucopy_multithread_cache(T* in, T* out, const size_t in_size, const size_t out_size, size_t* dims_and_strides, char* raw_cache, T* buffer) {
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx); 
    __private_dte__ tops_dte_ctx_t cache_ctx;
    tops::dte_scope s1(cache_ctx);

    int dst_dim[RANK];
    int dst_strides[RANK];
    for (int j = 0; j < RANK; ++j) {
      dst_dim[j] = dims_and_strides[j];
      dst_strides[j] = dims_and_strides[RANK + j];
    }

    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNum();

    const int TILESIZE = COPY_TILESIZE / BPE;

    tops::mdspan out_hbm(tops::Global, out, out_size);
    int N = out_size;
    int THREAD_STEP = 1;
    int thread_step = 1;
    GetThreadStep(N, thread_step, THREAD_STEP);

    bool cacheable_l1 = in_size > MAX_THREADS && in_size * sizeof(T) < COPY_L1SIZE;
    bool cacheable_l2 = in_size > MAX_THREADS && in_size * sizeof(T) < SHARE_BUFFER_SIZE;

    T* src_cached = reinterpret_cast<T*>(raw_cache);
    mapped_ptr src_l3_addr;
    if (cacheable_l1){
      src_cached = reinterpret_cast<T*>(buffer + TILESIZE);
      ctx.config_memcpy(
        tops::mdspan(tops::Private, src_cached, in_size),
        tops::mdspan(tops::Global, in, in_size));
      ctx.trigger_and_wait();
    } else if (cacheable_l2) {
      if (GetThreadIdxInBlock() == 0) {
        cache_ctx.config_memcpy(
          tops::mdspan(tops::Shared, src_cached, in_size),
          tops::mdspan(tops::Global, in, in_size));
        cache_ctx.trigger_and_wait();
      }
    } else {
      int in_map_size = AlignUp(in_size, NUMS_SPLIT) * sizeof(T);
      src_l3_addr = map_mem_ex(reinterpret_cast<generic_ptr>(in), in_map_size);
      src_cached = reinterpret_cast<T*>(src_l3_addr);
    }

    __syncthreads();

    for (int i = 0; i < thread_step; i+=TILESIZE) {
      int bufsize = (i + TILESIZE < thread_step) ? TILESIZE : thread_step - i;
      int offset = thread_id * THREAD_STEP + i;
      for (int j = 0; j< bufsize; j++) {
        size_t idx = offset + j;
        unsigned int strided_i = get_strided_index(idx, RANK, dst_dim, dst_strides);
        buffer[j] = src_cached[strided_i];
      }
      tops::mdspan buffer_l1(tops::Private, buffer, bufsize);
      tops::deslice(ctx, out_hbm, buffer_l1, {offset});
    }
    if (!cacheable_l1 && !cacheable_l2) {
      int in_map_size = AlignUp(in_size, NUMS_SPLIT) * sizeof(T);
      unmap_mem_ex(src_l3_addr, in_map_size);
    }  
}

#ifdef GATHER_COPY

template <int RANK>
__device__ __forceinline__ VecIndexType get_batch_strided_index(VecIndexType &indexes, VecIndexType results, VecIndexType dst_shape[], VecIndexType dst_strides[]) {
    VecIndexType vec_rem[RANK];
    vec_rem[0] = indexes;
    #pragma clang loop unroll(enable)
    for (int i = 0; i < RANK; i++) {
      unsigned int dim_idx = RANK - 1 - i;
      results = vadd(vmul(vrem(vec_rem[i], dst_shape[dim_idx]), dst_strides[dim_idx]), results); 
      vec_rem[i + 1] = vdiv(vec_rem[i], dst_shape[dim_idx]);
    }
    return results;
}

template <typename T, int RANK, int BPE>
__device__ void ucopy_multithread_gather(T* in, T* out, const size_t in_size, const size_t out_size, size_t* dims_and_strides, char* raw_cache, T* buffer) {
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx); 
    __private_dte__ tops_dte_ctx_t cache_ctx;
    tops::dte_scope s1(cache_ctx);
    tops::mdspan out_hbm(tops::Global, out, out_size);

    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNum();

    // constexpr int TILESIZE = vector_length<va16u32x4>::value;
    constexpr int TILESIZE = sizeof(typename tops::unified_scalar<T>::type) * TOPS_VECTOR_LENGTH / sizeof(T);
    const int BUFFER_SIZE = COPY_TILESIZE / BPE;
    constexpr int ITERATION = TILESIZE / TOPS_VECTOR_LENGTH;
    int N = out_size;
    int THREAD_STEP = 1;
    int thread_step = 1;
    GetThreadStep(N, thread_step, THREAD_STEP);

    int dst_dim[RANK];
    int dst_strides[RANK];
    for (int j = 0; j < RANK; ++j) {
      dst_dim[j] = dims_and_strides[j];
      dst_strides[j] = dims_and_strides[RANK + j];
    }

    bool cacheable_l1 = in_size > MAX_THREADS && in_size * sizeof(T) < COPY_L1SIZE;
    bool cacheable_l2 = in_size > MAX_THREADS && in_size * sizeof(T) < SHARE_BUFFER_SIZE;

    T* src_cached = reinterpret_cast<T*>(raw_cache);
    mapped_ptr src_l3_addr;

    if (cacheable_l1){
      src_cached = reinterpret_cast<T*>(buffer + BUFFER_SIZE);
      ctx.config_memcpy(
        tops::mdspan(tops::Private, src_cached, in_size),
        tops::mdspan(tops::Global, in, in_size));
      ctx.trigger_and_wait();
    } else if (cacheable_l2) {
      if (thread_id == 0) {
        cache_ctx.config_memcpy(
          tops::mdspan(tops::Shared, src_cached, in_size),
          tops::mdspan(tops::Global, in, in_size));
        cache_ctx.trigger_and_wait();
      }
    } else {
      int in_map_size = AlignUp(in_size, NUMS_SPLIT) * sizeof(T);
      src_l3_addr = map_mem_ex(reinterpret_cast<generic_ptr>(in), in_map_size);
      src_cached = reinterpret_cast<T*>(src_l3_addr);
    }

    __syncthreads();

    va16u32x4 vec_dst_shape[RANK];
    va16u32x4 vec_dst_strides[RANK];
    // using vtype = typename tops::scalar_to_vector<T, TOPS_VECTOR_LENGTH>::type;
    using vtype = typename tops::scalar_to_vector<T, TILESIZE>::type;
    for (int i = 0; i < RANK; ++i) {
      vec_dst_shape[i] = vbroadcast<va16u32x4>((unsigned int)dst_dim[i]);
      vec_dst_strides[i] = vbroadcast<va16u32x4>((unsigned int)dst_strides[i]);
    }
    va16u32x4 vec_bpe = vbroadcast<va16u32x4>((unsigned int)sizeof(T));
    va16u32x4 strided_indexes[ITERATION], strided_indexes_[ITERATION], indexes[ITERATION], indexes_[ITERATION];
    T *pbuffer = buffer;
    size_t buf_written = 0;
    int init_offset = -1;
    for (int i = 0; i < thread_step; i+=TILESIZE) {
      int bufsize = (i + TILESIZE < thread_step) ? TILESIZE : thread_step - i;
      int offset = thread_id * THREAD_STEP + i;
      if (init_offset < 0) init_offset = offset;
      va16u32x4 idx_results = vbroadcast<va16u32x4>((unsigned int)0);
      for (int k = 0; k < ITERATION; k++) {
        indexes[k] = viota<va16u32x4>((unsigned int)(offset + k * TOPS_VECTOR_LENGTH));
        strided_indexes[k] = get_batch_strided_index<RANK>(indexes[k], idx_results, vec_dst_shape, vec_dst_strides);
        strided_indexes_[k] = vmul(strided_indexes[k], vec_bpe);
      }
      if (bufsize == TILESIZE) {
        auto result = vgather<vtype>(src_cached, strided_indexes_);
        vstore(result, pbuffer);
      } else {
        for (int j = 0; j < bufsize; j++) {
          pbuffer[j] = src_cached[strided_indexes[j / TOPS_VECTOR_LENGTH][j % TOPS_VECTOR_LENGTH]];
        }
      }
      buf_written += bufsize;
      pbuffer += bufsize;
      if (buf_written + TILESIZE >= BUFFER_SIZE || i + bufsize >= thread_step - 1 || bufsize < TILESIZE) {
        tops::mdspan buffer_l1(tops::Private, buffer, buf_written);
        tops::deslice(ctx, out_hbm, buffer_l1, {init_offset});
        buf_written = 0;
        pbuffer = buffer;
        init_offset = -1;
      }
    }
    if (!cacheable_l1 && !cacheable_l2) {
      int in_map_size = AlignUp(in_size, NUMS_SPLIT) * sizeof(T);
      unmap_mem_ex(src_l3_addr, in_map_size);
    }
}
#endif

//dims_and_strides: dst shape, dst stride, dst layout, origin shape
template <typename T, int RANK>
__device__ void transpose_kernel(T* in, T* out, const size_t in_size, const size_t out_size, size_t* dims_and_strides, char* raw_cache, T* buffer) {
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx); 
    int src_shape[RANK];
    int dst_shape[RANK];
    int dst_layout[RANK];
    for (int j = 0; j < RANK; ++j) {
      dst_shape[j] = dims_and_strides[j];
      dst_layout[j] = dims_and_strides[j + 2 * RANK];
      src_shape[j] = dims_and_strides[j + 3 * RANK];
    }
    
    bool cacheable_l1 = in_size * sizeof(T) * 2 < COPY_L1SIZE;

    T* src_cached = reinterpret_cast<T*>(raw_cache);
    if (cacheable_l1){
      T* dst_cached = reinterpret_cast<T*>(buffer + in_size);
      tops::memcpy(ctx, 
        tops::mdspan(tops::Private, buffer, in_size),
        tops::mdspan(tops::Global, in, in_size));
      tops::transpose(ctx, tops::mdspan(tops::Private, dst_cached, dst_shape), tops::mdspan(tops::Private, buffer, src_shape), dst_layout);
      tops::deslice(ctx, tops::mdspan(tops::Global, out, out_size), tops::mdspan(tops::Private, dst_cached, out_size), 0);

    } else {
      tops::mdspan src_mem(tops::Global, in, src_shape);
      tops::mdspan dst_mem(tops::Global, out, dst_shape);
      tops::transpose(ctx, dst_mem, src_mem, dst_layout); 
    }
}

//dims_and_strides: dst shape, dst stride, dst layout, origin shape
template <typename T, int RANK>
__device__ void broadcast_kernel(T* in, T* out, int src_rank, const size_t in_size, const size_t out_size, size_t* dims_and_strides, char* raw_cache, T* buffer) {
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx); 
    int dst_shape[RANK];
    int src_shape[RANK];

    for (int j = 0; j < RANK; ++j) {
      dst_shape[j] = dims_and_strides[j];
    }

    for (int j = RANK - 1; j >=0; j--) {
      if (j - (RANK - src_rank) < 0) {
        src_shape[j] = 1;
      } else {
        src_shape[j] = dims_and_strides[3 * RANK + j - (RANK - src_rank)];
      }
    }
    bool cacheable_l1 = (in_size + out_size) * sizeof(T) < COPY_L1SIZE;
    bool cacheable_l2 = (in_size + out_size) * sizeof(T) < SHARE_BUFFER_SIZE;

    T* src_cached = reinterpret_cast<T*>(raw_cache);
    if (in_size == 1) {
        tops::mdspan src_mem(tops::Global, in, 1);
        tops::mdspan dst_mem(tops::Global, out, out_size);
        tops::broadcast(ctx, dst_mem, src_mem);
    } else if (cacheable_l1){
      T* dst_cached = reinterpret_cast<T*>(buffer + in_size);
      tops::memcpy(ctx, 
        tops::mdspan(tops::Private, buffer, in_size),
        tops::mdspan(tops::Global, in, in_size));
      tops::broadcast(ctx, tops::mdspan(tops::Private, dst_cached, dst_shape), tops::mdspan(tops::Private, buffer, src_shape));
      tops::deslice(ctx, tops::mdspan(tops::Global, out, out_size), tops::mdspan(tops::Private, dst_cached, out_size), 0);

    } else if (cacheable_l2) {
      T* dst_cached = reinterpret_cast<T*>(src_cached + in_size);
      tops::memcpy(ctx, 
        tops::mdspan(tops::Shared, src_cached, in_size),
        tops::mdspan(tops::Global, in, in_size));
      tops::broadcast(ctx, tops::mdspan(tops::Shared, dst_cached, dst_shape), tops::mdspan(tops::Shared, src_cached, src_shape));
      tops::deslice(ctx, tops::mdspan(tops::Global, out, out_size), tops::mdspan(tops::Shared, dst_cached, out_size), 0);
    } else {
      tops::mdspan src_mem(tops::Global, in, src_shape);
      tops::mdspan dst_mem(tops::Global, out, dst_shape);
      tops::broadcast(ctx, dst_mem, src_mem); 
    }
}

//dims_and_strides: dst shape, dst stride, dst layout, origin shape
template <typename T, int RANK>
__device__ void narrow_kernel(T* in, T* out, int src_rank, const size_t in_size, const size_t out_size, size_t* dims_and_strides, char* raw_cache, T* buffer) {
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx); 
    int dst_shape[RANK];
    int src_shape[RANK];
    int offsets[RANK];

    for (int j = 0; j < RANK; ++j) {
      dst_shape[j] = dims_and_strides[j];
      src_shape[j] = dims_and_strides[j + 3 * RANK];
      offsets[j] = 0;
    }

    bool cacheable_l1 = in_size * sizeof(T) < COPY_L1SIZE;
    bool cacheable_l2 = in_size * sizeof(T) < SHARE_BUFFER_SIZE;
    // PRINTHELPER(src_shape, RANK, "\nsrc_shape: ");
    // PRINTHELPER(dst_shape, RANK, "\ndst_shape: ");

    T* src_cached = reinterpret_cast<T*>(raw_cache);
    if (cacheable_l1){
      tops::memcpy(ctx, 
        tops::mdspan(tops::Private, buffer, in_size),
        tops::mdspan(tops::Global, in, in_size));
      tops::slice(ctx, tops::mdspan(tops::Global, out, dst_shape), tops::mdspan(tops::Private, buffer, src_shape), offsets);

    } else if (cacheable_l2) {
      tops::memcpy(ctx, 
        tops::mdspan(tops::Shared, src_cached, in_size),
        tops::mdspan(tops::Global, in, in_size));
      tops::slice(ctx, tops::mdspan(tops::Global, out, dst_shape), tops::mdspan(tops::Shared, src_cached, src_shape), offsets);
    } else {
      tops::mdspan src_mem(tops::Global, in, src_shape);
      tops::mdspan dst_mem(tops::Global, out, dst_shape);
      tops::slice(ctx, dst_mem, src_mem, offsets); 
    }
}

//dims_and_strides: dst shape, dst stride, dst layout, origin shape
#define UNARY_COPY_OP(KERNEL, TYPE, VT, FN_NAME, BPE) \
extern "C" __global__ void FN_NAME( \
    const size_t in_size, \
    const size_t out_size, \
    const size_t num_dims, \
    const size_t origin_num_dims, \
    size_t *dims_and_strides, \
    TYPE *in, \
    TYPE *out, \
    const size_t op_type) \
{ \
    extern __shared__ char raw_cache[]; \
    __local__ __valigned__ TYPE l1_cache[COPY_TILESIZE/BPE + COPY_L1SIZE/BPE]; \
    bool cont = true; \
    __local__ __valigned__ size_t info[128]; \
    tops_dte_ctx_t ctx; \
    tops::dte_scope s(ctx); \
    if (dims_and_strides) { \
      tops::mdspan srcInfo(tops::Global, dims_and_strides, num_dims * 3 + origin_num_dims); \
      tops::mdspan dstInfo(tops::Private, info, num_dims * 3 + origin_num_dims); \
      tops::memcpy(ctx, dstInfo, srcInfo); \
    } \
    if (op_type == 1 && in_size == out_size && num_dims < 5) { \
      if (GetThreadIdx() == 0) { \
        if (num_dims == 2)\
          transpose_kernel<TYPE, 2>(in, out, in_size, out_size, info, raw_cache, l1_cache); \
        else if (num_dims == 3)\
          transpose_kernel<TYPE, 3>(in, out, in_size, out_size, info, raw_cache, l1_cache); \
        else if (num_dims == 4)\
          transpose_kernel<TYPE, 4>(in, out, in_size, out_size, info, raw_cache, l1_cache); \
      }\
    } else if (op_type == 2 && in_size < out_size && origin_num_dims <= num_dims) { \
      if (GetThreadIdx() == 0) { \
        if (num_dims == 2)\
            broadcast_kernel<TYPE, 2>(in, out, origin_num_dims, in_size, out_size, info, raw_cache, l1_cache); \
        else if (num_dims == 3)\
            broadcast_kernel<TYPE, 3>(in, out, origin_num_dims, in_size, out_size, info, raw_cache, l1_cache); \
        else if (num_dims == 4)\
            broadcast_kernel<TYPE, 4>(in, out, origin_num_dims, in_size, out_size, info, raw_cache, l1_cache); \
        else if (num_dims == 5)\
            broadcast_kernel<TYPE, 5>(in, out, origin_num_dims, in_size, out_size, info, raw_cache, l1_cache); \
      }\
    } \
    else if (op_type == 4 && in_size > out_size) { \
      if (GetThreadIdx() == 0) { \
        if (num_dims == 2)\
            narrow_kernel<TYPE, 2>(in, out, origin_num_dims, in_size, out_size, info, raw_cache, l1_cache); \
        else if (num_dims == 3)\
            narrow_kernel<TYPE, 3>(in, out, origin_num_dims, in_size, out_size, info, raw_cache, l1_cache); \
        else if (num_dims == 4)\
            narrow_kernel<TYPE, 4>(in, out, origin_num_dims, in_size, out_size, info, raw_cache, l1_cache); \
        else if (num_dims == 5)\
            narrow_kernel<TYPE, 5>(in, out, origin_num_dims, in_size, out_size, info, raw_cache, l1_cache); \
      }\
    } \
    else { \
      if (num_dims == 2)\
        KERNEL<TYPE, 2, BPE>(in, out, in_size, out_size, info, raw_cache, l1_cache); \
      else if (num_dims == 3)\
        KERNEL<TYPE, 3, BPE>(in, out, in_size, out_size, info, raw_cache, l1_cache); \
      else if (num_dims == 4)\
        KERNEL<TYPE, 4, BPE>(in, out, in_size, out_size, info, raw_cache, l1_cache); \
      else if (num_dims == 5)\
        KERNEL<TYPE, 5, BPE>(in, out, in_size, out_size, info, raw_cache, l1_cache); \
      else if (num_dims <= 1)\
        KERNEL<TYPE, 1, BPE>(in, out, in_size, out_size, info, raw_cache, l1_cache); \
    } \
} \


UNARY_COPY_OP(ucopy_multithread_cache, __bf16, vbfloat, ucopy_bf16, 2) 
UNARY_COPY_OP(ucopy_multithread_cache, __fp16, vhalf, ucopy_f16, 2)
UNARY_COPY_OP(ucopy_multithread_cache, float, vfloat, ucopy_f32, 4)
UNARY_COPY_OP(ucopy_multithread_cache, u_int8_t, vuchar, ucopy_u8, 1)
UNARY_COPY_OP(ucopy_multithread_cache, char, vchar, ucopy_i8, 1)
UNARY_COPY_OP(ucopy_multithread_cache, u_int32_t, vuint, ucopy_u32, 4)
UNARY_COPY_OP(ucopy_multithread_cache, double, vfloatx2, ucopy_f64, 8) //remove this for gather copy

int main() {}