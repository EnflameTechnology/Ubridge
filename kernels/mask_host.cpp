
/**
 * Copyright 2020-2025 Enflame. All Rights Reserved.
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
#include <tops.h>
#include <tops/half.h>
#include <tops/bfloat.h>
#include <tops/tops_runtime.h>
#include <tops/topscc_types.h>
#include "utils/utils.h"
#include "tcle.h"
#define MAX_BATCH 1024
#define MAX_TOP_K 32

using IVType =
      typename tcle::altivector<u_int32_t, TCLE_MAX_VECTOR_LENGTH>::VT;
using FVType =
    typename tcle::altivector<float, TCLE_MAX_VECTOR_LENGTH>::VT;

#if 0
template <typename TYPE>
__device__ int atomic_mask(TYPE* src_ptr, float mask_v,
                               u_int32_t* base_index,
                               u_int32_t* index_ptr, 
                               int count_start,
                               int offset, int buf_size) {

    int count = 0;
    generic_ptr src_addr = reinterpret_cast<generic_ptr>(src_ptr);
    using in_vtype = typename tcle::altivector<TYPE, TCLE_MAX_VECTOR_LENGTH>::VT;

    auto src_leaptr = tcle::simple_leaptr<in_vtype>(src_addr);
    using mask_type = typename tcle::altivector_to_mask<IVType>::type;

    FVType velement = (FVType)(mask_v);
    IVType vnon_flag = (IVType)(-1);
    IVType vbase_index =
    *((__TCLE_AS__ IVType *)reinterpret_cast<IVType*>(base_index));
    __vector4 int vbase_128 = (__vector4 int)(TCLE_MAX_VECTOR_LENGTH);

    __vector4 int voffset = (__vector4 int)(offset);
    vbase_index = (vbase_index + voffset);

    FVType vsrc = tcle::cvt<FVType>(src_leaptr.load());
    mask_type vmask = velement != vsrc;
    IVType v_index = tcle::vsel(vmask, vbase_index, vnon_flag);
    for (int j = 0; j < TCLE_MAX_VECTOR_LENGTH; j++) {
        if (j >= buf_size) break;
        if (v_index[j] != -1) {
            index_ptr[count_start + count] = v_index[j];
            count++;
        }
    }
    return count;
}
#endif

__device__ int count_thread(u_int32_t thread_counts[], int max_thread) {
    int total = 0;
    for (int i=0; i<max_thread; i++) {
        total += thread_counts[i];
    }
    return total;
}

__device__ int offset_thread(u_int32_t thread_counts[], int thread_id, int max_thread) {
    int offset = 0;
    for (int i=0; i<max_thread; i++) {
        if (i == thread_id) break;
        offset += thread_counts[i];
    }
    return offset;
}

template <typename T>
__global__ __cooperative__ void mask_kernel(T* in, T mask_v, u_int32_t* out, 
        u_int32_t* workspace, u_int32_t* out_count, int batch, int dim_size) {
    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNum();
    int THREAD_STEP = 1;
    int thread_step = 1;
    __local__ __valigned__ T in_buffer[MAX_BATCH * MAX_TOP_K];
    tops_dte_ctx_t ctx_in;
    tops_dte_ctx_t ctx_count;
    tops_dte_ctx_t ctxs_index;
    ctx_in.init();
    ctx_count.init();
    ctxs_index.init();
    tops::mdspan l1_input(tops::Private, in_buffer, batch * dim_size);
    //input cache
    tops::memcpy(ctx_in, l1_input, tops::mdspan(tops::Global, in, batch * dim_size));
    GetThreadStep(batch * dim_size, thread_step, THREAD_STEP);
    __local__ __valigned__ u_int32_t row_ptr[TCLE_MAX_VECTOR_LENGTH * MAX_BATCH / MAX_TOP_K];
    __local__ __valigned__ u_int32_t col_ptr[TCLE_MAX_VECTOR_LENGTH * MAX_BATCH / MAX_TOP_K];
    __local__ __valigned__ u_int32_t thread_counts[128];
    __local__ __valigned__ char count_buffer[TCLE_MAX_VECTOR_LENGTH * 4];
    u_int32_t* counts = reinterpret_cast<u_int32_t*>(count_buffer);
    tops::mdspan hbm_workspace(tops::Global, workspace, MAX_THREADS);

    //per-thread count
    u_int32_t ct = 0;
    for (int i = 0; i < thread_step; i++) {
      u_int32_t index = thread_id * THREAD_STEP + i;
      if (index < batch * dim_size) {
        //index to indices
        if (in_buffer[index] == mask_v) {
            u_int32_t row = index / dim_size;
            u_int32_t col = index % dim_size;
            row_ptr[ct] = row;
            col_ptr[ct] = col;
            ct += 1;
        }
      }
    }
    counts[0] = ct;
    //record per-thread count
    tops::deslice(ctxs_index, hbm_workspace, tops::mdspan(tops::Private, counts, 1), {thread_id});

    __syncblocks();

    //calculate total count and current offset
    tops::memcpy(ctx_count, tops::mdspan(tops::Private, thread_counts, MAX_THREADS), hbm_workspace);
    int total_count = count_thread(thread_counts, MAX_THREADS);

    //write indices based on the thread offset
    if (ct > 0) {
        int global_offset = offset_thread(thread_counts, thread_id, MAX_THREADS);
        for (int rank=0; rank<2; rank++) {
            u_int32_t* cur_out = out + rank * total_count;
            tops::memcpy(ctx_count, tops::mdspan(tops::Global, cur_out + global_offset, ct), 
                tops::mdspan(tops::Private, rank == 0 ? row_ptr : col_ptr, ct));
        }
    }

    //write total count
    if (thread_id == 0) {
        counts[0] = total_count;
        tops::deslice(ctxs_index, tops::mdspan(tops::Global, out_count, 1), tops::mdspan(tops::Private, counts, 1), {0});
    }
}

#define MASK_OP(T, RUST_NAME) \
extern "C" u_int32_t mask_##RUST_NAME(  \
    T * in, T mask_v, u_int32_t* out, int batch, int dim_size, void* stream_ \
) { \
    topsStream_t stream = (topsStream_t)(stream_);\
    int numBlocks = 2;\
    int dimBlocks = 12;\
    u_int32_t *mask_count_dev, *workspace;\
    u_int32_t mask_count[512] = {0};\
    size_t workspace_size = numBlocks * dimBlocks * sizeof(u_int32_t);\
    workspace_size = ALIGN_UP(workspace_size, 512);\
    topsMallocAsync(&mask_count_dev, workspace_size, stream, topsDeviceMallocDefault);\
    topsMallocAsync(&workspace, workspace_size, stream, topsDeviceMallocDefault);\
    mask_kernel<T><<<2, 12, 0, stream>>>(in, mask_v, out, workspace, mask_count_dev, batch, dim_size);\
    topsFreeAsync(workspace, stream);\
    topsMemcpyAsync(&mask_count, mask_count_dev, 4, topsMemcpyDeviceToHost, stream);\
    topsFreeAsync(mask_count_dev, stream);\
    topsStreamSynchronize(stream);\
    return mask_count[0];\
}

MASK_OP(float, f32)
// MASK_OP(__fp16, f16)
// MASK_OP(__bf16, bf16)
MASK_OP(int32_t, i32)
// MASK_OP(int8_t, i8)
MASK_OP(uint32_t, u32)
// MASK_OP(uint8_t, u8)