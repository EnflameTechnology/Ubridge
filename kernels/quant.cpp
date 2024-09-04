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
 * @file    cast.cpp
 * @brief
 *
 * @author  Guoqing Bao
 * @date    2024-08-28 - 
 * @version V0.1
 * @par     Copyright (c) Enflame Tech Company.
 * @par     History: support ggml/gguf dequantization (Q8_0 supported)
 * @par     Comments: quant kernel for ggml/gguf format
 */

#include <stdio.h>
#include <type_traits>
#include <tops.h>
#include <krt/builtins.h>
#include <cstdint>

#include <tops/topsrtc.h>
#include <tops/half.h>
#include <tops/bfloat.h>
#include <tops/topsrtc.h>
// #include <binary.h>
#include <acore_op.h>

#include "tcle.h"
#include <tops/tops_runtime.h>

#include "utils/utils.h"
using namespace std;
using namespace tops;
#ifdef GGML_QKK_64
#define QK_K 64
#define K_SCALE_SIZE 4
#else
#define QK_K 256
#define K_SCALE_SIZE 12
#endif
#define GCU_DEQUANTIZE_BLOCK_SIZE 256
#define TILE_SIZE 1024

typedef uint16_t ggml_fp16_t;
typedef __fp16 half1;

typedef struct {
    __fp16 x; // x
    __fp16 y;  // y
} half2;

#define QK4_0 32
#define QR4_0 2
#define QI4_0 (QK4_0 / (4 * QR4_0))
typedef struct {
    half1    d;              // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;

#define QK4_1 32
#define QR4_1 2
#define QI4_1 (QK4_1 / (4 * QR4_1))
typedef struct {
    half2   dm;             // dm.x = delta, dm.y = min
    uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;

#define QK5_0 32
#define QR5_0 2
#define QI5_0 (QK5_0 / (4 * QR5_0))
typedef struct {
    half1 d;                 // delta
    uint8_t qh[4];          // 5-th bit of quants
    uint8_t qs[QK5_0 / 2];  // nibbles / quants
} block_q5_0;

#define QK5_1 32
#define QR5_1 2
#define QI5_1 (QK5_1 / (4 * QR5_1))
typedef struct {
    half2 dm;               // dm.x = delta, dm.y = min
    uint8_t qh[4];          // 5-th bit of quants
    uint8_t qs[QK5_1 / 2];  // nibbles / quants
} block_q5_1;

#define QK8_0 32
#define QR8_0 1
#define QI8_0 (QK8_0 / (4 * QR8_0))
typedef struct {
    half1    d;              // delta
    int8_t  qs[QK8_0];      // quants
} block_q8_0;

#define QK8_1 32
#define QR8_1 1
#define QI8_1 (QK8_1 / (4 * QR8_1))
typedef struct {
    half2   ds;             // ds.x = delta, ds.y = sum
    int8_t  qs[QK8_0];      // quants
} block_q8_1;


#define QR2_K 4
#define QI2_K (QK_K / (4*QR2_K))
typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    half2 dm;                // super-block scale for quantized scales/mins
} block_q2_K;

#define QR3_K 4
#define QI3_K (QK_K / (4*QR3_K))
typedef struct {
    uint8_t hmask[QK_K/8];     // quants - high bit
    uint8_t qs[QK_K/4];        // quants - low 2 bits
#ifdef GGML_QKK_64
    uint8_t scales[2]; // scales, quantized with 8 bits
#else
    uint8_t scales[K_SCALE_SIZE]; // scales, quantized with 6 bits
#endif
    half d;             // super-block scale
} block_q3_K;

#define QR4_K 2
#define QI4_K (QK_K / (4*QR4_K))
#ifdef GGML_QKK_64
typedef struct {
    half1    dm[2];             // super-block scales/mins
    uint8_t scales[2];         // 4-bit block scales/mins
    uint8_t qs[QK_K/2];        // 4--bit quants
} block_q4_K;
#else
typedef struct {
    half2 dm;                  // super-block scale for quantized scales/mins
    uint8_t scales[3*QK_K/64]; // scales, quantized with 6 bits
    uint8_t qs[QK_K/2];        // 4--bit quants
} block_q4_K;
#endif

#define QR5_K 2
#define QI5_K (QK_K / (4*QR5_K))
#ifdef GGML_QKK_64
typedef struct {
    half1 d;                  // super-block scale
    int8_t scales[QK_K/16];  // block scales
    uint8_t qh[QK_K/8];      // quants, high bit
    uint8_t qs[QK_K/2];      // quants, low 4 bits
} block_q5_K;
#else
typedef struct {
    half2 dm;                     // super-block scale for quantized scales/mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];           // quants, high bit
    uint8_t qs[QK_K/2];           // quants, low 4 bits
} block_q5_K;
#endif

#define QR6_K 2
#define QI6_K (QK_K / (4*QR6_K))
typedef struct {
    uint8_t ql[QK_K/2];   // quants, lower 4 bits
    uint8_t qh[QK_K/4];   // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales
    half1    d;         // delta
} block_q6_K;

// In llama.cpp this is only used for intermediate quantization and dot products
typedef struct {
    float   d;              // delta
    int8_t  qs[QK_K];       // quants
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16
} block_q8_K;


template <typename T, int BLOCK_DIM>
__device__ __forceinline__ void dequantize_q8_0(int8_t* input, T* buffer_scales, int blocks, T* buffer_out) {

  static const int vlength = TOPS_VECTOR_LENGTH * sizeof(float) / sizeof(T);

  using in_vtype = typename scalar_to_vector<int8_t, vlength>::type;
  using out_vtype = typename scalar_to_vector<T, vlength>::type;
  auto src_leaptr = tops::simple_leaptr<in_vtype>(input);
  auto dst_leaptr = tops::simple_leaptr<out_vtype>(buffer_out);
  auto scale_leaptr = tops::simple_leaptr<out_vtype>(buffer_scales);

  int group_num = blocks / (vlength / BLOCK_DIM);
  out_vtype vsrc, vres;
#pragma clang loop unroll_count(16)
  for (int i=0; i< group_num; i++) {
    auto scale = scale_leaptr.load();
    auto src = src_leaptr.load();
    vsrc = vcast<out_vtype>(src);
    vres = vmul(vsrc, scale);
    dst_leaptr.store(vres);
  }
}


template <typename T, int BLOCK_DIM>
__device__ void dequant_kernel(int8_t* in, T* out, const size_t elem_count) {
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx); 
    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNum();
    int num_blocks = (elem_count + (BLOCK_DIM - 1)) / BLOCK_DIM;
    __fp16* scale_ptr = reinterpret_cast<__fp16*>(in);
    int8_t* data_ptr = reinterpret_cast<int8_t*>(in) + num_blocks * sizeof(__fp16);

    __local__ __valigned__ int8_t buffer_in[TILE_SIZE * BLOCK_DIM];
    // __local__ __valigned__ float buffer_out[TILE_SIZE * BLOCK_DIM];
    __local__ __valigned__ T buffer_out_dequant[TILE_SIZE * BLOCK_DIM];
    __local__ __valigned__ __fp16 buffer_scales[TILE_SIZE];
    __local__ __valigned__ T buffer_scales1[TILE_SIZE];
    __local__ __valigned__ T buffer_scales2[TILE_SIZE * BLOCK_DIM];

    tops::mdspan buffer_l1(tops::Private, buffer_in, TILE_SIZE * BLOCK_DIM);
    tops::mdspan buffer_l1_scales(tops::Private, buffer_scales, TILE_SIZE);

    tops::mdspan out_hbm(tops::Global, out, elem_count);
    int N = num_blocks;
    int THREAD_STEP = 1;
    int thread_step = 1;
    if (N > MAX_THREADS) {
      THREAD_STEP = N / MAX_THREADS;
      thread_step = THREAD_STEP;
      if (N % MAX_THREADS != 0) {
        if (thread_id == MAX_THREADS - 1) {
          thread_step += N % MAX_THREADS; //last thread also process remains
        }
      }
    }

    for (int i = 0; i < thread_step; i+=TILE_SIZE) {
      int blocks = (i + TILE_SIZE < thread_step) ? TILE_SIZE : thread_step - i;
      int offset = thread_id * THREAD_STEP + i;
      int src_shape[2] = {blocks, 1};
      int dst_shape[2] = {blocks, BLOCK_DIM};

      tops::mdspan hbm_in(tops::Global, data_ptr + offset * BLOCK_DIM, blocks * BLOCK_DIM);
      tops::memcpy(ctx, buffer_l1, hbm_in);

      tops::mdspan hbm_in_scales(tops::Global, scale_ptr + offset, blocks);
      tops::memcpy(ctx, buffer_l1_scales, hbm_in_scales);

      convert<T, __fp16>(buffer_scales1, buffer_scales, TILE_SIZE);
      tops::broadcast(ctx, tops::mdspan(tops::Private, buffer_scales2, dst_shape), tops::mdspan(tops::Private, buffer_scales1, src_shape));
      dequantize_q8_0<T, BLOCK_DIM>(buffer_in, buffer_scales2, blocks, buffer_out_dequant);
      tops::memcpy(ctx, tops::mdspan(tops::Global, out + offset * BLOCK_DIM, blocks * BLOCK_DIM), tops::mdspan(tops::Private, buffer_out_dequant, blocks * BLOCK_DIM));
    }
}

#define DEQUANT_OP(T, BLOCK_DIM, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    int8_t *inp, \
    T *out,\
    const size_t numel) \
{ \
    dequant_kernel<T, BLOCK_DIM>(inp, out, numel); \
} \

DEQUANT_OP(__fp16, QK8_0, dequantize_block_q8_0_f16)
DEQUANT_OP(__bf16, QK8_0, dequantize_block_q8_0_bf16)
DEQUANT_OP(float, QK8_0, dequantize_block_q8_0_f32)

template <int BLOCK_DIM>
__device__ __forceinline__ void quantize_q8_0(float* input, int blocks, __fp16* buffer_scales, int8_t* out) {
    tops_dte_ctx_t ctx;
    const int BATCH = TOPS_VECTOR_LENGTH / BLOCK_DIM;
    va16f32 vinputs[BATCH];
    va16f32 vresults[BATCH];
    for (int i=0; i<blocks / BATCH; i++) {
        for (int j=0; j< BATCH; j++) {
            float* pIn = input + (i * BATCH + j) * BLOCK_DIM;
            vinputs[j] = vload<va16f32>(pIn);
            float amax = vreduce_max<float>(vabs(vinputs[j]));
            float d = (float)amax / ((1 << 7) - 1);
            float id = (d != 0.0 ? 1. / d :  0.);
            buffer_scales[i * BATCH + j] = (__fp16)d;
            vresults[j] = vmul(vinputs[j], vbroadcast<va16f32>(id));
            vstore(vresults[j], pIn);
        }
        //batch round & cast
        float* pIn = input + i * BATCH * BLOCK_DIM;
        v64i8 vout = vcast<v64i8>(tcle::round(vload<va16f32x4>(pIn)));
        int8_t* pOut = out + i * BATCH * BLOCK_DIM;
        vstore(vout, pOut);
    }
}

//quantized to data format of [scales...data]
template <typename T, int BLOCK_DIM>
__device__ void quant_kernel(T* in, int8_t* out, const size_t input_bytes, const size_t output_bytes) {
    int element_count = input_bytes / sizeof(T);
    int num_blocks = (element_count + (BLOCK_DIM - 1)) / BLOCK_DIM;
    __fp16* scale_ptr = reinterpret_cast<__fp16*>(out);
    int8_t* data_ptr = out + num_blocks * sizeof(__fp16);

    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx); 
    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNum();
    
    __local__ __valigned__ T buffer_in[TILE_SIZE * BLOCK_DIM];
    __local__ __valigned__ float buffer_in_f32[TILE_SIZE * BLOCK_DIM];

    __local__ __valigned__ int8_t buffer_out[TILE_SIZE * BLOCK_DIM];
    __local__ __valigned__ __fp16 buffer_scales[TILE_SIZE];

    tops::mdspan buffer_l1(tops::Private, buffer_in, TILE_SIZE * BLOCK_DIM);

    int N = num_blocks;
    int THREAD_STEP = 1;
    int thread_step = 1;
    if (N > MAX_THREADS) {
      THREAD_STEP = N / MAX_THREADS;
      thread_step = THREAD_STEP;
      if (N % MAX_THREADS != 0) {
        if (thread_id == MAX_THREADS - 1) {
          thread_step += N % MAX_THREADS; //last thread also process remains
        }
      }
    }

    for (int i = 0; i < thread_step; i+=TILE_SIZE) {
      int blocks = (i + TILE_SIZE < thread_step) ? TILE_SIZE : thread_step - i;
      int offset = thread_id * THREAD_STEP + i;
      tops::mdspan hbm_in(tops::Global, in + offset * BLOCK_DIM, blocks * BLOCK_DIM);
      tops::memcpy(ctx, buffer_l1, hbm_in);
      convert(buffer_in_f32, buffer_in, blocks * BLOCK_DIM);
      quantize_q8_0<BLOCK_DIM>(buffer_in_f32, blocks, buffer_scales, buffer_out);
      tops::memcpy(ctx, tops::mdspan(tops::Global, scale_ptr + offset, blocks), tops::mdspan(tops::Private, buffer_scales, blocks));
      tops::memcpy(ctx, tops::mdspan(tops::Global, data_ptr + offset * BLOCK_DIM, blocks * BLOCK_DIM), tops::mdspan(tops::Private, buffer_out, blocks * BLOCK_DIM));
    }
}

#define QUANT_OP(TYPE, BLOCK_DIM, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    TYPE *inp, \
    int8_t *out,\
    const size_t input_bytes, const size_t output_bytes) \
{ \
    quant_kernel<TYPE, BLOCK_DIM>(inp, out, input_bytes, output_bytes); \
} \


QUANT_OP(__bf16, QK8_0, quantize_block_q8_0_bf16)
QUANT_OP(__fp16, QK8_0, quantize_block_q8_0_f16)
QUANT_OP(float, QK8_0, quantize_block_q8_0_f32)



