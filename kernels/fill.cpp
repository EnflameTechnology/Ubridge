/**
 * Copyright 2020-2021 Enflame. All Rights Reserved.
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
 * @par Original implementation: http://git.enflame.cn/sw/tops/-/tree/master/library/computing/topsop/lib/kernel/cc_kernel/fill_
 * @par Comments: a simple fill kernel bought from TopsOP.
 */


#include <stdio.h>
#include <tops.h>
#include <tops/half.h>
#include <tops/bfloat.h>
#include "tops/tops_runtime.h"
#include "utils.h"

#define tile_size 1024
using namespace std;

__device__ __forceinline__ void init_tid_off_arr(int dim_size, int *tid_off_arr,
                                 int &thread_num) {
  for (int i = 0; i < thread_num; i++) {
    tid_off_arr[i] = 0;
  }
  int mBlocks = (dim_size + tile_size - 1) / tile_size;
  thread_num = mBlocks > thread_num ? thread_num : mBlocks;
  int block_arr[24] = {0};
  for (int i = 0; i < thread_num; i++) {
    block_arr[i] = mBlocks / thread_num;
  }
  for (int i = 0; i < mBlocks % thread_num; i++) {
    block_arr[i] += 1;
  }
  tid_off_arr[0] = block_arr[0];
  for (int i = 1; i < thread_num; i++) {
    tid_off_arr[i] += tid_off_arr[i - 1] + block_arr[i];
  }
}

template <typename T>
__device__ __forceinline__ void fill__Kernel(T *output, int total_size, T value, int bpe) {
  int thread_num = GetThreadNum();
  int thread_id = GetThreadIdx();
  tops_dte_ctx_t ctx;
  tops::dte_scope s(ctx);

  int nBlocks = (total_size + tile_size - 1) / tile_size;
  int tid_off_arr[24] = {0};
  init_tid_off_arr(total_size, tid_off_arr, thread_num);
  int off_start = thread_id == 0 ? 0 : tid_off_arr[thread_id - 1];
  int off_end = tid_off_arr[thread_id];
  // printf("thread_id:%d off_start:%d off_end:%d\n", thread_id, off_start,
  // off_end);

  for (int i = off_start; i < off_end; i++) {
    int dsize = i != tid_off_arr[thread_num - 2] ? tile_size
                                                 : total_size - i * tile_size;
    // printf("thread_id:%d off_start: %d dsize:%d\n", thread_id, i * tile_size,
    // dsize);
    tops::mdspan output_hbm(tops::Global, output + i * tile_size, dsize);
    tops::memset(ctx, output_hbm, value);
  }
}  // end of topscc kernel function


extern "C"  __global__ void fill_f32(float *output, int total_size,
                                      float value, int bpe){

    return fill__Kernel<float>(output, total_size, value, bpe);
}

extern "C"  __global__ void fill_f16(tops::half *output, int total_size,
                                      tops::half value, int bpe){

    return fill__Kernel<tops::half>(output, total_size, value, bpe);
}

extern "C"  __global__ void fill_bf16(tops::bfloat *output, int total_size,
                                      tops::bfloat value, int bpe){

    return fill__Kernel<tops::bfloat>(output, total_size, value, bpe);
}

extern "C"  __global__ void fill_f64(double *output, int total_size,
                                      double value, int bpe){

    return fill__Kernel<double>(output, total_size, value, bpe);
}

extern "C"  __global__ void fill_i32(int32_t *output, int total_size,
                                      int32_t value, int bpe){

    return fill__Kernel<int32_t>(output, total_size, value, bpe);
}

extern "C"  __global__ void fill_i16(int16_t *output, int total_size,
                                      int16_t value, int bpe){

    return fill__Kernel<int16_t>(output, total_size, value, bpe);
}

extern "C"  __global__ void fill_bool(bool *output, int total_size, bool value,
                                      int bpe){

    return fill__Kernel<bool>(output, total_size, value, bpe);
}

extern "C"  __global__ void fill_i8(int8_t *output, int total_size,
                                      int8_t value, int bpe){

    return fill__Kernel<int8_t>(output, total_size, value, bpe);
}

// __global__ void fill_u32(uint32_t *output, int total_size,
//                                       uint32_t value, int bpe){

//     return fill__Kernel<uint32_t>(output, total_size, value, bpe);
// }

// __global__ void fill_u8(uint8_t *output, int total_size,
//                                       uint8_t value, int bpe){

//     return fill__Kernel<uint8_t>(output, total_size, value, bpe);
// }

// __global__ void fill_u16(uint16_t *output, int total_size,
//                                       uint16_t value, int bpe){

//     return fill__Kernel<uint16_t>(output, total_size, value, bpe);
// }

int main() {
    return 0;
}