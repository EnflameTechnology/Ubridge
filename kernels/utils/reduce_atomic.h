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
 * @file    affine.cpp
 * @brief
 *
 * @author  Guoqing Bao
 * @date    2023-11-23 - 2024-02-04
 * @version V0.1
 * @par     Copyright (c) Enflame Tech Company.
 * @par     History: Fix compilation problem in reduce atomic op
 * @par     Comments: naive implementation for atomic reduce 
 */
#include <stdio.h>
#include <tops.h>
#include <tops/half.h>
#include <tops/bfloat.h>
#include <algorithm>
#include <vector>
#include "tops/tops_runtime.h"
#include "utils.h"
using namespace std;

template <typename T>
__device__ __forceinline__ T call_reduce_max(T *src_ptr,
                                              int size) {
  generic_ptr src_addr = reinterpret_cast<generic_ptr>(src_ptr);
  constexpr int bpe = sizeof(T);
  constexpr int vec_elems = sizeof(typename tops::unified_scalar<T>::type) * TOPS_VECTOR_LENGTH / bpe;
  using vtype = typename tops::scalar_to_vector<T, vec_elems>::type;

  auto src_leaptr = tops::simple_leaptr<vtype>(src_addr);
  int group_num = (size + vec_elems - 1) / vec_elems;
  vtype vsrc;
  T max_value = src_ptr[0];
  for (int i = 0; i < group_num; i++) {
    vsrc = src_leaptr.load();
    T cur_max = tops::vreduce_max<T, vtype>(vsrc);
    if (cur_max > max_value) {
      max_value = cur_max;
    }
  }
  return max_value;
}

template <typename TYPE>
__forceinline__ __device__ void atomic_reduce_max(TYPE* dst_ptr, TYPE* src_ptr,
                                            unsigned int channel_align) {}

template <>
__forceinline__ __device__ void  atomic_reduce_max(float* dst_ptr, float* src_ptr,
                                            unsigned int channel_align) {
  unsigned int num_remains =  channel_align % TOPS_VECTOR_LENGTH;
  unsigned int aligned_data_length =  channel_align - num_remains;                                 
  float atomic_max_value = channel_align < TOPS_VECTOR_LENGTH ? src_ptr[0] : call_reduce_max<float>(src_ptr, aligned_data_length);
  for (int i = 0; i< num_remains; i++) {//for unaligned remaining data
      if (src_ptr[aligned_data_length + i] > atomic_max_value) {
        atomic_max_value = src_ptr[aligned_data_length + i];
      }
  }
  dst_ptr[0] = atomic_max_value;
}

template <>
__forceinline__ __device__ void  atomic_reduce_max(__fp16* dst_ptr, __fp16* src_ptr,
                                            unsigned int channel_align) {
  unsigned int num_remains =  channel_align % TOPS_VECTOR_LENGTH;
  unsigned int aligned_data_length =  channel_align - num_remains;                                 
  __fp16 atomic_max_value = channel_align < TOPS_VECTOR_LENGTH ? src_ptr[0] : call_reduce_max<__fp16>(src_ptr, aligned_data_length);
  for (int i = 0; i< num_remains; i++) {//for unaligned remaining data
      if (src_ptr[aligned_data_length + i] > atomic_max_value) {
        atomic_max_value = src_ptr[aligned_data_length + i];
      }
  }
  dst_ptr[0] = atomic_max_value;
}

template <>
__forceinline__ __device__ void  atomic_reduce_max(__bf16* dst_ptr, __bf16* src_ptr,
                                            unsigned int channel_align) {
  unsigned int num_remains =  channel_align % TOPS_VECTOR_LENGTH;
  unsigned int aligned_data_length =  channel_align - num_remains;                                 
  __bf16 atomic_max_value = channel_align < TOPS_VECTOR_LENGTH ? src_ptr[0] : call_reduce_max<__bf16>(src_ptr, aligned_data_length);
  for (int i = 0; i< num_remains; i++) {//for unaligned remaining data
      if (src_ptr[aligned_data_length + i] > atomic_max_value) {
        atomic_max_value = src_ptr[aligned_data_length + i];
      }
  }
  dst_ptr[0] = atomic_max_value;
}

template <>
__forceinline__ __device__ void  atomic_reduce_max(int8_t* dst_ptr, int8_t* src_ptr,
                                            unsigned int channel_align) {
  dst_ptr[0] = src_ptr[0];
  for (int i=1; i< channel_align; i++) {
    if (src_ptr[i] > dst_ptr[0])
    {
      dst_ptr[0] = src_ptr[i];
    }
  }
}


template <typename T>
__device__ __forceinline__ T call_reduce_min(T *src_ptr,
                                              int size) {
  generic_ptr src_addr = reinterpret_cast<generic_ptr>(src_ptr);
  constexpr int bpe = sizeof(T);
  constexpr int vec_elems = sizeof(typename tops::unified_scalar<T>::type) * TOPS_VECTOR_LENGTH / bpe;
  using vtype = typename tops::scalar_to_vector<T, vec_elems>::type;

  auto src_leaptr = tops::simple_leaptr<vtype>(src_addr);
  int group_num = (size + vec_elems - 1) / vec_elems;
  vtype vsrc;
  T min_value = src_ptr[0];
  for (int i = 0; i < group_num; i++) {
    vsrc = src_leaptr.load();
    T cur_min = tops::vreduce_min<T, vtype>(vsrc);
    if (cur_min < min_value) {
      min_value = cur_min;
    }
  }
  return min_value;
}

template <typename TYPE>
__forceinline__ __device__ void atomic_reduce_min(TYPE* dst_ptr, TYPE* src_ptr,
                                            unsigned int channel_align) {}

template <>
__forceinline__ __device__ void  atomic_reduce_min(float* dst_ptr, float* src_ptr,
                                            unsigned int channel_align) {
  unsigned int num_remains =  channel_align % TOPS_VECTOR_LENGTH;
  unsigned int aligned_data_length =  channel_align - num_remains;                                 
  float atomic_min_value = channel_align < TOPS_VECTOR_LENGTH ? src_ptr[0] : call_reduce_min<float>(src_ptr, aligned_data_length);
  for (int i = 0; i < num_remains; i++) {//for unaligned remaining data
      if (src_ptr[aligned_data_length + i] < atomic_min_value) {
        atomic_min_value = src_ptr[aligned_data_length + i];
      }
  }
  dst_ptr[0] = atomic_min_value;
}

template <>
__forceinline__ __device__ void  atomic_reduce_min(__fp16* dst_ptr, __fp16* src_ptr,
                                            unsigned int channel_align) {
  unsigned int num_remains =  channel_align % TOPS_VECTOR_LENGTH;
  unsigned int aligned_data_length =  channel_align - num_remains;                                 
  __fp16 atomic_min_value = channel_align < TOPS_VECTOR_LENGTH ? src_ptr[0] : call_reduce_min<__fp16>(src_ptr, aligned_data_length);
  for (int i = 0; i < num_remains; i++) {//for unaligned remaining data
      if (src_ptr[aligned_data_length + i] < atomic_min_value) {
        atomic_min_value = src_ptr[aligned_data_length + i];
      }
  }
  dst_ptr[0] = atomic_min_value;
}

template <>
__forceinline__ __device__ void  atomic_reduce_min(__bf16* dst_ptr, __bf16* src_ptr,
                                            unsigned int channel_align) {
  unsigned int num_remains =  channel_align % TOPS_VECTOR_LENGTH;
  unsigned int aligned_data_length =  channel_align - num_remains;                                 
  __bf16 atomic_min_value = channel_align < TOPS_VECTOR_LENGTH ? src_ptr[0] : call_reduce_min<__bf16>(src_ptr, aligned_data_length);
  for (int i = 0; i < num_remains; i++) {//for unaligned remaining data
      if (src_ptr[aligned_data_length + i] < atomic_min_value) {
        atomic_min_value = src_ptr[aligned_data_length + i];
      }
  }
  dst_ptr[0] = atomic_min_value;
}

template <>
__forceinline__ __device__ void  atomic_reduce_min(int8_t* dst_ptr, int8_t* src_ptr,
                                            unsigned int channel_align) {
  dst_ptr[0] = src_ptr[0];
  for (int i=1; i< channel_align; i++) {
    if (src_ptr[i] < dst_ptr[0])
    {
      dst_ptr[0] = src_ptr[i];
    }
  }
}

template <typename T>
__device__ __forceinline__ T call_reduce_sum(T *src_ptr,
                                              int size) {
  generic_ptr src_addr = reinterpret_cast<generic_ptr>(src_ptr);
  constexpr int bpe = sizeof(T);
  constexpr int vec_elems = sizeof(typename tops::unified_scalar<T>::type) * TOPS_VECTOR_LENGTH / bpe;
  using vtype = typename tops::scalar_to_vector<T, vec_elems>::type;

  auto src_leaptr = tops::simple_leaptr<vtype>(src_addr);
  int group_num = (size + vec_elems - 1) / vec_elems;
  vtype vsrc;
  vtype vdst = tops::vzero<vtype>();
  for (int i = 0; i < group_num; i++) {
    vsrc = src_leaptr.load();
    vdst = tops::vadd(vdst, vsrc);
  }
  return tops::vreduce_sum<T, vtype>(vdst);
}

template <typename TYPE>
__forceinline__ __device__ void atomic_reduce_sum(TYPE* dst_ptr, TYPE* src_ptr,
                                            unsigned int channel_align) {}


template <>
__forceinline__ __device__ void  atomic_reduce_sum(float* dst_ptr, float* src_ptr,
                                            unsigned int channel_align) {
  unsigned int num_remains =  channel_align % TOPS_VECTOR_LENGTH;
  unsigned int aligned_data_length =  channel_align - num_remains;                                 
  float atomic_sum_value = channel_align < TOPS_VECTOR_LENGTH ? 0 : call_reduce_sum<float>(src_ptr, aligned_data_length);
  for (int i=0; i< num_remains; i++) {//for unaligned remaining data
      atomic_sum_value += src_ptr[aligned_data_length + i];
  }
  dst_ptr[0] = atomic_sum_value;
}

template <>
__forceinline__ __device__ void  atomic_reduce_sum(__fp16* dst_ptr, __fp16* src_ptr,
                                            unsigned int channel_align) {
  unsigned int num_remains =  channel_align % TOPS_VECTOR_LENGTH;
  unsigned int aligned_data_length =  channel_align - num_remains;                                 
  __fp16 atomic_sum_value = channel_align < TOPS_VECTOR_LENGTH ? 0 : call_reduce_sum<__fp16>(src_ptr, aligned_data_length);
  for (int i=0; i< num_remains; i++) {//for unaligned remaining data
      atomic_sum_value += src_ptr[aligned_data_length + i];
  }
  dst_ptr[0] = atomic_sum_value;
}

template <>
__forceinline__ __device__ void  atomic_reduce_sum(__bf16* dst_ptr, __bf16* src_ptr,
                                            unsigned int channel_align) {
  unsigned int num_remains =  channel_align % TOPS_VECTOR_LENGTH;
  unsigned int aligned_data_length =  channel_align - num_remains;                                 
  __bf16 atomic_sum_value = channel_align < TOPS_VECTOR_LENGTH ? 0 : call_reduce_sum<__bf16>(src_ptr, aligned_data_length);
  for (int i=0; i< num_remains; i++) {//for unaligned remaining data
      atomic_sum_value += src_ptr[aligned_data_length + i];
  }
  dst_ptr[0] = atomic_sum_value;
}

template <>
__forceinline__ __device__ void  atomic_reduce_sum(int8_t* dst_ptr, int8_t* src_ptr,
                                            unsigned int channel_align) {
  dst_ptr[0] = src_ptr[0];
  for (int i=1; i< channel_align; i++) {
      dst_ptr[0] += src_ptr[i];
  }
}

#define MAX_REDUCE_ARG_BATCH 4096 //4096 * 128

template <typename T>
__device__ __forceinline__ int call_reduce_argmax(T *src_ptr,
                                              int size) {
  generic_ptr src_addr = reinterpret_cast<generic_ptr>(src_ptr);
  constexpr int bpe = sizeof(T);
  constexpr int vec_elems = sizeof(typename tops::unified_scalar<T>::type) * TOPS_VECTOR_LENGTH / bpe;
  int positions[MAX_REDUCE_ARG_BATCH];
  using vtype = typename tops::scalar_to_vector<T, vec_elems>::type;
  auto src_leaptr = tops::simple_leaptr<vtype>(src_addr);
  int group_num = (size + vec_elems - 1) / vec_elems;
  vtype vsrc;
  for (int i = 0; i < group_num; i++) {
    vsrc = src_leaptr.load();
    int max_pos = tops::vreduce_argmax<vtype>(vsrc);
    positions[i] = max_pos;
  }
  int max_value_pos = positions[0];
  for (int i = 1; i < group_num; i++) {
    if (src_ptr[i * vec_elems + positions[i]] > src_ptr[max_value_pos]) {
      max_value_pos = i * vec_elems + positions[i];
    }  
  }
  return max_value_pos;
}

template <typename TYPE>
__forceinline__ __device__ void atomic_reduce_argmax(u_int32_t* dst_ptr, TYPE* src_ptr,
                                            unsigned int channel_align) {}

template <>
__forceinline__ __device__ void  atomic_reduce_argmax(u_int32_t* dst_ptr, float* src_ptr,
                                            unsigned int channel_align) {
  unsigned int num_remains =  channel_align % TOPS_VECTOR_LENGTH;
  unsigned int aligned_data_length =  channel_align - num_remains;                                 
  int atomic_max_value_pos = call_reduce_argmax<float>(src_ptr, aligned_data_length);
  for (int i = 0; i< num_remains; i++) {//for unaligned remaining data
      if (src_ptr[aligned_data_length + i] > src_ptr[atomic_max_value_pos]) {
        atomic_max_value_pos = aligned_data_length + i;
      }
  }
  dst_ptr[0] = (u_int32_t)atomic_max_value_pos;
}

template <>
__forceinline__ __device__ void  atomic_reduce_argmax(u_int32_t* dst_ptr, __fp16* src_ptr,
                                            unsigned int channel_align) {
  unsigned int num_remains =  channel_align % TOPS_VECTOR_LENGTH;
  unsigned int aligned_data_length =  channel_align - num_remains;                                 
  int atomic_max_value_pos = call_reduce_argmax<__fp16>(src_ptr, aligned_data_length);
  for (int i = 0; i< num_remains; i++) {//for unaligned remaining data
      if (src_ptr[aligned_data_length + i] > src_ptr[atomic_max_value_pos]) {
        atomic_max_value_pos = aligned_data_length + i;
      }
  }
  dst_ptr[0] = (u_int32_t)atomic_max_value_pos;
}

template <>
__forceinline__ __device__ void  atomic_reduce_argmax(u_int32_t* dst_ptr, __bf16* src_ptr,
                                            unsigned int channel_align) {
  unsigned int num_remains =  channel_align % TOPS_VECTOR_LENGTH;
  unsigned int aligned_data_length =  channel_align - num_remains;                                 
  int atomic_max_value_pos = call_reduce_argmax<__bf16>(src_ptr, aligned_data_length);
  for (int i = 0; i< num_remains; i++) {//for unaligned remaining data
      if (src_ptr[aligned_data_length + i] > src_ptr[atomic_max_value_pos]) {
        atomic_max_value_pos = aligned_data_length + i;
      }
  }
  dst_ptr[0] = (u_int32_t)atomic_max_value_pos;
}

template <>
__forceinline__ __device__ void  atomic_reduce_argmax(u_int32_t* dst_ptr, int8_t* src_ptr,
                                            unsigned int channel_align) {
  unsigned int num_remains =  channel_align % TOPS_VECTOR_LENGTH;
  unsigned int aligned_data_length =  channel_align - num_remains;                                 
  int atomic_max_value_pos = call_reduce_argmax<int8_t>(src_ptr, aligned_data_length);
  for (int i = 0; i< num_remains; i++) {//for unaligned remaining data
      if (src_ptr[aligned_data_length + i] > src_ptr[atomic_max_value_pos]) {
        atomic_max_value_pos = aligned_data_length + i;
      }
  }
  dst_ptr[0] = (u_int32_t)atomic_max_value_pos;
}



template <typename T>
__device__ __forceinline__ int call_reduce_argmin(T *src_ptr,
                                              int size) {
  generic_ptr src_addr = reinterpret_cast<generic_ptr>(src_ptr);
  constexpr int bpe = sizeof(T);
  constexpr int vec_elems = sizeof(typename tops::unified_scalar<T>::type) * TOPS_VECTOR_LENGTH / bpe;
  int positions[MAX_REDUCE_ARG_BATCH];
  using vtype = typename tops::scalar_to_vector<T, vec_elems>::type;

  auto src_leaptr = tops::simple_leaptr<vtype>(src_addr);
  int group_num = (size + vec_elems - 1) / vec_elems;
  vtype vsrc;
  for (int i = 0; i < group_num; i++) {
    vsrc = src_leaptr.load();
    int min_pos = tops::vreduce_argmin<vtype>(vsrc);
    positions[i] = min_pos;
  }

  int min_value_pos = positions[0];
  for (int i = 1; i < group_num; i++) {
    if (src_ptr[i * vec_elems + positions[i]] < src_ptr[min_value_pos]) {
      min_value_pos = i * vec_elems + positions[i];
    }  
  }
  return min_value_pos;
}

template <typename TYPE>
__forceinline__ __device__ void atomic_reduce_argmin(u_int32_t* dst_ptr, TYPE* src_ptr,
                                            unsigned int channel_align) {}

template <>
__forceinline__ __device__ void  atomic_reduce_argmin(u_int32_t* dst_ptr, float* src_ptr,
                                            unsigned int channel_align) {
  unsigned int num_remains =  channel_align % TOPS_VECTOR_LENGTH;
  unsigned int aligned_data_length =  channel_align - num_remains;                                 
  int atomic_min_value_pos = call_reduce_argmin<float>(src_ptr, aligned_data_length);
  for (int i = 0; i < num_remains; i++) {//for unaligned remaining data
      if (src_ptr[aligned_data_length + i] < src_ptr[atomic_min_value_pos]) {
        atomic_min_value_pos = aligned_data_length + i;
      }
  }
  dst_ptr[0] = atomic_min_value_pos;
}

template <>
__forceinline__ __device__ void  atomic_reduce_argmin(u_int32_t* dst_ptr, __fp16* src_ptr,
                                            unsigned int channel_align) {
  unsigned int num_remains =  channel_align % TOPS_VECTOR_LENGTH;
  unsigned int aligned_data_length =  channel_align - num_remains;                                 
  int atomic_min_value_pos = call_reduce_argmin<__fp16>(src_ptr, aligned_data_length);
  for (int i = 0; i< num_remains; i++) {//for unaligned remaining data
      if (src_ptr[aligned_data_length + i] < src_ptr[atomic_min_value_pos]) {
        atomic_min_value_pos = aligned_data_length + i;
      }
  }
  dst_ptr[0] = atomic_min_value_pos;
}

template <>
__forceinline__ __device__ void  atomic_reduce_argmin(u_int32_t* dst_ptr, __bf16* src_ptr,
                                            unsigned int channel_align) {
  unsigned int num_remains =  channel_align % TOPS_VECTOR_LENGTH;
  unsigned int aligned_data_length =  channel_align - num_remains;                                 
  int atomic_min_value_pos = call_reduce_argmin<__bf16>(src_ptr, aligned_data_length);
  for (int i = 0; i< num_remains; i++) {//for unaligned remaining data
      if (src_ptr[aligned_data_length + i] < src_ptr[atomic_min_value_pos]) {
        atomic_min_value_pos = aligned_data_length + i;
      }
  }
  dst_ptr[0] = atomic_min_value_pos;
}

template <>
__forceinline__ __device__ void  atomic_reduce_argmin(u_int32_t* dst_ptr, int8_t* src_ptr,
                                            unsigned int channel_align) {
  unsigned int num_remains =  channel_align % TOPS_VECTOR_LENGTH;
  unsigned int aligned_data_length =  channel_align - num_remains;                                 
  int atomic_min_value_pos = call_reduce_argmin<int8_t>(src_ptr, aligned_data_length);
  for (int i = 0; i< num_remains; i++) {//for unaligned remaining data
      if (src_ptr[aligned_data_length + i] < src_ptr[atomic_min_value_pos]) {
        atomic_min_value_pos = aligned_data_length + i;
      }
  }
  dst_ptr[0] = atomic_min_value_pos;
}
