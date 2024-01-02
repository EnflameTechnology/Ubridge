#include <stdio.h>
#include <tops.h>
#include <tops/half.h>
#include <algorithm>
#include <vector>
#include "tops/tops_runtime.h"
#include "utils.h"
using namespace std;
#include <acore/atomic_op.h>

template <typename TYPE>
__forceinline__ __device__ void atomic_reduce_max(TYPE* dst_ptr, TYPE* src_ptr,
                                            unsigned int channel_align) {}

template <>
__forceinline__ __device__ void  atomic_reduce_max(float* dst_ptr, float* src_ptr,
                                            unsigned int channel_align) {
  dst_ptr[0] = src_ptr[0];
  for (int i=1; i< channel_align; i++) {
    if (src_ptr[i] > dst_ptr[0])
    {
      dst_ptr[0] = src_ptr[i];
    }
  }
}

template <>
__forceinline__ __device__ void  atomic_reduce_max(tops::half* dst_ptr, tops::half* src_ptr,
                                            unsigned int channel_align) {
  dst_ptr[0] = src_ptr[0];
  for (int i=1; i< channel_align; i++) {
    if (src_ptr[i] > dst_ptr[0])
    {
      dst_ptr[0] = src_ptr[i];
    }
  }
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

template <typename TYPE>
__forceinline__ __device__ void atomic_reduce_min(TYPE* dst_ptr, TYPE* src_ptr,
                                            unsigned int channel_align) {}

template <>
__forceinline__ __device__ void  atomic_reduce_min(float* dst_ptr, float* src_ptr,
                                            unsigned int channel_align) {
  dst_ptr[0] = src_ptr[0];
  for (int i=1; i< channel_align; i++) {
    if (src_ptr[i] < dst_ptr[0])
    {
      dst_ptr[0] = src_ptr[i];
    }
  }
}

template <>
__forceinline__ __device__ void  atomic_reduce_min(tops::half* dst_ptr, tops::half* src_ptr,
                                            unsigned int channel_align) {
  dst_ptr[0] = src_ptr[0];
  for (int i=1; i< channel_align; i++) {
    if (src_ptr[i] < dst_ptr[0])
    {
      dst_ptr[0] = src_ptr[i];
    }
  }
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


template <typename TYPE>
__forceinline__ __device__ void atomic_reduce_sum(TYPE* dst_ptr, TYPE* src_ptr,
                                            unsigned int channel_align) {}


template <>
__forceinline__ __device__ void  atomic_reduce_sum(float* dst_ptr, float* src_ptr,
                                            unsigned int channel_align) {
  dst_ptr[0] = src_ptr[0];
  for (int i=1; i< channel_align; i++) {
      dst_ptr[0] += src_ptr[i];
  }
}

template <>
__forceinline__ __device__ void  atomic_reduce_sum(tops::half* dst_ptr, tops::half* src_ptr,
                                            unsigned int channel_align) {
  dst_ptr[0] = src_ptr[0];
  for (int i=1; i< channel_align; i++) {
      dst_ptr[0] += src_ptr[i];
  }
}

template <>
__forceinline__ __device__ void  atomic_reduce_sum(int8_t* dst_ptr, int8_t* src_ptr,
                                            unsigned int channel_align) {
  dst_ptr[0] = src_ptr[0];
  for (int i=1; i< channel_align; i++) {
      dst_ptr[0] += src_ptr[i];
  }
}