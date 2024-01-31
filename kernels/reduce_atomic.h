#include <stdio.h>
#include <tops.h>
#include <tops/half.h>
#include <algorithm>
#include <vector>
#include "tops/tops_runtime.h"
#include "utils.h"
using namespace std;

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
__forceinline__ __device__ void  atomic_reduce_max(tops::bfloat* dst_ptr, tops::bfloat* src_ptr,
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
__forceinline__ __device__ void  atomic_reduce_min(tops::bfloat* dst_ptr, tops::bfloat* src_ptr,
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
  if (channel_align % TOPS_VECTOR_LENGTH == 0) {
    dst_ptr[0] = call_reduce_sum<float>(src_ptr, channel_align);
    return;
  }
  dst_ptr[0] = src_ptr[0];
  for (int i=1; i< channel_align; i++) {
      dst_ptr[0] += src_ptr[i];
  }
}

template <>
__forceinline__ __device__ void  atomic_reduce_sum(__fp16* dst_ptr, __fp16* src_ptr,
                                            unsigned int channel_align) {
  if (channel_align % TOPS_VECTOR_LENGTH == 0) {
    dst_ptr[0] = call_reduce_sum<__fp16>(src_ptr, channel_align);
    return;
  }
  dst_ptr[0] = src_ptr[0];
  for (int i=1; i< channel_align; i++) {
      dst_ptr[0] += src_ptr[i];
  }
}

template <>
__forceinline__ __device__ void  atomic_reduce_sum(tops::bfloat* dst_ptr, tops::bfloat* src_ptr,
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

template <typename TYPE>
__forceinline__ __device__ float reduce_sum_scalar(TYPE* src_ptr,
                                            unsigned int channel_align) {}

template <>
__forceinline__ __device__ float  reduce_sum_scalar(int8_t* src_ptr,
                                            unsigned int channel_align) {
  float v = static_cast<float>(src_ptr[0]);
  for (int i=0; i< channel_align; i++) {
      v += src_ptr[i];
  }
  return v;
}

template <>
__forceinline__ __device__ float  reduce_sum_scalar(float* src_ptr,
                                            unsigned int channel_align) {
  float v = static_cast<float>(src_ptr[0]);
  for (int i=0; i< channel_align; i++) {
      v += src_ptr[i];
  }
  return v;
}

template <>
__forceinline__ __device__ float  reduce_sum_scalar(tops::half* src_ptr,
                                            unsigned int channel_align) {
  float v = static_cast<float>(src_ptr[0].value);
  for (int i=0; i< channel_align; i++) {
      v += src_ptr[i].value;
  }
  return v;
}

template <>
__forceinline__ __device__ float  reduce_sum_scalar(tops::bfloat* src_ptr,
                                            unsigned int channel_align) {
  float v = static_cast<float>(src_ptr[0].value);
  for (int i=0; i< channel_align; i++) {
      v += src_ptr[i].value;
  }
  return v;
}

