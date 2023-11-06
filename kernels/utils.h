#include <tops.h>

__device__ __forceinline__ int GetBlockNum(void) {
  return (gridDim.x * gridDim.y * gridDim.z);
}

__device__ __forceinline__ int GetBlockIdx(void) {
  return (blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x +
          blockIdx.x);
}

__device__ __forceinline__ int GetThreadNumEachBlock(void) {
  return (blockDim.x * blockDim.y * blockDim.z);
}

__device__ __forceinline__ int GetThreadNum(void) {
  return GetBlockNum() * GetThreadNumEachBlock();
}

__device__ __forceinline__ int GetThreadIdxInBlock(void) {
  return threadIdx.z * (blockDim.x * blockDim.y) +
      threadIdx.y * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ int GetThreadIdx(void) {
  int blockIdx = GetBlockIdx();
  int threadNumEachBlock = GetThreadNumEachBlock();

  return blockIdx * threadNumEachBlock + GetThreadIdxInBlock();
}

using FP_UNARY = void (*)(void*, void*, int);
