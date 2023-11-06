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


// ceiling division of two integers
template <typename Int_Type, typename Int_Type2 = Int_Type>
inline constexpr int CeilDiv(Int_Type x, Int_Type2 y) {
  return (x + y - 1) / y;
}

// align to a multiple of rhs no less than lhs
template <typename Int_Type, typename Int_Type2 = Int_Type>
inline Int_Type AlignUp(Int_Type x, Int_Type2 y) {
  return CeilDiv(x, y) * y;
}

// align to a multiple of rhs no more than lhs
template <typename Int_Type, typename Int_Type2 = Int_Type>
inline Int_Type AlignDown(Int_Type x, Int_Type2 y) {
  return (x / y) * y;
}

using FP_UNARY = void (*)(void*, void*, int);
