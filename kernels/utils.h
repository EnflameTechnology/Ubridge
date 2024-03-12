#ifndef OP_UTILS_
#define OP_UTILS_
#include <tops.h>
#include <cstdint>
#include <vector>
#include <mutex>
#include <algorithm>
#include <functional>
#include <numeric>
#include <krt/vector_infra.h>
using namespace std;
using namespace tops;
constexpr int MAX_RANK = 4;
constexpr int MAX_PAVO_CLUSTER_NUM = 4;
constexpr int MAX_PAVO_SIP_NUM = 6;
constexpr int MAX_DRD_CLUSTER_NUM = 2;
constexpr int MAX_DRD_SIP_NUM = 12;
constexpr int MAX_SCP_CLUSTER_NUM = 1;
constexpr int MAX_SCP_SIP_NUM = 12;
constexpr int SIP_VECTOR_LENGTH = 128;
#define SHARE_BUFFER_SIZE 1024 * 1024 * 24 //24MB

#if defined(__GCU_ARCH__) && (__GCU_ARCH__ == 200)
  constexpr static int VDMEM_SIZE = 0x80000;
#elif defined(__AGCU_ARCH__) && (__AGCU_ARCH__ == 200)
  constexpr static int VDMEM_SIZE = 0x80000;
#elif defined(__GCU_ARCH__) || defined(__AGCU_ARCH__)
  constexpr static int VDMEM_SIZE = 0x100000;
#else
  constexpr static int VDMEM_SIZE = 0x80000;
#endif

#if defined(__GCU_ARCH__) && (__GCU_ARCH__ == 300)
#if !defined(__AGCU_ARCH__)
  constexpr static int VDMEM_REAL_SIZE = 0x180000;
#elif defined(__AGCU_ARCH__) && (__AGCU_ARCH__ == 300)
  constexpr static int VDMEM_REAL_SIZE = 0x180000;
#else
  constexpr static int VDMEM_REAL_SIZE = VDMEM_SIZE;
#endif
#else
  constexpr static int VDMEM_REAL_SIZE = VDMEM_SIZE;
#endif

#if defined(__GCU_ARCH__)
#if __GCU_ARCH__ >= 200 && __GCU_ARCH__ <= 300
using IndexType = uint32_t;
#else
using IndexType = size_t;
#endif
#else
using IndexType = size_t;
#endif

#ifndef __UNDERLYING_TYPE__
#define __UNDERLYING_TYPE__
  template <typename T> struct UnderlyingType;
  template <> struct UnderlyingType<float> { using type = float; };
  template <> struct UnderlyingType<tops::half> { using type = __fp16; };
  template <> struct UnderlyingType<tops::bfloat16> { using type = __bf16; };
  template <> struct UnderlyingType<int32_t> { using type = int; };
  template <> struct UnderlyingType<int16_t> { using type = short; };
  template <> struct UnderlyingType<int8_t> { using type = char; };
  template <> struct UnderlyingType<uint32_t> { using type =  unsigned int; };
  template <> struct UnderlyingType<uint16_t> { using type = unsigned short; };
  template <> struct UnderlyingType<uint8_t> { using type = unsigned char; };
#endif

template <typename RET_TYPE>
inline KRT_API RET_TYPE viota(unsigned int start);
template <typename RET_TYPE, typename MB_TYPE>
inline KRT_API RET_TYPE viota_t(const MB_TYPE& mb, unsigned int start,
                                const RET_TYPE& remain);
template <typename RET_TYPE, typename MB_TYPE>
inline KRT_API RET_TYPE viota_f(const MB_TYPE& mb, unsigned int start,
                                const RET_TYPE& remain) {
  auto mask = mask_not(mb);
  return viota_t<RET_TYPE>(mask, start, remain);
}

template <typename RET_TYPE>
inline KRT_API RET_TYPE viota_dup(unsigned int start);
template <typename RET_TYPE, typename MB_TYPE>
inline KRT_API RET_TYPE viota_dup_t(const MB_TYPE& mb, unsigned int start,
                                    const RET_TYPE& remain);
template <typename RET_TYPE, typename MB_TYPE>
inline KRT_API RET_TYPE viota_dup_f(const MB_TYPE& mb, unsigned int start,
                                    const RET_TYPE& remain) {
  auto mask = mask_not(mb);
  return viota_dup_t<RET_TYPE>(mask, start, remain);
}

#if (__KRT_ARCH__ >= 300)
#define VIOTA_IMPL(TYPE, NUM, BOOL_NUM)                                       \
  template <>                                                                 \
  inline KRT_API va##NUM##TYPE##x4 viota(unsigned int start) {                \
    return __dtu_m_mid_m0_##TYPE(start);                                      \
  }                                                                           \
  template <>                                                                 \
  inline KRT_API va##NUM##TYPE##x4 viota_dup(unsigned int start) {            \
    return __dtu_m_mid_m1_##TYPE(start);                                      \
  }                                                                           \
  template <>                                                                 \
  inline KRT_API va##NUM##TYPE##x4 viota_t(const vbool##BOOL_NUM##_t& mb,     \
                                           unsigned int start,                \
                                           const va##NUM##TYPE##x4& remain) { \
    return __dtu_m_mid_m0_##TYPE##_vm(start, remain, mb);                     \
  }                                                                           \
  template <>                                                                 \
  inline KRT_API va##NUM##TYPE##x4 viota_dup_t(                               \
      const vbool##BOOL_NUM##_t& mb, unsigned int start,                      \
      const va##NUM##TYPE##x4& remain) {                                      \
    return __dtu_m_mid_m1_##TYPE##_vm(start, remain, mb);                     \
  }

VIOTA_IMPL(u32, 16, 64);
VIOTA_IMPL(u16, 32, 128);
VIOTA_IMPL(u8, 64, 256);
#undef VIOTA_IMPL
#endif  // __KRT_ARCH__ >= 300

template <int BPE> struct UnsignedByBPE;
template <> struct UnsignedByBPE<8> { using type = uint64_t; };
template <> struct UnsignedByBPE<4> { using type = uint32_t; };
template <> struct UnsignedByBPE<2> { using type = uint16_t; };
template <> struct UnsignedByBPE<1> { using type = uint8_t; };

constexpr int BYTES_FOR_VA = TOPS_VECTOR_LENGTH;
constexpr int BYTES_FOR_DA = TOPS_VECTOR_LENGTH * 2;
constexpr int BYTES_FOR_QA = TOPS_VECTOR_LENGTH * 4;
constexpr int NUMS_SPLIT = TOPS_VECTOR_LENGTH;

using IndexType = unsigned int;
using VecIndexType =
    typename scalar_to_vector<IndexType,
                              BYTES_FOR_QA / sizeof(IndexType)>::type;

template <int RANK> using ArrayVecIndexType = VecIndexType[RANK];

constexpr int FIXED_VEC_LENGTH = vector_length<VecIndexType>::value;
template <int BPE>
using FixedVecValueType =
    typename scalar_to_vector<typename UnsignedByBPE<BPE>::type,
                              FIXED_VEC_LENGTH>::type;
using FixedVecMaskType = typename vector_to_mask<VecIndexType>::type;

#define ALIGN_UP(a, b) (((a + b - 1) / b) * b)

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

__forceinline__ int GetGrids(int total_threads, int sip_number) {
  int grids = total_threads / sip_number;
  if (total_threads % sip_number != 0) grids ++;
  return grids;
}

// ceiling division of two integers
template <typename Int_Type, typename Int_Type2 = Int_Type>
__device__ inline constexpr int CeilDiv(Int_Type x, Int_Type2 y) {
  return (x + y - 1) / y;
}

// align to a multiple of rhs no less than lhs
template <typename Int_Type, typename Int_Type2 = Int_Type>
__device__ inline constexpr Int_Type AlignUp(Int_Type x, Int_Type2 y) {
  return CeilDiv(x, y) * y;
}

// align to a multiple of rhs no more than lhs
template <typename Int_Type, typename Int_Type2 = Int_Type>
__device__ inline constexpr Int_Type AlignDown(Int_Type x, Int_Type2 y) {
  return (x / y) * y;
}

template <typename T>
void check_data(T* d_result, T* h_result, int sz) {
  for (int i=0; i< sz; i++) {
    if (d_result[i] - h_result[i] > 0.001 || h_result[i] - d_result[i] > 0.001) {
      printf("Dif %.5f\n", d_result[i] - h_result[i]);
    }
  }

}


// struct memref {char *addr; int offset;};
#define CHECK(cmd) \
{\
    topsError_t error  = cmd;\
    if (error != topsSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", topsGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
	  }\
}


__device__ __forceinline__
int get_index() {
    std::size_t blockIndex = blockIdx.z*(gridDim.x*gridDim.y)
        + blockIdx.y*gridDim.x + blockIdx.x;
    std::size_t threadIndex = threadIdx.z*(blockDim.x*blockDim.y)
        + threadIdx.y*blockDim.x + threadIdx.x;
    return blockIndex*(blockDim.x*blockDim.y*blockDim.z) + threadIndex;
}

__device__ __forceinline__
int get_threadIndex() {
    std::size_t threadIndex = threadIdx.z*(blockDim.x*blockDim.y)
        + threadIdx.y*blockDim.x + threadIdx.x;
    return threadIndex;
}

__device__ __forceinline__ bool is_contiguous(
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {
    size_t acc = 1;
    for (unsigned int d = 0; d < num_dims; d++) {
        unsigned int dim_idx = num_dims - 1 - d;
        if (acc != strides[dim_idx]) {
            return false;
        }
        acc *= dims[dim_idx];
    }
    return true;
}


__device__ __forceinline__  bool is_non_overlapping_dense(size_t ndim, size_t* dims, size_t* strides, size_t* perm) {
  if (ndim == 1) {
    return strides[0] == 1;
  }
  if (ndim >= 2) {
    for (int i = 0; i < ndim - 1; ++i) {
      int idx = perm[i];
      int r_idx = perm[i + 1];
      if (strides[idx] != strides[r_idx] * dims[r_idx]) {
        return false;
      }
    }
  }
  return true;
}

__device__ __forceinline__ bool is_same_array(size_t rank, size_t*src_perm, size_t* dst_perm) {
  for (int i=0; i< rank; i++) {
    if (src_perm[i] != dst_perm[i]) {
      return false;
    }
  }
  return true;
}

__device__ __forceinline__ bool is_any_zero(size_t rank, size_t* strides) {
    for (int i = 0; i < rank; ++i) {
      if (strides[i] == 0) {
        return true;
      }
    }
    return false;
}

/**
 * @brief Cast a pointer or an interger to GCU intrinsic pointer
 *
 * @param any pointer type or integer type with same bits as a pointer
 * @return intrinsic pointer
 */
template <typename T, typename std::enable_if<std::is_integral<T>::value,
                                              bool>::type = true>
__device__ __forceinline__ __DTU_INTRIN_AS__ char* CastToGcuPtr(T ptr) {
  unsigned long long addr = static_cast<unsigned long long>(ptr);
  return (__DTU_INTRIN_AS__ char*)addr;
}

template <typename T,
          typename std::enable_if<std::is_pointer<T>::value, bool>::type = true>
__device__ __forceinline__ __DTU_INTRIN_AS__ char* CastToGcuPtr(T ptr) {
  return (__DTU_INTRIN_AS__ char*)ptr;
}

template <typename T>
struct DATA {
  T *lhs_d, *rhs_d, *out_d;
  T *lhs_h, *rhs_h, *out_h;
  T *expected;
  bool check;

  size_t size_lhs, size_rhs, size_out;

  explicit DATA(int b, int m, int k, int n, int tile_size, bool check) {
    this->check = check;
    size_lhs = b * m * k;
    size_rhs = b * k * n;
    size_out = b * m * n;
    printf("Prepare data for shape: [%d, %d, %d, %d]\n", b, m, k, n);
    lhs_h = reinterpret_cast<T *>(aligned_alloc(4096,
                                      size_lhs * sizeof(T)));
    rhs_h = reinterpret_cast<T *>(aligned_alloc(4096,
                                      size_rhs * sizeof(T)));
    out_h = reinterpret_cast<T *>(aligned_alloc(4096,
                                      size_out * sizeof(T)));
    if (check) {
        expected = reinterpret_cast<T *>(aligned_alloc(4096,
                                        size_out * sizeof(T)));
    }
    for (size_t i = 0; i < size_lhs; i++) {
      lhs_h[i] = (T)0.05;
      // if (lhs_h[i] == (T)0) {
      //   if (i%2) {lhs_h[i] = (T)1;} else {
      //   lhs_h[i] = (T)(-1);}
      // }
    }
    for (size_t i = 0; i < size_rhs; i++) {
      rhs_h[i] = (T)0.05;
      // if (rhs_h[i] == (T)0) {
      //   rhs_h[i] = (T)1;
      // }
    }
    if (check) {
        printf("Compute CPU Results...\n");
        for (int w=0; w<b; w++){
          T * p_expected = expected + w * (m * n);
          T * l_data = lhs_h + w * (m * k);
          T * r_data = rhs_h + w * (k * n);
          // //CPU results for each batch
          for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                p_expected[i * n + j] = (T)0;
                for (int v = 0; v < k/tile_size; v++) {
                T sum = (T)0;
                for (int c = 0; c < tile_size; c++) {
                    sum += l_data[i * k + v * tile_size + c] *
                            r_data[(v * tile_size + c) * n + j];
                }
                p_expected[i * n + j] += sum;
                }
            }
          }
        }

    }

    printf("Preprare device memory...\n");

    topsMalloc(&lhs_d, size_lhs * sizeof(T));
    topsMalloc(&rhs_d, size_rhs * sizeof(T));
    topsMalloc(&out_d, size_out * sizeof(T));

    topsMemcpy(lhs_d, lhs_h, size_lhs * sizeof(T), topsMemcpyHostToDevice);
    topsMemcpy(rhs_d, rhs_h, size_rhs * sizeof(T), topsMemcpyHostToDevice);
  }

  ~DATA() {
    topsFree(lhs_d);
    topsFree(rhs_d);
    topsFree(out_d);

    free(lhs_h);
    free(rhs_h);
    free(out_h);
    if(this->check)
        free(expected);
  }
};

namespace tops {

  template <typename T>
  __device__ __host__ __forceinline__ constexpr int hvlength() {
    return 128 / sizeof(T);
  }

  template<> __device__ __host__ __forceinline__
  constexpr int hvlength<vchar>() { return 128; }

  template<> __device__ __host__ __forceinline__
  constexpr int hvlength<vuchar>() { return 128; }

  template<> __device__ __host__ __forceinline__
  constexpr int hvlength<vshort>() { return 64; }

  template<> __device__ __host__ __forceinline__
  constexpr int hvlength<vushort>() { return 64; }

  template<> __device__ __host__ __forceinline__
  constexpr int hvlength<vint>() { return 32; }

  template<> __device__ __host__ __forceinline__
  constexpr int hvlength<vuint>() { return 32; }

  template<> __device__ __host__ __forceinline__
  constexpr int hvlength<vhalf>() { return 64; }

  template<> __device__ __host__ __forceinline__
  constexpr int hvlength<vbfloat>() { return 64; }

  template<> __device__ __host__ __forceinline__
  constexpr int hvlength<vfloat>() { return 32; }
} // namespace tops

using FP_UNARY = void (*)(void*, void*, int);
#endif