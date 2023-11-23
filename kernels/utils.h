#ifndef OP_UTILS_
#define OP_UTILS_
#include <tops.h>
#include <cstdint>

using namespace std;

constexpr int MAX_RANK = 4;
constexpr int MAX_PAVO_CLUSTER_NUM = 4;
constexpr int MAX_PAVO_SIP_NUM = 6;
constexpr int MAX_DRD_CLUSTER_NUM = 2;
constexpr int MAX_DRD_SIP_NUM = 12;
constexpr int MAX_SCP_CLUSTER_NUM = 1;
constexpr int MAX_SCP_SIP_NUM = 12;
constexpr int SIP_VECTOR_LENGTH = 128;
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
__device__ inline Int_Type AlignDown(Int_Type x, Int_Type2 y) {
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