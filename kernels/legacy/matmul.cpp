#include "tops.h"
#include <stdio.h>
#include <tops/tops_runtime.h>
#include <tops/topsrtc.h>
#include <tops/half.h>
#include <tops/bfloat.h>

#include "utils.h"

// #if __GCU_ARCH__ >= 300
// #include "include/common/atomic_op.h"
// #include "include/common/binary.h"
// #endif


#define CHECK(cmd) \
{\
    topsError_t error  = cmd;\
    if (error != topsSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", topsGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
	  }\
}
constexpr int MAX_RANK = 3;
constexpr int TILE_DIM=32;
constexpr int BLOCK_ROWS=8;
constexpr int MAX_DIM=4096;

template <typename T, typename VT>
__device__ __forceinline__ void matmul(T *matA, T *matB, T *matTmpB, T* out, int B, int M, int K, int N)
{
  tops_dte_ctx_t ctx;
  tops::dte_scope s(ctx);
  int batch = B;
  int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int threadId = blockId * blockDim.x + threadIdx.x;

  int batch_id = threadId / M;
  int task_id = threadId % M;
  if (batch_id >= batch) 
  {
    __syncthreads();
    return; 
  }

  // printf("Batch ID = {%d}, Task ID = {%d}", batch_id, task_id);
  int offsetA = batch_id * M * K;
  int offsetB = batch_id * K * N;
  int offsetOut = batch_id * M * N;

  if (task_id == 0) { //for worker thread in each batch perform matB transpose
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx);
    __valigned__ int shapeB_[2] = {K, N};
    tops::mdspan src(tops::Global, matB + offsetB, shapeB_);

    __valigned__ int shapeTransB_[2] = {N, K};
    tops::mdspan dst(tops::Global, matTmpB + offsetB, shapeTransB_);
    
    // layout parameter
    __valigned__ int layout[2] = {1, 0};
    //transpose of matB to temp buffer (matBbuffer)
    tops::transpose(ctx, dst, src, layout);
    // printf("Transpose in Batch ID = {%d}, Task ID = {%d}", batch_id, task_id);

  }
  
  __syncthreads();

// #if __GCU_ARCH__ < 300
  __valigned__ T bufferA[TILE_DIM];
  __valigned__ T bufferB[TILE_DIM];
  __valigned__ T bufferMul[TILE_DIM];
  tops::mdspan bufA(tops::Private, &bufferA, TILE_DIM);
  tops::mdspan bufB(tops::Private, &bufferB, TILE_DIM);
  tops::mdspan bufMul(tops::Private, &bufferMul, TILE_DIM);
// #endif

// #if __GCU_ARCH__ >= 300
//   __valigned__ T bufferA[MAX_DIM];
//   __valigned__ T bufferB[MAX_DIM];
//   tops::mdspan bufA(tops::Private, &bufferA, MAX_DIM);
//   tops::mdspan bufB(tops::Private, &bufferB, MAX_DIM);
// #endif

  __valigned__ T bufferOut[MAX_DIM];
  tops::mdspan bufOut(tops::Private, &bufferOut, MAX_DIM);

  for (int j=0; j<N; j+=1) {
      //sum of mul results
      T sums = T(0.0);
    // #if __GCU_ARCH__ < 300 
      for (int i=0; i<K; i+=TILE_DIM) {
        int tilesize = TILE_DIM;
        if (i * TILE_DIM + TILE_DIM >= K) {
          tilesize = K - i * TILE_DIM;
        }
        tops::mdspan srcA(tops::Global, matA + offsetA  + task_id * K + i, tilesize);
        tops::memcpy(ctx, bufA, srcA);
        //element-wise mul of one row of matB with one column of matB
        tops::mdspan srcB(tops::Global, matTmpB + offsetB + j * K + i, tilesize);
        tops::memcpy(ctx, bufB, srcB);
        const auto &v1_ = tops::vload<VT>(bufferA);
        const auto &v2_ = tops::vload<VT>(bufferB);
        tops::vstore(tops::vmul<VT>(v1_, v2_), bufferMul);
        for (int m=0; m<tilesize; m++){
          sums += bufferMul[m];
        }
      }
    // #endif
    
    // #if __GCU_ARCH__ >= 300
    //     tops::mdspan srcA(tops::Global, matA + offsetA  + task_id * K, K);
    //     tops::memcpy(ctx, bufA, srcA);
    //     //element-wise mul of one row of matB with one column of matB
    //     tops::mdspan srcB(tops::Global, matTmpB + offsetB + j * K, K);
    //     tops::memcpy(ctx, bufB, srcB);
    //     mul(reinterpret_cast<T*>(bufferB),reinterpret_cast<T*>(bufferA), reinterpret_cast<T*>(bufferB), K);
    //     for (int m=0; m<K; m++){
    //       sums += bufferB[m];
    //     }
    // #endif
      bufferOut[j] = sums;
      // printf("%.2f, ", sums);

  }
  // printf("\n");
  __syncthreads();
//   printf("Copy output from offset {%d}\n", offsetOut);
  // printf("\nCopy output in Batch ID = {%d}, Task ID = {%d}", batch_id, task_id);

  tops::mdspan dst(tops::Global, out + offsetOut + task_id * N, N);
  tops::memcpy(ctx, dst, bufOut);
  __syncthreads();
}

extern "C" __global__ void matmul_f32(float *matA, float *matB, float *matTmpB, float* out, int B, int M, int K, int N)
{
  matmul<float, vfloat>(matA, matB, matTmpB, out, B, M, K, N);
}

extern "C" __global__ void matmul_f16(__fp16 *matA, __fp16 *matB, __fp16 *matTmpB, __fp16* out, int B, int M, int K, int N)
{
  matmul<__fp16, vhalf>(matA, matB, matTmpB, out, B, M, K, N);

}

// extern "C" __global__ void matmul_bf16(__bf16 *matA, __bf16 *matB, __bf16 *matTmpB, __bf16* out, int B, int M, int K, int N)
// {
//   matmul<__bf16, vbfloat>(matA, matB, matTmpB, out, B, M, K, N);

// }

int main(int argc, char *argv[])
{
    CHECK(topsInit(0));
    CHECK(topsSetDevice(0));

    float *lhs_d, *rhs_d, *out_d, *rhs_tmp_d;
    float *lhs_h, *rhs_h, *out_h, *rhs_tmp_h;
    int *shape_lhs_d, *shape_rhs_d;
    int M = 13;
    int K = 13;
    int N = 128;
    const int MAX_RANK = 3;
    const int batch = 32;

    size_t size_lhs = M * K;
    size_t size_rhs = K * N;
    size_t size_out = M * N;
    int shape_lhs[MAX_RANK] = {batch, M, K};
    int shape_rhs[MAX_RANK] = {batch, K, N};


    lhs_h = (float *)aligned_alloc(4096, batch * size_lhs * sizeof(float));
    CHECK(lhs_h == 0 ? topsErrorMemoryAllocation : topsSuccess);
    rhs_h = (float *)aligned_alloc(4096, batch * size_rhs * sizeof(float));
    CHECK(rhs_h == 0 ? topsErrorMemoryAllocation : topsSuccess);
    rhs_tmp_h = (float *)aligned_alloc(4096, batch * size_rhs * sizeof(float));
    CHECK(rhs_tmp_h == 0 ? topsErrorMemoryAllocation : topsSuccess);
    out_h = (float *)aligned_alloc(4096, batch * size_out * sizeof(float));
    CHECK(out_h == 0 ? topsErrorMemoryAllocation : topsSuccess);
    
    for (size_t b = 0; b < batch; b++) { 
      for (size_t i = 0; i < size_lhs; i++) {
          lhs_h[b * size_lhs + i] = 0.2;
      }
      for (size_t i = 0; i < size_rhs; i++) {
          rhs_h[b * size_rhs + i] = 0.5;
      }
      for (size_t i = 0; i < size_rhs; i++) {
          rhs_tmp_h[b * size_rhs + i] = 0.0;
      }
      for (size_t i = 0; i < size_out; i++) {
          out_h[b * size_out + i] = 0.0;
      }
    }

    CHECK(topsMalloc(&lhs_d, batch * size_lhs * sizeof(float)));
    CHECK(topsMalloc(&rhs_d, batch * size_rhs * sizeof(float)));
    CHECK(topsMalloc(&rhs_tmp_d, batch * size_rhs * sizeof(float)));

    CHECK(topsMalloc(&out_d, batch * size_out * sizeof(float)));
    CHECK(topsMalloc(&shape_lhs_d, MAX_RANK * sizeof(int)));
    CHECK(topsMalloc(&shape_rhs_d, MAX_RANK * sizeof(int)));

    printf("info: copy Host2Device\n");
    CHECK(topsMemcpy(lhs_d, lhs_h, batch * size_lhs * sizeof(float),
                    topsMemcpyHostToDevice));
    CHECK(topsMemcpy(rhs_d, rhs_h, batch * size_rhs * sizeof(float),
                    topsMemcpyHostToDevice));
    CHECK(topsMemcpy(rhs_tmp_d, rhs_tmp_h, batch * size_rhs * sizeof(float),
                    topsMemcpyHostToDevice));
    CHECK(topsMemcpy(out_d, out_h, batch * size_out * sizeof(float),
                    topsMemcpyHostToDevice));

    CHECK(topsMemcpy(shape_lhs_d, &shape_lhs, MAX_RANK * sizeof(int),
                    topsMemcpyHostToDevice));

    CHECK(topsMemcpy(shape_rhs_d, &shape_rhs, MAX_RANK * sizeof(int),
                    topsMemcpyHostToDevice));

    topsEvent_t start, stop;
    int ITERATION = 20;
    float total_time = 0.0;
    for (int i=0; i< ITERATION; i++) {
      printf("kernel launch [%d %d]\n", M, batch);
      CHECK(topsEventCreate(&start));
      CHECK(topsEventCreate(&stop));

      CHECK(topsEventRecord(start));

      matmul_f32<<<dim3(M, batch, 1), dim3(1, 1, 1)>>>(lhs_d, rhs_d, rhs_tmp_d, out_d, batch, M, K, N);

      CHECK(topsEventRecord(stop));
      CHECK(topsEventSynchronize(stop));
      float time = 0.0;
      CHECK(topsEventElapsedTime(&time, start, stop));
      total_time += time;
    }
    
      printf("Average Time taken: %g ms, Shape: [%d %d %d %d] --------\n", total_time / ITERATION, batch, M, K, N);

    printf("info: copy Device2Host\n");
    CHECK(topsMemcpy(out_h, out_d, batch * size_out * sizeof(float), topsMemcpyDeviceToHost));

    // printf("info: pring results\n");
    // for (size_t b = 0; b < batch; b++) {
    //   for (size_t i = 0; i < M; i++) {
    //       for (size_t j = 0; j < N; j++) {
    //           printf("%.1f,", out_h[b * size_out + i * N + j]);
    //       }
    //       printf("\n");
    //   }
    //   printf("\n\n");
    // }

    // CHECK(topsMemcpy(rhs_tmp_h, rhs_tmp_d, batch * size_rhs * sizeof(float), topsMemcpyDeviceToHost));

    // for (size_t b = 0; b < batch; b++) {
    //   for (size_t i = 0; i < M; i++) {
    //       for (size_t j = 0; j < H; j++) {
    //           printf("%.2f,", rhs_h[b * size_rhs + i * H + j]);
    //       }
    //       printf("\n");
    //   }
    //   printf("\n\n");
    // }

    //       for (size_t j = 0; j < size_rhs; j++) {
    //           printf("%.2f,", rhs_h[j]);
    //       }
    //       printf("\n\n");
    // printf("print transpose results\n");
    // for (size_t b = 0; b < batch; b++) {
    //   for (size_t i = 0; i < H; i++) {
    //       for (size_t j = 0; j < M; j++) {
    //           printf("%.2f,", rhs_tmp_h[b * size_rhs + i * M + j]);
    //       }
    //       printf("\n");
    //   }
    //   printf("\n\n");
    // }

    return 0;
}
