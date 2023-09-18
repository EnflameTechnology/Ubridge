#include "tops.h"
#pragma clang force_cuda_host_device begin
#include <stdio.h>
#pragma clang force_cuda_host_device end
#include <stdio.h>
#include <tops/tops_runtime.h>
#include <tops/topsrtc.h>

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


template <typename T, std::size_t N>
__device__ void copy_to_buffer(
  tops_dte_ctx_t& ctx, 
  T *buf_l3, 
  T *buf_l1)
{
  tops_dte_ctx_t p_ctx;
  tops::dte_scope p_s(p_ctx);

  tops::mdspan from(tops::Global, buf_l3, N);
  tops::mdspan to(tops::Private, buf_l1, N);
  tops::memcpy(p_ctx, to, from);
}

//matA_shape [batch, m, k], matB_shape[batch, k, n], out shape [batch, m, n]
extern "C" __global__ void batch_matmul(float *matA, float *matB, float *matTmpB, float* out, int* matA_shape, int* matB_shape)
{
  
  tops_dte_ctx_t ctx;
  tops::dte_scope s(ctx);

  __valigned__ int shapeA[MAX_RANK];
  __valigned__ int shapeB[MAX_RANK];
  copy_to_buffer<int, MAX_RANK>(ctx, matA_shape, shapeA);
  copy_to_buffer<int, MAX_RANK>(ctx, matB_shape, shapeB);
  int batch = shapeA[0];
  int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int threadId = blockId * blockDim.x + threadIdx.x;
  int batch_id = threadId / shapeA[1];
  int task_id = threadId % shapeA[1];
  if (batch_id >= batch) 
  {
    __syncthreads();
    return; 
  }

  // printf("Batch ID = {%d}, Task ID = {%d}", batch_id, task_id);
  int strideA = shapeA[2];
  int strideOut = shapeB[2];
  int offsetA = batch_id * shapeA[1] * shapeA[2];
  int offsetB = batch_id * shapeB[1] * shapeB[2];
  int offsetOut = batch_id * shapeA[1] * shapeB[2];

  if (task_id == 0) { //for worker thread in each batch perform matB transpose
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx);
    __valigned__ int shapeB_[2] = {shapeB[1], shapeB[2]};
    tops::mdspan src(tops::Global, matB + offsetB, shapeB_);

    __valigned__ int shapeTransB_[2] = {shapeB[2], shapeB[1]};
    tops::mdspan dst(tops::Global, matTmpB + offsetB, shapeTransB_);
    
    // layout parameter
    __valigned__ int layout[2] = {1, 0};
    //transpose of matB to temp buffer (matBbuffer)
    tops::transpose(ctx, dst, src, layout);
    // printf("Transpose in Batch ID = {%d}, Task ID = {%d}", batch_id, task_id);

  }
  
  __syncthreads();


  __valigned__ float bufferA[TILE_DIM];
  __valigned__ float bufferB[TILE_DIM];
  __valigned__ float bufferMul[TILE_DIM];
  __valigned__ float bufferOut[TILE_DIM*400];

  tops::mdspan bufA(tops::Private, &bufferA, TILE_DIM);
  tops::mdspan bufB(tops::Private, &bufferB, TILE_DIM);
  tops::mdspan bufMul(tops::Private, &bufferMul, TILE_DIM);
  tops::mdspan bufOut(tops::Private, &bufferOut, strideOut);

  for (int j=0; j<strideOut; j+=1) {
      //sum of mul results
      float sums = 0.0;
      for (int i=0; i<strideA; i+=TILE_DIM) {
        int tilesize = TILE_DIM;
        if (i * TILE_DIM + TILE_DIM >= strideA) {
          tilesize = strideA - i * TILE_DIM;
        }
        tops::mdspan srcA(tops::Global, matA + offsetA  + task_id * strideA + i, tilesize);
        tops::memcpy(ctx, bufA, srcA);
        //element-wise mul of one row of matB with one column of matB
        tops::mdspan srcB(tops::Global, matTmpB + offsetB + j * strideA + i, tilesize);
        tops::memcpy(ctx, bufB, srcB);
        const auto &v1_ = tops::vload<vfloat>(bufferA);
        const auto &v2_ = tops::vload<vfloat>(bufferB);
        tops::vstore(tops::vmul<vfloat>(v1_, v2_), bufferMul);
        for (int m=0; m<tilesize; m++){
          sums += bufferMul[m];
        }
      }
      bufferOut[j] = sums;
      // printf("%.2f, ", sums);

  }
  // printf("\n");
  __syncthreads();
//   printf("Copy output from offset {%d}\n", offsetOut);
  // printf("\nCopy output in Batch ID = {%d}, Task ID = {%d}", batch_id, task_id);

  tops::mdspan dst(tops::Global, out + offsetOut + task_id * strideOut, strideOut);
  tops::memcpy(ctx, dst, bufOut);
  __syncthreads();
}


int main(int argc, char *argv[])
{
    CHECK(topsInit(0));
    CHECK(topsSetDevice(1));

    float *lhs_d, *rhs_d, *out_d, *rhs_tmp_d;
    float *lhs_h, *rhs_h, *out_h, *rhs_tmp_h;
    int *shape_lhs_d, *shape_rhs_d;
    int W = 32;
    int M = 16;
    int H = 64;
    const int MAX_RANK = 3;
    const int batch = 2;

    size_t size_lhs = W * M;
    size_t size_rhs = M * H;
    size_t size_out = W * H;
    int shape_lhs[MAX_RANK] = {batch, W, M};
    int shape_rhs[MAX_RANK] = {batch, M, H};


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

    // topsModuleLaunchKernel(kernel, W/4, 1, 1, 4, 1, 1, 0, nullptr, nullptr, config);
    batch_matmul<<<dim3(W, batch, 1), dim3(1, 1, 1)>>>(lhs_d, rhs_d, rhs_tmp_d, out_d, shape_lhs_d, shape_rhs_d);

    printf("info: copy Device2Host\n");
    CHECK(topsMemcpy(out_h, out_d, batch * size_out * sizeof(float), topsMemcpyDeviceToHost));

    printf("info: pring results\n");
    for (size_t b = 0; b < batch; b++) {
      for (size_t i = 0; i < W; i++) {
          for (size_t j = 0; j < H; j++) {
              printf("%.1f,", out_h[b * size_out + i * H + j]);
          }
          printf("\n");
      }
      printf("\n\n");
    }

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
