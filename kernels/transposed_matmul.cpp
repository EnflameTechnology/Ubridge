#include "tops.h"
#pragma clang force_cuda_host_device begin
#include <stdio.h>
#pragma clang force_cuda_host_device end
#include <stdio.h>
#include <tops/tops_runtime.h>
#include <tops/topsrtc.h>
#include <unistd.h>
#define CHECK(cmd) \
{\
    topsError_t error  = cmd;\
    if (error != topsSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", topsGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
	  }\
}
constexpr int MAX_RANK = 3;
constexpr int TILE_DIM=64;
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

//matA_shape [1, m, k], tranasposed matB_shape[1, n, k], out shape [1, m, n]
extern "C" __global__ void transposed_matmul(float *matA, float *matB, float* out, int* matA_shape, int* matB_shape)
{
  tops_dte_ctx_t ctx;
  tops::dte_scope s(ctx);

  __valigned__ int buffer_shapeA[MAX_RANK];
  copy_to_buffer<int, MAX_RANK>(ctx, matA_shape, buffer_shapeA);
  int splits = buffer_shapeA[1];
  int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int tid = blockId * blockDim.x + threadIdx.x;
  if (tid >= splits) {
    __syncthreads();
    return;
  }

  __valigned__ int buffer_shapeB[MAX_RANK];

  copy_to_buffer<int, MAX_RANK>(ctx, matB_shape, buffer_shapeB);

  int matA_stride = buffer_shapeA[2];
  int out_stride = buffer_shapeB[1];


  int offsetA = tid * matA_stride;
  int offsetOut = tid * out_stride;

  __valigned__ float bufferA[TILE_DIM];
  __valigned__ float bufferB[TILE_DIM];
  __valigned__ float bufferMul[TILE_DIM];
  __valigned__ float bufferOut[TILE_DIM*192];

  tops::mdspan bufA(tops::Private, &bufferA, TILE_DIM);
  tops::mdspan bufB(tops::Private, &bufferB, TILE_DIM);
  tops::mdspan bufMul(tops::Private, &bufferMul, TILE_DIM);
  tops::mdspan bufOut(tops::Private, &bufferOut, out_stride);

  for (int j=0; j<out_stride; j+=1) {
      //sum of mul results
      float sums = 0.0;
      for (int i=0; i<matA_stride; i+=TILE_DIM) {
        int tilesize = TILE_DIM;
        if (i * TILE_DIM + TILE_DIM >= matA_stride) {
          tilesize = matA_stride - i * TILE_DIM;
        }
        tops::mdspan srcA(tops::Global, matA + offsetA + i, tilesize);
        tops::memcpy(ctx, bufA, srcA);
        //element-wise mul of one row of matB with one column of matB
        tops::mdspan srcB(tops::Global, matB + j * matA_stride + i, tilesize);
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
  tops::mdspan dst(tops::Global, out + offsetOut, out_stride);
  tops::memcpy(ctx, dst, bufOut);
}

int main(int argc, char *argv[])
{
    CHECK(topsInit(0));
    CHECK(topsSetDevice(1));
    
    int W = 14;
    int M = 128;
    int H = 32;
    const int MAX_RANK = 3;
    const int batch = 1;

    size_t size_lhs = W * M;
    size_t size_rhs = M * H;
    size_t size_out = W * H;
    int shape_lhs[MAX_RANK] = {batch, W, M};
    int shape_rhs[MAX_RANK] = {batch, H, M}; //transposed

    int threads = 4;
    if (W % 4 >0) {threads += 1;}

    int grids = W/4;
    if (grids < 1) {
        threads = W;
        grids = 1;
    }

        float *lhs_d, *rhs_d, *out_d;
        int *shape_lhs_d, *shape_rhs_d;
        float *lhs_h, *rhs_h, *out_h;

        CHECK(topsHostMalloc((float**)&lhs_h, batch * size_lhs * sizeof(float)));
        CHECK(topsHostMalloc((float**)&rhs_h, batch * size_rhs * sizeof(float)));
        CHECK(topsHostMalloc((float**)&out_h, batch * size_out * sizeof(float)));

        for (size_t b = 0; b < batch; b++) { 
            for (size_t i = 0; i < size_lhs; i++) {
                lhs_h[b * size_lhs + i] = 0.005f;
            }
            for (size_t i = 0; i < size_rhs; i++) {
                rhs_h[b * size_rhs + i] = 0.6f;
            }
            for (size_t i = 0; i < size_out; i++) {
                out_h[b * size_out + i] = 0.0;
            }
        }
        CHECK(topsMalloc(&lhs_d, batch * size_lhs * sizeof(float)));
        CHECK(topsMalloc(&rhs_d, batch * size_rhs * sizeof(float)));
        CHECK(topsMalloc(&out_d, batch * size_out * sizeof(float)));
        CHECK(topsMalloc(&shape_lhs_d, MAX_RANK * sizeof(int)));
        CHECK(topsMalloc(&shape_rhs_d, MAX_RANK * sizeof(int)));

        printf("info: copy Host2Device\n");
        CHECK(topsMemcpy(lhs_d, lhs_h, batch * size_lhs * sizeof(float),
                        topsMemcpyHostToDevice));
        CHECK(topsMemcpy(rhs_d, rhs_h, batch * size_rhs * sizeof(float),
                        topsMemcpyHostToDevice));
        CHECK(topsMemcpy(out_d, out_h, batch * size_out * sizeof(float),
                        topsMemcpyHostToDevice));

        CHECK(topsMemcpy(shape_lhs_d, &shape_lhs, MAX_RANK * sizeof(int),
                        topsMemcpyHostToDevice));

        CHECK(topsMemcpy(shape_rhs_d, &shape_rhs, MAX_RANK * sizeof(int),
                        topsMemcpyHostToDevice));

        transposed_matmul<<<dim3(grids, 1, 1), dim3(threads, 1, 1)>>>(lhs_d, rhs_d, out_d, shape_lhs_d, shape_rhs_d);

        printf("info: copy Device2Host\n");
        CHECK(topsMemcpy(out_h, out_d, batch * size_out * sizeof(float), topsMemcpyDeviceToHost));

        for (size_t b = 0; b < batch; b++) {
          for (size_t i = 0; i < W; i++) {
              for (size_t j = 0; j < H; j++) {
                  printf("%.2f,", out_h[b * size_out + i * H + j]);
              }
              printf("\n");
          }
          printf("\n\n");
        }
        CHECK(topsHostFree(lhs_h));
        CHECK(topsHostFree(rhs_h));
        CHECK(topsHostFree(out_h));
        topsFree(lhs_d);
        topsFree(rhs_d);
        topsFree(out_d);
        topsFree(shape_lhs_d);
        topsFree(shape_rhs_d);
        // topsDeviceReset();
        // topsStreamDestroy(stream);
        // topsModuleUnload(module);
        sleep(2);
    return 0;
}
