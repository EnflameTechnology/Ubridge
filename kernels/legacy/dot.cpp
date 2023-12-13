#include "tops.h"
#include <stdio.h>
#include <tops/tops_runtime.h>
#include <tops/topsrtc.h>
#include <unistd.h>
#include <tops/half.h>
#include <tops/bfloat.h>

constexpr int TILE_DIM=64;

#define CHECK(cmd) \
{\
    topsError_t error  = cmd;\
    if (error != topsSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", topsGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
	  }\
}

namespace tops {
template <typename T>
__device__ __host__ __forceinline__ constexpr int hvlength() {
  return 128 / sizeof(T);
}

} // namespace tops


//matA_shape [1, m, k], tranasposed matB_shape[1, n, k], out shape [1, m, n]
template<typename T, typename VT>
__device__ void dot(const size_t m, const size_t k, const size_t n, T *matA, T *matB, T* out)
{
  tops_dte_ctx_t ctx;
  tops::dte_scope s(ctx);

  int splits = m;
  int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int tid = blockId * blockDim.x + threadIdx.x;
  if (tid >= splits) {
    __syncthreads();
    return;
  }

  int matA_stride = k;
  int out_stride = n;


  int offsetA = tid * matA_stride;
  int offsetOut = tid * out_stride;

  __valigned__ T bufferA[TILE_DIM];
  __valigned__ T bufferB[TILE_DIM];
  __valigned__ T bufferMul[TILE_DIM];
  __valigned__ T bufferOut[TILE_DIM*192];
  __valigned__ T SUM[tops::hvlength<VT>()];

  tops::mdspan bufA(tops::Private, &bufferA, TILE_DIM);
  tops::mdspan bufB(tops::Private, &bufferB, TILE_DIM);
  tops::mdspan bufMul(tops::Private, &bufferMul, TILE_DIM);
  tops::mdspan bufOut(tops::Private, &bufferOut, out_stride);

  for (int j=0; j<out_stride; j+=1) {
      //sum of mul results
      tops::vstore(tops::vzero<VT>(), SUM);
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
        const auto &v1_ = tops::vload<VT>(bufferA);
        const auto &v2_ = tops::vload<VT>(bufferB);
        tops::vstore(tops::vmul<VT>(v1_, v2_), bufferMul);
        for (int m=0; m<tilesize; m++){
          SUM[0] += bufferMul[m];
        }
      }
      bufferOut[j] = SUM[0];
      // printf("%.2f, ", sums);

  }
  // printf("\n");
  __syncthreads();
//   printf("Copy output from offset {%d}\n", offsetOut);
  tops::mdspan dst(tops::Global, out + offsetOut, out_stride);
  tops::memcpy(ctx, dst, bufOut);
}

extern "C" __global__ void dot_f16(const size_t m, const size_t k, const size_t n, tops::half *matA, tops::half *matB, tops::half* out)
{
    dot<tops::half, vhalf>(m, k, n, matA, matB, out);

}

extern "C" __global__ void dot_bf16(const size_t m, const size_t k, const size_t n, tops::bfloat *matA, tops::bfloat *matB, tops::bfloat* out)
{
    dot<tops::bfloat, vbfloat>(m, k, n, matA, matB, out);
}

extern "C" __global__ void dot_f32(const size_t m, const size_t k, const size_t n, float *matA, float *matB, float* out)
{
    dot<float, vfloat>(m, k, n, matA, matB, out);
}

int main(int argc, char *argv[])
{
    // CHECK(topsInit(0));
    // CHECK(topsSetDevice(1));
    
    int W = 14;
    int M = 4096;
    int H = 512;
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

        dot_f32<<<dim3(grids, 1, 1), dim3(threads, 1, 1)>>>(W, M, H, lhs_d, rhs_d, out_d);

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
        // sleep(2);
    return 0;
}
