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

extern "C" __global__ void transpose_kernel(float *idata, float *odata, int* size)
{
  tops_dte_ctx_t ctx;
  tops::dte_scope s(ctx);

  __valigned__ int shape[MAX_RANK];
  copy_to_buffer<int, MAX_RANK>(ctx, size, shape);
  int rows = shape[1];
  int cols = shape[2];

  int GRIDS = cols/TILE_DIM;
  if (GRIDS * TILE_DIM < cols) GRIDS += 1;
  int BLOCKS = rows/TILE_DIM;
  if (BLOCKS * TILE_DIM < rows) BLOCKS += 1;

  int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int threadId = blockId * blockDim.x + threadIdx.x;

  int x = threadId % GRIDS;
  int y = threadId / GRIDS;
  __valigned__ float tile[TILE_DIM][TILE_DIM];
  
  for (int j = 0; j < TILE_DIM; j += 1) {
    if (y * TILE_DIM + j >= rows) break; 
    int bufsize = TILE_DIM;
    if ((x + 1) * TILE_DIM > cols) {
        bufsize = cols - x * TILE_DIM;
    }
    tops::mdspan src(tops::Global, idata + ((y * TILE_DIM) + j) * cols + x * TILE_DIM, bufsize);
    tops::mdspan buf(tops::Private, &tile[j], bufsize);
    tops::memcpy(ctx, buf, src);
  }


  // __valigned__ int tileShape[2] = {TILE_DIM, TILE_DIM};
  // tops::mdspan src(tops::Private, tile, tileShape);
  __valigned__ float matBbuffer[TILE_DIM][TILE_DIM];    
  for (int i=0; i<TILE_DIM; i++) {
    for(int j=0; j<TILE_DIM; j++) {
      matBbuffer[i][j] = tile[j][i];
    }
  }
  // tops::mdspan dst1(tops::Private, matBbuffer, tileShape);
  // layout parameter
  // __valigned__ int layout[2] = {1, 0};
  //transpose of matB to temp buffer (matBbuffer)
  // tops::transpose(ctx, dst1, src, layout);

  __syncthreads();
  x = threadId / GRIDS;
  y = threadId % GRIDS;
  
  for (int j = 0; j < TILE_DIM; j += 1) {
    int bufsize = TILE_DIM;
    if (y * TILE_DIM + j >= cols) break; 
    if ((x + 1) * TILE_DIM > rows) {
        bufsize = rows - x * TILE_DIM;
    }
    tops::mdspan src(tops::Private, &matBbuffer[j], bufsize);
    tops::mdspan dst(tops::Global, odata + (y * TILE_DIM + j) * rows + x * TILE_DIM, bufsize);
    tops::memcpy(ctx, dst, src);
  }
  // __syncthreads();
}

// void transpose(float *idata, float *odata, int* size, int M, int N)
// {
//   // tops_dte_ctx_t ctx;
//   // tops::dte_scope s(ctx);

//   // __valigned__ int shape[MAX_RANK];
//   // copy_to_buffer<int, MAX_RANK>(ctx, size, shape);
//   // int M = shape[0];
//   // int N = shape[1];

//   int GRIDS = N/TILE_DIM;
//   if (GRIDS * TILE_DIM < N) GRIDS += 1;
//   int BLOCKS = M/TILE_DIM;
//   if (BLOCKS * TILE_DIM < M) BLOCKS += 1;
//   int PER_BLOCKS = 1;
//   if (BLOCKS > 4) {
//     PER_BLOCKS = 4;
//     if ((BLOCKS / PER_BLOCKS) * 4 < BLOCKS) {
//       BLOCKS /= PER_BLOCKS;
//       BLOCKS += 1;
//     } else {
//       BLOCKS /= PER_BLOCKS;
//     }
//   }
//   printf("info: LaunchKernel {%d} {%d} {%d}\n", GRIDS, BLOCKS, PER_BLOCKS);

//   transpose_kernel<<<dim3(GRIDS, BLOCKS, 1), dim3(PER_BLOCKS, 1, 1)>>>(idata, odata, size);
//   // __syncthreads();
// }


int test_transpose(int M, int N)
{
    float *lhs_d, *rhs_d, *out_d;
    float *lhs_h, *rhs_h, *out_h;
    int *shape_lhs_d, *shape_rhs_d;
    // int M = 48;
    // int N = 128;
    const int batch = 1;

    size_t size_lhs = M * N;
    size_t size_out = N * M;
    int shape_lhs[MAX_RANK] = {batch, M, N};


    lhs_h = (float *)aligned_alloc(4096, batch * size_lhs * sizeof(float));
    CHECK(lhs_h == 0 ? topsErrorMemoryAllocation : topsSuccess);

    out_h = (float *)aligned_alloc(4096, batch * size_out * sizeof(float));
    CHECK(out_h == 0 ? topsErrorMemoryAllocation : topsSuccess);
    
    for (size_t b = 0; b < batch; b++) { 
      for (size_t i = 0; i < size_lhs; i++) {
          lhs_h[b * size_lhs + i] = i * 1.0;
      }

      for (size_t i = 0; i < size_out; i++) {
          out_h[b * size_out + i] = 0.0;
      }
    }

    CHECK(topsMalloc(&lhs_d, batch * size_lhs * sizeof(float)));
    CHECK(topsMalloc(&out_d, batch * size_out * sizeof(float)));
    CHECK(topsMalloc(&shape_lhs_d, MAX_RANK * sizeof(int)));

    printf("info: copy Host2Device\n");
    CHECK(topsMemcpy(lhs_d, lhs_h, batch * size_lhs * sizeof(float),
                    topsMemcpyHostToDevice));

    CHECK(topsMemcpy(out_d, out_h, batch * size_out * sizeof(float),
                    topsMemcpyHostToDevice));

    CHECK(topsMemcpy(shape_lhs_d, &shape_lhs, MAX_RANK * sizeof(int),
                    topsMemcpyHostToDevice));

    int GRIDS = N/TILE_DIM;
    if (GRIDS * TILE_DIM < N) GRIDS += 1;
    int BLOCKS = M/TILE_DIM;
    if (BLOCKS * TILE_DIM < M) BLOCKS += 1;
    int PER_BLOCKS = 1;
    if (BLOCKS > 4) {
      PER_BLOCKS = 4;
      if ((BLOCKS / PER_BLOCKS) * 4 < BLOCKS) {
        BLOCKS /= PER_BLOCKS;
        BLOCKS += 1;
      } else {
        BLOCKS /= PER_BLOCKS;
      }
    }
    printf("info: topsModuleLaunchKernel {%d} {%d} {%d}\n", GRIDS, BLOCKS, PER_BLOCKS);

    transpose_kernel<<<dim3(GRIDS, BLOCKS, 1), dim3(PER_BLOCKS, 1, 1)>>>(lhs_d, out_d, shape_lhs_d);

    printf("info: copy Device2Host\n");
    CHECK(topsMemcpy(out_h, out_d, batch * size_out * sizeof(float), topsMemcpyDeviceToHost));

    printf("print results\n");
    topsDeviceSynchronize();
    // for (size_t i = 0; i < N; i++) {
    //       for (size_t j = 0; j < M; j++) {
    //           printf("%.2f,", out_h[i * M + j]);
    //       }
    //       printf("\n\r\n");
    // }

    topsHostFree(lhs_h);
    topsHostFree(out_h);

    for (int i=0; i< 20; i++) {
      topsFree(lhs_d);
      // topsFree(rhs_d);
      topsFree(out_d);
      topsFree(shape_lhs_d);
      // topsFree(shape_rhs_d);
    }



    // free(lhs_h);
    // free(rhs_h);
    // free(out_h);

    return 0;
}

int main(int argc, char *argv[])
{
  for (int i=0; i< 1; i++) 
    test_transpose(4096, 11008);
}