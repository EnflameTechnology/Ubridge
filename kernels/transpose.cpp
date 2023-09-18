#include "tops.h"
#include <stdio.h>
#include <tops/tops_runtime.h>
#include <tops/topsrtc.h>
#include <tops/half.h>
#include <tops/bfloat.h>

#define CHECK(cmd) \
{\
    topsError_t error  = cmd;\
    if (error != topsSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", topsGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
	  }\
}
constexpr int TILE_DIM=64;

//fast transpose without large temp buffer
template<typename T>
__device__ void transpose(const size_t rows, const size_t cols, T *idata, T *odata)
{
  tops_dte_ctx_t ctx;
  tops::dte_scope s(ctx);
  // printf("Rows %d, Cols %d", rows, cols);
  
  int GRIDS = cols/TILE_DIM;
  if (GRIDS * TILE_DIM < cols) GRIDS += 1;
  int BLOCKS = rows/TILE_DIM;
  if (BLOCKS * TILE_DIM < rows) BLOCKS += 1;

  int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int threadId = blockId * blockDim.x + threadIdx.x;

  int x = threadId % GRIDS;
  int y = threadId / GRIDS;
  __valigned__ T tile[TILE_DIM][TILE_DIM];
  
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

  __valigned__ T matBbuffer[TILE_DIM][TILE_DIM];    
  for (int i=0; i<TILE_DIM; i++) {
    for(int j=0; j<TILE_DIM; j++) {
      matBbuffer[i][j] = tile[j][i];
    }
  }
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

extern "C"  __global__ void transpose_f32(const size_t rows, const size_t cols, float *idata, float *odata)
{
    transpose<float>(rows, cols, idata, odata);

}

extern "C"  __global__ void transpose_f16(const size_t rows, const size_t cols, tops::half *idata, tops::half *odata)
{
    transpose<tops::half>(rows, cols, idata, odata);

}

extern "C"  __global__ void transpose_bf16(const size_t rows, const size_t cols, tops::bfloat *idata, tops::bfloat *odata)
{
    transpose<tops::bfloat>(rows, cols, idata, odata);

}

int main(int argc, char *argv[])
{
    return 0;
}