#include "tops.h"
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

//fast transpose without large temp buffer
extern "C" __global__ void transpose(const size_t rows, const size_t cols, float *idata, float *odata)
{
  tops_dte_ctx_t ctx;
  tops::dte_scope s(ctx);

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

  __valigned__ float matBbuffer[TILE_DIM][TILE_DIM];    
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


int main(int argc, char *argv[])
{
    return 0;
}