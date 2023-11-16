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

extern "C" __global__ void memory_kernel(float *idata, float *odata, int* size)
{
  // tops_dte_ctx_t ctx;
  // tops::dte_scope s(ctx);

}


int test_memory(int M, int N)
{
    float *lhs_d, *rhs_d, *out_d;
    float *lhs_h, *rhs_h, *out_h;
    int *shape_lhs_d, *shape_rhs_d;
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

    memory_kernel<<<dim3(GRIDS, BLOCKS, 1), dim3(PER_BLOCKS, 1, 1)>>>(lhs_d, out_d, shape_lhs_d);

    printf("info: copy Device2Host\n");
    CHECK(topsMemcpy(out_h, out_d, batch * size_out * sizeof(float), topsMemcpyDeviceToHost));

    topsDeviceSynchronize();
    printf("info: sync\n");

    topsFree(lhs_d);
    topsFree(out_d);
    topsFree(shape_lhs_d);

    free(lhs_h);
    free(out_h);
    return 0;
}

int main(int argc, char *argv[])
{
  for (int i=0; i< 100; i++) 
  {
    test_memory(4096, 11008);
    printf("\r\n----------Iteration %d finished\r\n", i);
  }  
}