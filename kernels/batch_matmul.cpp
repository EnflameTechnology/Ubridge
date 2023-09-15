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
constexpr int MAX_DIM = 512; //maximum matmul input size: 512 x 512
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
extern "C" __global__ void batch_matmul(float *matA, float *matB, float* out, int* matA_shape, int* matB_shape)
{
  
  tops_dte_ctx_t ctx;
  tops::dte_scope s(ctx);

  __valigned__ int buffer_shapeA[MAX_RANK];
  __valigned__ int buffer_shapeB[MAX_RANK];
  copy_to_buffer<int, MAX_RANK>(ctx, matA_shape, buffer_shapeA);
  copy_to_buffer<int, MAX_RANK>(ctx, matB_shape, buffer_shapeB);
  int batch = buffer_shapeA[0];
  int matA_stride = buffer_shapeA[1] * buffer_shapeA[2];
  int matB_stride = buffer_shapeB[1] * buffer_shapeB[2];
  int out_stride = buffer_shapeA[1] * buffer_shapeB[2];

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= batch) 
  {
    return; 
  }

  int offsetA = tid * matA_stride;
  int offsetB = tid * matB_stride;
  int offsetOut = tid * out_stride;

  int nmatB = matA_stride;
  int ncolumnsB = buffer_shapeB[2];
  int nmatA = matB_stride;
  int ncolumns = buffer_shapeA[2];

  __valigned__ int buffer_shapeB_[MAX_RANK-1];
  buffer_shapeB_[0] = buffer_shapeB[1];
  buffer_shapeB_[1] = buffer_shapeB[2];

  tops::mdspan src(tops::Global, matB + offsetB, buffer_shapeB_);

  __valigned__ int matB_output_shape[2] = {buffer_shapeB_[1], buffer_shapeB_[0]};  

  //size of matBbuffer will impact the maximum input dimension of matB
  __shared__ float matBbuffer[MAX_DIM*MAX_DIM];           
  tops::mdspan dst(tops::Global, matBbuffer, matB_output_shape);
  
  // layout parameter
  __valigned__ int matB_output_layout[2] = {1, 0};

  //transpose of matB to temp buffer (matBbuffer)
  tops::transpose(ctx, dst, src, matB_output_layout);

  __valigned__ float bufferA[MAX_DIM];
  __valigned__ float bufferB[MAX_DIM];
  __valigned__ float bufferMul[MAX_DIM];
  __valigned__ float bufferOut[MAX_DIM];

  tops::mdspan bufA(tops::Private, &bufferA, ncolumns);
  tops::mdspan bufB(tops::Private, &bufferB, ncolumns);
  tops::mdspan bufMul(tops::Private, &bufferMul, ncolumns);
  tops::mdspan bufOut(tops::Private, &bufferOut, ncolumnsB);

  //for each row of matA
  for (int i=0; i<nmatA; i+=ncolumns) {
    tops::mdspan srcA(tops::Global, matA + offsetA +i, ncolumns);
    tops::memcpy(ctx, bufA, srcA);

    for (int j=0; j<nmatB; j+=ncolumns) {
      //element-wise mul of one row of matB with one column of matB
        tops::mdspan srcB(tops::Global, matBbuffer+j, ncolumns);
        tops::memcpy(ctx, bufB, srcB);
        const auto &v1_ = tops::vload<vfloat>(bufferA);
        const auto &v2_ = tops::vload<vfloat>(bufferB);
        tops::vstore(tops::vmul<vfloat>(v1_, v2_), bufferMul);
        int line = j / ncolumns;

        //sum of mul results
        float sums = 0.0;
        for (int m=0; m<ncolumns; m++){
          sums += bufferMul[m];
        }
        bufferOut[line] = sums;
    }
    //copy to results
    tops::mdspan dst(tops::Global, out + offsetOut + (i / ncolumns) * ncolumnsB, ncolumnsB);
    tops::memcpy(ctx, dst, bufOut);
  }
  
}


int main(int argc, char *argv[])
{
    std::string kernel_path = "./batch_matmul.cpp-tops-dtu-enflame-tops.topsfb";

    topsModule_t module;
    topsFunction_t kernel;

    CHECK(topsModuleLoad(&module, kernel_path.c_str()));
    CHECK(topsModuleGetFunction(&kernel, module, "batch_matmul"));

    float *lhs_d, *rhs_d, *out_d;
    float *lhs_h, *rhs_h, *out_h;
    int *shape_lhs_d, *shape_rhs_d;
    int W = 16;
    int M = 10;
    int H = 24;
    const int MAX_RANK = 3;
    const int batch = 1;

    size_t size_lhs = W * M;
    size_t size_rhs = M * H;
    size_t size_out = W * H;
    int shape_lhs[MAX_RANK] = {batch, W, M};
    int shape_rhs[MAX_RANK] = {batch, M, H};


    lhs_h = (float *)aligned_alloc(4096, batch * size_lhs * sizeof(float));
    CHECK(lhs_h == 0 ? topsErrorMemoryAllocation : topsSuccess);
    rhs_h = (float *)aligned_alloc(4096, batch * size_rhs * sizeof(float));
    CHECK(rhs_h == 0 ? topsErrorMemoryAllocation : topsSuccess);
    out_h = (float *)aligned_alloc(4096, batch * size_out * sizeof(float));
    CHECK(out_h == 0 ? topsErrorMemoryAllocation : topsSuccess);
    
    for (size_t b = 0; b < batch; b++) { 
      for (size_t i = 0; i < size_lhs; i++) {
          lhs_h[b * size_lhs + i] = i * 0.001;
      }
      for (size_t i = 0; i < size_rhs; i++) {
          rhs_h[b * size_rhs + i] = i * 0.001;
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
    const unsigned blocks = 1;
    const unsigned threadsPerBlock = 1;

    struct {
        topsDeviceptr_t lhs_;
        topsDeviceptr_t rhs_;
        topsDeviceptr_t out_;
        int *shape_lhs_;
        int *shape_rhs_;
    } args{lhs_d, rhs_d, out_d, shape_lhs_d, shape_rhs_d};

    auto size = sizeof(args);
    void *config[] = {TOPS_LAUNCH_PARAM_BUFFER_POINTER, &args,
                        TOPS_LAUNCH_PARAM_BUFFER_SIZE, &size,
                        TOPS_LAUNCH_PARAM_END};

    // topsModuleLaunchKernel(kernel, W/4, 1, 1, 4, 1, 1, 0, nullptr, nullptr, config);
    topsModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, config);

    printf("info: copy Device2Host\n");
    CHECK(topsMemcpy(out_h, out_d, batch * size_out * sizeof(float), topsMemcpyDeviceToHost));

    printf("info: pring results\n");
    for (size_t b = 0; b < batch; b++) {
      for (size_t i = 0; i < M; i++) {
          for (size_t j = 0; j < H; j++) {
              printf("%.2f,", out_h[b * size_out + i * H + j]);
          }
          printf("\n");
      }
      printf("\n\n");
    }

    return 0;
}
