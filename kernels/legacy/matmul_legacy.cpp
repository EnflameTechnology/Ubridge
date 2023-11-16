#include "tops.h"
#pragma clang force_cuda_host_device begin
#include <stdio.h>
#pragma clang force_cuda_host_device end

constexpr int MAX_RANK = 4;
constexpr int MAX_DIM = 512; //maximum matmul input size: 512 x 512

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

extern "C" __global__ void matmul(float *matA, float *matB, float* out, int* matA_shape, int* matB_shape)
{
  
  tops_dte_ctx_t ctx;
  tops::dte_scope s(ctx);

  __valigned__ int buffer_shapeB[MAX_RANK];
  
  copy_to_buffer<int, MAX_RANK>(ctx, matB_shape, buffer_shapeB);
  int nmatB = buffer_shapeB[0] * buffer_shapeB[1];
  int ncolumnsB = buffer_shapeB[1];

  tops::mdspan src(tops::Global, matB, buffer_shapeB);

  __valigned__ int matB_output_shape[4] = {buffer_shapeB[1], buffer_shapeB[0], buffer_shapeB[2], buffer_shapeB[3]};  

  //size of matBbuffer will impact the maximum input dimension of matB
  __shared__ float matBbuffer[MAX_DIM*MAX_DIM];           
  tops::mdspan dst(tops::Global, matBbuffer, matB_output_shape);
  
  // layout parameter
  __valigned__ int matB_output_layout[4] = {1, 0, 2, 3};

  //transpose of matB to temp buffer (matBbuffer)
  tops::transpose(ctx, dst, src, matB_output_layout);

  __valigned__ int buffer_shapeA[MAX_RANK];
  copy_to_buffer<int, MAX_RANK>(ctx, matA_shape, buffer_shapeA);
  int nmatA = buffer_shapeA[0] * buffer_shapeA[1];

  __valigned__ float bufferA[MAX_DIM];
  __valigned__ float bufferB[MAX_DIM];
  __valigned__ float bufferMul[MAX_DIM];
  __valigned__ float bufferOut[MAX_DIM];

  int ncolumns = buffer_shapeA[1];

  tops::mdspan bufA(tops::Private, &bufferA, ncolumns);
  tops::mdspan bufB(tops::Private, &bufferB, ncolumns);
  tops::mdspan bufMul(tops::Private, &bufferMul, ncolumns);
  tops::mdspan bufOut(tops::Private, &bufferOut, ncolumnsB);

  //for each row of matA
  for (int i=0; i<nmatA; i+=ncolumns) {
    tops::mdspan srcA(tops::Global, matA+i, ncolumns);
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
    tops::mdspan dst(tops::Global, out + (i / ncolumns) * ncolumnsB, ncolumnsB);
    tops::memcpy(ctx, dst, bufOut);
  }
  
}
