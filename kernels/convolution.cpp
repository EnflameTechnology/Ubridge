#include <stdio.h>
#include <tops/tops_runtime.h>
#include <tops/topsrtc.h>

constexpr int BUF_SIZE = 1024 * 10;
constexpr int MAX_RANK = 4;
constexpr int MAX_DIM = 15; //max kernel size 15 x 15

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

// lhs (input tensorl), rhs (kernel tensor), out (output tensor), matShape ([width, height, 1, 1]), kernelShape ([width, height, 1, 1])
// channelInfo ([samples, input_channel, output_channel, stride])
extern "C" __attribute__((global)) void 
convolution(float* lhs, float* rhs, float* out, int* matShape, int* kernelShape, int* channelInfo) 
{

  tops_dte_ctx_t ctx;
  tops::dte_scope s(ctx);

  __valigned__ int mat_shape[MAX_RANK];
  copy_to_buffer<int, MAX_RANK>(ctx, matShape, mat_shape);
  int W = mat_shape[0];
  int H = mat_shape[1];

  __valigned__ int kernel_shape[MAX_RANK];
  copy_to_buffer<int, MAX_RANK>(ctx, kernelShape, kernel_shape);
  int W_k = kernel_shape[0];
  int H_k = kernel_shape[1];

  __valigned__ int channel_info[MAX_RANK];
  copy_to_buffer<int, MAX_RANK>(ctx, channelInfo, channel_info);
  int samples = channel_info[0];
  int input_channel = channel_info[1];
  int output_channel = channel_info[2];
  int stride = channel_info[3];
  
  int nW = W - W_k + 1;
  int nH = H - H_k + 1;

  //int lsize = samples * W * H * input_channel;
  int ksize = W_k * H_k * input_channel * output_channel;
  //int osize = samples * (W - W_k + 1) * (H - H_k + 1) * output_channel;

  __valigned__ float buffer_rhs[MAX_DIM*MAX_DIM];
  __valigned__ float buffer_lhs[MAX_DIM*MAX_DIM];
  __valigned__ float bufferMul[MAX_DIM*MAX_DIM];
  __valigned__ float bufferOut[MAX_DIM*MAX_DIM];
  
  tops::mdspan src(tops::Global, rhs, ksize);
  tops::mdspan buf_rhs(tops::Private, &buffer_rhs, ksize);
  tops::memcpy(ctx, buf_rhs, src);

  tops::mdspan bufOut(tops::Private, &bufferOut, nW);
  tops::mdspan bufMul(tops::Private, &bufferMul, ksize);

  for (int i=0; i<nH; i+=stride) {
    for (int j=0; j<nW; j+=stride) {
      int idx = i * W + j;

      //img2col of a patch
      for (int k=0; k<H_k; k++) {
        tops::mdspan src(tops::Global, lhs + idx + k * W, W_k);
        tops::mdspan buf_lhs(tops::Private, (float*)&buffer_lhs + k * W_k, W_k);
        tops::memcpy(ctx, buf_lhs, src);
      }

      //perform vector mul and sum
      const auto &v1_ = tops::vload<vfloat>(buffer_lhs);
      const auto &v2_ = tops::vload<vfloat>(buffer_rhs);
      tops::vstore(tops::vmul<vfloat>(v1_, v2_), bufferMul);

      //sum of mul results
      float sums = 0.0;
      for (int m=0; m<ksize; m++){
        sums += bufferMul[m];
      }
      bufferOut[j] = sums;
    }
   
    //copy to results
    tops::mdspan dst(tops::Global, out + i * nW, nW);
    tops::memcpy(ctx, dst, bufOut);
    
  }
  
}


int convolution_cpp(float* lhs, float* rhs, float* out, int* matShape, int* kernelShape, int* channelInfo) {
    convolution<<<dim3(1,1,1), dim3(1,1,1)>>>(lhs, rhs, out, matShape, kernelShape, channelInfo);
    return 0;
}

int main() {
    return 0;

}