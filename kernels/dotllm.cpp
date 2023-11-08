#include <tops.h>
#include <tops/half.h>
#include <tops/bfloat.h>
#include <krt/scalar.h>

#include <tops/tops_runtime.h>

#if __GCU_ARCH__ < 300 
#include "sip20intrin.h"
#define MAX_SIP_NUM 6
#endif

#if __GCU_ARCH__ >= 300 
#include "sip30intrin.h"
#define MAX_SIP_NUM 12
#endif

#pragma clang force_cuda_host_device begin
#include <stdio.h>
#pragma clang force_cuda_host_device end
#include "dot_core_kernels.h"


template <typename T, typename VT, FP dot_intrinsic>
__device__ void dot(
  T *lhs,
  T *rhs,
  T *out,
  int m,
  int k,
  int n) {
  constexpr int vlen = tops::hvlength<VT>();
  constexpr int tile_size = 1 * vlen;

  int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int threadId = blockId * blockDim.x + threadIdx.x;

  int lstride = tile_size;
  if (m < tile_size) {
    lstride = m;
  }

  int blockIndex = threadId / (n / tile_size);
  int threadIndex = threadId % (n / tile_size);
//   printf("blockIndex %d, threadIndex %d", blockIndex, threadIndex);

  __valigned__ T lhs_l1[lstride * tile_size];
  __valigned__ T rhs_l1[tile_size * tile_size];

  __valigned__ T out_l1[lstride * tile_size];
  __valigned__ T temp[lstride * tile_size];

  tops::mdspan out_l1_(out_l1, lstride, tile_size);

  tops::mdspan srcl_l1(lhs_l1, lstride, tile_size);
  tops::mdspan srcr_l1(rhs_l1, tile_size, tile_size);

  tops::mdspan srcl_l3(lhs, m, k);
  tops::mdspan srcr_l3(rhs, k, n);
  tops::mdspan dst_l3(out, m, n);

  tops_dte_ctx_t ctx;   //L1-L3
  tops::dte_scope s(ctx);

  int idx_y = blockIndex * lstride;
  int idx_x = threadIndex * tile_size;

  if (idx_y < m) {
    int offsets_l[] = {idx_y, 0};
    if (idx_x < n) {
     tops::memset<T>(ctx, out_l1_, T(0));
      tops::mdspan dst_l1(out_l1, lstride, tile_size);
      int offsets_r[] = {0, idx_x};

      for (int i = 0; i < k/tile_size; i++) {
        offsets_l[1] = i * tile_size;
        tops::slice(ctx, srcl_l1, srcl_l3, offsets_l);
        offsets_r[0] = i * tile_size;
        tops::slice(ctx, srcr_l1, srcr_l3, offsets_r);
        // //dot_no_transpose
        auto lhs_address = (__attribute__((address_space(5))) T *)(lhs_l1);
        auto rhs_address = (__attribute__((address_space(5))) T *)(rhs_l1);
        auto out_address = (__attribute__((address_space(5))) T *)(temp);

        dot_intrinsic(reinterpret_cast<long long>(lhs_address),
                       reinterpret_cast<long long>(rhs_address),
                       reinterpret_cast<long long>(out_address),
                       lstride,
                       tile_size,
                       tile_size
                       );

        for (auto i = 0; i < lstride * tile_size; i++) {
          out_l1[i] += temp[i];
        }
      }
      //L1->L3
      int offsets_o[] = {idx_y, idx_x};
      tops::deslice(ctx, dst_l3, dst_l1, offsets_o);
    } 
  } 
}


extern "C" __global__ void dotllm_f16(const size_t m, const size_t k, const size_t n, tops::half *matA, tops::half *matB, tops::half* out)
{
  if (m < 16) {
    dot<tops::half, vhalf, kernel_dot_batch_m_lt32_fp16>(matA, matB, out, m, k, n);
  } else {
    dot<tops::half, vhalf, kernel_dot_m_le256_fp16>(matA, matB, out, m, k, n);
  }
}

extern "C" __global__ void dotllm_bf16(const size_t m, const size_t k, const size_t n, tops::bfloat *matA, tops::bfloat *matB, tops::bfloat* out)
{
    // dot<tops::bfloat, vbfloat, kernel_dot_m_le256_fp16>(matA, matB, out, m, k, n);
}

extern "C" __global__ void dotllm_f32(const size_t m, const size_t k, const size_t n, float *matA, float *matB, float* out)
{
    if (m < 16) { 
      dot<float, vfloat, kernel_dot_batch_m_lt32_outfp32>(matA, matB, out, m, k, n);
    } else {
      dot<float, vfloat, kernel_dot_m_le256_outfp32>(matA, matB, out, m, k, n);
    }
}

int test(size_t M, size_t K, size_t N, bool check) {
  const int vlen = tops::hvlength<vhalf>();
  printf("tile size %d\n", vlen);

  const int tile_size = 1 * vlen;
  int gridsz = M / tile_size;
  if (gridsz < 1) {
    gridsz = 1;
  }
  int blocksz = N / tile_size;
  int perthreads = MAX_SIP_NUM;
  if (blocksz > MAX_SIP_NUM) {
    blocksz /= MAX_SIP_NUM;
  } else {
    perthreads = 1;
  }

  DATA<tops::half> data(M, K, N, tile_size, check);

  float time = 0.0;
  topsEvent_t start, stop;

  CHECK(topsEventCreate(&start));
  CHECK(topsEventCreate(&stop));

  CHECK(topsEventRecord(start));
  printf("Kernel launch... [%d, %d, %d]\n", gridsz, blocksz, perthreads);

   dotllm_f16
      <<<dim3(gridsz, blocksz, 1), dim3(perthreads, 1, 1)>>>( M,
                            K,
                            N,
                            data.lhs_d,
                            data.rhs_d,
                            data.out_d
                           );

  CHECK(topsGetLastError());

  CHECK(topsEventRecord(stop));
  CHECK(topsEventSynchronize(stop));
  CHECK(topsEventElapsedTime(&time, start, stop));
  printf("Time taken: %g ms, Shape: %d %d %d --------\n", time, M, K, N);

  CHECK(topsGetLastError());
  CHECK(topsMemcpy(data.out_h, data.out_d, data.size_out*sizeof(tops::half),
    topsMemcpyDeviceToHost));
  
  //CPU/GPU check_data
  if (check) { 
    printf("Compare with CPU data...\n");
    check_data<tops::half>(data.out_h, data.expected, M, K, N);
  }


  printf("intrinsic_fp16_kernel throughput is %8.2f GFLOPS\n",
   (2LL*M*K*N)/time/1000000000LL);
  return 0;
}

int main() {
  size_t M = 13;
  size_t K = 4096;
  size_t N = 11008;


  for (int i=0; i< 10; i++) {
    test(13, 4096, 4096, false);

    test(M, K, N, false);
  }
  return 0;
}
