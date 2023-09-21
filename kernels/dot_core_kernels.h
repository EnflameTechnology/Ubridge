#pragma once
#include <tops.h>
#include <tops/half.h>
#include <tops/bfloat.h>
#include <krt/scalar.h>

#include <tops/tops_runtime.h>
#include "sip20intrin.h"
#pragma clang force_cuda_host_device begin
#include <stdio.h>
#pragma clang force_cuda_host_device end

constexpr int MAX_RANK = 4;
constexpr int MAX_PAVO_CLUSTER_NUM = 4;
constexpr int MAX_PAVO_SIP_NUM = 6;
constexpr int MAX_DRD_CLUSTER_NUM = 2;
constexpr int MAX_DRD_SIP_NUM = 12;
constexpr int MAX_SCP_CLUSTER_NUM = 1;
constexpr int MAX_SCP_SIP_NUM = 12;
constexpr int SIP_VECTOR_LENGTH = 128;


template <typename T>
void check_data(T* d_result, T* h_result, int m, int k, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (d_result[i*n + j] > h_result[i*n + j]) {
        assert(
          d_result[i*n + j] - h_result[i*n + j] <=  0.05 * h_result[i*n + j]);
      } else {
        assert(
          h_result[i*n + j] - d_result[i*n + j] <=  0.05 * h_result[i*n + j]);
      }
    }
  }
}


// struct memref {char *addr; int offset;};
#define CHECK(cmd) \
{\
    topsError_t error  = cmd;\
    if (error != topsSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", topsGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
	  }\
}


__device__ __forceinline__
auto get_index() {
    std::size_t blockIndex = blockIdx.z*(gridDim.x*gridDim.y)
        + blockIdx.y*gridDim.x + blockIdx.x;
    std::size_t threadIndex = threadIdx.z*(blockDim.x*blockDim.y)
        + threadIdx.y*blockDim.x + threadIdx.x;
    return blockIndex*(blockDim.x*blockDim.y*blockDim.z) + threadIndex;
}

__device__ __forceinline__
auto get_threadIndex() {
    std::size_t threadIndex = threadIdx.z*(blockDim.x*blockDim.y)
        + threadIdx.y*blockDim.x + threadIdx.x;
    return threadIndex;
}

namespace tops {

  template<typename T>
  static __device__ __host__ __forceinline__
  int hvlength();

  template<> __device__ __host__ __forceinline__
  constexpr int hvlength<vchar>() { return 128; }

  template<> __device__ __host__ __forceinline__
  constexpr int hvlength<vuchar>() { return 128; }

  template<> __device__ __host__ __forceinline__
  constexpr int hvlength<vshort>() { return 64; }

  template<> __device__ __host__ __forceinline__
  constexpr int hvlength<vushort>() { return 64; }

  template<> __device__ __host__ __forceinline__
  constexpr int hvlength<vint>() { return 32; }

  template<> __device__ __host__ __forceinline__
  constexpr int hvlength<vuint>() { return 32; }

  template<> __device__ __host__ __forceinline__
  constexpr int hvlength<vhalf>() { return 64; }

  template<> __device__ __host__ __forceinline__
  constexpr int hvlength<vbfloat>() { return 64; }

  template<> __device__ __host__ __forceinline__
  constexpr int hvlength<vfloat>() { return 32; }
} // namespace tops
template <typename T>
struct DATA {
  T *lhs_d, *rhs_d, *out_d;
  T *lhs_h, *rhs_h, *out_h;
  T *expected;
  bool check;

  size_t size_lhs, size_rhs, size_out;

  explicit DATA(int m, int k, int n, int tile_size=64, bool check=false) {
    this->check = check;
    size_lhs = m * k;
    size_rhs = k * n;
    size_out = m * n;
    printf("Prepare data for shape: %d, %d, %d...\n", m, k, n);
    lhs_h = reinterpret_cast<T *>(aligned_alloc(4096,
                                      size_lhs * sizeof(T)));
    rhs_h = reinterpret_cast<T *>(aligned_alloc(4096,
                                      size_rhs * sizeof(T)));
    out_h = reinterpret_cast<T *>(aligned_alloc(4096,
                                      size_out * sizeof(T)));
    if (check) {
        expected = reinterpret_cast<T *>(aligned_alloc(4096,
                                        size_out * sizeof(T)));
    }
    for (size_t i = 0; i < size_lhs; i++) {
      lhs_h[i] = (T)0.1;
      if (lhs_h[i] == (T)0) {
        if (i%2) {lhs_h[i] = (T)1;} else {
        lhs_h[i] = (T)(-1);}
      }
    }
    for (size_t i = 0; i < size_rhs; i++) {
      rhs_h[i] = (T)0.1;
      if (rhs_h[i] == (T)0) {
        rhs_h[i] = (T)1;
      }
    }
    if (check) {
        printf("Compute CPU Results...\n");
        // //CPU results
        for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            expected[i * n + j] = (T)0;
            for (int v = 0; v < k/tile_size; v++) {
            T sum = (T)0;
            for (int c = 0; c < tile_size; c++) {
                sum += lhs_h[i * k + v * tile_size + c] *
                        rhs_h[(v * tile_size + c) * n + j];
            }
            expected[i * n + j] += sum;
            }
        }
        }
    }

    printf("Preprare device memory...\n");

    topsMalloc(&lhs_d, size_lhs * sizeof(T));
    topsMalloc(&rhs_d, size_rhs * sizeof(T));
    topsMalloc(&out_d, size_out * sizeof(T));

    topsMemcpy(lhs_d, lhs_h, size_lhs * sizeof(T), topsMemcpyHostToDevice);
    topsMemcpy(rhs_d, rhs_h, size_rhs * sizeof(T), topsMemcpyHostToDevice);
  }

  ~DATA() {
    topsFree(lhs_d);
    topsFree(rhs_d);
    topsFree(out_d);

    free(lhs_h);
    free(rhs_h);
    free(out_h);
    if(this->check)
        free(expected);
  }
};


__device__ __forceinline__ void dot_general_kernel_rhs_parallel_no_reduce_f16(
                                                    int lhs_addr,
                                                    int rhs_addr,
                                                    int out_addr,
                                                    int M,
                                                    int K,
                                                    int N,
                                                    int reduce_index,
                                                    int reduce_cnt)
    __attribute__((no_mem_alias_in_tar, loop_iterator_less_than_1024)) {
  smr_t smr;
  v32f16 vr_rhs0, vr_rhs1, vr_rhs2, vr_rhs3, vr_rhs4, vr_rhs5, vr_rhs6, vr_rhs7,
      vr_rhs8, vr_rhs9, vr_rhs10, vr_rhs11, vr_rhs12, vr_rhs13, vr_rhs14,
      vr_rhs15, vr_rhs16, vr_rhs17, vr_rhs18, vr_rhs19, vr_rhs20, vr_rhs21,
      vr_rhs22, vr_rhs23, vr_rhs24, vr_rhs25, vr_rhs26, vr_rhs27, vr_rhs28,
      vr_rhs29, vr_rhs30, vr_rhs31;
  v32f16 vr_lhs0, vr_lhs1, vr_lhs2, vr_lhs3, vr_lhs4, vr_lhs5, vr_lhs6, vr_lhs7,
      vr_lhs8, vr_lhs9, vr_lhs10, vr_lhs11, vr_lhs12, vr_lhs13, vr_lhs14,
      vr_lhs15, vr_lhs16, vr_lhs17, vr_lhs18, vr_lhs19, vr_lhs20, vr_lhs21,
      vr_lhs22, vr_lhs23, vr_lhs24, vr_lhs25, vr_lhs26, vr_lhs27, vr_lhs28,
      vr_lhs29, vr_lhs30, vr_lhs31;
  va16f32x2 vacc0, vacc1, vacc2, vacc3, vacc4, vacc5, vacc6, vacc7, vacc8,
      vacc9, vacc10, vacc11, vacc12, vacc13, vacc14, vacc15, vacc16, vacc17,
      vacc18, vacc19, vacc20, vacc21, vacc22, vacc23, vacc24, vacc25, vacc26,
      vacc27, vacc28, vacc29, vacc30, vacc31;

  //
  // Weight Address/Offset Configuration
  //
  int vmem_rhs_addr = reinterpret_cast<int>(rhs_addr >> 6);
  vmem_rhs_addr = (vmem_rhs_addr + 3) << 16 | (vmem_rhs_addr + 2);
  tar_t weight_addr_base = __dtu_c_movsr2targ(vmem_rhs_addr);
  int weight_offset_k = 2;
  int weight_offset_k_b = -6;
  int weight_offset_k_f = 10;
  int weight_offset_m = -(K * N / 32);
  tar_t t_weight_offset_k = __dtu_c_movsr2tari(
      (weight_offset_k << 16) | (weight_offset_k & 0xffff), weight_addr_base);
  tar_t t_weight_offset_k_b = __dtu_c_movsr2tari(
      (weight_offset_k_b << 16) | (weight_offset_k_b & 0xffff),
      weight_addr_base);
  tar_t t_weight_offset_k_f = __dtu_c_movsr2tari(
      (weight_offset_k_f << 16) | (weight_offset_k_f & 0xffff),
      weight_addr_base);
  tar_t t_weight_offset_m = __dtu_c_movsr2tari(
      (weight_offset_m << 16) | (weight_offset_m & 0xffff), weight_addr_base);

  vr_rhs0 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  vr_rhs1 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  vr_rhs2 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);
  vr_rhs3 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);
  vr_rhs4 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  vr_rhs5 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  vr_rhs6 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);
  vr_rhs7 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);
  vr_rhs8 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  vr_rhs9 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  vr_rhs10 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);
  vr_rhs11 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);
  vr_rhs12 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  vr_rhs13 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  vr_rhs14 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);
  vr_rhs15 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);
  vr_rhs16 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  vr_rhs17 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  vr_rhs18 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);
  vr_rhs19 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);
  vr_rhs20 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  vr_rhs21 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  vr_rhs22 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);
  vr_rhs23 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);
  vr_rhs24 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  vr_rhs25 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  vr_rhs26 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);
  vr_rhs27 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);
  vr_rhs28 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  vr_rhs29 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  vr_rhs30 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);
  vr_rhs31 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

  // Special Register Configuration
  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_m_s1(0);
  __dtu_c_movsr2vab_m_d(0);

  //
  // Input Address/Offset Configuration
  //
  // set targs base address
  int vmem_lhs_addr = reinterpret_cast<int>(lhs_addr >> 6);
  vmem_lhs_addr = ((vmem_lhs_addr) << 16) | vmem_lhs_addr;
  tar_t input_addr_base = __dtu_c_movsr2targ(vmem_lhs_addr);
  // set targs base address
  int input_offset_m = 64 - (2 * M * K / 64) + 2 * M - 64 - 1 + 2;
  int input_offset_n = -(2 * M * K / 64) + 2 * M - 64 - 1 + 2;
  int input_offset_k_dummy = -64 + 1 + 2;
  int input_offset_k_dummy1 = 2 * M - 64 - 1 + 2;
  int input_offset_k = (2);
  tar_t t_input_offset_n = __dtu_c_movsr2tari(
      (input_offset_n << 16) | (input_offset_n & 0xffff), input_addr_base);
  tar_t t_input_offset_k = __dtu_c_movsr2tari(
      (input_offset_k << 16) | (input_offset_k & 0xffff), input_addr_base);
  tar_t t_input_offset_k_dummy = __dtu_c_movsr2tari(
      (input_offset_k_dummy << 16) | (input_offset_k_dummy & 0xffff),
      input_addr_base);
  tar_t t_input_offset_k_dummy1 = __dtu_c_movsr2tari(
      (input_offset_k_dummy1 << 16) | (input_offset_k_dummy1 & 0xffff),
      input_addr_base);
  tar_t t_input_offset_m = __dtu_c_movsr2tari(
      (input_offset_m << 16) | (input_offset_m & 0xffff), input_addr_base);

  //
  // Output Address Configuraiton
  //
  int vmem_output_addr = reinterpret_cast<int>(out_addr >> 6);
  vmem_output_addr = ((vmem_output_addr + 1) << 16 | vmem_output_addr);
  tar_t output_addr_base = __dtu_c_movsr2targ(vmem_output_addr);

  int output_offset_n = N >> 5;
  int output_offset_n_dummy = -(N) + 2 + (N >> 5);
  int output_offset_m = ((31 * N) >> 5) - (N) + 2 + (N >> 5);
  tar_t t_output_offset_n = __dtu_c_movsr2tari(
      (output_offset_n << 16) | (output_offset_n & 0xffff), output_addr_base);
  tar_t t_output_offset_n_dummy = __dtu_c_movsr2tari(
      (output_offset_n_dummy << 16) | (output_offset_n_dummy & 0xffff),
      output_addr_base);
  tar_t t_output_offset_m = __dtu_c_movsr2tari(
      (output_offset_m << 16) | (output_offset_m & 0xffff), output_addr_base);

  //   int vab_shift = 0;
  int naccovr = 0x1;
  if (reduce_index == 0) {
    naccovr = 0x10001;
  }
  int k_count = K >> 6;

  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs0, 1);
  vr_lhs0 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs0 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs1, 2);
  vr_lhs1 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs1 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs2, 3);
  vr_lhs2 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs2 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs3, 0);
  vr_lhs3 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs3 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs4, 5);
  vr_lhs4 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs4 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs5, 6);
  vr_lhs5 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs5 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs6, 7);
  vr_lhs6 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs6 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs7, 4);
  vr_lhs7 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs7 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs8, 9);
  vr_lhs8 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs8 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs9, 10);
  vr_lhs9 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs9 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs10, 11);
  vr_lhs10 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs10 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs11, 8);
  vr_lhs11 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs11 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs12, 13);
  vr_lhs12 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs12 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs13, 14);
  vr_lhs13 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs13 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs14, 15);
  vr_lhs14 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs14 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs15, 12);
  vr_lhs15 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs15 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs16, 17);
  vr_lhs16 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs16 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs17, 18);
  vr_lhs17 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs17 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs18, 19);
  vr_lhs18 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs18 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs19, 16);
  vr_lhs19 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs19 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs20, 21);
  vr_lhs20 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs20 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs21, 22);
  vr_lhs21 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs21 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs22, 23);
  vr_lhs22 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs22 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs23, 20);
  vr_lhs23 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs23 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs24, 25);
  vr_lhs24 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs24 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs25, 26);
  vr_lhs25 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs25 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs26, 27);
  vr_lhs26 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs26 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs27, 24);
  vr_lhs27 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs27 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs28, 29);
  vr_lhs28 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs28 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs29, 30);
  vr_lhs29 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs29 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs30, 31);
  vr_lhs30 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
  vr_rhs30 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);
  smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs31, 28);
  vr_lhs31 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k_dummy);
  vr_rhs31 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

  //
  // Loop logic
  //
#pragma clang loop unroll(disable)
  for (int m_idx = 0; m_idx < M; m_idx = m_idx + 32) {
    if (m_idx != 0) {
      __dtu_l_tvsta_cvt2fp16_rnd(vacc0, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc1, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc2, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc3, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc4, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc5, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc6, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc7, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc8, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc9, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc10, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc11, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc12, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc13, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc14, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc15, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc16, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc17, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc18, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc19, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc20, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc21, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc22, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc23, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc24, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc25, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc26, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc27, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc28, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc29, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc30, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc31, output_addr_base, t_output_offset_m);
    }

#pragma clang loop unroll(disable)
    for (int n_idx = 0; n_idx < N - 64; n_idx = n_idx + 64) {
      __dtu_c_movsr2naccovr(naccovr);
#pragma clang loop unroll(disable)
      for (int k_idx = 0; k_idx < k_count - 1; k_idx += 1) {
        // Unroll h & w loop here
        // Load weight for oc0-15 ci0~ci15
        // Load input hxw 00, 01, 02, 10, 11, 12, 20, 21, 22 caculate
        vacc0 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc0, vr_lhs0, smr, vr_rhs0, 1);
        vr_lhs0 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs0 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc1 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc1, vr_lhs1, smr, vr_rhs1, 2);
        vr_lhs1 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs1 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc2 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc2, vr_lhs2, smr, vr_rhs2, 3);
        vr_lhs2 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs2 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

        vacc3 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc3, vr_lhs3, smr, vr_rhs3, 0);
        vr_lhs3 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs3 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

        vacc4 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc4, vr_lhs4, smr, vr_rhs4, 5);
        vr_lhs4 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs4 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc5 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc5, vr_lhs5, smr, vr_rhs5, 6);
        vr_lhs5 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs5 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc6 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc6, vr_lhs6, smr, vr_rhs6, 7);
        vr_lhs6 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs6 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

        vacc7 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc7, vr_lhs7, smr, vr_rhs7, 4);
        vr_lhs7 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs7 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

        vacc8 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc8, vr_lhs8, smr, vr_rhs8, 9);
        vr_lhs8 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs8 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc9 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc9, vr_lhs9, smr, vr_rhs9, 10);
        vr_lhs9 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs9 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc10 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc10, vr_lhs10, smr, vr_rhs10, 11);
        vr_lhs10 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs10 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

        vacc11 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc11, vr_lhs11, smr, vr_rhs11, 8);
        vr_lhs11 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs11 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

        vacc12 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc12, vr_lhs12, smr, vr_rhs12, 13);
        vr_lhs12 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs12 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc13 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc13, vr_lhs13, smr, vr_rhs13, 14);
        vr_lhs13 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs13 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc14 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc14, vr_lhs14, smr, vr_rhs14, 15);
        vr_lhs14 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs14 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

        vacc15 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc15, vr_lhs15, smr, vr_rhs15, 12);
        vr_lhs15 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs15 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

        vacc16 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc16, vr_lhs16, smr, vr_rhs16, 17);
        vr_lhs16 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs16 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc17 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc17, vr_lhs17, smr, vr_rhs17, 18);
        vr_lhs17 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs17 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc18 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc18, vr_lhs18, smr, vr_rhs18, 19);
        vr_lhs18 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs18 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

        vacc19 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc19, vr_lhs19, smr, vr_rhs19, 16);
        vr_lhs19 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs19 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

        vacc20 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc20, vr_lhs20, smr, vr_rhs20, 21);
        vr_lhs20 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs20 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc21 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc21, vr_lhs21, smr, vr_rhs21, 22);
        vr_lhs21 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs21 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc22 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc22, vr_lhs22, smr, vr_rhs22, 23);
        vr_lhs22 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs22 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

        vacc23 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc23, vr_lhs23, smr, vr_rhs23, 20);
        vr_lhs23 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs23 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

        vacc24 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc24, vr_lhs24, smr, vr_rhs24, 25);
        vr_lhs24 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs24 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc25 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc25, vr_lhs25, smr, vr_rhs25, 26);
        vr_lhs25 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs25 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc26 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc26, vr_lhs26, smr, vr_rhs26, 27);
        vr_lhs26 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs26 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

        vacc27 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc27, vr_lhs27, smr, vr_rhs27, 24);
        vr_lhs27 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs27 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

        vacc28 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc28, vr_lhs28, smr, vr_rhs28, 29);
        vr_lhs28 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs28 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc29 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc29, vr_lhs29, smr, vr_rhs29, 30);
        vr_lhs29 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs29 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc30 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc30, vr_lhs30, smr, vr_rhs30, 31);
        vr_lhs30 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs30 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

        vacc31 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc31, vr_lhs31, smr, vr_rhs31, 28);
        vr_lhs31 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k_dummy1);
        vr_rhs31 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

        smr = __dtu_v_swapsmr(smr);
        __dtu_c_movsr2naccovr(0x1);

        vacc0 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc0, vr_lhs0, smr, vr_rhs0, 1);
        vr_lhs0 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs0 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc1 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc1, vr_lhs1, smr, vr_rhs1, 2);
        vr_lhs1 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs1 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc2 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc2, vr_lhs2, smr, vr_rhs2, 3);
        vr_lhs2 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs2 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

        vacc3 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc3, vr_lhs3, smr, vr_rhs3, 0);
        vr_lhs3 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs3 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

        vacc4 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc4, vr_lhs4, smr, vr_rhs4, 5);
        vr_lhs4 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs4 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc5 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc5, vr_lhs5, smr, vr_rhs5, 6);
        vr_lhs5 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs5 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc6 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc6, vr_lhs6, smr, vr_rhs6, 7);
        vr_lhs6 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs6 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

        vacc7 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc7, vr_lhs7, smr, vr_rhs7, 4);
        vr_lhs7 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs7 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

        vacc8 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc8, vr_lhs8, smr, vr_rhs8, 9);
        vr_lhs8 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs8 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc9 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc9, vr_lhs9, smr, vr_rhs9, 10);
        vr_lhs9 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs9 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc10 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc10, vr_lhs10, smr, vr_rhs10, 11);
        vr_lhs10 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs10 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

        vacc11 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc11, vr_lhs11, smr, vr_rhs11, 8);
        vr_lhs11 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs11 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

        vacc12 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc12, vr_lhs12, smr, vr_rhs12, 13);
        vr_lhs12 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs12 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc13 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc13, vr_lhs13, smr, vr_rhs13, 14);
        vr_lhs13 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs13 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc14 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc14, vr_lhs14, smr, vr_rhs14, 15);
        vr_lhs14 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs14 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

        vacc15 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc15, vr_lhs15, smr, vr_rhs15, 12);
        vr_lhs15 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs15 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

        vacc16 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc16, vr_lhs16, smr, vr_rhs16, 17);
        vr_lhs16 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs16 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc17 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc17, vr_lhs17, smr, vr_rhs17, 18);
        vr_lhs17 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs17 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc18 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc18, vr_lhs18, smr, vr_rhs18, 19);
        vr_lhs18 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs18 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

        vacc19 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc19, vr_lhs19, smr, vr_rhs19, 16);
        vr_lhs19 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs19 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

        vacc20 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc20, vr_lhs20, smr, vr_rhs20, 21);
        vr_lhs20 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs20 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc21 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc21, vr_lhs21, smr, vr_rhs21, 22);
        vr_lhs21 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs21 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc22 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc22, vr_lhs22, smr, vr_rhs22, 23);
        vr_lhs22 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs22 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

        vacc23 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc23, vr_lhs23, smr, vr_rhs23, 20);
        vr_lhs23 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs23 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

        vacc24 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc24, vr_lhs24, smr, vr_rhs24, 25);
        vr_lhs24 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs24 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc25 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc25, vr_lhs25, smr, vr_rhs25, 26);
        vr_lhs25 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs25 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc26 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc26, vr_lhs26, smr, vr_rhs26, 27);
        vr_lhs26 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs26 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

        vacc27 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc27, vr_lhs27, smr, vr_rhs27, 24);
        vr_lhs27 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs27 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

        vacc28 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc28, vr_lhs28, smr, vr_rhs28, 29);
        vr_lhs28 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs28 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc29 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc29, vr_lhs29, smr, vr_rhs29, 30);
        vr_lhs29 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs29 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

        vacc30 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc30, vr_lhs30, smr, vr_rhs30, 31);
        vr_lhs30 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
        vr_rhs30 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

        vacc31 =
            __dtu_m_vmm_mode0_f_vs0_ld_row(vacc31, vr_lhs31, smr, vr_rhs31, 28);
        vr_lhs31 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k_dummy);
        vr_rhs31 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

        // Only the first time vacc need to be initialized
        smr = __dtu_v_swapsmr(smr);
      }  // K

      // Unroll h & w loop here
      // Load weight for oc0-15 ci0~ci15
      // Load input hxw 00, 01, 02, 10, 11, 12, 20, 21, 22 caculate
      vacc0 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc0, vr_lhs0, smr, vr_rhs0, 1);
      vr_lhs0 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs0 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc1 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc1, vr_lhs1, smr, vr_rhs1, 2);
      vr_lhs1 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs1 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc2 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc2, vr_lhs2, smr, vr_rhs2, 3);
      vr_lhs2 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs2 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc3 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc3, vr_lhs3, smr, vr_rhs3, 0);
      vr_lhs3 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs3 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      vacc4 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc4, vr_lhs4, smr, vr_rhs4, 5);
      vr_lhs4 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs4 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc5 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc5, vr_lhs5, smr, vr_rhs5, 6);
      vr_lhs5 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs5 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc6 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc6, vr_lhs6, smr, vr_rhs6, 7);
      vr_lhs6 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs6 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc7 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc7, vr_lhs7, smr, vr_rhs7, 4);
      vr_lhs7 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs7 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      vacc8 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc8, vr_lhs8, smr, vr_rhs8, 9);
      vr_lhs8 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs8 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc9 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc9, vr_lhs9, smr, vr_rhs9, 10);
      vr_lhs9 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs9 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc10 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc10, vr_lhs10, smr, vr_rhs10, 11);
      vr_lhs10 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs10 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc11 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc11, vr_lhs11, smr, vr_rhs11, 8);
      vr_lhs11 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs11 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      vacc12 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc12, vr_lhs12, smr, vr_rhs12, 13);
      vr_lhs12 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs12 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc13 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc13, vr_lhs13, smr, vr_rhs13, 14);
      vr_lhs13 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs13 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc14 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc14, vr_lhs14, smr, vr_rhs14, 15);
      vr_lhs14 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs14 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc15 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc15, vr_lhs15, smr, vr_rhs15, 12);
      vr_lhs15 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs15 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      vacc16 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc16, vr_lhs16, smr, vr_rhs16, 17);
      vr_lhs16 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs16 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc17 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc17, vr_lhs17, smr, vr_rhs17, 18);
      vr_lhs17 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs17 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc18 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc18, vr_lhs18, smr, vr_rhs18, 19);
      vr_lhs18 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs18 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc19 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc19, vr_lhs19, smr, vr_rhs19, 16);
      vr_lhs19 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs19 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      vacc20 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc20, vr_lhs20, smr, vr_rhs20, 21);
      vr_lhs20 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs20 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc21 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc21, vr_lhs21, smr, vr_rhs21, 22);
      vr_lhs21 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs21 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc22 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc22, vr_lhs22, smr, vr_rhs22, 23);
      vr_lhs22 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs22 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc23 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc23, vr_lhs23, smr, vr_rhs23, 20);
      vr_lhs23 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs23 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      vacc24 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc24, vr_lhs24, smr, vr_rhs24, 25);
      vr_lhs24 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs24 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc25 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc25, vr_lhs25, smr, vr_rhs25, 26);
      vr_lhs25 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs25 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc26 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc26, vr_lhs26, smr, vr_rhs26, 27);
      vr_lhs26 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs26 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc27 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc27, vr_lhs27, smr, vr_rhs27, 24);
      vr_lhs27 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs27 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      vacc28 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc28, vr_lhs28, smr, vr_rhs28, 29);
      vr_lhs28 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs28 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc29 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc29, vr_lhs29, smr, vr_rhs29, 30);
      vr_lhs29 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs29 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc30 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc30, vr_lhs30, smr, vr_rhs30, 31);
      vr_lhs30 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs30 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc31 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc31, vr_lhs31, smr, vr_rhs31, 28);
      vr_lhs31 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_n);
      vr_rhs31 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      smr = __dtu_v_swapsmr(smr);
      __dtu_c_movsr2naccovr(0x1);

      vacc0 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc0, vr_lhs0, smr, vr_rhs0, 1);
      vr_lhs0 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs0 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc1 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc1, vr_lhs1, smr, vr_rhs1, 2);
      vr_lhs1 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs1 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc2 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc2, vr_lhs2, smr, vr_rhs2, 3);
      vr_lhs2 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs2 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc3 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc3, vr_lhs3, smr, vr_rhs3, 0);
      vr_lhs3 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs3 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      vacc4 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc4, vr_lhs4, smr, vr_rhs4, 5);
      vr_lhs4 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs4 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc5 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc5, vr_lhs5, smr, vr_rhs5, 6);
      vr_lhs5 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs5 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc6 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc6, vr_lhs6, smr, vr_rhs6, 7);
      vr_lhs6 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs6 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc7 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc7, vr_lhs7, smr, vr_rhs7, 4);
      vr_lhs7 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs7 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      vacc8 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc8, vr_lhs8, smr, vr_rhs8, 9);
      vr_lhs8 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs8 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc9 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc9, vr_lhs9, smr, vr_rhs9, 10);
      vr_lhs9 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs9 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc10 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc10, vr_lhs10, smr, vr_rhs10, 11);
      vr_lhs10 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs10 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc11 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc11, vr_lhs11, smr, vr_rhs11, 8);
      vr_lhs11 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs11 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      vacc12 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc12, vr_lhs12, smr, vr_rhs12, 13);
      vr_lhs12 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs12 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc13 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc13, vr_lhs13, smr, vr_rhs13, 14);
      vr_lhs13 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs13 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc14 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc14, vr_lhs14, smr, vr_rhs14, 15);
      vr_lhs14 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs14 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc15 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc15, vr_lhs15, smr, vr_rhs15, 12);
      vr_lhs15 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs15 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      vacc16 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc16, vr_lhs16, smr, vr_rhs16, 17);
      vr_lhs16 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs16 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc17 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc17, vr_lhs17, smr, vr_rhs17, 18);
      vr_lhs17 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs17 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc18 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc18, vr_lhs18, smr, vr_rhs18, 19);
      vr_lhs18 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs18 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc19 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc19, vr_lhs19, smr, vr_rhs19, 16);
      vr_lhs19 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs19 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      vacc20 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc20, vr_lhs20, smr, vr_rhs20, 21);
      vr_lhs20 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs20 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc21 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc21, vr_lhs21, smr, vr_rhs21, 22);
      vr_lhs21 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs21 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc22 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc22, vr_lhs22, smr, vr_rhs22, 23);
      vr_lhs22 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs22 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc23 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc23, vr_lhs23, smr, vr_rhs23, 20);
      vr_lhs23 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs23 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      vacc24 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc24, vr_lhs24, smr, vr_rhs24, 25);
      vr_lhs24 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs24 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc25 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc25, vr_lhs25, smr, vr_rhs25, 26);
      vr_lhs25 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs25 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc26 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc26, vr_lhs26, smr, vr_rhs26, 27);
      vr_lhs26 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs26 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc27 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc27, vr_lhs27, smr, vr_rhs27, 24);
      vr_lhs27 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs27 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      vacc28 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc28, vr_lhs28, smr, vr_rhs28, 29);
      vr_lhs28 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs28 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc29 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc29, vr_lhs29, smr, vr_rhs29, 30);
      vr_lhs29 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs29 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc30 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc30, vr_lhs30, smr, vr_rhs30, 31);
      vr_lhs30 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs30 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc31 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc31, vr_lhs31, smr, vr_rhs31, 28);
      vr_lhs31 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k_dummy);
      vr_rhs31 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      // Only the first time vacc need to be initialized
      smr = __dtu_v_swapsmr(smr);

      __dtu_l_tvsta_cvt2fp16_rnd(vacc0, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc1, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc2, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc3, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc4, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc5, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc6, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc7, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc8, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc9, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc10, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc11, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc12, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc13, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc14, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc15, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc16, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc17, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc18, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc19, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc20, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc21, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc22, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc23, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc24, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc25, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc26, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc27, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc28, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc29, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc30, output_addr_base, t_output_offset_n);
      __dtu_l_tvsta_cvt2fp16_rnd(vacc31, output_addr_base,
                                 t_output_offset_n_dummy);
    }  // N

    __dtu_c_movsr2naccovr(naccovr);
#pragma clang loop unroll(disable)
    for (int k_idx = 0; k_idx < k_count - 1; k_idx += 1) {
      // Unroll h & w loop here
      // Load weight for oc0-15 ci0~ci15
      // Load input hxw 00, 01, 02, 10, 11, 12, 20, 21, 22 caculate
      vacc0 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc0, vr_lhs0, smr, vr_rhs0, 1);
      vr_lhs0 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs0 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc1 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc1, vr_lhs1, smr, vr_rhs1, 2);
      vr_lhs1 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs1 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc2 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc2, vr_lhs2, smr, vr_rhs2, 3);
      vr_lhs2 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs2 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc3 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc3, vr_lhs3, smr, vr_rhs3, 0);
      vr_lhs3 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs3 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      vacc4 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc4, vr_lhs4, smr, vr_rhs4, 5);
      vr_lhs4 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs4 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc5 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc5, vr_lhs5, smr, vr_rhs5, 6);
      vr_lhs5 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs5 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc6 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc6, vr_lhs6, smr, vr_rhs6, 7);
      vr_lhs6 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs6 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc7 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc7, vr_lhs7, smr, vr_rhs7, 4);
      vr_lhs7 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs7 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      vacc8 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc8, vr_lhs8, smr, vr_rhs8, 9);
      vr_lhs8 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs8 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc9 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc9, vr_lhs9, smr, vr_rhs9, 10);
      vr_lhs9 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs9 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc10 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc10, vr_lhs10, smr, vr_rhs10, 11);
      vr_lhs10 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs10 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc11 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc11, vr_lhs11, smr, vr_rhs11, 8);
      vr_lhs11 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs11 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      vacc12 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc12, vr_lhs12, smr, vr_rhs12, 13);
      vr_lhs12 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs12 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc13 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc13, vr_lhs13, smr, vr_rhs13, 14);
      vr_lhs13 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs13 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc14 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc14, vr_lhs14, smr, vr_rhs14, 15);
      vr_lhs14 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs14 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc15 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc15, vr_lhs15, smr, vr_rhs15, 12);
      vr_lhs15 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs15 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      vacc16 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc16, vr_lhs16, smr, vr_rhs16, 17);
      vr_lhs16 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs16 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc17 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc17, vr_lhs17, smr, vr_rhs17, 18);
      vr_lhs17 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs17 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc18 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc18, vr_lhs18, smr, vr_rhs18, 19);
      vr_lhs18 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs18 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc19 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc19, vr_lhs19, smr, vr_rhs19, 16);
      vr_lhs19 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs19 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      vacc20 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc20, vr_lhs20, smr, vr_rhs20, 21);
      vr_lhs20 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs20 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc21 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc21, vr_lhs21, smr, vr_rhs21, 22);
      vr_lhs21 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs21 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc22 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc22, vr_lhs22, smr, vr_rhs22, 23);
      vr_lhs22 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs22 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc23 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc23, vr_lhs23, smr, vr_rhs23, 20);
      vr_lhs23 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs23 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      vacc24 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc24, vr_lhs24, smr, vr_rhs24, 25);
      vr_lhs24 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs24 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc25 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc25, vr_lhs25, smr, vr_rhs25, 26);
      vr_lhs25 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs25 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc26 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc26, vr_lhs26, smr, vr_rhs26, 27);
      vr_lhs26 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs26 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc27 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc27, vr_lhs27, smr, vr_rhs27, 24);
      vr_lhs27 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs27 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      vacc28 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc28, vr_lhs28, smr, vr_rhs28, 29);
      vr_lhs28 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs28 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc29 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc29, vr_lhs29, smr, vr_rhs29, 30);
      vr_lhs29 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs29 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc30 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc30, vr_lhs30, smr, vr_rhs30, 31);
      vr_lhs30 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs30 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc31 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc31, vr_lhs31, smr, vr_rhs31, 28);
      vr_lhs31 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k_dummy1);
      vr_rhs31 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      smr = __dtu_v_swapsmr(smr);
      __dtu_c_movsr2naccovr(0x1);

      vacc0 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc0, vr_lhs0, smr, vr_rhs0, 1);
      vr_lhs0 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs0 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc1 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc1, vr_lhs1, smr, vr_rhs1, 2);
      vr_lhs1 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs1 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc2 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc2, vr_lhs2, smr, vr_rhs2, 3);
      vr_lhs2 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs2 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc3 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc3, vr_lhs3, smr, vr_rhs3, 0);
      vr_lhs3 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs3 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      vacc4 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc4, vr_lhs4, smr, vr_rhs4, 5);
      vr_lhs4 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs4 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc5 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc5, vr_lhs5, smr, vr_rhs5, 6);
      vr_lhs5 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs5 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc6 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc6, vr_lhs6, smr, vr_rhs6, 7);
      vr_lhs6 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs6 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc7 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc7, vr_lhs7, smr, vr_rhs7, 4);
      vr_lhs7 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs7 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      vacc8 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc8, vr_lhs8, smr, vr_rhs8, 9);
      vr_lhs8 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs8 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc9 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc9, vr_lhs9, smr, vr_rhs9, 10);
      vr_lhs9 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs9 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc10 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc10, vr_lhs10, smr, vr_rhs10, 11);
      vr_lhs10 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs10 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc11 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc11, vr_lhs11, smr, vr_rhs11, 8);
      vr_lhs11 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs11 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      vacc12 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc12, vr_lhs12, smr, vr_rhs12, 13);
      vr_lhs12 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs12 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc13 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc13, vr_lhs13, smr, vr_rhs13, 14);
      vr_lhs13 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs13 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc14 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc14, vr_lhs14, smr, vr_rhs14, 15);
      vr_lhs14 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs14 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc15 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc15, vr_lhs15, smr, vr_rhs15, 12);
      vr_lhs15 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs15 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      vacc16 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc16, vr_lhs16, smr, vr_rhs16, 17);
      vr_lhs16 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs16 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc17 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc17, vr_lhs17, smr, vr_rhs17, 18);
      vr_lhs17 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs17 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc18 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc18, vr_lhs18, smr, vr_rhs18, 19);
      vr_lhs18 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs18 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc19 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc19, vr_lhs19, smr, vr_rhs19, 16);
      vr_lhs19 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs19 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      vacc20 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc20, vr_lhs20, smr, vr_rhs20, 21);
      vr_lhs20 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs20 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc21 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc21, vr_lhs21, smr, vr_rhs21, 22);
      vr_lhs21 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs21 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc22 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc22, vr_lhs22, smr, vr_rhs22, 23);
      vr_lhs22 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs22 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc23 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc23, vr_lhs23, smr, vr_rhs23, 20);
      vr_lhs23 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs23 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      vacc24 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc24, vr_lhs24, smr, vr_rhs24, 25);
      vr_lhs24 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs24 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc25 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc25, vr_lhs25, smr, vr_rhs25, 26);
      vr_lhs25 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs25 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc26 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc26, vr_lhs26, smr, vr_rhs26, 27);
      vr_lhs26 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs26 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc27 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc27, vr_lhs27, smr, vr_rhs27, 24);
      vr_lhs27 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs27 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      vacc28 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc28, vr_lhs28, smr, vr_rhs28, 29);
      vr_lhs28 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs28 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc29 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc29, vr_lhs29, smr, vr_rhs29, 30);
      vr_lhs29 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs29 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

      vacc30 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc30, vr_lhs30, smr, vr_rhs30, 31);
      vr_lhs30 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
      vr_rhs30 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

      vacc31 =
          __dtu_m_vmm_mode0_f_vs0_ld_row(vacc31, vr_lhs31, smr, vr_rhs31, 28);
      vr_lhs31 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k_dummy);
      vr_rhs31 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

      // Only the first time vacc need to be initialized
      smr = __dtu_v_swapsmr(smr);
    }  // K

    { __dtu_s_tvld_itar(weight_addr_base, t_weight_offset_m); }
    // Unroll h & w loop here
    // Load weight for oc0-15 ci0~ci15
    // Load input hxw 00, 01, 02, 10, 11, 12, 20, 21, 22 caculate
    vacc0 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc0, vr_lhs0, smr, vr_rhs0, 1);
    vr_lhs0 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs0 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc1 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc1, vr_lhs1, smr, vr_rhs1, 2);
    vr_lhs1 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs1 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc2 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc2, vr_lhs2, smr, vr_rhs2, 3);
    vr_lhs2 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs2 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

    vacc3 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc3, vr_lhs3, smr, vr_rhs3, 0);
    vr_lhs3 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs3 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

    vacc4 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc4, vr_lhs4, smr, vr_rhs4, 5);
    vr_lhs4 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs4 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc5 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc5, vr_lhs5, smr, vr_rhs5, 6);
    vr_lhs5 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs5 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc6 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc6, vr_lhs6, smr, vr_rhs6, 7);
    vr_lhs6 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs6 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

    vacc7 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc7, vr_lhs7, smr, vr_rhs7, 4);
    vr_lhs7 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs7 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

    vacc8 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc8, vr_lhs8, smr, vr_rhs8, 9);
    vr_lhs8 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs8 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc9 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc9, vr_lhs9, smr, vr_rhs9, 10);
    vr_lhs9 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs9 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc10 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc10, vr_lhs10, smr, vr_rhs10, 11);
    vr_lhs10 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs10 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

    vacc11 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc11, vr_lhs11, smr, vr_rhs11, 8);
    vr_lhs11 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs11 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

    vacc12 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc12, vr_lhs12, smr, vr_rhs12, 13);
    vr_lhs12 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs12 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc13 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc13, vr_lhs13, smr, vr_rhs13, 14);
    vr_lhs13 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs13 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc14 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc14, vr_lhs14, smr, vr_rhs14, 15);
    vr_lhs14 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs14 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

    vacc15 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc15, vr_lhs15, smr, vr_rhs15, 12);
    vr_lhs15 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs15 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

    vacc16 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc16, vr_lhs16, smr, vr_rhs16, 17);
    vr_lhs16 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs16 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc17 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc17, vr_lhs17, smr, vr_rhs17, 18);
    vr_lhs17 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs17 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc18 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc18, vr_lhs18, smr, vr_rhs18, 19);
    vr_lhs18 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs18 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

    vacc19 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc19, vr_lhs19, smr, vr_rhs19, 16);
    vr_lhs19 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs19 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

    vacc20 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc20, vr_lhs20, smr, vr_rhs20, 21);
    vr_lhs20 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs20 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc21 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc21, vr_lhs21, smr, vr_rhs21, 22);
    vr_lhs21 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs21 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc22 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc22, vr_lhs22, smr, vr_rhs22, 23);
    vr_lhs22 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs22 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

    vacc23 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc23, vr_lhs23, smr, vr_rhs23, 20);
    vr_lhs23 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs23 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

    vacc24 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc24, vr_lhs24, smr, vr_rhs24, 25);
    vr_lhs24 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs24 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc25 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc25, vr_lhs25, smr, vr_rhs25, 26);
    vr_lhs25 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs25 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc26 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc26, vr_lhs26, smr, vr_rhs26, 27);
    vr_lhs26 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs26 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

    vacc27 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc27, vr_lhs27, smr, vr_rhs27, 24);
    vr_lhs27 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs27 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

    vacc28 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc28, vr_lhs28, smr, vr_rhs28, 29);
    vr_lhs28 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs28 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc29 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc29, vr_lhs29, smr, vr_rhs29, 30);
    vr_lhs29 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs29 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc30 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc30, vr_lhs30, smr, vr_rhs30, 31);
    vr_lhs30 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs30 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

    vacc31 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc31, vr_lhs31, smr, vr_rhs31, 28);
    vr_lhs31 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_m);
    vr_rhs31 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

    smr = __dtu_v_swapsmr(smr);
    __dtu_c_movsr2naccovr(0x1);

    vacc0 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc0, vr_lhs0, smr, vr_rhs0, 1);
    vr_lhs0 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs0 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc1 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc1, vr_lhs1, smr, vr_rhs1, 2);
    vr_lhs1 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs1 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc2 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc2, vr_lhs2, smr, vr_rhs2, 3);
    vr_lhs2 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs2 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

    vacc3 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc3, vr_lhs3, smr, vr_rhs3, 0);
    vr_lhs3 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs3 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

    vacc4 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc4, vr_lhs4, smr, vr_rhs4, 5);
    vr_lhs4 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs4 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc5 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc5, vr_lhs5, smr, vr_rhs5, 6);
    vr_lhs5 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs5 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc6 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc6, vr_lhs6, smr, vr_rhs6, 7);
    vr_lhs6 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs6 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

    vacc7 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc7, vr_lhs7, smr, vr_rhs7, 4);
    vr_lhs7 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs7 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

    vacc8 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc8, vr_lhs8, smr, vr_rhs8, 9);
    vr_lhs8 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs8 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc9 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc9, vr_lhs9, smr, vr_rhs9, 10);
    vr_lhs9 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs9 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc10 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc10, vr_lhs10, smr, vr_rhs10, 11);
    vr_lhs10 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs10 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

    vacc11 = __dtu_m_vmm_mode0_f_vs0_ld_row(vacc11, vr_lhs11, smr, vr_rhs11, 8);
    vr_lhs11 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs11 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

    vacc12 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc12, vr_lhs12, smr, vr_rhs12, 13);
    vr_lhs12 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs12 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc13 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc13, vr_lhs13, smr, vr_rhs13, 14);
    vr_lhs13 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs13 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc14 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc14, vr_lhs14, smr, vr_rhs14, 15);
    vr_lhs14 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs14 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

    vacc15 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc15, vr_lhs15, smr, vr_rhs15, 12);
    vr_lhs15 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs15 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

    vacc16 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc16, vr_lhs16, smr, vr_rhs16, 17);
    vr_lhs16 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs16 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc17 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc17, vr_lhs17, smr, vr_rhs17, 18);
    vr_lhs17 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs17 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc18 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc18, vr_lhs18, smr, vr_rhs18, 19);
    vr_lhs18 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs18 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

    vacc19 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc19, vr_lhs19, smr, vr_rhs19, 16);
    vr_lhs19 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs19 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

    vacc20 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc20, vr_lhs20, smr, vr_rhs20, 21);
    vr_lhs20 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs20 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc21 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc21, vr_lhs21, smr, vr_rhs21, 22);
    vr_lhs21 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs21 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc22 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc22, vr_lhs22, smr, vr_rhs22, 23);
    vr_lhs22 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs22 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

    vacc23 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc23, vr_lhs23, smr, vr_rhs23, 20);
    vr_lhs23 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs23 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

    vacc24 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc24, vr_lhs24, smr, vr_rhs24, 25);
    vr_lhs24 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs24 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc25 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc25, vr_lhs25, smr, vr_rhs25, 26);
    vr_lhs25 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs25 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc26 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc26, vr_lhs26, smr, vr_rhs26, 27);
    vr_lhs26 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs26 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

    vacc27 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc27, vr_lhs27, smr, vr_rhs27, 24);
    vr_lhs27 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs27 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

    vacc28 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc28, vr_lhs28, smr, vr_rhs28, 29);
    vr_lhs28 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs28 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc29 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc29, vr_lhs29, smr, vr_rhs29, 30);
    vr_lhs29 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs29 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k);

    vacc30 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc30, vr_lhs30, smr, vr_rhs30, 31);
    vr_lhs30 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k);
    vr_rhs30 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_b);

    vacc31 =
        __dtu_m_vmm_mode0_f_vs0_ld_row(vacc31, vr_lhs31, smr, vr_rhs31, 28);
    vr_lhs31 = __dtu_s_tvld_itar(input_addr_base, t_input_offset_k_dummy);
    vr_rhs31 = __dtu_s_tivld_itar(weight_addr_base, t_weight_offset_k_f);

    // Only the first time vacc need to be initialized
    smr = __dtu_v_swapsmr(smr);
  }  // M

  __dtu_l_tvsta_cvt2fp16_rnd(vacc0, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc1, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc2, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc3, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc4, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc5, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc6, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc7, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc8, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc9, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc10, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc11, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc12, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc13, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc14, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc15, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc16, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc17, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc18, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc19, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc20, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc21, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc22, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc23, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc24, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc25, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc26, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc27, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc28, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc29, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc30, output_addr_base, t_output_offset_n);
  __dtu_l_tvsta_cvt2fp16_rnd(vacc31, output_addr_base, t_output_offset_m);
}


using FP = void (*)(int, int, int, int, int, int, int, int);