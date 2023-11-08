#pragma once
#include <tops.h>
#include <tops/half.h>
#include <tops/bfloat.h>
#include <krt/scalar.h>

#include <tops/tops_runtime.h>
#if __GCU_ARCH__ < 300 
#include "sip20intrin.h"
#endif

#if __GCU_ARCH__ >= 300 
#include "sip30intrin.h"
#endif

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
int get_index() {
    std::size_t blockIndex = blockIdx.z*(gridDim.x*gridDim.y)
        + blockIdx.y*gridDim.x + blockIdx.x;
    std::size_t threadIndex = threadIdx.z*(blockDim.x*blockDim.y)
        + threadIdx.y*blockDim.x + threadIdx.x;
    return blockIndex*(blockDim.x*blockDim.y*blockDim.z) + threadIndex;
}

__device__ __forceinline__
int get_threadIndex() {
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

  explicit DATA(int m, int k, int n, int tile_size, bool check) {
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

/**
 * @brief kernel impl of dot, m max to 256(actually its unlimited), no need to
 * align with 32
 *
 * @param lhs_addr [M, K], fp16
 * @param rhs_addr [K, N], fp16
 * @param out_addr [M, N], fp16
 *
 */
__device__ __forceinline__  void kernel_dot_m_le256_fp16(int lhs_addr, int rhs_addr, int out_addr, int M,
                             int K, int N) {
  //   asm volatile(
  //       "    l.ldi16.u r12, 0x0  \n"
  //       "    c.movsr2spr NACCOVR, r12\n"
  //       "    c.movsr2spr VAB_M_S2, r12\n"
  //       "    c.movsr2spr VAB_M_S1, r12\n"
  //       "    c.movsr2spr VAB_M_D, r12\n"
  //       "    c.movsr2spr VAB_LV_S, r12\n"
  //       "    c.movsr2spr VAB_L_D, r12\n"
  //       "    c.movsr2spr VPR, r12\n"
  //       "    c.movsr2spr MPR, r12\n"
  //       "    c.movsr2spr CPR, r12\n"
  //       "    c.movsr2spr LPR, r12\n"
  //       "    c.movsr2spr TCTL, r12\n"
  //       "    c.movsr2spr VMM_VSEL_OVR, r12\n"
  //       "    c.movsr2spr VMM_RAW, r12\n"
  //       "    c.movsr2spr SVMM_SPR0, r12\n"
  //       "    c.movsr2spr LOOP_STS, r12\n"
  //       :
  //       :
  //       : "r12");
  __dtu_c_movsr2vab_lv_s(0x0);
  __dtu_c_movsr2vab_m_s1(0x0);
  __dtu_c_movsr2vab_m_s2(0x0);
  __dtu_c_movsr2vab_m_d(0x0);
  __dtu_c_movsr2vab_l_d(0x0);
  __dtu_c_movsr2tctl(0x0);
  smr_t smr;
  v32f16 vr_rhs0, vr_rhs1, vr_rhs2, vr_rhs3, vr_rhs4, vr_rhs5, vr_rhs6, vr_rhs7,
      vr_rhs8, vr_rhs9, vr_rhs10, vr_rhs11, vr_rhs12, vr_rhs13, vr_rhs14,
      vr_rhs15, vr_rhs16, vr_rhs17, vr_rhs18, vr_rhs19, vr_rhs20, vr_rhs21,
      vr_rhs22, vr_rhs23, vr_rhs24, vr_rhs25, vr_rhs26, vr_rhs27, vr_rhs28,
      vr_rhs29, vr_rhs30, vr_rhs31;
  v32f16 vr_lhs0;
  va16f32x2 vacc0;

  int naccovr_save = __dtu_movs_naccovr();
  int vab_save_lv_s = __dtu_movs_vab_lv_s();
  int vab_save_l_d = __dtu_movs_vab_l_d();
  int vab_save_m_s1 = __dtu_movs_vab_m_s1();
  int vab_save_m_d = __dtu_movs_vab_m_d();

  int n_step_cnt = N >> 5;
  int vmem_rhs_addr = reinterpret_cast<int>(rhs_addr >> 6);
  tar_t rhs_addr_base =
      __dtu_c_movsr2targ((vmem_rhs_addr + 1) << 16 | vmem_rhs_addr);

  int rhs_nextK = n_step_cnt;
  int rhs_back_nextN = -(K * n_step_cnt - 2);

  tar_t rhs_off_nextK =
      __dtu_c_movsr2tari((rhs_nextK << 16) | rhs_nextK, rhs_addr_base);
  tar_t rhs_off_back_nextN = __dtu_c_movsr2tari(
      (rhs_back_nextN << 16) | (rhs_back_nextN & 0xffff), rhs_addr_base);

  int k_step = (K >> 5);

  int vmem_lhs_addr = reinterpret_cast<int>(lhs_addr >> 6);
  tar_t lhs_addr_base = __dtu_c_movsr2targ((vmem_lhs_addr) << 16 |
                                           vmem_lhs_addr);  // vpt0 == vpt1
  tar_t lhs_off_step1 = __dtu_c_movsr2tari(0x00010001, lhs_addr_base);
  tar_t lhs_off_kback = __dtu_c_movsr2tari(
      ((-k_step) << 16) | ((-k_step) & 0xffff), lhs_addr_base);

  int vmem_output_addr = reinterpret_cast<int>(out_addr >> 6);
  tar_t output_addr_base =
      __dtu_c_movsr2targ((vmem_output_addr + 1) << 16 | vmem_output_addr);
  tar_t output_off_step1 = __dtu_c_movsr2tari((2 << 16) | 2, output_addr_base);

  for (int m_idx = 0; m_idx < M; m_idx++) {
    lhs_addr_base = __dtu_c_movsr2targ((vmem_lhs_addr + k_step * m_idx) << 16 |
                                       vmem_lhs_addr + k_step * m_idx);
    for (int n_idx = 0; n_idx < (N >> 6); n_idx++) {
      __dtu_c_movsr2naccovr(0x10001);
      for (int k_idx = 0; k_idx < (K >> 5); k_idx++) {
        vr_rhs0 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs1 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs2 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs3 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs4 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs5 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs6 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs7 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs8 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs9 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs10 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs11 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs12 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs13 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs14 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs15 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs16 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs17 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs18 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs19 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs20 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs21 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs22 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs23 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs24 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs25 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs26 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs27 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs28 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs29 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs30 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs31 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs0, 0);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs1, 1);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs2, 2);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs3, 3);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs4, 4);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs5, 5);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs6, 6);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs7, 7);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs8, 8);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs9, 9);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs10, 10);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs11, 11);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs12, 12);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs13, 13);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs14, 14);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs15, 15);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs16, 16);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs17, 17);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs18, 18);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs19, 19);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs20, 20);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs21, 21);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs22, 22);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs23, 23);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs24, 24);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs25, 25);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs26, 26);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs27, 27);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs28, 28);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs29, 29);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs30, 30);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs31, 31);

        vr_lhs0 = __dtu_s_tvld_itar(lhs_addr_base, lhs_off_step1);

        vacc0 = __dtu_m_vmm_mode0_f_vs0(vacc0, vr_lhs0, smr);

        __dtu_c_movsr2naccovr(0x1);
      }  // loop K

      // vacc0 = __dtu_l_movr2da(0x40F66666);

      __dtu_l_tvsta_cvt2fp16_rnd(vacc0, output_addr_base, output_off_step1);

      __dtu_s_tvld_itar(rhs_addr_base, rhs_off_back_nextN);

      __dtu_s_tvld_itar(lhs_addr_base, lhs_off_kback);
    }  // loop N
    rhs_addr_base =
        __dtu_c_movsr2targ((vmem_rhs_addr + 1) << 16 | vmem_rhs_addr);
  }  // loop M

  // reset spr
  __dtu_c_movsr2naccovr(naccovr_save);
  __dtu_c_movsr2vab_lv_s(vab_save_lv_s);
  __dtu_c_movsr2vab_l_d(vab_save_l_d);
  __dtu_c_movsr2vab_m_s1(vab_save_m_s1);
  __dtu_c_movsr2vab_m_d(vab_save_m_d);
}


/**
 * @brief kernel impl of dot, m max to 256(actually its unlimited), no need to
 * align with 32
 *
 * @param lhs_addr [M, K], fp16
 * @param rhs_addr [K, N], fp16
 * @param out_addr [M, N], fp16
 *
 */
__device__ __forceinline__ void kernel_dot_m_le256_outfp32(int lhs_addr, int rhs_addr, int out_addr, int M,
                                int K, int N) {
  //   asm volatile(
  //       "    l.ldi16.u r12, 0x0  \n"
  //       "    c.movsr2spr NACCOVR, r12\n"
  //       "    c.movsr2spr VAB_M_S2, r12\n"
  //       "    c.movsr2spr VAB_M_S1, r12\n"
  //       "    c.movsr2spr VAB_M_D, r12\n"
  //       "    c.movsr2spr VAB_LV_S, r12\n"
  //       "    c.movsr2spr VAB_L_D, r12\n"
  //       "    c.movsr2spr VPR, r12\n"
  //       "    c.movsr2spr MPR, r12\n"
  //       "    c.movsr2spr CPR, r12\n"
  //       "    c.movsr2spr LPR, r12\n"
  //       "    c.movsr2spr TCTL, r12\n"
  //       "    c.movsr2spr VMM_VSEL_OVR, r12\n"
  //       "    c.movsr2spr VMM_RAW, r12\n"
  //       "    c.movsr2spr SVMM_SPR0, r12\n"
  //       "    c.movsr2spr LOOP_STS, r12\n"
  //       :
  //       :
  //       : "r12");
  __dtu_c_movsr2vab_lv_s(0x0);
  __dtu_c_movsr2vab_m_s1(0x0);
  __dtu_c_movsr2vab_m_s2(0x0);
  __dtu_c_movsr2vab_m_d(0x0);
  __dtu_c_movsr2vab_l_d(0x0);
  __dtu_c_movsr2tctl(0x0);
  smr_t smr;
  v32f16 vr_rhs0, vr_rhs1, vr_rhs2, vr_rhs3, vr_rhs4, vr_rhs5, vr_rhs6, vr_rhs7,
      vr_rhs8, vr_rhs9, vr_rhs10, vr_rhs11, vr_rhs12, vr_rhs13, vr_rhs14,
      vr_rhs15, vr_rhs16, vr_rhs17, vr_rhs18, vr_rhs19, vr_rhs20, vr_rhs21,
      vr_rhs22, vr_rhs23, vr_rhs24, vr_rhs25, vr_rhs26, vr_rhs27, vr_rhs28,
      vr_rhs29, vr_rhs30, vr_rhs31;
  v32f16 vr_lhs0;
  va16f32x2 vacc0;
  v64i8 vr_out1, vr_out2;

  int naccovr_save = __dtu_movs_naccovr();
  int vab_save_lv_s = __dtu_movs_vab_lv_s();
  int vab_save_l_d = __dtu_movs_vab_l_d();
  int vab_save_m_s1 = __dtu_movs_vab_m_s1();
  int vab_save_m_d = __dtu_movs_vab_m_d();

  int n_step_cnt = N >> 5;
  int vmem_rhs_addr = reinterpret_cast<int>(rhs_addr >> 6);
  tar_t rhs_addr_base =
      __dtu_c_movsr2targ((vmem_rhs_addr + 1) << 16 | vmem_rhs_addr);
  int rhs_nextK = n_step_cnt;
  int rhs_back_nextN = -(K * n_step_cnt - 2);

  tar_t rhs_off_nextK =
      __dtu_c_movsr2tari((rhs_nextK << 16) | rhs_nextK, rhs_addr_base);
  tar_t rhs_off_back_nextN = __dtu_c_movsr2tari(
      (rhs_back_nextN << 16) | (rhs_back_nextN & 0xffff), rhs_addr_base);

  int k_step = (K >> 5);

  int vmem_lhs_addr = reinterpret_cast<int>(lhs_addr >> 6);
  tar_t lhs_addr_base = __dtu_c_movsr2targ((vmem_lhs_addr) << 16 |
                                           vmem_lhs_addr);  // vpt0 == vpt1
  tar_t lhs_off_step1 = __dtu_c_movsr2tari(0x00010001, lhs_addr_base);
  tar_t lhs_off_kback = __dtu_c_movsr2tari(
      ((-k_step) << 16) | ((-k_step) & 0xffff), lhs_addr_base);

  int vmem_output_addr = reinterpret_cast<int>(out_addr >> 6);
  tar_t output_addr_base =
      __dtu_c_movsr2targ((vmem_output_addr + 2) << 16 | vmem_output_addr);
  tar_t output_off_step1 = __dtu_c_movsr2tari(0x00010001, output_addr_base);
  tar_t output_off_step3 = __dtu_c_movsr2tari(0x00030003, output_addr_base);

  for (int m_idx = 0; m_idx < M; m_idx++) {
    lhs_addr_base = __dtu_c_movsr2targ((vmem_lhs_addr + k_step * m_idx) << 16 |
                                       vmem_lhs_addr + k_step * m_idx);
    for (int n_idx = 0; n_idx < (N >> 6); n_idx++) {
      __dtu_c_movsr2naccovr(0x10001);
      for (int k_idx = 0; k_idx < (K >> 5); k_idx++) {
        vr_rhs0 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs1 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs2 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs3 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs4 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs5 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs6 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs7 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs8 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs9 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs10 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs11 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs12 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs13 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs14 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs15 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs16 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs17 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs18 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs19 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs20 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs21 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs22 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs23 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs24 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs25 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs26 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs27 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs28 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs29 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs30 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        vr_rhs31 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs0, 0);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs1, 1);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs2, 2);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs3, 3);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs4, 4);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs5, 5);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs6, 6);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs7, 7);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs8, 8);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs9, 9);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs10, 10);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs11, 11);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs12, 12);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs13, 13);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs14, 14);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs15, 15);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs16, 16);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs17, 17);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs18, 18);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs19, 19);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs20, 20);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs21, 21);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs22, 22);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs23, 23);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs24, 24);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs25, 25);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs26, 26);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs27, 27);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs28, 28);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs29, 29);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs30, 30);
        smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs31, 31);

        vr_lhs0 = __dtu_s_tvld_itar(lhs_addr_base, lhs_off_step1);

        vacc0 = __dtu_m_vmm_mode0_f_vs0(vacc0, vr_lhs0, smr);

        __dtu_c_movsr2naccovr(0x1);
      }  // loop K

      // vacc0 = __dtu_l_movr2da(0x40F66666);

      // __dtu_l_tvsta_cvt2fp16_rnd(vacc0, output_addr_base, output_off_step1);
      vr_out1 = __dtu_l_extractda2vr(vacc0, 0);
      vr_out2 = __dtu_v_extractda2vr(vacc0, 1);

      //   __dtu_l_tvsta_w_d(vacc0, out_addr_base, out_off);
      __dtu_c_tvst_itar_f32(vr_out1, output_addr_base, output_off_step1);
      __dtu_c_tvst_itar_f32(vr_out2, output_addr_base, output_off_step3);

      __dtu_s_tvld_itar(rhs_addr_base, rhs_off_back_nextN);

      __dtu_s_tvld_itar(lhs_addr_base, lhs_off_kback);
    }  // loop N
    rhs_addr_base =
        __dtu_c_movsr2targ((vmem_rhs_addr + 1) << 16 | vmem_rhs_addr);
  }  // loop M

  // reset spr
  __dtu_c_movsr2naccovr(naccovr_save);
  __dtu_c_movsr2vab_lv_s(vab_save_lv_s);
  __dtu_c_movsr2vab_l_d(vab_save_l_d);
  __dtu_c_movsr2vab_m_s1(vab_save_m_s1);
  __dtu_c_movsr2vab_m_d(vab_save_m_d);
}

/**
 * @brief kernel impl of dot, m < 32, k is bbbbig like over 16k, n just normal
 like 4096
 *
 * @param lhs_addr ex: [1, 16k]
 * @param rhs_addr ex: [16k, 4096]
 * @param out_addr ex: [1, 4096]
 *
 */
__device__ __forceinline__  void kernel_dot_batch_m_lt32_fp16(int lhs_addr, int rhs_addr, int out_addr,
                                  int M, int K, int N)
    __attribute__((no_mem_alias_in_tar)) {
  // asm volatile(
  //     "    l.ldi16.u r12, 0x0  \n"
  //     "    c.movsr2spr NACCOVR, r12\n"
  //     "    c.movsr2spr VAB_M_S2, r12\n"
  //     "    c.movsr2spr VAB_M_S1, r12\n"
  //     "    c.movsr2spr VAB_M_D, r12\n"
  //     "    c.movsr2spr VAB_LV_S, r12\n"
  //     "    c.movsr2spr VAB_L_D, r12\n"
  //     "    c.movsr2spr VPR, r12\n"
  //     "    c.movsr2spr MPR, r12\n"
  //     "    c.movsr2spr CPR, r12\n"
  //     "    c.movsr2spr LPR, r12\n"
  //     "    c.movsr2spr TCTL, r12\n"
  //     "    c.movsr2spr VMM_VSEL_OVR, r12\n"
  //     "    c.movsr2spr VMM_RAW, r12\n"
  //     "    c.movsr2spr SVMM_SPR0, r12\n"
  //     "    c.movsr2spr LOOP_STS, r12\n"
  //     :
  //     :
  //     : "r12");
  __dtu_c_movsr2vab_lv_s(0x0);
  __dtu_c_movsr2vab_m_s1(0x0);
  __dtu_c_movsr2vab_m_s2(0x0);
  __dtu_c_movsr2vab_m_d(0x0);
  __dtu_c_movsr2vab_l_d(0x0);
  __dtu_c_movsr2tctl(0x0);
  smr_t smr;
  v32f16 vr_rhs0, vr_rhs1, vr_rhs2, vr_rhs3, vr_rhs4, vr_rhs5, vr_rhs6, vr_rhs7,
      vr_rhs8, vr_rhs9, vr_rhs10, vr_rhs11, vr_rhs12, vr_rhs13, vr_rhs14,
      vr_rhs15, vr_rhs16, vr_rhs17, vr_rhs18, vr_rhs19, vr_rhs20, vr_rhs21,
      vr_rhs22, vr_rhs23, vr_rhs24, vr_rhs25, vr_rhs26, vr_rhs27, vr_rhs28,
      vr_rhs29, vr_rhs30, vr_rhs31;
  v32f16 vr_lhs0;
  va16f32x2 vacc0;  // DACC

  // Special Register Configuration
  int vab_shift = 0;
  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_m_s1(vab_shift);
  __dtu_c_movsr2vab_m_d(vab_shift);

  int n_step_cnt = N >> 5;
  int k_step_cnt = K >> 5;

  //
  // Weight Address/Offset Configuration
  //
  int vmem_rhs_addr = reinterpret_cast<int>(rhs_addr >> 6);
  int rhs_nextK = n_step_cnt;
  int rhs_back_nextN = -(K * n_step_cnt - 2);
  tar_t rhs_addr_base =
      __dtu_c_movsr2targ((vmem_rhs_addr + 1) << 16 | vmem_rhs_addr);
  tar_t rhs_off_nextK =
      __dtu_c_movsr2tari((rhs_nextK << 16) | rhs_nextK, rhs_addr_base);
  tar_t rhs_off_back_nextN = __dtu_c_movsr2tari(
      (rhs_back_nextN << 16) | (rhs_back_nextN & 0xffff), rhs_addr_base);

  //
  // Input Address/Offset Configuration
  //
  int vmem_lhs_addr = reinterpret_cast<int>(lhs_addr >> 6);
  int lhs_nextM = k_step_cnt;
  int lhs_back_nextK = -(M * k_step_cnt - 1);
  tar_t lhs_addr_base = __dtu_c_movsr2targ((vmem_lhs_addr) << 16 |
                                           vmem_lhs_addr);  // vpt0 == vpt1
  tar_t lhs_off_nextM =
      __dtu_c_movsr2tari((lhs_nextM << 16) | lhs_nextM, lhs_addr_base);
  tar_t lhs_off_back_nextK = __dtu_c_movsr2tari(
      (lhs_back_nextK << 16) | (lhs_back_nextK & 0xffff), lhs_addr_base);

  //
  // Output Address Configuraiton
  //
  int vmem_output_addr = reinterpret_cast<int>(out_addr >> 6);
  int output_nextM = n_step_cnt;
  int output_back_nextN = -(M * n_step_cnt - 2);
  tar_t output_addr_base =
      __dtu_c_movsr2targ((vmem_output_addr + 1) << 16 | vmem_output_addr);
  tar_t output_off_step1 = __dtu_c_movsr2tari((2 << 16) | 2, output_addr_base);
  tar_t output_off_nextM =
      __dtu_c_movsr2tari((output_nextM << 16) | output_nextM, output_addr_base);
  tar_t output_off_back_nextN = __dtu_c_movsr2tari(
      (output_back_nextN << 16) | (output_back_nextN & 0xffff),
      output_addr_base);

#pragma clang loop unroll(disable)
  for (int n_idx = 0; n_idx < (N >> 6); n_idx++) {
    int vab_base = n_idx * M * 8;
    __dtu_c_movsr2naccovr(0x10001);

#pragma clang loop unroll(disable)
    for (int k_idx = 0; k_idx < k_step_cnt; k_idx++) {
      vab_shift = vab_base;
      __dtu_c_movsr2vab_m_s1(vab_shift);
      __dtu_c_movsr2vab_m_d(vab_shift);

      vr_rhs0 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs1 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs2 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs3 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs4 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs5 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs6 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs7 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs8 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs9 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs10 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs11 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs12 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs13 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs14 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs15 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs16 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs17 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs18 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs19 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs20 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs21 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs22 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs23 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs24 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs25 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs26 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs27 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs28 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs29 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs30 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs31 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs0, 0);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs1, 1);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs2, 2);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs3, 3);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs4, 4);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs5, 5);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs6, 6);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs7, 7);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs8, 8);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs9, 9);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs10, 10);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs11, 11);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs12, 12);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs13, 13);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs14, 14);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs15, 15);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs16, 16);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs17, 17);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs18, 18);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs19, 19);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs20, 20);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs21, 21);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs22, 22);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs23, 23);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs24, 24);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs25, 25);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs26, 26);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs27, 27);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs28, 28);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs29, 29);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs30, 30);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs31, 31);
#pragma clang loop unroll(disable)
      for (int m_idx = 0; m_idx < M; m_idx++) {
        vr_lhs0 = __dtu_s_tvld_itar(lhs_addr_base, lhs_off_nextM);

        vacc0 = __dtu_m_vmm_mode0_f_vs0(vacc0, vr_lhs0, smr);

        vab_shift += 8;
        __dtu_c_movsr2vab_m_s1(vab_shift);
        __dtu_c_movsr2vab_m_d(vab_shift);
      }  // loop M

      // jump lhs
      __dtu_s_tvld_itar(lhs_addr_base, lhs_off_back_nextK);
      __dtu_c_movsr2naccovr(0x1);
    }  // loop K

    // jump back rhs
    __dtu_s_tvld_itar(rhs_addr_base, rhs_off_back_nextN);
    // reset lhs targ
    lhs_addr_base = __dtu_c_movsr2targ((vmem_lhs_addr) << 16 | vmem_lhs_addr);
  }  // loop N

  // STORE
  for (int m_idx = 0; m_idx < M; m_idx++) {
    vab_shift = m_idx * 8;
    __dtu_c_movsr2vab_lv_s(vab_shift);
    for (int n_idx = 0; n_idx < (N >> 6); n_idx++) {
      __dtu_l_tvsta_cvt2fp16_rnd(vacc0, output_addr_base, output_off_step1);

      vab_shift += M * 8;
      __dtu_c_movsr2vab_lv_s(vab_shift);
    }
  }
}

/**
 * @brief kernel impl of dot, m < 32, k is bbbbig like over 16k, n just normal
 like 4096
 *
 * @param lhs_addr ex: [1, 16k]
 * @param rhs_addr ex: [16k, 4096]
 * @param out_addr ex: [1, 4096]
 *
 */
__device__ __forceinline__ void kernel_dot_batch_m_lt32_outfp32(int lhs_addr, int rhs_addr, int out_addr,
                                     int M, int K, int N)
    __attribute__((no_mem_alias_in_tar)) {
  // asm volatile(
  //     "    l.ldi16.u r12, 0x0  \n"
  //     "    c.movsr2spr NACCOVR, r12\n"
  //     "    c.movsr2spr VAB_M_S2, r12\n"
  //     "    c.movsr2spr VAB_M_S1, r12\n"
  //     "    c.movsr2spr VAB_M_D, r12\n"
  //     "    c.movsr2spr VAB_LV_S, r12\n"
  //     "    c.movsr2spr VAB_L_D, r12\n"
  //     "    c.movsr2spr VPR, r12\n"
  //     "    c.movsr2spr MPR, r12\n"
  //     "    c.movsr2spr CPR, r12\n"
  //     "    c.movsr2spr LPR, r12\n"
  //     "    c.movsr2spr TCTL, r12\n"
  //     "    c.movsr2spr VMM_VSEL_OVR, r12\n"
  //     "    c.movsr2spr VMM_RAW, r12\n"
  //     "    c.movsr2spr SVMM_SPR0, r12\n"
  //     "    c.movsr2spr LOOP_STS, r12\n"
  //     :
  //     :
  //     : "r12");
  __dtu_c_movsr2vab_lv_s(0x0);
  __dtu_c_movsr2vab_m_s1(0x0);
  __dtu_c_movsr2vab_m_s2(0x0);
  __dtu_c_movsr2vab_m_d(0x0);
  __dtu_c_movsr2vab_l_d(0x0);
  __dtu_c_movsr2tctl(0x0);
  smr_t smr;
  v32f16 vr_rhs0, vr_rhs1, vr_rhs2, vr_rhs3, vr_rhs4, vr_rhs5, vr_rhs6, vr_rhs7,
      vr_rhs8, vr_rhs9, vr_rhs10, vr_rhs11, vr_rhs12, vr_rhs13, vr_rhs14,
      vr_rhs15, vr_rhs16, vr_rhs17, vr_rhs18, vr_rhs19, vr_rhs20, vr_rhs21,
      vr_rhs22, vr_rhs23, vr_rhs24, vr_rhs25, vr_rhs26, vr_rhs27, vr_rhs28,
      vr_rhs29, vr_rhs30, vr_rhs31;
  v32f16 vr_lhs0;
  va16f32x2 vacc0;  // DACC

  // Special Register Configuration
  int vab_shift = 0;
  __dtu_c_movsr2vab_lv_s(0);
  __dtu_c_movsr2vab_m_s1(vab_shift);
  __dtu_c_movsr2vab_m_d(vab_shift);

  int n_step_cnt = N >> 5;
  int k_step_cnt = K >> 5;

  //
  // Weight Address/Offset Configuration
  //
  int vmem_rhs_addr = reinterpret_cast<int>(rhs_addr >> 6);
  int rhs_nextK = n_step_cnt;
  int rhs_back_nextN = -(K * n_step_cnt - 2);
  tar_t rhs_addr_base =
      __dtu_c_movsr2targ((vmem_rhs_addr + 1) << 16 | vmem_rhs_addr);
  tar_t rhs_off_nextK =
      __dtu_c_movsr2tari((rhs_nextK << 16) | rhs_nextK, rhs_addr_base);
  tar_t rhs_off_back_nextN = __dtu_c_movsr2tari(
      (rhs_back_nextN << 16) | (rhs_back_nextN & 0xffff), rhs_addr_base);

  //
  // Input Address/Offset Configuration
  //
  int vmem_lhs_addr = reinterpret_cast<int>(lhs_addr >> 6);
  int lhs_nextM = k_step_cnt;
  int lhs_back_nextK = -(M * k_step_cnt - 1);
  tar_t lhs_addr_base = __dtu_c_movsr2targ((vmem_lhs_addr) << 16 |
                                           vmem_lhs_addr);  // vpt0 == vpt1
  tar_t lhs_off_nextM =
      __dtu_c_movsr2tari((lhs_nextM << 16) | lhs_nextM, lhs_addr_base);
  tar_t lhs_off_back_nextK = __dtu_c_movsr2tari(
      (lhs_back_nextK << 16) | (lhs_back_nextK & 0xffff), lhs_addr_base);

  //
  // Output Address Configuraiton
  //
  int vmem_output_addr = reinterpret_cast<int>(out_addr >> 6);
  int output_nextM = n_step_cnt;
  int output_back_nextN = -(M * n_step_cnt - 2);
  tar_t output_addr_base =
      __dtu_c_movsr2targ((vmem_output_addr + 1) << 16 | vmem_output_addr);
  tar_t output_off_step1 = __dtu_c_movsr2tari((2 << 16) | 2, output_addr_base);
  tar_t output_off_nextM =
      __dtu_c_movsr2tari((output_nextM << 16) | output_nextM, output_addr_base);
  tar_t output_off_back_nextN = __dtu_c_movsr2tari(
      (output_back_nextN << 16) | (output_back_nextN & 0xffff),
      output_addr_base);

#pragma clang loop unroll(disable)
  for (int n_idx = 0; n_idx < (N >> 6); n_idx++) {
    int vab_base = n_idx * M * 8;
    __dtu_c_movsr2naccovr(0x10001);

#pragma clang loop unroll(disable)
    for (int k_idx = 0; k_idx < k_step_cnt; k_idx++) {
      vab_shift = vab_base;
      __dtu_c_movsr2vab_m_s1(vab_shift);
      __dtu_c_movsr2vab_m_d(vab_shift);

      vr_rhs0 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs1 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs2 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs3 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs4 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs5 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs6 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs7 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs8 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs9 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs10 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs11 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs12 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs13 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs14 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs15 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs16 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs17 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs18 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs19 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs20 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs21 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs22 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs23 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs24 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs25 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs26 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs27 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs28 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs29 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs30 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      vr_rhs31 = __dtu_s_tivld_itar(rhs_addr_base, rhs_off_nextK);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs0, 0);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs1, 1);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs2, 2);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs3, 3);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs4, 4);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs5, 5);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs6, 6);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs7, 7);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs8, 8);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs9, 9);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs10, 10);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs11, 11);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs12, 12);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs13, 13);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs14, 14);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs15, 15);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs16, 16);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs17, 17);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs18, 18);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs19, 19);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs20, 20);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs21, 21);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs22, 22);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs23, 23);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs24, 24);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs25, 25);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs26, 26);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs27, 27);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs28, 28);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs29, 29);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs30, 30);
      smr = __dtu_m_ldsmr_mode0_f_row(smr, vr_rhs31, 31);
#pragma clang loop unroll(disable)
      for (int m_idx = 0; m_idx < M; m_idx++) {
        vr_lhs0 = __dtu_s_tvld_itar(lhs_addr_base, lhs_off_nextM);

        vacc0 = __dtu_m_vmm_mode0_f_vs0(vacc0, vr_lhs0, smr);

        vab_shift += 8;
        __dtu_c_movsr2vab_m_s1(vab_shift);
        __dtu_c_movsr2vab_m_d(vab_shift);
      }  // loop M

      // jump lhs
      __dtu_s_tvld_itar(lhs_addr_base, lhs_off_back_nextK);
      __dtu_c_movsr2naccovr(0x1);
    }  // loop K

    // jump back rhs
    __dtu_s_tvld_itar(rhs_addr_base, rhs_off_back_nextN);
    // reset lhs targ
    lhs_addr_base = __dtu_c_movsr2targ((vmem_lhs_addr) << 16 | vmem_lhs_addr);
  }  // loop N

  // STORE
  for (int m_idx = 0; m_idx < M; m_idx++) {
    vab_shift = m_idx * 8;
    __dtu_c_movsr2vab_lv_s(vab_shift);
    for (int n_idx = 0; n_idx < (N >> 6); n_idx++) {
      __dtu_l_tvsta_w_d(vacc0, output_addr_base, output_off_step1);

      vab_shift += M * 8;
      __dtu_c_movsr2vab_lv_s(vab_shift);
    }
  }
}

using FP = void (*)(int, int, int, int, int, int);