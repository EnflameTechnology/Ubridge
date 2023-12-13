/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file    op_gemm_tuner.h
 * @brief
 *
 * @author
 * @date    2023-08-18
 * @version V1.0
 * @par     Copyright (c)
 *          Enflame Tech Company.
 * @par     History:
 */
#ifndef OP_ATEN_GEMM_TUNER_H_
#define OP_ATEN_GEMM_TUNER_H_
#include <algorithm>
#include <vector>
#include <utility>
#include <bitset>
#include <string>
#include <iostream>
#include "utils.h"
/** topsopDataType_t */
typedef enum {
  TOPSOP_DATA_NONE = -1,  /**< TOPSOP_DATA_NONE -1  */
  TOPSOP_DATA_I8 = 0,     /**< TOPSOP_DATA_I8 0  */
  TOPSOP_DATA_U8,         /**< TOPSOP_DATA_U8 1  */
  TOPSOP_DATA_I16,        /**< TOPSOP_DATA_I16 2  */
  TOPSOP_DATA_U16,        /**< TOPSOP_DATA_U16 3  */
  TOPSOP_DATA_FP16,       /**< TOPSOP_DATA_FP16 4  */
  TOPSOP_DATA_BF16,       /**< TOPSOP_DATA_BF16 5  */
  TOPSOP_DATA_I32,        /**< TOPSOP_DATA_I32 6  */
  TOPSOP_DATA_U32,        /**< TOPSOP_DATA_U32 7  */
  TOPSOP_DATA_FP32,       /**< TOPSOP_DATA_FP32 8  */
  TOPSOP_DATA_EF32,       /**< TOPSOP_DATA_EF32 9  */
  TOPSOP_DATA_TF32,       /**< TOPSOP_DATA_TF32 10  */
  TOPSOP_DATA_I64,        /**< TOPSOP_DATA_I64 11  */
  TOPSOP_DATA_U64,        /**< TOPSOP_DATA_U64 12  */
  TOPSOP_DATA_F64,        /**< TOPSOP_DATA_F64 13  */
  TOPSOP_DATA_PRED,       /**< TOPSOP_DATA_PRED 14  */
  TOPSOP_DATA_I4,         /**< TOPSOP_DATA_I4 15  */
} topsopDataType_t;

/** topsopActivationMode_t */
typedef enum {
  TOPSOP_ACTIVATION_NONE = 0,         /**< TOPSOP_ACTIVATION_NONE 0  */
  TOPSOP_ACTIVATION_RELU = 1,         /**< TOPSOP_ACTIVATION_RELU 1  */
  TOPSOP_ACTIVATION_SIGMOID = 2,      /**< TOPSOP_ACTIVATION_SIGMOID 2  */
  TOPSOP_ACTIVATION_CLIPPED_RELU = 3, /**< TOPSOP_ACTIVATION_CLIPPED_RELU 3  */
  TOPSOP_ACTIVATION_ELU = 4,          /**< TOPSOP_ACTIVATION_ELU 4  */
  TOPSOP_ACTIVATION_IDENTITY = 5,     /**< TOPSOP_ACTIVATION_IDENTITY 5  */
  TOPSOP_ACTIVATION_TANH = 6,         /**< TOPSOP_ACTIVATION_TANH 6  */
  TOPSOP_ACTIVATION_SWISH = 7,        /**< TOPSOP_ACTIVATION_SWISH 7  */
  TOPSOP_ACTIVATION_LEAKY_RELU = 8,   /**< TOPSOP_ACTIVATION_LEAKY_RELU 8  */
  TOPSOP_ACTIVATION_GELU = 9,         /**< TOPSOP_ACTIVATION_GELU 9  */
  TOPSOP_ACTIVATION_SWIGLU = 10,      /**< TOPSOP_ACTIVATION_SWIGLU 10  */
  TOPSOP_ACTIVATION_HARD_SWISH = 11,  /**< TOPSOP_ACTIVATION_HARD_SWISH 11 */
} topsopActivationMode_t;

// #define CeilDiv(x, y) (((x) + (y)-1) / y)

// // align to a multiple of rhs no less than lhs
// template <typename Int_Type, typename Int_Type2 = Int_Type>
// inline Int_Type AlignUp(Int_Type x, Int_Type2 y) {
//   return CeilDiv(x, y) * y;
// }

// // align to a multiple of rhs no more than lhs
// template <typename Int_Type, typename Int_Type2 = Int_Type>
// inline Int_Type AlignDown(Int_Type x, Int_Type2 y) {
//   return (x / y) * y;
// }

using namespace std;

// #define DEBUG
#define GEMM_EXECUTABLE_NAME_INT8 "scorpio_gemm_opn_int8"
#define GEMM_EXECUTABLE_NAME_FP32 "scorpio_gemm_opn_fp32"
#define GEMM_EXECUTABLE_NAME_FP16 "scorpio_gemm_opn_fp16"

// // Gemm Begin
#define INIT_SPLIT                                                            \
  std::string pattern_type;                                                   \
  int csb_batch = 1, sip_batch = 1, lhs_csb_m = 0, lhs_csb_k = 0,             \
      rhs_csb_k = 0, rhs_csb_n = 0;                                           \
  int sip_m = 0, sip_k = 0, sip_n = 0;                                        \
  bool lhs_tranpose = info.transa, rhs_tranpose = info.transb,                \
       out_tranpose = false;                                                  \
  bool batch_multicore = false, lhs_multicore = false, rhs_multicore = false; \
  bool cdma_lhs_pingpong = false, cdma_rhs_pingpong = false;                  \
  bool sdma_lhs_pingpong = false, sdma_rhs_pingpong = false;                  \
  bool rhs_repeatcopy = false;                                                \
  int M = info.M, N = info.N, K = info.K, B = info.batch;

#define END_SPLIT                              \
  tune->pattern_type = pattern_type;           \
  tune->csb_batch = csb_batch;                 \
  tune->sip_batch = sip_batch;                 \
  tune->lhs_csb_m = lhs_csb_m;                 \
  tune->lhs_csb_k = lhs_csb_k;                 \
  tune->rhs_csb_k = rhs_csb_k;                 \
  tune->rhs_csb_n = rhs_csb_n;                 \
  tune->sip_m = sip_m;                         \
  tune->sip_k = sip_k;                         \
  tune->sip_n = sip_n;                         \
  tune->batch_multicore = batch_multicore;     \
  tune->lhs_multicore = lhs_multicore;         \
  tune->rhs_multicore = rhs_multicore;         \
  tune->lhs_tranpose = lhs_tranpose;           \
  tune->rhs_tranpose = rhs_tranpose;           \
  tune->out_tranpose = out_tranpose;           \
  tune->cdma_lhs_pingpong = cdma_lhs_pingpong; \
  tune->cdma_rhs_pingpong = cdma_rhs_pingpong; \
  tune->sdma_lhs_pingpong = sdma_lhs_pingpong; \
  tune->sdma_rhs_pingpong = sdma_rhs_pingpong; \
  tune->rhs_repeatcopy = rhs_repeatcopy;

#define SET_SPLIT_POLICY(pattern_type_, lhs_csb_m_, lhs_csb_k_, rhs_csb_k_, \
                         rhs_csb_n_, sip_m_, sip_k_, sip_n_)                \
  {                                                                         \
    pattern_type = pattern_type_;                                           \
    lhs_csb_m = lhs_csb_m_;                                                 \
    lhs_csb_k = lhs_csb_k_;                                                 \
    rhs_csb_k = rhs_csb_k_;                                                 \
    rhs_csb_n = rhs_csb_n_;                                                 \
    sip_m = sip_m_;                                                         \
    sip_k = sip_k_;                                                         \
    sip_n = sip_n_;                                                         \
  }

#define SET_SPLIT_OPTION(batch_multicore_, lhs_multicore_, rhs_multicore_, \
                         cdma_lhs_pingpong_, cdma_rhs_pingpong_,           \
                         sdma_lhs_pingpong_, sdma_rhs_pingpong_,           \
                         rhs_repeatcopy_)                                  \
  {                                                                        \
    batch_multicore = batch_multicore_;                                    \
    lhs_multicore = lhs_multicore_;                                        \
    rhs_multicore = rhs_multicore_;                                        \
    cdma_lhs_pingpong = cdma_lhs_pingpong_;                                \
    cdma_rhs_pingpong = cdma_rhs_pingpong_;                                \
    sdma_lhs_pingpong = sdma_lhs_pingpong_;                                \
    sdma_rhs_pingpong = sdma_rhs_pingpong_;                                \
    rhs_repeatcopy = rhs_repeatcopy_;                                      \
  }


typedef struct {
  topsopDataType_t data_type;
  topsopDataType_t out_data_type;
  bool is_batch;
  int64_t batch;
  int64_t M;
  int64_t K;
  int64_t N;
  bool transa;
  bool transb;
} AtenGemmInfo;

typedef struct {
  int64_t csb_batch;
  int64_t sip_batch;
  int64_t lhs_csb_m;
  int64_t lhs_csb_k;
  int64_t rhs_csb_k;
  int64_t rhs_csb_n;

  int64_t sip_m;
  int64_t sip_k;
  int64_t sip_n;

  bool batch_multicore;
  bool lhs_multicore;
  bool rhs_multicore;
  bool cdma_lhs_pingpong;
  bool cdma_rhs_pingpong;
  bool sdma_lhs_pingpong;
  bool sdma_rhs_pingpong;
  bool rhs_repeatcopy;

  bool lhs_tranpose;
  bool rhs_tranpose;
  bool out_tranpose;

  std::string pattern_type;
} AtenGemmTune;

class AtenGemmTuner {
 public:
  AtenGemmTuner();
  ~AtenGemmTuner() {}

  int Tuner(const AtenGemmInfo& info, AtenGemmTune* tune);

  int TunerSGemm(const AtenGemmInfo& info, AtenGemmTune* tune);

  int TunerSGemmF32(const AtenGemmInfo& info, AtenGemmTune* tune);
  int TunerSGemmOpt(const AtenGemmInfo& info, AtenGemmTune* tune);
  int TunerHGemmOpt(const AtenGemmInfo& info, AtenGemmTune* tune);
  int TunerHGemmF16(const AtenGemmInfo& info, AtenGemmTune* tune);
   int TunerInt8GemmOpt(const AtenGemmInfo& info, AtenGemmTune* tune);
   int TunerInt8GemmI8(const AtenGemmInfo& info, AtenGemmTune* tune);

 private:
  std::string op_name_;
};


inline int AtenGemmTuner::Tuner(const AtenGemmInfo& info, AtenGemmTune* tune)
  {
      if (info.data_type == info.out_data_type) {
    if (info.data_type == TOPSOP_DATA_FP32) {
      // TunerSGemmOpt(info, tune);
      TunerSGemmF32(info, tune);
    } else if (info.data_type == TOPSOP_DATA_FP16) {
      // TunerHGemmOpt(info, tune);
      TunerHGemmF16(info, tune);
    } else if (info.data_type == TOPSOP_DATA_BF16) {
      TunerHGemmF16(info, tune);
    } else if (info.data_type == TOPSOP_DATA_I8) {
      // TunerInt8GemmOpt(info, tune);
      TunerInt8GemmI8(info, tune);
    }
  } else {
  }
  return 0;
  }

  inline  int AtenGemmTuner::TunerSGemm(const AtenGemmInfo& info, AtenGemmTune* tune){}

  inline  int AtenGemmTuner::TunerSGemmF32(const AtenGemmInfo& info, AtenGemmTune* tune) {
  INIT_SPLIT
  auto sum_mem = [](int m, int n, int k, int bpe, int bias) {
    return (2 * m * k + m * n + 2 * n * k + bias) * bpe * 2;
  };
  // sum_va_mem = ((m * n / (32 * 64)) * 256 * 16 ) * bpe * 2;
  auto sum_va_mem = [](int m, int n, int bpe) { return (m * n * 2) * bpe * 2; };
  const int32_t unit_sip_m = 32;
  const int32_t unit_sip_n = 64;
  const int32_t unit_sip_k = 32;
  const int32_t bpe = 4;
  int32_t sip_cnt = 6;
  int32_t l1_mem = 0;
  int32_t va_mem = 0;
  // topsDeviceProp_t device_prop;
  // topsGetDeviceProperties(&device_prop, 0);
  // std::string arch_name = device_prop.gcuArchName;
  // if (arch_name == "dtu-enflame-tops--gcu300") {
  if(true) {
    // EFDLOG(TOPS, OP) << "\nIn TunerSGemmOpn, device arch: " << arch_name
    //                  << std::endl;
    l1_mem = 1536 * 1024 - 1024;
    sip_cnt = 12;
    va_mem = 4096 * 16 * 2 /*VPT Thread*/ * 4;
  }
  int32_t l31_m_num = CeilDiv(M, unit_sip_m);
  int32_t l31_n_num = CeilDiv(N, unit_sip_n);
  if ((B >= l31_m_num) && (B >= l31_n_num)) {
    batch_multicore = true;
    lhs_multicore = false;
    rhs_multicore = false;
  } else {
    if (l31_m_num >= l31_n_num) {
      batch_multicore = false;
      lhs_multicore = true;
      rhs_multicore = false;
    } else {
      batch_multicore = false;
      lhs_multicore = false;
      rhs_multicore = true;
    }
  }

  int32_t sip_mem = 0;
  int32_t sip_va_mem = 0;
  int32_t l1_mem_sip_n = 0;
  int32_t va_mem_sip_n = 0;
  int32_t l1_mem_sip_m = 0;
  int32_t va_mem_sip_m = 0;
  int32_t l1_mem_sip_k = 0;
  int32_t K_Align = AlignUp(K, unit_sip_k);
  int32_t N_Align = AlignUp(N, unit_sip_n);

  if (sum_mem(unit_sip_m, unit_sip_n, K_Align, bpe, N_Align) <= l1_mem) {
    sip_k = K_Align;
    if (lhs_multicore || batch_multicore) {
      sip_m = unit_sip_m;
      sip_n = AlignUp(N, unit_sip_n);

      if (sum_mem(sip_m, sip_n, sip_k, bpe, N_Align) > l1_mem ||
          sum_va_mem(sip_m, sip_n, 4) > va_mem) {
        // l1_mem_sip_n =
        //     (l1_mem / 2 / bpe - 2 * K_Align * sip_m) / (2 * K_Align + sip_m);
        l1_mem_sip_n = (l1_mem / 2 / bpe - 2 * K_Align * sip_m - N_Align) /
                       (2 * K_Align + sip_m);
        // GEMM_LOG(l1_mem_sip_n);
        l1_mem_sip_n = AlignDown(l1_mem_sip_n, unit_sip_n);
        va_mem_sip_n = va_mem / 16 / sip_m;
        va_mem_sip_n = AlignDown(va_mem_sip_n, unit_sip_n);
        sip_n = l1_mem_sip_n;
        if (l1_mem_sip_n > va_mem_sip_n) {
          sip_n = va_mem_sip_n;
        }
      }
      // sip_n =64;
      // GEMM_LOG(sip_n);
      if (lhs_multicore) {
        if (2 * sip_n >= N) {
          SET_SPLIT_POLICY("cachediff", lhs_csb_m, lhs_csb_k, rhs_csb_k,
                           rhs_csb_n, sip_m, sip_k, sip_n);
        } else if (l31_m_num <= sip_cnt * 2) {
          SET_SPLIT_POLICY("cachesame", lhs_csb_m, lhs_csb_k, rhs_csb_k,
                           rhs_csb_n, sip_m, sip_k, sip_n);
        } else {
          SET_SPLIT_POLICY("general", lhs_csb_m, lhs_csb_k, rhs_csb_k,
                           rhs_csb_n, sip_m, sip_k, sip_n);
        }
        SET_SPLIT_OPTION(false, true, false, false, false, false, false, false);
      } else if (batch_multicore) {
        SET_SPLIT_POLICY("general", lhs_csb_m, lhs_csb_k, rhs_csb_k, rhs_csb_n,
                         sip_m, sip_k, sip_n);
        SET_SPLIT_OPTION(true, false, false, false, false, false, false, false);
      }
    } else if (rhs_multicore) {
      sip_n = unit_sip_n;
      sip_m = AlignUp(M, unit_sip_m);
      if (sum_mem(sip_m, sip_n, sip_k, bpe, N_Align) > l1_mem ||
          sum_va_mem(sip_m, sip_n, 4) > va_mem) {
        l1_mem_sip_m = (l1_mem / 2 / bpe - 2 * K_Align * sip_n - N_Align) /
                       (2 * K_Align + sip_n);
        l1_mem_sip_m = AlignDown(l1_mem_sip_m, unit_sip_m);
        va_mem_sip_m = va_mem / 16 / sip_n;
        va_mem_sip_m = AlignDown(va_mem_sip_m, unit_sip_m);
        sip_m = l1_mem_sip_m;
        if (l1_mem_sip_m > va_mem_sip_m) {
          sip_m = va_mem_sip_m;
        }
      }
      // sip_m =32;
      // GEMM_LOG(sip_m);
      if (2 * sip_m >= M) {
        SET_SPLIT_POLICY("cachediff", lhs_csb_m, lhs_csb_k, rhs_csb_k,
                         rhs_csb_n, sip_m, sip_k, sip_n);
      } else if (l31_n_num <= sip_cnt * 2) {
        SET_SPLIT_POLICY("cachesame", lhs_csb_m, lhs_csb_k, rhs_csb_k,
                         rhs_csb_n, sip_m, sip_k, sip_n);
      } else {
        SET_SPLIT_POLICY("general", lhs_csb_m, lhs_csb_k, rhs_csb_k, rhs_csb_n,
                         sip_m, sip_k, sip_n);
      }
      SET_SPLIT_OPTION(false, false, true, false, false, false, false, false);
    }
  } else {
    sip_n = unit_sip_n;
    sip_m = unit_sip_m;
    sip_k = K_Align;
    l1_mem_sip_k =
        (l1_mem / 2 / bpe - sip_m * sip_n - N_Align) / (2 * sip_m + 2 * sip_n);
    sip_k = AlignDown(l1_mem_sip_k, unit_sip_k);
    SET_SPLIT_POLICY("general", lhs_csb_m, lhs_csb_k, rhs_csb_k, rhs_csb_n,
                     sip_m, sip_k, sip_n);
    SET_SPLIT_OPTION(batch_multicore, lhs_multicore, rhs_multicore, false,
                     false, false, false, false);
    // GEMM_LOG(sip_k);
  }
  std::cout << "TunerSGemmF32" << std::endl;
  // std::cout << "pattern_type:" << pattern_type << std::endl;
  END_SPLIT
  return 0;
}
  inline  int AtenGemmTuner::TunerSGemmOpt(const AtenGemmInfo& info, AtenGemmTune* tune){}
  inline  int AtenGemmTuner::TunerHGemmOpt(const AtenGemmInfo& info, AtenGemmTune* tune){}
  inline  int AtenGemmTuner::TunerHGemmF16(const AtenGemmInfo& info, AtenGemmTune* tune)
  {
    INIT_SPLIT
  auto sum_mem = [](int m, int n, int k, int bpe, int bias) {
    return (2 * m * k + m * n + 2 * n * k + bias) * bpe * 2;
  };
  // sum_va_mem = ((m * n / (32 * 64)) * 256 * 16 ) * bpe * 2;
  auto sum_va_mem = [](int m, int n, int bpe) { return (m * n * 2) * bpe; };
  const int32_t unit_sip_m = 64;
  const int32_t unit_sip_n = 128;
  const int32_t unit_sip_k = 64;
  const int32_t bpe = 2;
  int32_t sip_cnt = 6;
  int32_t l1_mem = 0;
  int32_t va_mem = 0;
  // topsDeviceProp_t device_prop;
  // topsGetDeviceProperties(&device_prop, 0);
  // std::string arch_name = device_prop.gcuArchName;
  // if (arch_name == "dtu-enflame-tops--gcu300") {
  if (true) {
    // EFDLOG(TOPS, OP) << "\nIn TunerHGemmF16, device arch: " << arch_name
                    //  << std::endl;
    l1_mem = 1536 * 1024 - 1024;
    sip_cnt = 12;
    va_mem = 4096 * 16 * 2 /*VPT Thread*/ * 4;
  }
  int32_t l31_m_num = CeilDiv(M, unit_sip_m);
  int32_t l31_n_num = CeilDiv(N, unit_sip_n);
  if ((B >= l31_m_num) && (B >= l31_n_num)) {
    batch_multicore = true;
    lhs_multicore = false;
    rhs_multicore = false;
  } else {
    if (l31_m_num >= l31_n_num) {
      batch_multicore = false;
      lhs_multicore = true;
      rhs_multicore = false;
    } else {
      batch_multicore = false;
      lhs_multicore = false;
      rhs_multicore = true;
    }
  }
  int32_t sip_mem = 0;
  int32_t sip_va_mem = 0;
  int32_t l1_mem_sip_n = 0;
  int32_t va_mem_sip_n = 0;
  int32_t l1_mem_sip_m = 0;
  int32_t va_mem_sip_m = 0;
  int32_t l1_mem_sip_k = 0;
  int32_t K_Align = AlignUp(K, unit_sip_k);
  int32_t N_Align = AlignUp(N, unit_sip_n);
  if (sum_mem(unit_sip_m, unit_sip_n, K_Align, bpe, N_Align) <= l1_mem) {
    sip_k = K_Align;
    if (lhs_multicore || batch_multicore) {
      sip_m = unit_sip_m;
      sip_n = AlignUp(N, unit_sip_n);
      if (sum_mem(sip_m, sip_n, sip_k, bpe, N_Align) > l1_mem ||
          sum_va_mem(sip_m, sip_n, 4) > va_mem) {
        l1_mem_sip_n = (l1_mem / 2 / bpe - 2 * K_Align * sip_m - N_Align) /
                       (2 * K_Align + sip_m);
        l1_mem_sip_n = AlignDown(l1_mem_sip_n, unit_sip_n);
        va_mem_sip_n = va_mem / 8 / sip_m;
        va_mem_sip_n = AlignDown(va_mem_sip_n, unit_sip_n);
        sip_n = l1_mem_sip_n;
        if (l1_mem_sip_n > va_mem_sip_n) {
          sip_n = va_mem_sip_n;
        }
      }
      // sip_n =128;
      // GEMM_LOG(sip_n);
      if (lhs_multicore) {
        if (2 * sip_n >= N) {
          SET_SPLIT_POLICY("cachediff", lhs_csb_m, lhs_csb_k, rhs_csb_k,
                           rhs_csb_n, sip_m, sip_k, sip_n);
        } else if (l31_m_num <= sip_cnt * 2) {
          SET_SPLIT_POLICY("cachesame", lhs_csb_m, lhs_csb_k, rhs_csb_k,
                           rhs_csb_n, sip_m, sip_k, sip_n);
        } else if (batch_multicore) {
          SET_SPLIT_POLICY("general", lhs_csb_m, lhs_csb_k, rhs_csb_k,
                           rhs_csb_n, sip_m, sip_k, sip_n);
        }
        SET_SPLIT_OPTION(false, true, false, false, false, false, false, false);
      } else {
        SET_SPLIT_POLICY("general", lhs_csb_m, lhs_csb_k, rhs_csb_k, rhs_csb_n,
                         sip_m, sip_k, sip_n);
        SET_SPLIT_OPTION(true, false, false, false, false, false, false, false);
      }
    } else if (rhs_multicore) {
      sip_n = unit_sip_n;
      sip_m = AlignUp(M, unit_sip_m);
      if (sum_mem(sip_m, sip_n, sip_k, bpe, N_Align) > l1_mem ||
          sum_va_mem(sip_m, sip_n, 4) > va_mem) {
        l1_mem_sip_m = (l1_mem / 2 / bpe - 2 * K_Align * sip_n - N_Align) /
                       (2 * K_Align + sip_n);
        l1_mem_sip_m = AlignDown(l1_mem_sip_m, unit_sip_m);
        va_mem_sip_m = va_mem / 8 / sip_n;
        va_mem_sip_m = AlignDown(va_mem_sip_m, unit_sip_m);
        sip_m = l1_mem_sip_m;
        if (l1_mem_sip_m > va_mem_sip_m) {
          sip_m = va_mem_sip_m;
        }
      }
      // sip_m =32;
      // GEMM_LOG(sip_m);
      if (2 * sip_m >= M) {
        SET_SPLIT_POLICY("cachediff", lhs_csb_m, lhs_csb_k, rhs_csb_k,
                         rhs_csb_n, sip_m, sip_k, sip_n);
      } else if (l31_n_num <= sip_cnt * 2) {
        SET_SPLIT_POLICY("cachesame", lhs_csb_m, lhs_csb_k, rhs_csb_k,
                         rhs_csb_n, sip_m, sip_k, sip_n);
      } else {
        SET_SPLIT_POLICY("general", lhs_csb_m, lhs_csb_k, rhs_csb_k, rhs_csb_n,
                         sip_m, sip_k, sip_n);
      }
      SET_SPLIT_OPTION(false, false, true, false, false, false, false, false);
    }
  } else {
    sip_n = unit_sip_n;
    sip_m = unit_sip_m;
    sip_k = K_Align;
    l1_mem_sip_k =
        (l1_mem / 2 / bpe - sip_m * sip_n - N_Align) / (2 * sip_m + 2 * sip_n);
    sip_k = AlignDown(l1_mem_sip_k, unit_sip_k);
    SET_SPLIT_POLICY("general", lhs_csb_m, lhs_csb_k, rhs_csb_k, rhs_csb_n,
                     sip_m, sip_k, sip_n);
    SET_SPLIT_OPTION(batch_multicore, lhs_multicore, rhs_multicore, false,
                     false, false, false, false);
    // GEMM_LOG(sip_k);
  }
  std::cout << "TunerHGemmF16" << endl;
  // std::cout << "pattern_type:" << pattern_type << std::endl;
  END_SPLIT
  return 0;
  }
  inline  int AtenGemmTuner::TunerInt8GemmOpt(const AtenGemmInfo& info, AtenGemmTune* tune)
  {

  }
  inline  int AtenGemmTuner::TunerInt8GemmI8(const AtenGemmInfo& info, AtenGemmTune* tune){
    INIT_SPLIT
  auto sum_mem = [](int m, int n, int k, int bpe, int bias) {
    return (2 * m * k + m * n + 2 * n * k + bias) * bpe * 2;
  };
  auto sum_va_mem = [](int m, int n, int bpe) { return (m * n * 2) * bpe * 2; };
  const int32_t unit_sip_m = 32;
  const int32_t unit_sip_n = 128;
  const int32_t unit_sip_k = 128;
  const int32_t bpe = 1;
  int32_t sip_cnt = 6;
  int32_t l1_mem = 0;
  int32_t va_mem = 0;
  // topsDeviceProp_t device_prop;
  // topsGetDeviceProperties(&device_prop, 0);
  // std::string arch_name = device_prop.gcuArchName;
  // if (arch_name == "dtu-enflame-tops--gcu300") {
  if (true) {
    // EFDLOG(TOPS, OP) << "\nIn TunerSGemmOpn, device arch: " << arch_name
    //                  << std::endl;
    l1_mem = 1536 * 1024 - 1024;
    sip_cnt = 12;
    va_mem = 4096 * 16 * 2 /*VPT Thread*/ * 4;
  }
  int32_t l31_m_num = CeilDiv(M, unit_sip_m);
  int32_t l31_n_num = CeilDiv(N, unit_sip_n);
  if ((B >= l31_m_num) && (B >= l31_n_num)) {
    batch_multicore = true;
    lhs_multicore = false;
    rhs_multicore = false;
  } else {
    if (l31_m_num >= l31_n_num) {
      batch_multicore = false;
      lhs_multicore = true;
      rhs_multicore = false;
    } else {
      batch_multicore = false;
      lhs_multicore = false;
      rhs_multicore = true;
    }
  }
  int32_t sip_mem = 0;
  int32_t sip_va_mem = 0;
  int32_t l1_mem_sip_n = 0;
  int32_t va_mem_sip_n = 0;
  int32_t l1_mem_sip_m = 0;
  int32_t va_mem_sip_m = 0;
  int32_t l1_mem_sip_k = 0;
  int32_t K_Align = AlignUp(K, unit_sip_k);
  int32_t N_Align = AlignUp(N, unit_sip_n);
  if (sum_mem(unit_sip_m, unit_sip_n, K_Align, bpe, N_Align) <= l1_mem) {
    sip_k = K_Align;
    if (lhs_multicore || batch_multicore) {
      sip_m = unit_sip_m;
      sip_n = AlignUp(N, unit_sip_n);
      if (sum_mem(sip_m, sip_n, sip_k, bpe, N_Align) > l1_mem ||
          sum_va_mem(sip_m, sip_n, 4) > va_mem) {
        l1_mem_sip_n = (l1_mem / 2 / bpe - 2 * K_Align * sip_m - N_Align) /
                       (2 * K_Align + sip_m);
        l1_mem_sip_n = AlignDown(l1_mem_sip_n, unit_sip_n);
        va_mem_sip_n = va_mem / 16 / sip_m;
        va_mem_sip_n = AlignDown(va_mem_sip_n, unit_sip_n);
        sip_n = l1_mem_sip_n;
        if (l1_mem_sip_n > va_mem_sip_n) {
          sip_n = va_mem_sip_n;
        }
      }
      // sip_n =128;
      // GEMM_LOG(sip_n);
      if (lhs_multicore) {
        if (2 * sip_n >= N) {
          SET_SPLIT_POLICY("cachediff", lhs_csb_m, lhs_csb_k, rhs_csb_k,
                           rhs_csb_n, sip_m, sip_k, sip_n);
        } else if (l31_m_num <= sip_cnt * 2) {
          SET_SPLIT_POLICY("cachesame", lhs_csb_m, lhs_csb_k, rhs_csb_k,
                           rhs_csb_n, sip_m, sip_k, sip_n);
        } else {
          SET_SPLIT_POLICY("general", lhs_csb_m, lhs_csb_k, rhs_csb_k,
                           rhs_csb_n, sip_m, sip_k, sip_n);
        }
        SET_SPLIT_OPTION(false, true, false, false, false, false, false, false);
      } else if (batch_multicore) {
        SET_SPLIT_POLICY("general", lhs_csb_m, lhs_csb_k, rhs_csb_k, rhs_csb_n,
                         sip_m, sip_k, sip_n);
        SET_SPLIT_OPTION(true, false, false, false, false, false, false, false);
      }
    } else if (rhs_multicore) {
      sip_n = unit_sip_n;
      sip_m = AlignUp(M, unit_sip_m);
      if (sum_mem(sip_m, sip_n, sip_k, bpe, N_Align) > l1_mem ||
          sum_va_mem(sip_m, sip_n, 4) > va_mem) {
        l1_mem_sip_m = (l1_mem / 2 / bpe - 2 * K_Align * sip_n - N_Align) /
                       (2 * K_Align + sip_n);
        l1_mem_sip_m = AlignDown(l1_mem_sip_m, unit_sip_m);
        va_mem_sip_m = va_mem / 16 / sip_n;
        va_mem_sip_m = AlignDown(va_mem_sip_m, unit_sip_m);
        sip_m = l1_mem_sip_m;
        if (l1_mem_sip_m > va_mem_sip_m) {
          sip_m = va_mem_sip_m;
        }
      }
      // sip_m =32;
      // GEMM_LOG(sip_m);
      if (2 * sip_m >= M) {
        SET_SPLIT_POLICY("cachediff", lhs_csb_m, lhs_csb_k, rhs_csb_k,
                         rhs_csb_n, sip_m, sip_k, sip_n);
      } else if (l31_n_num <= sip_cnt * 2) {
        SET_SPLIT_POLICY("cachesame", lhs_csb_m, lhs_csb_k, rhs_csb_k,
                         rhs_csb_n, sip_m, sip_k, sip_n);
      } else {
        SET_SPLIT_POLICY("general", lhs_csb_m, lhs_csb_k, rhs_csb_k, rhs_csb_n,
                         sip_m, sip_k, sip_n);
      }
      SET_SPLIT_OPTION(false, false, true, false, false, false, false, false);
    }
  } else {
    sip_n = unit_sip_n;
    sip_m = unit_sip_m;
    sip_k = K_Align;
    l1_mem_sip_k =
        (l1_mem / 2 / bpe - sip_m * sip_n - N_Align) / (2 * sip_m + 2 * sip_n);
    sip_k = AlignDown(l1_mem_sip_k, unit_sip_k);
    SET_SPLIT_POLICY("general", lhs_csb_m, lhs_csb_k, rhs_csb_k, rhs_csb_n,
                     sip_m, sip_k, sip_n);
    SET_SPLIT_OPTION(batch_multicore, lhs_multicore, rhs_multicore, false,
                     false, false, false, false);
    // GEMM_LOG(sip_k);
  }
  // std::cout << "TunerInt8GemmI8" << std::endl;
  END_SPLIT
  return 0;
  }

  AtenGemmTuner::AtenGemmTuner() {}
#endif  // OP_OPS_DOT_TUNER_H_
