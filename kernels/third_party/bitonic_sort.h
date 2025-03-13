// =============================================================================
//
// Copyright 2021-2023 Enflame. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#ifndef __ATEN_TOPK_BITONIC_SORT_H__
#define __ATEN_TOPK_BITONIC_SORT_H__

#include <tops.h>
#include <tops/tops_runtime.h>

#include "../utils/utils.h"
#include "compute.h"
#include "tops/bfloat.h"
#include "tops/half.h"
#include "../utils/vector_ex.h"

using namespace tops;

#define GET_OFFSET_ARR          \
  for (int i = 0; i < n; i++) { \
    l = i ^ j;                  \
    if (l > i) {                \
      if ((i & k) == 0) {       \
        max_id[max_p] = i;      \
        max_id[max_q] = l;      \
        max_p += 2;             \
        max_q += 2;             \
      } else {                  \
        min_id[min_p] = i;      \
        min_id[min_q] = l;      \
        min_p += 2;             \
        min_q += 2;             \
      }                         \
    }                           \
  }

#define GET_ORDERED_VAL_AND_IDX(kernel_getm_prev, kernel_getm_next,          \
                                kernel_cmp_prev, kernel_cmp_next, off_id)    \
                                                                             \
  val_result[0] = kernel_getm_prev<DESCENDING, VType>(val_qa[0], val_qa[1]); \
  val_result[1] = kernel_getm_next<DESCENDING, VType>(val_qa[0], val_qa[1]); \
  val_result[2] = kernel_getm_prev<DESCENDING, VType>(val_qa[2], val_qa[3]); \
  val_result[3] = kernel_getm_next<DESCENDING, VType>(val_qa[2], val_qa[3]); \
  val_result[4] = kernel_getm_prev<DESCENDING, VType>(val_qa[4], val_qa[5]); \
  val_result[5] = kernel_getm_next<DESCENDING, VType>(val_qa[4], val_qa[5]); \
  val_result[6] = kernel_getm_prev<DESCENDING, VType>(val_qa[6], val_qa[7]); \
  val_result[7] = kernel_getm_next<DESCENDING, VType>(val_qa[6], val_qa[7]); \
                                                                             \
  vmask0 = kernel_cmp_prev<DESCENDING, VType, MT>(val_qa[0], val_qa[1]);     \
  idx_result[0] = vselect_t(vmask0, idx_qa[0], idx_qa[1]);                   \
  idx_result[1] = vselect_f(vmask0, idx_qa[0], idx_qa[1]);                   \
  vmask1 = kernel_cmp_prev<DESCENDING, VType, MT>(val_qa[2], val_qa[3]);     \
  idx_result[2] = vselect_t(vmask1, idx_qa[2], idx_qa[3]);                   \
  idx_result[3] = vselect_f(vmask1, idx_qa[2], idx_qa[3]);                   \
  vmask2 = kernel_cmp_prev<DESCENDING, VType, MT>(val_qa[4], val_qa[5]);     \
  idx_result[4] = vselect_t(vmask2, idx_qa[4], idx_qa[5]);                   \
  idx_result[5] = vselect_f(vmask2, idx_qa[4], idx_qa[5]);                   \
  vmask3 = kernel_cmp_prev<DESCENDING, VType, MT>(val_qa[6], val_qa[7]);     \
  idx_result[6] = vselect_t(vmask3, idx_qa[6], idx_qa[7]);                   \
  idx_result[7] = vselect_f(vmask3, idx_qa[6], idx_qa[7]);                   \
                                                                             \
  vstore<VType>(val_result[0],                                               \
                value_ptr + off_id[i + 0] * TOPS_VECTOR_LENGTH);             \
  vstore<VType>(val_result[1],                                               \
                value_ptr + off_id[i + 1] * TOPS_VECTOR_LENGTH);             \
  vstore<VType>(val_result[2],                                               \
                value_ptr + off_id[i + 2] * TOPS_VECTOR_LENGTH);             \
  vstore<VType>(val_result[3],                                               \
                value_ptr + off_id[i + 3] * TOPS_VECTOR_LENGTH);             \
  vstore<VType>(val_result[4],                                               \
                value_ptr + off_id[i + 4] * TOPS_VECTOR_LENGTH);             \
  vstore<VType>(val_result[5],                                               \
                value_ptr + off_id[i + 5] * TOPS_VECTOR_LENGTH);             \
  vstore<VType>(val_result[6],                                               \
                value_ptr + off_id[i + 6] * TOPS_VECTOR_LENGTH);             \
  vstore<VType>(val_result[7],                                               \
                value_ptr + off_id[i + 7] * TOPS_VECTOR_LENGTH);             \
                                                                             \
  vstore<IType>(idx_result[0],                                               \
                index_ptr + off_id[i + 0] * TOPS_VECTOR_LENGTH);             \
  vstore<IType>(idx_result[1],                                               \
                index_ptr + off_id[i + 1] * TOPS_VECTOR_LENGTH);             \
  vstore<IType>(idx_result[2],                                               \
                index_ptr + off_id[i + 2] * TOPS_VECTOR_LENGTH);             \
  vstore<IType>(idx_result[3],                                               \
                index_ptr + off_id[i + 3] * TOPS_VECTOR_LENGTH);             \
  vstore<IType>(idx_result[4],                                               \
                index_ptr + off_id[i + 4] * TOPS_VECTOR_LENGTH);             \
  vstore<IType>(idx_result[5],                                               \
                index_ptr + off_id[i + 5] * TOPS_VECTOR_LENGTH);             \
  vstore<IType>(idx_result[6],                                               \
                index_ptr + off_id[i + 6] * TOPS_VECTOR_LENGTH);             \
  vstore<IType>(idx_result[7], index_ptr + off_id[i + 7] * TOPS_VECTOR_LENGTH);

#define LOAD_VALUE_OR_INDEX(qacc_arr, input_ptr, off_id, TYPE) \
  qacc_arr[0] = vload<TYPE>(reinterpret_cast<char*>(           \
      input_ptr + off_id[i + 0] * TOPS_VECTOR_LENGTH));        \
  qacc_arr[1] = vload<TYPE>(reinterpret_cast<char*>(           \
      input_ptr + off_id[i + 1] * TOPS_VECTOR_LENGTH));        \
  qacc_arr[2] = vload<TYPE>(reinterpret_cast<char*>(           \
      input_ptr + off_id[i + 2] * TOPS_VECTOR_LENGTH));        \
  qacc_arr[3] = vload<TYPE>(reinterpret_cast<char*>(           \
      input_ptr + off_id[i + 3] * TOPS_VECTOR_LENGTH));        \
  qacc_arr[4] = vload<TYPE>(reinterpret_cast<char*>(           \
      input_ptr + off_id[i + 4] * TOPS_VECTOR_LENGTH));        \
  qacc_arr[5] = vload<TYPE>(reinterpret_cast<char*>(           \
      input_ptr + off_id[i + 5] * TOPS_VECTOR_LENGTH));        \
  qacc_arr[6] = vload<TYPE>(reinterpret_cast<char*>(           \
      input_ptr + off_id[i + 6] * TOPS_VECTOR_LENGTH));        \
  qacc_arr[7] = vload<TYPE>(reinterpret_cast<char*>(           \
      input_ptr + off_id[i + 7] * TOPS_VECTOR_LENGTH));

template <typename T>
__device__ void sort_inside(T *val_ptr, u_int32_t *idx_ptr, int k, int total) {
  using VType = typename tops::scalar_to_vector<T, TOPS_VECTOR_LENGTH>::type;
  using UType = typename tops::scalar_to_vector<u_int32_t, TOPS_VECTOR_LENGTH>::type;
  using MT = typename vector_to_mask<VType>::type;

  MT mask;

  VType last_val, cur_val;
  UType last_idx, cur_idx, min_idx;

  for (int i = 0; (i < k) && (i < total - 1); i++) {
    tops::leaptr<VType> val_addr = tops::simple_leaptr<VType>(val_ptr +
      i * TOPS_VECTOR_LENGTH);
    tops::leaptr<UType> idx_addr = tops::simple_leaptr<UType>(idx_ptr +
      i * TOPS_VECTOR_LENGTH);
    tops::leaptr<UType> idx_addr_wr = tops::simple_leaptr<UType>(idx_ptr +
      (i + 1) * TOPS_VECTOR_LENGTH);
    last_val = val_addr.load();
    min_idx = idx_addr.load();

    // index i shall be minimum
    for (int j = i + 1; j < total; j++) {
      cur_val = val_addr.load();
      cur_idx = idx_addr.load();
      // swap if val eq and cur idx is smaller
      mask = tops::mask_and(tops::veq<MT>(cur_val, last_val),
        tops::vlt<MT>(cur_idx, min_idx));
      last_idx = vselect_t(mask, min_idx, cur_idx);
      min_idx = vselect_t(mask, cur_idx, min_idx);
      idx_addr_wr.store(last_idx);

      // early quit, can rearrange with above inst
      mask = tops::vne<MT>(cur_val, last_val);
      int all_neq = __dtu_l_aggr_mb_w_qa_mode1_t0(mask);
      all_neq &= __dtu_l_aggr_mb_w_qa_mode1_t1(mask);
      if (all_neq) {
        break;
      }
    }

    vstore<UType>(min_idx, idx_ptr + i * TOPS_VECTOR_LENGTH);
  }
}

template <typename T>
__device__ void calc_num_kth_val(T *val_ptr, u_int32_t *idx_ptr, u_int32_t *num_ptr,
  int k, int total, int lane) {
  using VType = typename tops::scalar_to_vector<T, TOPS_VECTOR_LENGTH>::type;
  using UType = typename tops::scalar_to_vector<u_int32_t, TOPS_VECTOR_LENGTH>::type;
  using MT = typename vector_to_mask<VType>::type;

  int stride;
  VType cur_val, kth_val;
  UType cur_idx;
  UType qacc_0 = __dtu_l_movr2qa_u32(0);
  UType qacc_1 = __dtu_l_movr2qa_u32(1);
  UType num_of_kth_val = qacc_1;
  UType qacc_delta = __dtu_l_movr2qa_u32(total - k);
  UType qacc_iota = __dtu_m_mid_m0_u32(0);
  UType qacc_lane = __dtu_l_movr2qa_u32(lane);
  UType qacc_max = __dtu_l_movr2qa_u32(0x7FFFFFFF);

  generic_ptr val_addr0 = reinterpret_cast<generic_ptr>(val_ptr +
    (total - 1) * TOPS_VECTOR_LENGTH);
  generic_ptr val_addr1 = reinterpret_cast<generic_ptr>(val_ptr +
    (total - 1) * TOPS_VECTOR_LENGTH + TOPS_VECTOR_LENGTH / 2);
  stride = -1 * TOPS_VECTOR_LENGTH * static_cast<int>(sizeof(T));
  tops::leaptr<VType, 1> val_leaptr =
    tops::leaptr<VType, 1>(val_addr0, stride, val_addr1, stride);

  generic_ptr idx_addr0 = reinterpret_cast<generic_ptr>(idx_ptr +
    (total - 1) * TOPS_VECTOR_LENGTH);
  generic_ptr idx_addr1 = reinterpret_cast<generic_ptr>(idx_ptr +
    (total - 1) * TOPS_VECTOR_LENGTH + TOPS_VECTOR_LENGTH / 2);
  stride = -1 * TOPS_VECTOR_LENGTH * static_cast<int>(sizeof(int));
  tops::leaptr<UType, 1> idx_leaptr =
    tops::leaptr<UType, 1>(idx_addr0, stride, idx_addr1, stride);
  tops::leaptr<UType, 1> idx_leaptr_wr =
    tops::leaptr<UType, 1>(idx_addr0, stride, idx_addr1, stride);

  // calc number of kth val
  kth_val = val_leaptr.load();
  for (int i = total - 2; i >= 0; i--) {
    cur_val = val_leaptr.load();
    cur_idx = idx_leaptr.load();
    MT mask = tops::veq<MT>(cur_val, kth_val);
    num_of_kth_val = vadd_t(mask, num_of_kth_val, qacc_1, num_of_kth_val);
    // invalidate index
    cur_idx = vselect_t(mask, qacc_max, cur_idx);
    idx_leaptr_wr.store(cur_idx);
  }

  // exclude those not need
  qacc_delta = vmin(num_of_kth_val, qacc_delta);
  num_of_kth_val = vsub(num_of_kth_val, qacc_delta);

  // exclude empty lanes
  num_of_kth_val = vselect_t(tops::vlt<MT>(qacc_iota, qacc_lane),
    num_of_kth_val, qacc_0);

  vstore<UType>(num_of_kth_val, num_ptr);
}


template <typename T>
__device__ int collect_index_outside(T *val_ptr, u_int32_t *idx_ptr,
  int k, T *input_ptr, u_int32_t *num_ptr, int axis_size, int global_base) {
  using VType = typename tops::scalar_to_vector<T, TOPS_VECTOR_LENGTH>::type;
  using UType = typename tops::scalar_to_vector<u_int32_t, TOPS_VECTOR_LENGTH>::type;
  using MT = typename vector_to_mask<VType>::type;

  MT mask;
  tops::leaptr<VType> input_addr = tops::simple_leaptr<VType>(input_ptr);

  VType cur_val, kth_val;
  UType qacc_0 = __dtu_l_movr2qa_u32(0);
  UType qacc_1 = __dtu_l_movr2qa_u32(1);
  UType qacc_k = __dtu_l_movr2qa_u32(k);

  UType num_of_kth_val = vload<UType>(num_ptr);
  kth_val = vload<VType>(val_ptr + (k - 1) * TOPS_VECTOR_LENGTH);

  int idx_bpe = 4;
  UType qacc_iota = __dtu_m_mid_m0_u32(0);
  UType qacc_idx_bpe = __dtu_l_movr2qa_u32(idx_bpe);
  // 0, 4, 8, ...
  UType qacc_lane = __dtu_m_mop_mul_u32_qa(qacc_iota, qacc_idx_bpe);
  UType qacc_128_idx_bpe = __dtu_l_movr2qa_u32(idx_bpe * TOPS_VECTOR_LENGTH);
  UType idx = __dtu_l_movr2qa_u32(global_base);

  int all_zero = 0;

  // correct index of kth val
  for (int i = 0; i < axis_size; i++) {
    cur_val = input_addr.load();
    mask = tops::mask_and(tops::veq<MT>(cur_val, kth_val),
      tops::vgt<MT>(num_of_kth_val, qacc_0));

    // conditional scatter
    UType off = __dtu_m_mop_sub_u32_qa(qacc_k, num_of_kth_val);
    off = __dtu_m_mop_mul_u32_qa(off, qacc_128_idx_bpe);
    off = __dtu_m_mop_add_u32_qa(off, qacc_lane);
    vscatter_t(mask, idx_ptr, idx, off);
    idx = vadd(idx, qacc_1);

    // decrease num
    num_of_kth_val = vsub_t(mask, num_of_kth_val, qacc_1, num_of_kth_val);

    // early quit
    mask = tops::veq<MT>(num_of_kth_val, qacc_0);
    all_zero = __dtu_l_aggr_mb_w_qa_mode1_t0(mask);
    all_zero &= __dtu_l_aggr_mb_w_qa_mode1_t1(mask);
    if (all_zero) {
      break;
    }
  }

  vstore<UType>(num_of_kth_val, num_ptr);
  return all_zero;
}


template <typename T, bool DESCENDING>
__device__ void biconic_sort_first(T* value_ptr, u_int32_t* index_ptr,
                                                int base, int n) {
  using VType = typename tops::scalar_to_vector<T, TOPS_VECTOR_LENGTH>::type;
  using IType = typename tops::scalar_to_vector<u_int32_t, TOPS_VECTOR_LENGTH>::type;
  using MT = typename vector_to_mask<VType>::type;

  VType val_qa[8], val_result[8];
  IType idx_qa[8], idx_result[8];
  IType qacc_iota, qacc_base, qacc_n, qacc_off;
  MT vmask0, vmask1, vmask2, vmask3;

  qacc_iota = __dtu_m_mid_m0_u32(0);
  qacc_base = __dtu_l_movr2qa_u32(base);
  qacc_n = __dtu_l_movr2qa_u32(n);
  qacc_off = __dtu_m_mop_mul_u32_qa(qacc_iota, qacc_n);
  qacc_off = __dtu_m_mop_add_u32_qa(qacc_off, qacc_base);

  int k = 2, j = 1, l;
  int max_id[2048], max_p = 0, max_q = 1;
  int min_id[2048], min_p = 0, min_q = 1;

  GET_OFFSET_ARR;

  for (int i = 0; i < n / 2; i += 8) {
    LOAD_VALUE_OR_INDEX(val_qa, value_ptr, max_id, VType);

    idx_qa[0] = __dtu_l_movr2qa_u32(max_id[i]);
    idx_qa[1] = __dtu_l_movr2qa_u32(max_id[i + 1]);
    idx_qa[2] = __dtu_l_movr2qa_u32(max_id[i + 2]);
    idx_qa[3] = __dtu_l_movr2qa_u32(max_id[i + 3]);
    idx_qa[4] = __dtu_l_movr2qa_u32(max_id[i + 4]);
    idx_qa[5] = __dtu_l_movr2qa_u32(max_id[i + 5]);
    idx_qa[6] = __dtu_l_movr2qa_u32(max_id[i + 6]);
    idx_qa[7] = __dtu_l_movr2qa_u32(max_id[i + 7]);

    idx_qa[0] = __dtu_m_mop_add_u32_qa(idx_qa[0], qacc_off);
    idx_qa[1] = __dtu_m_mop_add_u32_qa(idx_qa[1], qacc_off);
    idx_qa[2] = __dtu_m_mop_add_u32_qa(idx_qa[2], qacc_off);
    idx_qa[3] = __dtu_m_mop_add_u32_qa(idx_qa[3], qacc_off);
    idx_qa[4] = __dtu_m_mop_add_u32_qa(idx_qa[4], qacc_off);
    idx_qa[5] = __dtu_m_mop_add_u32_qa(idx_qa[5], qacc_off);
    idx_qa[6] = __dtu_m_mop_add_u32_qa(idx_qa[6], qacc_off);
    idx_qa[7] = __dtu_m_mop_add_u32_qa(idx_qa[7], qacc_off);

    GET_ORDERED_VAL_AND_IDX(kernel_getV0, kernel_getV1, kernel_cmp0,
                            kernel_cmp1, max_id);
  }

  for (int i = 0; i < n / 2; i += 8) {
    LOAD_VALUE_OR_INDEX(val_qa, value_ptr, min_id, VType)

    idx_qa[0] = __dtu_l_movr2qa_u32(min_id[i]);
    idx_qa[1] = __dtu_l_movr2qa_u32(min_id[i + 1]);
    idx_qa[2] = __dtu_l_movr2qa_u32(min_id[i + 2]);
    idx_qa[3] = __dtu_l_movr2qa_u32(min_id[i + 3]);
    idx_qa[4] = __dtu_l_movr2qa_u32(min_id[i + 4]);
    idx_qa[5] = __dtu_l_movr2qa_u32(min_id[i + 5]);
    idx_qa[6] = __dtu_l_movr2qa_u32(min_id[i + 6]);
    idx_qa[7] = __dtu_l_movr2qa_u32(min_id[i + 7]);

    idx_qa[0] = __dtu_m_mop_add_u32_qa(idx_qa[0], qacc_off);
    idx_qa[1] = __dtu_m_mop_add_u32_qa(idx_qa[1], qacc_off);
    idx_qa[2] = __dtu_m_mop_add_u32_qa(idx_qa[2], qacc_off);
    idx_qa[3] = __dtu_m_mop_add_u32_qa(idx_qa[3], qacc_off);
    idx_qa[4] = __dtu_m_mop_add_u32_qa(idx_qa[4], qacc_off);
    idx_qa[5] = __dtu_m_mop_add_u32_qa(idx_qa[5], qacc_off);
    idx_qa[6] = __dtu_m_mop_add_u32_qa(idx_qa[6], qacc_off);
    idx_qa[7] = __dtu_m_mop_add_u32_qa(idx_qa[7], qacc_off);

    GET_ORDERED_VAL_AND_IDX(kernel_getV1, kernel_getV0, kernel_cmp1,
                            kernel_cmp0, min_id);
  }
}

template <typename T, bool DESCENDING>
__device__ void biconic_sort_middle(T* value_ptr, u_int32_t* index_ptr,
                                                 int n) {
  using VType = typename tops::scalar_to_vector<T, TOPS_VECTOR_LENGTH>::type;
  using IType = typename tops::scalar_to_vector<u_int32_t, TOPS_VECTOR_LENGTH>::type;
  using MT = typename vector_to_mask<VType>::type;

  VType val_qa[8], val_result[8];
  IType idx_qa[8], idx_result[8];
  IType qacc_iota, qacc_128, qacc_n;
  MT vmask0, vmask1, vmask2, vmask3;

  qacc_iota = __dtu_m_mid_m0_u32(0);
  qacc_128 = __dtu_l_movr2qa_u32(128);
  qacc_n = __dtu_l_movr2qa_u32(n);

  int l;
  int max_id[2048], max_p = 0, max_q = 1;
  int min_id[2048], min_p = 0, min_q = 1;
  for (int k = 4; k < n; k = k << 1) {
    for (int j = k / 2; j > 0; j = j >> 1) {
      max_p = 0;
      max_q = 1;
      min_p = 0;
      min_q = 1;
      GET_OFFSET_ARR;

      for (int i = 0; i < n / 2; i += 8) {
        LOAD_VALUE_OR_INDEX(val_qa, value_ptr, max_id, VType);
        LOAD_VALUE_OR_INDEX(idx_qa, index_ptr, max_id, IType);
        GET_ORDERED_VAL_AND_IDX(kernel_getV0, kernel_getV1, kernel_cmp0,
                                kernel_cmp1, max_id);
      }

      for (int i = 0; i < n / 2; i += 8) {
        LOAD_VALUE_OR_INDEX(val_qa, value_ptr, min_id, VType)
        LOAD_VALUE_OR_INDEX(idx_qa, index_ptr, min_id, IType);
        GET_ORDERED_VAL_AND_IDX(kernel_getV1, kernel_getV0, kernel_cmp1,
                                kernel_cmp0, min_id);
      }
    }
  }
}

template <typename T, bool DESCENDING>
__device__ void biconic_sort_merge(T* value_ptr, u_int32_t* index_ptr,
                                                int n) {
  using VType = typename tops::scalar_to_vector<T, TOPS_VECTOR_LENGTH>::type;
  using IType = typename tops::scalar_to_vector<u_int32_t, TOPS_VECTOR_LENGTH>::type;
  using MT = typename vector_to_mask<VType>::type;

  VType val_qa[8], val_result[8];
  IType idx_qa[8], idx_result[8];
  IType qacc_iota, qacc_128, qacc_n;
  MT vmask0, vmask1, vmask2, vmask3;

  qacc_iota = __dtu_m_mid_m0_u32(0);
  qacc_128 = __dtu_l_movr2qa_u32(128);
  qacc_n = __dtu_l_movr2qa_u32(n);

  int l;
  int max_id[2048], max_p = 0, max_q = 1;
  int min_id[2048], min_p = 0, min_q = 1;
  int k = n;
  for (int j = n / 2; j > 0; j = j >> 1) {
    max_p = 0;
    max_q = 1;
    min_p = 0;
    min_q = 1;
    GET_OFFSET_ARR;

    for (int i = 0; i < n; i += 8) {
      LOAD_VALUE_OR_INDEX(val_qa, value_ptr, max_id, VType);
      LOAD_VALUE_OR_INDEX(idx_qa, index_ptr, max_id, IType);
      GET_ORDERED_VAL_AND_IDX(kernel_getV0, kernel_getV1, kernel_cmp0,
                              kernel_cmp1, max_id);
    }
  }
}

template <typename T, bool DESCENDING>
__device__ void bitonic_block_select(T* input_ptr, u_int32_t* index_ptr,
                                                  int n) {
  using VType = typename tops::scalar_to_vector<T, TOPS_VECTOR_LENGTH>::type;
  using IType = typename tops::scalar_to_vector<u_int32_t, TOPS_VECTOR_LENGTH>::type;
  using MT = typename vector_to_mask<VType>::type;

  VType val_qa[8], val_result[4];
  IType idx_qa[8], idx_result[4];
  MT vmask0, vmask1, vmask2, vmask3;

  int val_bpe = sizeof(T);
  int idx_bpe = sizeof(u_int32_t);
  int v_stride = 2* TOPS_VECTOR_LENGTH * val_bpe;
  int i_stride = 2 * TOPS_VECTOR_LENGTH * idx_bpe;
  int v_stride_0 = -2 * TOPS_VECTOR_LENGTH * val_bpe;
  int i_stride_0 = -2 * TOPS_VECTOR_LENGTH * idx_bpe;
  int v_stride1 = -6 * TOPS_VECTOR_LENGTH * val_bpe;
  int i_stride1 = -6 * TOPS_VECTOR_LENGTH * idx_bpe;

  generic_ptr value_addr = reinterpret_cast<generic_ptr>(input_ptr);
  generic_ptr value1_addr =
      reinterpret_cast<generic_ptr>(input_ptr + TOPS_VECTOR_LENGTH);
  generic_ptr index_addr = reinterpret_cast<generic_ptr>(index_ptr);
  generic_ptr index1_addr =
      reinterpret_cast<generic_ptr>(index_ptr + TOPS_VECTOR_LENGTH);

  tops::leaptr<VType, 2> value_leaptr =
      tops::leaptr<VType, 2>(value_addr, v_stride, value1_addr, v_stride);
  tops::leaptr<IType, 2> index_leaptr =
      tops::leaptr<IType, 2>(index_addr, i_stride, index1_addr, i_stride);
  value_leaptr.template set_stride<1>(v_stride1, v_stride1);
  index_leaptr.template set_stride<1>(i_stride1, i_stride1);

  T* val_cmp_ptr =
      input_ptr + (n - 1) * TOPS_VECTOR_LENGTH + TOPS_VECTOR_LENGTH / 2;
  u_int32_t* idx_cmp_ptr =
      index_ptr + (n - 1) * TOPS_VECTOR_LENGTH + TOPS_VECTOR_LENGTH / 2;
  T* val1_cmp_ptr =
      input_ptr + (n - 2) * TOPS_VECTOR_LENGTH + TOPS_VECTOR_LENGTH / 2;
  u_int32_t* idx1_cmp_ptr =
      index_ptr + (n - 2) * TOPS_VECTOR_LENGTH + TOPS_VECTOR_LENGTH / 2;
  generic_ptr value_cmp_addr = reinterpret_cast<generic_ptr>(val_cmp_ptr);
  generic_ptr value1_cmp_addr = reinterpret_cast<generic_ptr>(val1_cmp_ptr);
  generic_ptr index_cmp_addr = reinterpret_cast<generic_ptr>(idx_cmp_ptr);
  generic_ptr index1_cmp_addr = reinterpret_cast<generic_ptr>(idx1_cmp_ptr);
  tops::leaptr<VType, 1> value_1_leaptr = tops::leaptr<VType, 1>(
      value_cmp_addr, v_stride_0, value1_cmp_addr, v_stride_0);
  tops::leaptr<IType, 1> index_1_leaptr = tops::leaptr<IType, 1>(
      index_cmp_addr, i_stride_0, index1_cmp_addr, i_stride_0);

  for (int i = 0; i < n / 2; i += 4) {
    val_qa[0] = value_leaptr.template load<0>();
    val_qa[2] = value_leaptr.template load<0>();
    val_qa[4] = value_leaptr.template load<0>();
    val_qa[6] = value_leaptr.template load<1>();
    val_qa[1] = value_1_leaptr.template load<0>();
    val_qa[3] = value_1_leaptr.template load<0>();
    val_qa[5] = value_1_leaptr.template load<0>();
    val_qa[7] = value_1_leaptr.template load<0>();

    idx_qa[0] = index_leaptr.template load<0>();
    idx_qa[2] = index_leaptr.template load<0>();
    idx_qa[4] = index_leaptr.template load<0>();
    idx_qa[6] = index_leaptr.template load<1>();
    idx_qa[1] = index_1_leaptr.template load<0>();
    idx_qa[3] = index_1_leaptr.template load<0>();
    idx_qa[5] = index_1_leaptr.template load<0>();
    idx_qa[7] = index_1_leaptr.template load<0>();

    val_result[0] = kernel_getV0<DESCENDING, VType>(val_qa[0], val_qa[1]);
    val_result[1] = kernel_getV0<DESCENDING, VType>(val_qa[2], val_qa[3]);
    val_result[2] = kernel_getV0<DESCENDING, VType>(val_qa[4], val_qa[5]);
    val_result[3] = kernel_getV0<DESCENDING, VType>(val_qa[6], val_qa[7]);

    vmask0 = kernel_cmp0<DESCENDING, VType, MT>(val_qa[0], val_qa[1]);
    idx_result[0] = vselect_t(vmask0, idx_qa[0], idx_qa[1]);
    vmask1 = kernel_cmp0<DESCENDING, VType, MT>(val_qa[2], val_qa[3]);
    idx_result[1] = vselect_t(vmask1, idx_qa[2], idx_qa[3]);
    vmask2 = kernel_cmp0<DESCENDING, VType, MT>(val_qa[4], val_qa[5]);
    idx_result[2] = vselect_t(vmask2, idx_qa[4], idx_qa[5]);
    vmask3 = kernel_cmp0<DESCENDING, VType, MT>(val_qa[6], val_qa[7]);
    idx_result[3] = vselect_t(vmask3, idx_qa[6], idx_qa[7]);

    value_leaptr.template store<0>(val_result[0]);
    value_leaptr.template store<0>(val_result[1]);
    value_leaptr.template store<0>(val_result[2]);
    value_leaptr.template store<0>(val_result[3]);

    index_leaptr.template store<0>(idx_result[0]);
    index_leaptr.template store<0>(idx_result[1]);
    index_leaptr.template store<0>(idx_result[2]);
    index_leaptr.template store<0>(idx_result[3]);
  }
  biconic_sort_merge<T, DESCENDING>(input_ptr, index_ptr, n);
}

template <typename T, bool DESCENDING>
__device__ void bitonic_sort(T* value_ptr, u_int32_t* index_ptr,
                                          int base, int n) {
  biconic_sort_first<T, DESCENDING>(value_ptr, index_ptr, base, n);
  biconic_sort_middle<T, DESCENDING>(value_ptr, index_ptr, n);
  biconic_sort_merge<T, DESCENDING>(value_ptr, index_ptr, n);
}

#endif
