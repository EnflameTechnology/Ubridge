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
#ifndef __ATEN_HEAP_SORT_H__
#define __ATEN_HEAP_SORT_H__

#include <tops.h>
#include <tops/tops_runtime.h>
#include <tcle.h>

#include <tops/bfloat.h>
#include <tops/half.h>

#include "../utils/utils.h"
#include "../utils/vector_ex.h"
#include "compute.h"

using namespace tops;

#define TAR32(low, high) (((low) & 0xffff) | (((high) & 0xffff) << 16))
#define CVT_U8(m) ((m << 24) | (m << 16) | (m << 8) | m)

#define INIT_SMR                               \
  va64u8x4 qa_tmp[16];                         \
  for (int i = 0; i < 2; i++) {                \
    qa_tmp[8 * i] = __dtu_l_vclr_f32_qa();     \
    qa_tmp[8 * i + 1] = __dtu_l_vclr_f32_qa(); \
    qa_tmp[8 * i + 2] = __dtu_l_vclr_f32_qa(); \
    qa_tmp[8 * i + 3] = __dtu_l_vclr_f32_qa(); \
    qa_tmp[8 * i + 4] = __dtu_l_vclr_f32_qa(); \
    qa_tmp[8 * i + 5] = __dtu_l_vclr_f32_qa(); \
    qa_tmp[8 * i + 6] = __dtu_l_vclr_f32_qa(); \
    qa_tmp[8 * i + 7] = __dtu_l_vclr_f32_qa(); \
  }                                            \
  va64u8x2 da_tmp[64];                         \
  smr_t smr_2 = __dtu_v_clrsmr();              \
  INIT_SMR_STEP(0);                            \
  INIT_SMR_STEP(1);                            \
  INIT_SMR_STEP(2);                            \
  INIT_SMR_STEP(3);

#define INIT_SMR_STEP(j)                                                       \
  qa_tmp[4 * j] = __dtu_l_movr2va_qa_u8(qa_tmp[4 * j], j * 16, 2);             \
  qa_tmp[4 * j] = __dtu_l_movr2va_qa_u8(qa_tmp[4 * j], j * 16 + 65, 2);        \
  qa_tmp[4 * j] = __dtu_l_movr2va_qa_u8(qa_tmp[4 * j], j * 16 + 130, 2);       \
  qa_tmp[4 * j] = __dtu_l_movr2va_qa_u8(qa_tmp[4 * j], j * 16 + 195, 2);       \
  qa_tmp[4 * j] = __dtu_l_movr2va_qa_u8(qa_tmp[4 * j], j * 16 + 256, 2);       \
  qa_tmp[4 * j] = __dtu_l_movr2va_qa_u8(qa_tmp[4 * j], j * 16 + 321, 2);       \
  qa_tmp[4 * j] = __dtu_l_movr2va_qa_u8(qa_tmp[4 * j], j * 16 + 386, 2);       \
  qa_tmp[4 * j] = __dtu_l_movr2va_qa_u8(qa_tmp[4 * j], j * 16 + 451, 2);       \
  qa_tmp[4 * j + 1] = __dtu_l_movr2va_qa_u8(qa_tmp[4 * j + 1], j * 16 + 4, 2); \
  qa_tmp[4 * j + 1] =                                                          \
      __dtu_l_movr2va_qa_u8(qa_tmp[4 * j + 1], j * 16 + 69, 2);                \
  qa_tmp[4 * j + 1] =                                                          \
      __dtu_l_movr2va_qa_u8(qa_tmp[4 * j + 1], j * 16 + 134, 2);               \
  qa_tmp[4 * j + 1] =                                                          \
      __dtu_l_movr2va_qa_u8(qa_tmp[4 * j + 1], j * 16 + 199, 2);               \
  qa_tmp[4 * j + 1] =                                                          \
      __dtu_l_movr2va_qa_u8(qa_tmp[4 * j + 1], j * 16 + 260, 2);               \
  qa_tmp[4 * j + 1] =                                                          \
      __dtu_l_movr2va_qa_u8(qa_tmp[4 * j + 1], j * 16 + 325, 2);               \
  qa_tmp[4 * j + 1] =                                                          \
      __dtu_l_movr2va_qa_u8(qa_tmp[4 * j + 1], j * 16 + 390, 2);               \
  qa_tmp[4 * j + 1] =                                                          \
      __dtu_l_movr2va_qa_u8(qa_tmp[4 * j + 1], j * 16 + 455, 2);               \
  qa_tmp[4 * j + 2] = __dtu_l_movr2va_qa_u8(qa_tmp[4 * j + 2], j * 16 + 8, 2); \
  qa_tmp[4 * j + 2] =                                                          \
      __dtu_l_movr2va_qa_u8(qa_tmp[4 * j + 2], j * 16 + 73, 2);                \
  qa_tmp[4 * j + 2] =                                                          \
      __dtu_l_movr2va_qa_u8(qa_tmp[4 * j + 2], j * 16 + 138, 2);               \
  qa_tmp[4 * j + 2] =                                                          \
      __dtu_l_movr2va_qa_u8(qa_tmp[4 * j + 2], j * 16 + 203, 2);               \
  qa_tmp[4 * j + 2] =                                                          \
      __dtu_l_movr2va_qa_u8(qa_tmp[4 * j + 2], j * 16 + 264, 2);               \
  qa_tmp[4 * j + 2] =                                                          \
      __dtu_l_movr2va_qa_u8(qa_tmp[4 * j + 2], j * 16 + 329, 2);               \
  qa_tmp[4 * j + 2] =                                                          \
      __dtu_l_movr2va_qa_u8(qa_tmp[4 * j + 2], j * 16 + 394, 2);               \
  qa_tmp[4 * j + 2] =                                                          \
      __dtu_l_movr2va_qa_u8(qa_tmp[4 * j + 2], j * 16 + 459, 2);               \
  qa_tmp[4 * j + 3] =                                                          \
      __dtu_l_movr2va_qa_u8(qa_tmp[4 * j + 3], j * 16 + 12, 2);                \
  qa_tmp[4 * j + 3] =                                                          \
      __dtu_l_movr2va_qa_u8(qa_tmp[4 * j + 3], j * 16 + 77, 2);                \
  qa_tmp[4 * j + 3] =                                                          \
      __dtu_l_movr2va_qa_u8(qa_tmp[4 * j + 3], j * 16 + 142, 2);               \
  qa_tmp[4 * j + 3] =                                                          \
      __dtu_l_movr2va_qa_u8(qa_tmp[4 * j + 3], j * 16 + 207, 2);               \
  qa_tmp[4 * j + 3] =                                                          \
      __dtu_l_movr2va_qa_u8(qa_tmp[4 * j + 3], j * 16 + 268, 2);               \
  qa_tmp[4 * j + 3] =                                                          \
      __dtu_l_movr2va_qa_u8(qa_tmp[4 * j + 3], j * 16 + 333, 2);               \
  qa_tmp[4 * j + 3] =                                                          \
      __dtu_l_movr2va_qa_u8(qa_tmp[4 * j + 3], j * 16 + 398, 2);               \
  qa_tmp[4 * j + 3] =                                                          \
      __dtu_l_movr2va_qa_u8(qa_tmp[4 * j + 3], j * 16 + 463, 2);               \
                                                                               \
  da_tmp[8 * j] = __dtu_extractqa2da(qa_tmp[4 * j], 0);                        \
  da_tmp[8 * j + 1] = __dtu_extractqa2da(qa_tmp[4 * j], 1);                    \
  da_tmp[8 * j + 2] = __dtu_extractqa2da(qa_tmp[4 * j + 1], 0);                \
  da_tmp[8 * j + 3] = __dtu_extractqa2da(qa_tmp[4 * j + 1], 1);                \
  da_tmp[8 * j + 4] = __dtu_extractqa2da(qa_tmp[4 * j + 2], 0);                \
  da_tmp[8 * j + 5] = __dtu_extractqa2da(qa_tmp[4 * j + 2], 1);                \
  da_tmp[8 * j + 6] = __dtu_extractqa2da(qa_tmp[4 * j + 3], 0);                \
  da_tmp[8 * j + 7] = __dtu_extractqa2da(qa_tmp[4 * j + 3], 1);                \
                                                                               \
  smr_2 = __dtu_m_ldsmr2_mode19_u8_row_va(smr_2, da_tmp[8 * j], 8 * j);        \
  smr_2 =                                                                      \
      __dtu_m_ldsmr2_mode19_u8_row_va(smr_2, da_tmp[8 * j + 1], 8 * j + 1);    \
  smr_2 =                                                                      \
      __dtu_m_ldsmr2_mode19_u8_row_va(smr_2, da_tmp[8 * j + 2], 8 * j + 2);    \
  smr_2 =                                                                      \
      __dtu_m_ldsmr2_mode19_u8_row_va(smr_2, da_tmp[8 * j + 3], 8 * j + 3);    \
  smr_2 =                                                                      \
      __dtu_m_ldsmr2_mode19_u8_row_va(smr_2, da_tmp[8 * j + 4], 8 * j + 4);    \
  smr_2 =                                                                      \
      __dtu_m_ldsmr2_mode19_u8_row_va(smr_2, da_tmp[8 * j + 5], 8 * j + 5);    \
  smr_2 =                                                                      \
      __dtu_m_ldsmr2_mode19_u8_row_va(smr_2, da_tmp[8 * j + 6], 8 * j + 6);    \
  smr_2 = __dtu_m_ldsmr2_mode19_u8_row_va(smr_2, da_tmp[8 * j + 7], 8 * j + 7);

#define ADJUST_DOWN(output_ptr, index_ptr)                                     \
  parent_off_qa = qacc_root_off;                                               \
  parent_off_idx_qa = qacc_root_idx_off;                                       \
  for (int h = 1; h < k; h = h * 2 + 1) {                                      \
    left_node_qa = qacc_1;                                                     \
    right_node_qa = qacc_2;                                                    \
    left_node_qa =                                                             \
        __dtu_m_vmm2_mode19_u8_acc0(left_node_qa, parent_node_vr, smr_2);      \
    right_node_qa =                                                            \
        __dtu_m_vmm2_mode19_u8_acc0(right_node_qa, parent_node_vr, smr_2);     \
    left_off_qa = __dtu_m_mop_mul_u32_qa(left_node_qa, qacc_128_bpe);          \
    left_off_qa = __dtu_m_mop_add_u32_qa(left_off_qa, qacc_root_off);          \
    left_off_idx_qa = __dtu_m_mop_mul_u32_qa(left_node_qa, qacc_128_idx_bpe);  \
    left_off_idx_qa =                                                          \
        __dtu_m_mop_add_u32_qa(left_off_idx_qa, qacc_root_idx_off);            \
    right_off_qa = __dtu_m_mop_mul_u32_qa(right_node_qa, qacc_128_bpe);        \
    right_off_qa = __dtu_m_mop_add_u32_qa(right_off_qa, qacc_root_off);        \
    right_off_idx_qa =                                                         \
        __dtu_m_mop_mul_u32_qa(right_node_qa, qacc_128_idx_bpe);               \
    right_off_idx_qa =                                                         \
        __dtu_m_mop_add_u32_qa(right_off_idx_qa, qacc_root_idx_off);           \
                                                                               \
    val_cur_qa[0] = tops::vgather<VType>(output_ptr, left_off_qa);             \
    val_cur_qa[1] = tops::vgather<VType>(output_ptr, right_off_qa);            \
    idx_cur_qa[0] = tops::vgather<UType>(index_ptr, left_off_idx_qa);          \
    idx_cur_qa[1] = tops::vgather<UType>(index_ptr, right_off_idx_qa);         \
    vmask0 = kernel_cmp1<DESCENDING, VType, MT>(                               \
      nan_to_inf<T, VType, MT>(val_cur_qa[1]),                                 \
      nan_to_inf<T, VType, MT>(val_cur_qa[0]));                                \
    val_child = vselect_t(vmask0, val_cur_qa[1], val_cur_qa[0]);               \
    idx_child = vselect_t(vmask0, idx_cur_qa[1], idx_cur_qa[0]);               \
    child_off_qa = vselect_t(vmask0, right_off_qa, left_off_qa);               \
    child_off_idx_qa = vselect_t(vmask0, right_off_idx_qa, left_off_idx_qa);   \
    child_node_qa = vselect_t(vmask0, right_node_qa, left_node_qa);            \
                                                                               \
    vmask1 = kernel_cmp1<DESCENDING, VType, MT>(                               \
      nan_to_inf<T, VType, MT>(val_child),                                     \
      nan_to_inf<T, VType, MT>(val_parent));                                   \
    val_cur_qa[0] = vselect_t(vmask1, val_child, val_parent);                  \
    val_cur_qa[1] = vselect_t(vmask1, val_parent, val_child);                  \
    idx_cur_qa[0] = vselect_t(vmask1, idx_child, idx_parent);                  \
    idx_cur_qa[1] = vselect_t(vmask1, idx_parent, idx_child);                  \
                                                                               \
    tops::vscatter<VType>(output_ptr, val_cur_qa[0], parent_off_qa);           \
    tops::vscatter<VType>(output_ptr, val_cur_qa[1], child_off_qa);            \
    tops::vscatter<UType>(index_ptr, idx_cur_qa[0], parent_off_idx_qa);        \
    tops::vscatter<UType>(index_ptr, idx_cur_qa[1], child_off_idx_qa);         \
                                                                               \
    idx_parent = idx_cur_qa[1];                                                \
    val_parent = val_cur_qa[1];                                                \
    parent_off_qa = child_off_qa;                                              \
    parent_off_idx_qa = child_off_idx_qa;                                      \
    parent_node_vr = __dtu_l_movva2vr_cvt2u8(child_node_qa);                   \
  }

template <typename T, typename VT, typename MASKTYPE>
__device__ __forceinline__ VT nan_to_inf(VT in) {
  if KERNEL_CONSTEXPR17 (std::is_same<T, float>::value ||
                         std::is_same<T, __fp16>::value ||
                         std::is_same<T, __bf16>::value) {
    T inf = get_pad_value<false, T>();
    MASKTYPE mask;
    mask = tops::visnan<MASKTYPE, VT>(in);
    return tops::vselect_t(mask, tops::vbroadcast<VT>(inf), in);
  } else {
    return in;
  }
}

template <typename T, bool DESCENDING, int BPE>
__attribute__((no_mem_alias_in_vldst_tar, noinline, enable_software_pipeliner,
               enable_bc_resolver, loop_iterator_less_than_1024))
__device__ void
heap_sort(T* input_ptr, T* output_ptr, u_int32_t* index_ptr, int col_index,
          int global_base, int size, int k) {
  using VType = typename tops::scalar_to_vector<T, TOPS_VECTOR_LENGTH>::type;
  using UType = typename tops::scalar_to_vector<u_int32_t, TOPS_VECTOR_LENGTH>::type;
  using MT = typename vector_to_mask<VType>::type;

  tops::leaptr<VType> input_addr = tops::simple_leaptr<VType>(input_ptr);
  tops::leaptr<VType> output_addr = tops::simple_leaptr<VType>(output_ptr);
  tops::leaptr<UType> index_addr = tops::simple_leaptr<UType>(index_ptr);

  VType val, val_root, val_parent, val_child, val_cur_qa[2];
  UType idx, idx_root, idx_cur_qa[2], off_cur_qa[2];
  UType left_off_qa, right_off_qa, parent_off_qa, child_off_qa;
  UType left_off_idx_qa, right_off_idx_qa, parent_off_idx_qa, child_off_idx_qa;
  UType idx_left, idx_right, idx_parent, idx_child;
  UType qacc_bpe, qacc_idx_bpe, qacc_iota, qacc_1, qacc_2;
  MT vmask0, vmask1;

  int idx_bpe = 4;

  qacc_bpe = __dtu_l_movr2qa_u32(BPE);
  qacc_idx_bpe = __dtu_l_movr2qa_u32(idx_bpe);
  qacc_1 = __dtu_l_movr2qa_u32(1);
  qacc_2 = __dtu_l_movr2qa_u32(2);
  qacc_iota = __dtu_m_mid_m0_u32(0);

  int cur_idx = global_base;
  if (col_index == 0) {
    // tops::krt_reset_clock();
    // Build a small heap with the first k numbers, using HeapInsert
    for (int i = global_base; i < global_base + k; i++) {
      val = input_addr.load();
      idx = __dtu_l_movr2qa_u32(i);
      output_addr.store(val);
      index_addr.store(idx);
    }

    for (int index = 1; index < k; index++) {
      for (int child = index; child > 0; child = (child - 1) / 2) {
        int parent = (child - 1) / 2;
        T* output_parent_ptr = output_ptr + parent * TOPS_VECTOR_LENGTH;
        T* output_child_ptr = output_ptr + child * TOPS_VECTOR_LENGTH;
        u_int32_t* index_parent_ptr = index_ptr + parent * TOPS_VECTOR_LENGTH;
        u_int32_t* index_child_ptr = index_ptr + child * TOPS_VECTOR_LENGTH;
        val_cur_qa[0] = vload<VType>(output_parent_ptr);
        val_cur_qa[1] = vload<VType>(output_child_ptr);
        idx_cur_qa[0] = vload<UType>(index_parent_ptr);
        idx_cur_qa[1] = vload<UType>(index_child_ptr);

        vmask0 =
            kernel_cmp1<DESCENDING, VType, MT>(
              nan_to_inf<T, VType, MT>(val_cur_qa[1]),
              nan_to_inf<T, VType, MT>(val_cur_qa[0]));
        val_parent = vselect_t(vmask0, val_cur_qa[1], val_cur_qa[0]);
        val_child = vselect_t(vmask0, val_cur_qa[0], val_cur_qa[1]);
        idx_parent = vselect_t(vmask0, idx_cur_qa[1], idx_cur_qa[0]);
        idx_child = vselect_t(vmask0, idx_cur_qa[0], idx_cur_qa[1]);

        vstore<VType>(val_parent, output_parent_ptr);
        vstore<VType>(val_child, output_child_ptr);
        vstore<UType>(idx_parent, index_parent_ptr);
        vstore<UType>(idx_child, index_child_ptr);
      }
    }
    // tops::krt_close_clock();
    // int cycles = tops::krt_clock();
    cur_idx += k;
  }

  INIT_SMR;

  UType qacc_root_off = __dtu_m_mop_mul_u32_qa(qacc_iota, qacc_bpe);
  UType qacc_root_idx_off = __dtu_m_mop_mul_u32_qa(qacc_iota, qacc_idx_bpe);
  UType qacc_128_bpe = __dtu_l_movr2qa_u32(BPE * TOPS_VECTOR_LENGTH);
  UType qacc_128_idx_bpe = __dtu_l_movr2qa_u32(idx_bpe * TOPS_VECTOR_LENGTH);
  v64u8 parent_node_vr;
  UType left_node_qa, right_node_qa, child_node_qa;

  __dtu_c_movsr2naccovr(0x1);

  // tops::krt_reset_clock();
  for (int i = cur_idx; i < global_base + size; i++) {
    // update root
    val_root = vload<VType>(reinterpret_cast<char*>(output_ptr));
    idx_root = vload<UType>(reinterpret_cast<char*>(index_ptr));

    val = input_addr.load();
    idx = __dtu_l_movr2qa_u32(i);

    vmask0 = kernel_cmp0<DESCENDING, VType, MT>(
      nan_to_inf<T, VType, MT>(val),
      nan_to_inf<T, VType, MT>(val_root));
    val_parent = vselect_t(vmask0, val, val_root);
    idx_parent = vselect_t(vmask0, idx, idx_root);
    parent_node_vr = __dtu_l_movr2va_u8(CVT_U8(0));

    ADJUST_DOWN(output_ptr, index_ptr);
  }
  // tops::krt_close_clock();
  // int cycles = tops::krt_clock();
}

template <typename T, bool DESCENDING, int BPE>
__attribute__((no_mem_alias_in_vldst_tar, noinline, enable_software_pipeliner,
               enable_bc_resolver, loop_iterator_less_than_1024))
__device__ void
heap_sort_reduce(T* input_ptr, u_int32_t* index_ptr, T* val_out_ptr, u_int32_t* idx_out_ptr,
                 int size, int k) {
  using VType = typename tops::scalar_to_vector<T, TOPS_VECTOR_LENGTH>::type;
  using UType = typename tops::scalar_to_vector<u_int32_t, TOPS_VECTOR_LENGTH>::type;
  using MT = typename vector_to_mask<VType>::type;

  tops::leaptr<VType> input_addr = tops::simple_leaptr<VType>(input_ptr);
  tops::leaptr<UType> index_addr = tops::simple_leaptr<UType>(index_ptr);

  VType val, val_root, val_parent, val_child, val_cur_qa[2];
  UType idx, idx_root, idx_cur_qa[2], off_cur_qa[2];
  UType left_off_qa, right_off_qa, parent_off_qa, child_off_qa;
  UType left_off_idx_qa, right_off_idx_qa, parent_off_idx_qa, child_off_idx_qa;
  UType idx_left, idx_right, idx_parent, idx_child;
  UType qacc_bpe, qacc_idx_bpe, qacc_iota, qacc_1, qacc_2;
  MT vmask0, vmask1;

  int idx_bpe = 4;

  qacc_bpe = __dtu_l_movr2qa_u32(BPE);
  qacc_idx_bpe = __dtu_l_movr2qa_u32(idx_bpe);
  qacc_1 = __dtu_l_movr2qa_u32(1);
  qacc_2 = __dtu_l_movr2qa_u32(2);
  qacc_iota = __dtu_m_mid_m0_u32(0);

  UType qacc_root_off = __dtu_m_mop_mul_u32_qa(qacc_iota, qacc_bpe);
  UType qacc_root_idx_off = __dtu_m_mop_mul_u32_qa(qacc_iota, qacc_idx_bpe);
  UType qacc_128_bpe = __dtu_l_movr2qa_u32(BPE * TOPS_VECTOR_LENGTH);
  UType qacc_128_idx_bpe = __dtu_l_movr2qa_u32(idx_bpe * TOPS_VECTOR_LENGTH);
  v64u8 parent_node_vr;
  UType left_node_qa, right_node_qa, child_node_qa;

  INIT_SMR;

  __dtu_c_movsr2naccovr(0x1);
  // tops::krt_reset_clock();
  for (int i = 0; i < size; i++) {
    // update root
    val_root = vload<VType>(reinterpret_cast<char*>(val_out_ptr));
    idx_root = vload<UType>(reinterpret_cast<char*>(idx_out_ptr));

    val = input_addr.load();
    idx = index_addr.load();

    vmask0 = kernel_cmp0<DESCENDING, VType, MT>(
        nan_to_inf<T, VType, MT>(val), nan_to_inf<T, VType, MT>(val_root));
    val_parent = vselect_t(vmask0, val, val_root);
    idx_parent = vselect_t(vmask0, idx, idx_root);
    parent_node_vr = __dtu_l_movr2va_u8(CVT_U8(0));

    ADJUST_DOWN(val_out_ptr, idx_out_ptr);
  }
  // tops::krt_close_clock();
  // int cycles = tops::krt_clock();
}

template <typename T, bool DESCENDING, int BPE>
__attribute__((noinline, enable_software_pipeliner, enable_bc_resolver,
               loop_iterator_less_than_1024)) __device__ void
heap_sort_heapify(T* input_ptr, u_int32_t* index_ptr, T* val_out_ptr,
                  u_int32_t* idx_out_ptr, int k, T inf_value) {
  using VType = typename tops::scalar_to_vector<T, TOPS_VECTOR_LENGTH>::type;
  using UType = typename tops::scalar_to_vector<u_int32_t, TOPS_VECTOR_LENGTH>::type;
  using MT = typename vector_to_mask<VType>::type;

  int idx_bpe = 4;
  T* val_out_addr = val_out_ptr + (k - 1) * TOPS_VECTOR_LENGTH;
  u_int32_t* idx_out_addr = idx_out_ptr + (k - 1) * TOPS_VECTOR_LENGTH;

  int v_stride = -1 * TOPS_VECTOR_LENGTH * BPE;
  int i_stride = -1 * TOPS_VECTOR_LENGTH * idx_bpe;

  generic_ptr value_addr0 = reinterpret_cast<generic_ptr>(val_out_addr);
  generic_ptr index_addr0 = reinterpret_cast<generic_ptr>(idx_out_addr);
  generic_ptr value_addr1 =
      reinterpret_cast<generic_ptr>(val_out_addr + TOPS_VECTOR_LENGTH / 2);
  generic_ptr index_addr1 =
      reinterpret_cast<generic_ptr>(idx_out_addr + TOPS_VECTOR_LENGTH / 2);

  tops::leaptr<VType, 1> output_value_leaptr =
      tops::leaptr<VType, 1>(value_addr0, v_stride, value_addr1, v_stride);
  tops::leaptr<UType, 1> output_index_leaptr =
      tops::leaptr<UType, 1>(index_addr0, i_stride, index_addr1, i_stride);

  VType val, val_root, val_parent, val_child, val_cur_qa[2];
  UType idx, idx_root, idx_cur_qa[2], off_cur_qa[2];
  UType left_off_qa, right_off_qa, parent_off_qa, child_off_qa;
  UType left_off_idx_qa, right_off_idx_qa, parent_off_idx_qa, child_off_idx_qa;
  UType idx_left, idx_right, idx_parent, idx_child;
  UType qacc_bpe, qacc_idx_bpe, qacc_iota, qacc_1, qacc_2;
  MT vmask0, vmask1;

  qacc_bpe = __dtu_l_movr2qa_u32(BPE);
  qacc_idx_bpe = __dtu_l_movr2qa_u32(idx_bpe);
  qacc_1 = __dtu_l_movr2qa_u32(1);
  qacc_2 = __dtu_l_movr2qa_u32(2);
  qacc_iota = __dtu_m_mid_m0_u32(0);

  UType qacc_root_off = __dtu_m_mop_mul_u32_qa(qacc_iota, qacc_bpe);
  UType qacc_root_idx_off = __dtu_m_mop_mul_u32_qa(qacc_iota, qacc_idx_bpe);
  UType qacc_128_bpe = __dtu_l_movr2qa_u32(BPE * TOPS_VECTOR_LENGTH);
  UType qacc_128_idx_bpe = __dtu_l_movr2qa_u32(idx_bpe * TOPS_VECTOR_LENGTH);
  v64u8 parent_node_vr;
  UType left_node_qa, right_node_qa, child_node_qa;
  VType qacc_inf;
  qacc_inf = vbroadcast<VType>(inf_value);

  // heapify
  val_root = vload<VType>(reinterpret_cast<char*>(input_ptr));
  idx_root = vload<UType>(reinterpret_cast<char*>(index_ptr));
  output_value_leaptr.template store<0>(val_root);
  output_index_leaptr.template store<0>(idx_root);

  T* heap_last_val_ptr = input_ptr + (k - 1) * TOPS_VECTOR_LENGTH;
  u_int32_t* heap_last_idx_ptr = index_ptr + (k - 1) * TOPS_VECTOR_LENGTH;
  VType val_last = vload<VType>(reinterpret_cast<char*>(heap_last_val_ptr));
  UType idx_last = vload<UType>(reinterpret_cast<char*>(heap_last_idx_ptr));

  vstore<VType>(val_last, input_ptr);
  vstore<UType>(idx_last, index_ptr);
  vstore<VType>(qacc_inf, heap_last_val_ptr);

  INIT_SMR;
  __dtu_c_movsr2naccovr(0x1);
  // tops::krt_reset_clock();
  // #pragma clang loop unroll_count(2)
  for (int count = k - 1; count > 0; count--) {
    // heapify
    val_parent = vload<VType>(reinterpret_cast<char*>(input_ptr));
    idx_parent = vload<UType>(reinterpret_cast<char*>(index_ptr));
    parent_node_vr = __dtu_l_movr2va_u8(CVT_U8(0));

    ADJUST_DOWN(input_ptr, index_ptr);

    val_root = vload<VType>(reinterpret_cast<char*>(input_ptr));
    idx_root = vload<UType>(reinterpret_cast<char*>(index_ptr));
    output_value_leaptr.template store<0>(val_root);
    output_index_leaptr.template store<0>(idx_root);

    heap_last_val_ptr = input_ptr + (count - 1) * TOPS_VECTOR_LENGTH;
    heap_last_idx_ptr = index_ptr + (count - 1) * TOPS_VECTOR_LENGTH;
    val_last = vload<VType>(reinterpret_cast<char*>(heap_last_val_ptr));
    idx_last = vload<UType>(reinterpret_cast<char*>(heap_last_idx_ptr));

    vstore<VType>(val_last, input_ptr);
    vstore<UType>(idx_last, index_ptr);
    vstore<VType>(qacc_inf, heap_last_val_ptr);
  }

  // tops::krt_close_clock();
  // int cycles = tops::krt_clock();
}

#endif
