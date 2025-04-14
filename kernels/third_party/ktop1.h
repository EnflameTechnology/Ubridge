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
#ifndef __ATEN_KTOP1_H__
#define __ATEN_KTOP1_H__

#include "compute.h"
#include "tcle.h"

using namespace tcle;

#define GET_TOP1_UNSTABLE                                                      \
  tcle::leaptr<VType, 1> input_value_leaptr =                                  \
      tcle::leaptr<VType, 1>(val_in_addr0, v_stride, val_in_addr1, v_stride);  \
                                                                               \
  VType max[16];                                                               \
  IType idx[16];                                                               \
  MType vmask[16];                                                             \
  max[0] = input_value_leaptr.template load<0>();                              \
  max[1] = input_value_leaptr.template load<0>();                              \
  max[2] = input_value_leaptr.template load<0>();                              \
  max[3] = input_value_leaptr.template load<0>();                              \
  max[4] = input_value_leaptr.template load<0>();                              \
  max[5] = input_value_leaptr.template load<0>();                              \
  max[6] = input_value_leaptr.template load<0>();                              \
  max[7] = input_value_leaptr.template load<0>();                              \
  max[8] = input_value_leaptr.template load<0>();                              \
  max[9] = input_value_leaptr.template load<0>();                              \
  max[10] = input_value_leaptr.template load<0>();                             \
  max[11] = input_value_leaptr.template load<0>();                             \
  max[12] = input_value_leaptr.template load<0>();                             \
  max[13] = input_value_leaptr.template load<0>();                             \
  max[14] = input_value_leaptr.template load<0>();                             \
  max[15] = input_value_leaptr.template load<0>();                             \
  idx[0] = qacc_num[0];                                                        \
  idx[1] = qacc_num[1];                                                        \
  idx[2] = qacc_num[2];                                                        \
  idx[3] = qacc_num[3];                                                        \
  idx[4] = qacc_num[4];                                                        \
  idx[5] = qacc_num[5];                                                        \
  idx[6] = qacc_num[6];                                                        \
  idx[7] = qacc_num[7];                                                        \
  idx[8] = qacc_num[8];                                                        \
  idx[9] = qacc_num[9];                                                        \
  idx[10] = qacc_num[10];                                                      \
  idx[11] = qacc_num[11];                                                      \
  idx[12] = qacc_num[12];                                                      \
  idx[13] = qacc_num[13];                                                      \
  idx[14] = qacc_num[14];                                                      \
  idx[15] = qacc_num[15];                                                      \
  int i = 0;                                                                   \
  for (; i < size - 16; i += 16) {                                             \
    VType tmp[16];                                                             \
    IType tmp_idx[16];                                                         \
    tmp[0] = input_value_leaptr.template load<0>();                            \
    tmp[1] = input_value_leaptr.template load<0>();                            \
    tmp[2] = input_value_leaptr.template load<0>();                            \
    tmp[3] = input_value_leaptr.template load<0>();                            \
    tmp[4] = input_value_leaptr.template load<0>();                            \
    tmp[5] = input_value_leaptr.template load<0>();                            \
    tmp[6] = input_value_leaptr.template load<0>();                            \
    tmp[7] = input_value_leaptr.template load<0>();                            \
    tmp[8] = input_value_leaptr.template load<0>();                            \
    tmp[9] = input_value_leaptr.template load<0>();                            \
    tmp[10] = input_value_leaptr.template load<0>();                           \
    tmp[11] = input_value_leaptr.template load<0>();                           \
    tmp[12] = input_value_leaptr.template load<0>();                           \
    tmp[13] = input_value_leaptr.template load<0>();                           \
    tmp[14] = input_value_leaptr.template load<0>();                           \
    tmp[15] = input_value_leaptr.template load<0>();                           \
    tmp_idx[0] = (IType)(i) + 16;                                              \
    tmp_idx[1] = tmp_idx[0] + qacc_num[1];                                     \
    tmp_idx[2] = tmp_idx[0] + qacc_num[2];                                     \
    tmp_idx[3] = tmp_idx[0] + qacc_num[3];                                     \
    tmp_idx[4] = tmp_idx[0] + qacc_num[4];                                     \
    tmp_idx[5] = tmp_idx[0] + qacc_num[5];                                     \
    tmp_idx[6] = tmp_idx[0] + qacc_num[6];                                     \
    tmp_idx[7] = tmp_idx[0] + qacc_num[7];                                     \
    tmp_idx[8] = tmp_idx[0] + qacc_num[8];                                     \
    tmp_idx[9] = tmp_idx[0] + qacc_num[9];                                     \
    tmp_idx[10] = tmp_idx[0] + qacc_num[10];                                   \
    tmp_idx[11] = tmp_idx[0] + qacc_num[11];                                   \
    tmp_idx[12] = tmp_idx[0] + qacc_num[12];                                   \
    tmp_idx[13] = tmp_idx[0] + qacc_num[13];                                   \
    tmp_idx[14] = tmp_idx[0] + qacc_num[14];                                   \
    tmp_idx[15] = tmp_idx[0] + qacc_num[15];                                   \
                                                                               \
    vmask[0] =                                                                 \
        kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[0], tmp[0]);       \
    idx[0] = vsel(vmask[0], idx[0], tmp_idx[0]);                               \
    vmask[1] =                                                                 \
        kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[1], tmp[1]);       \
    idx[1] = vsel(vmask[1], idx[1], tmp_idx[1]);                               \
    vmask[2] =                                                                 \
        kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[2], tmp[2]);       \
    idx[2] = vsel(vmask[2], idx[2], tmp_idx[2]);                               \
    vmask[3] =                                                                 \
        kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[3], tmp[3]);       \
    idx[3] = vsel(vmask[3], idx[3], tmp_idx[3]);                               \
    vmask[4] =                                                                 \
        kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[4], tmp[4]);       \
    idx[4] = vsel(vmask[4], idx[4], tmp_idx[4]);                               \
    vmask[5] =                                                                 \
        kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[5], tmp[5]);       \
    idx[5] = vsel(vmask[5], idx[5], tmp_idx[5]);                               \
    vmask[6] =                                                                 \
        kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[6], tmp[6]);       \
    idx[6] = vsel(vmask[6], idx[6], tmp_idx[6]);                               \
    vmask[7] =                                                                 \
        kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[7], tmp[7]);       \
    idx[7] = vsel(vmask[7], idx[7], tmp_idx[7]);                               \
    vmask[8] =                                                                 \
        kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[8], tmp[8]);       \
    idx[8] = vsel(vmask[8], idx[8], tmp_idx[8]);                               \
    vmask[9] =                                                                 \
        kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[9], tmp[9]);       \
    idx[9] = vsel(vmask[9], idx[9], tmp_idx[9]);                               \
    vmask[10] =                                                                \
        kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[10], tmp[10]);     \
    idx[10] = vsel(vmask[10], idx[10], tmp_idx[10]);                           \
    vmask[11] =                                                                \
        kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[11], tmp[11]);     \
    idx[11] = vsel(vmask[11], idx[11], tmp_idx[11]);                           \
    vmask[12] =                                                                \
        kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[12], tmp[12]);     \
    idx[12] = vsel(vmask[12], idx[12], tmp_idx[12]);                           \
    vmask[13] =                                                                \
        kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[13], tmp[13]);     \
    idx[13] = vsel(vmask[13], idx[13], tmp_idx[13]);                           \
    vmask[14] =                                                                \
        kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[14], tmp[14]);     \
    idx[14] = vsel(vmask[14], idx[14], tmp_idx[14]);                           \
    vmask[15] =                                                                \
        kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[15], tmp[15]);     \
    idx[15] = vsel(vmask[15], idx[15], tmp_idx[15]);                           \
    max[0] = kernel_tcle_get_value<DESCENDING, VType>(max[0], tmp[0]);         \
    max[1] = kernel_tcle_get_value<DESCENDING, VType>(max[1], tmp[1]);         \
    max[2] = kernel_tcle_get_value<DESCENDING, VType>(max[2], tmp[2]);         \
    max[3] = kernel_tcle_get_value<DESCENDING, VType>(max[3], tmp[3]);         \
    max[4] = kernel_tcle_get_value<DESCENDING, VType>(max[4], tmp[4]);         \
    max[5] = kernel_tcle_get_value<DESCENDING, VType>(max[5], tmp[5]);         \
    max[6] = kernel_tcle_get_value<DESCENDING, VType>(max[6], tmp[6]);         \
    max[7] = kernel_tcle_get_value<DESCENDING, VType>(max[7], tmp[7]);         \
    max[8] = kernel_tcle_get_value<DESCENDING, VType>(max[8], tmp[8]);         \
    max[9] = kernel_tcle_get_value<DESCENDING, VType>(max[9], tmp[9]);         \
    max[10] = kernel_tcle_get_value<DESCENDING, VType>(max[10], tmp[10]);      \
    max[11] = kernel_tcle_get_value<DESCENDING, VType>(max[11], tmp[11]);      \
    max[12] = kernel_tcle_get_value<DESCENDING, VType>(max[12], tmp[12]);      \
    max[13] = kernel_tcle_get_value<DESCENDING, VType>(max[13], tmp[13]);      \
    max[14] = kernel_tcle_get_value<DESCENDING, VType>(max[14], tmp[14]);      \
    max[15] = kernel_tcle_get_value<DESCENDING, VType>(max[15], tmp[15]);      \
  }                                                                            \
  vmask[0] = kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[1], max[0]);  \
  max[0] = kernel_tcle_get_value<DESCENDING, VType>(max[0], max[1]);           \
  idx[0] = vsel(vmask[0], idx[1], idx[0]);                                     \
  vmask[1] = kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[3], max[2]);  \
  max[2] = kernel_tcle_get_value<DESCENDING, VType>(max[2], max[3]);           \
  idx[2] = vsel(vmask[1], idx[3], idx[2]);                                     \
  vmask[2] = kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[5], max[4]);  \
  max[4] = kernel_tcle_get_value<DESCENDING, VType>(max[4], max[5]);           \
  idx[4] = vsel(vmask[2], idx[5], idx[4]);                                     \
  vmask[3] = kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[7], max[6]);  \
  max[6] = kernel_tcle_get_value<DESCENDING, VType>(max[6], max[7]);           \
  idx[6] = vsel(vmask[3], idx[7], idx[6]);                                     \
  vmask[4] = kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[9], max[8]);  \
  max[8] = kernel_tcle_get_value<DESCENDING, VType>(max[8], max[9]);           \
  idx[8] = vsel(vmask[4], idx[9], idx[8]);                                     \
  vmask[5] =                                                                   \
      kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[11], max[10]);       \
  max[10] = kernel_tcle_get_value<DESCENDING, VType>(max[10], max[11]);        \
  idx[10] = vsel(vmask[5], idx[11], idx[10]);                                  \
  vmask[6] =                                                                   \
      kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[13], max[12]);       \
  max[12] = kernel_tcle_get_value<DESCENDING, VType>(max[12], max[13]);        \
  idx[12] = vsel(vmask[6], idx[13], idx[12]);                                  \
  vmask[7] =                                                                   \
      kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[15], max[14]);       \
  max[14] = kernel_tcle_get_value<DESCENDING, VType>(max[14], max[15]);        \
  idx[14] = vsel(vmask[7], idx[15], idx[14]);                                  \
                                                                               \
  vmask[0] = kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[2], max[0]);  \
  idx[0] = vsel(vmask[0], idx[2], idx[0]);                                     \
  vmask[1] = kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[6], max[4]);  \
  idx[4] = vsel(vmask[1], idx[6], idx[4]);                                     \
  vmask[2] = kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[10], max[8]); \
  idx[8] = vsel(vmask[2], idx[10], idx[8]);                                    \
  vmask[3] =                                                                   \
      kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[14], max[12]);       \
  idx[12] = vsel(vmask[3], idx[14], idx[12]);                                  \
  max[0] = kernel_tcle_get_value<DESCENDING, VType>(max[0], max[2]);           \
  max[4] = kernel_tcle_get_value<DESCENDING, VType>(max[4], max[6]);           \
  max[8] = kernel_tcle_get_value<DESCENDING, VType>(max[8], max[10]);          \
  max[12] = kernel_tcle_get_value<DESCENDING, VType>(max[12], max[14]);        \
                                                                               \
  vmask[0] = kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[4], max[0]);  \
  idx[0] = vsel(vmask[0], idx[4], idx[0]);                                     \
  vmask[1] = kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[12], max[8]); \
  idx[8] = vsel(vmask[1], idx[12], idx[8]);                                    \
  max[0] = kernel_tcle_get_value<DESCENDING, VType>(max[0], max[4]);           \
  max[8] = kernel_tcle_get_value<DESCENDING, VType>(max[8], max[12]);          \
                                                                               \
  vmask[0] = kernel_tcle_cmp_value<DESCENDING, VType, MType>(max[8], max[0]);  \
  max[0] = kernel_tcle_get_value<DESCENDING, VType>(max[0], max[8]);           \
  idx[0] = vsel(vmask[0], idx[8], idx[0]);                                     \
                                                                               \
  vmask[0] = kernel_tcle_cmp_value<DESCENDING, VType, MType>(val_max, max[0]); \
  val_max = kernel_tcle_get_value<DESCENDING, VType>(max[0], val_max);         \
  idx_off_local = vsel(vmask[0], idx_off_local, idx[0]);

#define GET_TOP1_STABLE_STAGE1                                                 \
  idx_off_local = (IType)(0);                                                  \
  tcle::leaptr<VType, 1> input_value_leaptr =                                  \
      tcle::leaptr<VType, 1>(val_in_addr0, v_stride, val_in_addr1, v_stride);  \
                                                                               \
  in[0] = input_value_leaptr.template load<0>();                               \
  in[1] = input_value_leaptr.template load<0>();                               \
  in[2] = input_value_leaptr.template load<0>();                               \
  in[3] = input_value_leaptr.template load<0>();                               \
  in[4] = input_value_leaptr.template load<0>();                               \
  in[5] = input_value_leaptr.template load<0>();                               \
  in[6] = input_value_leaptr.template load<0>();                               \
  in[7] = input_value_leaptr.template load<0>();                               \
  in[8] = input_value_leaptr.template load<0>();                               \
  in[9] = input_value_leaptr.template load<0>();                               \
  in[10] = input_value_leaptr.template load<0>();                              \
  in[11] = input_value_leaptr.template load<0>();                              \
  in[12] = input_value_leaptr.template load<0>();                              \
  in[13] = input_value_leaptr.template load<0>();                              \
  in[14] = input_value_leaptr.template load<0>();                              \
  in[15] = input_value_leaptr.template load<0>();                              \
  int i = 0;

// Notice that last loop load should not read the invalid buff
#define GET_TOP1_STABLE_STAGE2                                                 \
  for (; i < size; i += 16) {                                                  \
    ind[0] = (IType)(i);                                                       \
    ind[1] = ind[0] + qacc_num[1];                                             \
    ind[2] = ind[0] + qacc_num[2];                                             \
    ind[3] = ind[0] + qacc_num[3];                                             \
    ind[4] = ind[0] + qacc_num[4];                                             \
    ind[5] = ind[0] + qacc_num[5];                                             \
    ind[6] = ind[0] + qacc_num[6];                                             \
    ind[7] = ind[0] + qacc_num[7];                                             \
                                                                               \
    out_val_max[0] = kernel_tcle_get_value<DESCENDING, VType>(in[0], in[1]);   \
    out_val_max[1] = kernel_tcle_get_value<DESCENDING, VType>(in[2], in[3]);   \
    out_val_max[2] = kernel_tcle_get_value<DESCENDING, VType>(in[4], in[5]);   \
    out_val_max[3] = kernel_tcle_get_value<DESCENDING, VType>(in[6], in[7]);   \
    out_val_max[4] = kernel_tcle_get_value<DESCENDING, VType>(in[8], in[9]);   \
    out_val_max[5] = kernel_tcle_get_value<DESCENDING, VType>(in[10], in[11]); \
    out_val_max[6] = kernel_tcle_get_value<DESCENDING, VType>(in[12], in[13]); \
    out_val_max[7] = kernel_tcle_get_value<DESCENDING, VType>(in[14], in[15]); \
                                                                               \
    vmask0 = kernel_tcle_cmp_value<DESCENDING, VType, MType>(in[1], in[0]);    \
    out_ind_max[0] = vsel(vmask0, ind[0] + qacc_1, ind[0]);                    \
    vmask1 = kernel_tcle_cmp_value<DESCENDING, VType, MType>(in[3], in[2]);    \
    out_ind_max[1] = vsel(vmask1, ind[1] + qacc_1, ind[1]);                    \
    vmask2 = kernel_tcle_cmp_value<DESCENDING, VType, MType>(in[5], in[4]);    \
    out_ind_max[2] = vsel(vmask2, ind[2] + qacc_1, ind[2]);                    \
    vmask3 = kernel_tcle_cmp_value<DESCENDING, VType, MType>(in[7], in[6]);    \
    out_ind_max[3] = vsel(vmask3, ind[3] + qacc_1, ind[3]);                    \
    vmask4 = kernel_tcle_cmp_value<DESCENDING, VType, MType>(in[9], in[8]);    \
    out_ind_max[4] = vsel(vmask4, ind[4] + qacc_1, ind[4]);                    \
    vmask5 = kernel_tcle_cmp_value<DESCENDING, VType, MType>(in[11], in[10]);  \
    out_ind_max[5] = vsel(vmask5, ind[5] + qacc_1, ind[5]);                    \
    vmask6 = kernel_tcle_cmp_value<DESCENDING, VType, MType>(in[13], in[12]);  \
    out_ind_max[6] = vsel(vmask6, ind[6] + qacc_1, ind[6]);                    \
    vmask7 = kernel_tcle_cmp_value<DESCENDING, VType, MType>(in[15], in[14]);  \
    out_ind_max[7] = vsel(vmask7, ind[7] + qacc_1, ind[7]);                    \
                                                                               \
    vmask0 = kernel_tcle_cmp_value<DESCENDING, VType, MType>(out_val_max[1],   \
                                                             out_val_max[0]);  \
    out_ind_max[0] = vsel(vmask0, out_ind_max[1], out_ind_max[0]);             \
    out_val_max[0] = kernel_tcle_get_value<DESCENDING, VType>(out_val_max[0],  \
                                                              out_val_max[1]); \
    vmask1 = kernel_tcle_cmp_value<DESCENDING, VType, MType>(out_val_max[3],   \
                                                             out_val_max[2]);  \
    out_ind_max[2] = vsel(vmask1, out_ind_max[3], out_ind_max[2]);             \
    out_val_max[2] = kernel_tcle_get_value<DESCENDING, VType>(out_val_max[2],  \
                                                              out_val_max[3]); \
    vmask2 = kernel_tcle_cmp_value<DESCENDING, VType, MType>(out_val_max[5],   \
                                                             out_val_max[4]);  \
    out_ind_max[4] = vsel(vmask2, out_ind_max[5], out_ind_max[4]);             \
    out_val_max[4] = kernel_tcle_get_value<DESCENDING, VType>(out_val_max[4],  \
                                                              out_val_max[5]); \
    vmask3 = kernel_tcle_cmp_value<DESCENDING, VType, MType>(out_val_max[7],   \
                                                             out_val_max[6]);  \
    out_ind_max[6] = vsel(vmask3, out_ind_max[7], out_ind_max[6]);             \
    out_val_max[6] = kernel_tcle_get_value<DESCENDING, VType>(out_val_max[6],  \
                                                              out_val_max[7]); \
                                                                               \
    vmask0 = kernel_tcle_cmp_value<DESCENDING, VType, MType>(out_val_max[2],   \
                                                             out_val_max[0]);  \
    out_ind_max[0] = vsel(vmask0, out_ind_max[2], out_ind_max[0]);             \
    out_val_max[0] = kernel_tcle_get_value<DESCENDING, VType>(out_val_max[0],  \
                                                              out_val_max[2]); \
    vmask1 = kernel_tcle_cmp_value<DESCENDING, VType, MType>(out_val_max[6],   \
                                                             out_val_max[4]);  \
    out_ind_max[4] = vsel(vmask1, out_ind_max[6], out_ind_max[4]);             \
    out_val_max[4] = kernel_tcle_get_value<DESCENDING, VType>(out_val_max[4],  \
                                                              out_val_max[6]); \
                                                                               \
    vmask0 = kernel_tcle_cmp_value<DESCENDING, VType, MType>(out_val_max[4],   \
                                                             out_val_max[0]);  \
    out_ind_max[0] = vsel(vmask0, out_ind_max[4], out_ind_max[0]);             \
    out_val_max[0] = kernel_tcle_get_value<DESCENDING, VType>(out_val_max[0],  \
                                                              out_val_max[4]); \
                                                                               \
    vmask0 = kernel_tcle_cmp_value<DESCENDING, VType, MType>(out_val_max[0],   \
                                                             val_max);         \
    idx_off_local = vsel(vmask0, out_ind_max[0], idx_off_local);               \
    val_max =                                                                  \
        kernel_tcle_get_value<DESCENDING, VType>(out_val_max[0], val_max);     \
                                                                               \
    in[0] = input_value_leaptr.template load<0>();                             \
    in[1] = input_value_leaptr.template load<0>();                             \
    in[2] = input_value_leaptr.template load<0>();                             \
    in[3] = input_value_leaptr.template load<0>();                             \
    in[4] = input_value_leaptr.template load<0>();                             \
    in[5] = input_value_leaptr.template load<0>();                             \
    in[6] = input_value_leaptr.template load<0>();                             \
    in[7] = input_value_leaptr.template load<0>();                             \
    in[8] = input_value_leaptr.template load<0>();                             \
    in[9] = input_value_leaptr.template load<0>();                             \
    in[10] = input_value_leaptr.template load<0>();                            \
    in[11] = input_value_leaptr.template load<0>();                            \
    in[12] = input_value_leaptr.template load<0>();                            \
    in[13] = input_value_leaptr.template load<0>();                            \
    in[14] = input_value_leaptr.template load<0>();                            \
    in[15] = input_value_leaptr.template load<0>();                            \
  }

template <typename T, bool DESCENDING>
__attribute__((noinline)) __TCLE_DEVICE_TYPE__ void
ktop1_stage1(T* value_ptr, T* output_ptr, u_int32_t* index_ptr, int col_index,
             int global_base, int size, int k, T inf_value_tmp) {
  using VType = typename altivector<T, TCLE_MAX_VECTOR_LENGTH>::VT;
  using IType = typename altivector<u_int32_t, TCLE_MAX_VECTOR_LENGTH>::VT;
  using MType = typename altivector_to_mask<VType>::type;
  using Inf_Type = typename dtype_mapping<T>::type;
  Inf_Type inf_val = *(reinterpret_cast<Inf_Type*>(&inf_value_tmp));
  int v_stride = TCLE_MAX_VECTOR_LENGTH * sizeof(T);
  int i_stride = TCLE_MAX_VECTOR_LENGTH * sizeof(u_int32_t);

  tcle::generic_ptr val_in_addr0 =
      reinterpret_cast<tcle::generic_ptr>(value_ptr);
  tcle::generic_ptr val_in_addr1 = reinterpret_cast<tcle::generic_ptr>(
      value_ptr + TCLE_MAX_VECTOR_LENGTH / 2);

  tcle::generic_ptr output_addr0 =
      reinterpret_cast<tcle::generic_ptr>(output_ptr);
  tcle::generic_ptr output_addr1 = reinterpret_cast<tcle::generic_ptr>(
      output_ptr + TCLE_MAX_VECTOR_LENGTH / 2);
  tcle::leaptr<VType, 1> output_value_leaptr =
      tcle::leaptr<VType, 1>(output_addr0, v_stride, output_addr1, v_stride);

  tcle::generic_ptr index_addr0 =
      reinterpret_cast<tcle::generic_ptr>(index_ptr);
  tcle::generic_ptr index_addr1 = reinterpret_cast<tcle::generic_ptr>(
      index_ptr + TCLE_MAX_VECTOR_LENGTH / 2);
  tcle::leaptr<IType, 1> output_index_leaptr =
      tcle::leaptr<IType, 1>(index_addr0, i_stride, index_addr1, i_stride);

  VType in[16], val_max, qa_inf_val;
  IType ind[8], idx_off_local, idx_max_global;
  IType qacc_1, qacc_base, qacc_order, qacc_128, qacc_bpe;
  IType qacc_num[8];
  VType out_val_max[8];
  IType out_ind_max[8];
  MType vmask0, vmask1, vmask2, vmask3;
  MType vmask4, vmask5, vmask6, vmask7;

  for (int i = 0; i < 8; i++) {
    qacc_num[i] = (IType)(2 * i);
  }
  qacc_1 = (IType)(1);
  qacc_base = (IType)(global_base);
  qacc_order = mid<__vector4 unsigned int, 0>(0);
  qacc_128 = (IType)(TCLE_MAX_VECTOR_LENGTH);
  qacc_bpe = (IType)(sizeof(T));

  qa_inf_val = (VType)(inf_val);
  for (int j = 0; j < k; j++) {
    val_max = load<VType>((__TCLE_AS__ char*)value_ptr);

    GET_TOP1_STABLE_STAGE1
    #pragma unroll 4
    GET_TOP1_STABLE_STAGE2
    output_value_leaptr.template store<0>(val_max);

    idx_max_global = idx_off_local + qacc_base;
    output_index_leaptr.template store<0>(idx_max_global);

    if (j < k - 1) {
      idx_off_local = idx_off_local * qacc_128;
      idx_off_local = idx_off_local + qacc_order;
      idx_off_local = idx_off_local * qacc_bpe;

      long addr0 = reinterpret_cast<long>(value_ptr);
      __TCLE_AS__ void* addr2 = reinterpret_cast<__TCLE_AS__ void*>(addr0);
      scatter(qa_inf_val, addr2, idx_off_local);
    }
  }
}

template <typename T, bool DESCENDING>
__attribute__((noinline)) __TCLE_DEVICE_TYPE__ void
ktop1_stage2(T* val_in_ptr, u_int32_t* idx_in_ptr, T* val_out_ptr, u_int32_t* idx_out_ptr,
             int size, int k, T inf_value_tmp) {
  using VType = typename altivector<T, TCLE_MAX_VECTOR_LENGTH>::VT;
  using IType = typename altivector<u_int32_t, TCLE_MAX_VECTOR_LENGTH>::VT;
  using MType = typename altivector_to_mask<VType>::type;
  int idx_bpe = sizeof(u_int32_t);
  int v_stride = TCLE_MAX_VECTOR_LENGTH * sizeof(T);
  int i_stride = TCLE_MAX_VECTOR_LENGTH * idx_bpe;
  using Inf_Type = typename dtype_mapping<T>::type;
  Inf_Type inf_val = *(reinterpret_cast<Inf_Type*>(&inf_value_tmp));

  tcle::generic_ptr val_in_addr0 =
      reinterpret_cast<tcle::generic_ptr>(val_in_ptr);
  tcle::generic_ptr val_in_addr1 = reinterpret_cast<tcle::generic_ptr>(
      val_in_ptr + TCLE_MAX_VECTOR_LENGTH / 2);

  tcle::generic_ptr output_addr0 =
      reinterpret_cast<tcle::generic_ptr>(val_out_ptr);
  tcle::generic_ptr output_addr1 = reinterpret_cast<tcle::generic_ptr>(
      val_out_ptr + TCLE_MAX_VECTOR_LENGTH / 2);
  tcle::leaptr<VType, 1> output_value_leaptr =
      tcle::leaptr<VType, 1>(output_addr0, v_stride, output_addr1, v_stride);

  VType in[16], val_max, qa_inf_val;
  IType ind[16], idx_off_local, val_off_local, qacc_idx_bpe;
  VType out_val_max[8];
  IType out_ind_max[8], qacc_num[8], qacc_1, qacc_order, qacc_128, qacc_bpe;
  MType vmask0, vmask1, vmask2, vmask3, vmask4, vmask5, vmask6, vmask7;
  IType qa_idx_last;

  for (int i = 0; i < 8; i++) {
    qacc_num[i] = (IType)(2 * i);
  }
  qacc_1 = (IType)(1);
  qacc_128 = (IType)(TCLE_MAX_VECTOR_LENGTH);
  qacc_bpe = (IType)(sizeof(T));
  qacc_idx_bpe = (IType)(idx_bpe);
  qacc_order = mid<__vector4 unsigned int, 0>(0);

  qa_inf_val = (VType)(inf_val);
  for (int j = 0; j < k; j++) {
    val_max = load<VType>((__TCLE_AS__ char*)val_in_ptr);

    GET_TOP1_STABLE_STAGE1
    #pragma unroll 2
    GET_TOP1_STABLE_STAGE2
    output_value_leaptr.template store<0>(val_max);

    // scatter value and gather global index
    idx_off_local = idx_off_local * qacc_128;
    idx_off_local = idx_off_local + qacc_order;
    val_off_local = idx_off_local * qacc_bpe;
    idx_off_local = idx_off_local * qacc_idx_bpe;

    long idx_addr0 = reinterpret_cast<long>(idx_in_ptr);
    __TCLE_AS__ void* idx_addr2 =
        reinterpret_cast<__TCLE_AS__ void*>(idx_addr0);
    qa_idx_last = gather<IType>(idx_addr2, idx_off_local);
    store(qa_idx_last,
          (__TCLE_AS__ char*)(idx_out_ptr + j * TCLE_MAX_VECTOR_LENGTH));
    if (j < k - 1) {
      long addr0 = reinterpret_cast<long>(val_in_ptr);
      __TCLE_AS__ void* addr2 = reinterpret_cast<__TCLE_AS__ void*>(addr0);
      scatter(qa_inf_val, addr2, val_off_local);
    }
  }
}

#endif
