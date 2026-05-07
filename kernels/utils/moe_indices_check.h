/**
 * Copyright 2020-2021 Enflame. All Rights Reserved.
 * Copied from topsop moe_align_block_size/indices_check.h (unchanged logic).
 */
#ifndef MOE_INDICES_CHECK_H_
#define MOE_INDICES_CHECK_H_

__attribute__((no_merge_block_into_predecessor, noinline,
               no_mem_alias_in_vldst_tar)) __device__ void
indicesOpAssert(int *indices, int vlength, int loop_num, int index_max,
                int index_min, int remain_len) {
  using id_vtype = tcle::altivector<int, TCLE_MAX_VECTOR_LENGTH>::VT;
  auto check_indices_ptr = tcle::simple_leaptr<id_vtype>(indices);
  id_vtype index_max_v = (id_vtype)(index_max);
  id_vtype index_min_v = (id_vtype)(index_min);
  using mask_i32_type = tcle::altivector_to_mask<__vector4 int>::type;
#pragma clang loop unroll_count(8)
  for (int i = 0; i < loop_num; i++) {
    auto index_v = check_indices_ptr.load();
    mask_i32_type index_lower_mask = (index_v < index_min_v);
    mask_i32_type index_upper_mask = (index_v >= index_max_v);
    mask_i32_type index_range_mask_tmp = (index_lower_mask | index_upper_mask);
    for (int err_pos = 0; err_pos < vlength; err_pos++) {
      if (index_range_mask_tmp[err_pos]) {
        op_assert(0,
                  "Error: InvokeFusedMoe index out of range! "
                  "range=[%d, %d), index[%d]=%d",
                  index_min, index_max, err_pos + i * vlength, index_v[err_pos]);
      }
    }
  }
  if (remain_len > 0) {
    auto index_v = check_indices_ptr.load();
    mask_i32_type init_mask = tcle::vset_mb<mask_i32_type>(remain_len);
    mask_i32_type index_lower_mask = (index_v < index_min_v);
    mask_i32_type index_upper_mask = (index_v >= index_max_v);
    mask_i32_type index_range_mask_tmp = (index_lower_mask | index_upper_mask);
    index_range_mask_tmp = index_range_mask_tmp & init_mask;
    for (int err_pos = 0; err_pos < remain_len; err_pos++) {
      if (index_range_mask_tmp[err_pos]) {
        op_assert(0,
                  "Error: InvokeFusedMoe index out of range! "
                  "range=[%d, %d), index[%d]=%d",
                  index_min, index_max, err_pos + loop_num * vlength,
                  index_v[err_pos]);
      }
    }
  }
}

__attribute__((device, noinline, no_mem_alias_in_vldst_tar,
               loop_iterator_less_than_1024, enable_software_pipeliner,
               enable_bc_resolver)) inline void
indices_check(int *indices, int indices_size, int index_max, int index_min) {
  using id_vtype = tcle::altivector<int, TCLE_MAX_VECTOR_LENGTH>::VT;
  int vlength = tcle::altivector_step<id_vtype>();
  int loop_num = indices_size / vlength;
  auto indices_ptr = tcle::simple_leaptr<id_vtype>(indices);
  using mask_i32_type = tcle::altivector_to_mask<__vector4 int>::type;

  id_vtype index_max_v = (id_vtype)(index_max);
  id_vtype index_min_v = (id_vtype)(index_min);
  int check_flag = 0;

#ifndef ENABLE_KERNEL_DEBUG
#pragma clang loop unroll_count(8)
#endif
  for (int i = 0; i < loop_num; i++) {
    auto index_v = indices_ptr.load();
    mask_i32_type index_lower_mask = (index_v < index_min_v);
    mask_i32_type index_upper_mask = (index_v >= index_max_v);
    mask_i32_type index_range_mask_tmp = (index_lower_mask | index_upper_mask);
    int t0_flag = tcle::aggr_mb<0, 0>(index_range_mask_tmp);
    int t1_flag = tcle::aggr_mb<0, 1>(index_range_mask_tmp);
    check_flag = check_flag | t0_flag;
    check_flag = check_flag | t1_flag;
  }
  int remain_len = indices_size - loop_num * vlength;
  if (remain_len > 0) {
    auto index_v = indices_ptr.load();
    mask_i32_type init_mask = tcle::vset_mb<mask_i32_type>(remain_len);
    mask_i32_type index_lower_mask = (index_v < index_min_v);
    mask_i32_type index_upper_mask = (index_v >= index_max_v);
    mask_i32_type index_range_mask_tmp = (index_lower_mask | index_upper_mask);
    index_range_mask_tmp = index_range_mask_tmp & init_mask;
    int t0_flag = tcle::aggr_mb<0, 0>(index_range_mask_tmp);
    int t1_flag = tcle::aggr_mb<0, 1>(index_range_mask_tmp);
    check_flag = check_flag | t0_flag;
    check_flag = check_flag | t1_flag;
  }
  if (0 != check_flag) {
    indicesOpAssert(indices, vlength, loop_num, index_max, index_min, remain_len);
  }
}

#endif  // MOE_INDICES_CHECK_H_
