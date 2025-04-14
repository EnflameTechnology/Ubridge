#include <tops/topscc_types.h>
#include "../utils/utils.h"
#include "bitonic_sort.h"  // NOLINT

#define TILE_SIZE_CSB TILE_SIZE * 2
#define TILE_SIZE ALIGN_UP((VDMEM_VALID_SIZE / 16), 512)
#define QACC_SIZE 128
#define DACC_SIZE 64

__host__ __device__ int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    };
    return n;
}

template <typename T, bool DESCENDING>
__global__ __cooperative__  void topk_kernel_bitonic(T *input, T *output,
                                                    u_int32_t *index, void* workspace, 
                                                    int dim0, int dim1, int dim2, int axis, int k, int n2k_1) {
  // __local__ __valigned__ int dims[128]; 

  // tops::private_dte ctx; 
  // ctx.init();
  // tops::mdspan srcInfo(tops::Global, dims_, 3); 
  // tops::mdspan dstInfo(tops::Private, dims, 3); 
  // tops::memcpy(ctx, dstInfo, srcInfo); 

  int n2k;
  if (k == 512) {
    // a margin is helpful to offload collect_index_outside
    n2k = next_power_of_2(k + 1);
  } else {
    n2k = next_power_of_2(k);
  }

  int block_size_bitonic = n2k * QACC_SIZE;

  int thread_id = GetThreadIdx();
  int thread_num = GetThreadNum();
  int thread_num_each_block = GetThreadNumEachBlock();

  tops::private_dte dte_ctx_in;
  dte_ctx_in.init();
  tops::private_dte dte_ctx_reduce_val;
  dte_ctx_reduce_val.init();
  tops::private_dte dte_ctx_reduce_idx;
  dte_ctx_reduce_idx.init();
  tops::private_dte dte_ctx_out_val;
  dte_ctx_out_val.init();
  tops::private_dte dte_ctx_out_idx;
  dte_ctx_out_idx.init();

  tops::private_cdte shared_dte_ctx_val_in;
  shared_dte_ctx_val_in.init();
  tops::private_cdte shared_dte_ctx_out_val;
  shared_dte_ctx_out_val.init();
  tops::private_cdte shared_dte_ctx_out_idx;
  shared_dte_ctx_out_idx.init();

  int slice_size_axis = block_size_bitonic;
  int l1_align_size = 512;

  __local__ __valigned__ char l1_buffer[VDMEM_VALID_SIZE-512];
  int val_bpe = sizeof(T);
  int idx_bpe = sizeof(u_int32_t);
  int val_out_size = n2k * val_bpe;
  int idx_out_size = n2k * idx_bpe;
  int val_in_size = slice_size_axis * val_bpe;
  int idx_in_size = slice_size_axis * idx_bpe;

  T *sip_value_buffer;
  u_int32_t *sip_index_buffer;
  int mem_off = 0;
  int mem_size = val_in_size;
  sip_value_buffer = reinterpret_cast<T *>(l1_buffer + mem_off);
  mem_off += mem_size;
  mem_off = ALIGN_UP(mem_off, l1_align_size);
  mem_size = idx_in_size;
  sip_index_buffer = reinterpret_cast<u_int32_t *>(l1_buffer + mem_off);
  mem_off += mem_size;
  mem_off = ALIGN_UP(mem_off, l1_align_size);
  mem_size = (TILE_SIZE / QACC_SIZE) * QACC_SIZE * val_bpe;
  T *sip_value_buffer2 = reinterpret_cast<T *>(l1_buffer + mem_off);
  mem_off += mem_size;
  mem_off = ALIGN_UP(mem_off, l1_align_size);
  mem_size = QACC_SIZE * idx_bpe;
  u_int32_t* sip_num_kth_val_buffer = reinterpret_cast<u_int32_t *>(l1_buffer + mem_off);
  mem_off += mem_size;
  mem_off = ALIGN_UP(mem_off, l1_align_size);

  // if (mem_size > VDMEM_VALID_SIZE) {
  //   printf("bitonic sort exceed %d\n", mem_size);
  //   // return;
  // }
  extern __shared__ __valigned__ char buf_l2[];

  mem_off = 0;
  mem_size = TILE_SIZE_CSB * thread_num_each_block * sizeof(T);
  T *buf_l2_shared_val = reinterpret_cast<T *>(buf_l2);
  mem_off += mem_size;
  mem_size = TILE_SIZE_CSB * thread_num_each_block * sizeof(u_int32_t);
  u_int32_t *buf_l2_shared_idx = reinterpret_cast<u_int32_t *>(buf_l2 + mem_off);
  mem_off += mem_size;
  mem_size = TILE_SIZE_CSB * thread_num_each_block * sizeof(T);
  T *buf_l2_shared_reduce_val = reinterpret_cast<T *>(buf_l2 + mem_off);
  mem_off += mem_size;
  mem_size = TILE_SIZE_CSB * thread_num_each_block * sizeof(u_int32_t);
  u_int32_t *buf_l2_shared_reduce_idx = reinterpret_cast<u_int32_t *>(buf_l2 + mem_off);

  T *buf_l2_val = reinterpret_cast<T *>(
      buf_l2_shared_val + TILE_SIZE_CSB * (thread_id % thread_num_each_block));
  u_int32_t *buf_l2_idx = reinterpret_cast<u_int32_t *>(
      buf_l2_shared_idx + TILE_SIZE_CSB * (thread_id % thread_num_each_block));
  T *buf_l2_reduce_val = reinterpret_cast<T *>(
      buf_l2_shared_reduce_val +
      TILE_SIZE_CSB * (thread_id % thread_num_each_block));
  u_int32_t *buf_l2_reduce_idx = reinterpret_cast<u_int32_t *>(
      buf_l2_shared_reduce_idx +
      TILE_SIZE_CSB * (thread_id % thread_num_each_block));

  // int dim0 = dims[0];
  // int dim1 = dims[1];
  // int dim2 = dims[2];

  // huge dim2 is hard to slice
  if ((axis == 2) && (dim2 > (1 << 24) - 1)) {
    dim0 = dim1;
    dim1 = dim2;
    dim2 = 1;
    axis = 1;
  }

  int out_dim0 = dim0;
  int out_dim1 = axis == 2 ? dim1 : k;
  int out_dim2 = axis == 2 ? k : dim2;
  int hbm_in_shape[] = {dim0, dim1, dim2};
  int hbm_out_shape[] = {out_dim0, out_dim1, out_dim2};
  tops::mdspan hbm_input(tops::Global, reinterpret_cast<T *>(input),
                         hbm_in_shape);
  tops::mdspan hbm_output(tops::Global, reinterpret_cast<T *>(output),
                          hbm_out_shape);
  tops::mdspan hbm_index(tops::Global, reinterpret_cast<u_int32_t *>(index),
                         hbm_out_shape);

  int dim_other_size = axis == 2 ? dim1 : dim2;
  int dim_axis_size = axis == 2 ? dim2 : dim1;

  int dim_other_pad = ALIGN_UP(dim_other_size, QACC_SIZE);

  // for columns
  int col_blocks = CeilDiv(dim_axis_size, slice_size_axis);
  int off_base, off_size;
  int block_vec[24];
  int thread_num_used =
    col_blocks > thread_num ? thread_num : col_blocks;
  for (int i = 0; i < thread_num_used; i++) {
    block_vec[i] = col_blocks / thread_num_used;
  }
  for (int i = thread_num_used; i < thread_num; i++) {
    block_vec[i] = 0;
  }
  for (int i = 0; i < col_blocks % thread_num_used; i++) {
    block_vec[i] = block_vec[i] + 1;
  }
  int tid_col =  thread_id % thread_num_used;
  int tid_row =  thread_id / thread_num_used;
  int block_min = 0, block_max = 0;
  for (int i = 0; i < tid_col; i++) {
    block_min += block_vec[i];
  }
  block_max = block_min + block_vec[tid_col];
  off_base = block_min * slice_size_axis;
  off_size = (block_max - block_min) * slice_size_axis;
  if (block_max * slice_size_axis > dim_axis_size) {
    off_size = dim_axis_size - off_base;
    if (off_base > dim_axis_size) {
      off_size = 0;
    }
  }


  int local_blocks = CeilDiv(off_size, slice_size_axis);

  // for rows
  int thread_num_row = thread_num / thread_num_used;
  int thread_num_used_row = dim_other_size > thread_num_row ? thread_num_row :
    dim_other_size;
  int dim_start = dim_other_size * tid_row / thread_num_used_row;
  int dim_end = dim_other_size * (tid_row + 1) / thread_num_used_row;
  if (dim_end > dim_other_size) {
    dim_end = dim_other_size;
    if (dim_start > dim_other_size) {
      dim_start = 0;
      dim_end = 0;
    }
  }

  char* l3_workspace = reinterpret_cast<char*>(workspace);
  mem_off = 0;
  mem_size = dim0 * val_bpe * thread_num * n2k * dim_other_pad;
  T* workspace_value = reinterpret_cast<T*>(l3_workspace);
  mem_off += mem_size;
  mem_size = dim0 * idx_bpe * thread_num * n2k * dim_other_pad;
  u_int32_t* workspace_index = reinterpret_cast<u_int32_t*>(l3_workspace + mem_off);

  int ws_shape[3] = {dim0, thread_num * n2k, dim_other_pad};
  tops::mdspan hbm_workspace_value(tops::Global, workspace_value, ws_shape);
  tops::mdspan hbm_workspace_index(tops::Global, workspace_index, ws_shape);

  T pad_value = get_pad_value<DESCENDING, T>();
  int sip_shape[] = {1, n2k, QACC_SIZE};
  int csb_shape[] = {1, QACC_SIZE, n2k};
  int layout_tr[] = {0, 2, 1};
  int offsets_csb[3] = {0, 0, 0};

  tops::mdspan csb_in_tr(tops::Shared, buf_l2_val, csb_shape);
  tops::memset(shared_dte_ctx_val_in, csb_in_tr, pad_value);

  int csb_reduce_shape[] = {1, n2k, QACC_SIZE};
  tops::mdspan val_csb_reduce(tops::Shared, buf_l2_reduce_val,
                              csb_reduce_shape);
  tops::mdspan idx_csb_reduce(tops::Shared, buf_l2_reduce_idx,
                              csb_reduce_shape);
  tops::memset(shared_dte_ctx_val_in, val_csb_reduce, pad_value);

  for (int t = 0; tid_row < thread_num_used_row && t < dim0; t++) {
    for (int row_index = dim_start; row_index < dim_end; row_index++) {
      int col_check_id = 0;
      for (int col_index = 0; col_index < local_blocks; col_index++) {
        int col_size = col_index == local_blocks - 1
                           ? off_size - col_index * slice_size_axis
                           : slice_size_axis;
        int col_size_pad = ALIGN_UP(col_size, block_size_bitonic);
        int col_padding_size = col_size_pad - col_size;
        if (col_padding_size > 0) {
          int val_shape_pad[] = {1, 1, col_padding_size};
          tops::mdspan val_csb_pad(tops::Shared, buf_l2_val + col_size,
                                   val_shape_pad);
          tops::memset(shared_dte_ctx_val_in, val_csb_pad, pad_value);
        }
        tops::mdspan val_sip(tops::Private, sip_value_buffer, sip_shape);
        tops::mdspan idx_sip(tops::Private, sip_index_buffer, sip_shape);
        if (axis == 2) {
          int val_shape_ori[] = {col_size};
          tops::mdspan val_csb_in(tops::Shared, buf_l2_val, val_shape_ori);

          tops::mdspan val_hbm_in(tops::Global,
            reinterpret_cast<T*>(input) + t * dim1 * dim2 +
            row_index * dim2 + off_base + col_index * slice_size_axis,
            val_shape_ori);

          tops::memcpy(shared_dte_ctx_val_in, val_csb_in, val_hbm_in);
        } else {
          int val_shape_ori[] = {col_size, 1};
          tops::mdspan val_csb_in(tops::Shared, buf_l2_val, val_shape_ori);

          int hbm_in_shape_tmp[] = {col_size, dim2};
          tops::mdspan hbm_input_tmp(tops::Global,
            reinterpret_cast<T*>(input) + t * dim1 * dim2 +
            (off_base + col_index * slice_size_axis) * dim2, hbm_in_shape_tmp);

          int offsets_src[] = {0, row_index};
          tops::slice(shared_dte_ctx_val_in, val_csb_in, hbm_input_tmp,
            offsets_src);
        }

        tops::slice_transpose(dte_ctx_in, val_sip, csb_in_tr, offsets_csb,
                              layout_tr);

        int global_base = off_base + col_index * slice_size_axis;
        bitonic_sort<T, DESCENDING>(sip_value_buffer, sip_index_buffer,
                                    global_base, n2k);
        __dtu_l_movs_barrier_fence(0x3);

        int v_2 = QACC_SIZE / 2;
        for (int t = QACC_SIZE; t > 1; t = t >> 1) {
          int sip_in_shape[] = {n2k, v_2, 2};
          int csb_tr_shape[] = {n2k, 2, v_2};
          tops::mdspan val_block_in(tops::Private, sip_value_buffer,
                                    sip_in_shape);
          tops::mdspan idx_block_in(tops::Private, sip_index_buffer,
                                    sip_in_shape);
          tops::mdspan val_csb(tops::Shared, buf_l2_val, csb_tr_shape);
          tops::mdspan idx_csb(tops::Shared, buf_l2_idx, csb_tr_shape);
          tops::transpose(dte_ctx_reduce_val, val_csb, val_block_in,
                          layout_tr);
          tops::transpose(dte_ctx_reduce_idx, idx_csb, idx_block_in,
                          layout_tr);

          tops::memcpy(dte_ctx_reduce_val, val_sip, val_csb);
          tops::memcpy(dte_ctx_reduce_idx, idx_sip, idx_csb);

          bitonic_block_select<T, DESCENDING>(sip_value_buffer,
                                              sip_index_buffer, n2k);
          __dtu_l_movs_barrier_fence(0x3);
        }

        int csb_out_shape[] = {1, n2k, 1};
        tops::mdspan val_csb_out(tops::Shared, buf_l2_val, csb_out_shape);
        tops::mdspan idx_csb_out(tops::Shared, buf_l2_idx, csb_out_shape);
        tops::slice(dte_ctx_reduce_val, val_csb_out, val_sip, {0, 0, 0});
        tops::slice(dte_ctx_reduce_idx, idx_csb_out, idx_sip, {0, 0, 0});

        if (local_blocks > 1) {
          int off = col_check_id;
          tops::deslice(dte_ctx_reduce_val, val_csb_reduce, val_csb_out,
                        {0, 0, off});
          tops::deslice(dte_ctx_reduce_idx, idx_csb_reduce, idx_csb_out,
                        {0, 0, off});
          col_check_id += 1;
          if (col_check_id == QACC_SIZE || col_index == local_blocks - 1) {
            tops::memset(dte_ctx_reduce_val, val_sip, pad_value);
            tops::slice(dte_ctx_reduce_val, val_sip, val_csb_reduce, {0, 0, 0});
            tops::slice(dte_ctx_reduce_idx, idx_sip, idx_csb_reduce, {0, 0, 0});

            for (int t = QACC_SIZE; t > 1; t = t >> 1) {
              int sip_in_shape[] = {n2k, v_2, 2};
              int csb_tr_shape[] = {n2k, 2, v_2};
              tops::mdspan val_block_in(tops::Private, sip_value_buffer,
                                        sip_in_shape);
              tops::mdspan idx_block_in(tops::Private, sip_index_buffer,
                                        sip_in_shape);
              tops::mdspan val_csb(tops::Shared, buf_l2_val, csb_tr_shape);
              tops::mdspan idx_csb(tops::Shared, buf_l2_idx, csb_tr_shape);
              tops::transpose(dte_ctx_reduce_val, val_csb, val_block_in,
                              layout_tr);
              tops::transpose(dte_ctx_reduce_idx, idx_csb, idx_block_in,
                              layout_tr);

              tops::memcpy(dte_ctx_reduce_val, val_sip, val_csb);
              tops::memcpy(dte_ctx_reduce_idx, idx_sip, idx_csb);

              bitonic_block_select<T, DESCENDING>(sip_value_buffer,
                                                  sip_index_buffer, n2k);
              __dtu_l_movs_barrier_fence(0x3);
            }
            int csb_out_shape[] = {1, n2k, 1};
            tops::mdspan val_csb_out(tops::Shared, buf_l2_val, csb_out_shape);
            tops::mdspan idx_csb_out(tops::Shared, buf_l2_idx, csb_out_shape);
            tops::slice(dte_ctx_reduce_val, val_csb_out, val_sip, {0, 0, 0});
            tops::slice(dte_ctx_reduce_idx, idx_csb_out, idx_sip, {0, 0, 0});

            if (col_index != local_blocks - 1) {
              tops::memset(shared_dte_ctx_val_in, val_csb_reduce, pad_value);
              tops::deslice(dte_ctx_reduce_val, val_csb_reduce, val_csb_out,
                            {0, 0, 0});
              tops::deslice(dte_ctx_reduce_idx, idx_csb_reduce, idx_csb_out,
                            {0, 0, 0});
              col_check_id = 1;
            }
          }
        }

        if (col_index == local_blocks - 1) {
          int offsets_ws[3] = {t, tid_col * n2k, row_index};
          tops::deslice(shared_dte_ctx_out_val, hbm_workspace_value,
            val_csb_out, offsets_ws);
          tops::deslice(shared_dte_ctx_out_idx, hbm_workspace_index,
            idx_csb_out, offsets_ws);
        }
      }
    }
  }

  __syncblocks();

  int thread_num_reduce_used =
    dim_other_size > thread_num ? thread_num : dim_other_size;
  dim_start = dim_other_size * thread_id / thread_num_reduce_used;
  dim_end = dim_other_size * (thread_id + 1) / thread_num_reduce_used;
  if (dim_end > dim_other_size) {
    dim_end = dim_other_size;
    if (dim_start > dim_other_size) {
      dim_start = 0;
      dim_end = 0;
    }
  }

  // reduce among threads, num of threads <= QACC_SIZE (128)
  for (int t = 0; thread_id < thread_num_reduce_used && t < dim0; t++) {
    for (int row_index = dim_start; row_index < dim_end; row_index++) {
      tops::mdspan csb_val_tr(tops::Shared, buf_l2_val, csb_shape);
      tops::mdspan csb_idx_tr(tops::Shared, buf_l2_idx, csb_shape);
      tops::memset(shared_dte_ctx_out_val, csb_val_tr, pad_value);
      tops::memset(shared_dte_ctx_out_idx, csb_idx_tr, pad_value);

      int csb_shape_tmp[] = {1, thread_num_used * n2k, 1};
      tops::mdspan val_csb(tops::Shared, buf_l2_val, csb_shape_tmp);
      tops::mdspan idx_csb(tops::Shared, buf_l2_idx, csb_shape_tmp);
      tops::slice(shared_dte_ctx_out_val, val_csb, hbm_workspace_value,
        {t, 0, row_index});
      tops::slice(shared_dte_ctx_out_idx, idx_csb, hbm_workspace_index,
        {t, 0, row_index});

      tops::mdspan val_sip(tops::Private, sip_value_buffer, sip_shape);
      tops::mdspan idx_sip(tops::Private, sip_index_buffer, sip_shape);
      tops::slice_transpose(dte_ctx_reduce_val, val_sip, csb_val_tr,
        offsets_csb, layout_tr);
      tops::slice_transpose(dte_ctx_reduce_idx, idx_sip, csb_idx_tr,
        offsets_csb, layout_tr);

      int v_2 = QACC_SIZE / 2;
      for (int t = QACC_SIZE; t > 1; t = t >> 1) {
        int sip_in_shape[] = {n2k, v_2, 2};
        int csb_tr_shape[] = {n2k, 2, v_2};
        tops::mdspan val_block_in(tops::Private, sip_value_buffer,
                                  sip_in_shape);
        tops::mdspan idx_block_in(tops::Private, sip_index_buffer,
                                  sip_in_shape);
        tops::mdspan val_csb(tops::Shared, buf_l2_val, csb_tr_shape);
        tops::mdspan idx_csb(tops::Shared, buf_l2_idx, csb_tr_shape);
        tops::transpose(dte_ctx_reduce_val, val_csb, val_block_in, layout_tr);
        tops::transpose(dte_ctx_reduce_idx, idx_csb, idx_block_in, layout_tr);

        tops::memcpy(dte_ctx_reduce_val, val_sip, val_csb);
        tops::memcpy(dte_ctx_reduce_idx, idx_sip, idx_csb);

        bitonic_block_select<T, DESCENDING>(sip_value_buffer,
                                            sip_index_buffer, n2k);
        __dtu_l_movs_barrier_fence(0x3);
      }
      int csb_out_shape[] = {1, n2k, 1};
      tops::mdspan val_csb_out(tops::Shared, buf_l2_val, csb_out_shape);
      tops::mdspan idx_csb_out(tops::Shared, buf_l2_idx, csb_out_shape);
      tops::slice(dte_ctx_reduce_val, val_csb_out, val_sip, {0, 0, 0});
      tops::slice(dte_ctx_reduce_idx, idx_csb_out, idx_sip, {0, 0, 0});

#define STABLE_SORT
#ifndef STABLE_SORT
      if (axis != 2) {
        int offsets_output[3] = {t, 0, row_index};
        tops::deslice(shared_dte_ctx_out_val, hbm_output, val_csb_out,
                      offsets_output);
        tops::deslice(shared_dte_ctx_out_idx, hbm_index, idx_csb_out,
                      offsets_output);
      } else {
        int out_shape[] = {k};
        tops::mdspan val_csb_out(tops::Shared, buf_l2_val, out_shape);
        tops::mdspan idx_csb_out(tops::Shared, buf_l2_idx, out_shape);

        tops::mdspan val_hbm_out(tops::Global,
          reinterpret_cast<T *>(output) + t * out_dim1 * out_dim2
          + row_index * out_dim2, out_shape);
        tops::mdspan idx_hbm_out(tops::Global,
          reinterpret_cast<u_int32_t *>(index) + t * out_dim1 * out_dim2
          + row_index * out_dim2, out_shape);
        tops::memcpy(shared_dte_ctx_out_val, val_hbm_out, val_csb_out);
        tops::memcpy(shared_dte_ctx_out_idx, idx_hbm_out, idx_csb_out);
      }
#else
      int offsets_output[3] = {t, 0, row_index};
      tops::deslice(shared_dte_ctx_out_val, hbm_workspace_value,
        val_csb_out, offsets_output);
      tops::deslice(shared_dte_ctx_out_idx, hbm_workspace_index,
        idx_csb_out, offsets_output);
#endif
    }
  }

#ifdef STABLE_SORT
  __syncblocks();

  // instable to stable
  int nBlocks = CeilDiv(dim_other_size, QACC_SIZE);
  thread_num_reduce_used = nBlocks > thread_num ? thread_num : nBlocks;
  int reduce_block_start = nBlocks * thread_id / thread_num_reduce_used;
  int reduce_block_end = nBlocks * (thread_id + 1) / thread_num_reduce_used;
  int reduce_off_base = reduce_block_start * QACC_SIZE;
  int reduce_off_size = (reduce_block_end - reduce_block_start) * QACC_SIZE;
  if (reduce_block_end * QACC_SIZE > dim_other_size) {
    reduce_off_size = dim_other_size - reduce_off_base;
    if (reduce_off_base > dim_other_size) {
      reduce_off_base = 0;
      reduce_off_size = 0;
    }
  }
  int reduce_nBlocks = CeilDiv(reduce_off_size, QACC_SIZE);

  for (int t = 0; thread_id < thread_num_reduce_used && t < dim0; t++) {
    for (int row_index = 0; row_index < reduce_nBlocks; row_index++) {
      int row_size = row_index == reduce_nBlocks - 1
                         ? reduce_off_size - row_index * QACC_SIZE
                         : QACC_SIZE;

      tops::mdspan val_sip(tops::Private, sip_value_buffer, sip_shape);
      tops::mdspan idx_sip(tops::Private, sip_index_buffer, sip_shape);
      int offsets_0[] = {t, 0, reduce_off_base + row_index * QACC_SIZE};
      tops::slice(dte_ctx_reduce_val, val_sip, hbm_workspace_value, offsets_0);
      tops::slice(dte_ctx_reduce_idx, idx_sip, hbm_workspace_index, offsets_0);

      calc_num_kth_val(sip_value_buffer, sip_index_buffer,
        sip_num_kth_val_buffer, k, n2k, row_size);

      int slice_size_axis = TILE_SIZE / QACC_SIZE;
      int local_blocks = CeilDiv(dim_axis_size, slice_size_axis);
      for (int col_index = 0; col_index < local_blocks; col_index++) {
        int col_size = col_index == local_blocks - 1
                           ? dim_axis_size - col_index * slice_size_axis
                           : slice_size_axis;
        int sip_shape_tmp[3] = {1, col_size, QACC_SIZE};
        tops::mdspan sip_in(tops::Private, sip_value_buffer2,
                            sip_shape_tmp);
        if (axis == 2) {
          int csb_shape_tmp[3] = {1, QACC_SIZE, col_size};
          tops::mdspan csb_in(tops::Shared, buf_l2_val, csb_shape_tmp);

          // CAUTION: do not allow dim2 exceed 2^24 - 1
          int hbm_in_shape_tmp[] = {1, row_size, dim2};
          tops::mdspan hbm_input_tmp(tops::Global,
            reinterpret_cast<T*>(input) + t * dim1 * dim2 +
            (reduce_off_base + row_index * QACC_SIZE) * dim2,
            hbm_in_shape_tmp);

          int offsets_0[] = {0, 0, col_index * slice_size_axis};
          tops::slice(shared_dte_ctx_val_in, csb_in, hbm_input_tmp,
            offsets_0);

          tops::slice_transpose(
              dte_ctx_in, sip_in, csb_in, offsets_csb, layout_tr);
        } else {
          int csb_shape_tmp[] = {1, col_size, QACC_SIZE};
          tops::mdspan csb_in(tops::Shared, buf_l2_val, csb_shape_tmp);

          int hbm_in_shape_tmp[] = {1, col_size, dim2};
          tops::mdspan hbm_input_tmp(tops::Global,
            reinterpret_cast<T*>(input) + t * dim1 * dim2 +
            col_index * slice_size_axis * dim2,
            hbm_in_shape_tmp);

          int offsets_0[] = {0, 0, reduce_off_base + row_index * QACC_SIZE};
          tops::slice(shared_dte_ctx_val_in, csb_in, hbm_input_tmp,
            offsets_0);

          tops::slice(dte_ctx_in, sip_in, csb_in, offsets_csb);
        }

        int global_base = col_index * slice_size_axis;

        int ret = collect_index_outside(
            sip_value_buffer, sip_index_buffer, k,
            sip_value_buffer2, sip_num_kth_val_buffer, col_size,
            global_base);

        // early quit
        if (ret) {
          break;
        }
      }

      // pad with first lane to resolve zero padding penalty
      if (row_size < QACC_SIZE) {
        int sip_shape[] = {1, n2k, QACC_SIZE};
        tops::mdspan sip_in(tops::Private, sip_value_buffer, sip_shape);
        int csb_shape[] = {1, n2k, QACC_SIZE - row_size};
        tops::mdspan csb_in(tops::Shared, buf_l2_val, csb_shape);
        unsigned int sip_shape_tmp[] = {1, static_cast<unsigned int>(n2k), 1};

        int offset_1[] = {0, 0, 0};
        int offset_2[] = {0, 0, row_size};

        tops::slice_broadcast(dte_ctx_in, csb_in, sip_in, offset_1,
          sip_shape_tmp);
        tops::deslice(dte_ctx_in, sip_in, csb_in, offset_2);
      }

      sort_inside(sip_value_buffer, sip_index_buffer, k, n2k);

      __dtu_l_movs_barrier_fence(0x3);
      if (axis != 2) {
        int offsets_output[3] = {t, 0, reduce_off_base + row_index * QACC_SIZE};
        tops::deslice(dte_ctx_out_val, hbm_output, val_sip, offsets_output);
        tops::deslice(dte_ctx_out_idx, hbm_index, idx_sip, offsets_output);
      } else {
        int csb_shape_trans[3] = {1, QACC_SIZE, n2k};
        tops::mdspan csb_out_val(tops::Shared, buf_l2_val, csb_shape_trans);
        tops::mdspan csb_out_idx(tops::Shared, buf_l2_idx, csb_shape_trans);

        tops::transpose(dte_ctx_out_val, csb_out_val, val_sip, layout_tr);
        tops::transpose(dte_ctx_out_idx, csb_out_idx, idx_sip, layout_tr);

        int out_shape[3] = {1, row_size, k};
        tops::mdspan sip_out_val(tops::Private, sip_value_buffer, out_shape);
        tops::mdspan sip_out_idx(tops::Private, sip_index_buffer, out_shape);

        tops::slice(dte_ctx_out_val, sip_out_val, csb_out_val, {0, 0, 0});
        tops::slice(dte_ctx_out_idx, sip_out_idx, csb_out_idx, {0, 0, 0});

        tops::mdspan hbm_out_val(tops::Global,
          reinterpret_cast<T*>(output) + t * out_dim1 * out_dim2 +
          (reduce_off_base + row_index * QACC_SIZE) * out_dim2, out_shape);
        tops::mdspan hbm_out_idx(tops::Global,
          reinterpret_cast<u_int32_t*>(index)+ t * out_dim1 * out_dim2 +
          (reduce_off_base + row_index * QACC_SIZE) * out_dim2, out_shape);

        tops::memcpy(dte_ctx_out_val, hbm_out_val, sip_out_val);
        tops::memcpy(dte_ctx_out_idx, hbm_out_idx, sip_out_idx);
      }
    }
  }
#endif

  dte_ctx_in.destroy();
  dte_ctx_reduce_val.destroy();
  dte_ctx_reduce_idx.destroy();
  dte_ctx_out_val.destroy();
  dte_ctx_out_idx.destroy();

  shared_dte_ctx_val_in.destroy();
  shared_dte_ctx_out_val.destroy();
  shared_dte_ctx_out_idx.destroy();
}  // end of topscc kernel function