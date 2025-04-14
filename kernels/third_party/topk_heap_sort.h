#include <tops/topscc_types.h>
// #include "bitonic_sort.h"  // NOLINT
#include <tops.h>
#include <tops/tops_runtime.h>

#include "../utils/utils.h"
#include "compute.h"
#include "tops/bfloat.h"
#include "tops/half.h"
#include "../utils/vector_ex.h"
#include "heap_sort.h"  // NOLINT
#define TILE_SIZE_CSB TILE_SIZE * 2
#define TILE_SIZE AlignUp((VDMEM_VALID_SIZE / 16), 512)
#define QACC_SIZE 128
#define DACC_SIZE 64

template <typename T, bool DESCENDING>
__global__ __cooperative__ void topk_kernel_heap_sort(T* input, T* output,
                                                    u_int32_t* index, void* workspace,
                                                    int dim0, int dim1, int dim2, int axis, int k, int n2k_1) {
  int thread_id = GetThreadIdx();
  int thread_num = GetThreadNum();
  int thread_num_each_block = GetThreadNumEachBlock();

  tops::private_dte dte_ctx_in[2];
  dte_ctx_in[0].init();
  dte_ctx_in[1].init();

  tops::private_dte dte_ctx_memset;
  dte_ctx_memset.init();
  tops::private_dte dte_ctx_reduce_val;
  dte_ctx_reduce_val.init();
  tops::private_dte dte_ctx_reduce_idx;
  dte_ctx_reduce_idx.init();

  tops::private_dte dte_ctx_out_val;
  dte_ctx_out_val.init();

  tops::private_dte dte_ctx_out_idx;
  dte_ctx_out_idx.init();

  tops::private_cdte shared_dte_ctx_in[2];
  shared_dte_ctx_in[0].init();
  shared_dte_ctx_in[1].init();
  tops::private_dte dte_ctx_bcast;
  dte_ctx_bcast.init();

  tops::private_cdte shared_dte_ctx_out_val;
  shared_dte_ctx_out_val.init();

  tops::private_cdte shared_dte_ctx_out_idx;
  shared_dte_ctx_out_idx.init();

  tops::event event_in[2];
  tops::event event_memset[2];
  tops::event l2_event_in[2];

  // conect rhs cdte and sdte
  shared_dte_ctx_in[0].connect(dte_ctx_in[0]);
  shared_dte_ctx_in[1].connect(dte_ctx_in[1]);

  int slice_size_axis = TILE_SIZE / QACC_SIZE;
  slice_size_axis = slice_size_axis > n2k_1 ? slice_size_axis : n2k_1;
  int l1_align_size = 256 * 4;

  __local__ __attribute__((aligned(1024))) char l1_buffer[VDMEM_VALID_SIZE-512];
  int val_bpe = sizeof(T);
  int idx_bpe = sizeof(int);
  int val_out_size = QACC_SIZE * val_bpe * n2k_1;
  int idx_out_size = QACC_SIZE * idx_bpe * n2k_1;
  int val_in_size = slice_size_axis * QACC_SIZE * val_bpe;
  int idx_in_size = slice_size_axis * QACC_SIZE * idx_bpe;

  T *sip_value_buffer[2], *sip_value_output_buffer, *sip_tmp_val_buffer;
  u_int32_t *sip_index_output_buffer, *sip_tmp_off_buffer;
  size_t mem_off = 0;
  size_t mem_size = val_in_size;
  sip_value_buffer[0] = reinterpret_cast<T*>(l1_buffer + mem_off);
  mem_off += mem_size;
  mem_off = AlignUp(mem_off, l1_align_size);
  mem_size = val_in_size;
  sip_value_buffer[1] = reinterpret_cast<T*>(l1_buffer + mem_off);
  mem_off += mem_size;
  mem_off = AlignUp(mem_off, l1_align_size);
  mem_size = val_out_size;
  sip_value_output_buffer = reinterpret_cast<T*>(l1_buffer + mem_off);
  mem_off += mem_size;
  mem_off = AlignUp(mem_off, l1_align_size);
  mem_size = idx_out_size;
  sip_index_output_buffer = reinterpret_cast<u_int32_t*>(l1_buffer + mem_off);

  int val_reduce_size = val_out_size;
  int idx_reduce_size = idx_out_size;

  T *sip_val_reduce_buffer, *sip_val_heapify_buffer;
  u_int32_t *sip_idx_reduce_buffer, *sip_idx_heapify_buffer;
  mem_off = 0;
  mem_size = val_reduce_size;
  sip_val_reduce_buffer = reinterpret_cast<T*>(l1_buffer + mem_off);
  mem_off += mem_size;
  mem_off = AlignUp(mem_off, l1_align_size);
  mem_size = idx_reduce_size;
  sip_idx_reduce_buffer = reinterpret_cast<u_int32_t*>(l1_buffer + mem_off);
  mem_off += mem_size;
  mem_off = AlignUp(mem_off, l1_align_size);
  mem_size = val_out_size;
  sip_val_heapify_buffer = reinterpret_cast<T*>(l1_buffer + mem_off);
  mem_off += mem_size;
  mem_off = AlignUp(mem_off, l1_align_size);
  mem_size = idx_out_size;
  sip_idx_heapify_buffer = reinterpret_cast<u_int32_t*>(l1_buffer + mem_off);

  int out_dim0 = dim0;
  int out_dim1 = axis == 2 ? dim1 : k;
  int out_dim2 = axis == 2 ? k : dim2;
  int sip_shape[] = {1, slice_size_axis, QACC_SIZE};
  int hbm_in_shape[] = {dim0, dim1, dim2};
  int hbm_out_shape[] = {out_dim0, out_dim1, out_dim2};

  tops::mdspan hbm_input(tops::Global, reinterpret_cast<T*>(input),
                         hbm_in_shape);
  tops::mdspan hbm_output(tops::Global, reinterpret_cast<T*>(output),
                          hbm_out_shape);
  tops::mdspan hbm_index(tops::Global, reinterpret_cast<u_int32_t*>(index),
                         hbm_out_shape);

  int dim_other_size = axis == 2 ? dim1 : dim2;
  int dim_axis_size = axis == 2 ? dim2 : dim1;

  int nBlocks = CeilDiv(dim_other_size, QACC_SIZE);
  int dim_other_pad = AlignUp(dim_other_size, QACC_SIZE);

  // for columns
  int col_global_blocks = CeilDiv(dim_axis_size, slice_size_axis);
  int off_base, off_size;
  int block_vec[24];
  int thread_num_used =
      col_global_blocks > thread_num ? thread_num : col_global_blocks;
  for (int i = 0; i < thread_num_used; i++) {
    block_vec[i] = col_global_blocks / thread_num_used;
  }
  for (int i = thread_num_used; i < thread_num; i++) {
    block_vec[i] = 0;
  }
  for (int i = 0; i < col_global_blocks % thread_num_used; i++) {
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
  int thread_num_used_row = nBlocks > thread_num_row ? thread_num_row : nBlocks;
  int block_start = nBlocks * tid_row / thread_num_used_row;
  int block_end = nBlocks * (tid_row + 1) / thread_num_used_row;
  int off_base_row = block_start * QACC_SIZE;
  int off_size_row = (block_end - block_start) * QACC_SIZE;
  if (block_end * QACC_SIZE > dim_other_size) {
    off_size_row = dim_other_size - off_base_row;
    if (off_base_row > dim_other_size) {
      off_base_row = 0;
      off_size_row = 0;
    }
  }
  int local_blocks_row = CeilDiv(off_size_row, QACC_SIZE);

  char* l3_workspace = reinterpret_cast<char*>(workspace);
  mem_off = 0;
  mem_size =
      static_cast<size_t>(dim0) * val_bpe * thread_num * n2k_1 * dim_other_pad;
  T* workspace_value = reinterpret_cast<T*>(l3_workspace);
  mem_off += mem_size;
  mem_size =
      static_cast<size_t>(dim0) * idx_bpe * thread_num * n2k_1 * dim_other_pad;
  u_int32_t* workspace_index = reinterpret_cast<u_int32_t*>(l3_workspace + mem_off);

  int ws_shape[3] = {dim0, thread_num * n2k_1, dim_other_pad};
  tops::mdspan hbm_workspace_value(tops::Global, workspace_value, ws_shape);
  tops::mdspan hbm_workspace_index(tops::Global, workspace_index, ws_shape);

  extern __shared__ __valigned__ char buf_l2_csb[];

  mem_off = 0;
  mem_size = static_cast<size_t>(TILE_SIZE) * thread_num_each_block * sizeof(T);
  T* buf_l2_shared = reinterpret_cast<T*>(buf_l2_csb);
  mem_off += mem_size;
  mem_size = static_cast<size_t>(TILE_SIZE) * thread_num_each_block * sizeof(T);
  T* buf_l2_shared_val = reinterpret_cast<T*>(buf_l2_csb + mem_off);
  mem_off += mem_size;
  mem_size =
      static_cast<size_t>(TILE_SIZE) * thread_num_each_block * sizeof(u_int32_t);
  u_int32_t* buf_l2_shared_idx = reinterpret_cast<u_int32_t*>(buf_l2_csb + mem_off);

  T* buf_l2 = reinterpret_cast<T*>(
      buf_l2_shared + TILE_SIZE * (thread_id % thread_num_each_block));
  T* buf_l2_val = reinterpret_cast<T*>(
      buf_l2_shared_val + TILE_SIZE * (thread_id % thread_num_each_block));
  u_int32_t* buf_l2_idx = reinterpret_cast<u_int32_t*>(
      buf_l2_shared_idx + TILE_SIZE * (thread_id % thread_num_each_block));

  // for reduce stage
  int thread_num_reduce_used = nBlocks > thread_num ? thread_num : nBlocks;
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

  T inf_value = get_inf_value<T>(true);
  T rem_value = get_pad_value<DESCENDING, T>();
  int layout_tr[] = {0, 2, 1};
  int offsets_csb[3] = {0, 0, 0};
  for (int t = 0; tid_row < thread_num_used_row && t < dim0; t++) {
    for (int row_index = 0; row_index < local_blocks_row; row_index++) {
      for (int col_index = 0; col_index < local_blocks; col_index++) {
        int col_size = col_index == local_blocks - 1
                           ? off_size - col_index * slice_size_axis
                           : slice_size_axis;
        if (col_index == 0) {
          int sip_shape_tmp[3] = {1, col_size, QACC_SIZE};
          tops::mdspan sip_in0(tops::Private, sip_value_buffer[0],
                               sip_shape_tmp);
          if (axis == 2) {
            int csb_shape_tmp[3] = {1, QACC_SIZE, col_size};
            tops::mdspan csb_in0(tops::Shared, buf_l2, csb_shape_tmp);
            int offsets_0[] = {t, off_base_row + row_index * QACC_SIZE,
                               off_base + col_index * slice_size_axis};
            l2_event_in[0] = tops::slice_async(shared_dte_ctx_in[0], csb_in0,
                                               hbm_input, offsets_0);
            event_in[0] = tops::slice_transpose_async(
                dte_ctx_in[0], sip_in0, csb_in0, offsets_csb, layout_tr);
          } else {
            int csb_shape_tmp[3] = {1, col_size, QACC_SIZE};
            tops::mdspan csb_in0(tops::Shared, buf_l2, csb_shape_tmp);
            int offsets_0[] = {t, off_base + col_index * slice_size_axis,
                               off_base_row + row_index * QACC_SIZE};
            l2_event_in[0] = tops::slice_async(shared_dte_ctx_in[0], csb_in0,
                                               hbm_input, offsets_0);
            event_in[0] =
                tops::slice_async(dte_ctx_in[0], sip_in0, csb_in0, offsets_csb);
          }
          if (col_size < n2k_1) {
            int sip_shape_rem[] = {1, n2k_1 - col_size, QACC_SIZE};
            tops::mdspan sip_in1_rem(tops::Private,
                                     sip_value_buffer[0] + col_size * QACC_SIZE,
                                     sip_shape_rem);
            tops::memset(dte_ctx_memset, sip_in1_rem, rem_value);
            col_size = n2k_1;
          }
        }

        tops::wait(event_in[col_index % 2]);
        if (col_index < local_blocks - 1) {
          int next_col_index = col_index + 1;
          int next_row_index = row_index;

          int next_col_size = next_col_index == local_blocks - 1
                                  ? off_size - next_col_index * slice_size_axis
                                  : slice_size_axis;

          int sip_shape_next[] = {1, next_col_size, QACC_SIZE};
          tops::mdspan sip_in1(tops::Private,
                               sip_value_buffer[(col_index + 1) % 2],
                               sip_shape_next);

          if (axis == 2) {
            int csb_shape_next[] = {1, QACC_SIZE, next_col_size};
            tops::mdspan csb_in1(tops::Shared, buf_l2, csb_shape_next);
            int offsets_1[] = {t, off_base_row + next_row_index * QACC_SIZE,
                               off_base + next_col_index * slice_size_axis};
            l2_event_in[(col_index + 1) % 2] =
                tops::slice_async(shared_dte_ctx_in[(col_index + 1) % 2],
                                  csb_in1, hbm_input, offsets_1);
            event_in[(col_index + 1) % 2] = tops::slice_transpose_async(
                dte_ctx_in[(col_index + 1) % 2], sip_in1, csb_in1, offsets_csb,
                layout_tr);
          } else {
            int csb_shape_next[] = {1, next_col_size, QACC_SIZE};
            tops::mdspan csb_in1(tops::Shared, buf_l2, csb_shape_next);
            int offsets_1[] = {t, off_base + next_col_index * slice_size_axis,
                               off_base_row + next_row_index * QACC_SIZE};
            l2_event_in[(col_index + 1) % 2] =
                tops::slice_async(shared_dte_ctx_in[(col_index + 1) % 2],
                                  csb_in1, hbm_input, offsets_1);
            event_in[(col_index + 1) % 2] = tops::slice_async(
                dte_ctx_in[(col_index + 1) % 2], sip_in1, csb_in1, offsets_csb);
          }
        }

        int global_base = off_base + col_index * slice_size_axis;
        // tops::krt_reset_clock();

        heap_sort<T, DESCENDING, sizeof(T)>(
            sip_value_buffer[col_index % 2], sip_value_output_buffer,
            sip_index_output_buffer, col_index, global_base, col_size, n2k_1);

        // tops::krt_close_clock();
        // int cycles = tops::krt_clock();
        int sip_output_shape[3] = {1, n2k_1, QACC_SIZE};
        tops::mdspan sip_output_value(tops::Private, sip_value_output_buffer,
                                      sip_output_shape);
        tops::mdspan sip_output_index(tops::Private, sip_index_output_buffer,
                                      sip_output_shape);

        if (col_index == local_blocks - 1) {
          __dtu_l_movs_barrier_fence(0x3);
          // {dim0, thread_num, nBlocks * QACC_SIZE};
          int offsets_ws[3] = {t, tid_col * n2k_1, off_base_row + row_index *
            QACC_SIZE};
          tops::deslice(dte_ctx_out_val, hbm_workspace_value, sip_output_value,
                        offsets_ws);
          tops::deslice(dte_ctx_out_idx, hbm_workspace_index, sip_output_index,
                        offsets_ws);
        }
      }
    }
  }

  __syncblocks();

  int thread_num_prev = thread_num_used;
  int thread_num_curr = CeilDiv(thread_num_used, 2);
  int mult = 1;
  while (thread_num_curr >= 1 && thread_num_curr != thread_num_prev) {
    int sip_reduce_shape[] = {1, n2k_1, QACC_SIZE};
    tops::mdspan sip_reduce_value(tops::Private, sip_val_heapify_buffer,
                                  sip_reduce_shape);
    tops::mdspan sip_reduce_index(tops::Private, sip_idx_heapify_buffer,
                                  sip_reduce_shape);
    tops::mdspan sip_in_val(tops::Private, sip_val_reduce_buffer,
                            sip_reduce_shape);
    tops::mdspan sip_in_idx(tops::Private, sip_idx_reduce_buffer,
                            sip_reduce_shape);
    for (int t = 0; thread_id < thread_num_curr && t < dim0; t++) {
      for (int row_index = 0; row_index < nBlocks; row_index++) {
        int offsets_0[] = {t, mult * 2 * thread_id * n2k_1,
                           row_index * QACC_SIZE};
        tops::slice(dte_ctx_reduce_val, sip_reduce_value, hbm_workspace_value,
                    offsets_0);
        tops::slice(dte_ctx_reduce_idx, sip_reduce_index, hbm_workspace_index,
                    offsets_0);

        int col_id = 2 * thread_id + 1;
        if (col_id < thread_num_prev) {
          int offsets_1[] = {t, mult * col_id * n2k_1, row_index * QACC_SIZE};
          tops::slice(dte_ctx_reduce_val, sip_in_val, hbm_workspace_value,
                      offsets_1);
          tops::slice(dte_ctx_reduce_idx, sip_in_idx, hbm_workspace_index,
                      offsets_1);
        } else {
          tops::memset(dte_ctx_memset, sip_in_val, rem_value);
        }

        heap_sort_reduce<T, DESCENDING, sizeof(T)>(
            sip_val_reduce_buffer, sip_idx_reduce_buffer,
            sip_val_heapify_buffer, sip_idx_heapify_buffer, n2k_1, n2k_1);

        __dtu_l_movs_barrier_fence(0x3);
        int offsets_out[] = {t, mult * 2 * thread_id * n2k_1,
                             row_index * QACC_SIZE};
        tops::deslice(dte_ctx_out_val, hbm_workspace_value, sip_reduce_value,
                      offsets_out);
        tops::deslice(dte_ctx_out_idx, hbm_workspace_index, sip_reduce_index,
                      offsets_out);
      }
    }
    thread_num_prev = thread_num_curr;
    thread_num_curr = CeilDiv(thread_num_curr, 2);
    mult *= 2;

    __syncblocks();
  }

  mem_off = 0;
  mem_size = val_in_size;
  sip_value_buffer[0] = reinterpret_cast<T*>(l1_buffer + mem_off);
  // slice_size_axis (512) > n2k_1 (max 511)
  // share val_in space
  sip_val_heapify_buffer = reinterpret_cast<T*>(l1_buffer + mem_off);
  mem_off += mem_size;
  mem_off = AlignUp(mem_off, l1_align_size);
  mem_size = idx_in_size;
  sip_value_buffer[1] = reinterpret_cast<T*>(l1_buffer + mem_off);
  sip_idx_heapify_buffer = reinterpret_cast<u_int32_t*>(l1_buffer + mem_off);
  mem_off += mem_size;
  mem_off = AlignUp(mem_off, l1_align_size);
  mem_size = val_out_size;
  sip_val_reduce_buffer = reinterpret_cast<T*>(l1_buffer + mem_off);
  mem_off += mem_size;
  mem_off = AlignUp(mem_off, l1_align_size);
  mem_size = idx_out_size;
  sip_idx_reduce_buffer = reinterpret_cast<u_int32_t*>(l1_buffer + mem_off);
  mem_off += mem_size;
  mem_off = AlignUp(mem_off, l1_align_size);
  mem_size = QACC_SIZE * idx_bpe;
  u_int32_t* sip_num_kth_val_buffer = reinterpret_cast<u_int32_t*>(l1_buffer + mem_off);
  mem_off += mem_size;
  mem_off = AlignUp(mem_off, l1_align_size);

  int sip_reduce_output_shape[] = {1, n2k_1, QACC_SIZE};
  tops::mdspan sip_reduce_output_value(tops::Private, sip_val_heapify_buffer,
                                       sip_reduce_output_shape);
  tops::mdspan sip_reduce_output_index(tops::Private, sip_idx_heapify_buffer,
                                       sip_reduce_output_shape);
  for (int t = 0; thread_id < thread_num_reduce_used && t < dim0; t++) {
    for (int row_index = 0; row_index < reduce_nBlocks; row_index++) {
      int row_size = row_index == reduce_nBlocks - 1
                         ? reduce_off_size - row_index * QACC_SIZE
                         : QACC_SIZE;
      int offsets_0[] = {t, 0, reduce_off_base + row_index * QACC_SIZE};
      tops::slice(dte_ctx_reduce_val, sip_reduce_output_value,
                  hbm_workspace_value, offsets_0);
      tops::slice(dte_ctx_reduce_idx, sip_reduce_output_index,
                  hbm_workspace_index, offsets_0);
      heap_sort_heapify<T, DESCENDING, sizeof(T)>(
          sip_val_heapify_buffer, sip_idx_heapify_buffer, sip_val_reduce_buffer,
          sip_idx_reduce_buffer, n2k_1, inf_value);
      int sip_output_shape[3] = {1, n2k_1, QACC_SIZE};

      tops::mdspan sip_output_value(tops::Private, sip_val_reduce_buffer,
                                    sip_output_shape);
      tops::mdspan sip_output_index(tops::Private, sip_idx_reduce_buffer,
                                    sip_output_shape);

#define STABLE_SORT
#ifdef STABLE_SORT
      calc_num_kth_val(sip_val_reduce_buffer, sip_idx_reduce_buffer,
        sip_num_kth_val_buffer, k, n2k_1, row_size);

      // require a loop, modify code from stage 1
      local_blocks = CeilDiv(dim_axis_size, slice_size_axis);
      for (int col_index = 0; col_index < local_blocks; col_index++) {
        int col_size = col_index == local_blocks - 1
                           ? dim_axis_size - col_index * slice_size_axis
                           : slice_size_axis;
        if (col_index == 0) {
          int sip_shape_tmp[3] = {1, col_size, QACC_SIZE};
          tops::mdspan sip_in0(tops::Private, sip_value_buffer[0],
                               sip_shape_tmp);
          if (axis == 2) {
            int csb_shape_tmp[3] = {1, QACC_SIZE, col_size};
            tops::mdspan csb_in0(tops::Shared, buf_l2, csb_shape_tmp);
            int offsets_0[] = {t, reduce_off_base + row_index * QACC_SIZE,
                               col_index * slice_size_axis};
            l2_event_in[0] = tops::slice_async(shared_dte_ctx_in[0], csb_in0,
                                               hbm_input, offsets_0);
            event_in[0] = tops::slice_transpose_async(
                dte_ctx_in[0], sip_in0, csb_in0, offsets_csb, layout_tr);
          } else {
            int csb_shape_tmp[3] = {1, col_size, QACC_SIZE};
            tops::mdspan csb_in0(tops::Shared, buf_l2, csb_shape_tmp);
            int offsets_0[] = {t, col_index * slice_size_axis,
                               reduce_off_base + row_index * QACC_SIZE};
            l2_event_in[0] = tops::slice_async(shared_dte_ctx_in[0], csb_in0,
                                               hbm_input, offsets_0);
            event_in[0] =
                tops::slice_async(dte_ctx_in[0], sip_in0, csb_in0, offsets_csb);
          }
        }

        tops::wait(event_in[col_index % 2]);
        if (col_index < local_blocks - 1) {
          int next_col_index = col_index + 1;
          int next_row_index = row_index;

          int next_col_size =
              next_col_index == local_blocks - 1
                  ? dim_axis_size - next_col_index * slice_size_axis
                  : slice_size_axis;

          int sip_shape_next[] = {1, next_col_size, QACC_SIZE};
          tops::mdspan sip_in1(tops::Private,
                               sip_value_buffer[(col_index + 1) % 2],
                               sip_shape_next);

          if (axis == 2) {
            int csb_shape_next[] = {1, QACC_SIZE, next_col_size};
            tops::mdspan csb_in1(tops::Shared, buf_l2, csb_shape_next);
            int offsets_1[] = {t, reduce_off_base + next_row_index * QACC_SIZE,
                               next_col_index * slice_size_axis};
            l2_event_in[(col_index + 1) % 2] =
                tops::slice_async(shared_dte_ctx_in[(col_index + 1) % 2],
                                  csb_in1, hbm_input, offsets_1);
            event_in[(col_index + 1) % 2] = tops::slice_transpose_async(
                dte_ctx_in[(col_index + 1) % 2], sip_in1, csb_in1, offsets_csb,
                layout_tr);
          } else {
            int csb_shape_next[] = {1, next_col_size, QACC_SIZE};
            tops::mdspan csb_in1(tops::Shared, buf_l2, csb_shape_next);
            int offsets_1[] = {t, next_col_index * slice_size_axis,
                               reduce_off_base + next_row_index * QACC_SIZE};
            l2_event_in[(col_index + 1) % 2] =
                tops::slice_async(shared_dte_ctx_in[(col_index + 1) % 2],
                                  csb_in1, hbm_input, offsets_1);
            event_in[(col_index + 1) % 2] = tops::slice_async(
                dte_ctx_in[(col_index + 1) % 2], sip_in1, csb_in1, offsets_csb);
          }
        }

        int global_base = col_index * slice_size_axis;

        int ret = collect_index_outside(
            sip_val_reduce_buffer, sip_idx_reduce_buffer, k,
            sip_value_buffer[col_index % 2], sip_num_kth_val_buffer, col_size,
            global_base);

        // early quit
        if (ret) {
          if (col_index < local_blocks - 1) {
            tops::wait(event_in[(col_index + 1) % 2]);
          }
          break;
        }
      }

      // pad with first lane to resolve zero padding penalty
      if (row_size < QACC_SIZE) {
        int sip_shape[] = {1, n2k_1, QACC_SIZE};
        tops::mdspan sip_in(tops::Private, sip_val_reduce_buffer, sip_shape);
        int csb_shape[] = {1, n2k_1, QACC_SIZE - row_size};
        tops::mdspan csb_in(tops::Shared, buf_l2_val, csb_shape);
        unsigned int sip_shape_tmp[] = {1, static_cast<unsigned int>(n2k_1), 1};

        int offset_1[] = {0, 0, 0};
        int offset_2[] = {0, 0, row_size};

        tops::slice_broadcast(dte_ctx_bcast, csb_in, sip_in, offset_1,
          sip_shape_tmp);
        tops::deslice(dte_ctx_bcast, sip_in, csb_in, offset_2);
      }

      sort_inside(sip_val_reduce_buffer, sip_idx_reduce_buffer, k, n2k_1);
#endif

      __dtu_l_movs_barrier_fence(0x3);
      if (axis != 2) {
        int offsets_output[3] = {t, 0, reduce_off_base + row_index * QACC_SIZE};
        tops::deslice(dte_ctx_out_val, hbm_output, sip_output_value,
                      offsets_output);
        tops::deslice(dte_ctx_out_idx, hbm_index, sip_output_index,
                      offsets_output);
      } else {
        int csb_shape_trans[3] = {1, QACC_SIZE, n2k_1};
        tops::mdspan csb_out_val(tops::Shared, buf_l2_val, csb_shape_trans);
        tops::mdspan csb_out_idx(tops::Shared, buf_l2_idx, csb_shape_trans);

        tops::transpose(dte_ctx_out_val, csb_out_val, sip_output_value,
                        layout_tr);
        tops::transpose(dte_ctx_out_idx, csb_out_idx, sip_output_index,
                        layout_tr);

        int out_shape[3] = {1, row_size, k};
        tops::mdspan sip_out_val(tops::Private, sip_val_reduce_buffer,
                                 out_shape);
        tops::mdspan sip_out_idx(tops::Private, sip_idx_reduce_buffer,
                                 out_shape);

        // int offsets_output[3] = {t, reduce_off_base + row_index * QACC_SIZE,
        // 0};
        tops::slice(dte_ctx_out_val, sip_out_val, csb_out_val, {0, 0, 0});
        tops::slice(dte_ctx_out_idx, sip_out_idx, csb_out_idx, {0, 0, 0});

        tops::mdspan hbm_out_val(
            tops::Global,
            reinterpret_cast<T*>(output) + t * out_dim1 * out_dim2 +
                (reduce_off_base + row_index * QACC_SIZE) * k,
            out_shape);
        tops::mdspan hbm_out_idx(
            tops::Global,
            reinterpret_cast<u_int32_t*>(index)+ t * out_dim1 * out_dim2 +
                (reduce_off_base + row_index * QACC_SIZE) * k,
            out_shape);

        tops::memcpy(dte_ctx_out_val, hbm_out_val, sip_out_val);
        tops::memcpy(dte_ctx_out_idx, hbm_out_idx, sip_out_idx);
      }
    }
  }
}

