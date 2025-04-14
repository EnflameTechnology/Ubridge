#include <tops/topscc_types.h>
// #include "bitonic_sort.h"  // NOLINT
#include <tops.h>
#include <tops/tops_runtime.h>

#include "../utils/utils.h"
#include "compute.h"
#include "tops/bfloat.h"
#include "tops/half.h"
#include "../utils/vector_ex.h"
#include "ktop1.h"  // NOLINT
#define TILE_SIZE_CSB TILE_SIZE * 2
#define TILE_SIZE AlignUp((VDMEM_VALID_SIZE / 16), 512)
#define QACC_SIZE 128
#define DACC_SIZE 64
#define ALIGN_512 __attribute__((aligned(512)))

template <typename T, bool DESCENDING>
__global__ __cooperative__ void topk_kernel_ktop1(T* input, T* output,
                                                  u_int32_t* index, void* workspace,
                                                    int dim0, int dim1, int dim2, int axis, int k, int n2k_1) {
  int thread_idx = GetThreadIdx();
  int thread_num = GetThreadNum();
  int thread_num_each_block = GetThreadNumEachBlock();
  int thread_idx_in_block = GetThreadIdxInBlock();

  tops::private_dte dte_ctx_in[2];
  dte_ctx_in[0].init();
  dte_ctx_in[1].init();

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

  tops::private_cdte shared_dte_ctx_out_val;
  shared_dte_ctx_out_val.init();

  tops::private_cdte shared_dte_ctx_out_idx;
  shared_dte_ctx_out_idx.init();


  tops::event event_in[2];
  tops::event l2_event_in[2];

  // conect rhs cdte and sdte
  shared_dte_ctx_in[0].connect(dte_ctx_in[0]);
  shared_dte_ctx_in[1].connect(dte_ctx_in[1]);

  int32_t kValBpe = sizeof(T);
  int32_t kIdxBpe = sizeof(uint32_t);
  int32_t l1_align_size = 512;
  int32_t axis_align_size = 16;
  int32_t slice_size_axis =
      AlignDown(VDMEM_VALID_SIZE / 16 / QACC_SIZE, axis_align_size);

  __local__ ALIGN_512 uint8_t l1_buffer[VDMEM_VALID_SIZE - 3072];
  int val_out_size = QACC_SIZE * kValBpe * k;
  int idx_out_size = QACC_SIZE * kIdxBpe * k;
  int val_in_size = slice_size_axis * QACC_SIZE * kValBpe;
  int idx_in_size = slice_size_axis * QACC_SIZE * kIdxBpe;

  T *sip_value_buffer[2];
  T *sip_value_output_buffer;
  u_int32_t *sip_index_output_buffer;
  T *sip_val_reduce_buffer;
  u_int32_t *sip_idx_reduce_buffer;
  size_t mem_off = 0;
  sip_value_buffer[0] = reinterpret_cast<T*>(l1_buffer + mem_off);
  mem_off += val_in_size;
  mem_off = AlignUp(mem_off, l1_align_size);
  sip_value_buffer[1] = reinterpret_cast<T*>(l1_buffer + mem_off);
  mem_off += val_in_size;
  mem_off = AlignUp(mem_off, l1_align_size);
  sip_value_output_buffer = reinterpret_cast<T*>(l1_buffer + mem_off);
  mem_off += val_out_size;
  mem_off = AlignUp(mem_off, l1_align_size);
  sip_index_output_buffer = reinterpret_cast<u_int32_t*>(l1_buffer + mem_off);
  mem_off += idx_out_size;
  mem_off = AlignUp(mem_off, l1_align_size);

  int l1_rem_size = VDMEM_VALID_SIZE - mem_off;
  int reduce_block_size = l1_rem_size / (QACC_SIZE * (kValBpe + kIdxBpe));
  reduce_block_size =
      reduce_block_size < slice_size_axis ? reduce_block_size : slice_size_axis;
  int val_reduce_size = reduce_block_size * QACC_SIZE * kValBpe;
  int idx_reduce_size = reduce_block_size * QACC_SIZE * kIdxBpe;
  sip_val_reduce_buffer = reinterpret_cast<T*>(l1_buffer + mem_off);
  mem_off += val_reduce_size;
  mem_off = AlignUp(mem_off, l1_align_size);
  sip_idx_reduce_buffer = reinterpret_cast<u_int32_t*>(l1_buffer + mem_off);
  mem_off += idx_reduce_size;
  mem_off = AlignUp(mem_off, l1_align_size);

  int sip_reduce_shape[] = {1, reduce_block_size, QACC_SIZE};
  tops::mdspan sip_reduce_value(tops::Private, sip_val_reduce_buffer,
                                sip_reduce_shape);
  tops::mdspan sip_reduce_index(tops::Private, sip_idx_reduce_buffer,
                                sip_reduce_shape);

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

  // for columns
  int off_size = 0;
  int thread_num_used = 0;
  int thr_col_blocks = 0;
  int col_global_blocks = 0;
  {
    col_global_blocks = CeilDiv(dim_axis_size, slice_size_axis);
    thread_num_used =
        col_global_blocks > thread_num ? thread_num : col_global_blocks;
    int blocks_div = col_global_blocks / thread_num_used;
    int blocks_rem = col_global_blocks % thread_num_used;
    thr_col_blocks = blocks_div;
    thr_col_blocks = thread_idx < blocks_rem
                            ? blocks_div + 1
                            : blocks_div;
  }

  // for rows
  int nBlocks = CeilDiv(dim_other_size, QACC_SIZE);
  int dim_other_pad = nBlocks * QACC_SIZE;
  int local_blocks_row = 0;
  int thread_num_used_row = 0;
  int off_base_row = 0;
  int off_size_row = 0;
  thread_num_used_row = nBlocks > thread_num ? thread_num : nBlocks;
  int blocks_div = nBlocks / thread_num_used_row;
  int blocks_rem = nBlocks % thread_num_used_row;
  int thread_blocks = thread_idx < blocks_rem
                          ? blocks_div + 1
                          : blocks_div;
  int block_start = thread_blocks * thread_idx;
  int block_end = thread_blocks * (thread_idx + 1);
  block_start = std::min(block_start, nBlocks);
  block_end = std::min(block_end, nBlocks);
  off_base_row = block_start * QACC_SIZE;
  off_size_row = (block_end - block_start) * QACC_SIZE;
  if (block_end * QACC_SIZE > dim_other_size) {
    off_size_row = dim_other_size - off_base_row;
    if (off_base_row > dim_other_size) {
      off_base_row = 0;
      off_size_row = 0;
    }
  }
  local_blocks_row = CeilDiv(off_size_row, QACC_SIZE);

  uint8_t* l3_workspace = reinterpret_cast<uint8_t*>(workspace);
  mem_off = 0;
  size_t mem_size =
      static_cast<size_t>(dim0) * kValBpe * thread_num * k * dim_other_pad;
  T* workspace_value = reinterpret_cast<T*>(l3_workspace);
  mem_off += mem_size;
  mem_size =
      static_cast<size_t>(dim0) * kIdxBpe * thread_num * k * dim_other_pad;
  u_int32_t* workspace_index = reinterpret_cast<u_int32_t*>(l3_workspace + mem_off);

  int ws_shape[3] = {dim0, thread_num * k, dim_other_pad};
  tops::mdspan hbm_workspace_value(tops::Global, workspace_value, ws_shape);
  tops::mdspan hbm_workspace_index(tops::Global, workspace_index, ws_shape);

  extern __shared__ __valigned__ uint8_t raw_l2_buff[];

  T* buf_l2_in[2];
  T* buf_l2_val = nullptr;
  u_int32_t* buf_l2_idx = nullptr;
  {
    int32_t mem_off = 0;
    int32_t input_base = slice_size_axis * QACC_SIZE;
    int32_t output_base = k * QACC_SIZE;
    T* buf_l2_shared0 = reinterpret_cast<T*>(raw_l2_buff + mem_off);
    mem_off += input_base * thread_num_each_block * sizeof(T);
    T* buf_l2_shared1 = reinterpret_cast<T*>(raw_l2_buff + mem_off);
    mem_off += input_base * thread_num_each_block * sizeof(T);
    T* buf_l2_shared_val = reinterpret_cast<T*>(raw_l2_buff + mem_off);
    mem_off += output_base * thread_num_each_block * sizeof(T);
    u_int32_t* buf_l2_shared_idx = reinterpret_cast<u_int32_t*>(raw_l2_buff + mem_off);

    buf_l2_in[0] = reinterpret_cast<T*>(
      buf_l2_shared0 + thread_idx_in_block * input_base);
    buf_l2_in[1] = reinterpret_cast<T*>(
      buf_l2_shared1 + thread_idx_in_block * input_base);
    buf_l2_val = reinterpret_cast<T*>(
      buf_l2_shared_val + thread_idx_in_block * output_base);
    buf_l2_idx = reinterpret_cast<u_int32_t*>(
      buf_l2_shared_idx + thread_idx_in_block * output_base);
  }

  // for reduce stage
  int thread_num_reduce_used = nBlocks > thread_num ? thread_num : nBlocks;
  int reduce_block_start = nBlocks * thread_idx / thread_num_reduce_used;
  int reduce_block_end = nBlocks * (thread_idx + 1) / thread_num_reduce_used;
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

  T pad_value = get_pad_value<DESCENDING, T>();
  int layout_tr[] = {0, 2, 1};
  int zero3d[3] = {0, 0, 0};
  for (int t = 0; t < dim0; t++) {
    for (int row_index = 0; row_index < nBlocks; row_index++) {
      int col_check_id = 0; // reduce check
      int thr_repeat_num = 0;
      for (int col_index = thread_idx; col_index < col_global_blocks;
           col_index += thread_num, thr_repeat_num++) {
        int axis_off = col_index * slice_size_axis;
        int col_size = col_index == col_global_blocks - 1
                           ? dim_axis_size - axis_off
                           : slice_size_axis;
        int col_size_pad = col_size < k ? k : col_size;
        col_size_pad = AlignUp(col_size_pad, axis_align_size);
        if (col_index == thread_idx) {
          int sip_shape_tmp[3] = {1, col_size_pad, QACC_SIZE};
          tops::mdspan sip_in0(tops::Private, sip_value_buffer[0],
                               sip_shape_tmp);
          if (axis == 2) {
            int csb_shape_tmp[3] = {1, QACC_SIZE, col_size_pad};
            tops::mdspan csb_in0(tops::Shared, buf_l2_in[0], csb_shape_tmp);
            int offsets_0[] = {t, row_index * QACC_SIZE,
                               col_index * slice_size_axis};
            l2_event_in[0] = tops::slice_async(shared_dte_ctx_in[0], csb_in0,
                                               hbm_input, offsets_0, pad_value);
            event_in[0] = tops::transpose_async(dte_ctx_in[0], sip_in0, csb_in0,
                                                layout_tr);

          } else {
            int csb_shape_tmp[3] = {1, col_size_pad, QACC_SIZE};
            tops::mdspan csb_in0(tops::Shared, buf_l2_in[0], csb_shape_tmp);
            int offsets_0[] = {t, col_index * slice_size_axis,
                               row_index * QACC_SIZE};
            l2_event_in[0] = tops::slice_async(shared_dte_ctx_in[0], csb_in0,
                                               hbm_input, offsets_0, pad_value);
            event_in[0] = tops::slice_async(dte_ctx_in[0], sip_in0, csb_in0,
                                            zero3d, pad_value);
          }
        }

        tops::wait(event_in[thr_repeat_num % 2]);
        if (col_index + thread_num < col_global_blocks) {
          int next_col_index = col_index + thread_num;
          int next_row_index = row_index;
          int next_col_off = next_col_index * slice_size_axis;

          int next_col_size = next_col_index == col_global_blocks - 1
                                  ? dim_axis_size - next_col_off
                                  : slice_size_axis;
          int next_col_size_pad = next_col_size < k ? k : next_col_size;
          next_col_size_pad = AlignUp(next_col_size_pad, axis_align_size);
          int sip_shape_next[] = {1, next_col_size_pad, QACC_SIZE};
          tops::mdspan sip_in1(tops::Private,
                               sip_value_buffer[(thr_repeat_num + 1) % 2],
                               sip_shape_next);

          if (axis == 2) {
            int csb_shape_next[] = {1, QACC_SIZE, next_col_size_pad};
            tops::mdspan csb_in1(tops::Shared,
                                 buf_l2_in[(thr_repeat_num + 1) % 2],
                                 csb_shape_next);
            int offsets_1[] = {t, next_row_index * QACC_SIZE,
                               next_col_index * slice_size_axis};
            l2_event_in[(thr_repeat_num + 1) % 2] =
                tops::slice_async(shared_dte_ctx_in[(thr_repeat_num + 1) % 2],
                                  csb_in1, hbm_input, offsets_1, pad_value);
            event_in[(thr_repeat_num + 1) % 2] =
                tops::transpose_async(dte_ctx_in[(thr_repeat_num + 1) % 2],
                                      sip_in1, csb_in1, layout_tr);
          } else {
            int csb_shape_next[] = {1, next_col_size_pad, QACC_SIZE};
            tops::mdspan csb_in1(tops::Shared,
                                 buf_l2_in[(thr_repeat_num + 1) % 2],
                                 csb_shape_next);
            int offsets_1[] = {t, next_col_index * slice_size_axis,
                               next_row_index * QACC_SIZE};
            l2_event_in[(thr_repeat_num + 1) % 2] =
                tops::slice_async(shared_dte_ctx_in[(thr_repeat_num + 1) % 2],
                                  csb_in1, hbm_input, offsets_1, pad_value);
            event_in[(thr_repeat_num + 1) % 2] =
                tops::slice_async(dte_ctx_in[(thr_repeat_num + 1) % 2], sip_in1,
                                  csb_in1, zero3d, pad_value);
          }
        }

        int global_base = col_index * slice_size_axis;
        // tops::krt_reset_clock();
        ktop1_stage1<T, DESCENDING>(
            sip_value_buffer[thr_repeat_num % 2], sip_value_output_buffer,
            sip_index_output_buffer, col_index, global_base, col_size_pad, k,
            pad_value);
        // tops::krt_close_clock();
        // int cycles = tops::krt_clock();
        int sip_output_shape[3] = {1, k, QACC_SIZE};
        tops::mdspan sip_output_value(tops::Private, sip_value_output_buffer,
                                      sip_output_shape);
        tops::mdspan sip_output_index(tops::Private, sip_index_output_buffer,
                                      sip_output_shape);

        if (thr_col_blocks > 1) {
          int offset_reduce[] = {0, col_check_id * k, 0};
          tops::deslice(dte_ctx_reduce_val, sip_reduce_value, sip_output_value,
                        offset_reduce);
          tops::deslice(dte_ctx_reduce_idx, sip_reduce_index, sip_output_index,
                        offset_reduce);

          col_check_id += 1;
          if ((col_check_id + 1) * k > reduce_block_size ||
              thr_repeat_num == thr_col_blocks - 1) {
            int mul_k_sizes = col_check_id * k;
            int mul_k_sizes_pad = AlignUp(mul_k_sizes, axis_align_size);

            if (mul_k_sizes < mul_k_sizes_pad) {
              int sip_shape_tmp_rem[3] = {1, mul_k_sizes_pad - mul_k_sizes,
                                          QACC_SIZE};
              tops::mdspan sip_reduce_val_rem(
                  tops::Private,
                  sip_val_reduce_buffer + mul_k_sizes * QACC_SIZE,
                  sip_shape_tmp_rem);
              tops::memset(dte_ctx_reduce_val, sip_reduce_val_rem, pad_value);
            }

            ktop1_stage2<T, DESCENDING>(
                sip_val_reduce_buffer, sip_idx_reduce_buffer,
                sip_value_output_buffer, sip_index_output_buffer,
                mul_k_sizes_pad, k, pad_value);

            tops::deslice(dte_ctx_reduce_val, sip_reduce_value,
                          sip_output_value, zero3d);
            tops::deslice(dte_ctx_reduce_idx, sip_reduce_index,
                          sip_output_index, zero3d);
            col_check_id = 1;
          }
        }

        if (thr_repeat_num == thr_col_blocks - 1) {
          // {dim0, thread_num, nBlocks * QACC_SIZE};
          int offsets_ws[3] = {0, thread_idx * k, row_index * QACC_SIZE};
          int ws_shape[3] = {dim0 - t, thread_num * k, dim_other_pad};
          tops::mdspan hbm_workspace_value(
              tops::Global,
              workspace_value + t * (thread_num * k * dim_other_pad), ws_shape);
          tops::mdspan hbm_workspace_index(
              tops::Global,
              workspace_index + t * (thread_num * k * dim_other_pad), ws_shape);
          tops::deslice(dte_ctx_out_val, hbm_workspace_value, sip_output_value,
                        offsets_ws);
          tops::deslice(dte_ctx_out_idx, hbm_workspace_index, sip_output_index,
                        offsets_ws);
        }
      }
    }
  }

  __syncblocks();

  if (thread_idx < thread_num_reduce_used) {
    for (int t = 0; t < dim0; t++) {
      for (int row_index = 0; row_index < reduce_nBlocks; row_index++) {
        int row_size = row_index == reduce_nBlocks - 1
                           ? reduce_off_size - row_index * QACC_SIZE
                           : QACC_SIZE;
        int reduce_size_tmp = thread_num_used * k;
        int reduce_size_tmp_pad = AlignUp(reduce_size_tmp, axis_align_size);
        int sip_shape_tmp[3] = {1, reduce_size_tmp, QACC_SIZE};
        tops::mdspan sip_in_val(tops::Private, sip_val_reduce_buffer,
                                sip_shape_tmp);
        tops::mdspan sip_in_idx(tops::Private, sip_idx_reduce_buffer,
                                sip_shape_tmp);
        if (reduce_size_tmp < reduce_size_tmp_pad) {
          int sip_shape_tmp_rem[3] = {1, reduce_size_tmp_pad - reduce_size_tmp,
                                      QACC_SIZE};
          tops::mdspan sip_in_val_rem(
              tops::Private,
              sip_val_reduce_buffer + reduce_size_tmp * QACC_SIZE,
              sip_shape_tmp_rem);
          tops::memset(dte_ctx_reduce_val, sip_in_val_rem, pad_value);
        }
        int offsets_0[] = {t, 0, reduce_off_base + row_index * QACC_SIZE};
        tops::slice(dte_ctx_reduce_val, sip_in_val, hbm_workspace_value,
                    offsets_0);
        tops::slice(dte_ctx_reduce_idx, sip_in_idx, hbm_workspace_index,
                    offsets_0);

        ktop1_stage2<T, DESCENDING>(
            sip_val_reduce_buffer, sip_idx_reduce_buffer,
            sip_value_output_buffer, sip_index_output_buffer,
            reduce_size_tmp_pad, k, pad_value);

        int sip_output_shape[3] = {1, k, QACC_SIZE};
        tops::mdspan sip_output_value(tops::Private, sip_value_output_buffer,
                                      sip_output_shape);
        tops::mdspan sip_output_index(tops::Private, sip_index_output_buffer,
                                      sip_output_shape);

        if (axis != 2) {
          int offsets_output[3] = {t, 0,
                                   reduce_off_base + row_index * QACC_SIZE};
          tcle::fence<0x3>();
          tops::deslice(dte_ctx_out_val, hbm_output, sip_output_value,
                        offsets_output);
          tops::deslice(dte_ctx_out_idx, hbm_index, sip_output_index,
                        offsets_output);
        } else {
          int csb_shape_trans[3] = {1, QACC_SIZE, k};
          tops::mdspan csb_out_val(tops::Shared, buf_l2_val, csb_shape_trans);
          tops::mdspan csb_out_idx(tops::Shared, buf_l2_idx, csb_shape_trans);

          tops::transpose(dte_ctx_out_val, csb_out_val, sip_output_value,
                          layout_tr);
          tops::transpose(dte_ctx_out_idx, csb_out_idx, sip_output_index,
                          layout_tr);

          int out_shape[3] = {1, row_size, k};
          tops::mdspan csb_out_val_tmp(tops::Shared, buf_l2_val, out_shape);
          tops::mdspan csb_out_idx_tmp(tops::Shared, buf_l2_idx, out_shape);

          tops::mdspan hbm_out_val(
              tops::Global,
              reinterpret_cast<T*>(output) + t * out_dim1 * out_dim2 +
                  (reduce_off_base + row_index * QACC_SIZE) * k,
              out_shape);
          tops::mdspan hbm_out_idx(
              tops::Global,
              reinterpret_cast<u_int32_t*>(index) + t * out_dim1 * out_dim2 +
                  (reduce_off_base + row_index * QACC_SIZE) * k,
              out_shape);
          tcle::fence<0x3>();
          tops::memcpy(shared_dte_ctx_out_val, hbm_out_val, csb_out_val_tmp);
          tops::memcpy(shared_dte_ctx_out_idx, hbm_out_idx, csb_out_idx_tmp);
        }
      }
    }
  }
}
