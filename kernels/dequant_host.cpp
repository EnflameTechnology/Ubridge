#include <acore_op.h>
#include <tops.h>
#include <tops/tops_runtime.h>
#include "utils/utils.h"
#define L1_ALIGN_SIZE (128)

// input [K, N] ((unsigned char)
// scale [2 * K / group_size, N] （bf16/fp16）
// zeros [2 * K / group_size, N] （bf16/fp16）
// out [2*K, N] (bf16, fp16)

// sample (group_size = 128, K = 4096, N = 1024)
//  input [4096, 1024] (actual [2048, 1024])
//  scale [64, 1024]
//  zeros [64, 1024]
//  out [8192, 1024]

template <typename T>
__global__ void dequant_kernel_4bit(
    T* out, unsigned char* rhs, T* scale, T* zeros, int K, int N,
    int weight_transpose, int group_size) {
  __local__ __valigned__ char buffer_sip[VDMEM_VALID_SIZE];
  int thread_num = GetThreadNum();
  int thread_id = GetThreadIdx();
  int sip_cnt_raw = thread_num;
  int thread_idx = thread_id;
  tops_dte_ctx_t dte_rhs[2];
  tops_dte_ctx_t dte_rhs_trans[2];
  tops_dte_ctx_t dte_out;
  tops_dte_ctx_t dte_scale;
  tops_dte_ctx_t dte_zeros;

  dte_rhs[0].init();
  dte_rhs[1].init();
  dte_rhs_trans[0].init();
  dte_rhs_trans[1].init();
  dte_out.init();
  dte_scale.init();
  dte_zeros.init();

  tops::event event_rhs0;
  tops::event event_rhs1;
  tops::event e_rhs_trans_0;
  tops::event e_rhs_trans_1;
  tops::event event_scale0;
  tops::event event_scale1;
  tops::event event_zeros0;
  tops::event event_zeros1;
  tops::event e_out;

  auto subn_size = 128;
  auto subk_size = 64;

  group_size = group_size == -1 ? K : group_size;
  int group_num = K / group_size;
  int k_group_num = CeilDiv(subk_size, group_size);
  if (group_size == -1) {
    if (K < subk_size) {
      k_group_num = 1;
    }
  }

  auto need_trans_rhs = weight_transpose;
  int32_t rhs_k = K;
  int32_t rhs_subk_size = subk_size;
  rhs_subk_size = subk_size / 2;
  if ((K % 128) == 0) {
    rhs_k = K / 2;
  } else {
    rhs_k = K / 2 + 32;
  }

  int32_t hbm_rhs_shape[2] = {rhs_k, N};
  int32_t hbm_out_shape[2] = {K * 2, N};

  if (need_trans_rhs) {
    hbm_rhs_shape[0] = N;
    hbm_rhs_shape[1] = rhs_k;
  }
  int32_t sip_rhs_shape[2] = {rhs_subk_size, subn_size};
  int32_t sip_out_shape[2] = {rhs_subk_size * 4, subn_size};
  int32_t sip_rhs_trans_shape[2] = {subn_size, rhs_subk_size};

  int32_t sip_scale_shape[2] = {k_group_num, subn_size};
  int32_t sip_zeros_shape[2] = {k_group_num, subn_size};

  int32_t sip_rhs_size = sip_rhs_shape[0] * sip_rhs_shape[1] *
                         sizeof(unsigned char);
  int32_t sip_out_size = sip_out_shape[0] * sip_out_shape[1] * sizeof(T);
  sip_rhs_size = sip_rhs_size * 2;
  int32_t sip_rhs_trans_size = need_trans_rhs > 0 ? sip_rhs_size : 0;
  int32_t sip_scale_size = 0;
  int32_t sip_zeros_size = 0;

  sip_scale_size = k_group_num * subn_size * sizeof(T);
  sip_zeros_size = k_group_num * subn_size * sizeof(T);
  tops::mdspan hbm_rhs(tops::Global, rhs, hbm_rhs_shape);
  tops::mdspan hbm_out(tops::Global, out, hbm_out_shape);
  tops::mdspan hbm_scale(tops::Global, scale, group_num, N);
  tops::mdspan hbm_zeros(tops::Global, zeros, group_num, N);
  // workspace is 2KB
  unsigned char* buffer_rhs0_trans =
      reinterpret_cast<unsigned char*>(reinterpret_cast<char*>(buffer_sip));
  unsigned char* buffer_rhs1_trans = reinterpret_cast<unsigned char*>(
      (reinterpret_cast<unsigned char*>(buffer_rhs0_trans)) +
      AlignUp(sip_rhs_trans_size, L1_ALIGN_SIZE));

  T* buffer_l1_out0 = reinterpret_cast<T*>(
      (reinterpret_cast<char*>(buffer_rhs1_trans)) +
      AlignUp(sip_rhs_trans_size, L1_ALIGN_SIZE));

  T* buffer_l1_out1 =
      reinterpret_cast<T*>((reinterpret_cast<char*>(buffer_l1_out0)) +
                           AlignUp(sip_out_size, L1_ALIGN_SIZE));
  T* buffer_scale0 =
      reinterpret_cast<T*>((reinterpret_cast<char*>(buffer_l1_out1)) +
                           AlignUp(sip_out_size, L1_ALIGN_SIZE));
  T* buffer_scale1 =
      reinterpret_cast<T*>((reinterpret_cast<char*>(buffer_scale0)) +
                           AlignUp(sip_scale_size, L1_ALIGN_SIZE));
  T* buffer_zeros0 =
      reinterpret_cast<T*>((reinterpret_cast<char*>(buffer_scale1)) +
                           AlignUp(sip_scale_size, L1_ALIGN_SIZE));
  T* buffer_zeros1 =
      reinterpret_cast<T*>((reinterpret_cast<char*>(buffer_zeros0)) +
                           AlignUp(sip_zeros_size, L1_ALIGN_SIZE));
  unsigned char* buffer_rhs0 = reinterpret_cast<unsigned char*>(
      (reinterpret_cast<char*>(buffer_zeros1)) +
      AlignUp(sip_zeros_size, L1_ALIGN_SIZE));
  unsigned char* buffer_rhs1 = reinterpret_cast<unsigned char*>(
      (reinterpret_cast<char*>(buffer_rhs0)) +
      AlignUp(sip_rhs_size, L1_ALIGN_SIZE));

  tops::mdspan l1_rhs0(tops::Private, buffer_rhs0, sip_rhs_shape);
  tops::mdspan l1_rhs1(tops::Private, buffer_rhs1, sip_rhs_shape);
  tops::mdspan l1_rhs0_trans(tops::Private, buffer_rhs0_trans,
                                  sip_rhs_trans_shape);
  tops::mdspan l1_rhs1_trans(tops::Private, buffer_rhs1_trans,
                                  sip_rhs_trans_shape);
  tops::mdspan l1_out0(tops::Private, buffer_l1_out0, sip_out_shape);
  tops::mdspan l1_out1(tops::Private, buffer_l1_out1, sip_out_shape);

  tops::mdspan l1_scale0(tops::Private, buffer_scale0, k_group_num,
                              subn_size);
  tops::mdspan l1_scale1(tops::Private, buffer_scale1, k_group_num,
                              subn_size);
  tops::mdspan l1_zeros0(tops::Private, buffer_zeros0, k_group_num,
                              subn_size);
  tops::mdspan l1_zeros1(tops::Private, buffer_zeros1, k_group_num,
                              subn_size);
  auto N_SIP_LOOP_CNT_TASKS = N / subn_size + (N % subn_size > 0 ? 1 : 0);
  auto subk_count = K / subk_size + (K % subk_size > 0 ? 1 : 0);
  auto sip_cnt = sip_cnt_raw;
  auto sdma_tasks_num = N_SIP_LOOP_CNT_TASKS;

  auto sip_num_used = sip_cnt;
  auto rhs_loop_step = sip_num_used;
  auto reminder = sdma_tasks_num % sip_cnt;
  auto loop_len_this_sip = (thread_idx < reminder)
                               ? (sdma_tasks_num / sip_cnt + 1)
                               : (sdma_tasks_num / sip_cnt);
  if (loop_len_this_sip == 0) {
    return;
  }

  auto subn_count = loop_len_this_sip;
  int vab_offset = 512;
  if (thread_idx < sip_num_used) {
    if (need_trans_rhs) {
      dte_rhs_trans[0].connect(dte_rhs[0]);
      dte_rhs_trans[1].connect(dte_rhs[1]);
    }
    auto n_hbm_offset = thread_idx * subn_size;
    if (need_trans_rhs) {
      e_rhs_trans_0 =
          tops::slice_async(dte_rhs_trans[0], l1_rhs0_trans, hbm_rhs,
                            {n_hbm_offset, 0});
      event_rhs0 = tops::transpose_async(dte_rhs[0], l1_rhs0,
                                         l1_rhs0_trans, {1, 0});
    } else {
      event_rhs0 = tops::slice_async(dte_rhs[0], l1_rhs0, hbm_rhs,
                                     {0, n_hbm_offset});
    }

    event_scale0 = tops::slice_async(dte_scale, l1_scale0, hbm_scale,
                                     {0, n_hbm_offset});
    event_zeros0 = tops::slice_async(dte_zeros, l1_zeros0, hbm_zeros,
                                     {0, n_hbm_offset});
    for (auto subn_index = 0; subn_index < subn_count; subn_index++) {
      int subn_index_global = subn_index * rhs_loop_step + thread_idx;
      int subn_offset_global = subn_index_global * subn_size;

      for (auto subk_index = 0; subk_index < subk_count; subk_index++) {
        auto global_loop_index = subn_index * subk_count + subk_index;
        auto next_rhs = (global_loop_index % 2) ? &l1_rhs0 : &l1_rhs1;
        auto subk_offset_global = subk_index * subk_size;

        auto next_rhs_trans = (global_loop_index % 2) == 1
                                  ? &l1_rhs0_trans
                                  : &l1_rhs1_trans;
        auto next_scale =
            (global_loop_index % 2) ? &l1_scale0 : &l1_scale1;
        auto next_zeros =
            (global_loop_index % 2) ? &l1_zeros0 : &l1_zeros1;

        int next_subk_index, next_subn_index;

        if (subk_index + 1 < subk_count) {
          next_subk_index = subk_index + 1;
          next_subn_index = subn_index;
        } else {
          next_subk_index = 0;
          if (subn_index + 1 < subn_count) {
            next_subn_index = subn_index + 1;
          } else {
            next_subn_index = 0;
          }
        }
        auto next_subk_offset_global = next_subk_index * subk_size;
        auto next_rhs_subk_offset_global = next_subk_index * rhs_subk_size;
        auto next_subn_offset_global =
            (next_subn_index * rhs_loop_step + thread_idx) * subn_size;

        auto cur_event_rhs =
            (global_loop_index % 2) == 0 ? event_rhs0 : event_rhs1;
        tops::wait(cur_event_rhs);
        auto cur_event_scale =
            (global_loop_index % 2) == 0 ? event_scale0 : event_scale1;
        auto cur_event_zeros =
            (global_loop_index % 2) == 0 ? event_zeros0 : event_zeros1;
        tops::wait(cur_event_scale);
        tops::wait(cur_event_zeros);

        auto next_event_rhs =
            (global_loop_index % 2) == 1 ? event_rhs0 : event_rhs1;
        auto next_event_scale =
            (global_loop_index % 2) == 1 ? event_scale0 : event_scale1;
        auto next_event_zeros =
            (global_loop_index % 2) == 1 ? event_zeros0 : event_zeros1;
        if ((next_subk_offset_global < K) && (next_subn_offset_global < N)) {
          if (need_trans_rhs) {
            e_rhs_trans_0 =
                tops::slice_async(dte_rhs_trans[0], *next_rhs_trans, hbm_rhs,
                                  {0, 0, next_subn_offset_global,
                                   next_rhs_subk_offset_global});
            next_event_rhs = tops::transpose_async(
                dte_rhs[0], *next_rhs, *next_rhs_trans, {1, 0});
          } else {
            next_event_rhs = tops::slice_async(
                dte_rhs[0], *next_rhs, hbm_rhs,
                {next_rhs_subk_offset_global,
                 next_subn_offset_global});
          }

          next_event_scale = tops::slice_async(
              dte_scale, *next_scale, hbm_scale,
              {next_subk_offset_global / group_size, next_subn_offset_global});
          next_event_zeros = tops::slice_async(
              dte_zeros, *next_zeros, hbm_zeros,
              {next_subk_offset_global / group_size, next_subn_offset_global});
        }
        auto rhs_ptr = (global_loop_index % 2) == 0 ? buffer_rhs0 : buffer_rhs1;
        auto scale_ptr =
            (global_loop_index % 2) == 0 ? buffer_scale0 : buffer_scale1;
        auto zeros_ptr =
            (global_loop_index % 2) == 0 ? buffer_zeros0 : buffer_zeros1;

        auto out_ptr =
            ((global_loop_index % 2) == 0 ? buffer_l1_out0
                                                 : buffer_l1_out1);
        auto private_out = ((global_loop_index % 2) == 0)
                                     ? &l1_out0
                                     : &l1_out1;

        dequant(reinterpret_cast<T*>(out_ptr), reinterpret_cast<unsigned char*>(rhs_ptr),
                reinterpret_cast<T*>(scale_ptr),
                reinterpret_cast<T*>(zeros_ptr), rhs_subk_size, subn_size,
                group_size);

        // output
        e_out = tops::deslice_async(
            dte_out, hbm_out, *private_out,
            {subk_offset_global * 2, subn_offset_global});
        tops::wait(e_out);
      }  // K loop
    }  // N loop
  }

  dte_rhs[0].destroy();
  dte_rhs[1].destroy();
  dte_rhs_trans[0].destroy();
  dte_rhs_trans[1].destroy();
  dte_out.destroy();
  dte_scale.destroy();
  dte_zeros.destroy();
}

//Enflame 4bit dequantization (w4a16 -> bf16/f16)
#define DEQUANT_OP(T, RUST_NAME)  \
extern "C" void dequant_##RUST_NAME(T* out, unsigned char* rhs, T* scale, T* zeros, int K, int N, \
    int weight_transpose, int group_size, void* stream_) {  \
  topsStream_t stream = (topsStream_t)(stream_);  \
  dequant_kernel_4bit<T><<<2, 12, 0, stream>>>(out, rhs, scale, zeros, K, N, weight_transpose, group_size); \
} \

DEQUANT_OP(__fp16, f16)
DEQUANT_OP(__bf16, bf16)