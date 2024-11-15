#include "tops/tops_runtime.h"
#include "utils.h"
#include <acore_op.h>
#include <tops.h>
#include <tops/bfloat.h>
#include <tops/half.h>
#include <tops/topscc_types.h>
#define PING_PONG_SIZE 2
#define BYTE_ALIGN 512
#define ALIGN_UP_512(x) (((x + 511) >> 9) << 9)

template <typename lhs_t, typename rhs_t, typename out_t, typename bias_t,
          typename scale_t>
__device__ __forceinline__ void matmul_kernel_lhs_l1(
    lhs_t *lhs, rhs_t *rhs, out_t *out, bias_t *bias, scale_t *scale,
    lhs_t *zeros, int input_dtype, int input_batch, int input_m, int input_k,
    int input_n, int lhs_multicore, int rhs_multicore, int batch_multicore,
    int lhs_transpose, int rhs_transpose, float alpha, float beta,
    float addmm_beta, int sip_m, int sip_k, int sip_n, int broadcasted_weight,
    int group_size, char *l1_buffer) {

  const int thread_id = GetThreadIdxInBlock();
  const int thread_num = GetThreadNum();
  const int global_thread_id = GetThreadIdx();
  const int block_num = GetBlockNum();
  const int block_id = GetBlockIdx();
  const int each_thread_num = GetThreadNumEachBlock();

  auto B = input_batch;
  auto M = input_m;
  auto K = input_k;
  auto N = input_n;
  auto need_trans_lhs = lhs_transpose;
  auto need_trans_rhs = rhs_transpose;

  int enable_act = 0;
  int enable_bias = 0;
  auto subm_size = sip_m;
  auto subn_size = sip_n;
  auto subk_size = sip_k;

  int quant_type = 0;
  int enable_quant = 0;
  if (std::is_same<lhs_t, __fp16>::value ||
      std::is_same<lhs_t, __bf16>::value) {
    if (std::is_same<rhs_t, uint8_t>::value) {
      quant_type = 1; // QuantType::W4A16;
      enable_quant = 1;
    } else if (std::is_same<rhs_t, int8_t>::value) {
      quant_type = 2; // QuantType::W8A16;
      enable_quant = 1;
    }
  }

  // printf("alpha beta is %f,%f\n", alpha, beta);
  int quant_vab = (enable_quant == 1) ? 512 : 0;
  int k_group_flag = (group_size != -1);
  group_size = k_group_flag ? group_size : K;
  int group_use_num = CeilDiv(K, group_size);
  int k_group_num;

  if (quant_type == 1) {
    k_group_num = K / group_size;
  }

  int dequant_K;
  if (quant_type == 1) {
    if ((K % 128) == 0) {
      dequant_K = K / 2;
    } else {
      dequant_K = K / 2 + 32;
    }
  }

  tops::private_cdte dte_shared_lhs_ctx;
  tops::event event_shared_lhs;

  int32_t hbm_lhs_shape[3] = {B, M, K};
  if (need_trans_lhs) {
    hbm_lhs_shape[1] = K;
    hbm_lhs_shape[2] = M;
  }

  int32_t M_loop = CeilDiv(M, subm_size);
  int32_t M_align = M_loop * subm_size;
  int32_t K_loop = CeilDiv(K, subk_size);
  int32_t K_align = K_loop * subk_size;
  int32_t N_align = AlignUp(N, subn_size);
  if (quant_type == 1) {
    group_use_num = K_align / group_size;
  }

  int32_t l2_lhs_shape0[3] = {1, M_align, K_align};
  if (need_trans_lhs) {
    l2_lhs_shape0[1] = K_align;
    l2_lhs_shape0[2] = M_align;
  }

  __shared__ char l2_buffer[SHARE_BUFFER_SIZE];

  // lhs -> L2 -> L1
  char *l2_lhs_ptr = reinterpret_cast<char *>(l2_buffer);

  auto n_unit_num = CeilDiv(N, subn_size);
  auto sip_num_used = std::min(thread_num, n_unit_num);
  auto block_num_used = (sip_num_used > 1) ? block_num : 1;

  auto sip_num_used_0 = sip_num_used / block_num + sip_num_used % block_num;
  auto sip_num_used_1 = sip_num_used / block_num;
  auto min_sip_num_used = std::min(sip_num_used_0, sip_num_used_1);

  int32_t lhs_dim1 = l2_lhs_shape0[1];
  int32_t lhs_size = l2_lhs_shape0[1] * l2_lhs_shape0[2] * sizeof(lhs_t);
  int32_t lhs_use_cdte_num =
      (lhs_size >= 256 * 1024 && (lhs_dim1 >> 2) * 3 < hbm_lhs_shape[1] &&
       min_sip_num_used >= 4)
          ? 4
          : 1;
  int32_t lhs_thread_dim1 = lhs_dim1 / lhs_use_cdte_num;
  l2_lhs_shape0[1] = lhs_thread_dim1;

  if (thread_id < lhs_use_cdte_num && block_id < block_num_used) {
    dte_shared_lhs_ctx.init();
    dte_shared_lhs_ctx.config_slice(
        tops::mdspan(tops::Shared,
                     reinterpret_cast<lhs_t *>(l2_lhs_ptr) +
                         lhs_thread_dim1 * l2_lhs_shape0[2] * thread_id,
                     l2_lhs_shape0),
        tops::mdspan(tops::Global, lhs, hbm_lhs_shape),
        {0, lhs_thread_dim1 * thread_id, 0});
    event_shared_lhs = dte_shared_lhs_ctx.trigger();
  }

  tops::private_dte dte_lhs_ctx;
  tops::event event_lhs;

  tops::private_cdte dte_private_rhs_ctx;
  dte_private_rhs_ctx.init();
  tops::private_dte dte_rhs_ctx;
  tops::event event_private_rhs;
  tops::event event_rhs;

  tops::private_dte dte_bias_ctx;
  tops::event event_bias;

  tops::private_dte dte_scale_ctx;
  tops::event event_scale;

  tops::private_dte dte_zeros_ctx;
  tops::event event_zeros;

  tops::private_dte dte_out_ctx;
  tops::event event_out;

  int32_t hbm_rhs_shape[3] = {B, K, N};
  if (need_trans_rhs) {
    hbm_rhs_shape[1] = N;
    hbm_rhs_shape[2] = K;
  }
  if (quant_type == 1) {
    if (need_trans_rhs) {
      hbm_rhs_shape[2] = dequant_K;
    } else {
      hbm_rhs_shape[1] = dequant_K;
    }
  }

  int32_t hbm_out_shape[3] = {B, M, N};

  int32_t l2_lhs_shape1[4] = {M_loop, subm_size, K_loop, subk_size};
  if (need_trans_lhs) {
    l2_lhs_shape1[0] = K_loop;
    l2_lhs_shape1[1] = subk_size;
    l2_lhs_shape1[2] = M_loop;
    l2_lhs_shape1[3] = subm_size;
  }

  int32_t l1_lhs_shape[4] = {K_loop, M_loop, subm_size, subk_size};

  int32_t l2_rhs_shape[3] = {1, subk_size, subn_size};
  if (need_trans_rhs) {
    l2_rhs_shape[1] = subn_size;
    l2_rhs_shape[2] = subk_size;
  }
  if (quant_type == 1) {
    if (need_trans_rhs) {
      l2_rhs_shape[2] = subk_size / 2;
    } else {
      l2_rhs_shape[1] = subk_size / 2;
    }
  }

  int32_t l1_rhs_shape[3] = {1, subk_size, subn_size};
  if (quant_type == 1) {
    l1_rhs_shape[1] = subk_size / 2;
  }

  int32_t l1_out_shape[3] = {1, M_align, subn_size};

  int32_t l1_lhs_size = M_align * K_align * sizeof(lhs_t);
  int32_t l1_rhs_size =
      subk_size * subn_size * sizeof(rhs_t); // use lhs type for convert
  int32_t l1_out_size = M_align * subn_size * sizeof(out_t); // use fp16
  int32_t l2_lhs_size = l1_lhs_size;
  int32_t l2_rhs_size = subk_size * subn_size * sizeof(rhs_t); // use rhs type
  int32_t l1_bias_size = 0;
  int32_t l1_scale_size = 0;
  int32_t l1_zeros_size = 0;
  int32_t hbm_scale_shape[2] = {group_use_num, N};
  int32_t hbm_zeros_shape[2];
  int32_t l1_scale_shape[2] = {group_use_num, subn_size};
  int32_t l1_zeros_shape[2];
  if (quant_type == 1) {
    hbm_scale_shape[0] = k_group_num;
    hbm_zeros_shape[0] = k_group_num;
    hbm_zeros_shape[1] = N;
    l1_zeros_shape[0] = group_use_num;
    l1_zeros_shape[1] = subn_size;
  }

  if (enable_quant) {
    l1_scale_size = group_use_num * subn_size * sizeof(scale_t);
  }
  if (quant_type == 1) {
    l1_zeros_size = l1_scale_size;
  }

  // prelu activation
  tops::private_dte dte_act_weight_ctx;
  tops::event event_act_weight;
  int32_t l1_act_weight_size = 0;
  int32_t l1_act_weight_len = 0;
  // __local__ __valigned__ char l1_buffer[VDMEM_VALID_SIZE];
  __local__ __attribute__((aligned(512))) int local_workspace[512];

  // lhs -> L2 -> L1
  char *l1_lhs_ptr = reinterpret_cast<char *>(l1_buffer);

  // rhs -> slice_padding to L2 -> transpose to L1
  char *l2_rhs_ptr0 = reinterpret_cast<char *>(l2_buffer + l2_lhs_size +
                                               thread_id * (l2_rhs_size * 2));
  char *l2_rhs_ptr1 = reinterpret_cast<char *>(l2_rhs_ptr0 + l2_rhs_size);
  char *l1_out_ptr0 = reinterpret_cast<char *>(l1_lhs_ptr + l1_lhs_size);
  char *l1_out_ptr1 = reinterpret_cast<char *>(l1_out_ptr0 + l1_out_size);
  char *l1_bias_ptr0 = reinterpret_cast<char *>(l1_out_ptr1 + l1_out_size);
  char *l1_bias_ptr1 = reinterpret_cast<char *>(l1_bias_ptr0 + l1_bias_size);
  char *l1_act_weight_ptr0 =
      l1_bias_ptr1 + ALIGN_UP_512(l1_bias_size); // no l1 mem assigned defaultly
  char *l1_act_weight_ptr1 = l1_act_weight_ptr0;
  char *l1_scale_ptr0 = reinterpret_cast<char *>(
      l1_act_weight_ptr1 + ALIGN_UP_512(l1_act_weight_size));
  char *l1_scale_ptr1 = reinterpret_cast<char *>(l1_scale_ptr0 + l1_scale_size);
  char *l1_rhs_ptr0 = reinterpret_cast<char *>(l1_scale_ptr1 + l1_scale_size);
  char *l1_rhs_ptr1 = reinterpret_cast<char *>(l1_rhs_ptr0 + l1_rhs_size);
  char *private_rhs_requant_buff =
      reinterpret_cast<char *>(l1_rhs_ptr1 + l1_rhs_size);
  char *l1_zeros_ptr0;
  char *l1_zeros_ptr1; // no l1 mem assigned
  if (quant_type == 1) {
    l1_zeros_ptr0 =
        reinterpret_cast<char *>(private_rhs_requant_buff + l1_rhs_size * 2);
    l1_zeros_ptr1 = reinterpret_cast<char *>(l1_zeros_ptr0 + l1_zeros_size);
  }

  if ((global_thread_id < sip_num_used_0) ||
      ((global_thread_id < sip_num_used_1 + each_thread_num) &&
       (global_thread_id >= each_thread_num))) {
    if (need_trans_rhs) {
      dte_private_rhs_ctx.connect(dte_rhs_ctx);
    }

    if (need_trans_lhs) {
      dte_lhs_ctx.config_transpose(
          tops::mdspan(tops::Private, reinterpret_cast<lhs_t *>(l1_lhs_ptr),
                       l1_lhs_shape),
          tops::mdspan(tops::Shared, reinterpret_cast<lhs_t *>(l2_lhs_ptr),
                       l2_lhs_shape1),
          {0, 2, 3, 1});
    } else {
      dte_lhs_ctx.config_transpose(
          tops::mdspan(tops::Private, reinterpret_cast<lhs_t *>(l1_lhs_ptr),
                       l1_lhs_shape),
          tops::mdspan(tops::Shared, reinterpret_cast<lhs_t *>(l2_lhs_ptr),
                       l2_lhs_shape1),
          {2, 0, 1, 3});
    }

    if (enable_quant) {
      dte_scale_ctx.config_slice(
          tops::mdspan(tops::Private,
                       reinterpret_cast<scale_t *>(l1_scale_ptr0),
                       l1_scale_shape),
          tops::mdspan(tops::Global, scale, hbm_scale_shape), {0, 0});
      if (quant_type == 1) {
        dte_zeros_ctx.config_slice(
            tops::mdspan(tops::Private,
                         reinterpret_cast<lhs_t *>(l1_zeros_ptr0),
                         l1_zeros_shape),
            tops::mdspan(tops::Global, zeros, hbm_zeros_shape), {0, 0});
      }
    }

    if (need_trans_rhs) {
      dte_private_rhs_ctx.config_slice(
          tops::mdspan(tops::Shared, reinterpret_cast<rhs_t *>(l2_rhs_ptr0),
                       l2_rhs_shape),
          tops::mdspan(tops::Global, rhs, hbm_rhs_shape), {0, 0, 0});
      dte_rhs_ctx.config_transpose(
          tops::mdspan(tops::Private, reinterpret_cast<rhs_t *>(l1_rhs_ptr0),
                       l1_rhs_shape),
          tops::mdspan(tops::Shared, reinterpret_cast<rhs_t *>(l2_rhs_ptr0),
                       l2_rhs_shape),
          {0, 2, 1});
    } else {
      dte_rhs_ctx.config_slice(
          tops::mdspan(tops::Private, reinterpret_cast<rhs_t *>(l1_rhs_ptr0),
                       l1_rhs_shape),
          tops::mdspan(tops::Global, rhs, hbm_rhs_shape), {0, 0, 0});
    }

    dte_out_ctx.config_deslice(
        tops::mdspan(tops::Global, out, hbm_out_shape),
        tops::mdspan(tops::Private, reinterpret_cast<out_t *>(l1_out_ptr0),
                     l1_out_shape),
        {0, 0, 0});

    int launch_times = 0;
    bool first_output = true;
    bool n_flag = true;
    bool k_flag = true;
    auto n_idx_base = subn_size * (block_num * thread_id + block_id);
    for (auto b_idx = 0; b_idx < B; b_idx++) {
      auto n_idx = n_idx_base;
      if (enable_quant) {
        SET_DST_CUR_CFG(dte_scale_ctx, l1_scale_ptr0, l1_scale_ptr1, n_flag);
        dte_scale_ctx.set_src_offset(0, 0);
        dte_scale_ctx.set_src_offset(1, n_idx);
        event_scale = dte_scale_ctx.trigger();
        if (quant_type == 1) {
          SET_DST_CUR_CFG(dte_zeros_ctx, l1_zeros_ptr0, l1_zeros_ptr1, n_flag);
          dte_zeros_ctx.set_src_offset(0, 0);
          dte_zeros_ctx.set_src_offset(1, n_idx);
          event_zeros = dte_zeros_ctx.trigger();
        }
      }

      if (need_trans_rhs) {
        SET_DST_CUR_CFG(dte_private_rhs_ctx, l2_rhs_ptr0, l2_rhs_ptr1, k_flag);
        SET_SRC_CUR_CFG(dte_rhs_ctx, l2_rhs_ptr0, l2_rhs_ptr1, k_flag);
        SET_DST_CUR_CFG(dte_rhs_ctx, l1_rhs_ptr0, l1_rhs_ptr1, k_flag);
        dte_private_rhs_ctx.set_src_offset(0, b_idx);
        dte_private_rhs_ctx.set_src_offset(1, n_idx);
        dte_private_rhs_ctx.set_src_offset(2, 0);
        event_private_rhs = dte_private_rhs_ctx.trigger();
        event_rhs = dte_rhs_ctx.trigger();
      } else {
        SET_DST_CUR_CFG(dte_rhs_ctx, l1_rhs_ptr0, l1_rhs_ptr1, k_flag);
        dte_rhs_ctx.set_src_offset(0, b_idx);
        dte_rhs_ctx.set_src_offset(1, 0);
        dte_rhs_ctx.set_src_offset(2, n_idx);
        event_rhs = dte_rhs_ctx.trigger();
      }

      if (b_idx > 0) {
        __syncthreads();
        if (thread_id < lhs_use_cdte_num) {
          dte_shared_lhs_ctx.set_src_offset(0, b_idx);
          event_shared_lhs = dte_shared_lhs_ctx.trigger();
        }
      }
      if (thread_id < lhs_use_cdte_num) {
        event_shared_lhs.wait();
      }
      __syncthreads();

      event_lhs = dte_lhs_ctx.trigger();

      for (; n_idx < N; n_idx += subn_size * thread_num) {
        long out_addr = n_flag ? (long)(l1_out_ptr0) : (long)(l1_out_ptr1);
        auto bias_ptr = n_flag ? l1_bias_ptr0 : l1_bias_ptr1;
        auto scale_ptr = n_flag ? l1_scale_ptr0 : l1_scale_ptr1;
        char *zeros_ptr;
        if (quant_type == 1) {
          zeros_ptr = n_flag ? l1_zeros_ptr0 : l1_zeros_ptr1;
        }

        // load scale
        if (enable_quant) {
          event_scale.wait();
          if (n_idx + subn_size * thread_num < N_align) {
            SET_DST_NEXT_CFG(dte_scale_ctx, l1_scale_ptr0, l1_scale_ptr1,
                             n_flag);
            dte_scale_ctx.set_src_offset(0, 0);
            dte_scale_ctx.set_src_offset(1, n_idx + subn_size * thread_num);
            event_scale = dte_scale_ctx.trigger();
          }
          if (quant_type == 1) {
            event_zeros.wait();
            if (n_idx + subn_size * thread_num < N_align) {
              SET_DST_NEXT_CFG(dte_zeros_ctx, l1_zeros_ptr0, l1_zeros_ptr1,
                               n_flag);
              dte_zeros_ctx.set_src_offset(0, 0);
              dte_zeros_ctx.set_src_offset(1, n_idx + subn_size * thread_num);
              event_zeros = dte_zeros_ctx.trigger();
            }
          }
        }

        for (auto k_idx = 0; k_idx < K; k_idx += subk_size) {
          long rhs_addr = k_flag ? (long)(l1_rhs_ptr0) : (long)(l1_rhs_ptr1);
          auto rhs_ptr = k_flag ? l1_rhs_ptr0 : l1_rhs_ptr1;
          auto next_k_idx = k_idx + subk_size;
          auto next_n_idx = n_idx;
          if (next_k_idx >= K) {
            next_k_idx = 0;
            next_n_idx += subn_size * thread_num;
          }

          event_rhs.wait();
          if (next_n_idx < N) {
            if (need_trans_rhs) {
              SET_DST_NEXT_CFG(dte_private_rhs_ctx, l2_rhs_ptr0, l2_rhs_ptr1,
                               k_flag);
              SET_SRC_NEXT_CFG(dte_rhs_ctx, l2_rhs_ptr0, l2_rhs_ptr1, k_flag);
              SET_DST_NEXT_CFG(dte_rhs_ctx, l1_rhs_ptr0, l1_rhs_ptr1, k_flag);
              dte_private_rhs_ctx.set_src_offset(0, b_idx);
              dte_private_rhs_ctx.set_src_offset(1, next_n_idx);
              if (quant_type == 1) {
                dte_private_rhs_ctx.set_src_offset(2, next_k_idx / 2);
              } else {
                dte_private_rhs_ctx.set_src_offset(2, next_k_idx);
              }
              event_private_rhs = dte_private_rhs_ctx.trigger();
              event_rhs = dte_rhs_ctx.trigger();
            } else {
              SET_DST_NEXT_CFG(dte_rhs_ctx, l1_rhs_ptr0, l1_rhs_ptr1, k_flag);
              dte_rhs_ctx.set_src_offset(0, b_idx);
              if (quant_type == 1) {
                dte_rhs_ctx.set_src_offset(1, next_k_idx / 2);
              } else {
                dte_rhs_ctx.set_src_offset(1, next_k_idx);
              }
              dte_rhs_ctx.set_src_offset(2, next_n_idx);
              event_rhs = dte_rhs_ctx.trigger();
            }
          }

          if (quant_type == 2) {
            int k_loop = k_group_flag ? group_size : subk_size;
            for (int k_group_idx = 0; k_group_idx < subk_size;
                 k_group_idx += k_loop) {
              if (k_idx + k_group_idx < K) {
                int k_gnum_idx =
                    k_group_flag ? (k_idx + k_group_idx) / group_size : 0;
                char *requant_buff =
                    reinterpret_cast<char *>(private_rhs_requant_buff) +
                    k_group_idx * subn_size * sizeof(lhs_t);
                char *rhs_buff =
                    reinterpret_cast<char *>(rhs_ptr) + k_group_idx * subn_size;
                char *scale_buff = reinterpret_cast<char *>(scale_ptr) +
                                   k_gnum_idx * subn_size * sizeof(scale_t);
                if (is_same<lhs_t, tops::half>()) {
                  mul_not_inplace<1, __fp16, char, __fp16>(
                      reinterpret_cast<__fp16 *>(requant_buff),
                      reinterpret_cast<char *>(rhs_buff),
                      reinterpret_cast<__fp16 *>(scale_buff), k_loop,
                      subn_size);
                } else if (is_same<lhs_t, tops::bfloat>()) {
                  mul_not_inplace<1, __bf16, char, __bf16>(
                      reinterpret_cast<__bf16 *>(requant_buff),
                      reinterpret_cast<char *>(rhs_buff),
                      reinterpret_cast<__bf16 *>(scale_buff), k_loop,
                      subn_size);
                }
              }
            }
          } else if (quant_type == 1) {
            int k_gnum_idx = k_idx / group_size;
            char *requant_buff =
                reinterpret_cast<char *>(private_rhs_requant_buff);
            char *rhs_buff = reinterpret_cast<char *>(rhs_ptr);
            char *scale_buff = reinterpret_cast<char *>(scale_ptr) +
                               k_gnum_idx * subn_size * sizeof(scale_t);
            char *zeros_buff = reinterpret_cast<char *>(zeros_ptr) +
                               k_gnum_idx * subn_size * sizeof(lhs_t);
            dequant(reinterpret_cast<lhs_t*>(requant_buff),
                    reinterpret_cast<unsigned char*>(rhs_buff),
                    reinterpret_cast<lhs_t*>(scale_buff),
                    reinterpret_cast<lhs_t*>(zeros_buff),
                    subk_size / 2, subn_size, group_size);
          }
          if (n_idx == n_idx_base && k_idx == 0) {
            event_lhs.wait();
          }

          for (auto m_idx = 0; m_idx < M; m_idx += subm_size) {
            long lhs_addr =
                (long)(l1_lhs_ptr +
                       (k_idx * M_align + m_idx * subk_size) * sizeof(lhs_t));
            long mloop_out_addr = out_addr + m_idx * subn_size * sizeof(out_t);
            lhs_t *rhs_pt = reinterpret_cast<lhs_t *>(
                (quant_type == 2 || quant_type == 1) ? private_rhs_requant_buff
                                                     : rhs_ptr);
            if (std::is_same<lhs_t, __bf16>::value) {
              c_func_bfmatmul_general(
                  lhs_addr, (long)rhs_pt, mloop_out_addr, subm_size, subn_size,
                  subk_size);
            } else {
              c_func_hmatmul_general(
                    lhs_addr, (long)rhs_pt, mloop_out_addr, subm_size, subn_size,
                    subk_size);
            }
          }
          k_flag = !k_flag;
        } // K loop
        if (!first_output) {
          event_out.wait();
        }
        SET_SRC_CUR_CFG(dte_out_ctx, l1_out_ptr0, l1_out_ptr1, n_flag);
        dte_out_ctx.set_dst_offset(0, b_idx);
        dte_out_ctx.set_dst_offset(2, n_idx);
        event_out = dte_out_ctx.trigger();

        first_output = false;

        n_flag = !n_flag;
      } // N loop
    }   // batch loop
    event_out.wait();
  } else {
    for (auto b_idx = 0; b_idx < B; b_idx++) {
      if (b_idx > 0) {
        __syncthreads();
      }
      __syncthreads();
    }
  }
} // func
