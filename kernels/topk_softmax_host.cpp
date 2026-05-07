/**
 * Top-k over softmax on expert logits.
 * Scalar path for f32 / f16 / bf16 — stable and toolchain-friendly.
 * All global accesses go through DMA (tops::memcpy) as required by GCU.
 */
#include <acore_op.h>
#include "utils/utils.h"

#include <cmath>
#include <limits>
#include <tops.h>
#include <tops/bfloat.h>
#include <tops/half.h>
#include <tops/tops_runtime.h>
#include <tops/topscc_types.h>

#if defined(__GCU_ARCH__)
#include "tcle.h"
using tcle::FenceType;
#endif

using namespace tops;

template <typename T>
__device__ __forceinline__ float to_float(T x) {
  return static_cast<float>(x);
}

template <typename T>
__device__ void topk_softmax_one_token(const T *input, T *output, int *index,
                                     int32_t token_idx, int32_t experts,
                                     int32_t topk, bool norm_topk_prob) {
  if (experts > 4096 || topk > 4096) {
    return;
  }

  __local__ T local_input[4096];
  __local__ float prob[4096];
  __local__ float wtmp[4096];
  __local__ T local_out[4096];
  __local__ int local_idx[4096];

  const int base = token_idx * experts;

  tops::private_dte ctx;
  ctx.init();
  tops::memcpy(ctx, mdspan(Private, local_input, experts),
               mdspan(Global, const_cast<T *>(input) + base, experts));

  float max_v = -std::numeric_limits<float>::infinity();
  for (int e = 0; e < experts; e++) {
    float v = to_float(local_input[e]);
    if (v > max_v) {
      max_v = v;
    }
  }
  float sum = 0.f;
  for (int e = 0; e < experts; e++) {
    sum += expf(to_float(local_input[e]) - max_v);
  }
  float inv_sum = sum > 0.f ? 1.f / sum : 1.f;
  for (int e = 0; e < experts; e++) {
    prob[e] = expf(to_float(local_input[e]) - max_v) * inv_sum;
  }

  for (int e = 0; e < experts; e++) {
    wtmp[e] = prob[e];
  }

  float pinf = std::numeric_limits<float>::infinity();
  float weight_sum = 0.f;
  for (int k = 0; k < topk; k++) {
    float best = -pinf;
    int best_i = -1;
    for (int e = 0; e < experts; e++) {
      if (wtmp[e] > best) {
        best = wtmp[e];
        best_i = e;
      }
    }
    if (best_i < 0) {
      break;
    }
    local_out[k] = static_cast<T>(prob[best_i]);
    local_idx[k] = best_i;
    weight_sum += prob[best_i];
    wtmp[best_i] = -pinf;
  }
  if (norm_topk_prob && weight_sum > 0.f) {
    float inv = 1.f / weight_sum;
    for (int k = 0; k < topk; k++) {
      float t = to_float(local_out[k]) * inv;
      local_out[k] = static_cast<T>(t);
    }
  }

  tops::memcpy(ctx, mdspan(Global, output + token_idx * topk, topk),
               mdspan(Private, local_out, topk));
  tops::memcpy(ctx, mdspan(Global, index + token_idx * topk, topk),
               mdspan(Private, local_idx, topk));
}

template <typename T>
__global__ void topk_softmax_kernel_impl(T *input, T *output, int *index,
                                         int32_t num_tokens, int32_t experts,
                                         int32_t topk, bool norm_topk_prob) {
  if (experts > 4096 || topk > 4096) {
    return;
  }

  __local__ T local_input[4096];
  __local__ float prob[4096];
  __local__ float wtmp[4096];
  __local__ T local_out[4096];
  __local__ int local_idx[4096];

  tops::private_dte ctx;
  ctx.init();

  const int thread_num = GetThreadNum();
  for (int32_t t = GetThreadIdx(); t < num_tokens; t += thread_num) {
    const int base = t * experts;
    tops::memcpy(ctx, mdspan(Private, local_input, experts),
                 mdspan(Global, const_cast<T *>(input) + base, experts));

    float max_v = -std::numeric_limits<float>::infinity();
    for (int e = 0; e < experts; e++) {
      float v = to_float(local_input[e]);
      if (v > max_v) max_v = v;
    }
    float sum = 0.f;
    for (int e = 0; e < experts; e++) {
      sum += expf(to_float(local_input[e]) - max_v);
    }
    float inv_sum = sum > 0.f ? 1.f / sum : 1.f;
    for (int e = 0; e < experts; e++) {
      prob[e] = expf(to_float(local_input[e]) - max_v) * inv_sum;
    }
    for (int e = 0; e < experts; e++) {
      wtmp[e] = prob[e];
    }

    float pinf = std::numeric_limits<float>::infinity();
    float weight_sum = 0.f;
    for (int k = 0; k < topk; k++) {
      float best = -pinf;
      int best_i = -1;
      for (int e = 0; e < experts; e++) {
        if (wtmp[e] > best) {
          best = wtmp[e];
          best_i = e;
        }
      }
      if (best_i < 0) break;
      local_out[k] = static_cast<T>(prob[best_i]);
      local_idx[k] = best_i;
      weight_sum += prob[best_i];
      wtmp[best_i] = -pinf;
    }
    if (norm_topk_prob && weight_sum > 0.f) {
      float inv = 1.f / weight_sum;
      for (int k = 0; k < topk; k++) {
        float tv = to_float(local_out[k]) * inv;
        local_out[k] = static_cast<T>(tv);
      }
    }

    tops::memcpy(ctx, mdspan(Global, output + t * topk, topk),
                 mdspan(Private, local_out, topk));
    tops::memcpy(ctx, mdspan(Global, index + t * topk, topk),
                 mdspan(Private, local_idx, topk));
  }
#if defined(__GCU_ARCH__)
  tcle::fence<FenceType::L2_MEM>();
#endif
}

extern "C" void topk_softmax_f32(float *input, float *output, int *index,
                                 int32_t num_tokens, int32_t num_experts,
                                 int32_t topk, int32_t norm_topk_prob,
                                 void *stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  topk_softmax_kernel_impl<float><<<2, 12, 0, stream>>>(
      input, output, index, num_tokens, num_experts, topk,
      norm_topk_prob != 0);
}

extern "C" void topk_softmax_f16(__fp16 *input, __fp16 *output, int *index,
                                 int32_t num_tokens, int32_t num_experts,
                                 int32_t topk, int32_t norm_topk_prob,
                                 void *stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  topk_softmax_kernel_impl<tops::half><<<2, 12, 0, stream>>>(
      reinterpret_cast<tops::half *>(input),
      reinterpret_cast<tops::half *>(output), index, num_tokens, num_experts,
      topk, norm_topk_prob != 0);
}

extern "C" void topk_softmax_bf16(__bf16 *input, __bf16 *output, int *index,
                                  int32_t num_tokens, int32_t num_experts,
                                  int32_t topk, int32_t norm_topk_prob,
                                  void *stream_) {
  topsStream_t stream = reinterpret_cast<topsStream_t>(stream_);
  topk_softmax_kernel_impl<tops::bfloat><<<2, 12, 0, stream>>>(
      reinterpret_cast<tops::bfloat *>(input),
      reinterpret_cast<tops::bfloat *>(output), index, num_tokens,
      num_experts, topk, norm_topk_prob != 0);
}
