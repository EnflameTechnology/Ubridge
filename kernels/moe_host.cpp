#include <acore_op.h>
#include "utils/utils.h"
#include <tops.h>
#include <tops/tops_runtime.h>
//Example:
// y = [N, M]
// idx=[K]
// top=[K]
// e_out = [K, M]
// w = [N, topk]
#define MAX_M_DIM 1024 * 8
#define MAX_IDX_DIM 1024 * 4
//case 1: batch = 1, seq_len = 7
//prefiling
//y [7, 5120], e_out [3, 5120], topk_weight [7, 6], idx [3], top [3]
//y [7, 5120], e_out [1, 5120], topk_weight [7, 6], idx [1], top [1]
//decoding
//y [1, 5120], e_out [1, 5120], topk_weight [1, 6], idx [1], top [1]

//case 2: batch==8, seq_len=28
//prefiling
//y [224, 5120], e_out [15, 5120], topk_weight [224, 6], idx [15], top [15]
//decoding
//y [8, 5120], e_out [8, 5120], topk_weight [8, 6], idx [8], top [8]

template<typename T, typename ID_TYPE>
__global__ void moe_kernel(T* y, T* e_out, float* w, ID_TYPE* idx, ID_TYPE* top, int N, int K, int M, int topk) {
    int thread_id = GetThreadIdx();
    int MAX_THREADS = GetThreadNum();
    int THREAD_STEP = 1;
    int thread_step = 1;
    __local__ __valigned__ T y_buffer[MAX_M_DIM];
    __local__ __valigned__ T e_buffer[MAX_M_DIM];
    __local__ __valigned__ T tmp_buffer[MAX_M_DIM];

    __local__ __valigned__ ID_TYPE idx_buffer[MAX_IDX_DIM];
    __local__ __valigned__ ID_TYPE top_buffer[MAX_IDX_DIM];
    __local__ __valigned__ float w_buffer[MAX_M_DIM*16];

    tops_dte_ctx_t ctx[3];
    tops_dte_ctx_t ctxs_y;
    tops_dte_ctx_t ctxs_e;
    tops_dte_ctx_t ctxs_o;
    tops::event ev_y;
    tops::event ev_e;
    tops::event ev[3];
    tops::mdspan l1_y(tops::Private, y_buffer, M);
    tops::mdspan l1_e(tops::Private, e_buffer, M);
    tops::mdspan l1_idx(tops::Private, idx_buffer, K);
    tops::mdspan l1_top(tops::Private, top_buffer, K);
    tops::mdspan l1_w(tops::Private, w_buffer, N * topk);
    
    ev[2] = tops::memcpy_async(ctx[0], l1_idx, tops::mdspan(tops::Global, idx, K));
    ev[1] = tops::memcpy_async(ctx[1], l1_top, tops::mdspan(tops::Global, top, K));
    ev[0] = tops::memcpy_async(ctx[2], l1_w, tops::mdspan(tops::Global, w, N * topk));
    GetThreadStep(K, thread_step, THREAD_STEP);

    for (int i=0; i<3; i++) {
      ev[i].wait();
    }
    for (int i = 0; i < thread_step; i++) {
      int k = thread_id * THREAD_STEP + i;
      if (k < K) {
        if (idx_buffer[k] < N) {
          tops::mdspan hbm_y1(tops::Global, y + idx_buffer[k] * M, M);
          tops::mdspan hbm_e1(tops::Global, e_out + k * M, M);
          uint32_t top_idx = top_buffer[k];
          if (top_idx < topk) {
            ev_e = tops::memcpy_async(ctxs_e, l1_e, hbm_e1);
            ev_y = tops::memcpy_async(ctxs_y, l1_y, hbm_y1);
            float w1 = w_buffer[idx_buffer[k] * topk + top_idx];
            ev_e.wait();
            mul<T, T, float>(reinterpret_cast<T*>(tmp_buffer), reinterpret_cast<T*>(e_buffer), w1, M);
            ev_y.wait();
            add(reinterpret_cast<T*>(e_buffer), reinterpret_cast<T*>(y_buffer), reinterpret_cast<T*>(tmp_buffer), M);
            tops::memcpy(ctxs_o, hbm_y1, l1_e);
          } 
        } 
      }
    }
}

#define MOE_OP(T, ID_TYPE, RUST_NAME) \
extern "C" void moe_##RUST_NAME(  \
    T* y, T* e_out, float* w, ID_TYPE* idx, ID_TYPE* top, int N, int K, int M, int topk, void* stream_ \
) { \
    topsStream_t stream = (topsStream_t)(stream_);\
    int numBlocks = 2;\
    int dimBlocks = 12;\
    moe_kernel<T, ID_TYPE><<<2, 12, 0, stream>>>(y, e_out, w, idx, top, N, K, M, topk);\
}\

MOE_OP(__fp16, u_int32_t, f16)
MOE_OP(__bf16, u_int32_t, bf16)

#include <tops/half.h>
#include <tops/bfloat.h>
#include "utils/utils.h"

//input: [batch, M (topk or 1), k]
//weight: [num_experts, n, k]
//indices: [batch, topk]
//output: [batch, topk, n]

//example:
//input [3355, 1, 2048]
//weight [128, 768, 2048]
//indices [3355, 8]
//output [3355, 8, 768]
#define MAX_TOP_K 128
#define K_TILE 128
#define N_TILE 128
#define MAX_K 4096
#define MAX_N 4096
#define TEMPLATE_ALIGN_UP(a, b) (((a + b - 1) / b) * b)
#define L1_ALIGN_SIZE (128)
template<typename T, typename ID_TYPE>
__global__ void indexed_moe_kernel(T* in, T* w, T* out, ID_TYPE* indices, int N, int K, int M, int batch, int topk, int num_experts) {
    bool enable_quant = false;
    int launch_times = 0;
    int vab_offset = (enable_quant == 1) ? 512 : 0;
    int THREAD_STEP = 1;
    int thread_step = 1;
    GetBlockThreadStep(batch, thread_step, THREAD_STEP);
    __local__ __valigned__ char buffer_sip[VDMEM_SIZE];
    int* local_workspace = reinterpret_cast<int*>(buffer_sip);
    ID_TYPE* idx_buffer = reinterpret_cast<ID_TYPE *>(
      (reinterpret_cast<char *>(local_workspace)) +
        TEMPLATE_ALIGN_UP(2048, L1_ALIGN_SIZE));

    T* in_buffer = reinterpret_cast<T *>(
      (reinterpret_cast<char *>(idx_buffer)) +
        TEMPLATE_ALIGN_UP(MAX_TOP_K * sizeof(ID_TYPE), L1_ALIGN_SIZE));

    T* o_buffer = reinterpret_cast<T *>(
      (reinterpret_cast<char *>(in_buffer)) +
        TEMPLATE_ALIGN_UP(MAX_K * sizeof(T), L1_ALIGN_SIZE));

    T* e_buffer = reinterpret_cast<T *>(
      (reinterpret_cast<char *>(o_buffer)) +
        TEMPLATE_ALIGN_UP(MAX_N * sizeof(T), L1_ALIGN_SIZE));

    tops_dte_ctx_t ctxs_in;
    tops_dte_ctx_t ctxs_e;
    tops_dte_ctx_t ctxs_idx;
    tops_dte_ctx_t ctxs_out;
    ctxs_in.init();
    ctxs_e.init();
    ctxs_idx.init();
    ctxs_out.init();

    int32_t l1_w_shape[2] = {N_TILE, K};
    int32_t hbm_w_shape[2] = {N, K};
    ctxs_e.config_slice(mdspan(Private, e_buffer, l1_w_shape),
                      mdspan(Global, w, hbm_w_shape), {0, 0});

    tops::event ev[3];
    tops::mdspan l1_in(tops::Private, in_buffer, K);
    tops::mdspan l1_idx(tops::Private, idx_buffer, topk);
    tops::mdspan l1_out(tops::Private, o_buffer, N);

    for (int j = 0; j < thread_step; j++) {
      int idx = GetThreadIdxInBlock() * THREAD_STEP + j;
      if (idx >= batch) { continue; }
      tops::memcpy(ctxs_idx, l1_idx, tops::mdspan(tops::Global, indices + idx * topk, topk));
      T* cur_out = out + idx * topk * N;
      for (int i = GetBlockIdx(); i < topk; i += GetBlockNum()) {
        T* cur_input = in + idx * M * K + ((M == 1) ? 0 : i * K);
        int expert_id = (int)idx_buffer[i];
        if ((M == 1 && i == GetBlockIdx()) || M > 1) {
          tops::memcpy(ctxs_in, l1_in, tops::mdspan(tops::Global, cur_input, K));
        }

        ctxs_e.set_src_addr(w + expert_id * N * K);
        for (int n_idx = 0; n_idx < N / N_TILE; n_idx++) {
          if (expert_id < num_experts) {
            ctxs_e.set_src_offset(0, n_idx * N_TILE);
            ctxs_e.trigger_and_wait();
            matmul<1, MK_NK>(o_buffer + n_idx * N_TILE, in_buffer, e_buffer, e_buffer, local_workspace,
                                K, N_TILE, 0, 1, false,
                                vab_offset, launch_times);
            launch_times += 1;
          }
        } // N loop

        __dtu_movs_barrier_all();
        #if __GCU_ARCH__ == 400
                tcle::fence<0xd2>();
        #elif __GCU_ARCH__ == 300
                tcle::fence<0x3>();
        #endif

        tops::memcpy_async(
            ctxs_out, tops::mdspan(tops::Global, cur_out + i * N, N), l1_out);
      }
    }
}

#define N_DECODE_TILE 64
template<typename T, typename ID_TYPE>
__global__ void indexed_moe_kernel_decoding(T* in, T* w, T* out, ID_TYPE* indices, int N, int K, int M, int batch, int topk, int num_experts) {
    bool enable_quant = false;
    int launch_times = 0;
    int vab_offset = (enable_quant == 1) ? 512 : 0;
    int THREAD_STEP = 1;
    int thread_step = 1;
    GetBlockThreadStep(N / N_DECODE_TILE, thread_step, THREAD_STEP);
    __local__ __valigned__ char buffer_sip[VDMEM_SIZE];
    int* local_workspace = reinterpret_cast<int*>(buffer_sip);
    ID_TYPE* idx_buffer = reinterpret_cast<ID_TYPE *>(
      (reinterpret_cast<char *>(local_workspace)) +
        TEMPLATE_ALIGN_UP(2048, L1_ALIGN_SIZE));

    T* in_buffer = reinterpret_cast<T *>(
      (reinterpret_cast<char *>(idx_buffer)) +
        TEMPLATE_ALIGN_UP(MAX_TOP_K * sizeof(ID_TYPE), L1_ALIGN_SIZE));

    T* o_buffer = reinterpret_cast<T *>(
      (reinterpret_cast<char *>(in_buffer)) +
        TEMPLATE_ALIGN_UP(MAX_K * sizeof(T), L1_ALIGN_SIZE));

    T* e_buffer = reinterpret_cast<T *>(
      (reinterpret_cast<char *>(o_buffer)) +
        TEMPLATE_ALIGN_UP(MAX_N * sizeof(T), L1_ALIGN_SIZE));

    tops_dte_ctx_t ctxs_in;
    tops_dte_ctx_t ctxs_e;
    tops_dte_ctx_t ctxs_idx;
    tops_dte_ctx_t ctxs_out;
    ctxs_in.init();
    ctxs_e.init();
    ctxs_idx.init();
    ctxs_out.init();

    int32_t l1_w_shape[2] = {N_DECODE_TILE, K};
    int32_t hbm_w_shape[2] = {N, K};
    ctxs_e.config_slice(mdspan(Private, e_buffer, l1_w_shape),
                      mdspan(Global, w, hbm_w_shape), {0, 0});

    tops::event ev[3];
    tops::mdspan l1_in(tops::Private, in_buffer, K);
    tops::mdspan l1_idx(tops::Private, idx_buffer, topk);
    tops::mdspan l1_out(tops::Private, o_buffer, N_DECODE_TILE);

    for (int idx = 0; idx < batch; idx++) {
      tops::memcpy(ctxs_idx, l1_idx, tops::mdspan(tops::Global, indices + idx * topk, topk));
      T* cur_out = out + idx * topk * N;
      for (int i = GetBlockIdx(); i < topk; i += GetBlockNum()) {
        T* cur_input = in + idx * M * K + ((M == 1) ? 0 : i * K);
        int expert_id = (int)idx_buffer[i];
        tops::memcpy(ctxs_in, l1_in, tops::mdspan(tops::Global, cur_input, K));

        ctxs_e.set_src_addr(w + expert_id * N * K);
        for (int j = 0; j < thread_step; j++) {
          int n_idx = GetThreadIdxInBlock() * THREAD_STEP + j;
          if (n_idx < N / N_DECODE_TILE && expert_id < num_experts) {
            ctxs_e.set_src_offset(0, n_idx * N_DECODE_TILE);
            ctxs_e.trigger_and_wait();
            matmul<32, MK_NK>(o_buffer, in_buffer, e_buffer, e_buffer, local_workspace,
                                K, N_DECODE_TILE, 0, 1, false,
                                vab_offset, launch_times);
            launch_times += 1;

            __dtu_movs_barrier_all();
            #if __GCU_ARCH__ == 400
                    tcle::fence<0xd2>();
            #elif __GCU_ARCH__ == 300
                    tcle::fence<0x3>();
            #endif
            tops::memcpy_async(
            ctxs_out, tops::mdspan(tops::Global, cur_out + i * N + n_idx * N_DECODE_TILE, N_DECODE_TILE), l1_out);
          }
        } // N loop
      } // topk loop
    } // batch loop
}

#define INDEXED_MOE_OP(T, ID_TYPE, RUST_NAME) \
extern "C" void indexed_moe_##RUST_NAME(  \
    T* in, T* w, T* out, ID_TYPE* indices, int N, int K, int M, int batch, int topk, int num_experts, void* stream_ \
) { \
    topsStream_t stream = (topsStream_t)(stream_);\
    int numBlocks = 2;\
    int dimBlocks = 12;\
    if (batch > 12) {\
      indexed_moe_kernel<T, ID_TYPE><<<2, 12, 0, stream>>>(in, w, out, indices, N, K, M, batch, topk, num_experts);\
    } else { \
      indexed_moe_kernel_decoding<T, ID_TYPE><<<2, 12, 0, stream>>>(in, w, out, indices, N, K, M, batch, topk, num_experts);\
    }\
}\

INDEXED_MOE_OP(__fp16, u_int32_t, f16)
INDEXED_MOE_OP(__bf16, u_int32_t, bf16)


// #define KERNEL_TEST

#ifdef KERNEL_TEST
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <type_traits>

template<typename T>
inline float load_as_float(const T* ptr, size_t idx);

template<>
inline float load_as_float<__fp16>(const __fp16* ptr, size_t idx) {
    return (float)(ptr[idx]);
}


template<typename T>
inline void store_from_float(T* ptr, size_t idx, float val);

template<>
inline void store_from_float<__fp16>(__fp16* ptr, size_t idx, float val) {
    ptr[idx] = (__fp16)(val);
}

// Core MoE compute
template<typename T>
void moe_matmul(
    const T* input,        // [batch, M, k]
    const T* weight,       // [num_experts, n, k]
    const u_int32_t* indices,    // [batch, topk]
    T* output,             // [batch, topk, n]
    int batch, int M, int topk, int k, int n, int num_experts
) {
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < topk; ++t) {
            int expert_id = (int)indices[b * topk + t];
            const T* expert_w = weight + expert_id * n * k;

            int m = (M == 1 ? 0 : t);
            const T* input_vec = input + (b * M + m) * k;

            for (int ni = 0; ni < n; ++ni) {
                float acc = 0.0f;
                const T* wrow = expert_w + ni * k;
                for (int ki = 0; ki < k; ++ki) {
                    float in_val = (float)input_vec[ki];
                    float w_val  = (float)wrow[ki];
                    acc += in_val * w_val;
                }
                output[(b * topk + t) * n + ni] = (T)acc;
                // store_from_float(output, (b * topk + t) * n + ni, acc);
            }
        }
    }
}

__fp16 get_rand_value(int num_experts) {
  int r = rand() % num_experts;
  if (r < num_experts / 5) {
    return 0.25;
  } 
  
  else if (r < num_experts / 4) {
    return 0.78;
  } else if (r < num_experts / 3) {
    return 0.95;
  } else if (r < num_experts / 2) {
    return 0.16;
  } 
  
  else {
    return 0.3;
  }
}
//example:
//input [3355, 1, 2048]
//weight [128, 768, 2048]
//indices [3355, 8]
//output [3355, 8, 768]
int main(int argc, char* argv[]) {
  int batch = 32;  // default value

  topsError_t err = topsSuccess;
  int M = 8;
  int K = 2048;
  int N = 384;
  int topk = 8;
  int num_experts = 128;

  for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];
      if (arg == "--batch" && i + 1 < argc) {
          batch = std::stoi(argv[i + 1]);  // convert string to int
          i++; // skip the value
      }

      if (arg == "--m" && i + 1 < argc) {
          M = std::stoi(argv[i + 1]);  // convert string to int
          i++; // skip the value
      }

      if (arg == "--k" && i + 1 < argc) {
          K = std::stoi(argv[i + 1]);  // convert string to int
          i++; // skip the value
      }

      if (arg == "--n" && i + 1 < argc) {
          N = std::stoi(argv[i + 1]);  // convert string to int
          i++; // skip the value
      }
  }
  int in_euem = batch * M * K;
  int out_euem = batch * topk * N;
  int weight_euem = num_experts * K * N;
  int indices_euem = batch * topk;

  int in_size = in_euem * sizeof(__fp16);
  int out_size = out_euem * sizeof(__fp16);
  int weight_size = weight_euem * sizeof(__fp16);
  int indices_size = indices_euem * sizeof(u_int32_t);
  std::cout << "Prepare data.. " << endl;

  __fp16 *host_in = reinterpret_cast<__fp16*>(aligned_alloc(128, in_size));
  __fp16 *host_weight = reinterpret_cast<__fp16*>(aligned_alloc(128, weight_size));
  u_int32_t *host_indices = reinterpret_cast<u_int32_t*>(aligned_alloc(128, indices_size));


  for (int i=0; i< batch * M * K; i++) {
      host_in[i] = (__fp16)(get_rand_value(num_experts));
  }

  for (int i=0; i< num_experts * K * N; i++) {
    host_weight[i] = (__fp16)(get_rand_value(num_experts));
  }

  for (int i=0; i< batch; i++) {
    for (int k=0; k<topk; k++) {
      host_indices[i * topk + k] = rand() % num_experts;
    }
  }
  __fp16 *host_out = reinterpret_cast<__fp16*>(aligned_alloc(128, out_size));
  __fp16 *host_out_cpu = reinterpret_cast<__fp16*>(aligned_alloc(128, out_size));

  __fp16 *dev_in = NULL;
  __fp16 *dev_weight = NULL;
  u_int32_t *dev_indices = NULL;

  CHECK(topsMalloc(reinterpret_cast<void **>(&dev_in), in_size));
  CHECK(topsMalloc(reinterpret_cast<void **>(&dev_weight), weight_size));
  CHECK(topsMalloc(reinterpret_cast<void **>(&dev_indices), indices_size));

  __fp16 *dev_out = NULL;
  CHECK(topsMalloc(reinterpret_cast<void **>(&dev_out), out_size));

  CHECK(topsMemcpy(dev_in, host_in, in_size, topsMemcpyHostToDevice));
  CHECK(topsMemcpy(dev_weight, host_weight, weight_size, topsMemcpyHostToDevice));
  CHECK(topsMemcpy(dev_indices, host_indices, indices_size, topsMemcpyHostToDevice));

  CHECK(topsMemset(dev_out, 0, out_size));

  printf("call indexed moe kernel!!!!!!!!!!!!!\n");
  topsStream_t stream;
  topsStreamCreate(&stream);
  topsEvent_t start, stop;
  topsEventCreate(&start);
  topsEventCreate(&stop);

  topsEventRecord(start, stream);

  auto start_t = std::chrono::system_clock::now();
  //three indexed_moe call per layer, 48 layers: one forward pass
  // for (int i=0; i< 3 * 48; i++)
  indexed_moe_kernel<__fp16, u_int32_t><<<2, 12, 0, stream>>>(dev_in, dev_weight, dev_out, dev_indices, N, K, M, batch, topk, num_experts);
  topsEventRecord(stop, stream);
  topsEventSynchronize(stop);
  auto end_t = std::chrono::system_clock::now();

  float ms = 0.0f;
  topsEventElapsedTime(&ms, start, stop);
  std::cout << "Kernel execution time: " << ms << " ms\n";

  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_t - start_t);
  std::cout << elapsed.count() << "ms" << endl;

  if (topsGetLastError() != topsSuccess) {
    fprintf(stderr, "Failed to launch indexed moe kernel (error code %s)!\n",
            topsGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  topsStreamSynchronize(stream);

  CHECK(topsMemcpy(host_out, dev_out, out_size, topsMemcpyDeviceToHost));
  printf("Checking results...\n");
  bool passed = true;
  moe_matmul<__fp16>(host_in, host_weight, host_indices, host_out_cpu, batch, M, topk, K, N, num_experts);
  for (int i=0; i< batch; i++) {
    for (int k=0; k<topk; k++) {
      for (int n=0; n<N; n++) {
        float a = (float)host_out_cpu[i * topk * N + k * N + k];
        float b = (float)host_out[i * topk * N + k * N + k];
        if (abs(a - b) > 0.00001) {
            printf("[%d, %d, %d] cpu %.6f gcu %.6f\n", i, k, n, a, b);
            passed = false;
        } else if (n == 0) {
            printf("OK [%d, %d, %d] cpu %.6f gcu %.6f\n", i, k, n, a, b);
        }
      }
    }
  }
  if (passed) {
    printf("Test PASSED\n");
  } else {
    printf("Test Faild\n");
  }

  CHECK(topsFree(dev_in));
  CHECK(topsFree(dev_weight));
  CHECK(topsFree(dev_indices));
  CHECK(topsFree(dev_out));
  free(host_in);
  free(host_weight);
  free(host_indices);
  free(host_out);
  return 0;
}
#endif
