
#include <limits>
#include <tops.h>
#include <tops/dte_chain.h>
#include <tops/half.h>
#include <tops/bfloat.h>
#include "tops/tops_runtime.h"
#include "utils/utils.h"
#define L1_ALIGN_SIZE (128)
#define TEMPLATE_ALIGN_UP(a, b) (((a + b - 1) / b) * b)
#define MIN_THREAD_STEP 1024

template <typename T, bool ascending>
__device__ void bitonic_sort_kernel(T* arr, u_int32_t * dst, int j, int k, unsigned int cols_pad) {
    __shared__ T temp_val[12];
    __shared__ u_int32_t temp_idx[12];
    int thread_id = GetThreadIdxInBlock(); // Thread index 
    int thread_block_step;
    int THREAD_BLOCK_STEP;
    GetBlockThreadStep(cols_pad, thread_block_step, THREAD_BLOCK_STEP);
    unsigned int idx = thread_id * THREAD_BLOCK_STEP;
    for (int i = idx; i < std::min(idx + thread_block_step, cols_pad); i++) {
        unsigned int ij = i ^ j;
        if (ij > i) {
            if constexpr (ascending) {
                if ((i & k) == 0) {
                    if (arr[i] > arr[ij]) {
                        temp_val[thread_id] = arr[i];
                        arr[i] = arr[ij];
                        arr[ij] = temp_val[thread_id];

                        temp_idx[thread_id] = dst[i];
                        dst[i] = dst[ij];
                        dst[ij] = temp_idx[thread_id];
                    }
                } else {
                    if (arr[i] < arr[ij]) {
                        temp_val[thread_id] = arr[i];
                        arr[i] = arr[ij];
                        arr[ij] = temp_val[thread_id];

                        temp_idx[thread_id] = dst[i];
                        dst[i] = dst[ij];
                        dst[ij] = temp_idx[thread_id];
                    }
                }
            }

            if constexpr (!ascending) {
                if ((i & k) != 0) {
                    if (arr[i] > arr[ij]) {
                        temp_val[thread_id] = arr[i];
                        arr[i] = arr[ij];
                        arr[ij] = temp_val[thread_id];

                        temp_idx[thread_id] = dst[i];
                        dst[i] = dst[ij];
                        dst[ij] = temp_idx[thread_id];
                    }
                } else {
                    if (arr[i] < arr[ij]) {
                        temp_val[thread_id] = arr[i];
                        arr[i] = arr[ij];
                        arr[ij] = temp_val[thread_id];

                        temp_idx[thread_id] = dst[i];
                        dst[i] = dst[ij];
                        dst[ij] = temp_idx[thread_id];
                    }
                }
            }
        }
    }
    __syncthreads();
}


int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    };
    return n;
}

template <typename T, bool Ascending>
__device__ void SortKernel(T *values, u_int32_t *indices, int rows, int cols, int ncols_pad, bool inplace) {
    tops_dte_ctx_t ctx;
    ctx.init();
    tops::dte_scope s(ctx); 
    extern __shared__ char l2_buffer[];
    T* shared_values = reinterpret_cast<T*>(l2_buffer);
    u_int32_t* shared_indices = reinterpret_cast<u_int32_t *>(
        (reinterpret_cast<char *>(shared_values)) +
            TEMPLATE_ALIGN_UP(ncols_pad * sizeof(T), L1_ALIGN_SIZE));
    int N = cols;
    int thread_block_step;
    int THREAD_BLOCK_STEP;
    GetBlockThreadStep(N, thread_block_step, THREAD_BLOCK_STEP);

    //Final merge for each row
    for (int row = 0; row < rows; row++) {
        int offset = GetThreadIdxInBlock() * THREAD_BLOCK_STEP;
        tops::mdspan shared_buffer(tops::Shared, shared_values + offset, thread_block_step);
        tops::mdspan global_buffer(tops::Global, values + row * cols + offset, thread_block_step);
        tops::memcpy(ctx, shared_buffer, global_buffer);
        for (int k=offset; k<offset+thread_block_step; k++) {
            shared_indices[k] = k;
        }
        if (GetThreadIdxInBlock() == 0) {
            for (int k=cols; k<ncols_pad; k++) {
                shared_values[k] = Ascending ? T(10000.0) : T(-10000.0); //padding
                shared_indices[k] = cols;
            }
        }
        __syncthreads();

        for (int k = 2; k <= ncols_pad; k <<= 1) {
            for (int j = k >> 1; j > 0; j = j >> 1) {
                bitonic_sort_kernel<T, Ascending>(shared_values, shared_indices, j, k, ncols_pad);
            }
        }
        
        __syncthreads();
        if (inplace)
            tops::memcpy(ctx, global_buffer, shared_buffer);
        tops::mdspan shared_ind(tops::Shared, shared_indices + offset, thread_block_step);
        tops::mdspan global_ind(tops::Global, indices + row * cols + offset, thread_block_step);
        tops::memcpy(ctx, global_ind, shared_ind);
    }
}

#define ASORT_OP(T, RUST_NAME, ASC) \
extern "C" __global__ void RUST_NAME(  \
    T * x, u_int32_t * dst, const int nrows, const int ncols, const int ncols_pad, bool inplace \
) { \
    SortKernel<T, ASC>(x, dst, nrows, ncols, ncols_pad, inplace);\
}\

ASORT_OP(__bf16, asort_asc_bf16, true)
ASORT_OP(__bf16, asort_desc_bf16, false)

ASORT_OP(__fp16, asort_asc_f16, true)
ASORT_OP(__fp16, asort_desc_f16, false)

ASORT_OP(float, asort_asc_f32, true)
ASORT_OP(double, asort_asc_f64, true)
ASORT_OP(uint8_t, asort_asc_u8, true)
ASORT_OP(uint32_t, asort_asc_u32, true)
ASORT_OP(int64_t, asort_asc_i64, true)

ASORT_OP(float, asort_desc_f32, false)
ASORT_OP(double, asort_desc_f64, false)
ASORT_OP(uint8_t, asort_desc_u8, false)
ASORT_OP(uint32_t, asort_desc_u32, false)
ASORT_OP(int64_t, asort_desc_i64, false)
