/**
 * Copyright 2020-2025 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "third_party/topk_ktop1.h"
#include "third_party/topk_bitonic.h"
#include "third_party/topk_heap_sort.h"
#include <tops.h>
#include <tops/tops_runtime.h>
#define TOPK_OP(T, RUST_NAME, DESC) \
extern "C" void topk_##RUST_NAME(  \
    T * in, T* out, u_int32_t * indices, int dim0, int dim1, int dim2, int k, void* stream_ \
) { \
    topsStream_t stream = (topsStream_t)(stream_);\
    const int AXIS = 2;\
    char* workspace;\
    int SHARED_SIZE = 65536 * 2 * 12 * (sizeof(float) * 2 + sizeof(int) * 2);\
    int h = 1, tmp = k;\
    while (tmp >>= 1) ++h;\
    int n2k_1 = (1 << h) - 1;\
    int numBlocks = 2;\
    int dimBlocks = 12;\
    size_t element = 1;\
    if (k < 512) {\
      element = n2k_1 * numBlocks * dimBlocks * dim0 * ALIGN_UP(dim1, 128);\
    } else if (k <= 1024) {\
      element = (n2k_1 + 1) * numBlocks * dimBlocks * dim0 * ALIGN_UP(dim1, 128);\
    } else if (k <= 60000) {\
      element = k * dim0 * ALIGN_UP(dim1, 128) + dim0 * dim1 * dim2 + 66 * numBlocks * dimBlocks;\
    }\
    size_t workspace_size = element * (sizeof(float) + sizeof(int32_t));\
    workspace_size = ALIGN_UP(workspace_size, 512);\
    topsMallocAsync(&workspace, workspace_size, stream, topsDeviceMallocDefault);\
    if (k < 10) {\
      topk_kernel_ktop1<T, DESC><<<2, 12, SHARED_SIZE, stream>>>(in, out, indices, workspace, dim0, dim1, dim2, AXIS, k, n2k_1);\
    } else if (k < 512) {\
      topk_kernel_heap_sort<T, DESC><<<2, 12, SHARED_SIZE, stream>>>(in, out, indices, workspace, dim0, dim1, dim2, AXIS, k, n2k_1);\
    } else {\
      topk_kernel_bitonic<T, DESC><<<2, 12, SHARED_SIZE, stream>>>(in, out, indices, workspace, dim0, dim1, dim2, AXIS, k, n2k_1);\
    }\
    topsFreeAsync(workspace, stream);\
}\

TOPK_OP(__fp16, f16, true)
TOPK_OP(__bf16, bf16, true)
TOPK_OP(float, f32, true)

#if 0
#include <random>
using namespace std;
int main() {
    const int NUM = 64;
    const int B = 11;

    float inputData[B * NUM] = {0};
    float outputData[B * NUM] = {0};

    u_int32_t indices[B * NUM];
    int dims[3] = {1, B, NUM};
    float* inputData_dev, *outputData_dev;
    const int K = 6;
    const int AXIS = 2;
    u_int32_t* indices_dev;
    char* workspace_dev;

    topsInit(0);
    topsSetDevice(0);
    topsStream_t stream;
    topsStreamCreate(&stream);

    topsMallocAsync(&inputData_dev, B * NUM * 4, stream, topsDeviceMallocDefault);
    topsMallocAsync(&outputData_dev, B * NUM * 4, stream, topsDeviceMallocDefault);
    topsMallocAsync(&indices_dev, B * NUM * 4, stream, topsDeviceMallocDefault);

    int workspace_size = dims[0] * (sizeof(float) + 4) * 24 * next_power_of_2(K) * ALIGN_UP(dims[1], 128);
    workspace_size = ALIGN_UP(workspace_size, 512);
    printf("workspace_size %d \n", workspace_size);
    topsMallocAsync(&workspace_dev, workspace_size, stream, topsDeviceMallocDefault);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    std::normal_distribution<float> dis(0, 1);
    for (int k = 0; k < B; k ++) {
      for (int i = 0; i < NUM; i++) {
        inputData[k * NUM + i] = dis(gen);
        indices[k * NUM + i] = i;
      }
    }


    topsMemcpyAsync(inputData_dev, inputData, B * NUM * 4,
                   topsMemcpyHostToDevice, stream);
    topsMemcpyAsync(indices_dev, indices, B * NUM * 4,
                   topsMemcpyHostToDevice, stream);

    topk_f32(inputData_dev, outputData_dev, indices_dev, reinterpret_cast<void*>(workspace_dev), dims[0], dims[1], dims[2], AXIS, K, stream);
    topsStreamSynchronize(stream);
    topsMemcpy(outputData, outputData_dev, B * NUM * 4,
                   topsMemcpyDeviceToHost);
    topsMemcpy(indices, indices_dev, B * NUM * 4,
                   topsMemcpyDeviceToHost);
    for (int k=0; k<B; k++) {
      printf("Batch %d \n", k);
      for (int i = 0; i < K; i++) {
        printf("indices %d, value %.5f\n", indices[k * K + i], outputData[k * K + i]);
      }
      printf("******* \n\n");

    }

}
#endif