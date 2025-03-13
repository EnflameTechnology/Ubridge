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

#include "third_party/topk_bitonic.h"
#include <tops.h>
#include <tops/tops_runtime.h>
#define TOPK_OP(T, RUST_NAME, DESC) \
extern "C" void topk_##RUST_NAME(  \
    T * in, T* out, u_int32_t * indices, void* workspace, int dim0, int dim1, int dim2, int axis, int k, void* stream_ \
) { \
    topsStream_t stream = (topsStream_t)(stream_);\
    topk_kernel_bitonic<T, DESC><<<2, 12, SHARE_BUFFER_SIZE, stream>>>(in, out, indices, workspace, dim0, dim1, dim2, axis, k);\
}\

TOPK_OP(__fp16, f16, true)
TOPK_OP(__bf16, bf16, true)
TOPK_OP(float, f32, true)

#if 0
#include <random>
using namespace std;
int main() {
    const int NUM = 1024 * 128;
    float inputData[NUM] = {0};
    float outputData[NUM] = {0};

    u_int32_t indices[NUM];
    int dims[3] = {1, 1, NUM};
    float* inputData_dev, *outputData_dev;
    const int K = 6;
    const int AXIS = 2;
    u_int32_t* indices_dev;
    char* workspace_dev;

    topsInit(0);
    topsSetDevice(0);
    topsStream_t stream;
    topsStreamCreate(&stream);

    topsMallocAsync(&inputData_dev, NUM * 4, stream, topsDeviceMallocDefault);
    topsMallocAsync(&outputData_dev, NUM * 4, stream, topsDeviceMallocDefault);
    topsMallocAsync(&indices_dev, NUM * 4, stream, topsDeviceMallocDefault);

    int workspace_size = dims[0] * (sizeof(float) + 4) * 24 * next_power_of_2(K) * ALIGN_UP(dims[1], 128);
    workspace_size = ALIGN_UP(workspace_size, 512);
    printf("workspace_size %d \n", workspace_size);
    topsMallocAsync(&workspace_dev, workspace_size, stream, topsDeviceMallocDefault);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    std::normal_distribution<float> dis(0, 1);
    for (int i = 0; i < NUM; i++) {
      inputData[i] = dis(gen);
      indices[i] = i;
    }

    topsMemcpyAsync(inputData_dev, inputData, NUM * 4,
                   topsMemcpyHostToDevice, stream);
    topsMemcpyAsync(indices_dev, indices, NUM * 4,
                   topsMemcpyHostToDevice, stream);

    topk_f32(inputData_dev, outputData_dev, indices_dev, reinterpret_cast<void*>(workspace_dev), dims[0], dims[1], dims[2], AXIS, K, stream);
    topsStreamSynchronize(stream);
    topsMemcpy(outputData, outputData_dev, NUM * 4,
                   topsMemcpyDeviceToHost);
    topsMemcpy(indices, indices_dev, NUM * 4,
                   topsMemcpyDeviceToHost);
    for (int i = 0; i < K; i++) {
      printf("indices %d, value %.5f\n", indices[i], outputData[i]);
    }
}
#endif