#include "tops.h"
#pragma clang force_cuda_host_device begin
#include <stdio.h>
#pragma clang force_cuda_host_device end
#include <stdio.h>
#include <tops/tops_runtime.h>
#include <tops/topsrtc.h>
#include <tops/half.h>
#include "../utils.h"
//supported activation kernels: relu (0), gelu (1), leaky relu (2), tanh (3)

template <typename T, std::size_t N>
__device__ void copy_to_buffer(
  tops_dte_ctx_t& ctx, 
  T *buf_l3, 
  T *buf_l1)
{
  tops_dte_ctx_t p_ctx;
  tops::dte_scope p_s(p_ctx);

  tops::mdspan from(tops::Global, buf_l3, N);
  tops::mdspan to(tops::Private, buf_l1, N);
  tops::memcpy(p_ctx, to, from);
}

//inputType [input rows, input cols, input activation type]
//activation type: relu (0), gelu (1), leaky relu (2), tanh (3)
template<typename T, typename VT>
__device__ void activation(T *x, int* param)
{
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx);
    constexpr int vlen = tops::hvlength<VT>();
    __valigned__ int buffer[2];
    copy_to_buffer<int, 2>(ctx, param, buffer);
    int size = buffer[0];
    int type = buffer[1];

    // printf("size: %d, activation type: %d", size, type);

    __valigned__ T bufferA[vlen];
    __valigned__ T bufferB[vlen];
    __valigned__ T bufferO[vlen];

    int bufsize = vlen;
    if (bufsize > size) {bufsize=size;}

    tops::mdspan bufA(tops::Private, &bufferA, bufsize);
    tops::mdspan bufB(tops::Private, &bufferB, bufsize);
    tops::mdspan bufO(tops::Private, &bufferO, bufsize);

    for (int i=0; i<size; i+=bufsize) {
        if (i + bufsize >= size) {
            bufsize = size - i;
        }
        tops::mdspan srcA(tops::Global, x+i, bufsize);
        tops::memcpy(ctx, bufA, srcA);
        
        if (type == 0 || type == 2){ //relu and leaky relu
            for (int j=0; j<bufsize; j++){
                if (bufferA[j] > 0) {
                    bufferB[j] = static_cast<T>(1.0);
                } else {
                    bufferB[j] = (type == 2)? static_cast<T>(0.1f) : static_cast<T>(0.0f);
                }
            }
            const auto &v1 = tops::vload<VT>(bufferA);
            const auto &v2 = tops::vload<VT>(bufferB);
            tops::vstore(tops::vmul<VT>(v1, v2), bufferO);

        } else if (type == 3) {//tanh
            const auto &v1 = tops::vload<VT>(bufferA);
            tops::vstore(tops::vtanh<VT>(v1), bufferO);

        } else if (type == 1) {//gelu
            const auto &v1 = tops::vload<VT>(bufferA);
            tops::vstore(tops::vgelu<VT>(v1), bufferO);
        }

        //copy to results
        tops::mdspan dst(tops::Global, x+i, bufsize);
        tops::memcpy(ctx, dst, bufO);
    }

}

extern "C"  __global__ void activationf32(float *x, int* param)
{
    activation<float, vfloat>(x, param);
}

extern "C"  __global__ void activationf16(tops::half *x, int* param)
{
    activation<tops::half, vhalf>(x, param);

}

int main(int argc, char *argv[]) {
    float *a, *a_d;
    int *p, *p_d;
    topsHostMalloc((float**)&a, 5 * sizeof(float));
    topsHostMalloc((int**)&p, 2 * sizeof(int));
    for (int i=0; i< 5; i++) {
        a[i] = i * 1.0;
        printf("%.2f, ", a[i]);
    }
    p[0] = 5; p[1] = 3;

    topsMalloc(&a_d, 5 * sizeof(float));
    topsMalloc(&p_d, 2 * sizeof(int));
    topsMemcpy(a_d, a, 5 * sizeof(float),
                        topsMemcpyHostToDevice);
    topsMemcpy(p_d, p, 2 * sizeof(int),
                        topsMemcpyHostToDevice);

    activationf32<<<dim3(1,1,1), dim3(1,1,1)>>>(a_d, p_d);
    topsMemcpy(a, a_d, 5 * sizeof(float), topsMemcpyDeviceToHost);
    for (int i=0; i< 5; i++) {
        printf("%.2f, ", a[i]);
    }
    return 0;
}