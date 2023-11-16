#include <stdio.h>
#include <tops/tops_runtime.h>
#include <tops/topsrtc.h>
#include <tops/half.h>
template<typename T, typename VT>
__device__ void element(T *matA, T* matB, T *to, int size, int type)
{
  tops_dte_ctx_t ctx;
  tops::dte_scope s(ctx);

  __valigned__ T bufferA[32];
  __valigned__ T bufferB[32];

  //printf("size: %d, type: %d", size, type);

  int bufsize = 32;
  if (bufsize > size) {bufsize=size;}

  tops::mdspan bufA(tops::Private, (T*)&bufferA, bufsize);
  tops::mdspan bufB(tops::Private, (T*)&bufferB, bufsize);

  for (unsigned i=0; i<size; i+=bufsize) {
    if (i + bufsize >= size) {
      bufsize = size - i;
    }
    tops::mdspan srcA(tops::Global, matA + i, bufsize);
    tops::mdspan srcB(tops::Global, matB + i, bufsize);

    tops::memcpy(ctx, bufA, srcA);
    tops::memcpy(ctx, bufB, srcB);

    const auto &v1 = tops::vload<VT>(bufferA);
    const auto &v2 = tops::vload<VT>(bufferB);

    if (type == 0) {
        tops::vstore<VT>(v1+v2, bufferB);
    } else if (type == 1) {
        tops::vstore<VT>(v1-v2, bufferB);
    } else if (type == 2) {
        tops::vstore<VT>(v1*v2, bufferB);
    } else if (type == 3) {
        tops::vstore<VT>(v1/v2, bufferB);
    }

    tops::mdspan dst(tops::Global, to + i, bufsize);
    tops::memcpy(ctx, dst, bufB);
  }

}

extern "C" __global__ void elementf32(float *matA, float* matB, float *to, int size, int type)
{
    element<float, vfloat>(matA, matB, to, size, type);

}

extern "C" __global__ void elementf16(tops::half *matA, tops::half* matB, tops::half *to, int size, int type)
{
    element<tops::half, vhalf>(matA, matB, to, size, type);
}

extern "C" __global__ void elementi32(int32_t *matA, int32_t* matB, int32_t *to, int size, int type)
{
    element<int32_t, vint>(matA, matB, to, size, type);

}

int element_cpp(float *matA, float* matB, float *to, int size, int type) {
    elementf32<<<dim3(1,1,1), dim3(1,1,1)>>>(matA, matB, to, size, type);
    return 0;
}

int main() {
    return 0;

}