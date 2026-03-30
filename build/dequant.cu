
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void dequant_int8(
    const int8_t* quantized, const float* scales, float* output,
    int n, int block_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = (float)quantized[i] * scales[i / block_size];
    }
}

extern "C" __declspec(dllexport) void launch_dequant_int8(
    const int8_t* q, const float* s, float* o, int n, int bs, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    dequant_int8<<<blocks, threads, 0, stream>>>(q, s, o, n, bs);
}
