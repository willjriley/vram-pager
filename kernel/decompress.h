#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

void launch_dequantize_int8(
    const int8_t* quantized, const __half* scales, __half* output,
    int num_elements, int block_size, cudaStream_t stream);

void launch_dequantize_int4(
    const uint8_t* quantized, const __half* scales, const __half* zeros,
    __half* output, int num_elements, int block_size, cudaStream_t stream);
