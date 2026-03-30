/**
 * VRAM Pager — GPU Decompression Kernels
 *
 * Decompresses quantized weights (INT4/INT8) to FP16 on the GPU
 * during the forward pass. Weights are stored compressed in system RAM,
 * transferred compressed across PCIe, and decompressed in GPU shared memory.
 *
 * This is the performance-critical path. Every microsecond matters here
 * because this runs for every layer of every step of every generation.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// ============================================================
// INT8 → FP16 Dequantization
// Each INT8 value is dequantized using a per-block scale factor:
//   fp16_value = int8_value * scale
// Block size: 128 elements share one FP16 scale
// ============================================================
extern "C" __global__ void dequantize_int8_to_fp16(
    const int8_t* __restrict__ quantized,    // INT8 weights (compressed)
    const __half* __restrict__ scales,       // Per-block scale factors
    __half* __restrict__ output,             // FP16 output (decompressed)
    int num_elements,
    int block_size)                           // Elements per scale block (default 128)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        int scale_idx = i / block_size;
        __half scale = scales[scale_idx];
        output[i] = __hmul(__int2half_rn((int)quantized[i]), scale);
    }
}

// ============================================================
// INT4 → FP16 Dequantization
// Two INT4 values packed per byte. Each nibble dequantized separately.
// Per-block scale + zero point for accuracy.
//   fp16_value = (int4_value - zero_point) * scale
// Block size: 32 elements (16 bytes) share one scale+zero
// ============================================================
extern "C" __global__ void dequantize_int4_to_fp16(
    const uint8_t* __restrict__ quantized,   // Packed INT4 (2 per byte)
    const __half* __restrict__ scales,       // Per-block scales
    const __half* __restrict__ zeros,        // Per-block zero points
    __half* __restrict__ output,             // FP16 output
    int num_elements,
    int block_size)                           // Elements per scale block (default 32)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int elem_idx = i * 2;  // Each thread processes 2 elements (1 byte)

    if (elem_idx < num_elements) {
        uint8_t packed = quantized[i];
        int scale_idx = elem_idx / block_size;
        __half scale = scales[scale_idx];
        __half zero = zeros[scale_idx];

        // Low nibble (first element)
        int val_lo = (int)(packed & 0x0F);
        output[elem_idx] = __hmul(__hsub(__int2half_rn(val_lo), zero), scale);

        // High nibble (second element)
        if (elem_idx + 1 < num_elements) {
            int val_hi = (int)((packed >> 4) & 0x0F);
            output[elem_idx + 1] = __hmul(__hsub(__int2half_rn(val_hi), zero), scale);
        }
    }
}

// ============================================================
// Launcher functions (called from Python via pybind11)
// ============================================================

void launch_dequantize_int8(
    const int8_t* quantized, const __half* scales, __half* output,
    int num_elements, int block_size, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    dequantize_int8_to_fp16<<<blocks, threads, 0, stream>>>(
        quantized, scales, output, num_elements, block_size);
}

void launch_dequantize_int4(
    const uint8_t* quantized, const __half* scales, const __half* zeros,
    __half* output, int num_elements, int block_size, cudaStream_t stream)
{
    int threads = 256;
    int packed_elements = (num_elements + 1) / 2;
    int blocks = (packed_elements + threads - 1) / threads;
    dequantize_int4_to_fp16<<<blocks, threads, 0, stream>>>(
        quantized, scales, zeros, output, num_elements, block_size);
}
