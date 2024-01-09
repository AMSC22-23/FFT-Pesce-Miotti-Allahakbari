#ifndef REAL_HPP_GPU
#define REAL_HPP_GPU

/**
 * @file FFTGPU.hpp.
 * @brief Declares functions to execute FFTs on a GPU using CUDA.
 * @todo Document the functions.
 */

#include <cuda_runtime.h>

#include <cuda/std/complex>

#include "Real.hpp"

namespace Transform {
namespace FourierTransform {

void run_fft_gpu(cuda::std::complex<real>* data, int size, int m, real base,
                 cudaStream_t stream_id = 0);

void run_block_fft_gpu(cuda::std::complex<real>* data, int n, real base,
                       int num_streams = 1, cudaStream_t stream_id = 0);

void bitreverse_gpu(cuda::std::complex<real>* in, cuda::std::complex<real>* out,
                    int size, int s, cudaStream_t stream_id = 0);

void transpose_gpu(cuda::std::complex<real>* in, cuda::std::complex<real>* out,
                   int n, cudaStream_t stream_id = 0);

void swap_row_col_gpu(cuda::std::complex<real>* in,
                      cuda::std::complex<real>* out, const int row,
                      const int col, int n, cudaStream_t stream_id = 0);

}  // namespace FourierTransform
}  // namespace Transform

#endif  // REAL_HPP