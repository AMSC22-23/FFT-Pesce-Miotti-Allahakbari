#ifndef REAL_HPP_GPU
#define REAL_HPP_GPU

/**
 * @file FFTGPU.hpp.
 * @brief Declares functions to execute Fourier Transforms on a GPU using CUDA.
 */

#include <cuda/std/complex>

#include "Real.hpp"

namespace Transform {
namespace FourierTransform {

/**
 * @brief Execute one layer of a 1D FFT.
 *
 * The function is a wrapper around a CUDA kernel that performs a layer in the
 * tree of an iterative 1D FFT.
 *
 * @param data Device pointer to the sequence with both input and output of the
 * FFT.
 * @param m Number of elements in data.
 * @param num_streams Number of CUDA streams for execution.
 * @param stream_id ID of the first CUDA stream to execute in.
 *
 * @see run_block_inverse_fft_gpu
 */
void run_fft1d_gpu(cuda::std::complex<real>* data, size_t size, size_t m,
                   real base_angle, cudaStream_t stream_id = 0);

/**
 * @brief Execute the bit reversal permutation of a sequence.
 *
 * The function is a wrapper around a CUDA kernel that performs the bit reversal
 * permutation of a sequence.
 *
 * @param in Device pointer to the input sequence.
 * @param out Device pointer to the output sequence.
 * @param size Size of the input and output sequences.
 * @param bitsize Number of bits in the representation of the index to reverse.
 * @param stream_id ID of the CUDA to execute in.
 *
 * @see run_block_inverse_fft_gpu
 */
void run_bitreverse_gpu(const cuda::std::complex<real>* in,
                        cuda::std::complex<real>* out, size_t size,
                        size_t bitsize, cudaStream_t stream_id = 0);

/**
 * @brief Swap a row and a column in a sequence.
 *
 * The function is a wrapper around a CUDA kernel that swaps the row and column
 * at the given indices.
 *
 * @param in Device pointer to the input sequence.
 * @param out Device pointer to the output sequence.
 * @param row Row index of the row to swap.
 * @param col column index of the row to swap.
 * @param n Size of the input and output sequences.
 * @param stream_id ID of the CUDA to execute in.
 */
void run_swap_row_col_gpu(const cuda::std::complex<real>* in,
                          cuda::std::complex<real>* out, const size_t row,
                          const size_t col, size_t n,
                          cudaStream_t stream_id = 0);

/**
 * @brief A 2D FFT algorithm on 8x8 blocks, using CUDA.
 *
 * The function is a wrapper around a CUDA kernel and performs a 2D FFT on 8x8
 * blocks of a subset of an image.
 *
 * @param data Device pointer to the sequence with both input and output of the
 * FFT.
 * @param n Number of elements in data.
 * @param num_streams Number of CUDA streams for execution.
 * @param stream_id ID of the CUDA stream to execute in.
 *
 * @see run_block_inverse_fft_gpu
 */
void run_block_fft_gpu(cuda::std::complex<real>* data, size_t n,
                       size_t num_streams = 1, cudaStream_t stream_id = 0);

/**
 * @brief A 2D IFFT algorithm on 8x8 blocks, using CUDA.
 *
 * The function is a wrapper around a CUDA kernel and performs a 2D IFFT on 8x8
 * blocks of a subset of an image. It is the inverse of run_block_fft_gpu.
 *
 * @param data Device pointer to the sequence with both input and output of the
 * IFFT.
 * @param n Number of elements in data.
 * @param num_streams Number of CUDA streams for execution.
 * @param stream_id ID of the first CUDA stream to execute in.
 *
 * @see run_block_fft_gpu
 */
void run_block_inverse_fft_gpu(cuda::std::complex<real>* data, size_t n,
                               size_t num_streams = 1,
                               cudaStream_t stream_id = 0);
}  // namespace FourierTransform
}  // namespace Transform

#endif  // REAL_HPP