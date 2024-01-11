#include "FFTGPU.hpp"

/**
 * @file FFTGPU.cu.
 * @brief Defines the kernels and wrappers declared in FFTGPU.hpp.
 */

#include <cuda_runtime.h>

#define TILE_SIZE 32

#define BLOCK_SIZE 8

namespace Transform {
namespace FourierTransform {

// A kernel to swap a row and a column of a 2D sequence.
__global__ void swap_row_col_kernel(const cuda::std::complex<real>* input,
                                    cuda::std::complex<real>* output,
                                    const size_t row, const size_t col,
                                    const size_t n) {
  // Use shared memory to reduce global memory transactions.
  __shared__ cuda::std::complex<real> tile[TILE_SIZE];

  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x < n) {
    const size_t in_index = row * n + x;

    // Load data from global memory to shared memory.
    tile[threadIdx.x] = input[in_index];

    // Synchronize threads in the block.
    __syncthreads();

    // Write transposed data to the output.
    const size_t out_index = x * n + col;
    output[out_index] = tile[threadIdx.x];
  }
}

// A kernel to perform the bit reversal permutation of a sequence.
__global__ void bitreverse_kernel(const cuda::std::complex<real>* in,
                                  cuda::std::complex<real>* out,
                                  const size_t bitsize) {
  const size_t id = blockIdx.x * blockDim.x + threadIdx.x;

  // Based on whether the indices are 32 or 64 bit long, use a different bit
  // reversal function.
  if constexpr (sizeof(size_t) == 4)
    out[__brev(id) >> (32 - bitsize)] = in[id];
  else
    out[__brevll(id) >> (64 - bitsize)] = in[id];
}

// A kernel to perform one layer of the 1D FFT of a sequence, assuming data has
// already been reordered.
__global__ void fft1d_kernel(cuda::std::complex<real>* data, const size_t n,
                             const size_t m, const real base_angle) {
  // Set aliases for a few quantities.
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t half_m = m >> 1;
  const size_t k = (tid / half_m) * m;
  const size_t j = tid % half_m;
  const size_t k_plus_j = k + j;

  // Perform on layer of the FFT.
  if ((k_plus_j + half_m) < n) {
    const cuda::std::complex<real> omega =
        cuda::std::exp(cuda::std::complex<real>{0, base_angle / half_m * j});
    const cuda::std::complex<real> t = omega * data[k_plus_j + half_m];
    const cuda::std::complex<real> u = data[k_plus_j];
    const cuda::std::complex<real> even = u + t;
    const cuda::std::complex<real> odd = u - t;
    data[k_plus_j] = even;
    data[k_plus_j + half_m] = odd;
  }
}

// A wrapper around the kernel to swap a row and a column a sequence.
void run_swap_row_col_gpu(const cuda::std::complex<real>* in,
                          cuda::std::complex<real>* out, const size_t row,
                          const size_t col, size_t n, cudaStream_t stream_id) {
  // Create CUDA block and grid dimensions.
  constexpr unsigned int block_dim = TILE_SIZE;
  const unsigned int grid_dim = (n + TILE_SIZE - 1) / TILE_SIZE;

  // Launch the kernel.
  swap_row_col_kernel<<<grid_dim, block_dim, 0, stream_id>>>(in, out, row, col,
                                                             n);

  // Check for errors.
  cudaError_t kernelError = cudaGetLastError();
  if (kernelError != cudaSuccess) {
    printf("Kernel launch failed with error: %s\n",
           cudaGetErrorString(kernelError));
  }
}

// A wrapper around the kernel to perform the bit reversal of a sequence.
void run_bitreverse_gpu(const cuda::std::complex<real>* in,
                        cuda::std::complex<real>* out, size_t n, size_t bitsize,
                        cudaStream_t stream_id) {
  // Create CUDA block and grid dimensions.
  constexpr unsigned int block_dim = TILE_SIZE;
  const unsigned int grid_dim = (n + TILE_SIZE - 1) / (TILE_SIZE);

  // Launch the kernel.
  bitreverse_kernel<<<grid_dim, block_dim, 0, stream_id>>>(in, out, bitsize);

  // Check for errors.
  cudaError_t kernelError = cudaGetLastError();
  if (kernelError != cudaSuccess) {
    printf("Kernel launch failed with error: %s on line: %d\n",
           cudaGetErrorString(kernelError), __LINE__);
  }
}

// A wrapper around the kernel to perform one layer of a 1D FFT.
void run_fft1d_gpu(cuda::std::complex<real>* data, size_t n, size_t m,
                   real base_angle, const cudaStream_t stream_id) {
  // Create CUDA block and grid dimensions.
  constexpr unsigned int block_dim = TILE_SIZE;
  const unsigned int grid_dim = (n + TILE_SIZE - 1) / (TILE_SIZE);

  // Launch the kernel.
  fft1d_kernel<<<grid_dim, block_dim, 0, stream_id>>>(data, n, m, base_angle);

  // Check for errors.
  cudaError_t kernelError = cudaGetLastError();
  if (kernelError != cudaSuccess) {
    printf("Kernel launch failed with error: %s on line: %d\n",
           cudaGetErrorString(kernelError), __LINE__);
  }
}

// A kernel for 8x8 2D FFT.
__global__ void block_fft_kernel(cuda::std::complex<real>* data, const size_t n,
                                 const size_t log_n_blk) {
  // Set an alias for the angle.
  constexpr real base_angle = -M_PI;

  // Set aliases for indices.
  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t blockX = threadIdx.x;
  const size_t blockY = threadIdx.y;

  // Declare a shared memory matrix.
  __shared__ cuda::std::complex<real> tile[BLOCK_SIZE][BLOCK_SIZE];

  // Perform bit reversal permutation of the input into the shared matrix.
  tile[blockY][__brev(blockX) >> (32 - log_n_blk)] = data[y * n + x];
  __syncthreads();

  // Perform the FFT, layer by layer
  for (size_t s = 1; s <= log_n_blk; s++) {
    const size_t m = 1UL << s;
    const size_t half_m = m >> 1;
    const size_t k = (threadIdx.x / half_m) * m;
    const size_t j = threadIdx.x % half_m;
    const size_t k_plus_j = k + j;

    if ((k_plus_j + half_m) < BLOCK_SIZE) {
      const cuda::std::complex<real> omega =
          cuda::std::exp(cuda::std::complex<real>{0, base_angle / half_m * j});
      const cuda::std::complex<real> t =
          omega * tile[blockY][k_plus_j + half_m];
      const cuda::std::complex<real> u = tile[blockY][k_plus_j];
      const cuda::std::complex<real> even = u + t;
      const cuda::std::complex<real> odd = u - t;

      tile[blockY][k_plus_j] = even;
      tile[blockY][k_plus_j + half_m] = odd;
    }
    __syncthreads();
  }

  // Copy data back to the main memory and transpose it.
  data[y * n + x] = tile[blockX][blockY];
}

// A kernel for 8x8 2D inverse FFT.
__global__ void block_inverse_fft_kernel(cuda::std::complex<real>* data,
                                         const size_t n,
                                         const size_t log_n_blk) {
  // Set an alias for the angle.
  constexpr real base_angle = M_PI;

  // Set aliases for indices.
  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t blockX = threadIdx.x;
  const size_t blockY = threadIdx.y;

  // Declare a shared memory matrix.
  __shared__ cuda::std::complex<real> tile[BLOCK_SIZE][BLOCK_SIZE];

  // Perform bit reversal permutation of the input into the shared matrix and
  // transpose it.
  tile[blockY][__brev(blockX) >> (32 - log_n_blk)] = data[y * n + x];
  __syncthreads();

  // Perform the FFT, layer by layer
  for (size_t s = 1; s <= log_n_blk; s++) {
    const size_t m = 1UL << s;
    const size_t half_m = m >> 1;
    const size_t k = (threadIdx.x / half_m) * m;
    const size_t j = threadIdx.x % half_m;
    const size_t k_plus_j = k + j;

    if ((k_plus_j + half_m) < BLOCK_SIZE) {
      const cuda::std::complex<real> omega =
          cuda::std::exp(cuda::std::complex<real>{0, base_angle / half_m * j});
      const cuda::std::complex<real> t =
          omega * tile[blockY][k_plus_j + half_m];
      const cuda::std::complex<real> u = tile[blockY][k_plus_j];
      const cuda::std::complex<real> even = u + t;
      const cuda::std::complex<real> odd = u - t;

      tile[blockY][k_plus_j] = even;
      tile[blockY][k_plus_j + half_m] = odd;
    }
    __syncthreads();
  }

  // Copy data back to the main memory.
  data[y * n + x] =
      tile[blockX][blockY] / cuda::std::complex<real>(BLOCK_SIZE, 0);
}

// A wrapper around the kernel for 8x8 2D FFT.
void run_block_fft_gpu(cuda::std::complex<real>* data, size_t n,
                       size_t num_streams, cudaStream_t stream_id) {
  // Create CUDA block and grid dimensions.
  constexpr dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
  const dim3 grid_dim((n + BLOCK_SIZE - 1) / (BLOCK_SIZE),
                      (n / num_streams + BLOCK_SIZE - 1) / (BLOCK_SIZE));

  // Launch the kernel.
  block_fft_kernel<<<grid_dim, block_dim, 0, stream_id>>>(data, n,
                                                          log2(BLOCK_SIZE));

  // Check for errors.
  cudaError_t kernelError = cudaGetLastError();
  if (kernelError != cudaSuccess) {
    printf("Kernel launch failed with error: %s on line: %d\n",
           cudaGetErrorString(kernelError), __LINE__);
  }
}

// A wrapper around the kernel for 8x8 2D inverse FFT.
void run_block_inverse_fft_gpu(cuda::std::complex<real>* data, size_t n,
                               size_t num_streams, cudaStream_t stream_id) {
  // Create CUDA block and grid dimensions.
  constexpr dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
  const dim3 grid_dim((n + BLOCK_SIZE - 1) / (BLOCK_SIZE),
                      (n / num_streams + BLOCK_SIZE - 1) / (BLOCK_SIZE));

  // Launch the kernel.
  block_inverse_fft_kernel<<<grid_dim, block_dim, 0, stream_id>>>(
      data, n, log2(BLOCK_SIZE));

  // Check for errors.
  cudaError_t kernelError = cudaGetLastError();
  if (kernelError != cudaSuccess) {
    printf("Kernel launch failed with error: %s on line: %d\n",
           cudaGetErrorString(kernelError), __LINE__);
  }
}

}  // namespace FourierTransform
}  // namespace Transform
