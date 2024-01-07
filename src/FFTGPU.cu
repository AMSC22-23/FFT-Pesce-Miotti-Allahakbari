#include <cuda_runtime.h>

#include "FFTGPU.hpp"

// The algorithm only works if n >= TILE_SIZE
#define TILE_SIZE 64

#define BLOCK_SIZE 32

namespace Transform {
namespace FourierTransform {

// CUDA kernel for transposing a 2D array efficiently using shared memory
__global__ void swap_row_col(cuda::std::complex<real>* input,
                             cuda::std::complex<real>* output, const int row,
                             const int col, const int n) {
  // Use shared memory to reduce global memory transactions
  __shared__ cuda::std::complex<real> tile[TILE_SIZE];
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x < n) {
    int in_index = row * n + x;

    // Load data from global memory to shared memory
    tile[threadIdx.x] = input[in_index];

    __syncthreads();

    // Write transposed data to the output

    int out_index = x * n + col;
    output[out_index] = tile[threadIdx.x];
  }
}

// CUDA kernel for transposing a 2D array efficiently using shared memory
__global__ void transpose(cuda::std::complex<real>* input,
                          cuda::std::complex<real>* output, const int n) {
  // Use shared memory to reduce global memory transactions
  __shared__ cuda::std::complex<real> tile[BLOCK_SIZE]
                                          [BLOCK_SIZE + 1];  // +1 for padding
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < n && y < n) {
    int in_index = y * n + x;

    // Load data from global memory to shared memory
    tile[threadIdx.y][threadIdx.x] = input[in_index];

    __syncthreads();

    // Write transposed data to the output
    x = blockIdx.y * blockDim.x + threadIdx.x;
    y = blockIdx.x * blockDim.y + threadIdx.y;

    int out_index = y * n + x;
    output[out_index] = tile[threadIdx.x][threadIdx.y];
  }
}

/**
 * Reorders array by bit-reversing the indexes.
 */
__global__ void bitrev_reorder(cuda::std::complex<real>* in,
                               cuda::std::complex<real>* out, int s) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if constexpr (sizeof(size_t) == 4)
    out[__brev(id) >> (32 - s)] = in[id];
  else
    out[__brevll(id) >> (64 - s)] = in[id];
}

__global__ void fft1d(cuda::std::complex<real>* data, int size, int m,
                      real base) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int half_m = m >> 1;
  size_t k = (tid / half_m) * m;
  size_t j = tid % half_m;
  const size_t k_plus_j = k + j;

  if ((k_plus_j + half_m) < size) {
    const cuda::std::complex<real> omega =
        cuda::std::exp(cuda::std::complex<real>{0, base / half_m * j});
    const cuda::std::complex<real> t = omega * data[k_plus_j + half_m];
    const cuda::std::complex<real> u = data[k_plus_j];
    const cuda::std::complex<real> even = u + t;
    const cuda::std::complex<real> odd = u - t;
    data[k_plus_j] = even;
    data[k_plus_j + half_m] = odd;
  }
}

void run_fft_gpu(cuda::std::complex<real>* data, int n, int m, real base,
                 cudaStream_t stream_id) {
  int block_dim = TILE_SIZE;
  int grid_dim = (n + TILE_SIZE - 1) / (TILE_SIZE);
  fft1d<<<grid_dim, block_dim, 0, stream_id>>>(data, n, m, base);
}

void bitreverse_gpu(cuda::std::complex<real>* in, cuda::std::complex<real>* out,
                    int size, int s, cudaStream_t stream_id) {
  int block_dim = TILE_SIZE;
  int grid_dim = (size + TILE_SIZE - 1) / (TILE_SIZE);

  bitrev_reorder<<<grid_dim, block_dim, 0, stream_id>>>(in, out, s);
}

void transpose_gpu(cuda::std::complex<real>* in, cuda::std::complex<real>* out,
                   int n, cudaStream_t stream_id) {
  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_dim((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

  transpose<<<grid_dim, block_dim, 0, stream_id>>>(in, out, n);
}

void swap_row_col_gpu(cuda::std::complex<real>* in,
                      cuda::std::complex<real>* out, const int row,
                      const int col, int n, cudaStream_t stream_id) {
  int block_dim = TILE_SIZE;
  int grid_dim = (n + TILE_SIZE - 1) / TILE_SIZE;

  swap_row_col<<<grid_dim, block_dim, 0, stream_id>>>(in, out, row, col, n);
}
}  // namespace FourierTransform
}  // namespace Transform
