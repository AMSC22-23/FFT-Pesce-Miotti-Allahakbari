#include <cuda_runtime.h>

#include "FFTGPU.hpp"

// The algorithm only works if n >= TILE_SIZE
#define TILE_SIZE 8

#define BLOCK_SIZE 8

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

__global__ void fft1d(cuda::std::complex<real>* data, int n, int m, real base) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int half_m = m >> 1;
  size_t k = (tid / half_m) * m;
  size_t j = tid % half_m;
  const size_t k_plus_j = k + j;

  if ((k_plus_j + half_m) < n) {
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

__global__ void block_fft(cuda::std::complex<real>* data, int n, int log_n_blk,
                          real base) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  __shared__ cuda::std::complex<real> tile[TILE_SIZE][TILE_SIZE];

  int blockX = threadIdx.x;
  int blockY = threadIdx.y;

  tile[blockY][__brev(blockX) >> (32 - log_n_blk)] = data[y * n + x];

  __syncthreads();

  for (size_t s = 1; s <= log_n_blk; s++) {
    const size_t m = 1UL << s;
    int half_m = m >> 1;
    size_t k = (threadIdx.x / half_m) * m;
    size_t j = threadIdx.x % half_m;
    const size_t k_plus_j = k + j;

    // printf("k + j+ half_m: %d\n", k_plus_j + half_m);
    if ((k_plus_j + half_m) < TILE_SIZE) {
      const cuda::std::complex<real> omega =
          cuda::std::exp(cuda::std::complex<real>{0, base / half_m * j});
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

  cuda::std::complex<real> temp = tile[blockY][blockX];
  tile[blockY][blockX] = tile[blockX][blockY];
  tile[blockX][blockY] = temp;

  __syncthreads();

  temp = tile[blockY][__brev(blockX) >> (32 - log_n_blk)];
  tile[blockY][__brev(blockX) >> (32 - log_n_blk)] = tile[blockY][blockX];
  tile[blockY][blockX] = temp;

  __syncthreads();

  for (size_t s = 1; s <= log_n_blk; s++) {
    const size_t m = 1UL << s;
    int half_m = m >> 1;
    size_t k = (threadIdx.x / half_m) * m;
    size_t j = threadIdx.x % half_m;
    const size_t k_plus_j = k + j;

    if ((k_plus_j + half_m) < TILE_SIZE) {
      const cuda::std::complex<real> omega =
          cuda::std::exp(cuda::std::complex<real>{0, base / half_m * j});
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

  data[y * n + x] = tile[blockX][blockY];
}

void run_fft_gpu(cuda::std::complex<real>* data, int n, int m, real base,
                 cudaStream_t stream_id) {
  int block_dim = TILE_SIZE;
  int grid_dim = (n + TILE_SIZE - 1) / (TILE_SIZE);
  fft1d<<<grid_dim, block_dim, 0, stream_id>>>(data, n, m, base);
}

void run_block_fft_gpu(cuda::std::complex<real>* data, int n, real base,
                       cudaStream_t stream_id) {
  dim3 block_dim(TILE_SIZE, TILE_SIZE);
  dim3 grid_dim((n + TILE_SIZE - 1) / (TILE_SIZE),
                (n + TILE_SIZE - 1) / TILE_SIZE);
  block_fft<<<grid_dim, block_dim, 0, stream_id>>>(data, n, log2(TILE_SIZE),
                                                   base);

  cudaError_t kernelError = cudaGetLastError();
  if (kernelError != cudaSuccess) {
    printf("Kernel launch failed with error: %s\n",
           cudaGetErrorString(kernelError));
    // Handle the error accordingly
  }
}

void bitreverse_gpu(cuda::std::complex<real>* in, cuda::std::complex<real>* out,
                    int n, int s, cudaStream_t stream_id) {
  int block_dim = TILE_SIZE;
  int grid_dim = (n + TILE_SIZE - 1) / (TILE_SIZE);

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
