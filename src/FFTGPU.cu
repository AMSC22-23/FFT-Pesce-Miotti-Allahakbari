#include <cuda_runtime.h>

#include "FFTGPU.hpp"

#define TILE_SIZE 32

#define TILE_DIM 32
#define BLOCK_ROWS 8

using namespace FourierTransform;

// CUDA kernel for transposing a 2D array efficiently using shared memory
__global__ void transpose(const cuda::std::complex<real>* idata,
                          cuda::std::complex<real>* odata) {
  __shared__ cuda::std::complex<real> tile[TILE_DIM * (TILE_DIM + 1)];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    tile[(threadIdx.y + j) * TILE_DIM + threadIdx.x] =
        idata[(y + j) * width + x];

  __syncthreads();

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    odata[(y + j) * width + x] =
        tile[(threadIdx.y + j) * TILE_DIM + threadIdx.x];
}

/**
 * Reorders array by bit-reversing the indexes.
 */
__global__ void bitrev_reorder(cuda::std::complex<real>* in,
                               cuda::std::complex<real>* out, int s) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (sizeof(size_t) == 4)
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

  const cuda::std::complex<real> omega =
      cuda::std::exp(cuda::std::complex<real>{0, base / half_m * j});
  const cuda::std::complex<real> t = omega * data[k_plus_j + half_m];
  const cuda::std::complex<real> u = data[k_plus_j];
  const cuda::std::complex<real> even = u + t;
  const cuda::std::complex<real> odd = u - t;
  data[k_plus_j] = even;
  data[k_plus_j + half_m] = odd;
}

void FourierTransform::run_fft_gpu(
    cuda::std::complex<FourierTransform::real>* data, int n, int m,
    FourierTransform::real base, cudaStream_t stream_id) {
  int block_dim = TILE_SIZE;
  int grid_dim = (n + TILE_SIZE) / (TILE_SIZE * 2);
  fft1d<<<grid_dim, block_dim, 0, stream_id>>>(data, n, m, base);
}

void FourierTransform::bitreverse_gpu(cuda::std::complex<real>* in,
                                      cuda::std::complex<real>* out, int size,
                                      int s, cudaStream_t stream_id) {
  int block_dim = TILE_SIZE;
  int grid_dim = size / TILE_SIZE;

  bitrev_reorder<<<grid_dim, block_dim, 0, stream_id>>>(in, out, s);
}

void FourierTransform::transpose_gpu(cuda::std::complex<real>* in,
                                     cuda::std::complex<real>* out, int n,
                                     cudaStream_t stream_id) {
  dim3 block_dim(TILE_DIM, BLOCK_ROWS);
  dim3 grid_dim(n / TILE_DIM, n / BLOCK_ROWS);

  transpose<<<grid_dim, block_dim, 0, stream_id>>>(in, out);
}