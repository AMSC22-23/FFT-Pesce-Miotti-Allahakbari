#include <cuda_runtime.h>

#include "FFTGPU.hpp"

#define TILE_SIZE 1024

using namespace FourierTransform;

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

__global__ void fft(cuda::std::complex<real>* data, int size, int m,
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
    cuda::std::complex<FourierTransform::real>* data, int size, int m,
    FourierTransform::real base) {
  int block_dim = TILE_SIZE;
  int grid_dim = (size + TILE_SIZE) / (TILE_SIZE * 2);
  fft<<<grid_dim, block_dim>>>(data, size, m, base);
}

void FourierTransform::bitreverse_gpu(cuda::std::complex<real>* in,
                                      cuda::std::complex<real>* out, int size,
                                      int s) {
  int block_dim = TILE_SIZE;
  int grid_dim = size / TILE_SIZE;

  bitrev_reorder<<<grid_dim, block_dim>>>(in, out, s);
}