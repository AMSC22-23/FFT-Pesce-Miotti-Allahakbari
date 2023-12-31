#include <cuda_runtime.h>

#include "FourierTransformGPU.hpp"

using namespace FourierTransform;
__global__ void fft(cuda::std::complex<real> *data, int size, int m,
                    real base) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int half_m = m >> 1;
  size_t k = (tid / half_m) * m;
  size_t j = tid % half_m;

  const cuda::std::complex<real> omega =
      cuda::std::exp(cuda::std::complex<real>{0, base / half_m * j});
  const size_t k_plus_j = k + j;
  const cuda::std::complex<real> t = omega * data[k_plus_j + half_m];
  const cuda::std::complex<real> u = data[k_plus_j];
  const cuda::std::complex<real> even = u + t;
  const cuda::std::complex<real> odd = u - t;
  data[k_plus_j] = even;
  data[k_plus_j + half_m] = odd;
}

void FourierTransform::run_fft_gpu(
    cuda::std::complex<FourierTransform::real> *data, int size, int m,
    FourierTransform::real base) {
  int block_dim = 1024;
  int grid_dim = size / (block_dim * 2);
  fft<<<grid_dim, block_dim>>>(data, size, m, base);
}