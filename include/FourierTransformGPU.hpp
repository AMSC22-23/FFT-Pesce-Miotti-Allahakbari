#ifndef REAL_HPP_GPU
#define REAL_HPP_GPU

#include <cuda_runtime.h>

#include <cuda/std/complex>

// A simple way to make the code generic with respect to the real type.

namespace FourierTransform {

#ifdef FLOAT
using real = float;
#else
#ifdef DOUBLE
using real = double;
#else
#ifdef LONG_DOUBLE
using real = long double;
#else
using real = double;
#endif
#endif
#endif

void run_fft_gpu(cuda::std::complex<FourierTransform::real> *data, int size,
                 int m, FourierTransform::real base);

}  // namespace FourierTransform

#endif  // REAL_HPP