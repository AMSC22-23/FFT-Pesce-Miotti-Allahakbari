#ifndef INVERSE_FOURIER_TRANSFORM_HPP
#define INVERSE_FOURIER_TRANSFORM_HPP

#include <complex>
#include <vector>
#include <numbers>

#include "Real.hpp"

// Perform the Inverse Fourier Transform of a sequence, using the O(n^2) algorithm.
std::vector<std::complex<real>> InverseDiscreteFourierTransform(const std::vector<std::complex<real>> &sequence);

// Perform the Inverse Fourier Transform of a sequence, using the iterative O(n log n) algorithm.
// This function is nearly identical to its direct counterpart, except for the sign of the exponent.
// Source: https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
std::vector<std::complex<real>> InverseFastFourierTransformIterative(const std::vector<std::complex<real>> &sequence);

#endif // INVERSE_FOURIER_TRANSFORM_HPP
