#ifndef INVERSE_FOURIER_TRANSFORM_HPP
#define INVERSE_TRANSFORM_HPP

#include <complex>
#include <vector>
#include <numbers>

#include "Real.hpp"

// Perform the Inverse Fourier Transform of a sequence, using the O(n^2) algorithm.
std::vector<std::complex<real>> InverseDiscreteFourierTransform(const std::vector<std::complex<real>> &sequence);

#endif // INVERSE_FOURIER_TRANSFORM_HPP
