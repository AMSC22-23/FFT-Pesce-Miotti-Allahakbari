#ifndef FOURIER_TRANSFORM_HPP
#define FOURIER_TRANSFORM_HPP

#include <complex>
#include <numbers>
#include <vector>

#include "Real.hpp"

// Perform the Fourier Transform of a sequence, using the O(n^2) algorithm.
std::vector<std::complex<real>> DiscreteFourierTransform(
    const std::vector<std::complex<real>> &sequence);

// Perform the Fourier Transform of a sequence, using the recursive O(n log n)
// algorithm. Source:
// https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
std::vector<std::complex<real>> FastFourierTransformRecursive(
    const std::vector<std::complex<real>> &sequence);

// Perform the Fourier Transform of a sequence, using the iterative O(n log n)
// algorithm. Source:
// https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
std::vector<std::complex<real>> FastFourierTransformIterative(
    const std::vector<std::complex<real>> &sequence);

void TimeEstimateFFT(const std::vector<std::complex<real>> &sequence,
                     unsigned int max_num_threads);
#endif  // FOURIER_TRANSFORM_HPP
