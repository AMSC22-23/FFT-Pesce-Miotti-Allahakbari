#ifndef FOURIER_TRANSFORM_HPP
#define FOURIER_TRANSFORM_HPP

#include <complex>
#include <vector>
#include <numbers>
#include "Real.hpp"

// Perform the Fourier Transform of a sequence, using the O(n^2) algorithm.
std::vector<std::complex<real>> DiscreteFourierTransform(const std::vector<std::complex<real>> &sequence);

// Perform the Fourier Transform of a sequence, using the recursive O(n log n) algorithm.
// Source: https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
std::vector<std::complex<real>> FastFourierTransformRecursive(const std::vector<std::complex<real>> &sequence);

// Perform the Fourier Transform of a sequence, using the iterative O(n log n) algorithm.
// Source: https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
std::vector<std::complex<real>> FastFourierTransformIterative(const std::vector<std::complex<real>> &sequence);

// Compare the values of "sequence" with those of "sequence_golden" and return true if
// the difference between the two is less than "precision" for all elements.
bool CompareResult(
    const std::vector<std::complex<real>> &sequence_golden,
    const std::vector<std::complex<real>> &sequence, 
    double precision, bool print_errors);

#endif //FOURIER_TRANSFORM_HPP
