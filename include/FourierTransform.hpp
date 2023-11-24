#ifndef FOURIER_TRANSFORM_HPP
#define FOURIER_TRANSFORM_HPP

#include <complex>
#include <vector>
#include <numbers>
#include "Real.hpp"

using vec = std::vector<std::complex<real>>;

// Perform the Fourier Transform of a sequence, using the O(n^2) algorithm.
vec DiscreteFourierTransform(const vec &sequence);

// Perform the Fourier Transform of a sequence, using the recursive O(n log n) algorithm.
// Source: https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
vec FastFourierTransformRecursive(const vec &sequence);

// Perform the Fourier Transform of a sequence, using the iterative O(n log n) algorithm.
// Source: Quinn, Chapter 15.
vec FastFourierTransformIterative(const vec &sequence);

// Compare the values of "sequence" with those of "sequence_golden" and return true if
// the difference between the two is less than "precision" for all elements.
bool CompareResult(const vec &sequence_golden, const vec &sequence, double precision, bool print_errors);

#endif //FOURIER_TRANSFORM_HPP
