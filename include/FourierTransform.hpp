#ifndef FOURIER_TRANSFORM_HPP
#define FOURIER_TRANSFORM_HPP

#include <complex>
#include <vector>
#include <numbers>

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

// Perform the Fourier Transform of a sequence, using the O(n^2) algorithm
std::vector<std::complex<real>> DiscreteFourierTransform(const std::vector<std::complex<real>> &sequence);

// Perform the Fourier Transform of a sequence, using the O(n log n) algorithm
// Note that this particular implementation uses recursion, which was discouraged in the assignment
std::vector<std::complex<real>> FastFourierTransformRecursive(const std::vector<std::complex<real>> &sequence);

// Perform the Fourier Transform of a sequence, using the O(n log n) algorithm
// This is the iterative implementation, taken from Quinn Chapter 15
std::vector<std::complex<real>> FastFourierTransformIterative(const std::vector<std::complex<real>> &sequence);

// Compare the values of "sequence" with those of "sequence_golden" and return if they have the same elements; if "print_errors" is true, print the errors in "sequence"
bool CompareResult(const std::vector<std::complex<real>> &sequence_golden, const std::vector<std::complex<real>> &sequence, double precision, bool print_errors);

#endif //FOURIER_TRANSFORM_HPP
