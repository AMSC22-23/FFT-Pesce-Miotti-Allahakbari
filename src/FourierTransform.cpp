#include "FourierTransform.hpp"

#include <omp.h>
#include <tgmath.h>

#include <cassert>
#include <chrono>
#include <iostream>

#include "BitReversalPermutation.hpp"
#include "Utility.hpp"

using vec = std::vector<std::complex<real>>;

vec FastFourierTransformIterativeOmegas(const vec &sequence);

void TimeEstimateFFT(const vec &sequence, unsigned int max_num_threads) {
  // Calculate sequence size.
  const size_t size = sequence.size();
  unsigned long serial_fast_time = 0;

  // For each thread number.
  for (unsigned int num_threads = 1; num_threads <= max_num_threads;
       num_threads *= 2) {
    // Set the number of threads.
    omp_set_num_threads(num_threads);

    // Execute the fft.
    auto t0 = std::chrono::high_resolution_clock::now();
    const vec fast_result = FastFourierTransformIterative(sequence);
    auto t1 = std::chrono::high_resolution_clock::now();
    const auto fast_time =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    if (num_threads == 1) serial_fast_time = fast_time;
    std::cout << "Time for parallel FFT with " << size << " elements and "
              << num_threads << " threads: " << fast_time << "μs" << std::endl;

    // Calculate and print speedups.
    std::cout << "Speedup over fast standard: "
              << static_cast<double>(serial_fast_time) / fast_time << "x"
              << std::endl;

    std::cout << std::endl;
  }
}
// Perform the Fourier Transform of a sequence, using the O(n^2) algorithm.
vec DiscreteFourierTransform(const vec &sequence) {
  // Defining some useful aliases.
  constexpr real pi = std::numbers::pi_v<real>;
  const size_t n = sequence.size();

  // Initializing the result vector.
  vec result;
  result.reserve(n);

  // Main loop: looping over result coefficients.
  for (size_t k = 0; k < n; k++) {
    std::complex<real> curr_coefficient = 0.0;

    // Internal loop: looping over input coefficients for a set result position.
    for (size_t m = 0; m < n; m++) {
      const std::complex<real> exponent =
          std::complex<real>{0, -2 * pi * k * m / n};
      curr_coefficient += sequence[m] * std::exp(exponent);
    }

    result.emplace_back(curr_coefficient);
  }

  return result;
}

// Perform the Fourier Transform of a sequence, using the O(n log n) algorithm.
// Source: https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
vec FastFourierTransformRecursive(const vec &sequence) {
  // Defining some useful aliases.
  constexpr real pi = std::numbers::pi_v<real>;
  const size_t n = sequence.size();

  // Trivial case: if the sequence is of length 1, return it.
  if (n == 1) return sequence;

  // Splitting the sequence into two halves.
  vec even_sequence;
  vec odd_sequence;

  for (size_t i = 0; i < n; i++) {
    if (i % 2 == 0)
      even_sequence.emplace_back(sequence[i]);
    else
      odd_sequence.emplace_back(sequence[i]);
  }

  // Recursively computing the Fourier Transform of the two halves.
  vec even_result = FastFourierTransformRecursive(even_sequence);
  vec odd_result = FastFourierTransformRecursive(odd_sequence);

  // Combining the two results.
  vec result(n, 0.0);

  // Implementing the Cooley-Tukey algorithm.
  for (size_t k = 0; k < n / 2; k++) {
    std::complex<real> p = even_result[k];
    std::complex<real> q =
        std::exp(std::complex<real>{0, -2 * pi * k / n}) * odd_result[k];

    result[k] = p + q;
    result[k + n / 2] = p - q;
  }

  return result;
}

// Perform the Fourier Transform of a sequence, using the iterative O(n log n)
// algorithm. Source:
// https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
vec FastFourierTransformIterative(const vec &sequence) {
  // Defining some useful aliases.
  constexpr real pi = std::numbers::pi_v<real>;
  const size_t n = sequence.size();

  // Check that the size of the sequence is a power of 2.
  const size_t log_n = static_cast<size_t>(log2(n));
  assert(1UL << log_n == n);

  // Initialization of output sequence.
  vec result = BitReversalPermutation(sequence);

  // Main loop: looping over the binary tree layers.
  for (size_t s = 1; s <= log_n; s++) {
    const size_t m = 1UL << s;
    const size_t half_m = m >> 1UL;

    const std::complex<real> omega_d =
        std::exp(std::complex<real>{0, -pi / half_m});

#pragma omp parallel for default(none) firstprivate(m, half_m, n, omega_d) \
    shared(result) schedule(static)
    for (size_t k = 0; k < n; k += m) {
      std::complex<real> omega(1, 0);
      for (size_t j = 0; j < half_m; j++) {
        const size_t k_plus_j = k + j;
        const std::complex<real> t = omega * result[k_plus_j + half_m];
        const std::complex<real> u = result[k_plus_j];
        result[k_plus_j] = u + t;
        result[k_plus_j + half_m] = u - t;
      }
      omega *= omega_d;
    }
  }

  return result;
}

// A version of FastFourierTransformIterative that allows for fusion of the two
// inner looops. Experimental.
vec FastFourierTransformIterativeOmegas(const vec &sequence) {
  // Defining some useful aliases.
  constexpr real pi = std::numbers::pi_v<real>;
  const size_t n = sequence.size();
  const size_t half_n = n >> 1;
  const unsigned int num_threads = omp_get_num_threads();

  // Check that the size of the sequence is a power of 2.
  const size_t log_n = static_cast<size_t>(log2(n));
  assert(1UL << log_n == n);

  // Initialization of output sequence.
  vec result = BitReversalPermutation(sequence);

  // Creation of a support vector to store values of omega.
  vec omegas(half_n, 0);

  // Main loop: looping over the binary tree layers.
  for (size_t s = 1; s <= log_n; s++) {
    const size_t m = 1UL << s;
    const size_t half_m = m >> 1UL;

    const std::complex<real> omega_d =
        std::exp(std::complex<real>{0, -pi / half_m});

#pragma omp parallel for default(none) shared(omegas) \
    firstprivate(num_threads, half_m, pi, omega_d)
    for (unsigned int thread = 0; thread < num_threads; thread++) {
      const size_t iterations = half_m / num_threads;
      const size_t base_index = iterations * thread;
      omegas[base_index] =
          std::exp(std::complex<real>{0, -base_index * pi / half_m});
      for (size_t i = base_index + 1; i < base_index + iterations; i++) {
        omegas[i] = omegas[i - 1] * omega_d;
      }
    }

#pragma omp parallel for default(none) firstprivate(m, half_m, n) \
    shared(result, omegas) schedule(static) collapse(2)
    for (size_t k = 0; k < n; k += m) {
      for (size_t j = 0; j < half_m; j++) {
        const size_t k_plus_j = k + j;
        const std::complex<real> t = omegas[j] * result[k_plus_j + half_m];
        const std::complex<real> u = result[k_plus_j];
        result[k_plus_j] = u + t;
        result[k_plus_j + half_m] = u - t;
      }
    }
  }

  return result;
}

// A version of FastFourierTransformIterativeOmegas that manually fuses the two
// inner looops. Experimental.
vec FastFourierTransformIterativeCollapsed(const vec &sequence) {
  // Defining some useful aliases.
  constexpr real pi = std::numbers::pi_v<real>;
  const size_t n = sequence.size();
  const size_t half_n = n >> 1;
  const unsigned int num_threads = omp_get_num_threads();

  // Check that the size of the sequence is a power of 2.
  const size_t log_n = static_cast<size_t>(log2(n));
  assert(1UL << log_n == n);

  // Initialization of output sequence.
  vec result = BitReversalPermutation(sequence);

  // Creation of a support vector to store values of omega.
  vec omegas(half_n, 0);

  // Main loop: looping over the binary tree layers.
  for (size_t s = 1; s <= log_n; s++) {
    const size_t m = 1UL << s;
    const size_t half_m = m >> 1UL;
    const size_t s_minus_1 = s - 1UL;

    const std::complex<real> omega_d =
        std::exp(std::complex<real>{0, -pi / half_m});

#pragma omp parallel for default(none) shared(omegas) \
    firstprivate(num_threads, half_m, pi, omega_d)
    for (unsigned int thread = 0; thread < num_threads; thread++) {
      const size_t iterations = half_m / num_threads;
      const size_t base_index = iterations * thread;
      omegas[base_index] =
          std::exp(std::complex<real>{0, -base_index * pi / half_m});
      for (size_t i = base_index + 1; i < base_index + iterations; i++) {
        omegas[i] = omegas[i - 1] * omega_d;
      }
    }

#pragma omp parallel for default(none)                                   \
    firstprivate(half_m, half_n, s_minus_1, s, m) shared(result, omegas) \
    schedule(static)
    for (size_t index = 0; index < half_n; index++) {
      const size_t k = (index >> s_minus_1) << s;
      const size_t j = index % half_m;
      const size_t k_plus_j = k + j;
      const std::complex<real> t = omegas[j] * result[k_plus_j + half_m];
      const std::complex<real> u = result[k_plus_j];
      result[k_plus_j] = u + t;
      result[k_plus_j + half_m] = u - t;
    }
  }

  return result;
}
