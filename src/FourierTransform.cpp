#include "FourierTransform.hpp"

#include <cuda_runtime.h>
#include <omp.h>
#include <tgmath.h>

#include <cassert>
#include <chrono>
#include <iostream>

#include "FFTGPU.hpp"
#include "FourierTransform.hpp"
#include "Utility.hpp"

namespace FourierTransform {

void ClassicalFourierTransformAlgorithm::operator()(
    const vec &input_sequence, vec &output_sequence) const {
  // Getting the input size.
  const size_t n = input_sequence.size();

  // Main loop: looping over result coefficients.
  for (size_t k = 0; k < n; k++) {
    std::complex<real> curr_coefficient = 0.0;

    // Internal loop: looping over input coefficients for a set result position.
    for (size_t m = 0; m < n; m++) {
      const std::complex<real> exponent =
          std::complex<real>{0, 2 * base_angle * k * m / n};
      curr_coefficient += input_sequence[m] * std::exp(exponent);
    }

    output_sequence[k] = curr_coefficient;
  }
}

void RecursiveFourierTransformAlgorithm::operator()(
    const vec &input_sequence, vec &output_sequence) const {
  // Getting the input size.
  const size_t n = input_sequence.size();

  // Trivial case: if the input sequence is of length 1, copy it.
  if (n == 1) {
    output_sequence[0] = input_sequence[0];
    return;
  }

  // Splitting the sequence into two halves.
  vec even_sequence;
  vec odd_sequence;

  for (size_t i = 0; i < n; i++) {
    if (i % 2 == 0)
      even_sequence.emplace_back(input_sequence[i]);
    else
      odd_sequence.emplace_back(input_sequence[i]);
  }

  // Recursively computing the Fourier Transform of the two halves.
  vec even_result(n / 2, 0);
  vec odd_result(n / 2, 0);
  RecursiveFourierTransformAlgorithm::operator()(even_sequence, even_result);
  RecursiveFourierTransformAlgorithm::operator()(odd_sequence, odd_result);

  // Implementing the Cooley-Tukey algorithm.
  for (size_t k = 0; k < n / 2; k++) {
    std::complex<real> p = even_result[k];
    std::complex<real> q =
        std::exp(std::complex<real>{0, 2 * base_angle * k / n}) * odd_result[k];

    output_sequence[k] = p + q;
    output_sequence[k + n / 2] = p - q;
  }
}

void IterativeFourierTransformAlgorithm::operator()(
    const vec &input_sequence, vec &output_sequence) const {
  // Getting the input size.
  const size_t n = input_sequence.size();

  // Check that the size of the sequence is a power of 2.
  const size_t log_n = static_cast<size_t>(log2(n));
  assert(1UL << log_n == n);

  // Perform bit reversal of the input sequence and store it into the output
  // sequence.
  (*bit_reversal_algorithm)(input_sequence, output_sequence);

  // Main loop: looping over the binary tree layers.
  for (size_t s = 1; s <= log_n; s++) {
    const size_t m = 1UL << s;
    const size_t half_m = m >> 1UL;

    const std::complex<real> omega_d =
        std::exp(std::complex<real>{0, base_angle / half_m});

#pragma omp parallel for schedule(static) default(none) \
    firstprivate(m, half_m, n, omega_d) shared(output_sequence)
    for (size_t k = 0; k < n; k += m) {
      std::complex<real> omega(1, 0);

      for (size_t j = 0; j < half_m; j++) {
        const size_t k_plus_j = k + j;
        const std::complex<real> t = omega * output_sequence[k_plus_j + half_m];
        const std::complex<real> u = output_sequence[k_plus_j];
        output_sequence[k_plus_j] = u + t;
        output_sequence[k_plus_j + half_m] = u - t;
        omega *= omega_d;
      }
    }
  }
}

void IterativeFFTGPU::operator()(const vec &input_sequence,
                                 vec &output_sequence) const {
  // Getting the input size.
  const size_t n = input_sequence.size();

  // Check that the size of the sequence is a power of 2.
  const size_t log_n = static_cast<size_t>(log2(n));
  assert(1UL << log_n == n);

  // Perform bit reversal of the input sequence and store it into the output
  // sequence.

  cuda::std::complex<real> *input_sequence_dev;
  cuda::std::complex<real> *output_sequence_dev;

  cudaMalloc(&input_sequence_dev, n * sizeof(cuda::std::complex<real>));
  cudaMalloc(&output_sequence_dev, n * sizeof(cuda::std::complex<real>));
  cudaMemcpy(input_sequence_dev, input_sequence.data(),
             n * sizeof(cuda::std::complex<real>), cudaMemcpyHostToDevice);

  bitreverse_gpu(input_sequence_dev, output_sequence_dev, n, log_n);
  // cudaDeviceSynchronize();

  for (size_t s = 1; s <= log_n; s++) {
    const size_t m = 1UL << s;
    run_fft_gpu(output_sequence_dev, n, m, base_angle);
  }

  cudaDeviceSynchronize();
  cudaMemcpy(output_sequence.data(), output_sequence_dev,
             n * sizeof(cuda::std::complex<real>), cudaMemcpyDeviceToHost);

  // #pragma acc exit data copyout(output_sequence)
}
/*
// A version of FastFourierTransformIterative that allows for fusion of the two
// inner looops. Experimental.
void IterativeFourierTransformAlgorithm::operator()(
    const vec &input_sequence, vec &output_sequence) const {
  // Defining some useful aliases.
  const size_t n = input_sequence.size();
  const size_t half_n = n >> 1;
  const unsigned int num_threads = omp_get_num_threads();

  // Check that the size of the sequence is a power of 2.
  const size_t log_n = static_cast<size_t>(log2(n));
  assert(1UL << log_n == n);

  // Perform bit reversal of the input sequence and store it into the output
  // sequence.
  (*bit_reversal_algorithm)(input_sequence, output_sequence);

  // Creation of a support vector to store values of omega.
  vec omegas(half_n, 0);

  // Main loop: looping over the binary tree layers.
  for (size_t s = 1; s <= log_n; s++) {
    const size_t m = 1UL << s;
    const size_t half_m = m >> 1UL;

    const std::complex<real> omega_d =
        std::exp(std::complex<real>{0, base_angle / half_m});

    const size_t iterations = half_m / num_threads;
#pragma omp parallel for default(none) shared(omegas) firstprivate( \
        num_threads, iterations, half_m, base_angle, omega_d) schedule(static)
    for (unsigned int thread = 0; thread < num_threads; thread++) {
      const size_t base_index = iterations * thread;
      omegas[base_index] =
          std::exp(std::complex<real>{0, base_index * base_angle / half_m});
      size_t end_index = base_index + iterations;
      for (size_t i = base_index + 1; i < end_index; i++) {
        omegas[i] = omegas[i - 1] * omega_d;
      }
    }

#pragma omp parallel for default(none) firstprivate(m, half_m, n) \
    shared(output_sequence, omegas) schedule(static) collapse(2)
    for (size_t k = 0; k < n; k += m) {
      for (size_t j = 0; j < half_m; j++) {
        const size_t k_plus_j = k + j;
        const std::complex<real> t =
            omegas[j] * output_sequence[k_plus_j + half_m];
        const std::complex<real> u = output_sequence[k_plus_j];
        output_sequence[k_plus_j] = u + t;
        output_sequence[k_plus_j + half_m] = u - t;
      }
    }
  }
}
*/

/*
// A version of the previous algorithm that manually fuses the two
// inner looops. Experimental.
void IterativeFourierTransformAlgorithm::operator()(
    const vec &input_sequence, vec &output_sequence) const {
  // Defining some useful aliases.
  const size_t n = input_sequence.size();
  const size_t half_n = n >> 1;
  const unsigned int num_threads = omp_get_num_threads();

  // Check that the size of the sequence is a power of 2.
  const size_t log_n = static_cast<size_t>(log2(n));
  assert(1UL << log_n == n);

  // Perform bit reversal of the input sequence and store it into the output
  // sequence.
  (*bit_reversal_algorithm)(input_sequence, output_sequence);

  // Creation of a support vector to store values of omega.
  vec omegas(half_n, 0);

  // Main loop: looping over the binary tree layers.
  for (size_t s = 1; s <= log_n; s++) {
    const size_t m = 1UL << s;
    const size_t half_m = m >> 1UL;
    const size_t s_minus_1 = s - 1UL;

    const std::complex<real> omega_d =
        std::exp(std::complex<real>{0, base_angle / half_m});

#pragma omp parallel for default(none) shared(omegas) \
    firstprivate(num_threads, half_m, base_angle, omega_d)
    for (unsigned int thread = 0; thread < num_threads; thread++) {
      const size_t iterations = half_m / num_threads;
      const size_t base_index = iterations * thread;
      omegas[base_index] =
          std::exp(std::complex<real>{0, base_index * base_angle / half_m});
      for (size_t i = base_index + 1; i < base_index + iterations; i++) {
        omegas[i] = omegas[i - 1] * omega_d;
      }
    }

#pragma omp parallel for default(none)            \
    firstprivate(half_m, half_n, s_minus_1, s, m) \
    shared(output_sequence, omegas) schedule(static)
    for (size_t index = 0; index < half_n; index++) {
      const size_t k = (index >> s_minus_1) << s;
      const size_t j = index % half_m;
      const size_t k_plus_j = k + j;
      const std::complex<real> t =
          omegas[j] * output_sequence[k_plus_j + half_m];
      const std::complex<real> u = output_sequence[k_plus_j];
      output_sequence[k_plus_j] = u + t;
      output_sequence[k_plus_j + half_m] = u - t;
    }
  }
}
*/

// Calculate time for execution using chrono.
unsigned long FourierTransformAlgorithm::calculateTime(
    const vec &input_sequence, vec &output_sequence) const {
  auto t0 = std::chrono::high_resolution_clock::now();
  this->operator()(input_sequence, output_sequence);
  auto t1 = std::chrono::high_resolution_clock::now();
  const auto time =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  return time;
}

void TimeEstimateFFT(std::unique_ptr<FourierTransformAlgorithm> &ft_algorithm,
                     const vec &sequence, unsigned int max_num_threads) {
  // Calculate sequence size.
  const size_t size = sequence.size();
  unsigned long serial_time = 0;

  // Create the output sequence.
  vec output_sequence(size, 0);

  // For each number of threads.
  for (unsigned int num_threads = 1; num_threads <= max_num_threads;
       num_threads *= 2) {
    // Set the number of threads.
    omp_set_num_threads(num_threads);

    // Execute the transform.
    const unsigned long time =
        ft_algorithm->calculateTime(sequence, output_sequence);
    if (num_threads == 1) serial_time = time;
    std::cout << "Time for parallel FFT with " << size << " elements and "
              << num_threads << " threads: " << time << "μs" << std::endl;

    // Calculate and print speedups.
    std::cout << "Speedup over fast standard: "
              << static_cast<double>(serial_time) / time << "x" << std::endl;

    std::cout << std::endl;
  }
}

}  // namespace FourierTransform
