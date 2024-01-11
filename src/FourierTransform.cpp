#include "FourierTransform.hpp"

/**
 * @file FourierTransform.cpp.
 * @brief Defines the methods and functions declared in
 * FourierTransform.hpp.
 */

// TODO: Remove commented code in the GPU implementations.

#include <cuda_runtime.h>
#include <omp.h>
#include <tgmath.h>

#include <cassert>
#include <chrono>
#include <iostream>
#include <numbers>

#include "FFTGPU.hpp"
#include "FourierTransform.hpp"
#include "Utility.hpp"

namespace Transform {
namespace FourierTransform {

// An alias for pi.
constexpr real pi = std::numbers::pi_v<real>;

// The classical O(n^2) Fourier Transform.
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

// A recursive implementation of the FFT.
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

// An iterative implementation of the FFT using OpenMP.
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

// A trivial implementation of the 2D Direct Fourier Transform.
void TrivialTwoDimensionalFourierTransformAlgorithm::operator()(
    const vec &input_sequence, vec &output_sequence) const {
  // Getting the input size.
  const size_t n = input_sequence.size();

  // Handle the case where n = 1 separately.
  if (n == 1) {
    output_sequence[0] = input_sequence[0];
    return;
  }

  // Get the square root of the input size.
  const size_t sqrt_n = static_cast<size_t>(sqrt(n));

  // Check that the side of the matrix is a power of 2.
  const size_t log_sqrt_n = static_cast<size_t>(log2(sqrt_n));
  assert(1UL << log_sqrt_n == sqrt_n);

  // Use the IterativeFourierTransformAlgorithm to compute the 1D FFT.
  std::unique_ptr<BitReversalPermutationAlgorithm> bit_reversal_algorithm =
      std::make_unique<MaskBitReversalPermutationAlgorithm>();
  IterativeFourierTransformAlgorithm fft_algorithm(bit_reversal_algorithm);

  // Set the base angle to -pi.
  fft_algorithm.setBaseAngle(-pi);

  // Use the 1D FFT algotithm to compute the 2D FFT.
  for (size_t i = 0; i < sqrt_n; i++) {
    // Get the i-th row of the input matrix.
    vec row(sqrt_n, 0);
    for (size_t j = 0; j < sqrt_n; j++) row[j] = input_sequence[i * sqrt_n + j];

    // Compute the i-th row of the output matrix.
    vec output_row(sqrt_n, 0);
    fft_algorithm(row, output_row);

    // Store the i-th row of the output matrix.
    for (size_t j = 0; j < sqrt_n; j++)
      output_sequence[i * sqrt_n + j] = output_row[j];
  }

  // Do the same for the columns.
  for (size_t j = 0; j < sqrt_n; j++) {
    // Get the j-th column of the input matrix.
    vec column(sqrt_n, 0);
    for (size_t i = 0; i < sqrt_n; i++)
      column[i] = output_sequence[i * sqrt_n + j];

    // Compute the j-th column of the output matrix.
    vec output_column(sqrt_n, 0);
    fft_algorithm(column, output_column);

    // Store the j-th column of the output matrix.
    for (size_t i = 0; i < sqrt_n; i++)
      output_sequence[i * sqrt_n + j] = output_column[i];
  }
}

// A trivial implementation of the 2D Inverse Fourier Transform.
void TrivialTwoDimensionalInverseFourierTransformAlgorithm::operator()(
    const vec &input_sequence, vec &output_sequence) const {
  // Getting the input size.
  const size_t n = input_sequence.size();

  // Handle the case where n = 1 separately.
  if (n == 1) {
    output_sequence[0] = input_sequence[0];
    return;
  }

  // Get the square root of the input size.
  const size_t sqrt_n = static_cast<size_t>(sqrt(n));

  // Check that the side of the matrix is a power of 2.
  const size_t log_sqrt_n = static_cast<size_t>(log2(sqrt_n));
  assert(1UL << log_sqrt_n == sqrt_n);

  // Use the IterativeFourierTransformAlgorithm to compute the 1D FFT.
  std::unique_ptr<BitReversalPermutationAlgorithm> bit_reversal_algorithm =
      std::make_unique<MaskBitReversalPermutationAlgorithm>();
  IterativeFourierTransformAlgorithm fft_algorithm(bit_reversal_algorithm);

  // Set the base angle to pi.
  fft_algorithm.setBaseAngle(pi);

  for (size_t j = 0; j < sqrt_n; j++) {
    // Get the j-th column of the input matrix.
    vec column(sqrt_n, 0);
    for (size_t i = 0; i < sqrt_n; i++)
      column[i] = input_sequence[i * sqrt_n + j];

    // Compute the j-th column of the output matrix.
    vec output_column(sqrt_n, 0);
    fft_algorithm(column, output_column);

    // Store the j-th column of the output matrix.
    for (size_t i = 0; i < sqrt_n; i++)
      output_sequence[i * sqrt_n + j] = output_column[i];
  }

  for (size_t i = 0; i < sqrt_n; i++) {
    // Get the i-th row of the input matrix.
    vec row(sqrt_n, 0);
    for (size_t j = 0; j < sqrt_n; j++)
      row[j] = output_sequence[i * sqrt_n + j];

    // Compute the i-th row of the output matrix.
    vec output_row(sqrt_n, 0);
    fft_algorithm(row, output_row);

    // Store the i-th row of the output matrix.
    for (size_t j = 0; j < sqrt_n; j++)
      output_sequence[i * sqrt_n + j] = output_row[j];
  }

  // Divide the output matrix by n.
  for (size_t i = 0; i < n; i++) output_sequence[i] /= n;
}

// A GPU implementation of the 1D iterative FFT, handling memory transfer and
// calling CUDA kernels.
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

  for (size_t s = 1; s <= log_n; s++) {
    const size_t m = 1UL << s;
    run_fft_gpu(output_sequence_dev, n, m, base_angle);
  }

  cudaDeviceSynchronize();
  cudaMemcpy(output_sequence.data(), output_sequence_dev,
             n * sizeof(cuda::std::complex<real>), cudaMemcpyDeviceToHost);

  cudaFree(input_sequence_dev);
  cudaFree(output_sequence_dev);
}

// A GPU implementation of the 2D iterative FFT, handling memory transfer and
// calling CUDA kernels.
void IterativeFFTGPU2D::operator()(const vec &input_sequence,
                                   vec &output_sequence) const {
  // Getting the input size.
  const size_t size = input_sequence.size();
  const size_t n = sqrt(size);

  // Check that the size of the sequence is a power of 2.
  const size_t log_n = static_cast<size_t>(log2(n));
  assert(1UL << log_n == n);

  // Perform bit reversal of the input sequence and store it into the output
  // sequence.

  cuda::std::complex<real> *input_sequence_dev;
  cuda::std::complex<real> *output_sequence_dev;
  cuda::std::complex<real> *transposed_sequence_dev;

  cudaMalloc(&input_sequence_dev, size * sizeof(cuda::std::complex<real>));
  cudaMalloc(&output_sequence_dev, size * sizeof(cuda::std::complex<real>));
  cudaMalloc(&transposed_sequence_dev, size * sizeof(cuda::std::complex<real>));

  // Loop over all the rows
  for (size_t i = 0; i < n; i++) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(&input_sequence_dev[i * n], &input_sequence.data()[i * n],
                    n * sizeof(cuda::std::complex<real>),
                    cudaMemcpyHostToDevice, stream);

    bitreverse_gpu(&input_sequence_dev[i * n], &output_sequence_dev[i * n], n,
                   log_n, stream);

    for (size_t s = 1; s <= log_n; s++) {
      const size_t m = 1UL << s;
      run_fft_gpu(&output_sequence_dev[i * n], n, m, base_angle, stream);
    }
    swap_row_col_gpu(output_sequence_dev, transposed_sequence_dev, i, i, n,
                     stream);
    cudaStreamDestroy(stream);
  }

  // transpose_gpu(output_sequence_dev, transposed_sequence_dev, n);
  cudaDeviceSynchronize();
  /// Loop over all the columns
  for (size_t i = 0; i < n; i++) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    bitreverse_gpu(&transposed_sequence_dev[i * n], &output_sequence_dev[i * n],
                   n, log_n, stream);

    for (size_t s = 1; s <= log_n; s++) {
      const size_t m = 1UL << s;
      run_fft_gpu(&output_sequence_dev[i * n], n, m, base_angle, stream);
    }
    swap_row_col_gpu(output_sequence_dev, input_sequence_dev, i, i, n, stream);

    // cudaMemcpyAsync(&output_sequence.data()[i * n], &input_sequence_dev[i *
    // n],
    //                 n * sizeof(cuda::std::complex<real>),
    //                 cudaMemcpyDeviceToHost, stream);

    cudaStreamDestroy(stream);
  }

  // transpose_gpu(output_sequence_dev, transposed_sequence_dev, n);

  cudaDeviceSynchronize();

  cudaMemcpy(output_sequence.data(), input_sequence_dev,
             size * sizeof(cuda::std::complex<real>), cudaMemcpyDeviceToHost);

  cudaFree(&input_sequence_dev);
  cudaFree(&output_sequence_dev);
  cudaFree(&transposed_sequence_dev);
}

void BlockFFTGPU2D::operator()(const vec &input_sequence,
                               vec &output_sequence) const {
  // Getting the input size.
  const size_t size = input_sequence.size();
  const size_t n = sqrt(size);
  const size_t n_blk = _block_size;

  // Check that the size of the sequence is a power of 2.
  const size_t log_n = static_cast<size_t>(log2(n));
  assert(1UL << log_n == n);

  const size_t log_n_blk = static_cast<size_t>(log2(n_blk));

  // Perform bit reversal of the input sequence and store it into the output
  // sequence.
  cuda::std::complex<real> *block_input_dev;

  cudaMalloc(&block_input_dev, size * sizeof(cuda::std::complex<real>));

  int num_streams = 8;
  cudaStream_t stream[num_streams];
  for (int i = 0; i < num_streams; i++) cudaStreamCreate(&stream[i]);

  for (int i = 0; i < num_streams; i++) {
    cudaMemcpyAsync(&block_input_dev[i * n / num_streams * n],
                    &input_sequence.data()[i * n / num_streams * n],
                    n / num_streams * n * sizeof(cuda::std::complex<real>),
                    cudaMemcpyHostToDevice, stream[i]);
    run_block_fft_gpu(&block_input_dev[i * n / num_streams * n], n, base_angle,
                      num_streams, stream[i]);
    run_block_fft_gpu(&block_input_dev[i * n / num_streams * n], n, base_angle,
                      num_streams, stream[i]);
  }
  for (int i = 0; i < num_streams; i++) {
    cudaMemcpyAsync(&output_sequence.data()[i * n / num_streams * n],
                    &block_input_dev[i * n / num_streams * n],
                    n / num_streams * n * sizeof(cuda::std::complex<real>),
                    cudaMemcpyDeviceToHost, stream[i]);
  }

  cudaDeviceSynchronize();
  for (int i = 0; i < num_streams; i++) cudaStreamDestroy(stream[i]);
  cudaFree(block_input_dev);
}

// Calculate time for execution of a Fourier Transform using chrono.
unsigned long FourierTransformAlgorithm::calculateTime(
    const vec &input_sequence, vec &output_sequence) const {
  auto t0 = std::chrono::high_resolution_clock::now();
  this->operator()(input_sequence, output_sequence);
  auto t1 = std::chrono::high_resolution_clock::now();
  const auto time =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  return time;
}

// Calculate time for execution of a Fourier Transform using chrono for multiple
// numbers of threads.
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
}  // namespace Transform
