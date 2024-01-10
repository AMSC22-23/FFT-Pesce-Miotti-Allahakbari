#include "fft_main.hpp"

/**
 * @file fft_main.cpp.
 * @brief Defines the main function for CPU FFT-related tests.
 */

#include <omp.h>

#include <numbers>
#include <string>

#include "FourierTransformCalculator.hpp"
#include "Utility.hpp"

void print_usage(size_t size, const std::string &mode,
                 unsigned int max_num_threads) {
  std::cerr << "Incorrect arguments!\n"
            << "Argument 1: fft\n"
            << "Argument 2: size of the sequence (default: " << size
            << "), must be a power of 2\n"
            << "Argument 3: execution mode (demo / bitReversalTest / "
               "scalingTest / timingTest (default: "
            << mode << "))\n"
            << "Argument 4: maximum number of threads (default: "
            << max_num_threads << ")\n"
            << "Argument 5: algorithm for timingTest mode (classic, recursive, "
               "iterative (default))"
            << std::endl;
}

int fft_main(int argc, char *argv[]) {
  using namespace Transform;
  using namespace FourierTransform;

  constexpr size_t default_size = 1UL << 10;
  constexpr unsigned int default_max_num_threads = 8;
  const std::string default_mode = "demo";
  constexpr real precision =
      std::max(std::numeric_limits<real>::epsilon() * 1e5, 1e-4);

  // Check the number of arguments.
  if (argc > 6) {
    print_usage(default_size, default_mode, default_max_num_threads);
    return 1;
  }

  // Get the size of the sequence.
  size_t size = default_size;
  if (argc >= 3) {
    char *error;

    const unsigned long long_size = strtoul(argv[2], &error, 10);
    // Check for errors.
    if (*error ||
        1UL << static_cast<unsigned long>(log2(long_size)) != long_size) {
      print_usage(default_size, default_mode, default_max_num_threads);
      return 1;
    }

    size = static_cast<size_t>(long_size);
  }

  // Get the FFT execution mode.
  std::string mode = default_mode;
  if (argc >= 4) mode = std::string(argv[3]);

  // Get the maximum number of threads.
  unsigned int max_num_threads = default_max_num_threads;
  if (argc >= 5) {
    char *error;

    const unsigned long long_max_num_threads = strtoul(argv[4], &error, 10);
    // Check for errors.
    if (*error) {
      print_usage(default_size, default_mode, default_max_num_threads);
      return 1;
    }

    max_num_threads = static_cast<unsigned int>(long_max_num_threads);
  }

  // Generate a sequence of complex numbers.
  vec input_sequence;
  input_sequence.reserve(size);
  for (size_t i = 0; i < size; i++) {
    // Add a random complex number to the sequence.
    input_sequence.emplace_back(rand() % 100, rand() % 100);
  }

  // Run a demo of the hands-on code.
  if (mode == std::string("demo")) {
    // Check the number of arguments.
    if (argc > 5) {
      print_usage(default_size, default_mode, default_max_num_threads);
      return 1;
    }

    // Set the maximum number of threads.
    omp_set_num_threads(max_num_threads);

    // Save the sequence to a file.
    WriteToFile(input_sequence, "input_sequence.csv");

    // Create the FourierTransformCalculator object.
    FourierTransformCalculator calculator;

    // Compute the O(n^2) Fourier Transform of the sequence.
    std::unique_ptr<FourierTransformAlgorithm> classical_dft =
        std::make_unique<ClassicalFourierTransformAlgorithm>();
    calculator.setDirectAlgorithm(classical_dft);
    vec classical_dft_result(size, 0);
    calculator.directTransform(input_sequence, classical_dft_result);
    WriteToFile(classical_dft_result, "classical_dft_result.csv");

    // Compute the O(n log n) Fourier Transform of the sequence with the
    // recursive algorithm.
    std::unique_ptr<FourierTransformAlgorithm> recursive_dft =
        std::make_unique<RecursiveFourierTransformAlgorithm>();
    calculator.setDirectAlgorithm(recursive_dft);
    vec recursive_dft_result(size, 0);
    calculator.directTransform(input_sequence, recursive_dft_result);
    WriteToFile(recursive_dft_result, "recursive_dft_result.csv");

    // Compute the O(n log n) Fourier Transform of the sequence with the
    // iterative algorithm.
    std::unique_ptr<BitReversalPermutationAlgorithm>
        mask_bit_reversal_algorithm =
            std::make_unique<MaskBitReversalPermutationAlgorithm>();
    std::unique_ptr<FourierTransformAlgorithm> iterative_dft =
        std::make_unique<IterativeFourierTransformAlgorithm>(
            mask_bit_reversal_algorithm);
    calculator.setDirectAlgorithm(iterative_dft);
    vec iterative_dft_result(size, 0);
    calculator.directTransform(input_sequence, iterative_dft_result);
    WriteToFile(iterative_dft_result, "iterative_dft_result.csv");

    std::unique_ptr<FourierTransformAlgorithm> iterative_fft_gpu =
        std::make_unique<IterativeFFTGPU>();
    calculator.setDirectAlgorithm(iterative_fft_gpu);
    vec iterative_gpu_result(size, 0);
    calculator.directTransform(input_sequence, iterative_gpu_result);
    WriteToFile(iterative_dft_result, "iterative_gpu_result.csv");

    // Check the results for errors.
    if (!CompareVectors(classical_dft_result, recursive_dft_result, precision,
                        false))
      std::cerr << "Errors detected in recursive direct FFT." << std::endl;
    if (!CompareVectors(classical_dft_result, iterative_dft_result, precision,
                        false))
      std::cerr << "Errors detected in iterative direct FFT." << std::endl;
    if (!CompareVectors(classical_dft_result, iterative_gpu_result, precision,
                        false))
      std::cerr << "Errors detected in iterative GPU FFT." << std::endl;

    // Compute the O(n^2) Fourier Transform of the result.
    std::unique_ptr<FourierTransformAlgorithm> classical_ift =
        std::make_unique<ClassicalFourierTransformAlgorithm>();
    calculator.setInverseAlgorithm(classical_ift);
    vec classical_ift_result(size, 0);
    calculator.inverseTransform(classical_dft_result, classical_ift_result);
    WriteToFile(classical_ift_result, "classical_ift_result.csv");

    // Compute the iterative O(n log) Inverse Fourier Transform of the result.
    std::unique_ptr<BitReversalPermutationAlgorithm>
        fast_bit_reversal_algorithm =
            std::make_unique<FastBitReversalPermutationAlgorithm>();
    std::unique_ptr<FourierTransformAlgorithm> iterative_ift =
        std::make_unique<IterativeFourierTransformAlgorithm>(
            fast_bit_reversal_algorithm);
    calculator.setInverseAlgorithm(iterative_ift);
    vec iterative_ift_result(size, 0);
    calculator.inverseTransform(classical_dft_result, iterative_ift_result);
    WriteToFile(iterative_ift_result, "iterative_ift_result.csv");

    // Check if the new inverse sequences are equal to the original one.
    if (!CompareVectors(input_sequence, classical_ift_result, precision, false))
      std::cerr << "Errors detected in classical inverse FFT." << std::endl;
    if (!CompareVectors(input_sequence, iterative_ift_result, precision, false))
      std::cerr << "Errors detected in iterative inverse FFT." << std::endl;

    // Test the 2D algorithms.
    // Generate a sequence.
    vec input_matrix;
    input_matrix.reserve(size * size);
    for (size_t i = 0; i < size * size; i++) {
      // Add a random complex number to the sequence.
      input_matrix.emplace_back(rand() % 100, rand() % 100);
    }

    // Run the algorithms.
    TrivialTwoDimensionalFourierTransformAlgorithm two_d_fft;
    vec two_d_fft_result(size * size, 0);
    two_d_fft(input_matrix, two_d_fft_result);
    TrivialTwoDimensionalInverseFourierTransformAlgorithm two_d_ift;
    vec two_d_ift_result(size * size, 0);
    two_d_ift(two_d_fft_result, two_d_ift_result);

    // Check the result.
    if (!CompareVectors(input_matrix, two_d_ift_result, precision, false))
      std::cerr << "Errors detected in 2D FFT." << std::endl;
  }

  // Run a performance comparison of different bit reversal permutation
  // techniques.
  else if (mode == std::string("bitReversalTest")) {
    // Check the number of arguments.
    if (argc > 5) {
      print_usage(default_size, default_mode, default_max_num_threads);
      return 1;
    }

    // Run a comparison between mask and fast bit reversal permutations.
    CompareBitReversalPermutationTimes(input_sequence, max_num_threads);
  }

  // Run a scaling test with the best performing algorithm.
  else if (mode == std::string("scalingTest")) {
    // Check the number of arguments.
    if (argc > 5) {
      print_usage(default_size, default_mode, default_max_num_threads);
      return 1;
    }

    // Calculate the times for up to max_num_threads threads for the iterative
    // fft.
    std::unique_ptr<BitReversalPermutationAlgorithm> bit_reversal_algorithm =
        std::make_unique<MaskBitReversalPermutationAlgorithm>();
    std::unique_ptr<IterativeFourierTransformAlgorithm>
        iterative_dft_algorithm =
            std::make_unique<IterativeFourierTransformAlgorithm>(
                bit_reversal_algorithm);
    iterative_dft_algorithm->setBaseAngle(-std::numbers::pi_v<real>);
    std::unique_ptr<FourierTransformAlgorithm> iterative_dft =
        std::move(iterative_dft_algorithm);
    TimeEstimateFFT(iterative_dft, input_sequence, max_num_threads);
  }

  // Execute a single algorithm and calculate the elapsed time.
  else if (mode == std::string("timingTest")) {
    // Get which algorithm to choose.
    std::string algorithm_name = "iterative";
    if (argc >= 6) algorithm_name = std::string(argv[5]);

    // Create the algorithm.
    std::unique_ptr<FourierTransformAlgorithm> algorithm;
    if (algorithm_name == std::string("classic")) {
      algorithm = std::make_unique<ClassicalFourierTransformAlgorithm>();
    } else if (algorithm_name == std::string("recursive")) {
      algorithm = std::make_unique<RecursiveFourierTransformAlgorithm>();
    } else if (algorithm_name == std::string("iterative")) {
      std::unique_ptr<BitReversalPermutationAlgorithm> bit_reversal_algorithm =
          std::make_unique<MaskBitReversalPermutationAlgorithm>();
      algorithm = std::make_unique<IterativeFourierTransformAlgorithm>(
          bit_reversal_algorithm);
    } else if (algorithm_name == std::string("iterativeGPU")) {
      algorithm = std::make_unique<IterativeFFTGPU>();
    } else {
      print_usage(default_size, default_mode, default_max_num_threads);
      return 1;
    }

    // Set the direct transform.
    algorithm->setBaseAngle(-std::numbers::pi_v<real>);

    // Create an output vector.
    vec result(size, 0);

    // Set the number of threads.
    omp_set_num_threads(max_num_threads);

    // Execute the algorithm and calculate the time.
    std::cout << algorithm->calculateTime(input_sequence, result) << "μs"
              << std::endl;
  }

  return 0;
}