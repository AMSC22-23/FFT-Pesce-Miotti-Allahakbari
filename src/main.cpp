#include <omp.h>
#include <tgmath.h>

#include <iostream>
#include <numbers>
#include <string>

#include "FourierTransformCalculator.hpp"
#include "Utility.hpp"
#include "VectorExporter.hpp"

void print_usage(size_t size, unsigned int max_num_threads);

int main(int argc, char* argv[]) {
  using namespace FourierTransform;

  size_t size = 1UL << 10;
  std::string mode = "demo";
  unsigned int max_num_threads = 8;

  // Check the number of arguments.
  if (argc > 5) {
    print_usage(size, max_num_threads);
    return 1;
  }

  // Get the size of the sequence.
  if (argc >= 2) {
    char* error;

    const unsigned long long_size = strtoul(argv[1], &error, 10);
    // Check for errors
    if (*error ||
        1UL << static_cast<unsigned long>(log2(long_size)) != long_size) {
      print_usage(size, max_num_threads);
      return 1;
    }

    size = static_cast<size_t>(long_size);
  }

  // Get which mode to execute the program in.
  if (argc >= 3) mode = std::string(argv[2]);

  // Get the maximum number of threads.
  if (argc >= 4) {
    char* error;

    const unsigned long long_max_num_threads = strtoul(argv[3], &error, 10);
    // Check for errors
    if (*error) {
      print_usage(size, max_num_threads);
      return 1;
    }

    max_num_threads = static_cast<size_t>(long_max_num_threads);
  }

  // Generate a sequence of complex numbers.
  vec input_sequence;
  input_sequence.reserve(size);
  for (size_t i = 0; i < size; i++) {
    // Add a random complex number to the sequence.
    input_sequence.emplace_back(rand() % 100, rand() % 100);
  }

  // Execute different code based on the chosen execution mode.
  // Default execution mode: give a general demonstration of the implemented
  // functionalities.
  if (mode == std::string("demo")) {
    omp_set_num_threads(max_num_threads);

    // Save the sequence to a file.
    WriteToFile(input_sequence, "input_sequence.csv");

    // Create the FourierTransformCalculator object.
    FourierTransformCalculator calculator;

    // Compute the O(n^2) Fourier Transform of the sequence.
    std::unique_ptr<FourierTransformAlgorithm> classical_dft(
        new ClassicalFourierTransformAlgorithm());
    calculator.setDirectAlgorithm(classical_dft);
    vec classical_dft_result(size, 0);
    calculator.directTransform(input_sequence, classical_dft_result);
    WriteToFile(classical_dft_result, "classical_dft_result.csv");

    // Compute the O(n log n) Fourier Transform of the sequence with the
    // recursive algorithm.
    std::unique_ptr<FourierTransformAlgorithm> recursive_dft(
        new RecursiveFourierTransformAlgorithm());
    calculator.setDirectAlgorithm(recursive_dft);
    vec recursive_dft_result(size, 0);
    calculator.directTransform(input_sequence, recursive_dft_result);
    WriteToFile(recursive_dft_result, "recursive_dft_result.csv");

    // Compute the O(n log n) Fourier Transform of the sequence with the
    // iterative algorithm.
    IterativeFourierTransformAlgorithm* iterative_dft_algorithm =
        new IterativeFourierTransformAlgorithm();
    std::unique_ptr<BitReversalPermutationAlgorithm> bit_reversal_algorithm(
        new MaskBitReversalPermutationAlgorithm());
    iterative_dft_algorithm->setBitReversalPermutationAlgorithm(
        bit_reversal_algorithm);
    std::unique_ptr<FourierTransformAlgorithm> iterative_dft(
        iterative_dft_algorithm);
    calculator.setDirectAlgorithm(iterative_dft);
    vec iterative_dft_result(size, 0);
    calculator.directTransform(input_sequence, iterative_dft_result);
    WriteToFile(iterative_dft_result, "iterative_dft_result.csv");

    // Check the results for errors.
    if (!CompareVectors(classical_dft_result, recursive_dft_result, 1e-4,
                        false))
      std::cerr << "Errors detected in recursive direct FFT." << std::endl;
    if (!CompareVectors(classical_dft_result, iterative_dft_result, 1e-4,
                        false))
      std::cerr << "Errors detected in iterative direct FFT." << std::endl;

    // Compute the O(n^2) Fourier Transform of the result.
    std::unique_ptr<FourierTransformAlgorithm> classical_ift(
        new ClassicalFourierTransformAlgorithm());
    calculator.setInverseAlgorithm(classical_ift);
    vec classical_ift_result(size, 0);
    calculator.inverseTransform(classical_dft_result, classical_ift_result);
    WriteToFile(classical_ift_result, "classical_ift_result.csv");

    // Compute the iterative O(n log) Inverse Fourier Transform of the result.
    IterativeFourierTransformAlgorithm* iterative_ift_algorithm =
        new IterativeFourierTransformAlgorithm();
    std::unique_ptr<BitReversalPermutationAlgorithm> bit_reversal_algorithm2(
        new FastBitReversalPermutationAlgorithm());
    iterative_ift_algorithm->setBitReversalPermutationAlgorithm(
        bit_reversal_algorithm2);
    std::unique_ptr<FourierTransformAlgorithm> iterative_ift(
        iterative_ift_algorithm);
    calculator.setInverseAlgorithm(iterative_ift);
    vec iterative_ift_result(size, 0);
    calculator.inverseTransform(classical_dft_result, iterative_ift_result);
    WriteToFile(iterative_ift_result, "iterative_ift_result.csv");

    // Check if the new inverse sequences are equal to the original one.
    if (!CompareVectors(input_sequence, classical_ift_result, 1e-4, false))
      std::cerr << "Errors detected in classical inverse FFT." << std::endl;
    if (!CompareVectors(input_sequence, iterative_ift_result, 1e-4, false))
      std::cerr << "Errors detected in iterative inverse FFT." << std::endl;
  }

  // Run a performance comparison of different bit reversal permutation
  // techniques.
  else if (mode == std::string("bitReversalTest")) {
    // Run a comparison between mask and fast bit reversal permutations.
    CompareTimesBitReversalPermutation(input_sequence, max_num_threads);
  }

  // Run a scaling test with the best performing algorithm.
  else if (mode == std::string("scalingTest")) {
    // Calculate the times for up to max_num_threads threads for the iterative
    // fft.
    IterativeFourierTransformAlgorithm* iterative_dft_algorithm =
        new IterativeFourierTransformAlgorithm();
    iterative_dft_algorithm->setBaseAngle(-std::numbers::pi_v<real>);
    std::unique_ptr<BitReversalPermutationAlgorithm> bit_reversal_algorithm(
        new MaskBitReversalPermutationAlgorithm());
    iterative_dft_algorithm->setBitReversalPermutationAlgorithm(
        bit_reversal_algorithm);
    std::unique_ptr<FourierTransformAlgorithm> iterative_dft(
        iterative_dft_algorithm);
    TimeEstimateFFT(iterative_dft, input_sequence, max_num_threads);
  }

  // Execute a single algorithm and calculate the elapsed time.
  else if (mode == std::string("timingTest")) {
    std::string algorithm_name = "iterative";

    // Get which algorithm to choose.
    if (argc >= 5) algorithm_name = std::string(argv[4]);

    // Create the algorithm.
    std::unique_ptr<FourierTransformAlgorithm> algorithm;
    if (algorithm_name == std::string("classic")) {
      algorithm = std::unique_ptr<FourierTransformAlgorithm>(
          new ClassicalFourierTransformAlgorithm());
    } else if (algorithm_name == std::string("recursive")) {
      algorithm = std::unique_ptr<FourierTransformAlgorithm>(
          new RecursiveFourierTransformAlgorithm());
    } else if (algorithm_name == std::string("iterative")) {
      IterativeFourierTransformAlgorithm* iterative_algorithm =
          new IterativeFourierTransformAlgorithm();
      std::unique_ptr<BitReversalPermutationAlgorithm> bit_reversal_algorithm(
          new MaskBitReversalPermutationAlgorithm());
      iterative_algorithm->setBitReversalPermutationAlgorithm(
          bit_reversal_algorithm);
      algorithm =
          std::unique_ptr<FourierTransformAlgorithm>(iterative_algorithm);
    } else {
      print_usage(size, max_num_threads);
      return 1;
    }

    // Set the direct transform.
    algorithm->setBaseAngle(-std::numbers::pi_v<real>);

    // Create an output vector.
    vec result(size, 0);

    // Execute the algorithm and calculate the time.
    std::cout << algorithm->calculateTime(input_sequence, result) << "μs"
              << std::endl;
  }

  // Wrong mode specified.
  else {
    print_usage(size, max_num_threads);
    return 1;
  }

  return 0;
}

void print_usage(size_t size, unsigned int max_num_threads) {
  std::cerr << "Incorrect arguments!\n"
            << "Argument 1: size of the sequence (default: " << size
            << "), must be a power of 2\n"
            << "Argument 2: execution mode (demo (default) / bitReversalTest / "
               "scalingTest, timeTest)\n"
            << "Argument 3: maximum number of threads (default: "
            << max_num_threads << ")\n"
            << "Argument 4: algorithm for timeTest mode (classic, recursive, "
               "iterative (default))"
            << std::endl;
}