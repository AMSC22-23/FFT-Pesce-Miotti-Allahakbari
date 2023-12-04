#include <omp.h>

#include <iostream>

#include "FourierTransformCalculator.hpp"
#include "Utility.hpp"
#include "VectorExporter.hpp"

int main(int argc, char* argv[]) {
  using namespace FourierTransform;

  // Set a default size for the sequence.
  size_t size = 1UL << 10;

  // Check the number of arguments.
  if (argc > 2) {
    std::cerr << "Too many arguments." << std::endl;
    std::cerr << "Argument 1: size of the sequence (default: " << size << ")"
              << std::endl;
    return 1;
  }

  // Get the size of the sequence.
  if (argc == 2) size = atoi(argv[1]);

  // Generating a sequence of complex numbers.
  vec input_sequence;
  input_sequence.reserve(size);

  for (size_t i = 0; i < size; i++) {
    // Add a random complex number to the sequence.
    input_sequence.emplace_back(rand() % 100, rand() % 100);
  }

  omp_set_num_threads(1);

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

  // Compute the O(n log n) Fourier Transform of the sequence with the recursive
  // algorithm.
  std::unique_ptr<FourierTransformAlgorithm> recursive_dft(
      new RecursiveFourierTransformAlgorithm());
  calculator.setDirectAlgorithm(recursive_dft);
  vec recursive_dft_result(size, 0);
  calculator.directTransform(input_sequence, recursive_dft_result);
  WriteToFile(recursive_dft_result, "recursive_dft_result.csv");

  // Compute the O(n log n) Fourier Transform of the sequence with the iterative
  // algorithm.
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
  if (!CompareVectors(classical_dft_result, recursive_dft_result, 1e-4, false))
    std::cerr << "Errors detected in recursive direct FFT." << std::endl;
  if (!CompareVectors(classical_dft_result, iterative_dft_result, 1e-4, false))
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

  // Bit reversal permutation test, recommended sequence size: 1UL << 27.
  CompareTimesBitReversalPermutation(input_sequence, 8);

  // Calculate the times for up to 8 cores for the iterative fft.
  IterativeFourierTransformAlgorithm* iterative_dft_algorithm2 =
      new IterativeFourierTransformAlgorithm();
  std::unique_ptr<BitReversalPermutationAlgorithm> bit_reversal_algorithm3(
      new MaskBitReversalPermutationAlgorithm());
  iterative_dft_algorithm2->setBitReversalPermutationAlgorithm(
      bit_reversal_algorithm3);
  std::unique_ptr<FourierTransformAlgorithm> iterative_dft2(
      iterative_dft_algorithm2);
  TimeEstimateFFT(iterative_dft2, input_sequence, 8);
  return 0;
}
