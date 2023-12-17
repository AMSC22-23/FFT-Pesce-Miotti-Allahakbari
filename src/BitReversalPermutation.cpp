#include "BitReversalPermutation.hpp"

#include <omp.h>
#include <tgmath.h>

#include <cassert>
#include <chrono>
#include <iostream>

#include "Utility.hpp"

namespace FourierTransform {

// Compute the reverse bit order of "index", given a bitsize.
size_t NaiveBitReverse(const size_t index, const size_t bitsize) {
  // Initialize the result.
  size_t result = 0;

  for (size_t i = 0; i < bitsize; i++) {
    // Set the i-th bit of the index to the same position, from the top.
    result |= ((1 << i) & index) >> i << (bitsize - i - 1);
  }

  return result;
}

// Compute the reverse bit order of "index", given a bitsize.
// More efficient implementation.
// Source: https://rosettacode.org/wiki/Fast_Fourier_transform#C.2B.2B
#pragma omp declare simd linear(index) uniform(bitsize) notinbranch
size_t MaskBitReverse(const size_t index, const size_t bitsize) {
  // Initialize the result.
  size_t result = index;
//@note to avoid possible problems I would use a static_assert
//      to verify that result is an unsigned long. Indeed this 
//      function works only if size_t is an unsigned long
//      (which is normally the case in a 64 bit architecture)
  result = (((result & 0xaaaaaaaaaaaaaaaa) >> 1) |
            ((result & 0x5555555555555555) << 1));
  result = (((result & 0xcccccccccccccccc) >> 2) |
            ((result & 0x3333333333333333) << 2));
  result = (((result & 0xf0f0f0f0f0f0f0f0) >> 4) |
            ((result & 0x0f0f0f0f0f0f0f0f) << 4));
  result = (((result & 0xff00ff00ff00ff00) >> 8) |
            ((result & 0x00ff00ff00ff00ff) << 8));
  result = (((result & 0xffff0000ffff0000) >> 16) |
            ((result & 0x0000ffff0000ffff) << 16));
  result = ((result >> 32) | (result << 32)) >> (64 - bitsize);
  return result;
}

// Call BitReverse on all elements of "input_sequence".
void NaiveBitReversalPermutationAlgorithm::operator()(
    const vec &input_sequence, vec &output_sequence) const {
  // Get the size and bitsize of the sequence, then perform a sanity check.
  const size_t n = input_sequence.size();
  const size_t bitsize = static_cast<size_t>(log2(n));
  assert(1UL << bitsize == n);

  // Call BitReverse on all elements of input_sequence, then copy the values
  // into output_sequence.
#pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
    output_sequence[i] = input_sequence[NaiveBitReverse(i, bitsize)];
  }
}

// Call MaskBitReverse on all elements of "input_sequence".
void MaskBitReversalPermutationAlgorithm::operator()(
    const vec &input_sequence, vec &output_sequence) const {
  // Get the size and bitsize of the sequence, then perform a sanity check.
  const size_t n = input_sequence.size();
  const size_t bitsize = static_cast<size_t>(log2(n));
  assert(1UL << bitsize == n);

  // Call BitReverse on all elements on the sequence.
  // While an alternative would be looping up to n/2 and swapping two values
  // each iteration, requiring half the calls to MaskBitReverse, the alternative
  // is actually slower in practice.
#pragma omp parallel for firstprivate(n, bitsize) schedule(static)
  for (size_t i = 0; i < n; i++) {
    output_sequence[i] = input_sequence[MaskBitReverse(i, bitsize)];
  }
}

// Perform BitReversalPermutation of the entire sequence in O(n) by reusing
// computed values.
void FastBitReversalPermutationAlgorithm::operator()(
    const vec &input_sequence, vec &output_sequence) const {
  // Get the size and bitsize of the sequence, then perform a sanity check.
  const size_t num_elements = input_sequence.size();
  const size_t bitsize = static_cast<size_t>(log2(num_elements));
  assert(1UL << bitsize == num_elements);

  // Base case: the sequence is 0 or 1 elements long.
  if (num_elements < 2) {
    output_sequence = vec(input_sequence);
  }

  // Initialize a temporary vector.
  std::vector<size_t> coefficients(num_elements, 0);

  // Set the values for the first 2 indices.
  coefficients[1] = 1;
  output_sequence[0] = input_sequence[0];
  output_sequence[1] = input_sequence[num_elements >> 1];

  // For each group separated by stiffled lines, after the first one.
  for (size_t q = 1; q < bitsize; q++) {
    const size_t L = 1UL << q;
    const size_t right_zeros = bitsize - q - 1;
    const size_t group_size = L >> 1;

    // For each element of the group, calculate indices and copy values.
#pragma omp parallel for default(none)                    \
    shared(output_sequence, coefficients, input_sequence) \
    firstprivate(L, right_zeros, group_size)
    for (size_t j = 0; j < group_size; j++) {
      const size_t coeff1 = coefficients[group_size + j];
      const size_t coeff2 = coeff1 + L;
      const size_t index1 = L + (j << 1);
      const size_t index2 = index1 + 1;

      coefficients[index1] = coeff1;
      coefficients[index2] = coeff2;
      output_sequence[index1] = input_sequence[coeff1 << right_zeros];
      output_sequence[index2] = input_sequence[coeff2 << right_zeros];
    }
  }
}

// Calculate time for execution using chrono.
unsigned long BitReversalPermutationAlgorithm::calculateTime(
    const vec &input_sequence, vec &output_sequence) const {
  auto t0 = std::chrono::high_resolution_clock::now();
  this->operator()(input_sequence, output_sequence);
  auto t1 = std::chrono::high_resolution_clock::now();
  const auto time =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  return time;
}

void CompareTimesBitReversalPermutation(const vec &sequence,
                                        unsigned int max_num_threads) {
  // Calculate sequence size.
  const size_t size = sequence.size();
  unsigned long serial_mask_time = 0;
  unsigned long serial_fast_time = 0;

  // Create the output sequences.
  vec mask_result(size, 0);
  vec fast_result(size, 0);

  // Create an instance of the needed algorithms.
  MaskBitReversalPermutationAlgorithm mask_algorithm;
  FastBitReversalPermutationAlgorithm fast_algorithm;

  // For each number of threads.
  for (unsigned int num_threads = 1; num_threads <= max_num_threads;
       num_threads *= 2) {
    // Set the number of threads.
    omp_set_num_threads(num_threads);

    // Execute the mask bit reversal.
    const unsigned long mask_time =
        mask_algorithm.calculateTime(sequence, mask_result);
    if (num_threads == 1) serial_mask_time = mask_time;
    std::cout << "Time for mask bit reversal permutation with " << size
              << " elements and " << num_threads << " threads: " << mask_time
              << "μs" << std::endl;

    // Execute the fast bit reversal.
    const unsigned long fast_time =
        fast_algorithm.calculateTime(sequence, fast_result);
    if (num_threads == 1) serial_fast_time = fast_time;
    std::cout << "Time for fast bit reversal permutation with " << size
              << " elements and " << num_threads << " threads: " << fast_time
              << "μs" << std::endl;

    // Calculate and print speedups.
    std::cout << "Mask speedup: "
              << static_cast<double>(serial_mask_time) / mask_time << "x"
              << std::endl;
    std::cout << "Fast speedup: "
              << static_cast<double>(serial_fast_time) / fast_time << "x"
              << std::endl;
    std::cout << "Winner: " << (fast_time < mask_time ? "Fast" : "Mask")
              << std::endl;

    // Compare the results.
    if (!CompareVectors(mask_result, fast_result, 0, false))
      std::cout << "The methods provide different results!" << std::endl;

    std::cout << std::endl;
  }
}

}  // namespace FourierTransform
