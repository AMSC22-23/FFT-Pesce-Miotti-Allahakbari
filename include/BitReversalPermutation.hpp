#ifndef BIT_REVERSAL_PERMUTATION_HPP
#define BIT_REVERSAL_PERMUTATION_HPP

#include "Real.hpp"

/**
 * @file BitReversalPermutation.hpp.
 * @brief Declares Bit Reversal Permutation algorithms.
 */

namespace Transform {
namespace FourierTransform {

/**
 * @brief Represents an abstract Bit Reversal Permutation algorithm.
 *
 * The BitReversalPermutationAlgorithm abstract class is designed as a common
 * interface for bit reversal permutation algorithms. It allows for execution of
 * the algorithm and computation of execution time.
 *
 * Example usage:
 * @code
 * std::unique_ptr<BitReversalPermutationAlgorithm> algorithm;
 *
 * ...
 *
 * algorithm(input_sequence, output_sequence);
 * @endcode
 */
class BitReversalPermutationAlgorithm {
 public:
  /**
   * @brief Perform a bit reversal permutation of input_sequence into
   * output_sequence.
   *
   * Store in output_sequence a permutation of input_sequence, s.t.
   * output_sequence[i] = input_sequence[rev(i)], where rev(i) is the value
   * obtained reversing the order of the bits in the bit representation of i.
   * The bit representation of i refers to an unsigned integer representation,
   * using the lowest number of bits required to store any index in the
   * sequence.
   *
   * @param input_sequence The input sequence.
   * @param output_sequence The output sequence.
   *
   * @note input_sequence and output_sequence must have the same length n, which
   * must be a power of 2.
   */
  virtual void operator()(const vec &input_sequence,
                          vec &output_sequence) const = 0;
  /**
   * @brief Execute the algorithm and return the execution time in microseconds.
   *
   * @param input_sequence The input sequence for execution.
   * @param output_sequence The output sequence for execution.
   *
   * @note input_sequence and output_sequence must be the same length, which
   * must be a power of 2.
   */
  unsigned long calculateTime(const vec &input_sequence,
                              vec &output_sequence) const;
  virtual ~BitReversalPermutationAlgorithm() = default;
};

/**
 * @brief A naive implementation of BitReversalPermutationAlgorithm.
 *
 * @note The algorithm has time complexity O(n log(n)).
 * @note The algorithm is massively parallel.
 */
class NaiveBitReversalPermutationAlgorithm
    : public BitReversalPermutationAlgorithm {
 public:
  void operator()(const vec &input_sequence,
                  vec &output_sequence) const override;
  ~NaiveBitReversalPermutationAlgorithm() = default;
};

/**
 * @brief An efficient implementation of BitReversalPermutationAlgorithm.
 *
 * @note The algorithm has time complexity O(n logn), with a small constant.
 * @note The algorithm is massively parallel.
 * @note Adapted from @cite
 * https://rosettacode.org/wiki/Fast_Fourier_transform#C.2B.2B.
 */
class MaskBitReversalPermutationAlgorithm
    : public BitReversalPermutationAlgorithm {
 public:
  void operator()(const vec &input_sequence,
                  vec &output_sequence) const override;
  ~MaskBitReversalPermutationAlgorithm() = default;
};

/**
 * @brief An asynthotically efficient implementation of
 * BitReversalPermutationAlgorithm.
 *
 * @note The algorithm has time complexity O(n).
 * @note The algorithm has a parallel time complexity of O(log(n)) with n
 * processors.
 * @note Adapted from @cite
 * https://rosettacode.org/wiki/Fast_Fourier_transform#C.2B.2B.
 */
class FastBitReversalPermutationAlgorithm
    : public BitReversalPermutationAlgorithm {
 public:
  void operator()(const vec &input_sequence,
                  vec &output_sequence) const override;
  ~FastBitReversalPermutationAlgorithm() = default;
};

/**
 * @brief Compare and print the execution times of
 * MaskBitReversalPermutationAlgorithm and FastBitReversalAlgorithm.
 *
 * Calculate the time needed to compute the bit reversal permutation of
 * sequence using an instance of MaskBitReversalPermutationAlgorithm and one of
 * FastBitReversalAlgorithm, using powers of 2 threads ranging from 1 to
 * max_num_threads. Execution times in microseconds are printed. Output
 * sequences are compared for differences to check for errors.
 *
 * @param sequence The sequence to use for comparison.
 * @param max_num_threads The maximum number of threads for comparisons.
 */
void CompareBitReversalPermutationTimes(const vec &sequence,
                                        unsigned int max_num_threads);

}  // namespace FourierTransform
}  // namespace Transform

#endif  // BIT_REVERSAL_PERMUTATION_HPP