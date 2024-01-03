#ifndef BIT_REVERSAL_PERMUTATION_HPP
#define BIT_REVERSAL_PERMUTATION_HPP

#include "Real.hpp"

namespace Transform {
namespace FourierTransform {

// A class that represents a generic bit reversal algorithm. Calling the
// operator () on (input, output) stores the bit reversal permutation of input
// into output. input must have a power of 2 elements.
class BitReversalPermutationAlgorithm {
 public:
  virtual void operator()(const vec &input_sequence,
                          vec &output_sequence) const = 0;
  // Execute the algorithm and return the time taken in microseconds.
  unsigned long calculateTime(const vec &input_sequence,
                              vec &output_sequence) const;
  virtual ~BitReversalPermutationAlgorithm() = default;
};

// A naive implementation of the algorithm in O(n logn). Massively parallel.
class NaiveBitReversalPermutationAlgorithm
    : public BitReversalPermutationAlgorithm {
 public:
  void operator()(const vec &input_sequence,
                  vec &output_sequence) const override;
  ~NaiveBitReversalPermutationAlgorithm() = default;
};

// An efficient implementation of the algorithm in O(n logn) with a very small
// constant. Massively parallel. Source:
// https://rosettacode.org/wiki/Fast_Fourier_transform#C.2B.2B.
class MaskBitReversalPermutationAlgorithm
    : public BitReversalPermutationAlgorithm {
 public:
  void operator()(const vec &input_sequence,
                  vec &output_sequence) const override;
  ~MaskBitReversalPermutationAlgorithm() = default;
};

// An implementation of the algorithm in O(n), adapted from:
// https://folk.idi.ntnu.no/elster/pubs/elster-bit-rev-1989.pdf.
class FastBitReversalPermutationAlgorithm
    : public BitReversalPermutationAlgorithm {
 public:
  void operator()(const vec &input_sequence,
                  vec &output_sequence) const override;
  ~FastBitReversalPermutationAlgorithm() = default;
};

// Calculate the time needed to compute the bit reversal permutation of
// "sequence" using an instance of MaskBitReversalPermutationAlgorithm and one
// of FastBitReversalAlgorithm, with 1 to "max_num_threads" threads. The results
// are printed expressed in microseconds and the output sequences are compared
// for differences.
void CompareTimesBitReversalPermutation(const vec &sequence,
                                        unsigned int max_num_threads);

}  // namespace FourierTransform
}  // namespace Transform

#endif  // BIT_REVERSAL_PERMUTATION_HPP