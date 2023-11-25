#ifndef BIT_REVERSAL_PERMUTATION_HPP
#define BIT_REVERSAL_PERMUTATION_HPP

#include <vector>
#include <complex>
#include "Real.hpp"

// Compute the permutation of "sequence" in which elements at index i and rev(i) are swapped
// where rev(i) is obtained from i by considering it as a log2(sequence.size())-bit word and reversing the bit order.
// "sequence" must have a power of 2 elements.
// O(n*log(n)) algorithm.
std::vector<std::complex<real>> BitReversalPermutation(const std::vector<std::complex<real>> &sequence);

// Compute the permutation of "sequence" in which elements at index i and rev(i) are swapped
// where rev(i) is obtained from i by considering it as a log2(sequence.size())-bit word and reversing the bit order.
// "sequence" must have a power of 2 elements.
// O(n) algorithm, adapted from: https://folk.idi.ntnu.no/elster/pubs/elster-bit-rev-1989.pdf
std::vector<std::complex<real>> FastBitReversalPermutation(const std::vector<std::complex<real>> &sequence);

// Calculate the time needed to compute "BitReversalPermutation(sequence)" and "FastBitReversalPermutation(sequence)" 
// with 1 to "max_num_threads" threads in microseconds, compare the output sequences print the results.
void TimeEstimateBitReversalPermutation(const std::vector<std::complex<real>> &sequence, unsigned int max_num_threads);

#endif //BIT_REVERSAL_PERMUTATION_HPP