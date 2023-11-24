#ifndef BIT_REVERSE_PERMUTATION_HPP
#define BIT_REVERSE_PERMUTATION_HPP

#include <cstddef>

// Compute the reverse bit order of "index", assuming it has "number_bits" bits
size_t BitReversePermutation(const size_t index, const size_t num_bits);

// Calculate the time needed to reverse the bits of the indexes of a sequence "number_elements" long
unsigned long CalculateTimeBitReversePermutation(size_t num_elements, unsigned int num_threads);

#endif //BIT_REVERSE_PERMUTATION_HPP
