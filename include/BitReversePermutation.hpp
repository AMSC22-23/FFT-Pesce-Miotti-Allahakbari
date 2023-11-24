#ifndef BIT_REVERSE_PERMUTATION_HPP
#define BIT_REVERSE_PERMUTATION_HPP

#include <cstddef>

// Compute the reverse bit order of "index", given a bitsize.
size_t BitReversePermutation(const size_t index, const size_t bitsize);

// Calculate the time needed to reverse the bits of the indexes of a sequence
// of size "number_elements", expressed in microseconds.
unsigned long CalculateTimeBitReversePermutation(size_t num_elements, unsigned int num_threads);

#endif //BIT_REVERSE_PERMUTATION_HPP
