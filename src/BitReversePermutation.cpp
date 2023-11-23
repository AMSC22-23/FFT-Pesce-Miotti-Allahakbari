#include "BitReversePermuation.hpp"

// Compute the reverse bit order of "index", assuming it has "number_bits" bits
// Note that this is an extremely naive implementation
size_t BitReversePermutation(const size_t index, const size_t number_bits) {
	size_t result = 0;
	for(size_t i=0; i<number_bits; i++) {
		result = result | (((1 << i) & index) >> i << (number_bits - i - 1));
	}
	return result;
}