#include "BitReversePermutation.hpp"
#include <chrono>
#include <tgmath.h>
#include <omp.h>
#include <iostream>

// Compute the reverse bit order of "index", given a bitsize.
// Note that this is an extremely naive implementation.
size_t BitReversePermutation(const size_t index, const size_t bitsize) {
	// Initialize the result.
	size_t result = 0;

	for(size_t i = 0; i < bitsize; i++) {
		// Set the i-th bit of the index to the same position, from the top.
		result |= ((1 << i) & index) >> i << (bitsize - i - 1);
	}

	return result;
}

// Calculate the time needed to reverse the bits of the indexes of a sequence
// of size "number_elements", expressed in microseconds.
unsigned long CalculateTimeBitReversePermutation(size_t num_elements, unsigned int num_threads, size_t &O3_workaround) {
	// Calculate the necessary number of bits to store the sequence.
	size_t num_bits = static_cast<size_t>(log2(num_elements));
	if (1 << num_bits != num_elements) num_bits++;
	
	// Set the number of threads.
	omp_set_num_threads(num_threads);
	
	// Apply the function to all elements.
	const auto t0 = std::chrono::high_resolution_clock::now();
	
	// OpenMP parallel for loop.
	#pragma omp parallel for
	for (size_t i=0; i<num_elements; i++)
	{
		O3_workaround = BitReversePermutation(i, num_bits);
	}

	// Return the time needed to reverse the bits of the indexes.
	const auto t1 = std::chrono::high_resolution_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

// A wrapper around the previous function to avoid O3 optimazation removing the for loop
unsigned long CalculateTimeBitReversePermutation(size_t num_elements, unsigned int num_threads) {
	size_t O3_workaround;

	return CalculateTimeBitReversePermutation(num_elements, num_threads, O3_workaround);
}
