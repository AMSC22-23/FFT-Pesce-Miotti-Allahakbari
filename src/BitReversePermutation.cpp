#include "BitReversePermutation.hpp"
#include <chrono>
#include <tgmath.h>
#include <omp.h>
#include <iostream>

// Compute the reverse bit order of "index", assuming it has "number_bits" bits
// Note that this is an extremely naive implementation
size_t BitReversePermutation(const size_t index, const size_t num_bits) {
	size_t result = 0;
	for(size_t i=0; i<num_bits; i++) {
		result = result | (((1 << i) & index) >> i << (num_bits - i - 1));
	}
	return result;
}

// Calculate the time needed to reverse the bits of the indexes of a sequence "number_elements" long
unsigned long CalculateTimeBitReversePermutation(size_t num_elements, unsigned int num_threads, size_t &result) {
	// Calculate the necessary number of bits to store the sequence
	size_t num_bits = static_cast<size_t>(log2(num_elements));
	if (1 << num_bits != num_elements) num_bits++;
	
	// Set the number of threads
	omp_set_num_threads(num_threads);
	
	// Apply the function to all elements
	const auto t0 = std::chrono::high_resolution_clock::now();
	#pragma omp parallel for
	for (size_t i=0; i<num_elements; i++)
	{
		result = BitReversePermutation(i, num_bits);
	}
	const auto t1 = std::chrono::high_resolution_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

// A wrapper around the previous function to avoid O3 optimazation removing the for loop
unsigned long CalculateTimeBitReversePermutation(size_t num_elements, unsigned int num_threads) {
	size_t result;

	return CalculateTimeBitReversePermutation(num_elements, num_threads, result);
}
