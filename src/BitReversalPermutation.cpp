#include <chrono>
#include <iostream>
#include <cassert>

#include <tgmath.h>
#include <omp.h>

#include "BitReversalPermutation.hpp"
#include "Utility.hpp"

using vec = std::vector<std::complex<real>>;

// Compute the reverse bit order of "index", given a bitsize.
size_t BitReverse(const size_t index, const size_t bitsize) 
{
	// Initialize the result.
	size_t result = 0;

	for(size_t i = 0; i < bitsize; i++) {
		// Set the i-th bit of the index to the same position, from the top.
		result |= ((1 << i) & index) >> i << (bitsize - i - 1);
	}

	return result;
}

// Compute the permutation of a sequence in which elements at index i and rev(i) are swapped,
// where rev(i) is obtained from i by considering it as a log2(sequence.size())-bit word and reversing the bit order.
// Note that sequence.size() must be a power of 2.
// O(n*log(n)) algorithm.
vec BitReversalPermutation(const vec &sequence) 
{
	// Get the size and bitsize of the sequence, then perform a sanity check.
	const size_t n = sequence.size();
	const size_t bitsize = static_cast<size_t>(log2(n));
	assert(1UL << bitsize == n);
	
	// Initialize the result vector.
	vec result(n,0);
	
	// Call BitReverse on all elements on the sequence.
	#pragma omp parallel for
	for (size_t i=0; i<n; i++)
	{
		result[i] = sequence[BitReverse(i, bitsize)];
	}
	return result;
}

// Compute the permutation of a sequence in which elements at index i and rev(i) are swapped,
// where rev(i) is obtained from i by considering it as a log2(sequence.size())-bit word and reversing the bit order.
// Note that sequence.size() must be a power of 2.
// O(n) algorithm, adapted from: https://folk.idi.ntnu.no/elster/pubs/elster-bit-rev-1989.pdf
vec FastBitReversalPermutation(const vec &sequence) 
{
	// Get the size and bitsize of the sequence, then perform a sanity check.
	const size_t num_elements = sequence.size();
	const size_t bitsize = static_cast<size_t>(log2(num_elements));
	assert(1UL << bitsize == num_elements);

	// Base case: the sequence is 0 or 1 elements long.
	if(num_elements < 2) {
		return vec(sequence);
	}
	
	// Initialize the output and a temporary vector.
	std::vector<size_t> coefficients(num_elements,0);
	vec result(num_elements,0);
	
	// Set the values for the first 2 indices.
	coefficients[1] = 1;
	result[0] = sequence[0];
	result[1] = sequence[num_elements >> 1];
	
	// For each group separated by stiffled lines, after the first one.
	for (size_t q=1; q<bitsize; q++)
	{
		const size_t L = 1UL << q;
		const size_t right_zeros = bitsize - q - 1;
		const size_t group_size = L >> 1;
		
		// For each element of the group, calculate indices and copy values.
		#pragma omp parallel for default(none) shared(result, coefficients, sequence) firstprivate(L, right_zeros, group_size)
		for (size_t j=0; j<group_size; j++)
		{
			const size_t coeff1 = coefficients[group_size + j];
			const size_t coeff2 = coeff1 + L;
			const size_t index1 = L + (j << 1);
			const size_t index2 = index1 + 1;

			coefficients[index1] = coeff1;
			coefficients[index2] = coeff2;
			result[index1] = sequence[coeff1 << right_zeros];
			result[index2] = sequence[coeff2 << right_zeros];
		}
	}

	return result;
}

// Calculate the time needed to compute "BitReversalPermutation(sequence)" and "FastBitReversalPermutation(sequence)" 
// with 1 to "max_num_threads" threads, expressed in microseconds, compare the output sequences, and print the results.
void TimeEstimateBitReversalPermutation(const vec &sequence, unsigned int max_num_threads) 
{
	// Calculate sequence size.
	const size_t size = sequence.size();
	unsigned long serial_standard_time = 0;
	unsigned long serial_fast_time = 0;

	// For each thread number.
	for(unsigned int num_threads=1; num_threads<=max_num_threads; num_threads *= 2)
	{
		// Set the number of threads.
		omp_set_num_threads(num_threads);

		// Execute the standard bit reversal.
		auto t0 = std::chrono::high_resolution_clock::now();
		const vec standard_result = BitReversalPermutation(sequence);
		auto t1 = std::chrono::high_resolution_clock::now();
		const auto standard_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
		if (num_threads == 1) serial_standard_time = standard_time;
		std::cout << "Time for standard bit reversal permutation with " << size << " elements and " << num_threads << " threads: " << standard_time << "μs" << std::endl;

		// Execute the fast bit reversal.
		t0 = std::chrono::high_resolution_clock::now();
		const vec fast_result = FastBitReversalPermutation(sequence);
		t1 = std::chrono::high_resolution_clock::now();
		const auto fast_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
		if (num_threads == 1) serial_fast_time = fast_time;
		std::cout << "Time for fast bit reversal permutation with " << size << " elements and " << num_threads << " threads: " << fast_time << "μs" << std::endl;

		// Calculate and print speedups.
		std::cout << "Speedup over parallel standard: " << static_cast<double>(standard_time)/fast_time << "x" << std::endl;
		std::cout << "Speedup over serial standard: " << static_cast<double>(serial_standard_time)/fast_time << "x" << std::endl;
		std::cout << "Speedup over fast standard: " << static_cast<double>(serial_fast_time)/fast_time << "x" << std::endl;

		// Check the results.
		if (!CompareResult(standard_result, fast_result, 0, false)) std::cout << "The methods provide different results!" << std::endl;

		std::cout << std::endl;
	}
}