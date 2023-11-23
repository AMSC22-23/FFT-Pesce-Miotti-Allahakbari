#include "FourierTransform.hpp"
#include "BitReversePermutation.hpp"
#include <tgmath.h>
#include <cassert>
#include <iostream>

// Perform the Fourier Transform of a sequence, using the O(n^2) algorithm
std::vector<std::complex<real>> DiscreteFourierTransform(const std::vector<std::complex<real>> &sequence)
{
	// Defining some useful aliases
	constexpr real pi = std::numbers::pi_v<real>;
	const size_t n = sequence.size();

	// Initializing the result vector
	std::vector<std::complex<real>> result;
	result.reserve(n);

	// Main loop: looping over result coefficients
	for (size_t k = 0; k < n; k++)
	{
		std::complex<real> curr_coefficient = 0.0;

		// Internal loop: looping over input coefficients for a set result position
		for (size_t m = 0; m < n; m++)
		{
			const std::complex<real> exponent = std::complex<real>{0, -2 * pi * k * m / n};
			curr_coefficient += sequence[m] * std::exp(exponent);
		}

		result.emplace_back(curr_coefficient);
	}

	return result;
}

// Perform the Fourier Transform of a sequence, using the O(n log n) algorithm
// Note that this particular implementation uses recursion, which was discouraged in the assignment
// Source: https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
std::vector<std::complex<real>> FastFourierTransformRecursive(const std::vector<std::complex<real>> &sequence)
{
	// Defining some useful aliases
	constexpr real pi = std::numbers::pi_v<real>;
	const size_t n = sequence.size();

	// Trivial case: if the sequence is of length 1, return it
	if (n == 1) return sequence; 

	// Splitting the sequence into two halves
	std::vector<std::complex<real>> even_sequence;
	std::vector<std::complex<real>> odd_sequence;

	for (size_t i = 0; i < n; i++)
	{
		if (i % 2 == 0) even_sequence.emplace_back(sequence[i]); 
		else odd_sequence.emplace_back(sequence[i]); 
	}

	// Recursively computing the Fourier Transform of the two halves
	std::vector<std::complex<real>> even_result = FastFourierTransformRecursive(even_sequence);
	std::vector<std::complex<real>> odd_result = FastFourierTransformRecursive(odd_sequence);

	// Combining the two results
	std::vector<std::complex<real>> result;
	result.reserve(n);

	// Dummy fill the result vector 
	for (size_t k = 0; k < n; k++)
	{
		result.emplace_back(0);
	}

	// Implementing the Cooley-Tukey algorithm
	for (size_t k = 0; k < n / 2; k++)
	{
		std::complex<real> p = even_result[k];
		std::complex<real> q = std::exp(std::complex<real>{0, -2 * pi * k / n}) * odd_result[k];

		result[k] = p + q;
		result[k + n / 2] = p - q;
	}

	return result;
}

// Perform the Fourier Transform of a sequence, using the O(n log n) algorithm
// This is the iterative implementation
// Source: https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
std::vector<std::complex<real>> FastFourierTransformIterative(const std::vector<std::complex<real>> &sequence)
{
	// Defining some useful aliases
	constexpr real pi = std::numbers::pi_v<real>;
	const size_t n = sequence.size();
	
	// Performing a sanity check: "sequence" should have a power of 2 elements
	const size_t log_n = static_cast<size_t>(log2(n));
	assert(1 << log_n == n);
	
	// Initialization of output sequence
	std::vector<std::complex<real>> result;
	result.reserve(n);
	
	// Bit reverse permutation of "sequence"
	for (size_t i = 0; i < n; i++) 
	{
		result.emplace_back(sequence[BitReversePermutation(i, log_n)]);
	}
	
	// Main loop: looping over the binary tree layers
	for (size_t s = 1; s <= log_n; s++)
	{
		const size_t m = 1 << s;
		const std::complex<real> omega_d = std::exp(std::complex<real>{0, -2 * pi / m});
		
		for (size_t k = 0; k < n; k += m) 
		{
			std::complex<real> omega{1, 0};
			
			for (size_t j = 0; j < m/2; j++) 
			{
				const std::complex<real> t = omega * result[k + j + m/2];
				const std::complex<real> u = result[k + j];
				result[k + j] = u + t;
				result[k + j + m/2] = u - t;
				omega *= omega_d;
			}
		}
	}
	
	return result;
}

// Compare the values of "sequence" with those of "sequence_golden" and return if they have the same elements; if "print_errors" is true, print the errors in "sequence"
bool CompareResult(const std::vector<std::complex<real>> &sequence_golden, const std::vector<std::complex<real>> &sequence, double precision, bool print_errors) 
{
	if (sequence_golden.size() != sequence.size()) {
		if (print_errors) std::cout << "The sequences have different lengths!" << std::endl;
		return false;
	}
	
	std::vector<std::complex<real>> errors;
	for (size_t i = 0; i < sequence_golden.size(); i++)
	{
		if (abs(sequence[i] - sequence_golden[i]) > precision) {
			if (!print_errors) return false;
			errors.emplace_back(i);
		}
	}
	
	if (errors.size() == 0) return true;
	
	std::cout << "Errors at indexes: ";
	for (size_t i = 0; i < errors.size(); i++)
	{
		std::cout << errors[i] << " ";
	}
	std::cout << std::endl;
	return false;
}
