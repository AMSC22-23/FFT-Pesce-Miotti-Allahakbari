#include <cassert>
#include <iostream>

#include <tgmath.h>

#include "FourierTransform.hpp"
#include "BitReversalPermutation.hpp"

using vec = std::vector<std::complex<real>>;

// Perform the Fourier Transform of a sequence, using the O(n^2) algorithm.
vec DiscreteFourierTransform(const vec &sequence)
{
	// Defining some useful aliases.
	constexpr real pi = std::numbers::pi_v<real>;
	const size_t n = sequence.size();

	// Initializing the result vector.
	vec result;
	result.reserve(n);

	// Main loop: looping over result coefficients.
	for (size_t k = 0; k < n; k++)
	{
		std::complex<real> curr_coefficient = 0.0;

		// Internal loop: looping over input coefficients for a set result position.
		for (size_t m = 0; m < n; m++)
		{
			const std::complex<real> exponent = std::complex<real>{0, -2 * pi * k * m / n};
			curr_coefficient += sequence[m] * std::exp(exponent);
		}

		result.emplace_back(curr_coefficient);
	}

	return result;
}

// Perform the Fourier Transform of a sequence, using the O(n log n) algorithm.
// Source: https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
vec FastFourierTransformRecursive(const vec &sequence)
{
	// Defining some useful aliases.
	constexpr real pi = std::numbers::pi_v<real>;
	const size_t n = sequence.size();

	// Trivial case: if the sequence is of length 1, return it.
	if (n == 1) return sequence; 

	// Splitting the sequence into two halves.
	vec even_sequence;
	vec odd_sequence;

	for (size_t i = 0; i < n; i++)
	{
		if (i % 2 == 0) even_sequence.emplace_back(sequence[i]); 
		else odd_sequence.emplace_back(sequence[i]); 
	}

	// Recursively computing the Fourier Transform of the two halves.
	vec even_result = FastFourierTransformRecursive(even_sequence);
	vec odd_result = FastFourierTransformRecursive(odd_sequence);

	// Combining the two results.
	vec result(n, 0.0);

	// Implementing the Cooley-Tukey algorithm.
	for (size_t k = 0; k < n / 2; k++)
	{
		std::complex<real> p = even_result[k];
		std::complex<real> q = std::exp(std::complex<real>{0, -2 * pi * k / n}) * odd_result[k];

		result[k] = p + q;
		result[k + n / 2] = p - q;
	}

	return result;
}

// Perform the Fourier Transform of a sequence, using the iterative O(n log n) algorithm.
// Source: https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
vec FastFourierTransformIterative(const vec &sequence)
{
	// Defining some useful aliases.
	constexpr real pi = std::numbers::pi_v<real>;
	const size_t n = sequence.size();
	
	// Check that the size of the sequence is a power of 2.
	const size_t log_n = static_cast<size_t>(log2(n));
	assert(1UL << log_n == n);
	
	// Initialization of output sequence.
	vec result = FastBitReversalPermutation(sequence);
	
	// Main loop: looping over the binary tree layers.
	for (size_t s = 1; s <= log_n; s++)
	{
		const size_t m = 1UL << s;
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