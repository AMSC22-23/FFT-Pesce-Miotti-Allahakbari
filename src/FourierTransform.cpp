#include "FourierTransform.hpp"

// Perform the Fourier Transform of a sequence, using the O(n^2) algorithm
std::vector<std::complex<real>> DiscreteFourierTransform(const std::vector<std::complex<real>> &sequence)
{
	// Defining some useful aliases
	constexpr std::complex<real> imm{0, 1};
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
std::vector<std::complex<real>> FastFourierTransform(const std::vector<std::complex<real>> &sequence)
{
	// Defining some useful aliases.
	constexpr std::complex<real> imm{0, 1};
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
	std::vector<std::complex<real>> even_result = FastFourierTransform(even_sequence);
	std::vector<std::complex<real>> odd_result = FastFourierTransform(odd_sequence);

	// Combining the two results
	std::vector<std::complex<real>> result;
	result.reserve(n);

	// Fill vector with the even results
	for (size_t k = 0; k < n / 2; k++)
	{
		result.emplace_back(even_result[k]);
	}

	// Fill vector with the odd results
	for (size_t k = 0; k < n / 2; k++)
	{
		result.emplace_back(odd_result[k]);
	}

	// Implementing the Cooley-Tukey algorithm
	// Source: https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
	for (size_t k = 0; k < n / 2; k++)
	{
		std::complex<real> p = result[k];
		std::complex<real> q = std::exp(std::complex<real>{0, -2 * pi * k / n}) * result[k + n / 2];

		result[k] = p + q;
		result[k + n / 2] = p - q;
	}

	return result;
}