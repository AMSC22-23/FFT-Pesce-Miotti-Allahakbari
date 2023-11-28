#include "InverseFourierTransform.hpp"

#include <cassert>

#include "BitReversalPermutation.hpp"

using vec = std::vector<std::complex<real>>;

// Perform the Inverse Fourier Transform of a sequence, using the O(n^2) algorithm.
vec InverseDiscreteFourierTransform(const vec &sequence)
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
			const std::complex<real> exponent = std::complex<real>{0, 2 * pi * k * m / n};
			curr_coefficient += sequence[m] * std::exp(exponent);
		}

		result.emplace_back(curr_coefficient);
	}

	return result;
}

// Perform the Inverse Fourier Transform of a sequence, using the iterative O(n log n) algorithm.
// This function is nearly identical to its direct counterpart, except for the sign of the exponent.
// Source: https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
vec InverseFastFourierTransformIterative(const vec &sequence)
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
		const std::complex<real> omega_d = std::exp(std::complex<real>{0, 2 * pi / m});
		
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