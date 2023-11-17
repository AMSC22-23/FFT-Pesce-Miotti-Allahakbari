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