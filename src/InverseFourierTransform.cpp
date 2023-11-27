#include "InverseFourierTransform.hpp"
#include "BitReversalPermutation.hpp"

using vec = std::vector<std::complex<real>>;

// Perform the Inverse Fourier Transform of a sequence, using the O(n^2) algorithm.
vec InverseDiscreteFourierTransform(const vec &sequence)
{
	const size_t size = sequence.size();
	vec result(size);

	// For each element in the sequence...
	for (size_t k = 0; k < size; ++k)
	{
		for (size_t n = 0; n < size; ++n)
		{
			// Compute the sum of the sequence multiplied by the nth root of unity, with step k.
			result[k] += sequence[n] * std::exp(std::complex<real>(0, 2 * M_PI * k * n / size));
		}
	}

	return result;	
}