#include <iostream>

#include "FourierTransform.hpp"
#include "BitReversalPermutation.hpp"
#include "VectorExporter.hpp"
#include "Utility.hpp"

using vec = std::vector<std::complex<real>>;

int main(int argc, char* argv[])
{
	// Set a default size for the sequence.
	size_t size = 1UL << 10;

	// Check the number of arguments.
	if (argc > 2) {
		std::cerr << "Too many arguments." << std::endl;
		std::cerr << "Argument 1: size of the sequence (default: " << size << ")" << std::endl;
		return 1;
	}

	// Get the size of the sequence.
	if (argc == 2) size = atoi(argv[1]);

	// Generating a sequence of complex numbers.
	vec sequence;
	sequence.reserve(size);

	for (size_t i = 0; i < size; i++)
	{
		// Add a random complex number to the sequence.
		sequence.emplace_back(rand() % 100, rand() % 100);
	} 	

	// Save the sequence to a file.
	WriteToFile(sequence, "sequence.csv");

	// Compute the O(n^2) Fourier Transform of the sequence.
	vec dft_result = DiscreteFourierTransform(sequence);
	WriteToFile(dft_result, "dft_result.csv");

	// Compute the O(n log n) Fourier Transform of the sequence with the recursive algorithm.
	vec fft_recursive_result = FastFourierTransformRecursive(sequence);
	WriteToFile(fft_recursive_result, "fft_recursive_result.csv");

	// Compute the O(n log n) Fourier Transform of the sequence with the iterative algorithm.
	vec fft_iterative_result = FastFourierTransformIterative(sequence);
	WriteToFile(fft_iterative_result, "fft_iterative_result.csv");

	// Check the results for errors.
	std::cout << std::endl;
	if (!CompareResult(dft_result, fft_recursive_result, 1e-4, false)) std::cerr << "Errors detected in recursive FFT." << std::endl;
	else if (!CompareResult(dft_result, fft_iterative_result, 1e-4, false)) std::cerr << "Errors detected in iterative FFT." << std::endl;
	else std::cout << "No errors detected." << std::endl << std::endl;

	// Bit permutation and OpenMP test, recommended sequence size: 1UL << 27.
	TimeEstimateBitReversalPermutation(sequence, 8);
	
	return 0;
}
