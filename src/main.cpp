#include <iostream>
#include "FourierTransform.hpp"
#include "BitReversePermutation.hpp"
#include "VectorExporter.hpp"

using vec = std::vector<std::complex<real>>;

int main(int argc, char* argv[])
{
	// Check the number of arguments.
	if (argc > 2) {
		std::cerr << "Too many arguments." << std::endl;
		std::cerr << "Argument 1: size of the sequence (default: 8)" << std::endl;
		return 1;
	}

	// Get the size of the sequence.
	size_t size = 8;
	if (argc == 2) size = atoi(argv[1]);

	// Generating a sequence of complex numbers.
	vec sequence;
	sequence.reserve(size);

	for (size_t i = 0; i < size; i++)
	{
		// Add a random complex number to the sequence.
		sequence.emplace_back(rand() % 100, rand() % 100);
	} 	

	// Compute the O(n^2) Fourier Transform of the sequence.
	vec dft_result = DiscreteFourierTransform(sequence);

	// Compute the O(n log n) Fourier Transform of the sequence with the recursive algorithm.
	vec fft_recursive_result = FastFourierTransformRecursive(sequence);
	
	// Compute the O(n log n) Fourier Transform of the sequence with the iterative algorithm.
	vec fft_iterative_result = FastFourierTransformIterative(sequence);

	// Check the results for errors.
	if (!CompareResult(dft_result, fft_recursive_result, 1e-4, false)) std::cerr << "Errors detected in recursive FFT." << std::endl;
	else if (!CompareResult(dft_result, fft_iterative_result, 1e-4, false)) std::cerr << "Errors detected in iterative FFT." << std::endl;
	else std::cerr << "No errors detected." << std::endl;

	// Write needed results to files, to be plotted with Python.
	WriteToFile(sequence, "sequence.csv");
	WriteToFile(fft_iterative_result, "result.csv");	

	// Bit permutation and OpenMP test.
	// Suggested value: 1L << 27
	size_t num_elements = 1UL << 20;
	
	unsigned int max_num_threads = 8;
	for(unsigned int num_threads=1; num_threads<=max_num_threads; num_threads *= 2)
	{
		std::cout << "Time for bit permutation with " << num_elements << " elements and " << num_threads << " threads: ";
		std::cout << CalculateTimeBitReversePermutation(num_elements, num_threads) << "μs" << std::endl;
	}
	
	return 0;
}
