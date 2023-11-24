#include <iostream>
#include "FourierTransform.hpp"
#include "BitReversePermutation.hpp"
#include "VectorExporter.hpp"

int main(int argc, char* argv[])
{
	// Testing the argument parsing
	if (argc != 1) { std::cerr << "Too many arguments." << std::endl; }

	// Set a size for the sequence
	size_t size = 8;

	// Generating a sequence of complex numbers
	std::vector<std::complex<real>> sequence;
	sequence.reserve(size);

	for (size_t i = 0; i < size; i++)
	{
		// Add a random complex number to the sequence
		sequence.emplace_back(rand() % 100, rand() % 100);
	} 	

	// Compute the O(n^2) Fourier Transform of the sequence
	std::vector<std::complex<real>> dft_result = DiscreteFourierTransform(sequence);

	// Compute the O(n log n) Fourier Transform of the sequence with the recursive algorithm
	std::vector<std::complex<real>> fft_recursive_result = FastFourierTransformRecursive(sequence);
	
	// Compute the O(n log n) Fourier Transform of the sequence with the iterative algorithm
	std::vector<std::complex<real>> fft_iterative_result = FastFourierTransformIterative(sequence);

	// Check the results
	if (!CompareResult(dft_result, fft_recursive_result, 1e-4, false)) std::cerr << "Errors detected in recursive FFT." << std::endl;
	else if (!CompareResult(dft_result, fft_iterative_result, 1e-4, false)) std::cerr << "Errors detected in iterative FFT." << std::endl;
	else std::cerr << "No errors detected." << std::endl;

	// Write needed results to files
	WriteToFile(sequence, "sequence.csv");
	WriteToFile(fft_iterative_result, "result.csv");	

	// Bit permutation and OpenMP test: 1L << 27 is good value for testing on my pc
	size_t num_elements = 1L << 20;
	
	unsigned int max_num_threads = 8;
	for(unsigned int num_threads=1; num_threads<=max_num_threads; num_threads *= 2)
	{
		std::cout << "Time for bit permutation with " << num_elements << " elements and " << num_threads << " threads: " << CalculateTimeBitReversePermutation(num_elements, num_threads) << "us" << std::endl;
	}
	
	return 0;
}
