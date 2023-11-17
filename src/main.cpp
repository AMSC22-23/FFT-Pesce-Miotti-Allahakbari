#include <iostream>
#include "FourierTransform.hpp"

int main(int argc, char* argv[])
{
	// Testing the argument parsing
	if (argc == 1) { std::cout << "No arguments detected." << std::endl; } 
	else if (argc == 2) { std::cout << "Argument detected: " << argv[1] << "." << std::endl; }
	else { std::cerr << "Too many arguments." << std::endl; }

	// Set a size for the sequence
	size_t size = 8;

	// Generating a sequence of complex numbers
	std::vector<std::complex<real>> sequence;
	sequence.reserve(size);

	std::cout << std::endl << "Initial sequence:" << std::endl;
	for (size_t i = 0; i < size; i++)
	{
		// Add a random complex number to the sequence
		sequence.emplace_back(rand() % 100, rand() % 100);

		// Print the sequence
		std::cout << sequence[i] << " ";
	} 	std::cout << std::endl;

	// Compute the O(n^2) Fourier Transform of the sequence
	std::vector<std::complex<real>> dft_result = DiscreteFourierTransform(sequence);

	// Compute the O(n log n) Fourier Transform of the sequence
	std::vector<std::complex<real>> fft_result = FastFourierTransform(sequence);

	// Print the results
	std::cout << std::endl << "DFT result:" << std::endl;
	for (size_t i = 0; i < dft_result.size(); i++)
	{
		std::cout << dft_result[i] << " ";
	}	std::cout << std::endl;

	std::cout << std::endl << "FFT result:" << std::endl;
	for (size_t i = 0; i < fft_result.size(); i++)
	{
		std::cout << fft_result[i] << " ";
	}	std::cout << std::endl;
	
	return 0;
}
