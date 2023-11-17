#include <iostream>
#include "FourierTransform.hpp"

int main(int argc, char* argv[]) {
	// Testing the argument parsing
	if (argc == 1) { std::cout << "No arguments detected." << std::endl; } 
	else if (argc == 2) { std::cout << "Argument detected: " << argv[1] << std::endl; }
	else { std::cerr << "Too many arguments!" << std::endl; }
	
	// Generating a sequence of complex numbers
	std::vector<std::complex<real>> sequence;
	sequence.reserve(16);

	for (size_t i = 0; i < 16; i++)
	{
		// Add a random complex number to the sequence
		sequence.emplace_back(rand() % 100, rand() % 100);

		// Print the sequence
		std::cout << sequence[i] << std::endl;
	}

	// Compute the O(n^2) Fourier Transform of the sequence
	std::vector<std::complex<real>> dft_result = FourierTransform(sequence);

	// Print the results
	std::cout << "DFT result:" << std::endl;
	for (size_t i = 0; i < dft_result.size(); i++)
	{
		std::cout << dft_result[i] << std::endl;
	}
	
	return 0;
}
