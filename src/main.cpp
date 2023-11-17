#include <iostream>
#include "FourierTransform.hpp"

int main(int argc, char* argv[]) {
	if(argc == 1) {
		std::cout << "Hello world!" << std::endl;
	} else if(argc == 2) {
		std::cout << "Hello " << argv[1] << "!" << std::endl;
	} else {
		std::cerr << "Too many arguments!" << std::endl;
	}
	
	/*
	//A simple test on computing the Fourier Transform with the naive method
	//Results correspond to those obtained running numpy.fft.fft([1,2,3]) in python
	std::vector<std::complex<real>> example_sequence = {std::complex<real>{1.0,0.0}, std::complex<real>{2.0,0.0}, std::complex<real>{3.0,0.0}};
	std::vector<std::complex<real>> output_sequence = FourierTransform(example_sequence);
	for(size_t i=0; i<output_sequence.size(); i++) {
		std::cout << output_sequence[i] << " ";
	}
	std::cout << std::endl;
	*/
	
	return 0;
}
