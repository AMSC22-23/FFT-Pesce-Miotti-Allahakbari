#include <iostream>

int main(int argc, char* argv[]) {
	if(argc == 1) {
		std::cout << "Hello world!" << std::endl;
	} else if(argc == 2) {
		std::cout << "Hello " << argv[1] << "!" << std::endl;
	} else {
		std::cerr << "Too many arguments!" << std::endl;
	}
	
	return 0;
}
