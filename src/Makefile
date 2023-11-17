build:
	g++ -std=c++20 -g -Wall -Wextra main.cpp FourierTransform.cpp -DDOUBLE -o main 
	
optimize:
	g++ -std=c++20 -Wall -Wextra -O3 -Ofast -march=native -funroll-all-loops main.cpp FourierTransform.cpp -DDOUBLE -o main	

clean:
	rm main
