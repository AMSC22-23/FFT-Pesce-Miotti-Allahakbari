#include <iostream>
#include <string>

/**
 * @file main.cpp.
 * @brief Defines the main function, which then calls one of other 4 functions
 * based on the first argument.
 */

#include "compression_main.hpp"
#include "cuda_main.hpp"
#include "fft_main.hpp"
#include "wavelet_main.hpp"

int main(int argc, char *argv[]) {
  if (argc <= 1) {
    std::cerr << "Incorrect arguments!\n"
              << "Specify the execution mode: fft (default) / compression / "
                 "cuda / wavelet"
              << std::endl;
    return 1;
  }

  // Get which mode to execute the program in.
  std::string mode = std::string(argv[1]);

  // Run a test with images.
  if (mode == std::string("compression")) {
    return compression_main(argc, argv);
  }
  // Run a test with CUDA.
  else if (mode == std::string("cuda")) {
    return cuda_main(argc, argv);
  }
  // Run a test of wavelets.
  else if (mode == std::string("wavelet")) {
    return wavelet_main(argc, argv);
  }
  // Run a FFT test.
  else if (mode == std::string("fft")) {
    return fft_main(argc, argv);
  }
  // Wrong mode.
  else {
    std::cerr << "Incorrect arguments!\n"
              << "Specify the execution mode: fft (default) / compression / "
                 "cuda / wavelet"
              << std::endl;
    return 1;
  }

  return 0;
}