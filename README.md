# FFT-Pesce-Miotti-Allahakbari

## Overview

Repository for the 2023/24 AMSC project FFT by Francesco Pesce, Michele Miotti and Mohammandhossein Allahakbari.
This program computes the Fast Fourier Transform (FFT) of a given data sequence of size `n`, where `n` is a power of 2.
This repository contains many different implementations of the FFT, which can be compared to each other in terms of performance.

## Usage

To use the program, follow these steps:

1. **Build the Program:**
   - Make sure you have CMake installed on your system.
   - Open a terminal and navigate to the project directory.
   - Run the following commands:
     ```bash
     mkdir build
     cd build
     cmake ..
     make
     ```

2. **Run the Program:**
   - After successfully building the program, you can run it by specifying the size of the data sequence. For example, to compute the FFT of a sequence of size 256, use the following command:
        ```bash
        ./fft 256
        ```
   - Replace `256` with the desired size of your data sequence, which must be a power of 2.

### Build Options

The following build options are available:


This option builds the program in release mode, which enables compiler optimizations. Namely, this option enables the `-O3` flag, as well as other optimization flags.

```bash
cmake .. -D CMAKE_BUILD_TYPE=Release
```

This option builds the program in debug mode, which disables compiler optimizations.

```bash
cmake .. -D CMAKE_BUILD_TYPE=Debug
```

## Implementations

+ **Classic DFT:** The classic implementation of the Discrete Fourier Transform (DFT) is provided as the `ClassicalFourierTransformAlgorithm` class. This algorithm is used as a reference for the other implementations.

+ **Recursive FFT:** The recursive implementation of the FFT is provided as the `RecursiveFourierTransformAlgorithm` class. This algorithm follows the recursive Cooley-Tukey algorithm, and has a complexity of `O(n log n)`.

+ **Iterative FFT:** The iterative implementation of the FFT is provided as the `IterativeFourierTransformAlgorithm` class. This algorithm follows the iterative Cooley-Tukey algorithm, and has a complexity of `O(n log n)`. This implementation is parallelized using OpenMP.

## Additional Information

All references used in the project are listed above function declarations in the source code.
Additional tools used for frequency visualization are listed in the `tools` directory.