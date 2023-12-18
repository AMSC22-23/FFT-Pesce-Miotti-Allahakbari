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
   - After successfully building the program, you can run it and decide to provide some arguments. For example, to execute a demo of the developed functionalities on a sequence of size 256 with up to 4 threads, use the following command:
        ```bash
        ./fft 256 demo 4
        ```
   - Replace `256` with the desired size of your data sequence, which must be a power of 2. The default size is 1024.
   - Replace `demo` with the desired execution mode, refer to the next section for details. This is the default execution mode.
   - Replace `4` with the desired maximum number of threads allowed. The default amount is 8.
   - If you are using mode `timingTest`, choose a method among the ones available, refer to the next section for details. The default value is "iterative".
   
### Program behaviour
The program can be executed in three different modes:
- `demo`: This is a demo of the implemented functionalities. A random sequence with the specified size is created, then all Fourier Transform algorithms are performed and results are compared. The transforms of the input are then used as inputs for all the implementations of the Inverse Fourier Transform algorithms, and results are again compared to the original sequences.
- `bitReversalTest`: A time and parallel speed-up comparison between the two best performing bit reversal permutation techniques will be performed on a random sequence.
- `scalingTest`: A scaling test will be performed on a random sequence using the parallelized direct iterative FFT.
- `timingTest`: The specified direct transform algorithm will be run once on a random sequence with the specified number of threads and time for execution in microseconds will be printed. The possible options are "classic", "recursive" and "iterative".

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

+ **Iterative FFT:** The iterative implementation of the FFT is provided as the `IterativeFourierTransformAlgorithm` class. This algorithm follows the iterative Cooley-Tukey algorithm, and has a complexity of `O(n log n)`. This implementation is parallelized using OpenMP. The implementation uses one of the algorithms presented in the following paragraph to perform bit reversal permutation of the input sequence.

### Bit reversal permutation

+ **Naive bit reversal:** A naive implementation of bit reversal permutation is provided as the `NaiveBitReversalPermutationAlgorithm` class. This implementation has a complexity of `O(n log n)` and is massively parallel.
+ **Mask bit reversal:** An optimized implementation of bit reversal permutation is provided as the `MaskBitReversalPermutationAlgorithm` class. This implementation has a complexity of `O(n log n)` and is massively parallel.
+ **Fast bit reversal:** A different optimized implementation of bit reversal permutation is provided as the `FastBitReversalPermutationAlgorithm` class. This implementation has a complexity of `O(n)` but is not massively parallel, requires an `n` long vector to store temporary results and has a larger constant than mask bit reversal.

## Additional Information

All references used in the project are listed above function declarations in the source code.
Additional tools used for frequency visualization and results collection are listed in the `tools` directory.
