# FFT-Pesce-Miotti-Allahakbari

## Overview

Repository for the 2023/24 AMSC project FFT by Francesco Pesce, Michele Miotti and Mohammadhossein Allahakbari.
The repository focuses on the implementation of multiple FFT algorithms, both with CPU and CUDA code, and their application to image processing with a custom JPEG-inspired algorithm. Additionally, two implementation of the 9/7 wavelet transform were implemented and can be used for simple denoising tasks.

## Dependencies
The project uses the C++20 standard and has the following dependencies:
   - CMake version 3.0.0 or higher.
   - CUDA Toolkit version 12.3 or higher. 
   - OpenMP version 4.5 or higher.
   - OpenCV version 4.5.4 or higher. In Ubuntu, the library is available via ```sudo apt install libopencv-dev```.
Note that older versions of the tools might not be supported. In particular, the code does not compile when using CUDA Toolkit version 11.5.

## Compiling
To compile, run:
```bash
mkdir build
cd build
cmake .. [Flags]
make
```
Where [Flags] are:
   -`-DCMAKE_CUDA_ARCHITECTURES=XX`. For Turing architectures `XX`=75, for Ampere `XX`=80,86,87, for Lovelace `XX`=89 and for Hopper `XX`=90. The code compiles even without setting this flag, but it might not be fully optimized.
   -`-DCMAKE\BUILD_TYPE=YY`, where `YY` is either `Debug` or `Release`, compiling the program in debug and release modes respectively. Both build modes use the flags `-Wall -Wextra` but the former includes debug symbols and uses default optimizations, while the latter uses multiple optimization flags, including `-Ofast`. 
   - `-DUSE_FLOAT=ZZ`, where `ZZ` is either `ON` or `OFF`. The entire codebase uses the `real` type to represent floating point types, its definition can be found in `Real.hpp`. If set, this flag forces the usage of floats, otherwise doubles are used. The default is `OFF` and changing the flag is not recommended, as for large sequences single floating point precision can be too low due to the large number of computations performed.

## Running
To run the program, run `./fft [args]` while in the `build` directory. The first argument in `[args]` is mandatory and it should be `fft`, `compression`, `cuda` or `wavelet`. Based on the first argument, a different execution mode is selected. Each execution mode serves as a demonstration of the implemented features in its corresponding area.

### FFT
This execution mode generates a random signal with complex coefficients with the number of elements given as the second argument, which must be a power of 2. It has 4 sub-modes, specified via the third argument: `demo` (default), `bitReversalTest`, `scalingTest` and `timingTest`. All modes use OpenMP, and the maximum number of threads can be specified as the fourth argument.
#### demo
All 1D direct and inverse Fourier Transform implementations are executed. Results of direct transforms are compared to those obtained with the classical `O(n^2)` algorithm, checking that they are the same up to a certain tolerance. Results of inverse transforms are compared to the original sequence. The trivial 2D direct and inverse FFT algorithm is then applied to a random matrix with the same side length as the length of the sequence and the result of the inverse transform is compared to the original matrix.
#### bitReversalTest
One instance each of `MaskBitReversalAlgorithm` and `FastBitReversalAlgorithm` are tested on the sequence and their execution times are compared for all numbers of threads that are powers of 2 ranging from 1 to the maximum number specified.
#### scalingTest
`IterativeFourierTransformAlgorithm` is tested on the sequence. For all numbers of threads that are powers of 2 ranging from 1 to the maximum number specified, execution times are speed-ups over the serial code are printed.
#### timingTest
An instance of the algorithm specified as the fifth argument is tested on the sequence and its execution time for the maximum number of threads is printed in microseconds.

### Compression
This execution mode loads the image specified as the second argument as grayscale image, compresses it and stores the compressed data in the `img` folder. Said data is then read from the same file, decoded and displayed. 

### CUDA
This section is used for CUDA-related tests, but its interface is still work in progress.

### Wavelet
This execution mode has 3 sub-modes, specified via the second argument: `demo` (default),`image` and `denoise`.
#### demo
A cubic signal with real coefficients is generated and a DWT and IDWT are applied to it for all 1D wavelet transform implementations. The resulting sequences are saved to a file and it is verified that after the inverse transform the sequences are equal to the original one, up to a certain tolerance. The same is done for all 2D wavelet transform implementations on a random matrix with the same side length as the length of the sequence. The sequence length can be specified as the third argument and different algorithms might apply different requirements on the lengths.
#### image
An image is loaded, converted to grayscale and displayed. All 2D DWT implementations are applied to it and the results are displayed and saved as an image. The image path and number of decomposition levels can be provided as the third and fourth arguments.
#### denoise
An image is loaded and denoised using the 2D DWT and IDWT implementations using `GPWaveletTransform97` and soft thresholding. The resulting image is displayed. The image path, number of decomposition levels and threshold can be provided as the third, fourth and fifth arguments.

## Code structure
The root directory of the repository contains the following folders:
   - `img` contains a set of images used for testing.
   - `include` contains the header files for the C++ code, containing declarations of functions and classes, type aliases and definitions of function templates. The latter as used to operate on sequences, writing to file and comparing the elements of a vector.
   - `report` contains the source files for the report for the project. The document can be generated by running ```pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex``` in said directory.
   - `src` contains the source files for the C++ code.
   - `tools` contains some Python tools for performance evaluation of FFT implementations, a way to check a DFT implementation for errors against the implementation by Numpy and some visualization tools.

## Documentation
The code is documented using Doxygen version 1.9.1 for functions and classes and regular comments inside functions. To generate Doxygen documentation, run 
```bash
doxygen Doxyfile
``` 
from the root directory of the repository. The documentation will then be available under `html/index.html`.
