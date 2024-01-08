This file is being used as a temporary note to keep track of progress about this branch. This is not needed for developers as a guide, but as a stepping stone for a full report to present.

## 1. Introduction
This project is composed by two main parts: the efficient, parallel implementation of the fast Fourier transform, and a set of applications that use it, namely a simplified version of the JPEG compression algorithm and the wavelet transform.

The fast Fourier transform (FFT) is an algorithm that computes the discrete Fourier transform (DFT) of a sequence, or its inverse (IDFT). The DFT is a mathematical transformation that decomposes a signal into its constituent frequencies. It is used in many fields, such as signal processing or image processing, and it is the basis of many compression algorithms, such as JPEG.

This report will focus on three main aspects: the parallelization and optimization of the two dimensional FFT, the implementation of the JPEG compression algorithm, and the implementation of the wavelet transform.

## 2. Parallelization and optimization of the two dimensional FFT
...

## 3. Implementation of the JPEG compression algorithm
Our implementation of the JPEG compression algorithm is based on the following steps:

1. Split the image into 8x8 blocks
2. Slide each block's values by 128, so that the values are centered around 0
3. Apply the two dimensional FFT to each block
4. Quantize the coefficients of each block, using a hand-made quantization matrix
5. Read the block in a zig-zag fashion, and apply run-length encoding to the coefficients

Noting that this method differs from the original JPEG algorithm in the following aspects:

- The original JPEG algorithm uses the YCbCr color space, while we use the grayscale color space
- The original JPEG algorithm uses a discrete cosine transform (DCT), while we use a fast Fourier transform (FFT)
- The original JPEG algorithm uses a different quantization matrix
- The original JPEG algorithm uses a more elaborate run-length encoding scheme

The second difference also implies that we need to encode both the real and imaginary parts of the FFT coefficients, while the original JPEG algorithm only encodes the real part of the DCT coefficients. This worsen the compression ratio by a factor of 2, but it is necessary to be able to reconstruct the image.

### 3.1. Splitting the image into 8x8 blocks
...

### 3.2. Sliding the values by 128
...

### 3.3. Applying the two dimensional FFT
...

### 3.4. Quantizing the coefficients
...

### 3.5. Run-length encoding
...

## 4. Implementation of the wavelet transform
...