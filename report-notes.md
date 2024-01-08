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
2. Apply the two dimensional FFT to each block
3. Quantize the coefficients of each block, using a hand-made quantization matrix
4. Read the block in a zig-zag fashion, and apply run-length encoding to the coefficients

Noting that this method differs from the original JPEG algorithm in the following aspects:

- The original JPEG algorithm uses the YCbCr color space, while we use the grayscale color space
- The original JPEG algorithm uses a discrete cosine transform (DCT), while we use a fast Fourier transform (FFT)
- The original JPEG algorithm uses a different quantization matrix
- The original JPEG algorithm uses a more elaborate run-length encoding scheme

The second difference also implies that we need to encode both the real and imaginary parts of the FFT coefficients, while the original JPEG algorithm only encodes the real part of the DCT coefficients. This worsen the compression ratio by a factor of 2, but it is necessary to be able to reconstruct the image.

### 3.1. Splitting the image into 8x8 blocks
The first step is to split the image into 8x8 blocks. This is done by the `splitBlocks` function. The blocks are stored in row-major order, so that the first n blocks are the first row of the image, the next n blocks are the second row of the image, and so on.

```c++
void GrayscaleImage::splitBlocks() {
  this->blocks.clear();

  // For each block row and column...
  for (int i = 0; i < this->blockGridHeight; i++) {
    for (int j = 0; j < this->blockGridWidth; j++) {
      // Create a new block.
      std::vector<int8_t> block;

      // For each row in the block...
      for (int k = 0; k < 8; k++) {
        // For each column in the block...
        for (int l = 0; l < 8; l++) {
          // Get the top-left pixel coordinates of the block.
          int x = j * 8;
          int y = i * 8;

          // Get the pixel coordinates.
          int pixelX = x + l;
          int pixelY = y + k;

          // Get the pixel value.
          uint8_t pixel =
              this->decoded[pixelY * this->blockGridWidth * 8 + pixelX];

          // Add the pixel value to the block.
          block.push_back(pixel);
        }
      }

      // Add the block to the blocks vector.
      this->blocks.push_back(block);
    }
  }
}
```

### 3.2. Applying the two dimensional FFT
All the values of the image are then slid by 128, so that they are centered around 0. This step is part of the original JPEG algorithm, and it's performed right before the FFT. After that, the FFT is applied to each block.

```c++
// Encode the last loaded or decoded image.
void GrayscaleImage::encode() {
  // Split the image in blocks of size 8x8.
  this->splitBlocks();

  // Initialize a TrivialTwoDimensionalDiscreteFourierTransform object.
  Transform::FourierTransform::TrivialTwoDimensionalFourierTransformAlgorithm
      fft;

  // For each block...
  this->imagBlocks.clear();
  for (size_t i = 0; i < this->blocks.size(); i++) {
    // Get the block.
    std::vector<int8_t> block = this->blocks[i];

    // Turn the block into a vec object.
    Transform::FourierTransform::vec vecBlock(64, 0.0);
    Transform::FourierTransform::vec outputVecBlock(64, 0.0);
    for (size_t j = 0; j < block.size(); j++) {
      vecBlock[j] = block[j];
      vecBlock[j] -= 128;
    }

    // Apply the Fourier transform to the block.
    fft(vecBlock, outputVecBlock);

    ...
  }

  ...
}
```

Note that the result of the FFT is a complex number, so we need to store both the real and imaginary parts of the result. The real and imaginary parts are stored in two separate vectors.

The idea behind the FFT is to decompose a signal into its constituent frequencies. In the case of a two dimensional FFT, the signal is a two dimensional array of values, and the frequencies are also two dimensional arrays of values. Expressing the image as a sum of frequencies allows us to discard the frequencies that the human eye is less sensitive to, and thus compress the image.

### 3.3. Quantizing the coefficients
To decide which frequencies to discard, and by how much, we use a quantization matrix. Each value of the matrix corresponds to a frequency, and the higher the value, the more we discard that frequency. Efficient quantization matrices are only available for the DCT, so in our case we compress all frequencies by the same amount, since we lack the knowledge to decide which frequencies are more important than others under the FFT's point of view.

Nevertheless, the top-left corner of the matrix is quite high, since the result of the FFT has a spike in the top-left corner. Having a low value in the top-left corner of the matrix would result in a compressed value that exceeds the maximum value of a signed 8-bit integer, which would cause an overflow.

```c++
// Static member variable to store the quantization table.
std::vector<int> GrayscaleImage::quantizationTable = {
    200, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100,
};

// Quantize the given vec into two blocks using the quantization table.
void GrayscaleImage::quantize(const Transform::FourierTransform::vec &vec,
                              std::vector<int8_t> &realBlock,
                              std::vector<int8_t> &imagBlock) {
  // For element in the block...
  for (int i = 0; i < 64; i++) {
    // Get the quantization table value.
    int quantizationTableValue = GrayscaleImage::quantizationTable[i];

    // Set the element value to the quantized value.
    realBlock[i] = vec[i].real() / quantizationTableValue;
    imagBlock[i] = vec[i].imag() / quantizationTableValue;
  }
}
```

### 3.4. Run-length encoding
The last step is to read the blocks in a zig-zag fashion, and apply run-length encoding to the coefficients. 
Since the block size is 8x8, the zig-zag pattern is stored in a 64-element array, to avoid having to compute it every time.

```c++
// Use entropy coding to encode all blocks.
void GrayscaleImage::entropyEncode() {
  ...

  // For each block...
  for (size_t i = 0; i < blockSet.size(); i++) {
    std::vector<int8_t> block = blockSet[i];

    // Initialize a zigZag vector.
    std::vector<int8_t> zigZagVector(64, 0);

    // For each step in zigzag path...
    for (int j = 0; j < 64; j++) {
      // Get the zigZag map coordinates.
      int x = GrayscaleImage::zigZagMap[j].first;
      int y = GrayscaleImage::zigZagMap[j].second;

      // Set the zigZag vector value to the block value.
      zigZagVector[j] = block[y * 8 + x];
    }

    ...
  }
}
```

Reading the block in a zig-zag fashion allows us to group together the coefficients that are more likely to be zero, and thus apply run-length encoding to them. Run-length encoding is a simple compression algorithm that replaces a sequence of repeated values with a single value and a count.

We use one byte to store the number of zeroes, and one byte to store the value. This means that we can only compress sequences of zeroes that are shorter than 255, but this is not a problem since the sequences of zeroes are at most 64 elements long. After interleaving the zeroes and the values, we store the result in a vector of bytes. The real and imaginary parts of the block are concatenated together.

## 4. Implementation of the wavelet transform
...