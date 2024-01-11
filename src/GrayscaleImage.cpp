#include "GrayscaleImage.hpp"

/**
 * @file GrayscaleImage.cpp.
 * @brief Defines the methods and functions declared in
 * GrayscaleImage.hpp.
 */

#include <omp.h>

#include <opencv2/opencv.hpp>

#include "FourierTransform.hpp"
#include "Utility.hpp"

// Load regular image from file.
bool GrayscaleImage::loadStandard(const std::string &filename) {
  const cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
  if (image.empty()) {
    return false;
  }

  // Get the image size.
  const int width = image.cols;
  const int height = image.rows;

  // Assert that the image size is a multiple of 8.
  assert(width % 8 == 0);
  assert(height % 8 == 0);

  // Memorize the block grid size.
  this->blockGridWidth = width / 8;
  this->blockGridHeight = height / 8;

  // For each row and column...
  this->decoded.reserve(width * height);
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      // Add the pixel value to the decoded image.
      const uint8_t pixel = image.at<uint8_t>(i, j);
      this->decoded.emplace_back(pixel);
    }
  }

  return true;
}

// Get the bitsize of the last loaded or decoded image.
unsigned int GrayscaleImage::getStandardBitsize() const {
  return this->blockGridWidth * this->blockGridHeight * 64 * 8;
}

// Display the last loaded or decoded image.
void GrayscaleImage::display() {
  // Create a new image.
  cv::Mat image(this->blockGridHeight * 8, this->blockGridWidth * 8, CV_8UC1);

  // For each row and column...
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      // Set the pixel value.
      const uint8_t pixel = this->decoded[i * image.cols + j];
      image.at<uint8_t>(i, j) = pixel;
    }
  }

  // Display the image.
  cv::imshow("Image", image);
  cv::waitKey(0);
}

// Split the image in blocks of size 8x8, and save the result in variable
// 'blocks'.
void GrayscaleImage::splitBlocks() {
  this->blocks.clear();

  // For each block row and column...
  blocks.reserve(this->blockGridHeight * this->blockGridWidth);
  for (int i = 0; i < this->blockGridHeight; i++) {
    for (int j = 0; j < this->blockGridWidth; j++) {
      // Create a new block.
      std::array<int8_t, 64> block;

      // For each row in the block...
      for (int k = 0; k < 8; k++) {
        // For each column in the block...
        for (int l = 0; l < 8; l++) {
          // Get the top-left pixel coordinates of the block.
          const int x = j * 8;
          const int y = i * 8;

          // Get the pixel coordinates.
          int pixelX = x + l;
          const int pixelY = y + k;

          // Get the pixel value.
          const uint8_t pixel =
              this->decoded[pixelY * this->blockGridWidth * 8 + pixelX];

          // Add the pixel value to the block.
          block[8 * k + l] = pixel;
        }
      }

      // Add the block to the blocks vector.
      this->blocks.emplace_back(block);
    }
  }
}

// Merge the blocks in variable 'blocks' and save the result in variable
// 'decoded'.
void GrayscaleImage::mergeBlocks() {
  // Clear the decoded vector.
  this->decoded.clear();

  // Fill the decoded vector with zeros.
  this->decoded.reserve(this->blockGridHeight * this->blockGridWidth * 64);
  for (int i = 0; i < this->blockGridHeight * 8; i++) {
    for (int j = 0; j < this->blockGridWidth * 8; j++) {
      this->decoded.emplace_back(0);
    }
  }

  // For each block row and column...
  for (int i = 0; i < this->blockGridHeight; i++) {
    for (int j = 0; j < this->blockGridWidth; j++) {
      // Get the block.
      const std::array<int8_t, 64> block =
          this->blocks[i * this->blockGridWidth + j];

      // For each row in the block...
      for (int k = 0; k < 8; k++) {
        // For each column in the block...
        for (int l = 0; l < 8; l++) {
          // Get the top-left pixel coordinates of the block.
          const int x = j * 8;
          const int y = i * 8;

          // Get the pixel coordinates.
          int pixelX = x + l;
          const int pixelY = y + k;

          // Get the pixel value.
          const uint8_t pixel = block[k * 8 + l];

          // Add the pixel value to the decoded vector, at the right position.
          this->decoded[pixelY * this->blockGridWidth * 8 + pixelX] = pixel;
        }
      }
    }
  }
}

// Static member variable to store the quantization table.
std::array<int, 64> GrayscaleImage::quantizationTable{
    200, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
};

// Quantize the given complex block into two blocks using the quantization
// table.
void GrayscaleImage::quantize(
    const std::array<Transform::FourierTransform::complex, 64> &complexBlock,
    std::array<int8_t, 64> &realBlock, std::array<int8_t, 64> &imagBlock) {
  // For element in the block...
  for (int i = 0; i < 64; i++) {
    // Get the quantization table value.
    const int quantizationTableValue = GrayscaleImage::quantizationTable[i];

    // Set the element value to the quantized value.
    realBlock[i] = complexBlock[i].real() / quantizationTableValue;
    imagBlock[i] = complexBlock[i].imag() / quantizationTableValue;
  }
}

// Unquantize the given block using the quantization table.
void GrayscaleImage::unquantize(
    std::array<Transform::FourierTransform::complex, 64> &complexBlock,
    const std::array<int8_t, 64> &realBlock,
    const std::array<int8_t, 64> &imagBlock) {
  for (int i = 0; i < 64; i++) {
    // Get the quantization table value.
    const int quantizationTableValue = GrayscaleImage::quantizationTable[i];

    // Unquantize the element value.
    const Transform::real unquantizedRealValue =
        realBlock[i] * quantizationTableValue;
    const Transform::real unquantizedImagValue =
        imagBlock[i] * quantizationTableValue;

    // Set the element value to the unquantized value.
    complexBlock[i] = Transform::FourierTransform::complex(
        unquantizedRealValue, unquantizedImagValue);
  }
}

// Encode the last loaded or decoded image.
void GrayscaleImage::encode() {
  // Split the image in blocks of size 8x8.
  this->splitBlocks();

  // Initialize a TrivialTwoDimensionalDiscreteFourierTransform object.
  const Transform::FourierTransform::
      TwoDimensionalDirectFFTCPU fft;

  // Disable nested parallelization (not efficient for consumer-grade
  // architectures).
  omp_set_nested(false);

  // Clear the imaginary blocks and initialize the vector, so that threads can
  // write in it in parallel.
  this->imagBlocks.clear();
  this->imagBlocks.reserve(this->blocks.size());
  for (size_t i = 0; i < this->blocks.size(); i++) {
    this->imagBlocks.emplace_back();
  }

  // For each block...
#pragma omp parallel for
  for (size_t i = 0; i < this->blocks.size(); i++) {
    // Get the block.
    const std::array<int8_t, 64> block = this->blocks[i];

    // Turn the block into a vec object.
    Transform::FourierTransform::vec vecBlock(64, 0.0);
    Transform::FourierTransform::vec outputVecBlock(64, 0.0);
    for (size_t j = 0; j < block.size(); j++) {
      vecBlock[j] = block[j];
      vecBlock[j] -= 128;
    }

    // Apply the Fourier transform to the block.
    fft(vecBlock, outputVecBlock);

    // Quantize the block.
    std::array<int8_t, 64> realBlock;
    std::array<int8_t, 64> imagBlock;
    std::array<Transform::FourierTransform::complex, 64> complexBlock;

    for (size_t i = 0; i < 64; i++) {
      complexBlock[i] = outputVecBlock[i];
    }

    this->quantize(complexBlock, realBlock, imagBlock);

    // Add the quantized block to the blocks vector.
    this->blocks[i] = realBlock;
    this->imagBlocks[i] = imagBlock;
  }

  // Perform run-length encoding.
  this->entropyEncode();
}

// Decode the last loaded or encoded image.
void GrayscaleImage::decode() {
  // Initialize a TrivialTwoDimensionalDiscreteFourierTransform object.
  const Transform::FourierTransform::
      TwoDimensionalInverseFFTCPU fft;

  // Decode the run-length encoding.
  this->entropyDecode();

  // Disable nested parallelization (not efficient for consumer-grade
  // architectures).
  omp_set_nested(false);

// For each block...
#pragma omp parallel for
  for (size_t i = 0; i < this->blocks.size(); i++) {
    // Get the block.
    const std::array<int8_t, 64> realBlock = this->blocks[i];
    const std::array<int8_t, 64> imagBlock = this->imagBlocks[i];

    // Unquantize the block.
    std::array<Transform::FourierTransform::complex, 64> complexBlock;
    this->unquantize(complexBlock, realBlock, imagBlock);

    // Apply the Inverse Fourier transform to the block.
    Transform::FourierTransform::vec vecBlock;
    Transform::FourierTransform::vec outputVecBlock(64, 0);

    vecBlock.reserve(64);
    for (size_t i = 0; i < 64; i++) {
      vecBlock.emplace_back(complexBlock[i]);
    }

    fft(vecBlock, outputVecBlock);

    // Turn the output vec object into a block.
    std::array<int8_t, 64> block;
    for (size_t j = 0; j < outputVecBlock.size(); j++) {
      block[j] = outputVecBlock[j].real();
      block[j] += 128;
    }

    this->blocks[i] = block;
  }

  // Merge the blocks in variable 'blocks'.
  this->mergeBlocks();
}

// Static member variable to store the zigZag map.
std::array<std::pair<int, int>, 64> GrayscaleImage::zigZagMap{
    {{0, 0}, {0, 1}, {1, 0}, {2, 0}, {1, 1}, {0, 2}, {0, 3}, {1, 2},
     {2, 1}, {3, 0}, {4, 0}, {3, 1}, {2, 2}, {1, 3}, {0, 4}, {0, 5},
     {1, 4}, {2, 3}, {3, 2}, {4, 1}, {5, 0}, {6, 0}, {5, 1}, {4, 2},
     {3, 3}, {2, 4}, {1, 5}, {0, 6}, {0, 7}, {1, 6}, {2, 5}, {3, 4},
     {4, 3}, {5, 2}, {6, 1}, {7, 0}, {7, 1}, {6, 2}, {5, 3}, {4, 4},
     {3, 5}, {2, 6}, {1, 7}, {2, 7}, {3, 6}, {4, 5}, {5, 4}, {6, 3},
     {7, 2}, {7, 3}, {6, 4}, {5, 5}, {4, 6}, {3, 7}, {4, 7}, {5, 6},
     {6, 5}, {7, 4}, {7, 5}, {6, 6}, {5, 7}, {6, 7}, {7, 6}, {7, 7}}};

// Use entropy coding to encode all blocks.
void GrayscaleImage::entropyEncode() {
  // Clear the encoded vector.
  this->encoded.clear();

  // Initialize a new blocks vector.
  std::vector<std::array<int8_t, 64>> blockSet;
  blockSet.reserve(this->blocks.size() + this->imagBlocks.size());

  // Add all blocks to the blocks vector.
  for (size_t i = 0; i < this->blocks.size(); i++) {
    blockSet.emplace_back(this->blocks[i]);
  }

  // Add all imaginary blocks to the blocks vector.
  for (size_t i = 0; i < this->imagBlocks.size(); i++) {
    blockSet.emplace_back(this->imagBlocks[i]);
  }

  // Initialize a unsigned int list for zero counters.
  // While we do not know the length of the sequence, we know it will have at
  // least an element of each block.
  std::vector<uint8_t> zeroCounters;
  zeroCounters.reserve(blockSet.size());

  // Initialize an int list for elements.
  // While we do not know the length of the sequence, we know it will have at
  // least an element of each block.
  std::vector<int8_t> elements;
  elements.reserve(blockSet.size());

  // For each block...
  for (size_t i = 0; i < blockSet.size(); i++) {
    const std::array<int8_t, 64> block = blockSet[i];

    // Initialize a zigZag vector.
    std::array<int8_t, 64> zigZagVector;

    // For each step in zigzag path...
    for (int j = 0; j < 64; j++) {
      // Get the zigZag map coordinates.
      const int x = GrayscaleImage::zigZagMap[j].first;
      const int y = GrayscaleImage::zigZagMap[j].second;

      // Set the zigZag vector value to the block value.
      zigZagVector[j] = block[y * 8 + x];
    }

    // Traverse the zigZag vector.
    uint8_t zeroCounter = 0;

    // For each element in the zigZag vector...
    for (int j = 0; j < 64; j++) {
      // Get the element value.
      const int8_t element = zigZagVector[j];

      // If the element value is zero...
      if (element == 0) {
        // Increment the zero counter.
        zeroCounter++;
        continue;
      }

      // If the element value is not zero...
      zeroCounters.emplace_back(zeroCounter);
      elements.emplace_back(element);

      // Reset the zero counter.
      zeroCounter = 0;
    }

    // Add end of block marker.
    zeroCounters.emplace_back(0xFF);
    elements.emplace_back(0);
  }

  // Concatenate the zero counters and the elements.
  this->encoded.reserve(zeroCounters.size() * 2);
  for (size_t i = 0; i < zeroCounters.size(); i++) {
    this->encoded.emplace_back(zeroCounters[i]);
    this->encoded.emplace_back(elements[i]);
  }
}

// Use entropy coding to decode all blocks.
void GrayscaleImage::entropyDecode() {
  // Clear the blocks vector.
  this->blocks.clear();
  this->imagBlocks.clear();

  // Create a new blocks vector.
  std::vector<std::array<int8_t, 64>> blocksSet;

  // Create a new reconstrcuted zigZag vector.
  std::vector<int8_t> reconstructedZigZagVector;
  reconstructedZigZagVector.reserve(64);

  // Initialize an unsigned int list of zero counters.
  std::vector<uint8_t> zeroCounters;
  zeroCounters.reserve(this->encoded.size());

  // Initialize a int list of elements.
  std::vector<int8_t> elements;
  elements.reserve(this->encoded.size());

  // Fill the zero counters with even elements of the encoded vector.
  for (size_t i = 0; i < this->encoded.size(); i++) {
    if (i % 2 == 0) {
      zeroCounters.emplace_back(this->encoded[i]);
    }
  }

  // Fill the elements with odd elements of the encoded vector.
  for (size_t i = 0; i < this->encoded.size(); i++) {
    if (i % 2 == 1) {
      elements.emplace_back(this->encoded[i]);
    }
  }

  // For each byte in the zero counters list...
  for (size_t i = 0; i < zeroCounters.size(); i++) {
    // Check if the byte is the end of block marker.
    if (zeroCounters[i] == 0xFF) {
      // Keep adding zeros to the reconstructed zigZag vector until the
      // reconstructed zigZag vector size is 64.
      while (reconstructedZigZagVector.size() < 64) {
        reconstructedZigZagVector.emplace_back(0);
      }

      // Create a new block.
      std::array<int8_t, 64> block;

      // For each element in the reconstructed zigZag vector...
      for (size_t j = 0; j < reconstructedZigZagVector.size(); j++) {
        // Get the zigZag map coordinates.
        const int x = GrayscaleImage::zigZagMap[j].first;
        const int y = GrayscaleImage::zigZagMap[j].second;

        // Set the block value to the reconstructed zigZag vector value.
        block[y * 8 + x] = reconstructedZigZagVector[j];
      }

      // Add the block to the blocks vector.
      blocksSet.emplace_back(block);

      // Clear the reconstructed zigZag vector.
      reconstructedZigZagVector.clear();

      // Continue to the next byte.
      continue;
    }

    // Add as many zeros to the reconstructed zigZag vector as the zero counter
    // value.
    for (int j = 0; j < zeroCounters[i]; j++) {
      reconstructedZigZagVector.emplace_back(0);
    }

    // Add the element to the reconstructed zigZag vector.
    reconstructedZigZagVector.emplace_back(elements[i]);
  }

  // Add the first half of the blockSet to the blocks vector.
  this->blocks.reserve(blocksSet.size());
  for (size_t i = 0; i < blocksSet.size() / 2; i++) {
    this->blocks.emplace_back(blocksSet[i]);
  }

  // Add the second half of the blockSet to the imaginary blocks vector.
  for (size_t i = blocksSet.size() / 2; i < blocksSet.size(); i++) {
    this->imagBlocks.emplace_back(blocksSet[i]);
  }

  // Set the block grid size as sqrt of the blocks vector size.
  this->blockGridWidth = sqrt(this->blocks.size());
  this->blockGridHeight = sqrt(this->blocks.size());
}

// Get the bitsize of the last loaded or encoded image.
unsigned int GrayscaleImage::getCompressedBitsize() const {
  return this->encoded.size() * 8;
}

// Apply a DWT on an image.
void GrayscaleImage::waveletTransform(
    std::unique_ptr<Transform::WaveletTransform::WaveletTransformAlgorithm>
        &algorithm,
    unsigned int levels) {
  using namespace Transform;
  using namespace WaveletTransform;

  // Convert the decoded image's values from uint8_t to real.
  std::vector<real> real_input;
  real_input.reserve(this->decoded.size());
  for (size_t i = 0; i < this->decoded.size(); i++) {
    real_input.emplace_back(static_cast<real>(this->decoded[i]));
  }

  // Allocate an output vector.
  std::vector<real> real_output(this->decoded.size(), 0);

  // Perform the direct wavelet transform.
  const TwoDimensionalWaveletTransformAlgorithm algorithm_2d(algorithm);
  algorithm_2d.directTransform(real_input, real_output, levels, true);

  // Map the values to [0, 256].
  affineMap(real_output, real(0), real(256));

  // Update the decoded image.
  for (size_t i = 0; i < this->decoded.size(); i++) {
    this->decoded[i] = static_cast<uint8_t>(real_output[i]);
  }
}

// Denoise an image with a DWT, IDWT and thresholding.
void GrayscaleImage::denoise(
    std::unique_ptr<Transform::WaveletTransform::WaveletTransformAlgorithm>
        &algorithm,
    unsigned int levels, Transform::real threshold,
    bool use_hard_thresholding) {
  using namespace Transform;
  using namespace WaveletTransform;

  // Convert the decoded image's values from uint8_t to real.
  std::vector<real> real_input;
  real_input.reserve(this->decoded.size());
  for (size_t i = 0; i < this->decoded.size(); i++) {
    real_input.emplace_back(static_cast<real>(this->decoded[i]));
  }

  // Allocate an output vector.
  std::vector<real> real_output(this->decoded.size(), 0);

  // Perform the direct wavelet transform.
  const TwoDimensionalWaveletTransformAlgorithm algorithm_2d(algorithm);
  algorithm_2d.directTransform(real_input, real_output, levels, true);

  // Use thresholding to denoise the image.
  for (size_t i = 0; i < real_output.size(); i++) {
    // Hard thresholding.
    if (use_hard_thresholding) {
      if (std::abs(real_output[i]) < threshold) {
        real_output[i] = 0;
      }
      // Soft thresholding.
    } else {
      if (std::abs(real_output[i]) < threshold) {
        real_output[i] = 0;
      } else if (real_output[i] > 0) {
        real_output[i] -= threshold;
      } else {
        real_output[i] += threshold;
      }
    }
  }

  // Perform the inverse wavelet transform.
  algorithm_2d.inverseTransform(real_output, real_output, levels, true);

  // Map the values to [0, 255].
  affineMap(real_output, real(0), real(255));

  // Update the decoded image.
  for (size_t i = 0; i < this->decoded.size(); i++) {
    this->decoded[i] = static_cast<uint8_t>(real_output[i]);
  }
}

// Load compressed image from file.
bool GrayscaleImage::loadCompressed(const std::string &filename) {
  std::ifstream file(filename);
  assert(file.is_open());

  if (!file.eof() && !file.fail()) {
    file.seekg(0, std::ios_base::end);
    const std::streampos fileSize = file.tellg();
    this->encoded.resize(fileSize);

    file.seekg(0, std::ios_base::beg);
    file.read(reinterpret_cast<char *>(&this->encoded[0]), fileSize);
  } else {
    return false;
  }

  return true;
}

// Save compressed image to file.
bool GrayscaleImage::saveCompressed(const std::string &filename) {
  // Open the file.
  std::ofstream file(filename);
  if (!file.is_open()) {
    return false;
  }

  // Write the image.
  file.write(reinterpret_cast<char *>(&this->encoded[0]), this->encoded.size());

  return !file.fail();
}

// Save image to file.
bool GrayscaleImage::saveStandard(const std::string &filename) {
  // Create an OpenCV image.
  cv::Mat image(this->blockGridHeight * 8, this->blockGridWidth * 8, CV_8UC1);

  // For each row and column...
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      // Set the pixel value.
      uint8_t pixel = this->decoded[i * image.cols + j];
      image.at<uint8_t>(i, j) = pixel;
    }
  }

  // Write the image.
  return cv::imwrite(filename, image);
}

// Get the encoded image.
std::vector<uint8_t> GrayscaleImage::getEncoded() { return this->encoded; }