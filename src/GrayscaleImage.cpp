#include "GrayscaleImage.hpp"

#include <opencv2/opencv.hpp>

#include "FourierTransform.hpp"
#include "Utility.hpp"

// Load regular image from file.
bool GrayscaleImage::loadStandard(const std::string &filename) {
  cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
  if (image.empty()) {
    return false;
  }

  // Get the image size.
  int width = image.cols;
  int height = image.rows;

  // Assert that the image size is a multiple of 8.
  assert(width % 8 == 0);
  assert(height % 8 == 0);

  // Memorize the block grid size.
  this->blockGridWidth = width / 8;
  this->blockGridHeight = height / 8;

  // For each row and column...
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      // Add the pixel value to the decoded image.
      uint8_t pixel = image.at<uint8_t>(i, j);
      this->decoded.push_back(pixel);
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
      uint8_t pixel = this->decoded[i * image.cols + j];
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

// Merge the blocks in variable 'blocks' and save the result in variable
// 'decoded'.
void GrayscaleImage::mergeBlocks() {
  // Clear the decoded vector.
  this->decoded.clear();

  // Fill the decoded vector with zeros.
  for (int i = 0; i < this->blockGridHeight * 8; i++) {
    for (int j = 0; j < this->blockGridWidth * 8; j++) {
      this->decoded.push_back(0);
    }
  }

  // For each block row and column...
  for (int i = 0; i < this->blockGridHeight; i++) {
    for (int j = 0; j < this->blockGridWidth; j++) {
      // Get the block.
      std::vector<int8_t> block = this->blocks[i * this->blockGridWidth + j];

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
          uint8_t pixel = block[k * 8 + l];

          // Add the pixel value to the decoded vector, at the right position.
          this->decoded[pixelY * this->blockGridWidth * 8 + pixelX] = pixel;
        }
      }
    }
  }
}

// Static member variable to store the quantization table.
std::vector<int> GrayscaleImage::quantizationTable = {
    200, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
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

// Unquantize the given block using the quantization table.
void GrayscaleImage::unquantize(Transform::FourierTransform::vec &vec,
                                std::vector<int8_t> &realBlock,
                                std::vector<int8_t> &imagBlock) {
  for (int i = 0; i < 64; i++) {
    // Get the quantization table value.
    int quantizationTableValue = GrayscaleImage::quantizationTable[i];

    // Unquantize the element value.
    Transform::real unquantizedRealValue =
        realBlock[i] * quantizationTableValue;
    Transform::real unquantizedImagValue =
        imagBlock[i] * quantizationTableValue;

    // Set the element value to the unquantized value.
    vec[i] = Transform::FourierTransform::complex(unquantizedRealValue,
                                                  unquantizedImagValue);
  }
}

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

    // Quantize the block.
    std::vector<int8_t> realBlock(64, 0);
    std::vector<int8_t> imagBlock(64, 0);

    this->quantize(outputVecBlock, realBlock, imagBlock);

    // Add the quantized block to the blocks vector.
    this->blocks[i] = realBlock;
    this->imagBlocks.push_back(imagBlock);
  }

  this->entropyEncode();
}

// Decode the last loaded or encoded image.
void GrayscaleImage::decode() {
  // Initialize a TrivialTwoDimensionalDiscreteFourierTransform object.
  Transform::FourierTransform::
      TrivialTwoDimensionalInverseFourierTransformAlgorithm fft;

  this->entropyDecode();

  // For each block...
  for (size_t i = 0; i < this->blocks.size(); i++) {
    // Get the block.
    std::vector<int8_t> realBlock = this->blocks[i];
    std::vector<int8_t> imagBlock = this->imagBlocks[i];

    // Unquantize the block.
    Transform::FourierTransform::vec vecBlock(64, 0);
    Transform::FourierTransform::vec outputVecBlock(64, 0);
    this->unquantize(vecBlock, realBlock, imagBlock);

    // Apply the Inverse Fourier transform to the block.
    fft(vecBlock, outputVecBlock);

    // Turn the output vec object into a block.
    std::vector<int8_t> block(64, 0);
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
std::vector<std::pair<int, int>> GrayscaleImage::zigZagMap = {
    {0, 0}, {0, 1}, {1, 0}, {2, 0}, {1, 1}, {0, 2}, {0, 3}, {1, 2},
    {2, 1}, {3, 0}, {4, 0}, {3, 1}, {2, 2}, {1, 3}, {0, 4}, {0, 5},
    {1, 4}, {2, 3}, {3, 2}, {4, 1}, {5, 0}, {6, 0}, {5, 1}, {4, 2},
    {3, 3}, {2, 4}, {1, 5}, {0, 6}, {0, 7}, {1, 6}, {2, 5}, {3, 4},
    {4, 3}, {5, 2}, {6, 1}, {7, 0}, {7, 1}, {6, 2}, {5, 3}, {4, 4},
    {3, 5}, {2, 6}, {1, 7}, {2, 7}, {3, 6}, {4, 5}, {5, 4}, {6, 3},
    {7, 2}, {7, 3}, {6, 4}, {5, 5}, {4, 6}, {3, 7}, {4, 7}, {5, 6},
    {6, 5}, {7, 4}, {7, 5}, {6, 6}, {5, 7}, {6, 7}, {7, 6}, {7, 7}};

// Use entropy coding to encode all blocks.
void GrayscaleImage::entropyEncode() {
  // Clear the encoded vector.
  this->encoded.clear();

  // Initialize a new blocks vector.
  std::vector<std::vector<int8_t>> blockSet;

  // Add all blocks to the blocks vector.
  for (size_t i = 0; i < this->blocks.size(); i++) {
    blockSet.push_back(this->blocks[i]);
  }

  // Add all imaginary blocks to the blocks vector.
  for (size_t i = 0; i < this->imagBlocks.size(); i++) {
    blockSet.push_back(this->imagBlocks[i]);
  }

  // Initialize a unsigned int list for zero counters.
  std::vector<uint8_t> zeroCounters;

  // Initialize an int list for elements.
  std::vector<int8_t> elements;

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

    // Traverse the zigZag vector.
    uint8_t zeroCounter = 0;

    // For each element in the zigZag vector...
    for (int j = 0; j < 64; j++) {
      // Get the element value.
      int8_t element = zigZagVector[j];

      // If the element value is zero...
      if (element == 0) {
        // Increment the zero counter.
        zeroCounter++;
        continue;
      }

      // If the element value is not zero...
      zeroCounters.push_back(zeroCounter);
      elements.push_back(element);

      // Reset the zero counter.
      zeroCounter = 0;
    }

    // Add end of block marker.
    zeroCounters.push_back(0xFF);
    elements.push_back(0);
  }

  // Concatenate the zero counters and the elements.
  for (size_t i = 0; i < zeroCounters.size(); i++) {
    this->encoded.push_back(zeroCounters[i]);
    this->encoded.push_back(elements[i]);
  }
}

// Use entropy coding to decode all blocks.
void GrayscaleImage::entropyDecode() {
  // Clear the blocks vector.
  this->blocks.clear();
  this->imagBlocks.clear();

  // Create a new blocks vector.
  std::vector<std::vector<int8_t>> blocksSet;

  // Create a new reconstrcuted zigZag vector.
  std::vector<int8_t> reconstructedZigZagVector;

  // Initialize an unsigned int list of zero counters.
  std::vector<uint8_t> zeroCounters;

  // Initialize a int list of elements.
  std::vector<int8_t> elements;

  // Fill the zero counters with even elements of the encoded vector.
  for (size_t i = 0; i < this->encoded.size(); i++) {
    if (i % 2 == 0) {
      zeroCounters.push_back(this->encoded[i]);
    }
  }

  // Fill the elements with odd elements of the encoded vector.
  for (size_t i = 0; i < this->encoded.size(); i++) {
    if (i % 2 == 1) {
      elements.push_back(this->encoded[i]);
    }
  }

  // For each byte in the zero counters list...
  for (size_t i = 0; i < zeroCounters.size(); i++) {
    // Check if the byte is the end of block marker.
    if (zeroCounters[i] == 0xFF) {
      // Keep adding zeros to the reconstructed zigZag vector until the
      // reconstructed zigZag vector size is 64.
      while (reconstructedZigZagVector.size() < 64) {
        reconstructedZigZagVector.push_back(0);
      }

      // Create a new block.
      std::vector<int8_t> block(64, 0);

      // For each element in the reconstructed zigZag vector...
      for (size_t j = 0; j < reconstructedZigZagVector.size(); j++) {
        // Get the zigZag map coordinates.
        int x = GrayscaleImage::zigZagMap[j].first;
        int y = GrayscaleImage::zigZagMap[j].second;

        // Set the block value to the reconstructed zigZag vector value.
        block[y * 8 + x] = reconstructedZigZagVector[j];
      }

      // Add the block to the blocks vector.
      blocksSet.push_back(block);

      // Clear the reconstructed zigZag vector.
      reconstructedZigZagVector.clear();

      // Continue to the next byte.
      continue;
    }

    // Add as many zeros to the reconstructed zigZag vector as the zero counter
    // value.
    for (int j = 0; j < zeroCounters[i]; j++) {
      reconstructedZigZagVector.push_back(0);
    }

    // Add the element to the reconstructed zigZag vector.
    reconstructedZigZagVector.push_back(elements[i]);
  }

  // Add the first half of the blockSet to the blocks vector.
  for (size_t i = 0; i < blocksSet.size() / 2; i++) {
    this->blocks.push_back(blocksSet[i]);
  }

  // Add the second half of the blockSet to the imaginary blocks vector.
  for (size_t i = blocksSet.size() / 2; i < blocksSet.size(); i++) {
    this->imagBlocks.push_back(blocksSet[i]);
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
  TwoDimensionalWaveletTransformAlgorithm algorithm_2d;
  algorithm_2d.setAlgorithm(algorithm);
  algorithm_2d.directTransform(real_input, real_output, levels);

  // Map the values to [0, 256].
  affineMap(real_output, real(0), real(256));

  // Update the decoded image.
  for (size_t i = 0; i < this->decoded.size(); i++) {
    this->decoded[i] = static_cast<uint8_t>(real_output[i]);
  }
}

// Denoise an image with a DWT, IWT and thresholding.
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
  TwoDimensionalWaveletTransformAlgorithm algorithm_2d;
  algorithm_2d.setAlgorithm(algorithm);
  algorithm_2d.directTransform(real_input, real_output, levels);

  // Use thresholding to denoise the image.
  for (size_t i = 0; i < real_output.size(); i++) {
    if (use_hard_thresholding) {
      if (std::abs(real_output[i]) < threshold) {
        real_output[i] = 0;
      }
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
  algorithm_2d.inverseTransform(real_output, real_output, levels);

  // Map the values to [0, 256].
  affineMap(real_output, real(0), real(256));

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
    std::streampos fileSize = file.tellg();
    this->encoded.resize(fileSize);

    file.seekg(0, std::ios_base::beg);
    file.read((char *)&this->encoded[0], fileSize);
  } else {
    return false;
  }

  return true;
}

// Save compressed image to file.
bool GrayscaleImage::save(const std::string &filename) {
  std::ofstream file(filename);
  assert(file.is_open());

  file.write((char *)&this->encoded[0], this->encoded.size());

  return true;
}

// Get the encoded image.
std::vector<uint8_t> GrayscaleImage::getEncoded() { return this->encoded; }