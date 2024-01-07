#include "GrayscaleImage.hpp"

#include <limits>
#include <numbers>
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

  // For each row...
  for (int i = 0; i < image.rows; i++) {
    // For each column...
    for (int j = 0; j < image.cols; j++) {
      // Get the pixel value.
      char pixel = image.at<char>(i, j);

      // Add the pixel value to the decoded image.
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

  // For each row...
  for (int i = 0; i < image.rows; i++) {
    // For each column...
    for (int j = 0; j < image.cols; j++) {
      // Get the pixel value.
      char pixel = this->decoded[i * image.cols + j];

      // Set the pixel value.
      image.at<char>(i, j) = pixel;
    }
  }

  // Display the image.
  cv::imshow("Image", image);
  cv::waitKey(0);
}

// Split the image in blocks of size 8x8, and save the result in variable
// 'blocks'.
void GrayscaleImage::splitBlocks() {
  // Clear the blocks vector.
  this->blocks.clear();

  // For each block row...
  for (int i = 0; i < this->blockGridHeight; i++) {
    // For each block column...
    for (int j = 0; j < this->blockGridWidth; j++) {
      // Create a new block.
      std::vector<char> block;

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
          unsigned char pixel =
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

  // For each block row...
  for (int i = 0; i < this->blockGridHeight; i++) {
    // For each block column...
    for (int j = 0; j < this->blockGridWidth; j++) {
      // Get the block.
      std::vector<char> block =
          this->blocks[i * this->blockGridWidth + j];

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
          unsigned char pixel = block[k * 8 + l];

          // Add the pixel value to the decoded vector, at the right position.
          this->decoded[pixelY * this->blockGridWidth * 8 + pixelX] = pixel;
        }
      }
    }
  }
}

// Static member variable to store the quantization table.
std::vector<int> GrayscaleImage::quantizationTable = {
   240, 60, 30, 30, 60, 60, 60, 60,
    60, 30, 30, 60, 60, 60, 60, 60,
    30, 30, 60, 60, 60, 60, 60, 60,
    30, 60, 60, 60, 60, 60, 60, 60,
    60, 60, 60, 60, 60, 60, 60, 60,
    60, 60, 60, 60, 60, 60, 60, 60,
    60, 60, 60, 60, 60, 60, 60, 60,
    60, 60, 60, 60, 60, 60, 60, 60};

// Quantize the given vec into two blocks using the quantization table.
void GrayscaleImage::quantize(const Transform::FourierTransform::vec &vec,
                              std::vector<char> &realBlock,
                              std::vector<char> &imagBlock) {
  // For element in the block...
  for (int i = 0; i < 64; i++) {
    // Get the quantization table value.
    unsigned char quantizationTableValue = GrayscaleImage::quantizationTable[i];

    // Set the element value to the quantized value.
    realBlock[i] = vec[i].real() / quantizationTableValue;
    imagBlock[i] = vec[i].imag() / quantizationTableValue;
  }
}

// Unquantize the given block using the quantization table.
void GrayscaleImage::unquantize(Transform::FourierTransform::vec &vec,
                                std::vector<char> &realBlock,
                                std::vector<char> &imagBlock) {
  for (int i = 0; i < 64; i++) {
    // Get the quantization table value.
    unsigned char quantizationTableValue = GrayscaleImage::quantizationTable[i];

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
    std::vector<char> block = this->blocks[i];

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
    std::vector<char> realBlock(64, 0);
    std::vector<char> imagBlock(64, 0);

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
  Transform::FourierTransform::TrivialTwoDimensionalInverseFourierTransformAlgorithm
      fft;

  this->entropyDecode();
  //size_t randomIndex = rand() % this->blocks.size();
  

  // For each block...
  for (size_t i = 0; i < this->blocks.size(); i++) {
    // Get the block.
    std::vector<char> realBlock = this->blocks[i];
    std::vector<char> imagBlock = this->imagBlocks[i];

    // If the block is the first block...
    /* if (i == randomIndex) {
      // Print the first block. Add a newline after group of 8 elements.
      for (size_t j = 0; j < realBlock.size(); j++) {
        std::cout << static_cast<int>(realBlock[j]) << " ";
        if ((j + 1) % 8 == 0) {
          std::cout << std::endl;
        }
      }
    } */

    // Unquantize the block.
    Transform::FourierTransform::vec vecBlock(64, 0);
    Transform::FourierTransform::vec outputVecBlock(64, 0);
    this->unquantize(vecBlock, realBlock, imagBlock);

    // Apply the Inverse Fourier transform to the block.
    fft(vecBlock, outputVecBlock);

    // Turn the output vec object into a block.
    std::vector<char> block(64, 0);
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
/*************************************************
{0, 0},
{0, 1}, {1, 0},
{2, 0}, {1, 1}, {0, 2},
{0, 3}, {1, 2}, {2, 1}, {3, 0},
{4, 0}, {3, 1}, {2, 2}, {1, 3}, {0, 4},
{0, 5}, {1, 4}, {2, 3}, {3, 2}, {4, 1}, {5, 0},
{6, 0}, {5, 1}, {4, 2}, {3, 3}, {2, 4}, {1, 5}, {0, 6},
{0, 7}, {1, 6}, {2, 5}, {3, 4}, {4, 3}, {5, 2}, {6, 1}, {7, 0},
{7, 1}, {6, 2}, {5, 3}, {4, 4}, {3, 5}, {2, 6}, {1, 7},
{2, 7}, {3, 6}, {4, 5}, {5, 4}, {6, 3}, {7, 2},
{7, 3}, {6, 4}, {5, 5}, {4, 6}, {3, 7},
{4, 7}, {5, 6}, {6, 5}, {7, 4},
{7, 5}, {6, 6}, {5, 7},
{6, 7}, {7, 6},
{7, 7}
*************************************************/

// Use entropy coding to encode all blocks.
void GrayscaleImage::entropyEncode() {
  // Clear the encoded vector.
  this->encoded.clear();

  // Initialize a new blocks vector.
  std::vector<std::vector<char>> blocks;

  // Add all blocks to the blocks vector.
  for (size_t i = 0; i < this->blocks.size(); i++) {
    blocks.push_back(this->blocks[i]);
  }

  // Add all imaginary blocks to the blocks vector.
  for (size_t i = 0; i < this->imagBlocks.size(); i++) {
    blocks.push_back(this->imagBlocks[i]);
  }

  // For each block...
  for (size_t i = 0; i < this->blocks.size(); i++) {
    // Get the block.
    std::vector<char> block = blocks[i];

    // Initialize a zigZag vector.
    std::vector<char> zigZagVector(64, 0);

    // For each step in zigzag path...
    for (int j = 0; j < 64; j++) {
      // Get the zigZag map coordinates.
      int x = GrayscaleImage::zigZagMap[j].first;
      int y = GrayscaleImage::zigZagMap[j].second;

      // Set the zigZag vector value to the block value.
      zigZagVector[j] = block[y * 8 + x];
    }

    // Traverse the zigZag vector.
    char zeroCounter = 0;

    // For each element in the zigZag vector...
    for (int j = 0; j < 64; j++) {
      // Get the element value.
      char element = zigZagVector[j];

      // If the element value is zero...
      if (element == 0) {
        // Increment the zero counter.
        zeroCounter++;
        continue;
      }

      // If the element value is not zero...
      this->encoded.push_back(zeroCounter);
      this->encoded.push_back(element);
    }

    this->encoded.push_back(zeroCounter);
    this->encoded.push_back(0x00);
  }
}

// Use entropy coding to decode all blocks.
void GrayscaleImage::entropyDecode() {
  // Clear the blocks vector.
  this->blocks.clear();
  this->imagBlocks.clear();

  // Create a new blocks vector.
  std::vector<std::vector<char>> blocks;

  // Create a new reconstrcuted zigZag vector.
  std::vector<char> reconstructedZigZagVector;

  // For each byte in the encoded vector...
  for (size_t i = 0; i < this->encoded.size(); i++) {
    // If the length of the reconstructed zigZag vector is 64...
    if (reconstructedZigZagVector.size() == 64) {
      // Initialize a new block.
      std::vector<char> block(64, 0);

      // For each step in zigzag path...
      for (int j = 0; j < 64; j++) {
        // Get the zigZag map coordinates.
        int x = GrayscaleImage::zigZagMap[j].first;
        int y = GrayscaleImage::zigZagMap[j].second;

        // Set the block value to the zigZag vector value.
        block[y * 8 + x] = reconstructedZigZagVector[j];
      }

      // Add the block to the blocks vector.
      blocks.push_back(block);

      // Clear the reconstructed zigZag vector.
      reconstructedZigZagVector.clear();
    }

    // Get the byte value.
    char byte = this->encoded[i];

    // Check zero counter.
    if (i % 2 == 0) {
      // Add as many zeros as the zero counter value.
      for (int j = 0; j < byte; j++) {
        reconstructedZigZagVector.push_back(0);
      }

      continue;
    }

    // Add the byte value to the reconstructed zigZag vector.
    reconstructedZigZagVector.push_back(byte);
  }

  // Print the length of the blocks vector.
  std::cout << "Blocks vector length: " << blocks.size() << std::endl;

  // Add the first half of the blocks vector to the blocks vector.
  for (size_t i = 0; i < blocks.size() / 2; i++) {
    this->blocks.push_back(blocks[i]);
  }
  
  // Add the second half of the blocks vector to the imaginary blocks vector.
  for (size_t i = blocks.size() / 2; i < blocks.size(); i++) {
    this->imagBlocks.push_back(blocks[i]);
  }
}

// Get the bitsize of the last loaded or encoded image.
unsigned int GrayscaleImage::getCompressedBitsize() const {
  return this->encoded.size() * 8;
}

void GrayscaleImage::waveletTransform(
    const std::shared_ptr<
        Transform::WaveletTransform::WaveletTransformAlgorithm>
        algorithm,
    bool direct) {
  using namespace Transform;
  using namespace WaveletTransform;

  // Convert the decoded image's values from char to real.
  std::vector<real> real_input;
  real_input.reserve(this->decoded.size());
  for (size_t i = 0; i < this->decoded.size(); i++) {
    real_input.emplace_back(static_cast<real>(this->decoded[i]));
  }

  // Allocate an output vector.
  std::vector<real> real_output(this->decoded.size(), 0);

  // Perform the direct wavelet transform.
  TwoDimensionalWaveletTransformAlgorithm algorithm_2d;
  if (direct) {
    algorithm_2d.directTransform(real_input, real_output, algorithm);
  } else {
    algorithm_2d.inverseTransform(real_input, real_output, algorithm);
  }

  /*
  // Calculate the maximum and minimum coefficients.
  real max_value = -std::numeric_limits<real>::max();
  real min_value = std::numeric_limits<real>::max();
  for (size_t i = 0; i < this->decoded.size(); i++) {
    if (real_output[i] > max_value) {
      max_value = real_output[i];
    }
    if (real_output[i] < min_value) {
      min_value = real_output[i];
    }
  }
  std::cout << min_value << " " << max_value << std::endl;
  */

  // Update the decoded image.
  for (size_t i = 0; i < this->decoded.size(); i++) {
    this->decoded[i] = static_cast<char>(real_output[i]);
  }
}