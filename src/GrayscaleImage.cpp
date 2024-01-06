#include "GrayscaleImage.hpp"

#include <opencv2/opencv.hpp>

// Load regular image from file.
bool GrayscaleImage::loadStandard(const std::string &filename)
{
  cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
  if (image.empty())
  {
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
  for (int i = 0; i < image.rows; i++)
  {
    // For each column...
    for (int j = 0; j < image.cols; j++)
    {
      // Get the pixel value.
      char pixel = image.at<char>(i, j);

      // Add the pixel value to the decoded image.
      this->decoded.push_back(pixel);
    }
  }

  return true;
}

// Get the bitsize of the last loaded or decoded image.
unsigned int GrayscaleImage::getStandardBitsize() const
{
  return this->blockGridWidth * this->blockGridHeight * 64 * 8;
}

// Display the last loaded or decoded image.
void GrayscaleImage::display()
{
  // Create a new image.
  cv::Mat image(this->blockGridHeight * 8, this->blockGridWidth * 8, CV_8UC1);

  // For each row...
  for (int i = 0; i < image.rows; i++)
  {
    // For each column...
    for (int j = 0; j < image.cols; j++)
    {
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
void GrayscaleImage::splitBlocks()
{
  // Clear the blocks vector.
  this->blocks.clear();

  // For each block row...
  for (int i = 0; i < this->blockGridHeight; i++)
  {
    // For each block column...
    for (int j = 0; j < this->blockGridWidth; j++)
    {
      // Create a new block.
      std::vector<char> block;

      // For each row in the block...
      for (int k = 0; k < 8; k++)
      {
        // For each column in the block...
        for (int l = 0; l < 8; l++)
        {
          // Get the top-left pixel coordinates of the block.
          int x = j * 8;
          int y = i * 8;

          // Get the pixel coordinates.
          int pixelX = x + l;
          int pixelY = y + k;

          // Get the pixel value.
          char pixel =
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
void GrayscaleImage::mergeBlocks()
{
  // Clear the decoded vector.
  this->decoded.clear();

  // For each block row...
  for (int i = 0; i < this->blockGridHeight; i++)
  {
    // For each block column...
    for (int j = 0; j < this->blockGridWidth; j++)
    {
      // Get the block.
      std::vector<char> block = this->blocks[i * this->blockGridWidth + j];

      // For each row in the block...
      for (int k = 0; k < 8; k++)
      {
        // For each column in the block...
        for (int l = 0; l < 8; l++)
        {
          // Get the top-left pixel coordinates of the block.
          int x = j * 8;
          int y = i * 8;

          // Get the pixel coordinates.
          int pixelX = x + l;
          int pixelY = y + k;

          // Get the pixel value.
          char pixel = block[k * 8 + l];

          // Add the pixel value to the decoded vector, at the right position.
          this->decoded[pixelY * this->blockGridWidth * 8 + pixelX] = pixel;
        }
      }
    }
  }
}

// Static member variable to store the quantization table.
std::vector<char> GrayscaleImage::quantizationTable = {
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99};

// Quantize the given block using the quantization table.
std::vector<char> GrayscaleImage::quantize(
    const std::vector<char> &block)
{
  // Create a new block.
  std::vector<char> quantizedBlock;

  // For element in the block...
  for (int i = 0; i < 64; i++)
  {
    // Get the element value.
    char element = block[i];

    // Get the quantization table value.
    char quantizationTableValue = GrayscaleImage::quantizationTable[i];

    // Quantize the element value.
    char quantizedElement = element / quantizationTableValue;

    // Add the quantized element value to the quantized block.
    quantizedBlock.push_back(quantizedElement);
  }

  return quantizedBlock;
}

// Unquantize the given block using the quantization table.
std::vector<char> GrayscaleImage::unquantize(
    const std::vector<char> &block)
{
  // Create a new block.
  std::vector<char> unquantizedBlock;

  // For element in the block...
  for (int i = 0; i < 64; i++)
  {
    // Get the element value.
    char element = block[i];

    // Get the quantization table value.
    char quantizationTableValue = GrayscaleImage::quantizationTable[i];

    // Unquantize the element value.
    char unquantizedElement = element * quantizationTableValue;

    // Add the unquantized element value to the unquantized block.
    unquantizedBlock.push_back(unquantizedElement);
  }

  return unquantizedBlock;
}

// Static member variable to store the zigZag map.
std::vector<std::pair<int, int>> GrayscaleImage::zigZagMap = {{0, 0}, {0, 1}, {1, 0}, {2, 0}, {1, 1}, {0, 2}, {0, 3}, {1, 2}, {2, 1}, {3, 0}, {4, 0}, {3, 1}, {2, 2}, {1, 3}, {0, 4}, {0, 5}, {1, 4}, {2, 3}, {3, 2}, {4, 1}, {5, 0}, {6, 0}, {5, 1}, {4, 2}, {3, 3}, {2, 4}, {1, 5}, {0, 6}, {0, 7}, {1, 6}, {2, 5}, {3, 4}, {4, 3}, {5, 2}, {6, 1}, {7, 0}, {7, 1}, {6, 2}, {5, 3}, {4, 4}, {3, 5}, {2, 6}, {1, 7}, {2, 7}, {3, 6}, {4, 5}, {5, 4}, {6, 3}, {7, 2}, {7, 3}, {6, 4}, {5, 5}, {4, 6}, {3, 7}, {4, 7}, {5, 6}, {6, 5}, {7, 4}, {7, 5}, {6, 6}, {5, 7}, {6, 7}, {7, 6}, {7, 7}};
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
void GrayscaleImage::entropyEncode()
{
  // Clear the encoded vector.
  this->encoded.clear();

  // Note the unassigned bits in the last byte.
  int unassignedBits = 0;

  // For each block...
  for (size_t i = 0; i < this->blocks.size(); i++)
  {
    // Create a linearized zigZag vector.
    std::vector<char> zigZagVector;

    // Get the already quantized block.
    std::vector<char> block = this->blocks[i];

    // For each element in the block...
    for (int j = 0; j < 64; j++)
    {
      // Get the zigZag map coordinates.
      int x = GrayscaleImage::zigZagMap[j].first;
      int y = GrayscaleImage::zigZagMap[j].second;

      // Get the element value.
      char element = block[y * 8 + x];

      // Add the element value to the zigZag block.
      zigZagVector.push_back(element);
    }

    // Initialize a counter for the number of zeros.
    int zeroCounter = 0;

    // Begin the entropy encoding by iterating over the zigZag vector.
    for (size_t j = 0; j < zigZagVector.size(); j++)
    {
      // Get the element value.
      char element = zigZagVector[j];

      // If the element value is zero...
      if (element == 0)
      {
        // Increment the zero counter.
        zeroCounter++;
        continue;
      }

      // Else, if the zero counter is greater than 15...
      while (zeroCounter > 15)
      {
        // Add the special code to the encoded vector.
        this->encoded.push_back(0xF0);
        this->encoded.push_back(0x00);

        // Decrement the zero counter by 16.
        zeroCounter -= 16;
      }

      // Get the bit size of the element value.
      int bitSize = 0;
      char temp = element;
      if (temp < 0)
      {
        bitSize = 8;
      }
      else
      {
        while (temp > 0)
        {
          temp = temp << 1;
          bitSize++;
        }
      }

      // Initialize the first byte.
      char firstByte = 0;

      // Set the first 4 bits of the first byte to the zero counter.
      firstByte |= zeroCounter << 4;

      // Set the last 4 bits of the first byte to the bit size.
      firstByte |= bitSize;

      // Add the first byte to the encoded vector, by filling the unassigned
      // bits of the last byte first, and then adding the first byte.
      // This operation does not change the number of unassigned bits, since
      // we're adding a byte.
      if (unassignedBits == 0)
      {
        this->encoded.push_back(firstByte);
      }
      else
      {
        this->encoded[this->encoded.size() - 1] |= firstByte >> (8 - unassignedBits);
        this->encoded.push_back(firstByte << unassignedBits);
      }

      // Add the element value to the encoded vector, by filling the unassigned
      // bits of the last byte first, and then adding the element value.
      // This operation changes the number of unassigned bits, since we're
      // adding a variable number of bits.
      if (unassignedBits == 0)
      {
        this->encoded.push_back(element << (8 - bitSize));
      }
      else
      {
        this->encoded[this->encoded.size() - 1] |= element >> (8 - unassignedBits) << (8 - bitSize);
        if (bitSize > unassignedBits)
        {
          this->encoded.push_back(element << (8 - bitSize + unassignedBits));
        }
      }

      // Update the number of unassigned bits.
      unassignedBits = (unassignedBits + bitSize) % 8;

      // Reset the zero counter.
      zeroCounter = 0;
    }
  }

  // Add an EOB code to the encoded vector.
  // We're not dealing with unassigned bits here, since we're adding only zeros.
  this->encoded.push_back(0x00);
}