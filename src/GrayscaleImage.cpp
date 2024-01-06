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
      unsigned char pixel = image.at<unsigned char>(i, j);

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
      unsigned char pixel = this->decoded[i * image.cols + j];

      // Set the pixel value.
      image.at<unsigned char>(i, j) = pixel;
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
      std::vector<unsigned char> block;

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
      std::vector<unsigned char> block = this->blocks[i * this->blockGridWidth + j];

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
          unsigned char pixel = block[k * 8 + l];

          // Add the pixel value to the decoded vector, at the right position.
          this->decoded[pixelY * this->blockGridWidth * 8 + pixelX] = pixel;
        }
      }
    }
  }
}

// Static member variable to store the quantization table.
std::vector<unsigned char> GrayscaleImage::quantizationTable = {
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99};

// Quantize the given block using the quantization table.
std::vector<unsigned char> GrayscaleImage::quantize(
    const std::vector<unsigned char> &block)
{
  // Create a new block.
  std::vector<unsigned char> quantizedBlock;

  // For element in the block...
  for (int i = 0; i < 64; i++)
  {
    // Get the element value.
    unsigned char element = block[i];

    // Get the quantization table value.
    unsigned char quantizationTableValue = GrayscaleImage::quantizationTable[i];

    // Quantize the element value.
    unsigned char quantizedElement = element / quantizationTableValue;

    // Add the quantized element value to the quantized block.
    quantizedBlock.push_back(quantizedElement);
  }

  return quantizedBlock;
}

// Unquantize the given block using the quantization table.
std::vector<unsigned char> GrayscaleImage::unquantize(
    const std::vector<unsigned char> &block)
{
  // Create a new block.
  std::vector<unsigned char> unquantizedBlock;

  // For element in the block...
  for (int i = 0; i < 64; i++)
  {
    // Get the element value.
    unsigned char element = block[i];

    // Get the quantization table value.
    unsigned char quantizationTableValue = GrayscaleImage::quantizationTable[i];

    // Unquantize the element value.
    unsigned char unquantizedElement = element * quantizationTableValue;

    // Add the unquantized element value to the unquantized block.
    unquantizedBlock.push_back(unquantizedElement);
  }

  return unquantizedBlock;
}