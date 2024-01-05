#include "GrayscaleImage.hpp"

#include <iostream>

int main()
{
  using namespace FourierTransform;

  // Create a GrayscaleImage object.
  GrayscaleImage grayscaleImage;

  // Load the image.
  std::cout << "Loading image..." << std::endl;
  bool success = grayscaleImage.loadStandard("../img/image.jpg");

  // Check if the image was loaded successfully.
  if (!success)
  {
    std::cout << "Failed to load image." << std::endl;
    return 1;
  }

  // Get the bitsize of the image.
  unsigned int bitsize = grayscaleImage.getStandardBitsize();

  // Print the bitsize.
  std::cout << "The loaded image has a bitsize of " << bitsize << " bits." << std::endl;

  return 0;
}