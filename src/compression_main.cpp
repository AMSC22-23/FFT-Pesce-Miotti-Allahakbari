#include "compression_main.hpp"

/**
 * @file compression_main.cpp.
 * @brief Defines the main function for compression-related tests.
 */

#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "GrayscaleImage.hpp"

int compression_main(int argc, char *argv[]) {
  using namespace Transform;
  using namespace FourierTransform;

  std::string image_path = "../img/cat.jpg";

  // We expect at most 1 extra argument.
  if (argc > 3) {
    std::cerr << "Incorrect arguments!\n"
              << "Argument 1: jpeg\n"
              << "Argument 2: image path (default: " << image_path << ")\n"
              << std::endl;
    return 1;
  }

  // Get the image path.
  if (argc > 2) image_path = std::string(argv[2]);

  // Create a GrayscaleImage object.
  GrayscaleImage grayscaleImage;

  // Load the image.
  std::cout << "Loading image..." << std::endl;
  bool success = grayscaleImage.loadStandard(image_path);

  // Check if the image was loaded successfully.
  if (!success) {
    std::cout << "Failed to load image." << std::endl;
    return 1;
  }

  // Encode the image.
  std::cout << "Encoding image..." << std::endl;
  grayscaleImage.encode();

  // Get the bitsizes of the image.
  unsigned int bitsize = grayscaleImage.getStandardBitsize();
  unsigned int compressed_bitsize = grayscaleImage.getCompressedBitsize();

  // Get the compression ratio.
  double compression_ratio =
      static_cast<double>(compressed_bitsize) / static_cast<double>(bitsize);
  std::cout << "Compression ratio: " << compression_ratio << std::endl;

  // Save the compressed image.
  std::cout << "Saving compressed image..." << std::endl;
  grayscaleImage.saveCompressed("../img/compressed-image.data");

  // Create a new GrayscaleImage object.
  GrayscaleImage grayscaleImage2;

  // Load the compressed image.
  std::cout << "Loading compressed image..." << std::endl;
  success = grayscaleImage2.loadCompressed("../img/compressed-image.data");

  // Check if the image was loaded successfully.
  if (!success) {
    std::cout << "Failed to load compressed image." << std::endl;
    return 1;
  }

  // Get the encoded bytes of grayscaleImages with getEncoded().
  const std::vector<uint8_t> encoded_bytes = grayscaleImage.getEncoded();
  const std::vector<uint8_t> encoded_bytes2 = grayscaleImage2.getEncoded();

  // Decode the image.
  std::cout << "Decoding image..." << std::endl;
  grayscaleImage2.decode();

  // Display the image.
  grayscaleImage2.display();

  return 0;
}