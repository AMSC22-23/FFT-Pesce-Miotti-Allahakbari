#ifndef GRAYSCALE_IMAGE_HPP
#define GRAYSCALE_IMAGE_HPP

#include "Real.hpp"
#include "WaveletTransform.hpp"

// A class that represents a grayscale image.
// This class may be used to load, save, encode, decode and display grayscale
// images. Please note that the image's width and height must be a multiple
// of 8.
class GrayscaleImage {
 public:
  // Load regular image from file.
  bool loadStandard(const std::string &filename);

  // Load compressed image from file.
  bool loadCompressed(const std::string &filename);

  // Save compressed image to file.
  bool save(const std::string &filename);

  // Encode the last loaded or decoded image.
  void encode();

  // Decode the last loaded or encoded image.
  void decode();

  // Display the last loaded or decoded image.
  void display();

  // Get the bitsize of the last loaded or decoded image.
  unsigned int getStandardBitsize() const;

  // Get the bitsize of the last loaded or encoded image.
  unsigned int getCompressedBitsize() const;

  // Perform a wavelet transform on the decoded image. It is direct is "direct"
  // is true and inverse otherwise.
  void waveletTransform(
      const std::shared_ptr<
          Transform::WaveletTransform::WaveletTransformAlgorithm>
          algorithm,
      unsigned int levels, bool direct);

 private:
  // Split the image in blocks of size 8x8, and save the result in variable
  // 'blocks'.
  void splitBlocks();

  // Merge the blocks in variable 'blocks' and save the result in variable
  // 'decoded'.
  void mergeBlocks();

  // Quantize the given vec using the quantization table.
  void quantize(const Transform::FourierTransform::vec &vec,
                std::vector<char> &realBlock, std::vector<char> &imagBlock);

  // Unquantize the given vec using the quantization table.
  void unquantize(Transform::FourierTransform::vec &vec,
                  std::vector<char> &realBlock, std::vector<char> &imagBlock);

  // Use entropy coding to encode all blocks.
  void entropyEncode();

  // Use entropy coding to decode all blocks.
  void entropyDecode();

  // Static member variable to store the quantization table.
  static std::vector<int> quantizationTable;

  // Static member variable to store the zigZag map.
  static std::vector<std::pair<int, int>> zigZagMap;

  // The image in uncompressed form.
  std::vector<char> decoded;

  // The image in compressed form (expressed as a sequence of bytes).
  std::vector<char> encoded;

  // An array of 8x8 blocks. Each block is a vector of 64 elements.
  std::vector<std::vector<char>> blocks;
  std::vector<std::vector<char>> imagBlocks;

  // Block grid width.
  int blockGridWidth;

  // Block grid height.
  int blockGridHeight;
};

#endif  // GRAYSCALE_IMAGE_HPP