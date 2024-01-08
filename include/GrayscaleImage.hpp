#ifndef GRAYSCALE_IMAGE_HPP
#define GRAYSCALE_IMAGE_HPP

#include <cstdint>

#include "Real.hpp"
#include "WaveletTransform.hpp"

/**
 * @brief Represents a Grayscale Image.
 *
 * The GrayscaleImage class is designed to handle and manipulate grayscale images.
 * It provides functionalities for loading, encoding, decoding, and saving grayscale images.
 * 
 * Example usage:
 * @code
 * GrayscaleImage grayscaleImage;
 * 
 * bool success = grayscaleImage.loadStandard(image_path);
 * if (!success) {
 *  ...
 * }
 * 
 * grayscaleImage.encode();
 * grayscaleImage.decode();
 * grayscaleImage.display();
 * @endcode
 *
 * @note This class automatically converts common image formats to grayscale.
 * @note The image's width and height must be a multiple of 8.
 */
class GrayscaleImage {
 public:
  /**
   * @brief Load a grayscale image from file, using a standard format.
   * 
   * @param filename The path to the image file.
   * @return true If the image was loaded successfully.
   * @return false If the image could not be loaded.
   */
  bool loadStandard(const std::string &filename);

  /**
   * @brief Load a grayscale image from file, using a compressed format.
   * 
   * @param filename The path to the image file.
   * @return true If the image was loaded successfully.
   * @return false If the image could not be loaded.
   */
  bool loadCompressed(const std::string &filename);

  /**
   * @brief Save the image (in compressed format) to file.
   * 
   * @param filename The path to the image file to be saved.
   * @return true If the image was saved successfully.
   * @return false If the image could not be saved.
   */
  bool save(const std::string &filename);

  /**
   * @brief Encode the last loaded image.
   * 
   * Image encoding is done in 5 steps:
   * 1. Split the image in 8x8 blocks.
   * 2. Shift the block values to the range [-128, 127].
   * 3. Perform the 2D FFT on each block.
   * 4. Quantize the real and imaginary parts of each block.
   * 5. Store the quantized values in a sequence of bytes using entropy coding.
   */
  void encode();

  /**
   * @brief Decode the last loaded image.
   * 
   * Image decoding is done in 5 steps:
   * 1. Use entropy decoding to get the quantized values of each block.
   * 2. Unquantize the real and imaginary parts of each block.
   * 3. Perform the 2D inverse FFT on each block.
   * 4. Shift the block values back to the range [0, 255].
   * 5. Merge the blocks into a single image.
   */
  void decode();

  /**
   * @brief Display the last loaded or decoded image.
   * 
   * A window will pop up displaying the image. Any key can be pressed to close the window.
   * If multiple requests to display the image are made, the new image will be displayed 
   * as soon as the previous window is closed.
   * 
   * @note This function uses OpenCV to display the image.
   */
  void display();

  /**
   * @brief Get the bitsize of the last uncompressed image in memory.
   * 
   * This function returns the bitsize of the last uncompressed image in memory,
   * assuming that the image is represented as a sequence of bytes, each byte
   * representing a value in the range [0, 255], assigned to a pixel.
   * 
   * @return unsigned int The bitsize of the last loaded or decoded image.
   */
  unsigned int getStandardBitsize() const;

  /**
   * @brief Get the bitsize of the last compressed image in memory.
   * 
   * The compressed image is represented as a sequence of bytes. This function
   * counts the number of bytes in the sequence and returns 8 times that number.
   * 
   * @return unsigned int The bitsize of the last compressed image.
   */
  unsigned int getCompressedBitsize() const;

  /**
   * @brief Perform a direct wavelet transform on the decoded image and store the result into decoded.
   *
   * @todo Complete this documentation. 
   */
  void waveletTransform(
      const std::shared_ptr<
          Transform::WaveletTransform::WaveletTransformAlgorithm>
          algorithm,
      unsigned int levels);

  /**
   * Perform a direct wavelet transform on the decoded image, use thresholding, 
   * perform the inverse transform and store the result into decoded.
   * 
   * @todo Complete this documentation.
   */
  void denoise(const std::shared_ptr<
                   Transform::WaveletTransform::WaveletTransformAlgorithm>
                   direct_algorithm,
               const std::shared_ptr<
                   Transform::WaveletTransform::WaveletTransformAlgorithm>
                   inverse_algorithm,
               unsigned int levels, Transform::real threshold,
               bool hard_thresholding);

 private:
  // Split the image in blocks of size 8x8, and save the result in variable
  // 'blocks'.
  void splitBlocks();

  // Merge the blocks in variable 'blocks' and save the result in variable
  // 'decoded'.
  void mergeBlocks();

  // Quantize the given vec using the quantization table.
  void quantize(const Transform::FourierTransform::vec &vec,
                std::vector<int8_t> &realBlock, std::vector<int8_t> &imagBlock);

  // Unquantize the given vec using the quantization table.
  void unquantize(Transform::FourierTransform::vec &vec,
                  std::vector<int8_t> &realBlock, std::vector<int8_t> &imagBlock);

  // Use entropy coding to encode all blocks.
  void entropyEncode();

  // Use entropy coding to decode all blocks.
  void entropyDecode();

  // Static member variable to store the quantization table.
  static std::vector<int> quantizationTable;

  // Static member variable to store the zigZag map.
  static std::vector<std::pair<int, int>> zigZagMap;

  // The image in uncompressed form.
  std::vector<uint8_t> decoded;

  // The image in compressed form (expressed as a sequence of bytes).
  std::vector<uint8_t> encoded;

  // An array of 8x8 blocks. Each block is a vector of 64 elements.
  std::vector<std::vector<int8_t>> blocks;
  std::vector<std::vector<int8_t>> imagBlocks;

  // Block grid width.
  int blockGridWidth;

  // Block grid height.
  int blockGridHeight;
};

#endif  // GRAYSCALE_IMAGE_HPP