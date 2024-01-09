#ifndef GRAYSCALE_IMAGE_HPP
#define GRAYSCALE_IMAGE_HPP

/**
 * @file GrayscaleImage.hpp.
 * @brief Declares a class for handling Grayscale images.
 */

#include <cstdint>

#include "WaveletTransform.hpp"

/**
 * @brief Represents a Grayscale Image.
 *
 * The GrayscaleImage class is designed to handle and manipulate grayscale
 * images. It provides functionalities for loading, encoding, decoding, and
 * saving grayscale images.
 *
 * Example usage:
 * @code
 * GrayscaleImage grayscaleImage;
 *
 * bool success = grayscaleImage.loadStandard("image.jpg");
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
  // Get the encoded image.
  std::vector<uint8_t> getEncoded();

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
  bool saveCompressed(const std::string &filename);

  /**
   * @brief Save the latest loaded or decoded image to file.
   *
   * @param filename The path to the image file to be saved.
   * @return true If the image was saved successfully.
   * @return false If the image could not be saved.
   */
  bool saveImage(const std::string &filename);

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
   * A window will pop up displaying the image. Any key can be pressed to close
   * the window. If multiple requests to display the image are made, the new
   * image will be displayed as soon as the previous window is closed.
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
   * @brief Perform a direct wavelet transform on the last loaded image and
   * replace it with an image representing its DWT.
   *
   * @param algorithm A unique pointer to the algorithm to use for the DWT, it
   * will be moved when calling the function.
   * @param levels The number of levels for the DWT.
   *
   * @note The image must be square and the number of pixels in a row must be a
   * power of 2 and greater than 1.
   * @note Unless omp_set_num_threads() has been called, the algorithm will use
   * all available OpenMP threads.
   */
  void waveletTransform(
      std::unique_ptr<Transform::WaveletTransform::WaveletTransformAlgorithm>
          &algorithm,
      unsigned int levels);

  /**
   * @brief Denoise the last loaded image using thresholding.
   *
   * The method applies the DWT to the image, uses thresholding with the
   * specified parameter and then applies the IDWT on the result and updates the
   * image.
   *
   * @param algorithm A unique pointer to the algorithm to use for the DWT and
   * IDWT, it will be moved when calling the function.
   * @param levels The number of levels for the DWT and IDWT.
   * @param threshold The threshold for the thresholding step.
   * @param use_hard_thresholding If true, hard thresholding is used, otherwise
   * soft thresholding is used instead.
   *
   * @note Unless omp_set_num_threads() has been called, the algorithm will use
   * all available OpenMP threads.
   * @note The image must be square and the number of pixels in a row must be a
   * power of 2 and greater than 1.
   */
  void denoise(
      std::unique_ptr<Transform::WaveletTransform::WaveletTransformAlgorithm>
          &algorithm,
      unsigned int levels, Transform::real threshold,
      bool use_hard_thresholding);

 private:
  /**
   * @brief Split the image in 8x8 blocks and store the result in variable
   * 'blocks'.
   *
   * @note This function assumes that the image's width and height are a
   * multiple of 8.
   */
  void splitBlocks();

  /**
   * @brief Merge the blocks into a single image and store the result in
   * variable 'decoded'.
   *
   * @note This function assumes that the image's width and height are a
   * multiple of 8.
   */
  void mergeBlocks();

  /**
   * @brief Use a quantization table to quantize the given vec.
   *
   * This function quantizes the given vec using the quantization table. Each
   * element of the vec is divided by the corresponding element of the
   * quantization table. Then, the result is split into a real and imaginary
   * part, and each part is stored in a separate block.
   *
   * @param vec The vec to be quantized.
   * @param realBlock The resulting real part of the quantized vec.
   * @param imagBlock The resulting imaginary part of the quantized vec.
   */
  void quantize(const Transform::FourierTransform::vec &vec,
                std::vector<int8_t> &realBlock, std::vector<int8_t> &imagBlock);

  /**
   * @brief Use a quantization table to unquantize the given vec.
   *
   * This function unquantizes the given real and imaginary parts using the
   * quantization table. The real and imaginary parts are combined into a single
   * vec, which is multiplied by the corresponding element of the quantization
   * table, and stored in variable 'vec'.
   *
   * @param vec The resulting vec.
   * @param realBlock The real part of the quantized vec.
   * @param imagBlock The imaginary part of the quantized vec.
   */
  void unquantize(Transform::FourierTransform::vec &vec,
                  std::vector<int8_t> &realBlock,
                  std::vector<int8_t> &imagBlock);

  /**
   * @brief Use a simplified version of entropy coding to encode all blocks.
   *
   * This function uses a simplified version of entropy coding to encode all
   * blocks. The encoding is done in 3 steps:
   * 1. Use the zig-zag map to convert the block into a linear sequence of
   * values.
   * 2. Use run-length encoding to encode the sequence of values, by storing the
   *    number of consecutive zeros and the next non-zero value, along with the
   *    special end-of-block symbol.
   * 3. Store the encoded sequence of values in the 'encoded' variable.
   */
  void entropyEncode();

  /**
   * @brief Use a simplified version of entropy coding to decode all blocks.
   *
   * This function uses a simplified version of entropy coding to decode all
   * blocks. The decoding is done in 3 steps:
   * 1. Use run-length decoding to decode the sequence of values, by reading the
   *    number of consecutive zeros and the next non-zero value, along with the
   *    special end-of-block symbol.
   * 2. Use the zig-zag map to convert the sequence of values into a block.
   * 3. Store the decoded block in the 'blocks' variable.
   */
  void entropyDecode();

  /**
   * @brief The quantization table.
   *
   * This is a static member variable, which means that it is shared by all
   * instances of the GrayscaleImage class.
   *
   * The quantization table is used to quantize and unquantize the real and
   * imaginary parts of each block. The table is used to divide the real and
   * imaginary parts of each block by the corresponding element of the table.
   *
   * Very efficient quantization tables can be found online, to be used with the
   * JPEG standard. These quantization tables are designed to be used with the
   * cosine transform, which is not the same as the Fourier transform. In our
   * case, we use a very simple quantization table, which is eyeballed to give
   * good results, as quantization effectiveness is dependent on the
   * relationship between human visual perception of image frequencies and the
   * actual frequencies of the image, of which we have no precise knowledge.
   *
   * 200 100 100 100 100 100 100 100
   * 100 100 100 100 100 100 100 100
   * 100 100 100 100 100 100 100 100
   * 100 100 100 100 100 100 100 100
   * 100 100 100 100 100 100 100 100
   * 100 100 100 100 100 100 100 100
   * 100 100 100 100 100 100 100 100
   * 100 100 100 100 100 100 100 100
   */
  static std::vector<int> quantizationTable;

  /**
   * @brief The zig-zag map.
   *
   * This is a static member variable, which means that it is shared by all
   * instances of the GrayscaleImage class. The zig-zag map is used to convert a
   * block into a linear sequence of values. The map associates each element of
   * the block (expressed as a pair of integer coordinates) with a position
   * (their index) in the linear sequence.
   *
   * In our case, the zig-zag map is not size-dependent, as we only use 8x8
   * blocks. Using this restriction to our advantage, we can store the whole map
   * without having to compute it every time.
   *
   * {0, 0},
   * {0, 1}, {1, 0},
   * {2, 0}, {1, 1}, {0, 2},
   * {0, 3}, {1, 2}, {2, 1}, {3, 0},
   * {4, 0}, {3, 1}, {2, 2}, {1, 3}, {0, 4},
   * {0, 5}, {1, 4}, {2, 3}, {3, 2}, {4, 1}, {5, 0},
   * {6, 0}, {5, 1}, {4, 2}, {3, 3}, {2, 4}, {1, 5}, {0, 6},
   * {0, 7}, {1, 6}, {2, 5}, {3, 4}, {4, 3}, {5, 2}, {6, 1}, {7, 0},
   * {7, 1}, {6, 2}, {5, 3}, {4, 4}, {3, 5}, {2, 6}, {1, 7},
   * {2, 7}, {3, 6}, {4, 5}, {5, 4}, {6, 3}, {7, 2},
   * {7, 3}, {6, 4}, {5, 5}, {4, 6}, {3, 7},
   * {4, 7}, {5, 6}, {6, 5}, {7, 4},
   * {7, 5}, {6, 6}, {5, 7},
   * {6, 7}, {7, 6},
   * {7, 7}
   */
  static std::vector<std::pair<int, int>> zigZagMap;

  /**
   * @brief The image in uncompressed form (expressed as a sequence of bytes).
   *
   * This variable is used to store the image in uncompressed form, as a
   * sequence of bytes. Each byte represents a value in the range [0, 255],
   * assigned to a pixel.
   */
  std::vector<uint8_t> decoded;

  /**
   * @brief The image in compressed form (expressed as a sequence of bytes).
   *
   * The first half of this variable is used to store the real parts of the
   * quantized values of each block. The second half is used to store the
   * imaginary parts.
   */
  std::vector<uint8_t> encoded;

  /**
   * @brief An array of blocks.
   *
   * This variable is used to store the image in the form of blocks. Each block
   * is represented as a sequence of bytes, each byte representing a value. The
   * contents of the blocks do not always correspond to the same image format,
   * as this variable is used to store the image in different stages of the
   * encoding/decoding process.
   */
  std::vector<std::vector<int8_t>> blocks;

  /**
   * @brief An array of blocks, containing imaginary parts.
   *
   * This variable is used to store the imaginary part of the result of the 2D
   * FFT on each block. This is used during the encoding process.
   *
   * @see blocks
   */
  std::vector<std::vector<int8_t>> imagBlocks;

  /**
   * @brief The image width, expressed in blocks.
   */
  int blockGridWidth;

  /**
   * @brief The image height, expressed in blocks.
   *
   */
  int blockGridHeight;
};

#endif  // GRAYSCALE_IMAGE_HPP