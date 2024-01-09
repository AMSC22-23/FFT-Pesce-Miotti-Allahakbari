#include "cuda_main.hpp"

/**
 * @file cuda_main.cpp.
 * @brief Defines the main function for CUDA-related tests.
 */

// TODO: Remove commented code and improve comments.

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "FourierTransformCalculator.hpp"

int cuda_main(int argc, char *argv[]) {
  using namespace Transform;
  using namespace FourierTransform;

  if (argc > 3) {
    std::cerr << "Incorrect arguments!\n"
              << "Argument 1: cuda\n"
              << "Argument 2: image path (default: ../img/image.jpg)\n"
              << std::endl;
    return 1;
  }

  std::string image_path = "../img/image.jpg";

  // Get which algorithm to choose.
  if (argc >= 3) image_path = std::string(argv[2]);

  // Load the image (grayscale)
  cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

  if (image.empty()) {
    std::cerr << "Error: Unable to load the image." << std::endl;
    return EXIT_FAILURE;
  }

  // Check image properties.
  std::cout << "Image size: " << image.size() << std::endl;
  std::cout << "Image type: " << image.type() << std::endl;

  // Convert the image to a vector of complex numbers.
  vec input_sequence;
  vec output_sequence;

  input_sequence.reserve(image.rows * image.cols);
  output_sequence.reserve(image.rows * image.cols);

  for (int i = 0; i < image.rows; i++)
    for (int j = 0; j < image.cols; j++)
      input_sequence.emplace_back(image.at<uchar>(i, j), 0);

  std::unique_ptr<FourierTransformAlgorithm> algorithm =
      std::make_unique<BlockFFTGPU2D>();

  // Set the direct transform.
  algorithm->setBaseAngle(-std::numbers::pi_v<real>);

  // Execute the algorithm and calculate the time.
  std::cout << algorithm->calculateTime(input_sequence, output_sequence) << "μs"
            << std::endl;

  // Ensure the image dimensions are divisible by 8
  int height = image.rows;
  int width = image.cols;

  int new_height = height - (height % 8);
  int new_width = width - (width % 8);

  cv::Rect roi(0, 0, new_width, new_height);
  image = image(roi);

  // Segment the image into 8x8 blocks and store with row/column indices
  std::vector<std::pair<int, int>>
      blockIndices;  // For storing row/column indices
  std::vector<cv::Mat> blocks;

  for (int y = 0; y < new_height; y += 8) {
    for (int x = 0; x < new_width; x += 8) {
      cv::Mat block = image(cv::Rect(x, y, 8, 8));
      blocks.push_back(block);

      // Store row and column indices as a pair
      blockIndices.push_back(std::make_pair(y / 8, x / 8));
    }
  }

  // Perform Complex Double Precision DFT on each block
  for (size_t bid = 0; bid < blocks.size(); ++bid) {
    cv::Mat dftInput;
    blocks[bid].convertTo(dftInput,
                          CV_64F);  // Convert block to double precision

    cv::Mat dftOutput;
    cv::dft(dftInput, dftOutput, cv::DFT_COMPLEX_OUTPUT);
    std::vector<cv::Mat> channels;
    cv::split(dftOutput, channels);

    cv::Mat realPart = channels[0];  // Real part of the DFT output
    cv::Mat imagPart = channels[1];
    // Process the DFT result (you can do further operations here)
    double epsilon = 1e-4;

    // std::cout << "Calculations at block: (" << blockIndices[bid].first << ",
    // "
    //           << blockIndices[bid].second << ")" << std::endl;
    for (int i = 0; i < 8; i++)
      for (int j = 0; j < 8; j++) {
        if (std::fabs(
                output_sequence[(blockIndices[bid].first * 8 + i) * image.rows +
                                blockIndices[bid].second * 8 + j]
                    .real() -
                realPart.at<double>(i, j)) > epsilon ||
            std::fabs(
                output_sequence[(blockIndices[bid].first * 8 + i) * image.rows +
                                blockIndices[bid].second * 8 + j]
                    .imag() -
                imagPart.at<double>(i, j)) > epsilon) {
          std::cout << "Calculations at: " << i << ", " << j << " GPU: "
                    << output_sequence[(blockIndices[bid].first * 8 + i) *
                                           image.rows +
                                       blockIndices[bid].second * 8 + j]
                    << ", CPU: (" << realPart.at<double>(i, j) << ", "
                    << imagPart.at<double>(i, j) << ")" << std::endl;
        }
      }
  }

  return EXIT_SUCCESS;
}