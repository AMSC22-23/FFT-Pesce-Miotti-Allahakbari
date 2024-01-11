#include "cuda_main.hpp"

/**
 * @file cuda_main.cpp.
 * @brief Defines the main function for CUDA-related tests.
 */

#include <chrono>
#include <cstdint>
#include <iostream>
#include <numbers>
#include <opencv2/opencv.hpp>
#include <string>

#include "FourierTransformCalculator.hpp"
#include "GrayscaleImage.hpp"
#include "Utility.hpp"

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

  const std::string default_path = "../img/cat.jpg";
  constexpr real precision =
      std::max(std::numeric_limits<real>::epsilon() * 1e5, 1e-4);
  constexpr unsigned int num_cuda_streams = 8;

  // Get the image path.
  std::string image_path = default_path;
  if (argc >= 3) image_path = std::string(argv[2]);

  // Load the image (grayscale).
  cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

  // Check for errors.
  if (image.empty()) {
    std::cerr << "Error: Unable to load the image." << std::endl;
    return 1;
  }

  // Print image properties.
  std::cout << "Image size: " << image.size() << std::endl;
  std::cout << "Image type: " << image.type() << std::endl;

  // Ensure the image dimensions are divisible by 8
  const size_t height = image.rows;
  const size_t width = image.cols;
  const size_t total_size = height * width;
  if (height % 8 != 0 || width % 8 != 0) {
    std::cerr << "Error: The image sizes are not divisible by 8." << std::endl;
    return 1;
  }

  // Allocate space for the needed sequences.
  vec input_sequence;
  input_sequence.reserve(total_size);
  vec fft_output_sequence(total_size, 0);
  vec ifft_output_sequence(total_size, 0);

  // Convert the image to a vector of complex numbers.
  for (size_t i = 0; i < height; i++)
    for (size_t j = 0; j < width; j++)
      input_sequence.emplace_back(image.at<uint8_t>(i, j), 0);

  // Create the algorithms.
  TwoDimensionalDirectBlockFFTGPU fft2d;
  TwoDimensionalInverseBlockFFTGPU ifft2d;

  // Perform a warm-up run.
  fft2d(input_sequence, fft_output_sequence, num_cuda_streams);

  // Execute the direct algorithm and calculate the time.
  auto start = std::chrono::high_resolution_clock::now();
  fft2d(input_sequence, fft_output_sequence, num_cuda_streams);
  auto end = std::chrono::high_resolution_clock::now();
  auto fft_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();

  std::cout << "Execution time for GPU FFT: " << fft_duration << " μs"
            << std::endl;

  // Execute the inverse algorithm and calculate the time.
  start = std::chrono::high_resolution_clock::now();
  ifft2d(fft_output_sequence, ifft_output_sequence, num_cuda_streams);
  end = std::chrono::high_resolution_clock::now();
  auto ifft_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();

  std::cout << "Execution time for GPU IFFT: " << ifft_duration << " μs"
            << std::endl;

  // Check if the inverse result is the same as the input sequence.
  if (!CompareVectors(input_sequence, ifft_output_sequence, precision, false,
                      false))
    std::cerr << "Errors detected in GPU inverse Fourier transform."
              << std::endl;

  // Check the direct results with OpenCV.
  // Create an OpenCV image.
  cv::Rect roi(0, 0, width, height);
  image = image(roi);

  // Segment the image into 8x8 blocks and store row/column indices
  std::vector<std::pair<size_t, size_t>> blockIndices;
  std::vector<cv::Mat> blocks;

  for (size_t y = 0; y < height; y += 8) {
    for (size_t x = 0; x < width; x += 8) {
      cv::Mat block = image(cv::Rect(x, y, 8, 8));
      blocks.push_back(block);

      // Store row and column indices as a pair.
      blockIndices.push_back(std::make_pair(y / 8, x / 8));
    }
  }

  // Perform a direct FFT on each block.
  for (size_t bid = 0; bid < blocks.size(); ++bid) {
    // Convert the block to the correct precision.
    cv::Mat dftInput;
    if constexpr (sizeof(real) == sizeof(double)) {
      blocks[bid].convertTo(dftInput, CV_64F);
    } else {
      blocks[bid].convertTo(dftInput, CV_32F);
    }

    // Perform the FFTs with OpenCV.
    cv::Mat dftOutput;
    cv::dft(dftInput, dftOutput, cv::DFT_COMPLEX_OUTPUT);

    // Split the data into real and immaginary channels.
    std::vector<cv::Mat> channels;
    cv::split(dftOutput, channels);

    cv::Mat realPart = channels[0];
    cv::Mat imagPart = channels[1];

    // Convert OpenCV and GPU results for this block into vecs.
    vec cv_result;
    cv_result.reserve(64);
    vec gpu_result;
    gpu_result.reserve(64);

    for (size_t i = 0; i < 8; i++) {
      for (size_t j = 0; j < 8; j++) {
        cv_result.emplace_back(realPart.at<real>(i, j),
                               imagPart.at<real>(i, j));
        gpu_result.emplace_back(
            fft_output_sequence[(blockIndices[bid].first * 8 + i) * height +
                                blockIndices[bid].second * 8 + j]);
      }
    }

    // Check the result.
    if (!CompareVectors(cv_result, gpu_result, precision, false, false)) {
      std::cerr << "The GPU result is different from the one by OpenCV"
                << std::endl;
      break;
    }
  }

  // Compare the time to CPU times.
  // Reload the image.
  GrayscaleImage grayscaleImage;
  bool success = grayscaleImage.loadStandard(image_path);

  // Check if the image was loaded successfully.
  if (!success) {
    std::cout << "Failed to load image." << std::endl;
    return 1;
  }

  // Get the CPU time (only FFT time included).
  unsigned long cpu_duration = grayscaleImage.calculateEncodingTime();

  // Print the result.
  std::cout << "Execution time for CPU FFT: " << cpu_duration << " μs"
            << std::endl;
  std::cout << "Speedup: " << static_cast<double>(cpu_duration) / fft_duration
            << "x" << std::endl;

  return 0;
}