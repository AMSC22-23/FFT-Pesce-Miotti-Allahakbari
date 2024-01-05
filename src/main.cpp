#include "GrayscaleImage.hpp"

#include <iostream>
#include <numbers>
#include <opencv2/opencv.hpp>
#include <string>

#include "FourierTransformCalculator.hpp"
#include "Utility.hpp"
#include "VectorExporter.hpp"

using namespace FourierTransform;

void print_usage(size_t size, const std::string& mode,
                 unsigned int max_num_threads);

int main(int argc, char* argv[]) {
  // Load the image (grayscale)
  cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

  if (image.empty()) {
    std::cerr << "Error: Unable to load the image." << std::endl;
    return EXIT_FAILURE;
  }

  // Check image properties
  std::cout << "Image size: " << image.size() << std::endl;
  std::cout << "Image type: " << image.type() << std::endl;

  // convert image to complex vector
  vec input_sequence;
  vec output_sequence;

  input_sequence.reserve(image.rows * image.cols);
  output_sequence.reserve(image.rows * image.cols);

  for (int i = 0; i < image.rows; i++)
    for (int j = 0; j < image.cols; j++)
      input_sequence.emplace_back(image.at<uchar>(i, j), 0);

  IterativeFFTGPU2D* iterativeGPU2D_algorithm = new IterativeFFTGPU2D();
  std::unique_ptr<FourierTransformAlgorithm> algorithm =
      std::unique_ptr<FourierTransformAlgorithm>(iterativeGPU2D_algorithm);

  // Set the direct transform.
  algorithm->setBaseAngle(-std::numbers::pi_v<real>);

  // Execute the algorithm and calculate the time.
  std::cout << algorithm->calculateTime(input_sequence, output_sequence) << "μs"
            << std::endl;

  image.convertTo(image, CV_64FC1);
  cv::Mat planes[] = {image, cv::Mat::zeros(image.size(), CV_64FC1)};
  cv::Mat complexInput;
  cv::merge(planes, 2, complexInput);
  cv::dft(planes[0], planes[1], cv::DFT_COMPLEX_OUTPUT);
  cv::Mat complex[2];

  cv::split(planes[1], complex);

  double epsilon = 1e-4;

  for (int i = 0; i < image.rows; i++)
    for (int j = 0; j < image.cols; j++) {
      if (std::fabs(output_sequence[i * image.cols + j].real() -
                    complex[0].at<double>(i, j)) > epsilon ||
          std::fabs(output_sequence[i * image.cols + j].imag() -
                    complex[1].at<double>(i, j)) > epsilon) {
        std::cout << "Error in GPU calculations at: " << i << ", " << j
                  << " GPU: " << output_sequence[i * image.cols + j]
                  << ", CPU: (" << complex[0].at<double>(i, j) << ", "
                  << complex[1].at<double>(i, j) << ")" << std::endl;
        // break;
      }
      // std::cout << "Calculated by gpu: " << output_sequence[i * image.cols +
      // j]
      //           << ", Calculated by opencv: (" << complex[0].at<float>(i, j)
      //           << ", " << complex[1].at<float>(i, j) << ")" << std::endl;
    }
  // // Display the image using OpenCV (optional)
  // cv::imshow("Loaded Image", image);
  // cv::waitKey(0);
  // cv::destroyAllWindows();

  return EXIT_SUCCESS;
}

int main_random_sequence_mode(int argc, char* argv[]) {
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
  
    // Compute the O(n log n) Fourier Transform of the sequence with the
    // recursive algorithm.
    std::unique_ptr<FourierTransformAlgorithm> recursive_dft(
        new RecursiveFourierTransformAlgorithm());
    calculator.setDirectAlgorithm(recursive_dft);
    vec recursive_dft_result(size, 0);
    calculator.directTransform(input_sequence, recursive_dft_result);
    WriteToFile(recursive_dft_result, "recursive_dft_result.csv");

    // Compute the O(n log n) Fourier Transform of the sequence with the
    // iterative algorithm.
    IterativeFourierTransformAlgorithm* iterative_dft_algorithm =
        new IterativeFourierTransformAlgorithm();
    std::unique_ptr<BitReversalPermutationAlgorithm> bit_reversal_algorithm(
        new MaskBitReversalPermutationAlgorithm());
    iterative_dft_algorithm->setBitReversalPermutationAlgorithm(
        bit_reversal_algorithm);
    std::unique_ptr<FourierTransformAlgorithm> iterative_dft(
        iterative_dft_algorithm);
    calculator.setDirectAlgorithm(iterative_dft);
    vec iterative_dft_result(size, 0);
    calculator.directTransform(input_sequence, iterative_dft_result);
    WriteToFile(iterative_dft_result, "iterative_dft_result.csv");

    IterativeFFTGPU* fft_gpu_algorithm = new IterativeFFTGPU();
    std::unique_ptr<FourierTransformAlgorithm> iterative_fft_gpu(
        fft_gpu_algorithm);
    calculator.setDirectAlgorithm(iterative_fft_gpu);
    vec iterative_gpu_result(size, 0);
    calculator.directTransform(input_sequence, iterative_gpu_result);
    WriteToFile(iterative_dft_result, "iterative_gpu_result.csv");

    // Check the results for errors.
    if (!CompareVectors(classical_dft_result, recursive_dft_result, 1e-4,
                        false))
      std::cerr << "Errors detected in recursive direct FFT." << std::endl;
    if (!CompareVectors(classical_dft_result, iterative_dft_result, 1e-4,
                        false))
      std::cerr << "Errors detected in iterative direct FFT." << std::endl;
    if (!CompareVectors(classical_dft_result, iterative_gpu_result, 1e-4,
                        false))
      std::cerr << "Errors detected in iterative GPU FFT." << std::endl;

    // Compute the O(n^2) Fourier Transform of the result.
    std::unique_ptr<FourierTransformAlgorithm> classical_ift(
        new ClassicalFourierTransformAlgorithm());
    calculator.setInverseAlgorithm(classical_ift);
    vec classical_ift_result(size, 0);
    calculator.inverseTransform(classical_dft_result, classical_ift_result);
    WriteToFile(classical_ift_result, "classical_ift_result.csv");

    // Compute the iterative O(n log) Inverse Fourier Transform of the result.
    IterativeFourierTransformAlgorithm* iterative_ift_algorithm =
        new IterativeFourierTransformAlgorithm();
    std::unique_ptr<BitReversalPermutationAlgorithm> bit_reversal_algorithm2(
        new FastBitReversalPermutationAlgorithm());
    iterative_ift_algorithm->setBitReversalPermutationAlgorithm(
        bit_reversal_algorithm2);
    std::unique_ptr<FourierTransformAlgorithm> iterative_ift(
        iterative_ift_algorithm);
    calculator.setInverseAlgorithm(iterative_ift);
    vec iterative_ift_result(size, 0);
    calculator.inverseTransform(classical_dft_result, iterative_ift_result);
    WriteToFile(iterative_ift_result, "iterative_ift_result.csv");

    // Check if the new inverse sequences are equal to the original one.
    if (!CompareVectors(input_sequence, classical_ift_result, 1e-4, false))
      std::cerr << "Errors detected in classical inverse FFT." << std::endl;
    if (!CompareVectors(input_sequence, iterative_ift_result, 1e-4, false))
      std::cerr << "Errors detected in iterative inverse FFT." << std::endl;
  }

  // Run a performance comparison of different bit reversal permutation
  // techniques.
  else if (mode == std::string("bitReversalTest")) {
    // Run a comparison between mask and fast bit reversal permutations.
    CompareTimesBitReversalPermutation(input_sequence, max_num_threads);
  }

  // Run a scaling test with the best performing algorithm.
  else if (mode == std::string("scalingTest")) {
    // Calculate the times for up to max_num_threads threads for the iterative
    // fft.
    IterativeFourierTransformAlgorithm* iterative_dft_algorithm =
        new IterativeFourierTransformAlgorithm();
    iterative_dft_algorithm->setBaseAngle(-std::numbers::pi_v<real>);
    std::unique_ptr<BitReversalPermutationAlgorithm> bit_reversal_algorithm(
        new MaskBitReversalPermutationAlgorithm());
    iterative_dft_algorithm->setBitReversalPermutationAlgorithm(
        bit_reversal_algorithm);
    std::unique_ptr<FourierTransformAlgorithm> iterative_dft(
        iterative_dft_algorithm);
    TimeEstimateFFT(iterative_dft, input_sequence, max_num_threads);
  }

  // Execute a single algorithm and calculate the elapsed time.
  else if (mode == std::string("timingTest")) {
    std::string algorithm_name = "iterativeGPU";

  // Print the bitsize.
  std::cout << "The loaded image has a bitsize of " << bitsize << " bits." << std::endl;

  // Display the image.
  std::cout << "Displaying image..." << std::endl;
  grayscaleImage.display();

    // Create the algorithm.
    std::unique_ptr<FourierTransformAlgorithm> algorithm;
    if (algorithm_name == std::string("classic")) {
      algorithm = std::unique_ptr<FourierTransformAlgorithm>(
          new ClassicalFourierTransformAlgorithm());
    } else if (algorithm_name == std::string("recursive")) {
      algorithm = std::unique_ptr<FourierTransformAlgorithm>(
          new RecursiveFourierTransformAlgorithm());
    } else if (algorithm_name == std::string("iterative")) {
      IterativeFourierTransformAlgorithm* iterative_algorithm =
          new IterativeFourierTransformAlgorithm();
      std::unique_ptr<BitReversalPermutationAlgorithm> bit_reversal_algorithm(
          new MaskBitReversalPermutationAlgorithm());
      iterative_algorithm->setBitReversalPermutationAlgorithm(
          bit_reversal_algorithm);
      algorithm =
          std::unique_ptr<FourierTransformAlgorithm>(iterative_algorithm);
    } else if (algorithm_name == std::string("iterativeGPU")) {
      IterativeFFTGPU* iterativeGPU_algorithm = new IterativeFFTGPU();
      algorithm =
          std::unique_ptr<FourierTransformAlgorithm>(iterativeGPU_algorithm);
    } else {
      print_usage(default_size, default_mode, default_max_num_threads);
      return 1;
    }

    // Set the direct transform.
    algorithm->setBaseAngle(-std::numbers::pi_v<real>);

    // Create an output vector.
    vec result(size, 0);

    // Set the number of threads.
    omp_set_num_threads(max_num_threads);

    // Execute the algorithm and calculate the time.
    std::cout << algorithm->calculateTime(input_sequence, result) << "μs"
              << std::endl;
  }

  // Wrong mode specified.
  else {
    print_usage(default_size, default_mode, default_max_num_threads);
    return 1;
  }

  return 0;
}

void print_usage(size_t size, const std::string& mode,
                 unsigned int max_num_threads) {
  std::cerr << "Incorrect arguments!\n"
            << "Argument 1: size of the sequence (default: " << size
            << "), must be a power of 2\n"
            << "Argument 2: execution mode (demo / bitReversalTest / "
               "scalingTest, timingTest) (default: "
            << mode << ")\n"
            << "Argument 3: maximum number of threads (default: "
            << max_num_threads << ")\n"
            << "Argument 4: algorithm for timingTest mode (classic, recursive, "
               "iterative, IterativeGPU(default))"
            << std::endl;
}