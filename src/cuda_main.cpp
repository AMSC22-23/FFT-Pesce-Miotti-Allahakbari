#include "cuda_main.hpp"

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

  std::unique_ptr<FourierTransformAlgorithm> algorithm =
      std::make_unique<IterativeFFTGPU2D>();

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
      // std::cout << "Calculated by gpu: " << output_sequence[i * image.cols
      // + j]
      //           << ", Calculated by opencv: (" << complex[0].at<float>(i,
      //           j)
      //           << ", " << complex[1].at<float>(i, j) << ")" << std::endl;
    }
  // // Display the image using OpenCV (optional)
  // cv::imshow("Loaded Image", image);
  // cv::waitKey(0);
  // cv::destroyAllWindows();

  return EXIT_SUCCESS;
}