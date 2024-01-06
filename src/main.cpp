#include <omp.h>

#include <numbers>
#include <opencv2/opencv.hpp>
#include <string>

#include "FourierTransformCalculator.hpp"
#include "GrayscaleImage.hpp"
#include "Utility.hpp"
#include "WaveletTransform.hpp"

void print_usage_fft(size_t size, const std::string &mode,
                     unsigned int max_num_threads);
void print_usage_wavelet(size_t size);

int jpeg_main(int argc, char *argv[]);
int cuda_main(int argc, char *argv[]);
int wavelet_main(int argc, char *argv[]);
int fft_main(int argc, char *argv[]);

using namespace Transform;

int main(int argc, char *argv[])
{
  if (argc <= 1)
  {
    std::cerr << "Incorrect arguments!\n"
              << "Specify the execution mode!" << std::endl;
    return 1;
  }

  // Get which mode to execute the program in.
  std::string mode = std::string(argv[1]);

  // Run a test with images.
  if (mode == std::string("jpeg"))
  {
    return jpeg_main(argc, argv);
  }
  // Run a test with CUDA.
  else if (mode == std::string("cuda"))
  {
    return cuda_main(argc, argv);
  }
  // Run a test of wavelets.
  else if (mode == std::string("wavelet"))
  {
    return wavelet_main(argc, argv);
  }
  // Run a FFT test.
  else if (mode == std::string("fft"))
  {
    return fft_main(argc, argv);
  }
  // Wrong mode.
  else
  {
    std::cerr
        << "Incorrect arguments!\n"
        << "Specify the execution mode: fft (default) / jpeg / cuda / wavelet"
        << std::endl;
    return 1;
  }

  return 0;
}

int jpeg_main(int argc, char *argv[])
{
  using namespace FourierTransform;

  // We expect at most 1 extra argument.
  if (argc > 3)
  {
    std::cerr << "Incorrect arguments!\n"
              << "Argument 1: jpeg\n"
              << "Argument 2: image path (default: ../img/image.jpg)\n"
              << std::endl;
    return 1;
  }

  // Get the image path.
  std::string image_path = "../img/image.jpg";
  if (argc >= 2)
    image_path = std::string(argv[2]);

  // Create a GrayscaleImage object.
  GrayscaleImage grayscaleImage;

  // Load the image.
  std::cout << "Loading image..." << std::endl;
  bool success = grayscaleImage.loadStandard(image_path);

  // Check if the image was loaded successfully.
  if (!success)
  {
    std::cout << "Failed to load image." << std::endl;
    return 1;
  }

  // Get the bitsize of the image.
  unsigned int bitsize = grayscaleImage.getStandardBitsize();

  // Print the bitsize.
  std::cout << "Bitsize: " << bitsize << std::endl;

  return 0;
}

int cuda_main(int argc, char *argv[])
{
  using namespace FourierTransform;

  if (argc > 3)
  {
    std::cerr << "Incorrect arguments!\n"
              << "Argument 1: cuda\n"
              << "Argument 2: image path (default: ../img/image.jpg)\n"
              << std::endl;
    return 1;
  }

  std::string image_path = "../img/image.jpg";

  // Get which algorithm to choose.
  if (argc >= 3)
    image_path = std::string(argv[2]);

  // Load the image (grayscale)
  cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

  if (image.empty())
  {
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
    for (int j = 0; j < image.cols; j++)
    {
      if (std::fabs(output_sequence[i * image.cols + j].real() -
                    complex[0].at<double>(i, j)) > epsilon ||
          std::fabs(output_sequence[i * image.cols + j].imag() -
                    complex[1].at<double>(i, j)) > epsilon)
      {
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

int wavelet_main(int argc, char *argv[])
{
  using namespace WaveletTransform;

  constexpr size_t default_size = 1UL << 10;
  constexpr real precision = 1e-4;

  if (argc > 3)
  {
    print_usage_wavelet(default_size);
    return 1;
  }

  // Get the size of the sequence.
  size_t size = default_size;
  if (argc >= 2)
  {
    char *error;

    const unsigned long long_size = strtoul(argv[1], &error, 10);
    // Check for errors
    if (*error)
    {
      print_usage_wavelet(default_size);
      return 1;
    }

    size = static_cast<size_t>(long_size);
  }

  // Create a cubic signal. Source of the example:
  // https://web.archive.org/web/20120305164605/http://www.embl.de/~gpau/misc/dwt97.c.
  std::vector<real> input_sequence;
  input_sequence.reserve(size);
  for (size_t i = 0; i < size; i++)
    input_sequence.emplace_back(5 + i + 0.4 * i * i - 0.02 * i * i * i);

  // Save the sequence to a file.
  WriteToFile(input_sequence, "input_sequence.csv");

  // Do the forward 9/7 transform with the old algorithm.
  std::vector<real> fwt_result(input_sequence);
  DirectWaveletTransform97(fwt_result);

  // Save the sequence to a file.
  WriteToFile(fwt_result, "fwt_result.csv");

  // Do the inverse 9/7 transform with the old algorithm.
  std::vector<real> iwt_result(fwt_result);
  InverseWaveletTransform97(iwt_result);

  // Check if the result is the same as the input sequence.
  if (!CompareVectors(input_sequence, iwt_result, precision, false))
    std::cerr << "Errors detected in wavelet transforms." << std::endl;

  // Do the forward 9/7 transform with the new algorithm.
  std::vector<real> high_sequence(size / 2, 0);
  std::vector<real> low_sequence(size / 2, 0);
  NewDirectWaveletTransform97(input_sequence, high_sequence, low_sequence);

  // Save the sequences to a file.
  WriteToFile(high_sequence, "high_sequence.csv");
  WriteToFile(low_sequence, "low_sequence.csv");

  // Do the inverse 9/7 transform with the new algorithm.
  std::vector<real> final_result(size, 0);
  NewInverseWaveletTransform97(final_result, high_sequence, low_sequence);

  // Check if the result is the same as the input sequence.
  if (!CompareVectors(input_sequence, final_result, precision, true))
    std::cerr << "Errors detected in new wavelet transforms." << std::endl;

  return 0;
}

int fft_main(int argc, char *argv[])
{
  using namespace FourierTransform;

  constexpr size_t default_size = 1UL << 10;
  constexpr unsigned int default_max_num_threads = 8;
  const std::string default_mode = "demo";
  constexpr real precision = 1e-4;

  // Check the number of arguments.
  if (argc > 6)
  {
    print_usage_fft(default_size, default_mode, default_max_num_threads);
    return 1;
  }

  // Get the size of the sequence.
  size_t size = default_size;
  if (argc >= 3)
  {
    char *error;

    const unsigned long long_size = strtoul(argv[2], &error, 10);
    // Check for errors
    if (*error ||
        1UL << static_cast<unsigned long>(log2(long_size)) != long_size)
    {
      print_usage_fft(default_size, default_mode, default_max_num_threads);
      return 1;
    }

    size = static_cast<size_t>(long_size);
  }

  // Get the FFT execution mode.
  std::string mode = default_mode;
  if (argc >= 4)
    mode = std::string(argv[3]);

  // Get the maximum number of threads.
  unsigned int max_num_threads = default_max_num_threads;
  if (argc >= 5)
  {
    char *error;

    const unsigned long long_max_num_threads = strtoul(argv[4], &error, 10);
    // Check for errors
    if (*error)
    {
      print_usage_fft(default_size, default_mode, default_max_num_threads);
      return 1;
    }

    max_num_threads = static_cast<size_t>(long_max_num_threads);
  }

  // Generate a sequence of complex numbers.
  vec input_sequence;
  input_sequence.reserve(size);
  for (size_t i = 0; i < size; i++)
  {
    // Add a random complex number to the sequence.
    input_sequence.emplace_back(rand() % 100, rand() % 100);
  }

  // Run a demo of the hands-on code.
  if (mode == std::string("demo"))
  {
    omp_set_num_threads(max_num_threads);

    // Save the sequence to a file.
    WriteToFile(input_sequence, "input_sequence.csv");

    // Create the FourierTransformCalculator object.
    FourierTransformCalculator calculator;

    // Compute the O(n^2) Fourier Transform of the sequence.
    std::unique_ptr<FourierTransformAlgorithm> classical_dft =
        std::make_unique<ClassicalFourierTransformAlgorithm>();
    calculator.setDirectAlgorithm(classical_dft);
    vec classical_dft_result(size, 0);
    calculator.directTransform(input_sequence, classical_dft_result);
    WriteToFile(classical_dft_result, "classical_dft_result.csv");

    // Compute the O(n log n) Fourier Transform of the sequence with the
    // recursive algorithm.
    std::unique_ptr<FourierTransformAlgorithm> recursive_dft =
        std::make_unique<RecursiveFourierTransformAlgorithm>();
    calculator.setDirectAlgorithm(recursive_dft);
    vec recursive_dft_result(size, 0);
    calculator.directTransform(input_sequence, recursive_dft_result);
    WriteToFile(recursive_dft_result, "recursive_dft_result.csv");

    // Compute the O(n log n) Fourier Transform of the sequence with the
    // iterative algorithm.
    std::unique_ptr<IterativeFourierTransformAlgorithm>
        iterative_dft_algorithm =
            std::make_unique<IterativeFourierTransformAlgorithm>();
    std::unique_ptr<BitReversalPermutationAlgorithm>
        mask_bit_reversal_algorithm =
            std::make_unique<MaskBitReversalPermutationAlgorithm>();
    iterative_dft_algorithm->setBitReversalPermutationAlgorithm(
        mask_bit_reversal_algorithm);
    std::unique_ptr<FourierTransformAlgorithm> iterative_dft =
        std::move(iterative_dft_algorithm);
    calculator.setDirectAlgorithm(iterative_dft);
    vec iterative_dft_result(size, 0);
    calculator.directTransform(input_sequence, iterative_dft_result);
    WriteToFile(iterative_dft_result, "iterative_dft_result.csv");

    std::unique_ptr<FourierTransformAlgorithm> iterative_fft_gpu =
        std::make_unique<IterativeFFTGPU>();
    calculator.setDirectAlgorithm(iterative_fft_gpu);
    vec iterative_gpu_result(size, 0);
    calculator.directTransform(input_sequence, iterative_gpu_result);
    WriteToFile(iterative_dft_result, "iterative_gpu_result.csv");

    // Check the results for errors.
    if (!CompareVectors(classical_dft_result, recursive_dft_result, precision,
                        false))
      std::cerr << "Errors detected in recursive direct FFT." << std::endl;
    if (!CompareVectors(classical_dft_result, iterative_dft_result, precision,
                        false))
      std::cerr << "Errors detected in iterative direct FFT." << std::endl;
    if (!CompareVectors(classical_dft_result, iterative_gpu_result, precision,
                        false))
      std::cerr << "Errors detected in iterative GPU FFT." << std::endl;

    // Compute the O(n^2) Fourier Transform of the result.
    std::unique_ptr<FourierTransformAlgorithm> classical_ift =
        std::make_unique<ClassicalFourierTransformAlgorithm>();
    calculator.setInverseAlgorithm(classical_ift);
    vec classical_ift_result(size, 0);
    calculator.inverseTransform(classical_dft_result, classical_ift_result);
    WriteToFile(classical_ift_result, "classical_ift_result.csv");

    // Compute the iterative O(n log) Inverse Fourier Transform of the result.
    std::unique_ptr<IterativeFourierTransformAlgorithm>
        iterative_ift_algorithm =
            std::make_unique<IterativeFourierTransformAlgorithm>();
    std::unique_ptr<BitReversalPermutationAlgorithm>
        fast_bit_reversal_algorithm =
            std::make_unique<FastBitReversalPermutationAlgorithm>();
    iterative_ift_algorithm->setBitReversalPermutationAlgorithm(
        fast_bit_reversal_algorithm);
    std::unique_ptr<FourierTransformAlgorithm> iterative_ift =
        std::move(iterative_ift_algorithm);
    calculator.setInverseAlgorithm(iterative_ift);
    vec iterative_ift_result(size, 0);
    calculator.inverseTransform(classical_dft_result, iterative_ift_result);
    WriteToFile(iterative_ift_result, "iterative_ift_result.csv");

    // Check if the new inverse sequences are equal to the original one.
    if (!CompareVectors(input_sequence, classical_ift_result, precision, false))
      std::cerr << "Errors detected in classical inverse FFT." << std::endl;
    if (!CompareVectors(input_sequence, iterative_ift_result, precision, false))
      std::cerr << "Errors detected in iterative inverse FFT." << std::endl;
  }

  // Run a performance comparison of different bit reversal permutation
  // techniques.
  else if (mode == std::string("bitReversalTest"))
  {
    // Run a comparison between mask and fast bit reversal permutations.
    CompareTimesBitReversalPermutation(input_sequence, max_num_threads);
  }

  // Run a scaling test with the best performing algorithm.
  else if (mode == std::string("scalingTest"))
  {
    // Calculate the times for up to max_num_threads threads for the iterative
    // fft.
    std::unique_ptr<IterativeFourierTransformAlgorithm>
        iterative_dft_algorithm =
            std::make_unique<IterativeFourierTransformAlgorithm>();
    iterative_dft_algorithm->setBaseAngle(-std::numbers::pi_v<real>);
    std::unique_ptr<BitReversalPermutationAlgorithm> bit_reversal_algorithm =
        std::make_unique<MaskBitReversalPermutationAlgorithm>();
    iterative_dft_algorithm->setBitReversalPermutationAlgorithm(
        bit_reversal_algorithm);
    std::unique_ptr<FourierTransformAlgorithm> iterative_dft =
        std::move(iterative_dft_algorithm);
    TimeEstimateFFT(iterative_dft, input_sequence, max_num_threads);
  }

  // Execute a single algorithm and calculate the elapsed time.
  else if (mode == std::string("timingTest"))
  {
    std::string algorithm_name = "iterative";

    // Get which algorithm to choose.
    if (argc >= 6)
      algorithm_name = std::string(argv[5]);

    // Create the algorithm.
    std::unique_ptr<FourierTransformAlgorithm> algorithm;
    if (algorithm_name == std::string("classic"))
    {
      algorithm = std::make_unique<ClassicalFourierTransformAlgorithm>();
    }
    else if (algorithm_name == std::string("recursive"))
    {
      algorithm = std::make_unique<RecursiveFourierTransformAlgorithm>();
    }
    else if (algorithm_name == std::string("iterative"))
    {
      std::unique_ptr<IterativeFourierTransformAlgorithm>
          iterative_dft_algorithm =
              std::make_unique<IterativeFourierTransformAlgorithm>();
      std::unique_ptr<BitReversalPermutationAlgorithm> bit_reversal_algorithm =
          std::make_unique<MaskBitReversalPermutationAlgorithm>();
      iterative_dft_algorithm->setBitReversalPermutationAlgorithm(
          bit_reversal_algorithm);
      algorithm = std::move(iterative_dft_algorithm);
    }
    else if (algorithm_name == std::string("iterativeGPU"))
    {
      algorithm = std::make_unique<IterativeFFTGPU>();
    }
    else
    {
      print_usage_fft(default_size, default_mode, default_max_num_threads);
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

  return 0;
}

void print_usage_wavelet(size_t size)
{
  std::cerr << "Incorrect arguments!\n"
            << "Argument 1: wavelet\n"
            << "Argument 2: size of the sequence (default: " << size << ")"
            << std::endl;
}

void print_usage_fft(size_t size, const std::string &mode,
                     unsigned int max_num_threads)
{
  std::cerr << "Incorrect arguments!\n"
            << "Argument 1: fft\n"
            << "Argument 2: size of the sequence (default: " << size
            << "), must be a power of 2\n"
            << "Argument 3: execution mode (demo / bitReversalTest / "
               "scalingTest / timingTest (default: "
            << mode << "))\n"
            << "Argument 4: maximum number of threads (default: "
            << max_num_threads << ")\n"
            << "Argument 5: algorithm for timingTest mode (classic, recursive, "
               "iterative(default)"
            << std::endl;
}