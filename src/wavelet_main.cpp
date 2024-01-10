#include "wavelet_main.hpp"

/**
 * @file wavelet_main.cpp.
 * @brief Defines the main function for wavelet-related tests.
 */

#include <numbers>
#include <string>

#include "GrayscaleImage.hpp"
#include "Utility.hpp"
#include "WaveletTransform.hpp"

void print_usage(size_t size, const std::string &mode, const std::string &path,
                 unsigned int levels, Transform::real threshold) {
  std::cerr << "Incorrect arguments!\n"
            << "Argument 1: wavelet\n"
            << "Argument 2: mode (demo / image / denoise (default: " << mode
            << "))\n"
            << "Argument 3: demo: size of the sequence (default: " << size
            << "), image/denoise: image path (default: " << path << ")\n"
            << "Argument 4: image/denoise: number of 2D DWT levels (default: "
            << levels << ")\n"
            << "Argument 5: denoise: denoise threshold (default: " << threshold
            << ")" << std::endl;
}

int wavelet_main(int argc, char *argv[]) {
  using namespace Transform;
  using namespace WaveletTransform;

  constexpr size_t default_size = 1UL << 10;
  const std::string default_image = "../img/image.jpg";
  const std::string default_mode = "demo";
  constexpr real precision =
      std::max(std::numeric_limits<real>::epsilon() * 1e5, 1e-4);
  constexpr unsigned int default_levels = 1;
  constexpr real default_threshold = 50;

  if (argc > 6) {
    print_usage(default_size, default_mode, default_image, default_levels,
                default_threshold);
    return 1;
  }

  // Get the wavelet execution mode.
  std::string mode = default_mode;
  if (argc >= 3) mode = std::string(argv[2]);

  // Perform a test on an image, either displaying wavelet transforms or
  // denoising.
  if (mode == std::string("image") || mode == std::string("denoise")) {
    // Get the image path.
    std::string path = default_image;
    if (argc >= 4) path = std::string(argv[3]);

    // Get the number of levels.
    unsigned int levels = default_levels;
    if (argc >= 5) {
      char *error;

      unsigned long long_levels = strtoul(argv[4], &error, 10);
      // Check for errors.
      if (*error) {
        print_usage(default_size, default_mode, default_image, default_levels,
                    default_threshold);
        return 1;
      }

      levels = static_cast<unsigned int>(long_levels);
    }

    // Perform a test displaying wavelet transforms.
    if (mode == std::string("image")) {
      if (argc > 5) {
        print_usage(default_size, default_mode, default_image, default_levels,
                    default_threshold);
        return 1;
      }

      // Load the image.
      GrayscaleImage image;
      bool success = image.loadStandard(path);

      // Check if the image was loaded successfully.
      if (!success) {
        std::cout << "Failed to load image." << std::endl;
        return 1;
      }

      // Display the image.
      image.display();

      // Create the two wavelet transform algorithms.
      std::unique_ptr<WaveletTransformAlgorithm> gp_algorithm =
          std::make_unique<GPWaveletTransform97>();
      std::unique_ptr<WaveletTransformAlgorithm> db_algorithm =
          std::make_unique<DaubechiesWaveletTransform97>();

      // Perform the transform with the algorithm by Gregoire Pau.
      image.waveletTransform(gp_algorithm, levels);
      image.display();

      // Save the image.
      success = image.saveStandard("gp_transform.jpg");
      // Check if the image was saved successfully.
      if (!success) {
        std::cout << "Failed to save image." << std::endl;
        return 1;
      }

      // Load a second copy of the image.
      GrayscaleImage image2;
      success = image2.loadStandard(path);

      // Check if the image was loaded successfully.
      if (!success) {
        std::cout << "Failed to load image." << std::endl;
        return 1;
      }

      // Perform the transform with the algorithm by Daubechies.
      image2.waveletTransform(db_algorithm, levels);
      image2.display();

      // Save the image.
      success = image.saveStandard("db_transform.jpg");
      // Check if the image was saved successfully.
      if (!success) {
        std::cout << "Failed to save image." << std::endl;
        return 1;
      }

      // Perform a denoising test.
    } else {
      // Get the threshold.
      real threshold = default_threshold;
      if (argc >= 6) {
        char *error;

        threshold = strtod(argv[5], &error);
        // Check for errors.
        if (*error) {
          print_usage(default_size, default_mode, default_image, default_levels,
                      default_threshold);
          return 1;
        }
      }

      // Load the image.
      GrayscaleImage image;
      bool success = image.loadStandard(path);

      // Check if the image was loaded successfully.
      if (!success) {
        std::cout << "Failed to load image." << std::endl;
        return 1;
      }

      // Create a wavelet algorithm.
      std::unique_ptr<WaveletTransformAlgorithm> gp_algorithm =
          std::make_unique<GPWaveletTransform97>();

      // Denoise the image with the algorithm by Gregoire Pau.
      image.denoise(gp_algorithm, levels, threshold, false);
      image.display();
    }

    // Perform a demo of the wavelet transforms.
  } else if (mode == std::string("demo")) {
    // Check the number of arguments.
    if (argc > 4) {
      print_usage(default_size, default_mode, default_image, default_levels,
                  default_threshold);
      return 1;
    }

    // Get the size of the sequence.
    size_t size = default_size;
    if (argc >= 4) {
      char *error;

      const unsigned long long_size = strtoul(argv[3], &error, 10);
      // Check for errors.
      if (*error) {
        print_usage(default_size, default_mode, default_image, default_levels,
                    default_threshold);
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

    // Do the forward 9/7 transform with the algorithm by Gregoire Pau.
    std::vector<real> fwt_result(input_sequence);
    const GPWaveletTransform97 gp_algorithm;
    gp_algorithm.directTransform(input_sequence, fwt_result);

    // Save the sequence to a file.
    WriteToFile(fwt_result, "gp_fwt_result.csv");

    // Do the inverse 9/7 transform with the algorithm by Gregoire Pau.
    std::vector<real> iwt_result(fwt_result);
    gp_algorithm.inverseTransform(fwt_result, iwt_result);

    // Check if the result is the same as the input sequence.
    if (!CompareVectors(input_sequence, iwt_result, precision, false))
      std::cerr << "Errors detected in GP wavelet transforms." << std::endl;

    // Do the forward 9/7 transform with the algorithm by Daubechies.
    const DaubechiesWaveletTransform97 db_algorithm;
    db_algorithm.directTransform(input_sequence, fwt_result);

    // Save the sequences to a file.
    WriteToFile(fwt_result, "daubechies_fwt_result.csv");

    // Do the inverse 9/7 transform with the algorithm by Daubechies.
    db_algorithm.inverseTransform(fwt_result, iwt_result);

    // Check if the result is the same as the input sequence.
    if (!CompareVectors(input_sequence, iwt_result, precision, false))
      std::cerr << "Errors detected in DB wavelet transforms." << std::endl;

    // Test the 2D algorithms.
    const unsigned int max_levels =
        std::min(8U, static_cast<unsigned int>(log2(size)));
    for (unsigned int levels = 1; levels < max_levels; levels++) {
      // Create a 2D example.
      std::vector<real> input_matrix;
      input_matrix.reserve(size * size);
      for (size_t i = 0; i < size * size; i++)
        input_matrix.emplace_back(rand() % 100);
      std::vector<real> dwt_matrix(size * size, 0);
      std::vector<real> iwt_matrix(size * size, 0);

      // Test the algorithm by Gregoire Pau.
      std::unique_ptr<WaveletTransformAlgorithm> gp_algorithm =
          std::make_unique<GPWaveletTransform97>();
      const TwoDimensionalWaveletTransformAlgorithm gp_algorithm_2d(
          gp_algorithm);
      gp_algorithm_2d.directTransform(input_matrix, dwt_matrix, levels, true);
      gp_algorithm_2d.inverseTransform(dwt_matrix, iwt_matrix, levels, true);
      if (!CompareVectors(input_matrix, iwt_matrix, precision, false))
        std::cerr << "Errors detected in 2D GP wavelet transforms with "
                  << levels << " levels." << std::endl;

      // Test the algorithm by Daubechies.
      std::unique_ptr<WaveletTransformAlgorithm> db_algorithm =
          std::make_unique<DaubechiesWaveletTransform97>();
      const TwoDimensionalWaveletTransformAlgorithm db_algorithm_2d(
          db_algorithm);
      db_algorithm_2d.directTransform(input_matrix, dwt_matrix, levels, true);
      db_algorithm_2d.inverseTransform(dwt_matrix, iwt_matrix, levels, true);
      if (!CompareVectors(input_matrix, iwt_matrix, precision, false))
        std::cerr << "Errors detected in 2D DB wavelet transforms with "
                  << levels << " levels." << std::endl;
    }

  } else {
    print_usage(default_size, default_mode, default_image, default_levels,
                default_threshold);
    return 1;
  }

  return 0;
}
