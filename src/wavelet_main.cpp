#include "wavelet_main.hpp"

#include <string>

#include "GrayscaleImage.hpp"
#include "Utility.hpp"
#include "WaveletTransform.hpp"

void print_usage(size_t size, const std::string &mode,
                 const std::string &path) {
  std::cerr << "Incorrect arguments!\n"
            << "Argument 1: wavelet\n"
            << "Argument 2: mode (demo / image (default: " << mode << "))\n"
            << "Argument 3: demo: size of the sequence (default: " << size
            << "), image: image path (default: " << path << ")" << std::endl;
}

int wavelet_main(int argc, char *argv[]) {
  using namespace Transform;
  using namespace WaveletTransform;

  constexpr size_t default_size = 1UL << 10;
  const std::string default_image = "../img/image.jpg";
  const std::string default_mode = "demo";
  constexpr real precision = 1e-4;

  if (argc > 4) {
    print_usage(default_size, default_mode, default_image);
    return 1;
  }

  // Get the wavelet execution mode.
  std::string mode = default_mode;
  if (argc >= 3) mode = std::string(argv[2]);

  // Perform a test on an image.
  if (mode == std::string("image")) {
    // Get the image path.
    std::string path = default_image;
    if (argc >= 4) path = std::string(argv[3]);

    GrayscaleImage image;
    bool success = image.loadStandard(path);

    // Check if the image was loaded successfully.
    if (!success) {
      std::cout << "Failed to load image." << std::endl;
      return 1;
    }

    // Perform a demo of the wavelet transforms.
  } else if (mode == std::string("demo")) {
    // Get the size of the sequence.
    size_t size = default_size;
    if (argc >= 4) {
      char *error;

      const unsigned long long_size = strtoul(argv[3], &error, 10);
      // Check for errors
      if (*error) {
        print_usage(default_size, default_mode, default_image);
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
    GPDirectWaveletTransform97 gp_direct;
    gp_direct(input_sequence, fwt_result);

    // Save the sequence to a file.
    WriteToFile(fwt_result, "gp_fwt_result.csv");

    // Do the inverse 9/7 transform with the algorithm by Gregoire Pau.
    std::vector<real> iwt_result(fwt_result);
    GPInverseWaveletTransform97 gp_inverse;
    gp_inverse(fwt_result, iwt_result);

    // Check if the result is the same as the input sequence.
    if (!CompareVectors(input_sequence, iwt_result, precision, false))
      std::cerr << "Errors detected in wavelet transforms." << std::endl;

    // Do the forward 9/7 transform with the algorithm by Daubechies.
    DaubechiesDirectWaveletTransform97 db_direct;
    db_direct(input_sequence, fwt_result);

    // Save the sequences to a file.
    WriteToFile(fwt_result, "daubechies_fwt_result.csv");

    // Do the inverse 9/7 transform with the algorithm by Daubechies.
    DaubechiesInverseWaveletTransform97 db_inverse;
    db_inverse(fwt_result, iwt_result);

    // Check if the result is the same as the input sequence.
    if (!CompareVectors(input_sequence, iwt_result, precision, true))
      std::cerr << "Errors detected in new wavelet transforms." << std::endl;
  } else {
    print_usage(default_size, default_mode, default_image);
    return 1;
  }

  return 0;
}
