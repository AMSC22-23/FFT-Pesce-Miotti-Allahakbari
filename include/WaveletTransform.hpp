#ifndef WAVELET_TRANSFORM_HPP
#define WAVELET_TRANSFORM_HPP

#include <memory>

#include "Real.hpp"

namespace Transform {
namespace WaveletTransform {

/** @namespace */

class WaveletTransformAlgorithm {
 public:
  virtual void directTransform(const std::vector<real> &input_sequence,
                               std::vector<real> &output_sequence) const = 0;
  virtual void inverseTransform(const std::vector<real> &input_sequence,
                                std::vector<real> &output_sequence) const = 0;
  virtual ~WaveletTransformAlgorithm() = default;
};

class GPWaveletTransform97 : public WaveletTransformAlgorithm {
 public:
  // Forward biorthogonal 9/7 wavelet transform (lifting implementation).
  // "input_sequence" is an input signal, while "output_sequnce" is the
  // transformed output. Their lengths must be a power of 2 and larger than 1.
  // The first half part of the output signal contains the approximation
  // coefficients. The second half part contains the detail coefficients (aka.
  // the wavelets coefficients). Source:
  // https://web.archive.org/web/20120305164605/http://www.embl.de/~gpau/misc/dwt97.c.
  void directTransform(const std::vector<real> &input_sequence,
                       std::vector<real> &output_sequence) const override;
  // Inverse biorthogonal 9/7 wavelet transform. This is the inverse of
  // DirectWaveletTransform97. "input_sequence" is an input signal, while
  // "output_sequnce" is the transformed output. Their length must be a power of
  // 2 and larger than 1. Source:
  // https://web.archive.org/web/20120305164605/http://www.embl.de/~gpau/misc/dwt97.c.
  void inverseTransform(const std::vector<real> &input_sequence,
                        std::vector<real> &output_sequence) const override;
  ~GPWaveletTransform97() = default;
};

class DaubechiesWaveletTransform97 : public WaveletTransformAlgorithm {
 public:
  // New implementation of the forward 9/7 DWT, following
  // https://services.math.duke.edu/~ingrid/publications/J_Four_Anal_Appl_4_p247.pdf
  // and https://www.sciencedirect.com/science/article/pii/S016516841100199X.
  // Since the articles do not specify what to do with values at array
  // boundaries, the same approach as
  // https://web.archive.org/web/20120305164605/http://www.embl.de/~gpau/misc/dwt97.c
  // was used. The length of "sequence" must be even and larger than 0. The
  // high-passed sequence is stored in "high_sequence", while the low-passed one
  // is stored in "low_sequence". This method gives a different result to the
  // old algorithm.
  void directTransform(const std::vector<real> &input_sequence,
                       std::vector<real> &output_sequence) const override;

  // Inverse of DaubechiesDirectWaveletTransform97. "high_sequence" and
  // "low_sequence" are the outputs from the direct DWT and will be changed
  // during execution. "sequence" contains the result of the inverse
  // transformation.
  void inverseTransform(const std::vector<real> &input_sequence,
                        std::vector<real> &output_sequence) const override;
  ~DaubechiesWaveletTransform97() = default;
};

// Perform a 2D wavelet direct or inverse transform with "levels" levels, using
// the provided algorithm. Each level, the direct transform is performed
// transforming each row and then each column and the inverse transforms each
// column and then each row. The direct transform starts with the entire picture
// and then moves to the top left corner, halfing the side length each
// iteration, for "levels" iterations. The inverse transform does the opposite.
class TwoDimensionalWaveletTransformAlgorithm {
 public:
  void setAlgorithm(std::unique_ptr<WaveletTransformAlgorithm> &algorithm);
  void directTransform(const std::vector<real> &input_matrix,
                       std::vector<real> &output_matrix,
                       unsigned int levels) const;
  void inverseTransform(const std::vector<real> &input_matrix,
                        std::vector<real> &output_matrix,
                        unsigned int levels) const;
  ~TwoDimensionalWaveletTransformAlgorithm() = default;

 private:
  void transformRows(std::vector<real> &matrix, size_t n, bool direct) const;
  void transformColumns(std::vector<real> &matrix, size_t n, bool direct) const;
  std::unique_ptr<WaveletTransformAlgorithm> algorithm;
};

}  // namespace WaveletTransform
}  // namespace Transform

#endif  // WAVELET_TRANSFORM_HPP