#ifndef FOURIER_TRANSFORM_CALCULATOR_HPP
#define FOURIER_TRANSFORM_CALCULATOR_HPP

#include "FourierTransform.hpp"

namespace Transform {
namespace FourierTransform {

// A class that represents a Fourier transform calculator, which can set an
// algorithm for both the direct and the inverse transform computation. In the
// future, more methods will be added for intermedia computations between the
// two transforms.
class FourierTransformCalculator {
 public:
  void setDirectAlgorithm(
      std::unique_ptr<FourierTransformAlgorithm> &algorithm);
  void setInverseAlgorithm(
      std::unique_ptr<FourierTransformAlgorithm> &algorithm);
  // Perform the direct Fourier transform of "input_sequence" and store it in
  // "output_sequence".
  void directTransform(const vec &input_sequence, vec &output_sequence);
  // Perform the inverse Fourier transform of "input_sequence" and store it in
  // "output_sequence".
  void inverseTransform(const vec &input_sequence, vec &output_sequence);

 private:
  std::unique_ptr<FourierTransformAlgorithm> direct_algorithm;
  std::unique_ptr<FourierTransformAlgorithm> inverse_algorithm;

  // Multiply all elements of "sequence" by "scalar".
  void scaleVector(vec &sequence, real scalar);
};

}  // namespace FourierTransform
}  // namespace Transform

#endif  // FOURIER_TRANSFORM_CALCULATOR_HPP