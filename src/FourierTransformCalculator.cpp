#include "FourierTransformCalculator.hpp"

#include <numbers>

#include "Utility.hpp"

namespace FourierTransform {

constexpr real pi = std::numbers::pi_v<real>;

void FourierTransformCalculator::setDirectAlgorithm(
    std::unique_ptr<FourierTransformAlgorithm> &algorithm) {
  direct_algorithm = std::move(algorithm);
  direct_algorithm->setBaseAngle(-pi);
}

void FourierTransformCalculator::setInverseAlgorithm(
    std::unique_ptr<FourierTransformAlgorithm> &algorithm) {
  inverse_algorithm = std::move(algorithm);
  inverse_algorithm->setBaseAngle(pi);
}

// Perform the direct Fourier transform of "input_sequence" and store it in
// "output_sequence".
void FourierTransformCalculator::directTransform(const vec &input_sequence,
                                                 vec &output_sequence) {
  (*direct_algorithm)(input_sequence, output_sequence);
}

// Perform the direct Fourier transform of "input_sequence" and store it in
// "output_sequence". This consists in two steps: calling the algorithm (after
// setting the correct angle) and scaling all coefficients by dividing them by
// the size of the sequence.
void FourierTransformCalculator::inverseTransform(const vec &input_sequence,
                                                  vec &output_sequence) {
  (*inverse_algorithm)(input_sequence, output_sequence);
  ScaleVector(output_sequence, real(1.0) / output_sequence.size());
}

}  // namespace FourierTransform