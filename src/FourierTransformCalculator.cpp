#include "FourierTransformCalculator.hpp"

#include <numbers>

namespace Transform {
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
  FourierTransformCalculator::scaleVector(output_sequence,
                                          real(1.0) / output_sequence.size());
}

void FourierTransformCalculator::scaleVector(vec &sequence, real scalar) {
  // Get the size of the sequence
  const size_t n = sequence.size();

#pragma omp parallel for default(none) shared(sequence) firstprivate(n, scalar)
  for (size_t i = 0; i < n; i++) {
    sequence[i] *= scalar;
  }
}

}  // namespace FourierTransform
}  // namespace Transform