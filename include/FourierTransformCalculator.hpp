#ifndef FOURIER_TRANSFORM_CALCULATOR_HPP
#define FOURIER_TRANSFORM_CALCULATOR_HPP

/**
 * @file FourierTransformCalculator.hpp.
 * @brief Declares a utility for Fourier Transform computations.
 */

#include "FourierTransform.hpp"

namespace Transform {
namespace FourierTransform {

/**
 * @brief Gives high level users easier usage of FourierTransformAlgorithm.
 *
 * The class allows to execute a direct and inverse Fourier Transform without
 * having to set the angle and scaling the elements of the sequence obtained
 * with the inverse transform. Since all algorithms provide the same result,
 * different algorithms can be used for the direct and inverse transform.
 */
class FourierTransformCalculator {
 public:
  /**
   * @brief Set the algorithm for the direct Fourier Transform.
   *
   * @param algorithm The algorithm for the direct Fourier Transform.
   *
   * @note The algorithm will be moved.
   */
  void setDirectAlgorithm(
      std::unique_ptr<FourierTransformAlgorithm> &algorithm);
  /**
   * @brief Set the algorithm for the inverse Fourier Transform.
   *
   * @param algorithm The algorithm for the inverse Fourier Transform.
   *
   * @note The algorithm will be moved.
   */
  void setInverseAlgorithm(
      std::unique_ptr<FourierTransformAlgorithm> &algorithm);
  /**
   * @brief Perform a direct Fourier Transform on the sequence.
   *
   * @param input_sequence The input sequence.
   * @param output_sequence The output sequence, containing the transformed
   * input.
   *
   * @note The sequences must have the same length n. Depending on the provided
   * algorithm, n might have to be a power of 2.
   */
  void directTransform(const vec &input_sequence, vec &output_sequence);
  /**
   * @brief Perform an inverse Fourier Transform on the sequence.
   *
   * @param input_sequence The input sequence.
   * @param output_sequence The output sequence, containing the anti-transformed
   * input.
   *
   * @note The sequences must have the same length n. Depending on the provided
   * algorithm, n might have to be a power of 2.
   */
  void inverseTransform(const vec &input_sequence, vec &output_sequence);

 private:
  /** @brief The algorithm for the direct transform. */
  std::unique_ptr<FourierTransformAlgorithm> direct_algorithm;

  /** @brief The algorithm for the direct transform. */
  std::unique_ptr<FourierTransformAlgorithm> inverse_algorithm;

  /**
   * @brief Scale a sequence by a scalar.
   *
   * Multiply all elements in the provided sequence by the provided scalar.
   *
   * @param input_sequence The sequence.
   * @param scalar The scaling factor.
   */
  void scaleVector(vec &sequence, real scalar);
};

}  // namespace FourierTransform
}  // namespace Transform

#endif  // FOURIER_TRANSFORM_CALCULATOR_HPP