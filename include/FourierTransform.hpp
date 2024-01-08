#ifndef FOURIER_TRANSFORM_HPP
#define FOURIER_TRANSFORM_HPP

#include <memory>

/**
 * @file FourierTransform.hpp.
 * @brief Declares Fourier Transform algorithms.
 */

#include "BitReversalPermutation.hpp"

namespace Transform {
namespace FourierTransform {

/**
 * @brief Represents an abstract direct or inverse Fourier Transform algorithm.
 *
 * The distinction between direct and inverse is made via the setBaseAngle
 * method, which specifies the direct method if angle = -pi. The inverse can be
 * obtained by setting angle = pi and dividing all elements in the result by the
 * length of the sequence. The algorithm is executed via the () operator.
 *
 * @see FourierTransformCalculator for higher level abstractions.
 */
class FourierTransformAlgorithm {
 public:
  /**
   * @brief Operator that executes the algorithm.
   *
   * @param input_sequence The input sequence.
   * @param output_sequence The output sequence.
   *
   * @note The sequences must have the same length n.
   */
  virtual void operator()(const vec &input_sequence,
                          vec &output_sequence) const = 0;

  /**
   * @brief Set the Base Angle value.
   *
   * This value should be -pi for the direct transform and pi for the inverse.
   *
   * @param angle The angle.
   */
  void setBaseAngle(real angle) { base_angle = angle; }

  virtual ~FourierTransformAlgorithm() = default;

  /**
   * @brief Execute the algorithm and return the execution time in microseconds.
   *
   * @param input_sequence The input sequence.
   * @param output_sequence The output sequence.
   * @return The time needed to execute the algorithm in microseconds.
   *
   * @note The sequences must have the same length n.
   */
  unsigned long calculateTime(const vec &input_sequence,
                              vec &output_sequence) const;

 protected:
  /**
   * @brief The internal angle used by the algorithm.
   */
  real base_angle;
};

/**
 * @brief The standard implementation of the Fourier Transform.
 *
 * @note The algorithm has time complexity O(n^2).
 */
class ClassicalFourierTransformAlgorithm : public FourierTransformAlgorithm {
 public:
  void operator()(const vec &input_sequence,
                  vec &output_sequence) const override;
  ~ClassicalFourierTransformAlgorithm() = default;
};

/**
 * @brief A recursive implmentation of the FFT.
 *
 * @note The algorithm has time complexity O(n log(n)).
 * @note Adapted from @cite
 * https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm.
 * @note The sequences' length must be a power of 2.
 */
class RecursiveFourierTransformAlgorithm : public FourierTransformAlgorithm {
 public:
  void operator()(const vec &input_sequence,
                  vec &output_sequence) const override;
  ~RecursiveFourierTransformAlgorithm() = default;
};

/**
 * @brief An iterative implementation of the FFT.
 *
 * An iterative implementation of the FFT using OpenMP. The algorithm will use
 * the bit reversal permutation algorithm provided to the constructor.
 *
 * @note The algorithm has time complexity O(n log(n)).
 * @note Adapted from @cite
 * https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm.
 * @note The sequences' length must be a power of 2.
 * @note Unless omp_set_num_threads() has been called, the algorithm will use
 * all available OpenMP threads.
 */
class IterativeFourierTransformAlgorithm : public FourierTransformAlgorithm {
 public:
  void operator()(const vec &input_sequence,
                  vec &output_sequence) const override;
  IterativeFourierTransformAlgorithm(
      std::unique_ptr<BitReversalPermutationAlgorithm> &algorithm)
      : bit_reversal_algorithm(std::move(algorithm)){};
  ~IterativeFourierTransformAlgorithm() = default;

 private:
  /**
   * @brief The algorithm used to perform the bit reversal permutation of the
   * input.
   */
  std::unique_ptr<BitReversalPermutationAlgorithm> bit_reversal_algorithm;
};

/**
 * @brief A 2D FFT algorithm, for non-NVIDIA devices.
 *
 * The algorithm is based on the 1D FFT algorithm. This algorithm is not
 * optimized and is only used to test the correctness of other algorithms, in a
 * machine without CUDA support. This class does not inherit from
 * FourierTransformAlgorithm since it does not implement its inverse. The
 * algorithm applies a 1D FFT to all rows and then to all columns of the
 * temporary results.
 *
 * @see TrivialTwoDimensionalInverseFourierTransformAlgorithm
 */
class TrivialTwoDimensionalFourierTransformAlgorithm {
 public:
  /**
   * @brief Apply the 2D FFT to a sequence.
   *
   * @param input_matrix The sequence to use as an input to the 2D FFT,
   * interpreted as a square matrix.
   * @param output_sequence An output sequence containing the transformed
   * input, to be interpreted as a matrix.
   * @note input_sequence and output_sequence must have the same length n,
   * which must be a power of 2 and a perfect square.
   */
  void operator()(const vec &input_sequence, vec &output_sequence) const;
  ~TrivialTwoDimensionalFourierTransformAlgorithm() = default;
};

/**
 * @brief A 2D IFFT algorithm, for non-NVIDIA devices.
 *
 * The algorithm is based on the 1D IFFT algorithm. This algorithm is not
 * optimized and is only used to test the correctness of other algorithms, in a
 * machine without CUDA support. The algorithm is the inverse of
 * TrivialTwoDimensionalFourierTransformAlgorithm.
 *
 * @note input_sequence and output_sequence must have the same length n, which
 * has to be a power of 2 and a perfect square.
 *
 * @see TrivialTwoDimensionalFourierTransformAlgorithm
 */
class TrivialTwoDimensionalInverseFourierTransformAlgorithm {
 public:
  /**
   * @brief Apply the 2D IFFT to a sequence.
   *
   * @param input_matrix The sequence to use as an input to the 2D IFFT,
   * interpreted as a square matrix.
   * @param output_sequence An output sequence containing the anti-transformed
   * input, to be interpreted as a matrix.
   * @note input_sequence and output_sequence must have the same length n,
   * which must be a power of 2 and a perfect square.
   * @note Notice that using this operator performs the full 2D IFFT of the
   * input, while the operator in FourierTransformAlgorithm does not scale the
   * result by n.
   */
  void operator()(const vec &input_sequence, vec &output_sequence) const;
  ~TrivialTwoDimensionalInverseFourierTransformAlgorithm() = default;
};

/**
 * @todo Document this class.
 */
class IterativeFFTGPU : public FourierTransformAlgorithm {
 public:
  void operator()(const vec &input_sequence,
                  vec &output_sequence) const override;

  ~IterativeFFTGPU() = default;
};

/**
 * @todo Document this class.
 */
class IterativeFFTGPU2D : public FourierTransformAlgorithm {
 public:
  void operator()(const vec &input_sequence,
                  vec &output_sequence) const override;

  ~IterativeFFTGPU2D() = default;
};

/**
 * @brief Get a time estimate for the execution of a Fourier transform
 * algorithm in microseconds.
 *
 * This function calculates the time needed to compute the Fourier transform of
 * the given sequence using an instance of the provided algorithm, with power of
 * two threads ranging from 1 to max_num_threads. The results are printed
 * expressed in microseconds.
 *
 * @param algorithm The Fourier transform algorithm.
 * @param sequence The input sequence.
 * @param max_num_threads The maximum number of threads to use.
 */
void TimeEstimateFFT(std::unique_ptr<FourierTransformAlgorithm> &algorithm,
                     const vec &sequence, unsigned int max_num_threads);

}  // namespace FourierTransform
}  // namespace Transform

#endif  // FOURIER_TRANSFORM_HPP
