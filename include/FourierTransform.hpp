#ifndef FOURIER_TRANSFORM_HPP
#define FOURIER_TRANSFORM_HPP

#include <memory>

#include "BitReversalPermutation.hpp"

namespace Transform {
namespace FourierTransform {

/**
 * @brief A class that represents a generic direct or inverse Fourier transform algorithm.
 * 
 * The distinction between direct and inverse is made via the setBaseAngle method,
 * which specifies the direct method if angle = -pi and the inverse method up to
 * the constant sqrt(n) if angle = pi. The algorithm is executed via the () operator.
 */
class FourierTransformAlgorithm {
 public:
  /**
   * @brief Operator that executes the algorithm.
   * 
   * @param input_sequence The input sequence.
   * @param output_sequence The output sequence.
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
  
  /**
   * @brief Destroy the Fourier Transform Algorithm object.
   */
  virtual ~FourierTransformAlgorithm() = default;

  /**
   * @brief Execute the algorithm and calculate the time needed to do so.
   * 
   * @note The time is calculated in microseconds.
   * 
   * @param input_sequence The input sequence. 
   * @param output_sequence The output sequence.
   * @return unsigned long The time needed to execute the algorithm, in microseconds.
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
 * @brief The most naive algorithm, with O(n^2) complexity.
 */
class ClassicalFourierTransformAlgorithm : public FourierTransformAlgorithm {
 public:
  void operator()(const vec &input_sequence,
                  vec &output_sequence) const override;
  ~ClassicalFourierTransformAlgorithm() = default;
};


/**
 * @brief A recursive O(n log n) algorithm.
 * 
 * Source: https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm.
 * It is assumed that input_sequence has a power of 2 elements.
 */
class RecursiveFourierTransformAlgorithm : public FourierTransformAlgorithm {
 public:
  void operator()(const vec &input_sequence,
                  vec &output_sequence) const override;
  ~RecursiveFourierTransformAlgorithm() = default;
};

/**
 * @brief An iterative O(n log n) algorithm.
 * 
 * Source: https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm.
 * An algorithm to perform the bit reversal permutation of the input must
 * be specified. It is assumed that input_sequence has a power of 2 elements.
 */
class IterativeFourierTransformAlgorithm : public FourierTransformAlgorithm {
 public:
  void operator()(const vec &input_sequence,
                  vec &output_sequence) const override;

  /**
   * @brief Set the Bit Reversal Permutation Algorithm object
   * 
   * This algorithm is used to perform the bit reversal permutation of the input,
   * which is a part of the FFT algorithm.
   * 
   * @param algorithm The algorithm object. 
   */
  void setBitReversalPermutationAlgorithm(
      std::unique_ptr<BitReversalPermutationAlgorithm> &algorithm) {
    bit_reversal_algorithm = std::move(algorithm);
  }
  ~IterativeFourierTransformAlgorithm() = default;

 private:
  /**
   * @brief The algorithm used to perform the bit reversal permutation of the input.
   */
  std::unique_ptr<BitReversalPermutationAlgorithm> bit_reversal_algorithm;
};

/**
 * @brief A 2D FFT algorithm, for non-NVIDIA devices.
 * 
 * We assume that the input matrix is square and its size is a power of 2.
 * The algorithm is based on the 1D FFT algorithm. This algorithm is very slow
 * and is only used to test the correctness of other algorithms, in a machine
 * without CUDA support. This class does not inherit from FourierTransformAlgorithm
 * since it is not the same as its inverse.
 * 
 * @see TrivialTwoDimensionalInverseFourierTransformAlgorithm
 */
class TrivialTwoDimensionalFourierTransformAlgorithm {
 public:
  void operator()(const vec &input_sequence, vec &output_sequence) const;
  ~TrivialTwoDimensionalFourierTransformAlgorithm() = default;
};

// A 2D IFFT algorithm. We assume that the input matrix is square and its size
// is a power of 2. The algorithm is based on the 1D FFT algorithm.
// This algorithm is very slow and is only used to test the correctness of
// other algorithms, in a machine without CUDA support.

/**
 * @brief A 2D IFFT algorithm, for non-NVIDIA devices.
 * 
 * We assume that the input matrix is square and its size is a power of 2.
 * The algorithm is based on the 1D IFFT algorithm. This algorithm is very slow
 * and is only used to test the correctness of other algorithms, in a machine
 * without CUDA support.
 * 
 * @see TrivialTwoDimensionalFourierTransformAlgorithm
 */
class TrivialTwoDimensionalInverseFourierTransformAlgorithm {
 public:
  void operator()(const vec &input_sequence, vec &output_sequence) const;
  ~TrivialTwoDimensionalInverseFourierTransformAlgorithm() = default;
};

class IterativeFFTGPU : public FourierTransformAlgorithm {
 public:
  void operator()(const vec &input_sequence,
                  vec &output_sequence) const override;

  ~IterativeFFTGPU() = default;
};

class IterativeFFTGPU2D : public FourierTransformAlgorithm {
 public:
  void operator()(const vec &input_sequence,
                  vec &output_sequence) const override;

  ~IterativeFFTGPU2D() = default;
};

/**
 * @brief Get a time estimate for the execution of a Fourier transform algorithm.
 * 
 * This function calculates the time needed to compute the Fourier transform of
 * "sequence" using an instance of "ft_algorithm", with 1 to "max_num_threads"
 * threads. The results are printed expressed in microseconds.
 * 
 * @param ft_algorithm The Fourier transform algorithm.
 * @param sequence The input sequence.
 * @param max_num_threads The maximum number of threads to use.
 */
void TimeEstimateFFT(std::unique_ptr<FourierTransformAlgorithm> &ft_algorithm,
                     const vec &sequence, unsigned int max_num_threads);

}  // namespace FourierTransform
}  // namespace Transform

#endif  // FOURIER_TRANSFORM_HPP
