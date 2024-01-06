#ifndef FOURIER_TRANSFORM_HPP
#define FOURIER_TRANSFORM_HPP

#include <memory>

#include "BitReversalPermutation.hpp"

namespace Transform {
namespace FourierTransform {

// A class that represents a generic direct or inverse Fourier transform
// algorithm. The distinction between direct and inverse is made via the
// setBaseAngle method, which specifies the direct method if angle = -pi and the
// inverse method up to the constant sqrt(n) if angle = pi.
class FourierTransformAlgorithm {
 public:
  virtual void operator()(const vec &input_sequence,
                          vec &output_sequence) const = 0;
  void setBaseAngle(real angle) { base_angle = angle; }
  virtual ~FourierTransformAlgorithm() = default;
  // Execute the algorithm and return the time taken in microseconds.
  unsigned long calculateTime(const vec &input_sequence,
                              vec &output_sequence) const;

 protected:
  real base_angle;
};

// The classical O(n^2) algorithm.
class ClassicalFourierTransformAlgorithm : public FourierTransformAlgorithm {
 public:
  void operator()(const vec &input_sequence,
                  vec &output_sequence) const override;
  ~ClassicalFourierTransformAlgorithm() = default;
};

// A recursive O(n log n) algorithm. Source:
// https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm. It is
// assumed that input_sequence has a power of 2 elements.
class RecursiveFourierTransformAlgorithm : public FourierTransformAlgorithm {
 public:
  void operator()(const vec &input_sequence,
                  vec &output_sequence) const override;
  ~RecursiveFourierTransformAlgorithm() = default;
};

// An iterative O(n log n) algorithm. Source:
// https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm. An
// algorithm to perform the bit reversal permutation of the input must be
// specified. It is assumed that input_sequence has a power of 2 elements.
class IterativeFourierTransformAlgorithm : public FourierTransformAlgorithm {
 public:
  void operator()(const vec &input_sequence,
                  vec &output_sequence) const override;
  void setBitReversalPermutationAlgorithm(
      std::unique_ptr<BitReversalPermutationAlgorithm> &algorithm) {
    bit_reversal_algorithm = std::move(algorithm);
  }
  ~IterativeFourierTransformAlgorithm() = default;

 private:
  std::unique_ptr<BitReversalPermutationAlgorithm> bit_reversal_algorithm;
};

// A 2D FFT algorithm. We assume that the input matrix is square and its size
// is a power of 2. The algorithm is based on the 1D FFT algorithm.
// This algorithm is very slow and is only used to test the correctness of
// other algorithms, in a machine without CUDA support.
class TrivialTwoDimensionalFourierTransformAlgorithm
    : public FourierTransformAlgorithm {
 public:
  void operator()(const vec &input_sequence,
                  vec &output_sequence) const override;
  ~TrivialTwoDimensionalFourierTransformAlgorithm() = default;
};

// A 2D IFFT algorithm. We assume that the input matrix is square and its size
// is a power of 2. The algorithm is based on the 1D FFT algorithm.
// This algorithm is very slow and is only used to test the correctness of
// other algorithms, in a machine without CUDA support.
class TrivialTwoDimensionalInverseFourierTransformAlgorithm
    : public FourierTransformAlgorithm {
 public:
  void operator()(const vec &input_sequence,
                  vec &output_sequence) const override;
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

// Calculate the time needed to compute the Fourier transform of
// "sequence" using an instance of "ft_algorithm", with 1 to "max_num_threads"
// threads. The results are printed expressed in microseconds.
void TimeEstimateFFT(std::unique_ptr<FourierTransformAlgorithm> &ft_algorithm,
                     const vec &sequence, unsigned int max_num_threads);

}  // namespace FourierTransform
}  // namespace Transform

#endif  // FOURIER_TRANSFORM_HPP
