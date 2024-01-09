#ifndef WAVELET_TRANSFORM_HPP
#define WAVELET_TRANSFORM_HPP

/**
 * @file WaveletTransform.hpp.
 * @brief Declares Wavelet Transform algorithms.
 */

#include <memory>

#include "Real.hpp"

namespace Transform {
namespace WaveletTransform {

/** @namespace */

/**
 * @brief Represents an abstract 1D Wavelet Transform algorithm.
 *
 * The WaveletTransformAlgorithm abstract class is designed as a common
 * interface for Wavelet Transform algorithms. It allows for execution of a DWT
 * (direct Discrete Wavelet Transform) and IDWT (Inverse Discrete Wavelet
 * Transform).
 *
 * Example usage:
 * @code
 * std::unique_ptr<WaveletTransformAlgorithm> algorithm;
 *
 * ...
 *
 * algorithm->directTransform(original_sequence, dwt);
 * algorithm->inverseTransform(dwt, idwt);
 * @endcode
 *
 * @note Different DWT algorithms are not equivalent, so the same algorithm must
 * be used if an IDWT is required.
 */
class WaveletTransformAlgorithm {
 public:
  /**
   * @brief Perform the DWT of a sequence.
   *
   * @param input_sequence The sequence to use as an input to the DWT.
   * @param output_sequence An output sequence containing the transformed input.
   *
   * @note The sequences must have the same length n.
   */
  virtual void directTransform(const std::vector<real> &input_sequence,
                               std::vector<real> &output_sequence) const = 0;

  /**
   * @brief Perform the IDWT of a sequence.
   *
   * @param input_sequence The sequence to use as an input to the IDWT.
   * @param output_sequence An output sequence containing the anti-transformed
   * input.
   *
   * @note The sequences must have the same length n.
   */
  virtual void inverseTransform(const std::vector<real> &input_sequence,
                                std::vector<real> &output_sequence) const = 0;

  virtual ~WaveletTransformAlgorithm() = default;
};

/**
 * @brief An implementation of the 9/7 wavelet transform.
 *
 * @note The length of the sequences n must be a power of 2 and larger than 1.
 * @note The algorithm has time complexity O(n).
 * @note Adapted from @cite
 * https://web.archive.org/web/20120305164605/http://www.embl.de/~gpau/misc/dwt97.c.
 */
class GPWaveletTransform97 : public WaveletTransformAlgorithm {
 public:
  void directTransform(const std::vector<real> &input_sequence,
                       std::vector<real> &output_sequence) const override;
  void inverseTransform(const std::vector<real> &input_sequence,
                        std::vector<real> &output_sequence) const override;
  ~GPWaveletTransform97() = default;
};

/**
 * @brief An implementation of the 9/7 wavelet transform.
 *
 * @note The length of the sequences n must be even and larger than 1.
 * @note The algorithm has time complexity O(n).
 * @note The algorithm is massively parallel.
 * @note Adapted from @cite
 * https://services.math.duke.edu/~ingrid/publications/J_Four_Anal_Appl_4_p247.pdf
 * (publicated by Daubechies) and @cite
 * https://www.sciencedirect.com/science/article/pii/S016516841100199X. Since
 * the sources do not specify how to handle values at the boundary, the same
 * approach as @cite
 * https://web.archive.org/web/20120305164605/http://www.embl.de/~gpau/misc/dwt97.c
 * was used.
 */
class DaubechiesWaveletTransform97 : public WaveletTransformAlgorithm {
 public:
  void directTransform(const std::vector<real> &input_sequence,
                       std::vector<real> &output_sequence) const override;
  void inverseTransform(const std::vector<real> &input_sequence,
                        std::vector<real> &output_sequence) const override;
  ~DaubechiesWaveletTransform97() = default;
};

/**
 * @brief Represents a 2D Wavelet Transform algorithm.
 *
 * The WaveletTransformAlgorithm class is designed to execute a 2D Wavelet
 * Transform algorithm, after being provided the corresponding 1D algorithm.
 *
 * Example usage:
 * @code
 * std::unique_ptr<WaveletTransformAlgorithm> algorithm;
 *
 * ...
 *
 * TwoDimensionalWaveletTransformAlgorithm algorithm_2d(algorithm);
 * unsigned int levels = 1;
 * algorithm_2d.directTransform(original_sequence, dwt, levels);
 * algorithm_2d.inverseTransform(dwt, idwt, levels);
 * @endcode
 */
class TwoDimensionalWaveletTransformAlgorithm {
 public:
  /**
   * @brief Perform the 2D DWT of a sequence.
   *
   * The 1 level algorithm works by applying the 1D DWT to all rows of the input
   * matrix and then to all columns of the temporary result. Multi level 2D DWTs
   * apply the same algorithm iteratively to the upper left quadrant of the
   * matrix. The algorithm will use the bit reversal permutation algorithm
   * provided to the constructor, which will be moved.
   *
   * @param input_matrix The sequence to use as an input to the DWT, interpred
   * as a square matrix.
   * @param output_sequence An output sequence containing the transformed input,
   * to be interpreted as a matrix.
   * @param levels The number of levels for the 2D DWT.
   * @param use_openmp If true, OpenMP is used for parallelizing the outer loop,
   * otherwise it is not. Regardless, it is not used for the 1D DWT.
   *
   * @note Unless omp_set_num_threads() has been called, the algorithm will use
   * all available OpenMP threads.
   * @note The sequences must have the same length n, which must be a power of
   * two, a perfect square and larger than 1 for each decomposition level.
   */
  void directTransform(const std::vector<real> &input_matrix,
                       std::vector<real> &output_matrix, unsigned int levels,
                       bool use_openmp = false) const;

  /**
   * @brief Perform the 2D IDWT of a sequence.
   *
   * The method performs the reverse algorithm of directTransform.
   *
   * @param input_matrix The sequence to use as an input to the IDWT,
   * interpreted as a square matrix.
   * @param output_sequence An output sequence containing the anti-transformed
   * input, to be interpreted as a matrix.
   * @param levels The number of levels for the 2D IWDT.
   * @param use_openmp If true, OpenMP is used for parallelizing the outer loop,
   * otherwise it is not. Regardless, it is not used for the 1D DWT.
   *
   * @note Unless omp_set_num_threads() has been called, the algorithm will use
   * all available OpenMP threads.
   * @note The sequences must have the same length n, which must be a power of
   * two, a perfect square and larger than 1 for each recomposition level.
   */
  void inverseTransform(const std::vector<real> &input_matrix,
                        std::vector<real> &output_matrix, unsigned int levels,
                        bool use_openmp = false) const;

  ~TwoDimensionalWaveletTransformAlgorithm() = default;

  TwoDimensionalWaveletTransformAlgorithm(
      std::unique_ptr<WaveletTransformAlgorithm> &algorithm_)
      : algorithm(std::move(algorithm_)){};

 private:
  /** @brief The 1D wavelet transform algorithm. */
  const std::unique_ptr<WaveletTransformAlgorithm> algorithm;

  /**
   * @brief Apply the 1D algorithm to the first n elements of the first n rows
   * of the matrix.
   * @param matrix The matrix to transform.
   * @param n The number of rows and length of each subrow to transform.
   * @param direct_algorithm If true, the DWT is applied, otherwise the IDWT is
   * applied instead.
   * @param use_openmp If true, OpenMP is used, otherwise it is not.
   */
  void transformRows(std::vector<real> &matrix, size_t n, bool direct_algorithm,
                     bool use_openmp = false) const;

  /**
   * @brief Apply the 1D algorithm to the first n elements of the first n
   * columns of the matrix.
   * @param matrix The matrix to transform.
   * @param n The number of columns and length of each subcolumn to transform.
   * @param direct_algorithm If true, the DWT is applied, otherwise the IDWT is
   * applied instead.
   * @param use_openmp If true, OpenMP is used, otherwise it is not.
   */
  void transformColumns(std::vector<real> &matrix, size_t n,
                        bool direct_algorithm, bool use_openmp = false) const;
};

}  // namespace WaveletTransform
}  // namespace Transform

#endif  // WAVELET_TRANSFORM_HPP