#include "WaveletTransform.hpp"

#include <tgmath.h>

#include <cassert>
#include <iostream>

namespace Transform {
namespace WaveletTransform {

// Constants used in the 9/7 fast wavelet transform. Source:
// https://services.math.duke.edu/~ingrid/publications/J_Four_Anal_Appl_4_p247.pdf.
constexpr real alpha = -1.586134342;
constexpr real beta = -0.05298011854;
constexpr real gamma = 0.8829110762;
constexpr real delta = 0.4435068522;
constexpr real xi = 1.0 / 1.149604398;

void GPWaveletTransform97::directTransform(
    const std::vector<real> &input_sequence,
    std::vector<real> &output_sequence) const {
  // Copy the input.
  std::vector<real> temporary_sequence(input_sequence);

  // Get the input size.
  const size_t n = temporary_sequence.size();

  // Check that the size of the sequence is a power of 2.
  assert(1UL << static_cast<size_t>(log2(n)) == n);

  // Predict 1.
  for (size_t i = 1; i < n - 2; i += 2) {
    temporary_sequence[i] +=
        alpha * (temporary_sequence[i - 1] + temporary_sequence[i + 1]);
  }
  temporary_sequence[n - 1] += 2 * alpha * temporary_sequence[n - 2];

  // Update 1.
  for (size_t i = 2; i < n; i += 2) {
    temporary_sequence[i] +=
        beta * (temporary_sequence[i - 1] + temporary_sequence[i + 1]);
  }
  temporary_sequence[0] += 2 * beta * temporary_sequence[1];

  // Predict 2.
  for (size_t i = 1; i < n - 2; i += 2) {
    temporary_sequence[i] +=
        gamma * (temporary_sequence[i - 1] + temporary_sequence[i + 1]);
  }
  temporary_sequence[n - 1] += 2 * gamma * temporary_sequence[n - 2];

  // Update 2.
  for (size_t i = 2; i < n; i += 2) {
    temporary_sequence[i] +=
        delta * (temporary_sequence[i - 1] + temporary_sequence[i + 1]);
  }
  temporary_sequence[0] += 2 * delta * temporary_sequence[1];

  // Scale.
  for (size_t i = 0; i < n; i++) {
    if (i % 2)
      temporary_sequence[i] *= xi;
    else
      temporary_sequence[i] /= xi;
  }

  // Pack and copy data into the output sequence.
  for (size_t i = 0; i < n; i++) {
    if (i % 2 == 0)
      output_sequence[i / 2] = temporary_sequence[i];
    else
      output_sequence[n / 2 + i / 2] = temporary_sequence[i];
  }
}

void GPWaveletTransform97::inverseTransform(
    const std::vector<real> &input_sequence,
    std::vector<real> &output_sequence) const {
  // Copy the input.
  std::vector<real> temporary_sequence(input_sequence);

  // Get the input size.
  const size_t n = temporary_sequence.size();

  // Check that the size of the sequence is a power of 2.
  assert(1UL << static_cast<size_t>(log2(n)) == n);

  // Unpack.
  for (size_t i = 0; i < n / 2; i++) {
    temporary_sequence[i * 2] = input_sequence[i];
    temporary_sequence[i * 2 + 1] = input_sequence[i + n / 2];
  }

  // Copy data into the result sequence.
  for (size_t i = 0; i < n; i++) {
    output_sequence[i] = temporary_sequence[i];
  }

  // Undo scale.
  for (size_t i = 0; i < n; i++) {
    if (i % 2)
      output_sequence[i] /= xi;
    else
      output_sequence[i] *= xi;
  }

  // Undo update 2.
  for (size_t i = 2; i < n; i += 2) {
    output_sequence[i] -=
        delta * (output_sequence[i - 1] + output_sequence[i + 1]);
  }
  output_sequence[0] -= 2 * delta * output_sequence[1];

  // Undo predict 2.
  for (size_t i = 1; i < n - 2; i += 2) {
    output_sequence[i] -=
        gamma * (output_sequence[i - 1] + output_sequence[i + 1]);
  }
  output_sequence[n - 1] -= 2 * gamma * output_sequence[n - 2];

  // Undo update 1.
  for (size_t i = 2; i < n; i += 2) {
    output_sequence[i] -=
        beta * (output_sequence[i - 1] + output_sequence[i + 1]);
  }
  output_sequence[0] -= 2 * beta * output_sequence[1];

  // Undo predict 1.
  for (size_t i = 1; i < n - 2; i += 2) {
    output_sequence[i] -=
        alpha * (output_sequence[i - 1] + output_sequence[i + 1]);
  }
  output_sequence[n - 1] -= 2 * alpha * output_sequence[n - 2];
}

void DaubechiesWaveletTransform97::directTransform(
    const std::vector<real> &input_sequence,
    std::vector<real> &output_sequence) const {
  // Get the input size.
  const size_t n = input_sequence.size();

  // Allocate two temporary vectors.
  std::vector<real> low_sequence;
  std::vector<real> high_sequence;
  low_sequence.reserve(n / 2);
  high_sequence.reserve(n / 2);

  // Splitting phase.
  for (size_t i = 0; i < n / 2; i++) {
    low_sequence.emplace_back(input_sequence[2 * i]);
    high_sequence.emplace_back(input_sequence[2 * i + 1]);
  }

  // Prediction phase 1.
  for (size_t i = 0; i < n / 2 - 1; i++) {
    high_sequence[i] += alpha * (low_sequence[i] + low_sequence[i + 1]);
  }
  high_sequence[n / 2 - 1] += 2 * alpha * low_sequence[n / 2 - 1];

  // Updating phase 1.
  for (size_t i = 1; i < n / 2; i++) {
    low_sequence[i] += beta * (high_sequence[i - 1] + high_sequence[i]);
  }
  low_sequence[0] += 2 * beta * high_sequence[0];

  // Prediction phase 2.
  for (size_t i = 0; i < n / 2 - 1; i++) {
    high_sequence[i] += gamma * (low_sequence[i] + low_sequence[i + 1]);
  }
  high_sequence[n / 2 - 1] += 2 * gamma * low_sequence[n / 2 - 1];

  // Updating phase 2.
  for (size_t i = 1; i < n / 2; i++) {
    low_sequence[i] += delta * (high_sequence[i - 1] + high_sequence[i]);
  }
  low_sequence[0] += 2 * delta * high_sequence[0];

  // Scaling.
  for (size_t i = 0; i < n / 2; i++) {
    low_sequence[i] *= xi;
    high_sequence[i] /= xi;
  }

  // Merging into a single sequence.
  for (size_t i = 0; i < n / 2; i++) {
    output_sequence[i] = low_sequence[i];
  }
  for (size_t i = n / 2; i < n; i++) {
    output_sequence[i] = high_sequence[i - n / 2];
  }
}

void DaubechiesWaveletTransform97::inverseTransform(
    const std::vector<real> &input_sequence,
    std::vector<real> &output_sequence) const {
  // Get the input size.
  const size_t n = input_sequence.size();

  // Allocate two temporary vectors.
  std::vector<real> low_sequence;
  std::vector<real> high_sequence;
  low_sequence.reserve(n / 2);
  high_sequence.reserve(n / 2);

  // Undo merging into a single sequence.
  for (size_t i = 0; i < n / 2; i++) {
    low_sequence.emplace_back(input_sequence[i]);
  }
  for (size_t i = n / 2; i < n; i++) {
    high_sequence.emplace_back(input_sequence[i]);
  }

  // Undo Scaling.
  for (size_t i = 0; i < n / 2; i++) {
    low_sequence[i] /= xi;
    high_sequence[i] *= xi;
  }

  // Undo updating phase 2.
  for (size_t i = 1; i < n / 2; i++) {
    low_sequence[i] -= delta * (high_sequence[i - 1] + high_sequence[i]);
  }
  low_sequence[0] -= 2 * delta * high_sequence[0];

  // Undo prediction phase 2.
  for (size_t i = 0; i < n / 2 - 1; i++) {
    high_sequence[i] -= gamma * (low_sequence[i] + low_sequence[i + 1]);
  }
  high_sequence[n / 2 - 1] -= 2 * gamma * low_sequence[n / 2 - 1];

  // Undo updating phase 1.
  for (size_t i = 1; i < n / 2; i++) {
    low_sequence[i] -= beta * (high_sequence[i - 1] + high_sequence[i]);
  }
  low_sequence[0] -= 2 * beta * high_sequence[0];

  // Undo prediction phase 1.
  for (size_t i = 0; i < n / 2 - 1; i++) {
    high_sequence[i] -= alpha * (low_sequence[i] + low_sequence[i + 1]);
  }
  high_sequence[n / 2 - 1] -= 2 * alpha * low_sequence[n / 2 - 1];

  // Undo splitting phase.
  for (size_t i = 0; i < n / 2; i++) {
    output_sequence[2 * i] = low_sequence[i];
    output_sequence[2 * i + 1] = high_sequence[i];
  }
}

void TwoDimensionalWaveletTransformAlgorithm::transformRows(
    std::vector<real> &matrix, size_t n, bool direct) const {
  // Get the length of the input.
  const size_t N_squared = matrix.size();
  const size_t N = static_cast<size_t>(sqrt(N_squared));

  // Check that the sizes are a power of 2 and at least 2, that the vector is
  // large enough and that the matrix is square.
  assert(n <= N);
  assert(n >= 2 && (1UL << static_cast<size_t>(log2(n))) == n);
  assert(1UL << static_cast<size_t>(log2(N)) == N);
  assert(N * N == N_squared);

  // Perform the transform on all rows.
  for (size_t i = 0; i < n; i++) {
    // Get a vector with the current row.
    std::vector<real> row;
    row.reserve(n);
    for (size_t j = 0; j < n; j++) {
      row.emplace_back(matrix[N * i + j]);
    }

    // Allocate space for the result.
    std::vector<real> transformed_row(n, 0);

    // Transform the row.
    if (direct) {
      this->algorithm->directTransform(row, transformed_row);
    } else {
      this->algorithm->inverseTransform(row, transformed_row);
    }

    // Write the result to the output matrix.
    for (size_t j = 0; j < n; j++) {
      matrix[N * i + j] = transformed_row[j];
    }
  }
}

void TwoDimensionalWaveletTransformAlgorithm::transformColumns(
    std::vector<real> &matrix, size_t n, bool direct) const {
  // Get the length of the input.
  const size_t N_squared = matrix.size();
  const size_t N = static_cast<size_t>(sqrt(N_squared));

  // Check that the sizes are a power of 2 and at least 2, that the vector is
  // large enough and that the matrix is square.
  assert(n <= N);
  assert(n >= 2 && (1UL << static_cast<size_t>(log2(n))) == n);
  assert(1UL << static_cast<size_t>(log2(N)) == N);
  assert(N * N == N_squared);

  // Perform the transform on all columns.
  for (size_t i = 0UL; i < n; i++) {
    // Get a vector with the current columns.
    std::vector<real> column;
    column.reserve(n);
    for (size_t j = 0; j < n; j++) {
      column.emplace_back(matrix[i + N * j]);
    }

    // Allocate space for the result.
    std::vector<real> transformed_column(n, 0);

    // Transform the column.
    if (direct) {
      this->algorithm->directTransform(column, transformed_column);
    } else {
      this->algorithm->inverseTransform(column, transformed_column);
    }

    // Write the result to the output matrix.
    for (size_t j = 0; j < n; j++) {
      matrix[i + N * j] = transformed_column[j];
    }
  }
}

void TwoDimensionalWaveletTransformAlgorithm::directTransform(
    const std::vector<real> &input_matrix, std::vector<real> &output_matrix,
    unsigned int levels) const {
  // Check that the inputs have the same size.
  assert(input_matrix.size() == output_matrix.size());

  // Copy the input into the output.
  for (size_t i = 0UL; i < input_matrix.size(); i++) {
    output_matrix[i] = input_matrix[i];
  }

  // Get the initial size and assert that the input has a length that is a
  // perfect square.
  size_t n_squared = input_matrix.size();
  size_t n = static_cast<size_t>(sqrt(n_squared));
  assert(n * n == n_squared);

  // Apply the algorithm to all rows and columns in the top left corner. Halve
  // the length of a side each step.
  for (unsigned int level = 0U; level < levels; level++) {
    TwoDimensionalWaveletTransformAlgorithm::transformRows(output_matrix, n,
                                                           true);
    TwoDimensionalWaveletTransformAlgorithm::transformColumns(output_matrix, n,
                                                              true);
    n = n >> 1UL;
  }
}

void TwoDimensionalWaveletTransformAlgorithm::inverseTransform(
    const std::vector<real> &input_matrix, std::vector<real> &output_matrix,
    unsigned int levels) const {
  // Check that the inputs have the same size.
  assert(input_matrix.size() == output_matrix.size());

  // Copy the input into the output.
  for (size_t i = 0UL; i < input_matrix.size(); i++) {
    output_matrix[i] = input_matrix[i];
  }

  // Get the initial size and assert that the input has a length that is a
  // perfect square.
  size_t n_squared = input_matrix.size();
  size_t n = static_cast<size_t>(sqrt(n_squared));
  assert(n * n == n_squared);
  n = n >> (levels - 1U);

  // Apply the algorithm to all rows and columns in the top left corner.
  // Double the length of a side each step.
  for (unsigned int level = 0U; level < levels; level++) {
    TwoDimensionalWaveletTransformAlgorithm::transformColumns(output_matrix, n,
                                                              false);
    TwoDimensionalWaveletTransformAlgorithm::transformRows(output_matrix, n,
                                                           false);
    n = n << 1UL;
  }
}

void TwoDimensionalWaveletTransformAlgorithm::setAlgorithm(
    std::unique_ptr<WaveletTransformAlgorithm> &algorithm) {
  this->algorithm = std::move(algorithm);
}

}  // namespace WaveletTransform
}  // namespace Transform