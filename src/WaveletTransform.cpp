#include "WaveletTransform.hpp"

#include <tgmath.h>

#include <cassert>

namespace Transform {
namespace WaveletTransform {

// Constants used in the 9/7 fast wavelet transform. Source:
// https://getreuer.info/posts/waveletcdf97/index.html.
constexpr real alpha = -1.586134342;
constexpr real beta = -0.05298011854;
constexpr real delta = 0.8829110762;
constexpr real gamma = 0.4435068522;
constexpr real kappa = 1.149604398;

void DirectWaveletTransform97(std::vector<real> &sequence) {
  // Getting the input size.
  const size_t n = sequence.size();

  // Check that the size of the sequence is a power of 2.
  const size_t log_n = static_cast<size_t>(log2(n));
  assert(1UL << log_n == n);

  // Allocate space for a temporary sequence.
  std::vector<real> temporary_sequence(n, 0);

  // Predict 1.
  for (size_t i = 1; i < n - 2; i += 2) {
    sequence[i] += alpha * (sequence[i - 1] + sequence[i + 1]);
  }
  sequence[n - 1] += 2 * alpha * sequence[n - 2];

  // Update 1.
  for (size_t i = 2; i < n; i += 2) {
    sequence[i] += beta * (sequence[i - 1] + sequence[i + 1]);
  }
  sequence[0] += 2 * beta * sequence[1];

  // Predict 2.
  for (size_t i = 1; i < n - 2; i += 2) {
    sequence[i] += delta * (sequence[i - 1] + sequence[i + 1]);
  }
  sequence[n - 1] += 2 * delta * sequence[n - 2];

  // Update 2.
  for (size_t i = 2; i < n; i += 2) {
    sequence[i] += gamma * (sequence[i - 1] + sequence[i + 1]);
  }
  sequence[0] += 2 * gamma * sequence[1];

  // Scale.
  for (size_t i = 0; i < n; i++) {
    if (i % 2)
      sequence[i] /= kappa;
    else
      sequence[i] *= kappa;
  }

  // Pack.
  for (size_t i = 0; i < n; i++) {
    if (i % 2 == 0)
      temporary_sequence[i / 2] = sequence[i];
    else
      temporary_sequence[n / 2 + i / 2] = sequence[i];
  }

  // Copy data into the final sequence.
  for (size_t i = 0; i < n; i++) {
    sequence[i] = temporary_sequence[i];
  }
}

void InverseWaveletTransform97(std::vector<real> &sequence) {
  // Getting the input size.
  const size_t n = sequence.size();

  // Check that the size of the sequence is a power of 2.
  const size_t log_n = static_cast<size_t>(log2(n));
  assert(1UL << log_n == n);

  // Allocate space for a temporary sequence.
  std::vector<real> temporary_sequence(n, 0);

  // Unpack.
  for (size_t i = 0; i < n / 2; i++) {
    temporary_sequence[i * 2] = sequence[i];
    temporary_sequence[i * 2 + 1] = sequence[i + n / 2];
  }

  // Copy data into the result sequence.
  for (size_t i = 0; i < n; i++) {
    sequence[i] = temporary_sequence[i];
  }

  // Undo scale.
  for (size_t i = 0; i < n; i++) {
    if (i % 2)
      sequence[i] *= kappa;
    else
      sequence[i] /= kappa;
  }

  // Undo update 2.
  for (size_t i = 2; i < n; i += 2) {
    sequence[i] -= gamma * (sequence[i - 1] + sequence[i + 1]);
  }
  sequence[0] -= 2 * gamma * sequence[1];

  // Undo predict 2.
  for (size_t i = 1; i < n - 2; i += 2) {
    sequence[i] -= delta * (sequence[i - 1] + sequence[i + 1]);
  }
  sequence[n - 1] -= 2 * delta * sequence[n - 2];

  // Undo update 1.
  for (size_t i = 2; i < n; i += 2) {
    sequence[i] -= beta * (sequence[i - 1] + sequence[i + 1]);
  }
  sequence[0] -= 2 * beta * sequence[1];

  // Undo predict 1.
  for (size_t i = 1; i < n - 2; i += 2) {
    sequence[i] -= alpha * (sequence[i - 1] + sequence[i + 1]);
  }
  sequence[n - 1] -= 2 * alpha * sequence[n - 2];
}

}  // namespace WaveletTransform
}  // namespace Transform