#ifndef UTILITY_HPP
#define UTILITY_HPP

/**
 * @file Utility.hpp.
 * @brief Declares utility functions to operate on vectors.
 */

#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "Real.hpp"

/**
 * @brief Compare two vectors of numbers for errors.
 *
 * @tparam T The type of the elements contained in the sequences.
 * @param sequence_golden The sequence that is assumed to be correct.
 * @param sequence The sequence whose correctness has to be checked.
 * @param tolerance The tolerance on the error between any two
 * elements in the sequences.
 * @param print_errors If true, print the indices in which the sequences differ
 * more than the specified tolerance.
 * @param use_relative_error If true, errors are checked using the relative
 * error, otherwise the absolute error is used.
 * @return true If the sequences are equal up to the specified tolerance.
 * @return false If the sequences are not equal up to the specified tolerance.
 */
template <typename T>
bool CompareVectors(const std::vector<T> &sequence_golden,
                    const std::vector<T> &sequence, Transform::real tolerance,
                    bool print_errors, bool use_relative_error = true) {
  // Assert that the two sequences have the same length.
  if (sequence_golden.size() != sequence.size()) {
    if (print_errors)
      std::cout << "The sequences have different lengths!" << std::endl;
    return false;
  }

  std::vector<size_t> errors;

  // Check that the difference between the two sequences is small enough.
  for (size_t i = 0; i < sequence_golden.size(); i++) {
    // Consider the case where absolute tolerance is used, or when one of the
    // elements is zero.
    if (!use_relative_error || std::abs(sequence_golden[i]) == 0 ||
        std::abs(sequence[i]) == 0) {
      if (std::abs(sequence[i] - sequence_golden[i]) > tolerance) {
        if (!print_errors) return false;
        errors.emplace_back(i);
      }
      // Otherwise, consider the relative tolerance.
    } else if (std::abs(sequence[i] - sequence_golden[i]) >
               tolerance * std::abs(sequence_golden[i])) {
      std::cout << sequence[i] << " " << sequence_golden[i] << std::endl;
      if (!print_errors) return false;
      errors.emplace_back(i);
    }
  }

  // If no errors were found, return true.
  if (errors.size() == 0) return true;

  // Otherwise, print the errors and return false.
  std::cout << "Errors at indices: ";
  for (size_t i = 0; i < errors.size(); i++) {
    std::cout << errors[i] << " ";
  }
  std::cout << std::endl;

  return false;
}

/**
 * @brief Write a sequence of numbers to a file, one element to line.
 *
 * @tparam T The type of the elements contained in the sequence.
 * @param sequence The sequence to write.
 * @param filename The path of the file to write to.
 */
template <typename T>
void WriteToFile(const std::vector<T> &sequence, const std::string &file_path) {
  // Open the file and check at it was opened correctly.
  std::ofstream file(file_path);
  assert(!file.fail());

  // Write the sequence to the file in .csv format, with full precision.
  for (size_t i = 0; i < sequence.size(); i++) {
    file << std::setprecision(std::numeric_limits<T>::max_digits10);
    file << sequence[i] << std::endl;
  }

  // Close the file and check at it was closed correctly.
  file.close();
  assert(!file.fail());

  // Notify the user.
  std::cout << "Written data to [" << file_path << "]." << std::endl;
}

/**
 * @brief Write a sequence of complex numbers to a file, one element to line.
 *
 * Write a sequence of complex numbers to a file, one element to line, with real
 * and complex part separated by a comma and without spaces.
 *
 * @param sequence The sequence to write.
 * @param file_path The path of the file to write to.
 */
template <>
inline void WriteToFile(const Transform::FourierTransform::vec &sequence,
                        const std::string &file_path) {
  // Open the file and check at it was opened correctly.
  std::ofstream file(file_path);
  assert(!file.fail());

  // Write the sequence to the file in .csv format, with full precision.
  for (size_t i = 0; i < sequence.size(); i++) {
    file << std::setprecision(
        std::numeric_limits<Transform::real>::max_digits10);
    file << sequence[i].real() << "," << sequence[i].imag() << std::endl;
  }

  // Close the file and check at it was closed correctly.
  file.close();
  assert(!file.fail());

  // Notify the user.
  std::cout << "Written data to [" << file_path << "]." << std::endl;
}

// Map values to the range [min_value, max_value] with an affine map, mapping
// the lowest element in values to min_value and the highest to max_value. If
// all elements in values are the same, they are all mapped into the the average
// of min_value and max_value.
/**
 * @brief Perform an affine map on a vector of numbers.
 *
 * Map the elements in the vector into the range [min_value, max_value] using an
 * affine map. The lowest element is mapped to min_value and the highest to
 * max_value. If all elements have the same value, they are all mapped into the
 * average of max_value and min_value.
 *
 * @tparam T The type of the elements in the vector.
 * @param value The vector whose elements have to be remapped.
 * @param min_value The minimum value in the output range.
 * @param min_value The maximum value in the output range.
 */
template <typename T>
void affineMap(std::vector<T> &values, T min_value, T max_value) {
  const T range_start = *std::min_element(values.begin(), values.end());
  const T range_end = *std::max_element(values.begin(), values.end());
  const T range = range_end - range_start;

  for (size_t i = 0; i < values.size(); i++) {
    // Default case.
    if (range != 0) {
      values[i] = (values[i] - range_start) / range * (max_value - min_value) +
                  min_value;
      // Case where all values are the same.
    } else {
      values[i] = (min_value + max_value) / 2;
    }
  }
}

#endif  // UTILITY_HPP