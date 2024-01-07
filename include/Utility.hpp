#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "Real.hpp"

// Compare the values of "sequence" with those of "sequence_golden" and return
// true if the difference between the two is less than "precision" for all
// elements.
template <typename T>
bool CompareVectors(const std::vector<T> &sequence_golden,
                    const std::vector<T> &sequence, double precision,
                    bool print_errors) {
  // Assert that the two sequences have the same length.
  if (sequence_golden.size() != sequence.size()) {
    if (print_errors)
      std::cout << "The sequences have different lengths!" << std::endl;
    return false;
  }

  std::vector<size_t> errors;

  // Check that the difference between the two sequences is small enough.
  for (size_t i = 0; i < sequence_golden.size(); i++) {
    if (std::abs(sequence[i] - sequence_golden[i]) > precision) {
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

// Write "sequence" to a file, one element per line.
template <typename T>
void WriteToFile(const std::vector<T> &sequence, const std::string &filename) {
  // Open the file.
  std::ofstream file(filename);

  // Write the sequence to the file in .csv format, with full precision.
  for (size_t i = 0; i < sequence.size(); i++) {
    file << std::setprecision(
        std::numeric_limits<Transform::real>::max_digits10);
    file << sequence[i] << std::endl;
  }

  file.close();

  // Notify the user.
  std::cout << "Written data to [" << filename << "]." << std::endl;
}

// Specialization of writeToFile for complex numbers.
template <>
inline void WriteToFile(const Transform::FourierTransform::vec &sequence,
                        const std::string &filename) {
  // Open the file.
  std::ofstream file(filename);

  // Write the sequence to the file in .csv format, with full precision.
  for (size_t i = 0; i < sequence.size(); i++) {
    file << std::setprecision(
        std::numeric_limits<Transform::real>::max_digits10);
    file << sequence[i].real() << "," << sequence[i].imag() << std::endl;
  }

  file.close();

  // Notify the user.
  std::cout << "Written data to [" << filename << "]." << std::endl;
}

#endif  // UTILITY_HPP