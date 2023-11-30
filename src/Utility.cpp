#include "Utility.hpp"

#include <iostream>

using vec = std::vector<std::complex<real>>;

// Compare the values of "sequence" with those of "sequence_golden" and return
// true if the difference between the two is less than "precision" for all
// elements.
bool CompareResult(const vec &sequence_golden, const vec &sequence,
                   double precision, bool print_errors) {
  // Assert that the two sequences have the same length.
  if (sequence_golden.size() != sequence.size()) {
    if (print_errors)
      std::cout << "The sequences have different lengths!" << std::endl;
    return false;
  }

  vec errors;

  // Check that the difference between the two sequences is small enough.
  for (size_t i = 0; i < sequence_golden.size(); i++) {
    if (abs(sequence[i] - sequence_golden[i]) > precision) {
      if (!print_errors) return false;
      errors.emplace_back(i);
    }
  }

  // If no errors were found, return true.
  if (errors.size() == 0) return true;

  // Otherwise, print the errors and return false.
  std::cout << "Errors at indexes: ";
  for (size_t i = 0; i < errors.size(); i++) {
    std::cout << errors[i] << " ";
  }
  std::cout << std::endl;

  return false;
}