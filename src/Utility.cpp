#include "Utility.hpp"

#include <iostream>

using namespace FourierTransform;

// Compare "sequence_golden" and "sequence", assuming the first one is correct,
// with floating point precision "precision".
bool CompareVectors(const vec &sequence_golden, const vec &sequence,
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
    //@note The abs function is defined in <cmath>. It is overloaded for
    //     different types. Better use that version instead of the one in
    //     <cstdlib>. The latter is for integers.
    
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

void ScaleVector(vec &vector, const real scalar) {
  // Get the size of the vector
  const size_t n = vector.size();

#pragma omp parallel for default(none) shared(vector) firstprivate(n, scalar)
  for (size_t i = 0; i < n; i++) {
    vector[i] *= scalar;
  }
}