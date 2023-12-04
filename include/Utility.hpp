#include <complex>
#include <vector>

#include "Real.hpp"

// Compare the values of "sequence" with those of "sequence_golden" and return
// true if the difference between the two is less than "precision" for all
// elements.
bool CompareVectors(const FourierTransform::vec &sequence_golden, const FourierTransform::vec &sequence,
                   double precision, bool print_errors);

// Multiply all elements of "vector" by "scalar".
void ScaleVector(FourierTransform::vec &vector, const FourierTransform::real scalar);