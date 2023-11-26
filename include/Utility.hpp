#include <vector>
#include <complex>

#include "Real.hpp"

// Compare the values of "sequence" with those of "sequence_golden" and return true if
// the difference between the two is less than "precision" for all elements.
bool CompareResult(
    const std::vector<std::complex<real>> &sequence_golden, 
    const std::vector<std::complex<real>> &sequence, 
    double precision, bool print_errors);