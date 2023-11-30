#ifndef VECTOR_EXPORTER_HPP
#define VECTOR_EXPORTER_HPP

#include <complex>
#include <string>
#include <vector>

#include "Real.hpp"

// Utility function to write a sequence of complex numbers to a file.
void WriteToFile(const std::vector<std::complex<real>> &sequence,
                 const std::string &filename);

#endif  // VECTOR_EXPORTER_HPP