#ifndef VECTOR_EXPORTER_HPP
#define VECTOR_EXPORTER_HPP

#include <vector>
#include <complex>
#include <string>
#include "Real.hpp"

using vec = std::vector<std::complex<real>>;

// Utility function to write a sequence of complex numbers to a file.
void WriteToFile(const vec &sequence, const std::string &filename);

#endif //VECTOR_EXPORTER_HPP