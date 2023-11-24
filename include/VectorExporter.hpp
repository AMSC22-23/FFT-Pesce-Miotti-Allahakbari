#ifndef VECTOR_EXPORTER_HPP
#define VECTOR_EXPORTER_HPP

#include <vector>
#include <complex>
#include <string>
#include "Real.hpp"

// Utility function to write a sequence of complex numbers to a file
void WriteToFile(const std::vector<std::complex<real>> &sequence, const std::string &filename);

#endif //VECTOR_EXPORTER_HPP