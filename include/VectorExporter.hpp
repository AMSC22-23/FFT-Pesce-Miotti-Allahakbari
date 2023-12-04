#ifndef VECTOR_EXPORTER_HPP
#define VECTOR_EXPORTER_HPP

#include <string>

#include "Real.hpp"

// Utility function to write a sequence of complex numbers to a file in the .csv
// format.
void WriteToFile(const FourierTransform::vec &sequence,
                 const std::string &filename);

#endif  // VECTOR_EXPORTER_HPP