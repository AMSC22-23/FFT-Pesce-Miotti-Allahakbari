#ifndef VECTOR_EXPORTER_HPP
#define VECTOR_EXPORTER_HPP

#include "FourierTransform.hpp"
#include <fstream>
#include <iostream>

void WriteToFile(const std::vector<std::complex<real>> &sequence, const std::string &filename);

#endif //VECTOR_EXPORTER_HPP