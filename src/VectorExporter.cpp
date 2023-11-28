#include "VectorExporter.hpp"

#include <fstream>
#include <iostream>
#include <iomanip>

// Utility function to write a sequence of complex numbers to a file.
void WriteToFile(const std::vector<std::complex<real>> &sequence, const std::string &filename)
{
    // Open the file.
    std::ofstream file(filename);

    // Write the sequence to the file in .csv format, with full precision.
    for (size_t i = 0; i < sequence.size(); i++)
    {
        file << std::setprecision(std::numeric_limits<real>::max_digits10);
        file << sequence[i].real() << "," << sequence[i].imag() << std::endl;
    }

    file.close();

    // Notify the user.
    std::cout << "Written data to [" << filename << "]." << std::endl;
}