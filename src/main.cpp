#include <omp.h>
#include <tgmath.h>

#include <iostream>
#include <numbers>
#include <string>

#include "FourierTransformCalculator.hpp"
#include "Utility.hpp"
#include "VectorExporter.hpp"

int main()
{
  using namespace FourierTransform;

  // Set a matrix size.
  size_t size = 8;

  // Create a vector that will represent the matrix.
  vec sequence(size * size, 0);

  // For each row.
  for (size_t i = 0; i < size; i++)
  {
    // For each column.
    for (size_t j = 0; j < size; j++)
    {
      // Generate a random real number between -10 and 10.
      const double random_number = (rand() % 2000 - 1000) / 100.0;

      // Set the value of the element.
      sequence[i * size + j] = random_number;
    }
  }

  // Create a vector that will represent the output matrix.
  vec output_sequence(size * size, 0);

  // Initialize a TwoDimensionalFourierTransformAlgorithm object.
  std::unique_ptr<FourierTransformAlgorithm> ft_algorithm =
      std::make_unique<TrivialTwoDimensionalFourierTransformAlgorithm>();

  // Initialize a FourierTransformCalculator object.
  FourierTransformCalculator ft_calculator;

  // Set the algorithm for the direct transform.
  ft_calculator.setDirectAlgorithm(ft_algorithm);

  // Perform the direct transform.
  ft_calculator.directTransform(sequence, output_sequence);

  // Print the input matrix.
  std::cout << "Input matrix:" << std::endl;
  for (size_t i = 0; i < size; i++)
  {
    std::cout << "[ ";
    for (size_t j = 0; j < size; j++)
      std::cout << sequence[i * size + j] << " ";
    std::cout << "]" << std::endl;
  }

  // Print the output matrix.
  std::cout << "Output matrix:" << std::endl;
  for (size_t i = 0; i < size; i++)
  {
    std::cout << "[ ";
    for (size_t j = 0; j < size; j++)
      std::cout << output_sequence[i * size + j] << " ";
    std::cout << "]" << std::endl;
  }

  return 0;
}