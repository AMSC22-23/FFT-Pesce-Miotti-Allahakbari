#ifndef REAL_HPP
#define REAL_HPP

#include <complex>
#include <vector>

/** @file Real.hpp.
 *  @brief Defines a type for real numbers using a trait and aliases for complex
 * numbers and vector of complex numbers.
 */

namespace Transform {

/** @namespace */

#ifdef FLOAT
using real = float;
#else
#ifdef DOUBLE
using real = double;
#else
#ifdef LONG_DOUBLE
using real = long double;
#else
using real = double;
#endif
#endif
#endif

namespace FourierTransform {

/** @namespace */

using complex = std::complex<real>;
using vec = std::vector<complex>;

}  // namespace FourierTransform

}  // namespace Transform

#endif  // REAL_HPP