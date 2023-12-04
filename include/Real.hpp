#ifndef REAL_HPP
#define REAL_HPP

#include <complex>
#include <vector>

// A simple way to make the code generic with respect to the real type.

namespace FourierTransform {

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

using vec = std::vector<std::complex<real>>;

}  // namespace FourierTransform

#endif  // REAL_HPP