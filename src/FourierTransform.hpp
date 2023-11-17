#include <complex>
#include <vector>
#include <numbers>

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

//Perform the Fourier Transform of a sequence, using the O(n^2) algorithm
std::vector<std::complex<real>> FourierTransform(const std::vector<std::complex<real>> &sequence);
