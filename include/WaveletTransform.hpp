#ifndef WAVELET_TRANSFORM_HPP
#define WAVELET_TRANSFORM_HPP

#include "Real.hpp"

namespace Transform {
namespace WaveletTransform {

// Forward biorthogonal 9/7 wavelet transform (lifting implementation).
// "sequence" is an input signal, which will be replaced by its output
// transform. Its length must be a power of 2 and larger than 1. The first half
// part of the output signal contains the approximation coefficients. The second
// half part contains the detail coefficients (aka. the wavelets coefficients).
// Source:
// https://web.archive.org/web/20120305164605/http://www.embl.de/~gpau/misc/dwt97.c.
void DirectWaveletTransform97(std::vector<real> &sequence);
// Inverse biorthogonal 9/7 wavelet transform. This is the inverse of
// DirectWaveletTransform97. "sequence" is an input signal, which will be
// replaced by its output transform. Its length must be a power of 2 and larger
// than 1. Source:
// https://web.archive.org/web/20120305164605/http://www.embl.de/~gpau/misc/dwt97.c.
void InverseWaveletTransform97(std::vector<real> &sequence);

}  // namespace WaveletTransform
}  // namespace Transform

#endif  // WAVELET_TRANSFORM_HPP