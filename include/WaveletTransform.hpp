#ifndef WAVELET_TRANSFORM_HPP
#define WAVELET_TRANSFORM_HPP

#include "Real.hpp"

namespace Transform {
namespace WaveletTransform {

// Forward biorthogonal 9/7 wavelet transform (lifting implementation).
// "input_sequence" is an input signal, while "output_sequnce" is the
// transformed output. Their lengths must be a power of 2 and larger than 1. The
// first half part of the output signal contains the approximation coefficients.
// The second half part contains the detail coefficients (aka. the wavelets
// coefficients). Source:
// https://web.archive.org/web/20120305164605/http://www.embl.de/~gpau/misc/dwt97.c.
void GPDirectWaveletTransform97(const std::vector<real> &input_sequence,
                                std::vector<real> &output_sequence);
// Inverse biorthogonal 9/7 wavelet transform. This is the inverse of
// DirectWaveletTransform97. "input_sequence" is an input signal, while
// "output_sequnce" is the transformed output. Their length must be a power of 2
// and larger than 1. Source:
// https://web.archive.org/web/20120305164605/http://www.embl.de/~gpau/misc/dwt97.c.
void GPInverseWaveletTransform97(const std::vector<real> &input_sequence,
                                 std::vector<real> &output_sequence);

// New implementation of the forward 9/7 DWT, following
// https://services.math.duke.edu/~ingrid/publications/J_Four_Anal_Appl_4_p247.pdf
// and https://www.sciencedirect.com/science/article/pii/S016516841100199X.
// Since the articles do not specify what to do with values at array boundaries,
// the same approach as
// https://web.archive.org/web/20120305164605/http://www.embl.de/~gpau/misc/dwt97.c
// was used. The length of "sequence" must be even and larger than 0. The
// high-passed sequence is stored in "high_sequence", while the low-passed one
// is stored in "low_sequence". This method gives a different result to the old
// algorithm.
void DaubechiesDirectWaveletTransform97(const std::vector<real> &input_sequence,
                                        std::vector<real> &output_sequence);

// Inverse of NewDirectWaveletTransform97. "high_sequence" and "low_sequence"
// are the outputs from the direct DWT and will be changed during execution.
// "sequence" contains the result of the inverse transformation.
void DaubechiesInverseWaveletTransform97(const std::vector<real> &input_sequence,
                                         std::vector<real> &output_sequence);

}  // namespace WaveletTransform
}  // namespace Transform

#endif  // WAVELET_TRANSFORM_HPP