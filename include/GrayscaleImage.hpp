#ifndef GRAYSCALE_IMAGE_HPP
#define GRAYSCALE_IMAGE_HPP

#include "FourierTransform.hpp"

namespace FourierTransform
{
    class GrayscaleImage
    {
    public:
        // Load image from file.
        bool loadFile(const std::string &filename);

        // Save compressed image to file.
        bool saveFile(const std::string &filename);

        // Encode the image in variable 'uncompressed' and save the result in variable 'compressed'.
        void encode();

        // Decode the image in variable 'compressed' and save the result in variable 'uncompressed'.
        void decode();

    private:
        // Split the image in blocks of size 8x8, and save the result in variable 'blocks'.
        void splitBlocks();

        // Merge the blocks in variable 'blocks' and save the result in variable 'output'.
        void mergeBlocks();

        // Quantize the given block using a quantization table (class variable).
        void quantize(vec &block);

        // Unquantize the given block using a quantization table (class variable).
        void unQuantize();

        // Use entropy coding to encode the given block.
        void internalEncode();

        // Use entropy coding to decode the given block.
        void internalDecode();

        // The image in uncompressed form.
        vec uncompressed;

        // The image in compressed form.
        vec compressed;

        // An array of 8x8 blocks. Each block is a vector of 64 elements.
        std::vector<vec> blocks;

        // The quantization table.
        vec quantizationTable;
    };
} // namespace FourierTransform

#endif // GRAYSCALE_IMAGE_HPP