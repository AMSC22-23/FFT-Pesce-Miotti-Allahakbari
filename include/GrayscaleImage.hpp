#ifndef GRAYSCALE_IMAGE_HPP
#define GRAYSCALE_IMAGE_HPP

#include "FourierTransform.hpp"

namespace FourierTransform
{
    // A class that represents a grayscale image.
    // This class may be used to load, save, encode, decode and display grayscale images.
    // Please note that the image's width and height must be a multiple of 8.
    class GrayscaleImage
    {
    public:
        // Load regular image from file.
        bool loadStandard(const std::string &filename);

        // Load compressed image from file.
        bool loadCompressed(const std::string &filename);

        // Save compressed image to file.
        bool save(const std::string &filename);

        // Encode the last loaded or decoded image.
        void encode();

        // Decode the last loaded or encoded image.
        void decode();

        // Display the last loaded or decoded image.
        void display();

        // Get the bitsize of the last loaded or decoded image.
        unsigned int getStandardBitsize() const;

        // Get the bitsize of the last loaded or encoded image.
        unsigned int getCompressedBitsize() const;

    private:
        // Split the image in blocks of size 8x8, and save the result in variable 'blocks'.
        void splitBlocks();

        // Merge the blocks in variable 'blocks' and save the result in variable 'decoded'.
        void mergeBlocks();

        // Quantize the given block using the quantization table.
        vec quantize(const vec &block);

        // Unquantize the given block using the quantization table.
        vec unquantize(const vec &block);

        // Use entropy coding to encode the given block.
        void entropyEncode();

        // Use entropy coding to decode the given block.
        void entropyDecode();

        // Static member variable to store the quantization table.
        static vec quantizationTable;

        // The image in uncompressed form.
        std::vector<unsigned char> decoded;

        // The image in compressed form (expressed as a sequence of bytes).
        std::vector<unsigned char> encoded;

        // An array of 8x8 blocks. Each block is a vector of 64 elements.
        std::vector<vec> blocks;

        // Block grid width.
        int blockGridWidth;

        // Block grid height.
        int blockGridHeight;
    };
} // namespace FourierTransform

#endif // GRAYSCALE_IMAGE_HPP