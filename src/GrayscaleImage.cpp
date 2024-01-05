#include "GrayscaleImage.hpp"

#include <opencv2/opencv.hpp>

namespace FourierTransform
{
    // Load regular image from file.
    bool GrayscaleImage::loadStandard(const std::string &filename)
    {
        cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
        if (image.empty())
        {
            return false;
        }

        // Get the image size.
        int width = image.cols;
        int height = image.rows;

        // Assert that the image size is a multiple of 8.
        assert(width % 8 == 0);
        assert(height % 8 == 0);

        // Memorize the block grid size.
        this->blockGridWidth = width / 8;
        this->blockGridHeight = height / 8;

        // For each row...
        for (int i = 0; i < image.rows; i++)
        {
            // For each column...
            for (int j = 0; j < image.cols; j++)
            {
                // Get the pixel value.
                unsigned char pixel = image.at<unsigned char>(i, j);

                // Add the pixel value to the decoded image.
                this->decoded.push_back(pixel);
            }
        }

        return true;
    }

    // Get the bitsize of the last loaded or decoded image.
    unsigned int GrayscaleImage::getStandardBitsize() const
    {
        return this->blockGridWidth * this->blockGridHeight * 64 * 8;
    }

    // Display the last loaded or decoded image.
    void GrayscaleImage::display()
    {
        // Create a new image.
        cv::Mat image(this->blockGridHeight * 8, this->blockGridWidth * 8, CV_8UC1);

        // For each row...
        for (int i = 0; i < image.rows; i++)
        {
            // For each column...
            for (int j = 0; j < image.cols; j++)
            {
                // Get the pixel value.
                unsigned char pixel = this->decoded[i * image.cols + j];

                // Set the pixel value.
                image.at<unsigned char>(i, j) = pixel;
            }
        }

        // Display the image.
        cv::imshow("Image", image);
        cv::waitKey(0);
    }

    // Split the image in blocks of size 8x8, and save the result in variable 'blocks'.
    void GrayscaleImage::splitBlocks()
    {
        // Clear the blocks vector.
        this->blocks.clear();

        // For each block row...
        for (int i = 0; i < this->blockGridHeight; i++)
        {
            // For each block column...
            for (int j = 0; j < this->blockGridWidth; j++)
            {
                // Create a new block.
                std::vector<unsigned char> block;

                // For each row in the block...
                for (int k = 0; k < 8; k++)
                {
                    // For each column in the block...
                    for (int l = 0; l < 8; l++)
                    {
                        // Get the top-left pixel coordinates of the block.
                        int x = j * 8;
                        int y = i * 8;

                        // Get the pixel coordinates.
                        int pixelX = x + l;
                        int pixelY = y + k;

                        // Get the pixel value.
                        unsigned char pixel = this->decoded[pixelY * this->blockGridWidth * 8 + pixelX];

                        // Add the pixel value to the block.
                        block.push_back(pixel);
                    }
                }

                // Add the block to the blocks vector.
                this->blocks.push_back(block);
            }
        }
    }
}