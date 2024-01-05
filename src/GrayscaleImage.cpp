#include "GrayscaleImage.hpp"

#include <opencv2/opencv.hpp>

namespace FourierTransform
{
    bool GrayscaleImage::loadStandard(const std::string &filename)
    {
        cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
        if (image.empty())
        {
            return false;
        }

        // Get the image size.
        unsigned int width = image.cols;
        unsigned int height = image.rows;

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
                this->decoded->push_back(pixel);
            }
        }
    }

    unsigned int GrayscaleImage::getStandardBitsize() const
    {
        return this->decoded->size() * 8;
    }
}