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
}