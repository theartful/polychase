#pragma once

#include <opencv2/core/mat.hpp>

struct GFTTOptions {
    double quality_level = 0.01;
    double min_distance = 5.0;
    int block_size = 3;
    int gradient_size = 3;
    int max_corners = 0;

    // If we want to use Harris
    bool use_harris = false;
    double harris_k = 0.04;

    // Grid-based thresholding to ensure better feature distribution
    // The image is split into a grid and thresholding is applied separately for
    // each block
    int grid_rows = 4;
    int grid_cols = 4;
};

void GoodFeaturesToTrack(cv::InputArray image, cv::InputArray _mask,
                         cv::OutputArray corners,
                         cv::OutputArray corners_quality,
                         const GFTTOptions& options);
