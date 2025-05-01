#pragma once

#include <cstdint>
#include <functional>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>

using FrameAccessorFunction = std::function<std::optional<cv::Mat>(uint32_t frame_id)>;

using ProgressCallback = std::function<bool(float progress, const std::string& progress_message)>;

struct VideoInfo {
    uint32_t width;
    uint32_t height;
    uint32_t first_frame;
    uint32_t num_frames;
};

struct FeatureDetectorOptions {
    double quality_level = 0.01;
    double min_distance = 2.0;
    int block_size = 7;
    int gradient_size = 7;
    int max_corners = 0;

    // If we want to use Harris
    bool use_harris = false;
    double k = 0.04;
};

struct OpticalFlowOptions {
    double window_size = 10;
    int max_level = 3;
    double term_epsilon = 0.01;
    int term_max_iters = 30;
    double min_eigen_threshold = 1e-4;
};

void GenerateOpticalFlowDatabase(const VideoInfo& video_info, FrameAccessorFunction frame_accessor,
                                 ProgressCallback callback, const std::string& database_path,
                                 const FeatureDetectorOptions& detector_options = {},
                                 const OpticalFlowOptions& flow_options = {}, bool write_images = false);
