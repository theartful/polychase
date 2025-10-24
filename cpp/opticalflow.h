// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

#pragma once

#include <cstdint>
#include <functional>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>

#include "feature_detection/gftt.h"

using FrameAccessorFunction =
    std::function<std::optional<cv::Mat>(int32_t frame_id)>;

using OpticalFlowProgressCallback =
    std::function<bool(float progress, const std::string& progress_message)>;

struct VideoInfo {
    uint32_t width;
    uint32_t height;
    int32_t first_frame;
    uint32_t num_frames;
};

struct OpticalFlowOptions {
    int window_size = 10;
    int max_level = 3;
    int term_max_iters = 30;
    double term_epsilon = 0.01;
    double min_eigen_threshold = 1e-4;
};

void GenerateOpticalFlowDatabase(const VideoInfo& video_info,
                                 FrameAccessorFunction frame_accessor,
                                 OpticalFlowProgressCallback callback,
                                 const std::string& database_path,
                                 const GFTTOptions& detector_options = {},
                                 const OpticalFlowOptions& flow_options = {},
                                 bool write_images = false);
