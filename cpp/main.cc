#include <fmt/format.h>

#include <Eigen/Core>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "database.h"
#include "opticalflow.h"

cv::Mat GetFrame(uint32_t frame_id) {
    std::string path = fmt::format("/home/theartful/Work/exp/frames/{:09}.jpg", frame_id);
    return cv::imread(path);
}

int main() {
    GenerateOpticalFlowDatabase(
        VideoInfo{
            .width = 1920,
            .height = 1080,
            .first_frame = 1,
            .num_frames = 10,
        },
        GetFrame, nullptr, "/home/theartful/test.db");
}
