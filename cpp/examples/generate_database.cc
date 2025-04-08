#include <gflags/gflags.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vector>

#include "opticalflow.h"
#include "utils.h"

DEFINE_string(images_dir, "", "Images directory");
DEFINE_string(images_ext, ".jpg", "Images extension");
DEFINE_string(output, "", "Output database path");

VideoInfo GetVideoInfo(const std::vector<std::filesystem::path> images) {
    cv::Mat image = cv::imread(images[0]);
    return VideoInfo{
        .width = static_cast<uint32_t>(image.cols),
        .height = static_cast<uint32_t>(image.rows),
        .first_frame = 0,
        .num_frames = static_cast<uint32_t>(images.size()),
    };
}

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_images_dir.empty()) {
        return EXIT_FAILURE;
    }

    if (FLAGS_output.empty()) {
        return EXIT_FAILURE;
    }

    std::filesystem::path images_dir{FLAGS_images_dir};
    CHECK(std::filesystem::is_directory(images_dir));

    std::vector<std::filesystem::path> images;
    std::ranges::for_each(std::filesystem::directory_iterator(images_dir), [&](const std::filesystem::path& entry) {
        if (std::filesystem::is_regular_file(entry) && (entry.extension() == FLAGS_images_ext)) {
            images.push_back(entry);
        }
    });
    std::ranges::sort(images);

    if (images.empty()) {
        return EXIT_FAILURE;
    }

    auto get_frame = [&](uint32_t frame_id) {
        cv::Mat img = cv::imread(images[frame_id]);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        return img;
    };
    auto progress = [&](float progress, const std::string& message) {
        spdlog::info("{:3}%: {}", std::roundl(progress * 100), message);
        return true;
    };

    GenerateOpticalFlowDatabase(GetVideoInfo(images), get_frame, progress, FLAGS_output);

    return EXIT_SUCCESS;
}
