#include <fmt/format.h>
#include <gflags/gflags.h>

#include <filesystem>
#include <opencv2/opencv.hpp>

#include "database.h"
#include "utils.h"

DEFINE_string(images_dir, "", "Images directory");
DEFINE_string(images_ext, ".jpg", "Images extension");
DEFINE_string(database_path, "", "Database path");
DEFINE_string(output_dir, "", "Output directory");

void DrawKeypoints(const cv::Mat& img, const Keypoints& keypoints,
                   const std::vector<cv::Scalar>& colors,
                   const KeypointsIndices& indices,
                   const std::filesystem::path& filepath) {
    cv::Mat img_clone = img.clone();

    for (size_t i = 0; i < keypoints.size(); i++) {
        cv::Point2f kp = {keypoints[i].x(), keypoints[i].y()};
        cv::Scalar color = indices.size() == 0 ? colors[i] : colors[indices[i]];
        cv::drawMarker(img_clone, kp, color, cv::MARKER_CROSS, 10);
    }

    std::filesystem::create_directories(filepath.parent_path());
    cv::imwrite(filepath, img_clone);
}

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_images_dir.empty()) {
        return EXIT_FAILURE;
    }

    if (FLAGS_database_path.empty()) {
        return EXIT_FAILURE;
    }

    if (FLAGS_output_dir.empty()) {
        return EXIT_FAILURE;
    }

    std::filesystem::path images_dir{FLAGS_images_dir};
    CHECK(std::filesystem::is_directory(images_dir));

    std::vector<std::filesystem::path> images;
    std::ranges::for_each(std::filesystem::directory_iterator(images_dir),
                          [&](const std::filesystem::path& entry) {
                              if (std::filesystem::is_regular_file(entry) &&
                                  (entry.extension() == FLAGS_images_ext)) {
                                  images.push_back(entry);
                              }
                          });
    std::ranges::sort(images);

    if (images.empty()) {
        return EXIT_FAILURE;
    }

    std::filesystem::path output_dir{FLAGS_output_dir};
    Database database{FLAGS_database_path};

    cv::RNG rng = cv::theRNG();
    for (size_t idx = 0; idx < images.size(); idx++) {
        const int32_t id1 = static_cast<int32_t>(idx + 1);
        const Keypoints keypoints = database.ReadKeypoints(id1);

        if (keypoints.empty()) {
            continue;
        }

        std::vector<cv::Scalar> colors;
        for (size_t i = 0; i < keypoints.size(); i++) {
            colors.emplace_back(rng(256), rng(256), rng(256), 255);
        }

        const cv::Mat img = cv::imread(images[idx]);
        DrawKeypoints(img, keypoints, colors, {},
                      output_dir / fmt::format("{:06}", id1) /
                          fmt::format("{:06}.jpg", id1));

        std::vector<int32_t> image_ids =
            database.FindOpticalFlowsFromImage(id1);
        for (int32_t id2 : image_ids) {
            ImagePairFlow flow = database.ReadImagePairFlow(id1, id2);

            cv::Mat img2 = cv::imread(images[id2 - 1]);
            DrawKeypoints(img2, flow.tgt_kps, colors, flow.src_kps_indices,
                          output_dir / fmt::format("{:06}", id1) /
                              fmt::format("{:06}.jpg", id2));
        }
    }
}
