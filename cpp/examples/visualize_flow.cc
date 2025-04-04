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

void DrawKeypoints(const cv::Mat& img, const KeypointsMatrix& keypoints, const std::vector<cv::Scalar>& colors,
                   const KeypointsIndicesMatrix& indices, const std::filesystem::path& filepath) {
    cv::Mat img_clone = img.clone();

    for (Eigen::Index row = 0; row < keypoints.rows(); row++) {
        cv::Point2f kp = {keypoints.coeff(row, 0), keypoints.coeff(row, 1)};
        cv::Scalar color = indices.rows() == 0 ? colors[row] : colors[indices[row]];
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
    std::ranges::for_each(std::filesystem::directory_iterator(images_dir), [&](const std::filesystem::path& entry) {
        if (std::filesystem::is_regular_file(entry) && (entry.extension() == FLAGS_images_ext)) {
            images.push_back(entry);
        }
    });
    std::ranges::sort(images);

    if (images.empty()) {
        return EXIT_FAILURE;
    }

    std::filesystem::path output_dir{FLAGS_output_dir};
    Database database{FLAGS_database_path};
    for (size_t i = 0; i < images.size(); i++) {
        KeypointsMatrix keypoints = database.ReadKeypoints(i);
        std::vector<cv::Scalar> colors;
        cv::RNG rng = cv::theRNG();
        for (Eigen::Index i = 0; i < keypoints.rows(); i++) {
            colors.emplace_back(rng(256), rng(256), rng(256), 255);
        }

        cv::Mat img = cv::imread(images[i]);
        DrawKeypoints(img, keypoints, colors, {}, output_dir / fmt::format("{:06}", i) / fmt::format("{:06}.jpg", i));

        std::vector<uint32_t> image_ids = database.FindOpticalFlowsFromImage(i);
        for (uint32_t j : image_ids) {
            ImagePairFlow flow = database.ReadImagePairFlow(i, j);

            cv::Mat img2 = cv::imread(images[j]);
            DrawKeypoints(img2, flow.tgt_kps, colors, flow.src_kps_indices,
                          output_dir / fmt::format("{:06}", i) / fmt::format("{:06}.jpg", j));
        }
    }
}
