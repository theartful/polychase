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

    cv::RNG rng = cv::theRNG();
    for (size_t idx = 0; idx < images.size(); idx++) {
        const uint32_t id1 = static_cast<uint32_t>(idx + 1);
        const KeypointsMatrix keypoints = database.ReadKeypoints(id1);

        std::vector<cv::Scalar> colors;
        for (Eigen::Index i = 0; i < keypoints.rows(); i++) {
            colors.emplace_back(rng(256), rng(256), rng(256), 255);
        }

        const cv::Mat img = cv::imread(images[idx]);
        DrawKeypoints(img, keypoints, colors, {},
                      output_dir / fmt::format("{:06}", id1) / fmt::format("{:06}.jpg", id1));

        std::vector<uint32_t> image_ids = database.FindOpticalFlowsFromImage(id1);
        for (uint32_t id2 : image_ids) {
            ImagePairFlow flow = database.ReadImagePairFlow(id1, id2);

            cv::Mat img2 = cv::imread(images[id2 - 1]);
            DrawKeypoints(img2, flow.tgt_kps, colors, flow.src_kps_indices,
                          output_dir / fmt::format("{:06}", id1) / fmt::format("{:06}.jpg", id2));
        }
    }
}
