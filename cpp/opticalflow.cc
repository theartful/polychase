#include "opticalflow.h"

#include <array>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "database.h"
#include "utils.h"

struct OpticalFlowCache {
    std::vector<cv::Point2f> tracked_features;
    std::vector<uchar> status;
    std::vector<float> err;
    KeypointsIndices frame1_filtered_feats_indices;
    Keypoints frame2_filtered_feats;
    FlowErrors filtered_errors;

    void Clear() {
        tracked_features.clear();
        status.clear();
        err.clear();
        frame1_filtered_feats_indices.clear();
        frame2_filtered_feats.clear();
        filtered_errors.clear();
    }
};

constexpr std::array image_skips = std::to_array<uint32_t>({1, 2, 4, 8});

static void SaveImageForDebugging(const cv::Mat& image, uint32_t frame_id,
                                  const std::filesystem::path& dir,
                                  const std::vector<cv::Point2f>& features) {
    cv::Mat bgr;
    cv::cvtColor(image, bgr, cv::COLOR_RGB2BGR);

    cv::imwrite(dir / fmt::format("{:06}.jpg", frame_id), bgr);

    cv::RNG rng = cv::theRNG();
    for (const cv::Point2f& feat : features) {
        cv::Scalar color(rng(256), rng(256), rng(256));
        cv::drawMarker(bgr, feat, color, cv::MARKER_CROSS, 10);
    }

    cv::imwrite(dir / fmt::format("keypoints_{:06}.jpg", frame_id), bgr);
}

const std::vector<Eigen::Vector2f>& PointVectorToEigen(
    const std::vector<cv::Point2f>& points) {
    static_assert(sizeof(Eigen::Vector2f) == sizeof(cv::Point2f));
    static_assert(sizeof(std::vector<Eigen::Vector2f>) ==
                  sizeof(std::vector<cv::Point2f>));
    static_assert(std::is_trivially_destructible_v<Eigen::Vector2f>);
    static_assert(std::is_trivially_destructible_v<cv::Point2f>);

    // This is dangerous
    return *reinterpret_cast<const std::vector<Eigen::Vector2f>*>(&points);
}

std::vector<Eigen::Vector2f>& PointVectorToEigen(
    std::vector<cv::Point2f>& points) {
    static_assert(sizeof(Eigen::Vector2f) == sizeof(cv::Point2f));
    static_assert(sizeof(std::vector<Eigen::Vector2f>) ==
                  sizeof(std::vector<cv::Point2f>));
    static_assert(std::is_trivially_destructible_v<Eigen::Vector2f>);
    static_assert(std::is_trivially_destructible_v<cv::Point2f>);

    // This is dangerous
    return *reinterpret_cast<std::vector<Eigen::Vector2f>*>(&points);
}

static void GenerateOpticalFlowForAPair(cv::InputArray frame1_pyr,
                                        cv::InputArray frame2_pyr,
                                        uint32_t frame_id1, uint32_t frame_id2,
                                        cv::InputArray frame1_features,
                                        const OpticalFlowOptions& options,
                                        Database& database,
                                        OpticalFlowCache& cache) {
    cache.Clear();

    cv::calcOpticalFlowPyrLK(
        frame1_pyr, frame2_pyr, frame1_features, cache.tracked_features,
        cache.status, cache.err,
        cv::Size(options.window_size, options.window_size), options.max_level,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                         options.term_max_iters, options.term_epsilon),
        options.min_eigen_threshold);

    CHECK_EQ(cache.tracked_features.size(), cache.status.size());
    CHECK_EQ(cache.tracked_features.size(), cache.err.size());

    size_t num_valid_feats = 0;
    for (uchar s : cache.status) {
        if (s) num_valid_feats++;
    }

    cache.frame1_filtered_feats_indices.reserve(num_valid_feats);
    cache.frame2_filtered_feats.reserve(num_valid_feats);
    cache.filtered_errors.reserve(num_valid_feats);

    for (size_t i = 0; i < cache.tracked_features.size(); i++) {
        if (cache.status[i] == 1) {
            cache.frame1_filtered_feats_indices.push_back(
                static_cast<uint32_t>(i));
            cache.frame2_filtered_feats.push_back(
                {cache.tracked_features[i].x, cache.tracked_features[i].y});
            cache.filtered_errors.push_back(cache.err[i]);
        }
    }

    database.WriteImagePairFlow(
        frame_id1, frame_id2, cache.frame1_filtered_feats_indices,
        cache.frame2_filtered_feats, cache.filtered_errors);
}

static void GenerateKeypoints(const cv::Mat& frame, uint32_t frame_id,
                              Database& database,
                              const FeatureDetectorOptions& options,
                              std::vector<cv::Point2f>& features) {
    CHECK_EQ(frame.channels(), 1);

    features.clear();

    // INVESTIGATE: Can the qualities of the features help us in the final
    // solution? Should we write them to the database as well? For the sake of
    // simplicity, I will ignore that for now.
    cv::goodFeaturesToTrack(frame, features, options.max_corners,
                            options.quality_level, options.min_distance, {},
                            options.block_size, options.gradient_size,
                            options.use_harris, options.k);

    // INVESTIGATE: Should we use cv::cornerSubPix to refine features even more?

    // Write to database
    database.WriteKeypoints(frame_id, PointVectorToEigen(features));
}

static void ReadOrGenerateKeypoints(const cv::Mat& frame, uint32_t frame_id,
                                    Database& database,
                                    const FeatureDetectorOptions& options,
                                    std::vector<cv::Point2f>& features) {
    features.clear();
    database.ReadKeypoints(frame_id, PointVectorToEigen(features));

    if (features.empty()) {
        GenerateKeypoints(frame, frame_id, database, options, features);
    }
}

static void GeneratePyramid(const cv::Mat& frame,
                            const OpticalFlowOptions& options,
                            std::vector<cv::Mat>& pyramid) {
    pyramid.clear();
    cv::buildOpticalFlowPyramid(
        frame, pyramid, cv::Size(options.window_size, options.window_size),
        options.max_level);
}

static std::optional<cv::Mat> RequestFrame(
    FrameAccessorFunction& frame_accessor, const VideoInfo& video_info,
    uint32_t frame_id) {
    std::optional<cv::Mat> frame = frame_accessor(frame_id);

    if (frame) {
        CHECK_EQ(static_cast<uint32_t>(frame->rows), video_info.height);
        CHECK_EQ(static_cast<uint32_t>(frame->cols), video_info.width);
        CHECK_EQ(static_cast<uint32_t>(frame->channels()), 3);
    }

    return frame;
}

void GenerateOpticalFlowDatabase(const VideoInfo& video_info,
                                 FrameAccessorFunction frame_accessor,
                                 OpticalFlowProgressCallback callback,
                                 const std::string& database_path,
                                 const FeatureDetectorOptions& detector_options,
                                 const OpticalFlowOptions& flow_options,
                                 bool write_images) {
    CHECK(frame_accessor);

    Database database{database_path};

    const uint32_t from = video_info.first_frame;
    const uint32_t to = video_info.first_frame + video_info.num_frames;

    cv::Mat frame1_gray;
    cv::Mat frame2_gray;
    std::vector<cv::Point2f> features;
    std::vector<cv::Mat> frame1_pyramid;
    std::vector<cv::Mat> frame2_pyramid;

    const std::filesystem::path frames_dir =
        std::filesystem::path(database_path).parent_path() / "frames";
    if (write_images) {
        std::filesystem::create_directory(frames_dir);
    }

    OpticalFlowCache cache;

    for (uint32_t frame_id1 = from; frame_id1 < to; frame_id1++) {
        if (callback) {
            const double progress =
                static_cast<float>((frame_id1 - from)) / video_info.num_frames;
            const bool ok = callback(
                progress, fmt::format("Processing frame {}", frame_id1));
            if (!ok) {
                callback(1.0, "Cancelled");
                return;
            }
        }

        const std::optional<cv::Mat> maybe_frame1 =
            RequestFrame(frame_accessor, video_info, frame_id1);
        if (!maybe_frame1) {
            break;
        }
        const cv::Mat& frame1 = *maybe_frame1;

        // INVESTIGATE: Is it okay really to lose color informmation when
        // detecting and tracking features?
        // FIXME: The image can be BGR if we're using opencv to load it
        cv::cvtColor(frame1, frame1_gray, cv::COLOR_RGB2GRAY);

        ReadOrGenerateKeypoints(frame1_gray, frame_id1, database,
                                detector_options, features);
        GeneratePyramid(frame1_gray, flow_options, frame1_pyramid);

        if (write_images) {
            SaveImageForDebugging(frame1, frame_id1, frames_dir, features);
        }

        // Forward flow
        for (uint32_t skip : image_skips) {
            const uint32_t frame_id2 = frame_id1 + skip;
            if (frame_id2 >= to) {
                break;
            }

            const std::optional<cv::Mat> maybe_frame2 =
                RequestFrame(frame_accessor, video_info, frame_id2);
            if (!maybe_frame2) {
                break;
            }
            const cv::Mat& frame2 = *maybe_frame2;

            cv::cvtColor(frame2, frame2_gray, cv::COLOR_RGB2GRAY);

            if (!database.ImagePairFlowExists(frame_id1, frame_id2)) {
                GeneratePyramid(frame2_gray, flow_options, frame2_pyramid);
                GenerateOpticalFlowForAPair(frame1_pyramid, frame2_pyramid,
                                            frame_id1, frame_id2, features,
                                            flow_options, database, cache);
            }
        }

        // Backward flow
        for (uint32_t skip : image_skips) {
            if (frame_id1 < skip) break;

            const uint32_t frame_id2 = frame_id1 - skip;
            if (frame_id2 < from) {
                break;
            }

            const std::optional<cv::Mat> maybe_frame2 =
                RequestFrame(frame_accessor, video_info, frame_id2);
            if (!maybe_frame2) {
                break;
            }
            const cv::Mat& frame2 = *maybe_frame2;
            cv::cvtColor(frame2, frame2_gray, cv::COLOR_RGB2GRAY);

            if (!database.ImagePairFlowExists(frame_id1, frame_id2)) {
                GeneratePyramid(frame2_gray, flow_options, frame2_pyramid);
                GenerateOpticalFlowForAPair(frame1_pyramid, frame2_pyramid,
                                            frame_id1, frame_id2, features,
                                            flow_options, database, cache);
            }
        }
    }

    if (callback) {
        callback(1.0, "Done");
    }
}
