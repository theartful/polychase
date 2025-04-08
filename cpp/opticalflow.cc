#include "opticalflow.h"

#include <array>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "database.h"
#include "utils.h"

#define SAVE_FRAMES_FOR_DEBUGGING
#define FRAMES_DIR "./frames/"

#undef SAVE_FRAMES_FOR_DEBUGGING

#ifdef SAVE_FRAMES_FOR_DEBUGGING
#include <filesystem>

static void SaveImageForDebugging(const cv::Mat& image, uint32_t frame_id) {
    const char* dir = FRAMES_DIR;
    std::filesystem::create_directories(dir);

    cv::Mat bgr;
    cv::cvtColor(image, bgr, cv::COLOR_RGB2BGR);
    cv::imwrite(std::filesystem::path(dir) / fmt::format("{:06}.jpg", frame_id), bgr);
}
#endif

struct Cache {
    std::vector<cv::Point2f> tracked_features;
    std::vector<uchar> status;
    std::vector<float> err;
    std::vector<uint32_t> frame1_filtered_feats_indices;
    std::vector<cv::Point2f> frame2_filtered_feats;
    std::vector<float> filtered_errors;

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

Eigen::Map<const KeypointsMatrix> PointVectorToEigen(const std::vector<cv::Point2f>& points) {
    static_assert(sizeof(cv::Point2f) == sizeof(float) * 2);
    static_assert(KeypointsMatrix::ColsAtCompileTime == 2);
    static_assert(KeypointsMatrix::IsRowMajor);

    return Eigen::Map<const KeypointsMatrix>(reinterpret_cast<const float*>(points.data()), points.size(), 2);
}

static void GenerateOpticalFlowForAPair(cv::InputArray frame1_pyr, cv::InputArray frame2_pyr, uint32_t frame_id1,
                                        uint32_t frame_id2, cv::InputArray frame1_features,
                                        const OpticalFlowOptions& options, Database& database, Cache& cache) {
    cache.Clear();

    cv::calcOpticalFlowPyrLK(frame1_pyr, frame2_pyr, frame1_features, cache.tracked_features, cache.status, cache.err,
                             cv::Size(options.window_size, options.window_size), options.max_level,
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, options.term_max_iters,
                                              options.term_epsilon),
                             options.min_eigen_threshold);

    CHECK_EQ(cache.tracked_features.size(), cache.status.size());
    CHECK_EQ(cache.tracked_features.size(), cache.err.size());

    size_t num_valid_feats = 0;
    for (uchar s : cache.status) {
        if (s) num_valid_feats++;
    }

    // TODO: Reuse these vectors
    std::vector<uint32_t> frame1_filtered_feats_indices;
    std::vector<cv::Point2f> frame2_filtered_feats;
    std::vector<float> filtered_errors;

    frame1_filtered_feats_indices.reserve(num_valid_feats);
    frame2_filtered_feats.reserve(num_valid_feats);
    filtered_errors.reserve(num_valid_feats);

    for (size_t i = 0; i < cache.tracked_features.size(); i++) {
        if (cache.status[i] == 1) {
            frame1_filtered_feats_indices.push_back(static_cast<uint32_t>(i));
            frame2_filtered_feats.push_back(cache.tracked_features[i]);
            filtered_errors.push_back(filtered_errors[i]);
        }
    }

    static_assert(KeypointsIndicesMatrix::ColsAtCompileTime == 1);
    static_assert(KeypointsMatrix::ColsAtCompileTime == 2);
    static_assert(sizeof(KeypointsIndicesMatrix::Scalar) == sizeof(uint32_t));

    database.WriteImagePairFlow(
        frame_id1, frame_id2,
        Eigen::Map<const KeypointsIndicesMatrix>{frame1_filtered_feats_indices.data(),
                                                 static_cast<Eigen::Index>(frame1_filtered_feats_indices.size()), 1},
        Eigen::Map<const KeypointsMatrix>{reinterpret_cast<float*>(frame2_filtered_feats.data()),
                                          static_cast<Eigen::Index>(frame2_filtered_feats.size()), 2},
        Eigen::Map<const FlowErrorsMatrix>{filtered_errors.data(), static_cast<Eigen::Index>(filtered_errors.size()),
                                           1});
}

static void GenerateKeypoints(const cv::Mat& frame, uint32_t frame_id, Database& database,
                              const FeatureDetectorOptions& options, std::vector<cv::Point2f>& features) {
    CHECK_EQ(frame.channels(), 1);

    features.clear();

    // INVESTIGATE: Can the qualities of the features help us in the final solution?
    // Should we write them to the database as well?
    // For the sake of simplicity, I will ignore that for now.
    cv::goodFeaturesToTrack(frame, features, options.max_corners, options.quality_level, options.min_distance, {},
                            options.block_size, options.gradient_size, options.use_harris, options.k);

    // INVESTIGATE: Should we use cv::cornerSubPix to refine features even more?

    // Write to database
    database.WriteKeypoints(frame_id, PointVectorToEigen(features));
}

static void ReadKeypoints(uint32_t frame_id, Database& database, std::vector<cv::Point2f>& features) {
    features.clear();

    KeypointsMatrix keypoints = database.ReadKeypoints(frame_id);
    for (Eigen::Index row = 0; row < keypoints.rows(); row++) {
        Eigen::Vector2f keypoint = keypoints.row(row);
        features.push_back(cv::Point2f(keypoint.x(), keypoint.y()));
    }
}

static void ReadOrGenerateKeypoints(const cv::Mat& frame, uint32_t frame_id, Database& database,
                                    const FeatureDetectorOptions& options, std::vector<cv::Point2f>& features) {
    features.clear();

    ReadKeypoints(frame_id, database, features);
    if (features.empty()) {
        GenerateKeypoints(frame, frame_id, database, options, features);
    }
}

static void GeneratePyramid(const cv::Mat& frame, const OpticalFlowOptions& options, std::vector<cv::Mat>& pyramid) {
    pyramid.clear();
    cv::buildOpticalFlowPyramid(frame, pyramid, cv::Size(options.window_size, options.window_size), options.max_level);
}

static cv::Mat RequestFrame(FrameAccessorFunction& frame_accessor, const VideoInfo& video_info, uint32_t frame_id) {
    cv::Mat frame = frame_accessor(frame_id);

    CHECK_EQ(static_cast<uint32_t>(frame.rows), video_info.height);
    CHECK_EQ(static_cast<uint32_t>(frame.cols), video_info.width);
    CHECK_EQ(static_cast<uint32_t>(frame.channels()), 3);

    return frame;
}

void GenerateOpticalFlowDatabase(const VideoInfo& video_info, FrameAccessorFunction frame_accessor,
                                 ProgressCallback callback, const std::string& database_path,
                                 const FeatureDetectorOptions& detector_options,
                                 const OpticalFlowOptions& flow_options) {
    CHECK(frame_accessor);

    Database database{database_path};

    const uint32_t from = video_info.first_frame;
    const uint32_t to = video_info.first_frame + video_info.num_frames;

    cv::Mat frame1_gray;
    cv::Mat frame2_gray;
    std::vector<cv::Point2f> features;
    std::vector<cv::Mat> frame1_pyramid;
    std::vector<cv::Mat> frame2_pyramid;

    Cache cache;

    // Forward flow
    for (uint32_t frame_id1 = from; frame_id1 < to; frame_id1++) {
        if (callback) {
            const double progress = static_cast<float>((frame_id1 - from)) / video_info.num_frames;
            const bool ok = callback(progress, fmt::format("Processing frame {}", frame_id1));
            if (!ok) {
                callback(1.0, "Cancelled");
                return;
            }
        }

        cv::Mat frame1 = RequestFrame(frame_accessor, video_info, frame_id1);

#ifdef SAVE_FRAMES_FOR_DEBUGGING
        SaveImageForDebugging(frame1, frame_id1);
#endif

        // INVESTIGATE: Is it okay really to lose color informmation when detecting and tracking features?
        // FIXME: The image can be BGR if we're using opencv to load it
        cv::cvtColor(frame1, frame1_gray, cv::COLOR_RGB2GRAY);

        ReadOrGenerateKeypoints(frame1_gray, frame_id1, database, detector_options, features);
        GeneratePyramid(frame1_gray, flow_options, frame1_pyramid);

        // Forward flow
        for (uint32_t skip : image_skips) {
            const uint32_t frame_id2 = frame_id1 + skip;
            if (frame_id2 >= to) {
                break;
            }

            cv::Mat frame2 = RequestFrame(frame_accessor, video_info, frame_id2);
            cv::cvtColor(frame2, frame2_gray, cv::COLOR_RGB2GRAY);

            if (!database.ImagePairFlowExists(frame_id1, frame_id2)) {
                GeneratePyramid(frame2_gray, flow_options, frame2_pyramid);
                GenerateOpticalFlowForAPair(frame1_pyramid, frame2_pyramid, frame_id1, frame_id2, features,
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

            cv::Mat frame2 = RequestFrame(frame_accessor, video_info, frame_id2);
            cv::cvtColor(frame2, frame2_gray, cv::COLOR_RGB2GRAY);

            if (!database.ImagePairFlowExists(frame_id1, frame_id2)) {
                GeneratePyramid(frame2_gray, flow_options, frame2_pyramid);
                GenerateOpticalFlowForAPair(frame1_pyramid, frame2_pyramid, frame_id1, frame_id2, features,
                                            flow_options, database, cache);
            }
        }
    }

    if (callback) {
        callback(1.0, "Done");
    }
}
