#include "opticalflow.h"

#include <tbb/tbb.h>

#include <array>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "database.h"
#include "utils.h"

struct OpticalFlowCache {
    cv::Mat frame2_gray;
    std::vector<cv::Mat> frame2_pyramid;

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

class GuardedDatabase {
   public:
    GuardedDatabase(const std::string& database_path)
        : database(database_path) {}

    class Guard {
       public:
        Guard(Database& database, std::unique_lock<std::mutex> lk)
            : database(database), lk(std::move(lk)) {}

        Database* operator->() { return &database; }
        const Database* operator->() const { return &database; }

       private:
        Database& database;
        std::unique_lock<std::mutex> lk;
    };

    Guard Lock() {
        return Guard(const_cast<Database&>(database),
                     std::unique_lock<std::mutex>{mtx});
    }
    const Guard Lock() const {
        return Guard(const_cast<Database&>(database),
                     std::unique_lock<std::mutex>{mtx});
    }

   private:
    Database database;
    mutable std::mutex mtx;
};

// TODO: Investigate
#if 0
constexpr std::array image_skips =
    std::to_array<int32_t>({-4, -3, -2, -1, 1, 2, 3, 4});
#else
constexpr std::array image_skips =
    std::to_array<int32_t>({-8, -4, -2, -1, 1, 2, 4, 8});
#endif

static void SaveImageForDebugging(const cv::Mat& image, int32_t frame_id,
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
                                        int32_t frame_id1, int32_t frame_id2,
                                        cv::InputArray frame1_features,
                                        const OpticalFlowOptions& options,
                                        GuardedDatabase& guarded_db,
                                        OpticalFlowCache& cache) {
    cache.Clear();

    cv::calcOpticalFlowPyrLK(
        frame1_pyr, frame2_pyr, frame1_features, cache.tracked_features,
        cache.status, cache.err,
        cv::Size(options.window_size, options.window_size), options.max_level,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                         options.term_max_iters, options.term_epsilon),
        0, options.min_eigen_threshold);

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

    guarded_db.Lock()->WriteImagePairFlow(
        frame_id1, frame_id2, cache.frame1_filtered_feats_indices,
        cache.frame2_filtered_feats, cache.filtered_errors);
}

static void GenerateKeypoints(const cv::Mat& frame, int32_t frame_id,
                              GuardedDatabase& guarded_db,
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
    guarded_db.Lock()->WriteKeypoints(frame_id, PointVectorToEigen(features));
}

static void ReadOrGenerateKeypoints(const cv::Mat& frame, int32_t frame_id,
                                    GuardedDatabase& guarded_db,
                                    const FeatureDetectorOptions& options,
                                    std::vector<cv::Point2f>& features) {
    features.clear();
    guarded_db.Lock()->ReadKeypoints(frame_id, PointVectorToEigen(features));

    if (features.empty()) {
        GenerateKeypoints(frame, frame_id, guarded_db, options, features);
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
    int32_t frame_id, std::mutex& mtx) {
    std::lock_guard lk{mtx};
    std::optional<cv::Mat> frame = frame_accessor(frame_id);

    if (frame) {
        CHECK_EQ(static_cast<uint32_t>(frame->rows), video_info.height);
        CHECK_EQ(static_cast<uint32_t>(frame->cols), video_info.width);
        CHECK_EQ(static_cast<uint32_t>(frame->channels()), 3);
    }

    return frame;
}

static bool FlowExists(const GuardedDatabase& guarded_db, int32_t frame_id1,
                       int32_t frame_id2) {
    return guarded_db.Lock()->ImagePairFlowExists(frame_id1, frame_id2);
}

void GenerateOpticalFlowDatabase(const VideoInfo& video_info,
                                 FrameAccessorFunction frame_accessor,
                                 OpticalFlowProgressCallback callback,
                                 const std::string& database_path,
                                 const FeatureDetectorOptions& detector_options,
                                 const OpticalFlowOptions& flow_options,
                                 bool write_images) {
    CHECK(frame_accessor);

    GuardedDatabase guarded_db{database_path};

    const int32_t from = video_info.first_frame;
    const int32_t to = video_info.first_frame + video_info.num_frames;

    cv::Mat frame1_gray;

    std::vector<cv::Point2f> features;
    std::vector<cv::Mat> frame1_pyramid;

    const std::filesystem::path frames_dir =
        std::filesystem::path(database_path).parent_path() / "frames";
    if (write_images) {
        std::filesystem::create_directory(frames_dir);
    }

    tbb::enumerable_thread_specific<OpticalFlowCache> cache_tl{};
    std::mutex accessor_mtx;

    for (int32_t frame_id1 = from; frame_id1 < to; frame_id1++) {
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
            RequestFrame(frame_accessor, video_info, frame_id1, accessor_mtx);
        if (!maybe_frame1) {
            break;
        }
        const cv::Mat& frame1 = *maybe_frame1;

        // INVESTIGATE: Is it okay really to lose color informmation when
        // detecting and tracking features?
        // FIXME: The image can be BGR if we're using opencv to load it
        cv::cvtColor(frame1, frame1_gray, cv::COLOR_RGB2GRAY);

        ReadOrGenerateKeypoints(frame1_gray, frame_id1, guarded_db,
                                detector_options, features);
        GeneratePyramid(frame1_gray, flow_options, frame1_pyramid);

        if (write_images) {
            SaveImageForDebugging(frame1, frame_id1, frames_dir, features);
        }

        // TODO: Make max_allowed_parallelism customizable
        tbb::global_control tbb_global_control(
            tbb::global_control::max_allowed_parallelism, 4);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, image_skips.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                OpticalFlowCache& cache = cache_tl.local();
                for (size_t i = range.begin(); i != range.end(); i++) {
                    int32_t skip = image_skips[i];

                    const int32_t frame_id2 = frame_id1 + skip;
                    if (frame_id2 < from || frame_id2 >= to) {
                        continue;
                    }

                    const std::optional<cv::Mat> maybe_frame2 = RequestFrame(
                        frame_accessor, video_info, frame_id2, accessor_mtx);
                    if (!maybe_frame2) {
                        break;
                    }
                    const cv::Mat& frame2 = *maybe_frame2;

                    cv::cvtColor(frame2, cache.frame2_gray, cv::COLOR_RGB2GRAY);

                    if (!FlowExists(guarded_db, frame_id1, frame_id2)) {
                        GeneratePyramid(cache.frame2_gray, flow_options,
                                        cache.frame2_pyramid);
                        GenerateOpticalFlowForAPair(
                            frame1_pyramid, cache.frame2_pyramid, frame_id1,
                            frame_id2, features, flow_options, guarded_db,
                            cache);
                    }
                }
            });
    }

    if (callback) {
        callback(1.0, "Done");
    }
}
