// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

#pragma once

#include <spdlog/spdlog.h>
#include <tbb/concurrent_queue.h>

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <variant>

#include "database.h"
#include "opticalflow.h"
#include "utils.h"

struct OpticalFlowProgress {
    float progress;
    std::string progress_message;
};

struct OpticalFlowRequest {
    int32_t frame_id;
};

using OpticalFlowThreadMessage =
    std::variant<OpticalFlowProgress, OpticalFlowRequest, bool,
                 std::unique_ptr<std::exception>>;

template <int CacheSize>
struct SequentialWrapper {
    SequentialWrapper(FrameAccessorFunction accessor)
        : accessor{std::move(accessor)} {}

    std::optional<cv::Mat> RequestFrame(int32_t frame_id) {
        if (invalid) {
            return std::nullopt;
        }
        std::optional<cv::Mat> maybe_frame = accessor(frame_id);
        if (!maybe_frame) {
            invalid = true;
        }
        return maybe_frame;
    }

    std::optional<cv::Mat> operator()(int32_t frame_id) {
        const size_t frame_idx = frame_id % CacheSize;

        if (highest_frame_id == kInvalidId) {
            highest_frame_id = frame_id;
            frames[frame_idx] = accessor(frame_id);
            return frames[frame_idx];
        }

        if (frame_id <= highest_frame_id) {
            CHECK(highest_frame_id - frame_id < CacheSize);
            return frames[frame_idx];
        }

        CHECK_LT(frame_id - highest_frame_id, CacheSize);

        for (int32_t id = highest_frame_id + 1; id <= frame_id; id++) {
            const size_t idx = id % CacheSize;
            frames[idx] = RequestFrame(id);
        }

        highest_frame_id = frame_id;
        return frames[frame_idx];
    }

    FrameAccessorFunction accessor;
    int32_t highest_frame_id = kInvalidId;
    bool invalid = false;
    std::optional<cv::Mat> frames[CacheSize];
};

class OpticalFlowThread {
   public:
    OpticalFlowThread(VideoInfo video_info, std::string database_path,
                      GFTTOptions detector_options = {},
                      OpticalFlowOptions flow_options = {},
                      bool write_images = false)
        : video_info(video_info),
          database_path(std::move(database_path)),
          detector_options(detector_options),
          flow_options(flow_options),
          write_images(write_images) {
        worker_thread =
            std::jthread{std::bind_front(&OpticalFlowThread::Work, this)};
    }

    void RequestStop() {
        {
            std::lock_guard<std::mutex> lk(provided_frame_mtx);
            worker_thread.request_stop();
        }
        provided_frame_cv.notify_all();
    }

    void Join() {
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
    }

    std::optional<OpticalFlowThreadMessage> TryPop() {
        OpticalFlowThreadMessage result;
        const bool ok = results_queue.try_pop(result);
        if (ok) {
            return result;
        } else {
            return std::nullopt;
        }
    }

    void ProvideFrame(int32_t frame_id, cv::Mat frame) {
        // frame is provided from python, and on destruction, the gil has to be
        // held, which causes a deadlock, since it can die on a separate cpp
        // thread while the python interpreter is waiting on it to die by
        // calling join
        cv::Mat frame_copy;
        frame.copyTo(frame_copy);
        {
            std::lock_guard<std::mutex> lk(provided_frame_mtx);
            provided_frame = {frame_id, frame_copy};
        }
        provided_frame_cv.notify_all();
    }

    bool Empty() const { return results_queue.empty(); }

    ~OpticalFlowThread() { Join(); }

   private:
    void Work(std::stop_token stop_token) {
        auto frame_accessor =
            [this, &stop_token](int32_t frame_id) -> std::optional<cv::Mat> {
            results_queue.push(OpticalFlowRequest{.frame_id = frame_id});

            std::unique_lock<std::mutex> lk(provided_frame_mtx);
            provided_frame_cv.wait_for(lk, std::chrono::seconds(10), [&]() {
                return provided_frame.has_value() ||
                       stop_token.stop_requested();
            });

            if (stop_token.stop_requested()) {
                return std::nullopt;
            }

            if (provided_frame->first != frame_id) {
                throw std::runtime_error(
                    fmt::format("Requested frame {} but got {}", frame_id,
                                provided_frame->first));
            }

            cv::Mat frame = std::move(provided_frame->second);
            provided_frame = {};

            return frame;
        };

        auto progress_callback = [this, &stop_token](float progress,
                                                     const std::string& msg) {
            results_queue.push(OpticalFlowProgress{.progress = progress,
                                                   .progress_message = msg});
            return !stop_token.stop_requested();
        };

        try {
            GenerateOpticalFlowDatabase(
                video_info, SequentialWrapper<17>{frame_accessor},
                progress_callback, database_path, detector_options,
                flow_options, write_images);
        } catch (const std::exception& exception) {
            SPDLOG_ERROR("Error: {}", exception.what());
            results_queue.push(std::make_unique<std::exception>(exception));
        } catch (...) {
            SPDLOG_ERROR("Unknown exception type thrown");
            results_queue.push(std::make_unique<std::runtime_error>(
                "Unknown exception type. This should never happen!"));
        }
        results_queue.push(true);
    }

   private:
    // Opticalflow arguments
    const VideoInfo video_info;
    const std::string database_path;
    const GFTTOptions detector_options;
    const OpticalFlowOptions flow_options;
    const bool write_images = false;

    // Results queue
    tbb::concurrent_queue<OpticalFlowThreadMessage> results_queue;

    std::optional<std::pair<int32_t, cv::Mat>> provided_frame;
    std::mutex provided_frame_mtx;
    std::condition_variable provided_frame_cv;

    std::jthread worker_thread;
};
