// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

#pragma once

#include <spdlog/spdlog.h>
#include <tbb/concurrent_queue.h>

#include <functional>
#include <optional>
#include <thread>
#include <variant>

#include "tracker.h"

using TrackerThreadMessage =
    std::variant<FrameTrackingResult, bool, std::unique_ptr<std::exception>>;

class TrackerThread {
   public:
    TrackerThread(std::string database_path, int32_t frame_from,
                  int32_t frame_to_inclusive,
                  SceneTransformations scene_transform,
                  std::shared_ptr<const AcceleratedMesh> accel_mesh,
                  bool optimize_focal_length, bool optimize_principal_point,
                  BundleOptions bundle_opts)
        : database_path(std::move(database_path)),
          frame_from(frame_from),
          frame_to_inclusive(frame_to_inclusive),
          scene_transform(scene_transform),
          accel_mesh(std::move(accel_mesh)),
          optimize_focal_length(optimize_focal_length),
          optimize_principal_point(optimize_principal_point),
          bundle_opts(bundle_opts) {
        worker_thread =
            std::jthread{std::bind_front(&TrackerThread::Work, this)};
    }

    void RequestStop() { worker_thread.request_stop(); }

    void Join() {
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
    }

    std::optional<TrackerThreadMessage> TryPop() {
        TrackerThreadMessage result;
        const bool ok = results_queue.try_pop(result);
        if (ok) {
            return result;
        } else {
            return std::nullopt;
        }
    }

    bool Empty() const { return results_queue.empty(); }

    ~TrackerThread() { Join(); }

   private:
    void Work(std::stop_token stop_token) {
        auto callback =
            [this, &stop_token](const FrameTrackingResult& tracking_result) {
                results_queue.push(tracking_result);
                return !stop_token.stop_requested();
            };

        try {
            TrackSequence(database_path, frame_from, frame_to_inclusive,
                          scene_transform, *accel_mesh, callback,
                          optimize_focal_length, optimize_principal_point,
                          bundle_opts);
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
    // Tracking arguments
    const std::string database_path;
    const int32_t frame_from;
    const int32_t frame_to_inclusive;
    const SceneTransformations scene_transform;
    const std::shared_ptr<const AcceleratedMesh> accel_mesh;
    const bool optimize_focal_length;
    const bool optimize_principal_point;
    const BundleOptions bundle_opts;

    // Results queue
    tbb::concurrent_queue<TrackerThreadMessage> results_queue;

    std::jthread worker_thread;
};
