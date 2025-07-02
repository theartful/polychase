#pragma once

#include <spdlog/spdlog.h>
#include <tbb/concurrent_queue.h>

#include <functional>
#include <optional>
#include <variant>

#include "refiner.h"

using RefinerThreadMessage =
    std::variant<RefineTrajectoryUpdate, bool, std::unique_ptr<std::exception>>;

class RefinerThread {
   public:
    RefinerThread(std::string database_path, CameraTrajectory& traj,
                  RowMajorMatrix4f model_matrix, const AcceleratedMesh& mesh,
                  bool optimize_focal_length, bool optimize_principal_point,
                  BundleOptions bundle_opts)
        : database_path(database_path),
          traj(traj),
          model_matrix(model_matrix),
          mesh(mesh),
          optimize_focal_length(optimize_focal_length),
          optimize_principal_point(optimize_principal_point),
          bundle_opts(bundle_opts) {
        worker_thread =
            std::jthread{std::bind_front(&RefinerThread::Work, this)};
    }

    void RequestStop() { worker_thread.request_stop(); }

    void Join() {
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
    };

    std::optional<RefinerThreadMessage> TryPop() {
        RefinerThreadMessage result;
        const bool ok = results_queue.try_pop(result);
        if (ok) {
            return result;
        } else {
            return std::nullopt;
        }
    }

    bool Empty() const { return results_queue.empty(); }

    ~RefinerThread() { Join(); }

   private:
    void Work(std::stop_token stop_token) {
        auto callback = [this,
                         &stop_token](RefineTrajectoryUpdate refine_update) {
            results_queue.push(refine_update);
            return !stop_token.stop_requested();
        };

        try {
            RefineTrajectory(database_path, traj, model_matrix, mesh,
                             optimize_focal_length, optimize_principal_point,
                             callback, bundle_opts);
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
    // Refine arguments
    const std::string database_path;
    CameraTrajectory& traj;
    const RowMajorMatrix4f model_matrix;
    const AcceleratedMesh& mesh;
    const bool optimize_focal_length;
    const bool optimize_principal_point;
    const BundleOptions bundle_opts;

    tbb::concurrent_queue<RefinerThreadMessage> results_queue;

    std::jthread worker_thread;
};
