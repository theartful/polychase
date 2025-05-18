#include "refiner.h"

#include <chrono>
#include <thread>

#include "database.h"

static bool RefineTrajectory(const Database& database, CameraTrajectory& traj, const RowMajorMatrix4f& model_matrix,
                             AcceleratedMeshSptr mesh, bool optimize_focal_length, bool optimize_principal_point,
                             RefineTrajectoryCallback callback) {
    RefineTrajectoryUpdate update;
    for (int i = 0; i < 1000; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        update.message = "Working on it!";
        update.progress = (i + 1) / 1000.0f;

        if (!callback(update)) {
            break;
        }
    }

    return true;
}

bool RefineTrajectory(const std::string& database_path, CameraTrajectory& traj, const RowMajorMatrix4f& model_matrix,
                      AcceleratedMeshSptr mesh, bool optimize_focal_length, bool optimize_principal_point,
                      RefineTrajectoryCallback callback) {
    Database database{database_path};
    return RefineTrajectory(database, traj, model_matrix, mesh, optimize_focal_length, optimize_principal_point,
                            callback);
}
