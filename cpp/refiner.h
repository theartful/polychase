#pragma once

#include <functional>
#include <string>

#include "camera_trajectory.h"
#include "eigen_typedefs.h"
#include "ray_casting.h"

struct RefineTrajectoryUpdate {
    float progress;
    std::string message;

    BundleStats stats;
};

using RefineTrajectoryCallback = std::function<bool(RefineTrajectoryUpdate)>;

bool RefineTrajectory(const std::string& database_path, CameraTrajectory& traj,
                      const RowMajorMatrix4f& model_matrix,
                      AcceleratedMeshSptr mesh, bool optimize_focal_length,
                      bool optimize_principal_point,
                      RefineTrajectoryCallback callback,
                      BundleOptions bundle_opts);
