// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

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

void RefineTrajectory(const std::string& database_path, CameraTrajectory& traj,
                      const RowMajorMatrix4f& model_matrix,
                      const AcceleratedMesh& mesh, bool optimize_focal_length,
                      bool optimize_principal_point,
                      RefineTrajectoryCallback callback,
                      BundleOptions bundle_opts);
