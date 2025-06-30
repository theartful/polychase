// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "camera_trajectory.h"
#include "database.h"
#include "geometry.h"
#include "ray_casting.h"

struct FrameTrackingResult {
    int32_t frame;
    Pose pose;
    CameraIntrinsics intrinsics;
    BundleStats bundle_stats;
    Float inlier_ratio;
};

using TrackingCallback = std::function<bool(const FrameTrackingResult&)>;

// TODO: Drop this function, and use CameraTrajectory similar to what we do in
// the refiner
bool TrackSequence(const std::string& database_path, int32_t frame_from,
                   int32_t frame_to_inclusive,
                   const SceneTransformations& scene_transform,
                   const AcceleratedMesh& accel_mesh, TrackingCallback callback,
                   bool optimize_focal_length, bool optimize_principal_point,
                   BundleOptions opts);

bool TrackCameraTrajectory(
    const Database& database, CameraTrajectory& camera_traj, int32_t frame_from,
    int32_t frame_to_inclusive, const RowMajorMatrix4f& model_matrix,
    const AcceleratedMesh& accel_mesh, TrackingCallback callback,
    bool optimize_focal_length, bool optimize_principal_point,
    const BundleOptions& opts);
