#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>

#include "camera_trajectory.h"
#include "database.h"
#include "geometry.h"
#include "ray_casting.h"

struct FrameTrackingResult {
    int32_t frame;
    Pose pose;
    std::optional<CameraIntrinsics> intrinsics;
    std::optional<BundleStats> bundle_stats;
};

using TrackingCallback = std::function<bool(const FrameTrackingResult&)>;

bool TrackSequence(const std::string& database_path, int32_t frame_from, int32_t frame_to_inclusive,
                   const SceneTransformations& scene_transform, const AcceleratedMeshSptr& accel_mesh,
                   TransformationType trans_type, TrackingCallback callback, bool optimize_focal_length,
                   bool optimize_principal_point, BundleOptions opts);

bool TrackCameraSequence(const Database& database, CameraTrajectory& camera_traj, int32_t frame_from,
                         int32_t frame_to_inclusive, const RowMajorMatrix4f& model_matrix,
                         const AcceleratedMeshSptr& accel_mesh, TrackingCallback callback, bool optimize_focal_length,
                         bool optimize_principal_point, const BundleOptions& opts);
