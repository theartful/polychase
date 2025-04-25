#pragma once

#include <cstdint>
#include <functional>

#include "eigen_typedefs.h"
#include "geometry.h"
#include "ray_casting.h"

// FIXME: Maybe use Sophus or define a proper SE3 class.
using Pose = CameraPose;

struct FrameTrackingResult {
    int32_t frame;
    Pose pose;
    std::optional<CameraIntrinsics> intrinsics;
    std::optional<RansacStats> ransac_stats;
    std::optional<BundleStats> bundle_stats;
};

using TrackingCallback = std::function<bool(FrameTrackingResult)>;

bool TrackForwards(const std::string& database_path, int32_t frame_from, size_t num_frames,
                   const SceneTransformations& scene_transform, const AcceleratedMeshSptr& accel_mesh,
                   TransformationType trans_type, TrackingCallback callback);

bool TrackBackwards(const std::string& database_path, int32_t frame_from, size_t num_frames,
                    const SceneTransformations& scene_transform, const AcceleratedMeshSptr& accel_mesh,
                    TransformationType trans_type, TrackingCallback callback);
