#pragma once

#include <cstdint>
#include <functional>

#include "eigen_typedefs.h"
#include "geometry.h"
#include "ray_casting.h"

using TrackingCallback = std::function<bool(uint32_t frame_id, RowMajorMatrix4f pose)>;

bool TrackForwards(const std::string& database_path, uint32_t frame_from, size_t num_frames,
                   const SceneTransformations& scene_transform, const AcceleratedMeshSptr& accel_mesh,
                   TransformationType trans_type, TrackingCallback callback);

bool TrackBackwards(const std::string& database_path, uint32_t frame_from, size_t num_frames,
                    const SceneTransformations& scene_transform, const AcceleratedMeshSptr& accel_mesh,
                    TransformationType trans_type, TrackingCallback callback);
