#pragma once

#include <Eigen/Core>

#include "geometry.h"

struct PinUpdate {
    uint32_t pin_idx;
    Eigen::Vector2f pos;
};

SceneTransformations FindTransformation(
    const ConstRefRowMajorMatrixX3f& object_points,  // Using Ref so that pybind doesn't copy numpy data
    const SceneTransformations& initial_scene_transform, const SceneTransformations& current_scene_transform,
    const PinUpdate& update, TransformationType trans_type);
