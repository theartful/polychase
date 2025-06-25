// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

#pragma once

#include <Eigen/Core>

#include "geometry.h"

struct PinUpdate {
    uint32_t pin_idx;
    Eigen::Vector2f pos;
};

SceneTransformations FindTransformation(
    // Using Ref so that pybind doesn't copy numpy data
    const RefConstRowMajorMatrixX3f& object_points,
    const SceneTransformations& initial_scene_transform,
    const SceneTransformations& current_scene_transform,
    const PinUpdate& update, TransformationType trans_type,
    bool optimize_focal_length, bool optimize_principal_point);
