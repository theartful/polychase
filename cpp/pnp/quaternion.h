// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "eigen_typedefs.h"

inline Eigen::Quaternionf QuatStepPost(const Eigen::Quaternionf &q,
                                       const Eigen::Vector3f &w_delta) {
    const Float angle = w_delta.norm();
    if (angle > 0) {
        const Eigen::Vector3f axis = w_delta.block<3, 1>(0, 0) / angle;
        return q * Eigen::AngleAxis(angle, axis);
    } else {
        return q;
    }
}
