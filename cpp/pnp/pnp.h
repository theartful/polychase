#pragma once

#include <Eigen/Core>

#include "eigen_typedefs.h"

enum class PnPSolveMethod { Iterative, Ransac };

// Only supporting SIMPLE_PINHOLE camera model for now (in colmap terms).
struct CameraIntrinsics {
    float fx;
    float fy;
    float cx;
    float cy;
};

struct PnPResult {
    Eigen::Vector3f translation;
    RowMajorMatrix3f rotation;
};

// Assumes OpenGL projection matrix, where we're looking at the negative Z direction.
bool SolvePnP(const ConstRefRowMajorMatrixX3f& object_points, const ConstRefRowMajorMatrixX2f& image_points,
              const RowMajorMatrix4f& projection_matrix, PnPSolveMethod method, PnPResult& result);
