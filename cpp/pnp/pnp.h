#pragma once

#include <Eigen/Core>
#include <optional>

#include "eigen_typedefs.h"
#include "pnp/types.h"

enum class PnPSolveMethod { Iterative, Ransac };

struct PnPResult {
    CameraState camera;
    std::optional<BundleStats> bundle_stats;
    std::optional<RansacStats> ransac_stats;
};

// Assumes OpenGL projection matrix, where we're looking at the negative Z direction.
bool SolvePnPOpenGL(const ConstRefRowMajorMatrixX3f &object_points, const ConstRefRowMajorMatrixX2f &image_points,
                    PnPSolveMethod method, PnPResult &result);
