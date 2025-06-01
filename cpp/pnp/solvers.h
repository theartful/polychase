#pragma once

#include "eigen_typedefs.h"
#include "pnp/types.h"

struct PnPResult {
    CameraState camera;
    BundleStats bundle_stats;
    Float inlier_ratio = 0.0f;
};

struct PnPOptions {
    BundleOptions bundle_opts;
    Float max_inlier_error;
    bool optimize_focal_length;
    bool optimize_principal_point;
};

void SolvePnPIterative(const RefConstRowMajorMatrixX3f& object_points, const RefConstRowMajorMatrixX2f& image_points,
                       const PnPOptions& opts, PnPResult& result);

void SolvePnPIterative(const RefConstRowMajorMatrixX3f& object_points, const RefConstRowMajorMatrixX2f& image_points,
                       const RefConstArrayXf& weights, const PnPOptions& opts, PnPResult& result);
