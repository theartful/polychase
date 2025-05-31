#pragma once

#include "eigen_typedefs.h"
#include "pnp/types.h"

struct PnPResult {
    CameraState camera;
    BundleStats bundle_stats;
};

void SolvePnPIterative(const ConstRefRowMajorMatrixX3f& object_points, const ConstRefRowMajorMatrixX2f& image_points,
                       const BundleOptions& bundle_opts, bool optimize_focal_length, bool optimize_principal_point,
                       PnPResult& result);

void SolvePnPIterative(const ConstRefRowMajorMatrixX3f& object_points, const ConstRefRowMajorMatrixX2f& image_points,
                       const ConstRefArrayXf& weights, const BundleOptions& bundle_opts, bool optimize_focal_length,
                       bool optimize_principal_point, PnPResult& result);
