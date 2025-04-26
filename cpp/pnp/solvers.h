#pragma once

#include <optional>

#include "eigen_typedefs.h"
#include "pnp/types.h"

struct PnPResult {
    CameraState camera;
    std::optional<BundleStats> bundle_stats;
    std::optional<RansacStats> ransac_stats;
};

void SolvePnPIterative(const ConstRefRowMajorMatrixX3f& object_points, const ConstRefRowMajorMatrixX2f& image_points,
                       const BundleOptions& bundle_opts, bool optimize_focal_length, bool optimize_principal_point,
                       PnPResult& result);

void SolvePnPRansac(const ConstRefRowMajorMatrixX3f& object_points, const ConstRefRowMajorMatrixX2f& image_points,
                    const RansacOptions& ransac_opts, const BundleOptions& bundle_opts, bool optimize_focal_length,
                    bool optimize_principal_point, PnPResult& result);
