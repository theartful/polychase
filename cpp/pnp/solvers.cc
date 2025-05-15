#include "solvers.h"

#include <PoseLib/robust.h>
#include <PoseLib/robust/jacobian_impl.h>

#include "jacobian.h"
#include "lev_marq.h"
#include "lm_impl.h"
#include "pnp_opengl.h"
#include "robust_loss.h"
#include "utils.h"

template <typename LossFunction, typename ResidualWeightVector>
static inline void SolvePnPIterative(const ConstRefRowMajorMatrixX3f &object_points,
                                     const ConstRefRowMajorMatrixX2f &image_points, const BundleOptions &bundle_opts,
                                     bool optimize_focal_length, bool optimize_principal_point,
                                     ResidualWeightVector &&weights, PnPResult &result) {
    LossFunction loss_fn(bundle_opts.loss_scale);
    // CameraJacobianAccumulator<LossFunction, ResidualWeightVector> accum(
    //     image_points, object_points, loss_fn, optimize_focal_length, optimize_principal_point,
    //     result.camera.intrinsics.width, result.camera.intrinsics.height, result.camera.intrinsics.convention,
    //     std::forward<ResidualWeightVector>(weights));

    // result.bundle_stats = lm_impl<decltype(accum)>(accum, &result.camera, bundle_opts, nullptr);

    CameraJacobian problem(image_points, object_points, optimize_focal_length, optimize_principal_point,
                           result.camera.intrinsics.width, result.camera.intrinsics.height,
                           result.camera.intrinsics.convention);
    result.bundle_stats = LevMarqDenseSolve(problem, loss_fn, weights, &result.camera, bundle_opts);
}

template <typename ResidualWeightVector>
static inline void SolvePnPIterative(const ConstRefRowMajorMatrixX3f &object_points,
                                     const ConstRefRowMajorMatrixX2f &image_points, const BundleOptions &bundle_opts,
                                     bool optimize_focal_length, bool optimize_principal_point,
                                     ResidualWeightVector &&weights, PnPResult &result) {
    CHECK_EQ(object_points.rows(), image_points.rows());
    CHECK(object_points.rows() >= 3);

    switch (bundle_opts.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                                      \
    SolvePnPIterative<LossFunction>(object_points, image_points, bundle_opts, optimize_focal_length, \
                                    optimize_principal_point, std::forward<ResidualWeightVector>(weights), result);
        SWITCH_LOSS_FUNCTIONS
#undef SWITCH_LOSS_FUNCTION_CASE
        default:
            throw std::runtime_error(fmt::format("Unknown loss type: {}", static_cast<int>(bundle_opts.loss_type)));
    }
}

void SolvePnPIterative(const ConstRefRowMajorMatrixX3f &object_points, const ConstRefRowMajorMatrixX2f &image_points,
                       const BundleOptions &bundle_opts, bool optimize_focal_length, bool optimize_principal_point,
                       PnPResult &result) {
    SolvePnPIterative(object_points, image_points, bundle_opts, optimize_focal_length, optimize_principal_point,
                      poselib::UniformWeightVector(), result);
}

struct InlierWeightsVector {
    double operator[](std::size_t idx) const {
        CHECK_LT(idx, inliers.size());
        return inliers[idx] ? 1.0 : 0.0;
    }
    const std::vector<char> &inliers;
};

void SolvePnPRansac(const ConstRefRowMajorMatrixX3f &object_points, const ConstRefRowMajorMatrixX2f &image_points,
                    const RansacOptions &ransac_opts, const BundleOptions &bundle_opts, bool optimize_focal_length,
                    bool optimize_principal_point, SolvePnPRansacCache &cache, PnPResult &result) {
    CHECK_EQ(object_points.rows(), image_points.rows());

    OpenGLPnPAdapter adapter{object_points, result};

    cache.Clear();
    cache.points2D_calib.reserve(static_cast<size_t>(image_points.rows()));
    cache.points3D.reserve(static_cast<size_t>(image_points.rows()));

    for (Eigen::Index i = 0; i < image_points.rows(); ++i) {
        cache.points2D_calib.push_back(
            result.camera.intrinsics.Unproject(image_points.row(i)).head<2>().cast<double>());
        cache.points3D.push_back(object_points.row(i).cast<double>());
    }

    RansacOptions ransac_opt_scaled = ransac_opts;
    ransac_opt_scaled.max_reproj_error /= result.camera.intrinsics.Focal();

    poselib::CameraPose pose(Eigen::Vector4d(result.camera.pose.q.cast<double>()),
                             Eigen::Vector3d(result.camera.pose.t.cast<double>()));
    result.ransac_stats =
        poselib::ransac_pnp(cache.points2D_calib, cache.points3D, ransac_opt_scaled, &pose, &cache.inliers);

    result.camera.pose = CameraPose(Eigen::Vector4f(pose.q.cast<float>()), Eigen::Vector3f(pose.t.cast<float>()));

    if (result.ransac_stats->num_inliers > 3) {
        SolvePnPIterative(object_points, image_points, bundle_opts, optimize_focal_length, optimize_principal_point,
                          InlierWeightsVector{cache.inliers}, result);
    }
}

void SolvePnPRansac(const ConstRefRowMajorMatrixX3f &object_points, const ConstRefRowMajorMatrixX2f &image_points,
                    const RansacOptions &ransac_opts, const BundleOptions &bundle_opts, bool optimize_focal_length,
                    bool optimize_principal_point, PnPResult &result) {
    SolvePnPRansacCache cache;
    SolvePnPRansac(object_points, image_points, ransac_opts, bundle_opts, optimize_focal_length,
                   optimize_principal_point, cache, result);
}
