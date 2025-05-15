#include "solvers.h"

#include <PoseLib/robust.h>
#include <PoseLib/robust/jacobian_impl.h>

#include "lev_marq.h"
#include "pnp_opengl.h"
#include "pnp_problem.h"
#include "robust_loss.h"
#include "utils.h"

template <typename LossFunction, typename ResidualWeightVector>
static inline void SolvePnPIterative(const ConstRefRowMajorMatrixX3f &object_points,
                                     const ConstRefRowMajorMatrixX2f &image_points, const BundleOptions &bundle_opts,
                                     bool optimize_focal_length, bool optimize_principal_point,
                                     ResidualWeightVector &&weights, PnPResult &result) {
    constexpr Float min_fov = 15 * M_PI / 180;
    constexpr Float max_fov = 160 * M_PI / 180;

    // TODO: constexpr
    const Float min_tan_fov_2 = std::tan(min_fov / 2);
    const Float max_tan_fov_2 = std::tan(max_fov / 2);

    Float f_low;
    Float f_high;

    if (result.camera.intrinsics.convention == CameraConvention::OpenGL) {
        f_low = -(result.camera.intrinsics.width / 2.0f) / min_tan_fov_2;
        f_high = -(result.camera.intrinsics.width / 2.0f) / max_tan_fov_2;
    } else {
        f_high = (result.camera.intrinsics.width / 2.0f) / min_tan_fov_2;
        f_low = (result.camera.intrinsics.width / 2.0f) / max_tan_fov_2;
    }

    const Float cx_low = 0.0f;
    const Float cx_high = result.camera.intrinsics.width;

    const Float cy_low = 0.0f;
    const Float cy_high = result.camera.intrinsics.height;

    CHECK(f_low < f_high);
    CHECK(cx_low < cx_high);
    CHECK(cy_low < cy_high);

    LossFunction loss_fn(bundle_opts.loss_scale);
    PnPProblem problem(image_points, object_points, optimize_focal_length, optimize_principal_point,
                       // TODO: Bounds should be an argument to SolvePnPIterative
                       PnPProblem::Bounds{
                           .f_low = f_low,
                           .f_high = f_high,
                           .cx_low = cx_low,
                           .cx_high = cx_high,
                           .cy_low = cy_low,
                           .cy_high = cy_high,
                       });
    PnPProblem::Parameters params{
        .cam = result.camera,
        .R = result.camera.pose.R(),
    };

    result.bundle_stats = LevMarqDenseSolve(problem, loss_fn, weights, &params, bundle_opts);
    result.camera = params.cam;
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

    result.camera.pose = CameraPose(Eigen::Vector4f(pose.q.cast<Float>()), Eigen::Vector3f(pose.t.cast<Float>()));

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
