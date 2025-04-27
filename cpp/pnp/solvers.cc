#include "solvers.h"

#include <PoseLib/robust.h>
#include <PoseLib/robust/jacobian_impl.h>

#include "jacobian.h"
#include "lm_impl.h"
#include "pnp_opengl.h"
#include "robust_loss.h"
#include "utils.h"

template <typename LossFunction>
static inline void SolvePnPIterative(const ConstRefRowMajorMatrixX3f& object_points,
                                     const ConstRefRowMajorMatrixX2f& image_points, const BundleOptions& bundle_opts,
                                     bool optimize_focal_length, bool optimize_principal_point, PnPResult& result) {
    LossFunction loss_fn(bundle_opts.loss_scale);
    CameraJacobianAccumulator<LossFunction> accum(image_points, object_points, loss_fn, optimize_focal_length,
                                                  optimize_principal_point, result.camera.intrinsics.width,
                                                  result.camera.intrinsics.height, result.camera.intrinsics.convention,
                                                  poselib::UniformWeightVector());

    result.bundle_stats = lm_impl<decltype(accum)>(accum, &result.camera, bundle_opts, nullptr);
}

void SolvePnPIterative(const ConstRefRowMajorMatrixX3f& object_points, const ConstRefRowMajorMatrixX2f& image_points,
                       const BundleOptions& bundle_opts, bool optimize_focal_length, bool optimize_principal_point,
                       PnPResult& result) {
    CHECK_EQ(object_points.rows(), image_points.rows());
    CHECK(object_points.rows() >= 3);

    switch (bundle_opts.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                                      \
    SolvePnPIterative<LossFunction>(object_points, image_points, bundle_opts, optimize_focal_length, \
                                    optimize_principal_point, result);
        SWITCH_LOSS_FUNCTIONS
#undef SWITCH_LOSS_FUNCTION_CASE
        default:
            throw std::runtime_error(fmt::format("Unknown loss type: {}", static_cast<int>(bundle_opts.loss_type)));
    }
}

void SolvePnPRansac(const ConstRefRowMajorMatrixX3f& object_points, const ConstRefRowMajorMatrixX2f& image_points,
                    const RansacOptions& ransac_opts, const BundleOptions& bundle_opts, bool optimize_focal_length,
                    bool optimize_principal_point, PnPResult& result) {
    CHECK_EQ(object_points.rows(), image_points.rows());

    OpenGLPnPAdapter adapter{object_points, result};

    std::vector<Eigen::Vector2d> points2D_calib;
    std::vector<Eigen::Vector3d> points3D;

    points2D_calib.reserve(static_cast<size_t>(image_points.rows()));
    points3D.reserve(image_points.rows());

    for (Eigen::Index i = 0; i < image_points.rows(); ++i) {
        points2D_calib.push_back(result.camera.intrinsics.unproject(image_points.row(i)).cast<double>());
        points3D.push_back(object_points.row(i).cast<double>());
    }

    RansacOptions ransac_opt_scaled = ransac_opts;
    ransac_opt_scaled.max_reproj_error /= result.camera.intrinsics.focal();

    std::vector<char> inliers;
    poselib::CameraPose pose(Eigen::Vector4d(result.camera.pose.q.cast<double>()),
                             Eigen::Vector3d(result.camera.pose.t.cast<double>()));
    result.ransac_stats = poselib::ransac_pnp(points2D_calib, points3D, ransac_opt_scaled, &pose, &inliers);

    result.camera.pose = CameraPose(Eigen::Vector4f(pose.q.cast<float>()), Eigen::Vector3f(pose.t.cast<float>()));

    if (result.ransac_stats->num_inliers > 3) {
        // Collect inliers
        // FIXME: I hate these allocations
        RowMajorMatrixX3f inlier_object_points(result.ransac_stats->num_inliers, 3);
        RowMajorMatrixX2f inlier_image_points(result.ransac_stats->num_inliers, 2);

        Eigen::Index row = 0;
        for (size_t i = 0; i < inliers.size(); i++) {
            if (inliers[i]) {
                inlier_object_points.row(row) = points3D[i].cast<float>();
                inlier_image_points.row(row) = image_points.row(i);
                row++;
            }
        }

        SolvePnPIterative(inlier_object_points, inlier_image_points, bundle_opts, optimize_focal_length,
                          optimize_principal_point, result);
    }
}
