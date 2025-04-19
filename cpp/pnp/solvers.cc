#include "solvers.h"

#include <PoseLib/robust.h>
#include <PoseLib/robust/jacobian_impl.h>

#include "jacobian.h"
#include "lm_impl.h"
#include "utils.h"

class CauchyLoss {
   public:
    CauchyLoss(double threshold) : inv_sq_thr(1.0 / (threshold * threshold)) {}
    double loss(double r2) const { return std::log1p(r2 * inv_sq_thr); }
    double weight(double r2) const {
        return std::max(std::numeric_limits<double>::min(), inv_sq_thr / (1.0 + r2 * inv_sq_thr));
    }

   private:
    const double inv_sq_thr;
};

class TrivialLoss {
   public:
    TrivialLoss(double) {}  // dummy to ensure we have consistent calling interface
    TrivialLoss() {}
    double loss(double r2) const { return r2; }
    double weight(double) const { return 1.0; }
};

bool SolvePnPIterative(const ConstRefRowMajorMatrixX3f& object_points, const ConstRefRowMajorMatrixX2f& image_points,
                       PnPResult& result) {
    CHECK_EQ(object_points.rows(), image_points.rows());
    CHECK(object_points.rows() >= 3);
    CHECK(object_points.rows() == image_points.rows());

    poselib::BundleOptions opt = {};

    TrivialLoss loss_fn(opt.loss_scale);
    CameraJacobianAccumulator<TrivialLoss> accum(image_points, object_points, loss_fn, poselib::UniformWeightVector());

    result.bundle_stats = lm_impl<decltype(accum)>(accum, &result.camera, opt, nullptr);
    result.ransac_stats = {};

    return true;
}

bool SolvePnPRansac(const ConstRefRowMajorMatrixX3f& object_points, const ConstRefRowMajorMatrixX2f& image_points,
                    PnPResult& result) {
    // Use PoseLib estimate_absolute_pose
    poselib::Camera poselib_cam;
    poselib_cam.model_id = poselib::PinholeCameraModel::model_id;
    poselib_cam.width = 2.0;  // poselib doesn't really use the width and height params
    poselib_cam.height = 2.0;
    poselib_cam.params = {result.camera.intrinsics.fx, result.camera.intrinsics.fy, result.camera.intrinsics.cx,
                          result.camera.intrinsics.cy};

    // TODO: Edit poselib so that we don't have to allocate these variables.
    std::vector<poselib::Point2D> points2d;
    std::vector<poselib::Point3D> points3d;

    CHECK(object_points.rows() == image_points.rows());
    points3d.reserve(image_points.rows());
    points2d.reserve(object_points.rows());

    for (Eigen::Index i = 0; i < object_points.rows(); i++) {
        points3d.push_back(object_points.row(i).cast<double>());
        points2d.push_back(image_points.row(i).cast<double>());
    }
    std::vector<char> pose_lib_inliers;
    poselib::RansacOptions ransac_opts;
    poselib::BundleOptions bundle_opts;
    poselib::CameraPose pose;

    result.ransac_stats = poselib::estimate_absolute_pose(points2d, points3d, poselib_cam, ransac_opts, bundle_opts,
                                                          &pose, &pose_lib_inliers);
    result.camera.pose = {Eigen::Vector4f{pose.q.cast<float>()}, Eigen::Vector3f{pose.t.cast<float>()}};

    return true;
}
