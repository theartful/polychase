#include "solvers.h"

#include <PoseLib/robust.h>
#include <PoseLib/robust/jacobian_impl.h>
#include <PoseLib/robust/lm_impl.h>

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

bool SolvePnPIterative(const ConstRefRowMajorMatrixX3f& object_points, const ConstRefRowMajorMatrixX2f& image_points,
                       const CameraIntrinsics& camera, PnPResult& result) {
    CHECK_EQ(object_points.rows(), image_points.rows());
    CHECK(object_points.rows() >= 3);

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

    poselib::Camera poselib_cam;
    poselib_cam.model_id = poselib::PinholeCameraModel::model_id;
    poselib_cam.width = 2.0;  // poselib doesn't really use the width and height params
    poselib_cam.height = 2.0;
    poselib_cam.params = {camera.fx, camera.fy, camera.cx, camera.cy};

    poselib::BundleOptions opt = {};

    CauchyLoss loss_fn(opt.loss_scale);
    poselib::CameraJacobianAccumulator<poselib::PinholeCameraModel, CauchyLoss> accum(
        points2d, points3d, poselib_cam, loss_fn, poselib::UniformWeightVector());

    poselib::CameraPose pose(poselib::rotmat_to_quat(result.rotation.cast<double>()),
                             result.translation.cast<double>());
    poselib::lm_impl<decltype(accum)>(accum, &pose, opt, nullptr);

    result.rotation = pose.R().cast<float>();
    result.translation = pose.t.cast<float>();

    return true;
}

bool SolvePnPRansac(const ConstRefRowMajorMatrixX3f& object_points, const ConstRefRowMajorMatrixX2f& image_points,
                    const CameraIntrinsics& camera, PnPResult& result) {
    // Use PoseLib estimate_absolute_pose
    poselib::Camera poselib_cam;
    poselib_cam.model_id = poselib::PinholeCameraModel::model_id;
    poselib_cam.width = 2.0;  // poselib doesn't really use the width and height params
    poselib_cam.height = 2.0;
    poselib_cam.params = {camera.fx, camera.fy, camera.cx, camera.cy};

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

    [[maybe_unused]] poselib::RansacStats stats = poselib::estimate_absolute_pose(
        points2d, points3d, poselib_cam, ransac_opts, bundle_opts, &pose, &pose_lib_inliers);

    result = {.translation = pose.t.cast<float>(), .rotation = pose.R().cast<float>()};
    return true;
}
