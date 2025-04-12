#include "pnp.h"

#include <PoseLib/misc/colmap_models.h>
#include <PoseLib/robust.h>
#include <spdlog/spdlog.h>

#include <opencv2/calib3d.hpp>

#include "utils.h"

// clang-format off
const RowMajorMatrix3f NEGATIVE_Z = (RowMajorMatrix3f() <<
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, -1.0
).finished();
// clang-format on

struct CameraIntrinsics {
    float fx;
    float fy;
    float cx;
    float cy;
};

static inline Eigen::Matrix3f CreateOpenCVCameraIntrinsicsMatrix(float fx, float fy, float cx, float cy) {
    // clang-format off
    return (RowMajorMatrix3f() <<
        fx,  0.0, cx,
        0.0, fy,  cy,
        0.0, 0.0, 1.0
    ).finished();
    // clang-format on
}

static inline Eigen::Matrix3f CreateOpenCVCameraIntrinsicsMatrix(const CameraIntrinsics& camera) {
    return CreateOpenCVCameraIntrinsicsMatrix(camera.fx, camera.fy, camera.cx, camera.cy);
}

static void NegateZAxisOfObjectPoints(const ConstRefRowMajorMatrixX3f& object_points) {
    // Doing it in-place when we promised const is horrible. But maybe it's better than allocating a whole new array.
    // TODO: Modify the implementation of OpenCV to support OpenGL convention.
    for (Eigen::Index row = 0; row < object_points.rows(); row++) {
        // Disgusting
        const_cast<float*>(object_points.data())[3 * row + 2] *= -1;
    }
}

static bool SolvePnPRansac(const ConstRefRowMajorMatrixX3f& object_points,
                           const ConstRefRowMajorMatrixX2f& image_points, const CameraIntrinsics& camera,
                           PnPResult& result) {
    // Use PoseLib estimate_absolute_pose
    poselib::Camera poselib_cam;
    poselib_cam.model_id = poselib::PinholeCameraModel::model_id;
    poselib_cam.width = 2.0;  // poselib doesn't really use the width and height params
    poselib_cam.height = 2.0;
    poselib_cam.params = {camera.fx, camera.fy, camera.cx, camera.cy};

    std::vector<Eigen::Vector2d> points2d;
    std::vector<Eigen::Vector3d> points3d;

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

    // spdlog::info("TOTAL NUMBER = {} NUM INLIERS = {}", object_points.rows(), stats.num_inliers);
    result = {.translation = pose.t.cast<float>(), .rotation = pose.R().cast<float>()};
    return true;
}

static bool SolvePnPIterative(const ConstRefRowMajorMatrixX3f& object_points,
                              const ConstRefRowMajorMatrixX2f& image_points, const CameraIntrinsics& camera,
                              PnPResult& result) {
    const RowMajorMatrixX3f K = CreateOpenCVCameraIntrinsicsMatrix(camera);

    Eigen::Vector3f rvec = {};
    cv::Mat rot_cv{3, 3, CV_32F, result.rotation.data()};
    cv::Mat tvec_cv{3, 1, CV_32F, result.translation.data()};
    cv::Mat rvec_cv{3, 1, CV_32F, rvec.data()};

    cv::Rodrigues(rot_cv, rvec_cv);

    // Transform image_points/object_points/K_opencv to cv::Mat
    const cv::Mat object_points_cv{static_cast<int>(object_points.rows()), 3, CV_32F,
                                   const_cast<float*>(object_points.data())};
    const cv::Mat image_points_cv{static_cast<int>(image_points.rows()), 2, CV_32F,
                                  const_cast<float*>(image_points.data())};
    const cv::Mat K_cv{3, 3, CV_32F, const_cast<float*>(K.data())};

    const bool ok =
        cv::solvePnP(object_points_cv, image_points_cv, K_cv, {}, rvec_cv, tvec_cv, true, cv::SOLVEPNP_ITERATIVE);

    cv::Rodrigues(rvec_cv, rot_cv);

    return ok;
}

bool SolvePnP(const ConstRefRowMajorMatrixX3f& object_points, const ConstRefRowMajorMatrixX2f& image_points,
              const RowMajorMatrix4f& projection_matrix, PnPSolveMethod method, PnPResult& result) {
    // We're trying to find the matrix T that minimizes the reprojection error:
    //       argmin(T) of || Proj(K, T * object_points) - image_points ||
    //
    // Where K is the intrinsics matrix.
    //
    // OpenCV convention: X right / Y down / looking at +Z
    // OpenGL convention: X right / Y up   / looking at -Z
    //
    // Normally, when we're dealing with OpenCV convention, the intrinsics matrix is:
    //        K_opencv  = | fx  0   cx |
    //                    | 0   fy  cy |
    //                    | 0   0   1  |
    //
    // But since this is OpenGL convention, we have:
    //        K_opengl  = | fx  0   cx |
    //                    | 0   fy  cy |
    //                    | 0   0   -1 |
    //                              ^
    //                              this is different, as we're looking at negative Z instead
    //
    // We have to do the computation using K_opencv, otherwise OpenCV gets sad, so we're solving:
    //        argmin(T) of || Proj(K_opencv, T * object_points) - image_points ||
    //
    // But we want instead to solve:
    //        argmin(U) of || Proj(K_opengl, U * object_points) - image_points ||
    //
    // To do that, we can do the transformation `K_opencv = K_opengl * inv(K_opengl) * K_opencv`
    // which is equivalent to `K_opencv = K_opengl * FIX`:
    //
    //        argmin(T) of || Proj(K_opencv, T * object_points) - image_points ||
    //        argmin(T) of || Proj(K_opengl * FIX,  T * object_points) - image_points ||
    //        argmin(T) of || Proj(K_opengl,  FIX * T * object_points) - image_points ||
    //
    // This essentially solves the problem we're interested in, with `U = FIX * T`.
    // The problem with this solution is that FIX flips handedness.
    //
    // So instead, let's solve this problem:
    //      argmin(T) of || Proj(K_opencv, T * (FIX * object_points)) - image_points ||
    //
    // And with the same analysis, we get U = FIX * T * FIX, which maintains handedness.
    //
    // Finally, we can also do another trick, and use
    //        K_opencv  = | fx  0   -cx |
    //                    | 0   fy  -cy |
    //                    | 0   0   1   |
    //
    // which will make FIX = NEGATIVE_Z
    CHECK(object_points.rows() > 2);
    CHECK_EQ(projection_matrix.coeff(3, 2), -1.0f);

    const CameraIntrinsics camera_cv = {
        .fx = projection_matrix.coeff(0, 0),
        .fy = projection_matrix.coeff(1, 1),
        .cx = -projection_matrix.coeff(0, 2),
        .cy = -projection_matrix.coeff(1, 2),
    };

    // Negate the z direction of object_points. We have to make sure to restore it before returning from the function,
    // because we promised that data is const. This however makes it so that object_points cannot be used in a
    // multithreaded fashion if this function is called.
    NegateZAxisOfObjectPoints(object_points);

    // Setup initial solution
    //
    // Negate the z-direction, since this initial solution is in K_opengl terms, while we're using K_opencv
    result.rotation = NEGATIVE_Z * result.rotation * NEGATIVE_Z;
    result.translation.z() *= -1;

    bool ok;
    switch (method) {
        case PnPSolveMethod::Iterative: {
            ok = SolvePnPIterative(object_points, image_points, camera_cv, result);
            break;
        }
        case PnPSolveMethod::Ransac: {
            ok = SolvePnPRansac(object_points, image_points, camera_cv, result);
            break;
        }
        default:
            NegateZAxisOfObjectPoints(object_points);
            throw std::runtime_error(fmt::format("Invalid PnPSolveMethod value: {}", static_cast<int>(method)));
    }

    // Restore object_points
    NegateZAxisOfObjectPoints(object_points);

    // Restore result to OpenGL land.
    result.rotation = NEGATIVE_Z * result.rotation * NEGATIVE_Z;
    result.translation.z() *= -1;

    return ok;
}
