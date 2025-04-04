#include "pnp.h"

#include <opencv2/calib3d.hpp>

#include "utils.h"

// clang-format off
const RowMajorMatrix3f NEGATIVE_Z = (RowMajorMatrix3f() <<
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, -1.0
).finished();
// clang-format on

static void NegateZAxisOfObjectPoints(const ConstRefRowMajorMatrixX3f& object_points) {
    // Doing it in-place when we promised const is horrible. But maybe it's better than allocating a whole new array.
    // TODO: Modify the implementation of OpenCV to support OpenGL convention.
    for (Eigen::Index row = 0; row < object_points.rows(); row++) {
        // Disgusting
        const_cast<float*>(object_points.data())[3 * row + 2] *= -1;
    }
}

bool SolvePnP(const ConstRefRowMajorMatrixX3f& object_points, const ConstRefRowMajorMatrixX2f& image_points,
              const CameraIntrinsics& camera, PnPResult& result) {
    // We're trying to find the matrix T that minimizes the reprojection error:
    //       argmin(T) of || Proj(K, T * object_points) - image_points ||
    //
    // Where K is the intrinsics matrix.
    //
    // OpenCV convention: X right - Y down - looking at Z
    // OpenGL convention: X right - Y up   - looking at -Z
    //
    // Normally, when we're dealing with OpenCV convention, the intrinsics matrix is:
    //        K_opencv  = | fx  0   cx |
    //                    | 0   fy  cy |
    //                    | 0   0   1  |
    //
    // But since this is OpenGL convention, we have:
    //        K_opengl  = | fx  0   -cx |
    //                    | 0   fy  -cy |
    //                    | 0   0   -1  |
    //
    // We have to do the computation using K_opencv, otherwise OpenCV gets sad, so we're solving:
    //        argmin(T) of || Proj(K_opencv, T * object_points) - image_points ||
    //
    // But we want instead to solve:
    //        argmin(U) of || Proj(K_opengl, U * object_points) - image_points ||
    //
    // To do that, we can do the transformation `K_opencv = K_opengl * inv(K_opengl) * K_opencv`
    // which is equivalent to `K_opencv = K_opengl * NEGATIVE_Z`:
    //
    //        argmin(T) of || Proj(K_opencv, T * object_points) - image_points ||
    //        argmin(T) of || Proj(K_opengl * NEGATIVE_Z, T * object_points) - image_points ||
    //        argmin(T) of || Proj(K_opengl, NEGATIVE_Z * T * object_points) - image_points ||
    //
    // This essentially solves the problem we're interested in, with `U = NEGAtIVE_Z * T`.
    // The problem with this solution is that NEGATIVE_Z flips handedness.
    //
    // So instead, let's solve this problem:
    //      argmin(T) of || Proj(K_opencv, T * (NEGATIVE_Z * object_points)) - image_points ||
    //
    // And with the same analysis, we get U = NEGATIVE_Z * T * NEGATIVE_Z, which maintains handedness.
    //
    CHECK(object_points.rows() > 2);

    const RowMajorMatrixX3f K = CreateOpenCVCameraIntrinsicsMatrix(camera);

    // Negate the z direction of object_points. We have to make sure to restore it before returning from the function,
    // because we promised that data is const. This however makes it so that object_points cannot be used in a
    // multithreaded fashion if this function is called.
    NegateZAxisOfObjectPoints(object_points);

    // Setup initial solution
    //
    // Negate the z-direction, since this initial solution is in K_opengl terms, while we're using K_opencv
    result.rotation = NEGATIVE_Z * result.rotation * NEGATIVE_Z;
    result.translation.z() *= -1;

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

    // Restore object_points
    NegateZAxisOfObjectPoints(object_points);

    if (!ok) {
        return false;
    }

    cv::Rodrigues(rvec_cv, rot_cv);

    // Restore result to OpenGL land.
    result.rotation = NEGATIVE_Z * result.rotation * NEGATIVE_Z;
    result.translation.z() *= -1;

    return true;
}
