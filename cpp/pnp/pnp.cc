#include "pnp.h"

#include <PoseLib/misc/colmap_models.h>
#include <PoseLib/robust.h>
#include <spdlog/spdlog.h>

#include "solvers.h"
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
