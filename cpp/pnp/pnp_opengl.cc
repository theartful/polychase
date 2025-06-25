// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

#include "pnp_opengl.h"

#include <spdlog/spdlog.h>

#include "solvers.h"
#include "utils.h"

// We're trying to find the matrix T that minimizes the reprojection error:
//       argmin(T) of || Proj(K, T * object_pts) - image_pts ||
//
// Where K is the intrinsics matrix.
//
// OpenCV convention: X right / Y down / looking at +Z
// OpenGL convention: X right / Y up   / looking at -Z
//
// Normally, when we're dealing with OpenCV convention, the intrinsics matrix
// is:
//      K_opencv  = | fx  0   cx |
//                  | 0   fy  cy |
//                  | 0   0   1  |
//
// But since this is OpenGL convention, we have:
//      K_opengl  = | fx  0   cx |
//                  | 0   fy  cy |
//                  | 0   0   -1 |
//                            ^
//                            this is different, as we're looking at negative
//                            Z instead
//
// We have to do the computation using K_opencv, otherwise poselib p3p gets sad,
// so we're solving:
//      argmin(T) of || Proj(K_opencv, T * object_pts) - image_pts ||
//
// But we want instead to solve:
//      argmin(U) of || Proj(K_opengl, U * object_pts) - image_pts ||
//
// To do that, we can do the transformation:
//      K_opencv = K_opengl * inv(K_opengl) * K_opencv
//
// Let FIX = inv(K_opengl) * K_opencv. This leads to:
//      argmin(T) of || Proj(K_opencv, T * object_pts) - image_pts ||
//      argmin(T) of || Proj(K_opengl * FIX,  T * object_pts) - image_pts ||
//      argmin(T) of || Proj(K_opengl,  FIX * T * object_pts) - image_pts ||
//
// This essentially solves the problem we're interested in, with `U = FIX * T`.
// The problem with this solution is that FIX flips handedness.
//
// So instead, let's solve this problem:
//      argmin(T) of || Proj(K_opencv, T * (FIX * object_pts)) - image_pts ||
//
// And with the same analysis, we get U = FIX * T * FIX, which maintains
// handedness.
//
// Finally, we can also do another trick, and use
//        K_opencv  = | fx  0   -cx |
//                    | 0   fy  -cy |
//                    | 0   0   1   |
//
// which will make FIX = diag(1, 1, -1)

static void NegateZAxisOfObjectPoints(RefRowMajorMatrixX3f& object_points) {
    for (Eigen::Index row = 0; row < object_points.rows(); row++) {
        object_points(row, 2) *= -1;
    }
}

void PnPOpenGLPreprocessing(RefRowMajorMatrixX3f object_points,
                            PnPResult& result) {
    CHECK_EQ(result.camera.intrinsics.convention, CameraConvention::OpenGL);
    NegateZAxisOfObjectPoints(object_points);

    result.camera.intrinsics.fx *= -1.0f;
    result.camera.intrinsics.fy *= -1.0f;
    result.camera.intrinsics.convention = CameraConvention::OpenCV;

    result.camera.pose.q = {result.camera.pose.q[0], -result.camera.pose.q[1],
                            -result.camera.pose.q[2], result.camera.pose.q[3]};
    result.camera.pose.t.z() *= -1;
}

void PnPOpenGLPostprocessing(RefRowMajorMatrixX3f object_points,
                             PnPResult& result) {
    CHECK_EQ(result.camera.intrinsics.convention, CameraConvention::OpenCV);
    NegateZAxisOfObjectPoints(object_points);

    result.camera.intrinsics.fx *= -1.0f;
    result.camera.intrinsics.fy *= -1.0f;
    result.camera.intrinsics.convention = CameraConvention::OpenGL;

    result.camera.pose.q = {result.camera.pose.q[0], -result.camera.pose.q[1],
                            -result.camera.pose.q[2], result.camera.pose.q[3]};
    result.camera.pose.t.z() *= -1;
}
