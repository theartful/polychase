#pragma once

#include <Eigen/Core>

#include "eigen_typedefs.h"

enum class PnPSolveMethod { Iterative, Ransac };

struct PnPResult {
    Eigen::Vector3f translation;
    RowMajorMatrix3f rotation;
};

struct CameraIntrinsics {
    float fx;
    float fy;
    float cx;
    float cy;
};

// Assumes OpenGL camera, where we're looking at the negative Z direction.
bool SolvePnP(const ConstRefRowMajorMatrixX3f& object_points, const ConstRefRowMajorMatrixX2f& image_points,
              const CameraIntrinsics& camera, PnPSolveMethod method, PnPResult& result);

static inline bool SolvePnP(const ConstRefRowMajorMatrixX3f& object_points,
                            const ConstRefRowMajorMatrixX2f& image_points, const RowMajorMatrix4f& projection_matrix,
                            PnPSolveMethod method, PnPResult& result) {
    const CameraIntrinsics camera = {
        .fx = projection_matrix.coeff(0, 0),
        .fy = projection_matrix.coeff(1, 1),
        .cx = -projection_matrix.coeff(0, 2),
        .cy = -projection_matrix.coeff(1, 2),
    };
    return SolvePnP(object_points, image_points, camera, method, result);
}

static inline Eigen::Matrix3f CreateOpenGLCameraIntrinsicsMatrix(float fx, float fy, float cx, float cy) {
    // clang-format off
    return (RowMajorMatrix3f() <<
        fx,  0.0, -cx, 
        0.0, fy,  -cy, 
        0.0, 0.0, -1.0 // cx, cy, and 1.0 should be negative, but OpenCV doesn't handle the OpenGL convention
    ).finished();
    // clang-format on
}

static inline Eigen::Matrix3f CreateOpenGLCameraIntrinsicsMatrix(const CameraIntrinsics& camera) {
    return CreateOpenGLCameraIntrinsicsMatrix(camera.fx, camera.fy, camera.cx, camera.cy);
}

static inline Eigen::Matrix3f CreateOpenCVCameraIntrinsicsMatrix(float fx, float fy, float cx, float cy) {
    // clang-format off
    return (RowMajorMatrix3f() <<
        fx,  0.0, cx, 
        0.0, fy,  cy, 
        0.0, 0.0, 1.0 // cx, cy, and 1.0 should be negative, but OpenCV doesn't handle the OpenGL convention
    ).finished();
    // clang-format on
}

static inline Eigen::Matrix3f CreateOpenCVCameraIntrinsicsMatrix(const CameraIntrinsics& camera) {
    return CreateOpenCVCameraIntrinsicsMatrix(camera.fx, camera.fy, camera.cx, camera.cy);
}
