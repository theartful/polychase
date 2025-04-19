#pragma once

#include <PoseLib/types.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "eigen_typedefs.h"
#include "quaternion.h"

struct CameraIntrinsics {
    float fx;
    float fy;
    float cx;
    float cy;
    float aspect_ratio;

    static inline CameraIntrinsics FromProjectionMatrix(const RowMajorMatrix4f &projection_matrix) {
        return {
            .fx = projection_matrix(0, 0),
            .fy = projection_matrix(1, 1),
            .cx = projection_matrix(0, 2),
            .cy = projection_matrix(1, 2),
            .aspect_ratio = projection_matrix(0, 0) / projection_matrix(1, 1),
        };
    }

    Eigen::Vector2f project(const Eigen::Vector2f &x) const { return Eigen::Vector2f(fx * x[0] + cx, fy * x[1] + cy); }

    void project_with_jac(const Eigen::Vector2f &x, Eigen::Vector2f *xp, RowMajorMatrixf<2, 5> *jac) const {
        (*xp)(0) = fx * x[0] + cx;
        (*xp)(1) = fy * x[1] + cy;

        // We're assuming that fx/fy = aspect_ratio should always be true, so
        // fy is the free parameter, while fx=aspect_ratio*fy

        // clang-format off
        *jac <<
            fx,  0,  x[0] * aspect_ratio, 1.0, 0.0,
            0.0, fy, x[1],                0.0, 1.0;
        // clang-format on
    }
    void project_with_jac(const Eigen::Vector2f &x, Eigen::Vector2f *xp, RowMajorMatrix2f *jac) const {
        (*xp)(0) = fx * x[0] + cx;
        (*xp)(1) = fy * x[1] + cy;

        // clang-format off
        *jac <<
            fx,  0,
            0.0, fy;
        // clang-format on
    }

    Eigen::Vector2f unproject(const Eigen::Vector2f &x) const {
        return Eigen::Vector2f((x[0] - cx) / fx, (x[1] - cy) / fy);
    }
};

struct CameraPose {
    // Rotation is represented as a unit quaternion
    // with real part first, i.e. QW, QX, QY, QZ
    Eigen::Vector4f q;
    Eigen::Vector3f t;

    // Constructors (Defaults to identity camera)
    CameraPose() : q(1.0, 0.0, 0.0, 0.0), t(0.0, 0.0, 0.0) {}
    CameraPose(const Eigen::Vector4f &qq, const Eigen::Vector3f &tt) : q(qq), t(tt) {}
    CameraPose(const RowMajorMatrix3f &R, const Eigen::Vector3f &tt) : q(rotmat_to_quat(R)), t(tt) {}

    // Helper functions
    inline RowMajorMatrix3f R() const { return quat_to_rotmat(q); }
    inline RowMajorMatrix34f Rt() const {
        RowMajorMatrix34f tmp;
        tmp.block<3, 3>(0, 0) = quat_to_rotmat(q);
        tmp.col(3) = t;
        return tmp;
    }
    inline Eigen::Vector3f rotate(const Eigen::Vector3f &p) const { return quat_rotate(q, p); }
    inline Eigen::Vector3f derotate(const Eigen::Vector3f &p) const { return quat_rotate(quat_conj(q), p); }
    inline Eigen::Vector3f apply(const Eigen::Vector3f &p) const { return rotate(p) + t; }

    inline Eigen::Vector3f center() const { return -derotate(t); }

    static inline CameraPose FromRt(const RowMajorMatrix4f &mat) {
        return CameraPose(RowMajorMatrix3f(mat.template block<3, 3>(0, 0)),
                          Eigen::Vector3f(mat.template block<3, 1>(0, 3)));
    }

    static inline CameraPose FromRt(const RowMajorMatrix34f &mat) {
        return CameraPose(RowMajorMatrix3f(mat.template block<3, 3>(0, 0)),
                          Eigen::Vector3f(mat.template block<3, 1>(0, 3)));
    }
};

struct CameraState {
    CameraIntrinsics intrinsics;
    CameraPose pose;
};

static inline CameraIntrinsics ProjectionMatrixToIntrinsics(const ConstRefRowMajorMatrix4f &projection_matrix) {
    return {
        .fx = projection_matrix(0, 0),
        .fy = projection_matrix(1, 1),
        .cx = projection_matrix(0, 2),
        .cy = projection_matrix(1, 2),
        .aspect_ratio = projection_matrix(0, 0) / projection_matrix(1, 1),
    };
}

using BundleOptions = poselib::BundleOptions;
using BundleStats = poselib::BundleStats;
using RansacStats = poselib::RansacStats;
