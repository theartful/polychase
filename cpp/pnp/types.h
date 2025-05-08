#pragma once

#include <PoseLib/types.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "eigen_typedefs.h"
#include "quaternion.h"

enum class CameraConvention {
    OpenGL,  // Looking at -Z direction
    OpenCV,  // Looking at +Z direction
};

struct CameraIntrinsics {
    float fx;
    float fy;
    float cx;
    float cy;
    float aspect_ratio;
    float width;
    float height;

    // We can get rid of this field, and only use negative fx/fy/cx/cy, but adding it might be useful for debugging.
    CameraConvention convention;

    RowMajorMatrix4f To4x4ProjectionMatrix() const {
        // FIXME: I'm completely ignoring the view frustrum near and far clipping planes.
        // The z component due to using this projection matrix is completely bogus
        // This is just the camera intrinsics matrix (K).

        constexpr float f = 100.0;
        constexpr float n = 10.0;
        constexpr float p22 = -(f + n) / (f - n);
        constexpr float p23 = -2.0f * f * n / (f - n);

        // clang-format off
        return (RowMajorMatrix4f() <<
            fx,   0.0f, cx,   0.0f,
            0.0f, fy,   cy,   0.0f,
            0.0f, 0.0f, p22,  p23,
            0.0f, 0.0f, 1.0f, 0.0f
        ).finished();
        // clang-format on
    }

    RowMajorMatrix3f To3x3ProjectionMatrix() const {
        // clang-format off
        return (RowMajorMatrix3f() <<
            fx,   0.0f, cx,
            0.0f, fy,   cy,
            0.0f, 0.0f, 1.0f
        ).finished();
        // clang-format on
    }

    Eigen::Vector2f Project(const Eigen::Vector2f &x) const { return Eigen::Vector2f(fx * x(0) + cx, fy * x(1) + cy); }

    void ProjectWithJac(const Eigen::Vector2f &x, Eigen::Vector2f *xp, RowMajorMatrixf<2, 5> *jac) const {
        (*xp)(0) = fx * x(0) + cx;
        (*xp)(1) = fy * x(1) + cy;

        // We're assuming that fx/fy = aspect_ratio should always be true, so
        // fy is the free parameter, while fx=aspect_ratio*fy

        // clang-format off
        *jac <<
            fx,   0.0f, x(0) * aspect_ratio, 1.0f, 0.0f,
            0.0f, fy,   x(1),                0.0f, 1.0f;
        // clang-format on
    }
    void ProjectWithJac(const Eigen::Vector2f &x, Eigen::Vector2f *xp, RowMajorMatrix2f *jac) const {
        (*xp)(0) = fx * x(0) + cx;
        (*xp)(1) = fy * x(1) + cy;

        // clang-format off
        *jac <<
            fx,   0.0f,
            0.0f, fy;
        // clang-format on
    }

    Eigen::Vector2f Unproject(const Eigen::Vector2f &x) const {
        return Eigen::Vector2f((x(0) - cx) / fx, (x(1) - cy) / fy);
    }

    float Focal() const { return std::abs((fx + fy) / 2.0f); }

    inline bool IsBehind(const Eigen::Vector3f &p) const {
        return convention == CameraConvention::OpenCV ? p.z() < 0.0f : p.z() > 0.0f;
    }

    CameraIntrinsics Rescale(float scale) const {
        return CameraIntrinsics{
            .fx = fx * scale,
            .fy = fy * scale,
            .cx = cx * scale,
            .cy = cy * scale,
            .aspect_ratio = aspect_ratio,
            .width = width,
            .height = height,
            .convention = convention,
        };
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
    inline RowMajorMatrix4f Rt4x4() const {
        RowMajorMatrix4f tmp;
        tmp.block<3, 3>(0, 0) = quat_to_rotmat(q);
        tmp.block<3, 1>(0, 3) = t;
        tmp(3, 3) = 1.0f;
        tmp(3, 0) = 0.0f;
        tmp(3, 1) = 0.0f;
        tmp(3, 2) = 0.0f;
        return tmp;
    }
    inline Eigen::Vector3f Rotate(const Eigen::Vector3f &p) const { return quat_rotate(q, p); }
    inline Eigen::Vector3f Derotate(const Eigen::Vector3f &p) const { return quat_rotate(quat_conj(q), p); }
    inline Eigen::Vector3f Apply(const Eigen::Vector3f &p) const { return Rotate(p) + t; }

    inline Eigen::Vector3f Center() const { return -Derotate(t); }

    inline CameraPose Inverse() { return CameraPose(quat_conj(q), -Derotate(t)); }

    static inline CameraPose FromRt(const RowMajorMatrix4f &mat) {
        return CameraPose(RowMajorMatrix3f(mat.block<3, 3>(0, 0)), Eigen::Vector3f(mat.block<3, 1>(0, 3)));
    }

    static inline CameraPose FromSRt(const RowMajorMatrix4f &mat) {
        RowMajorMatrix3f mat_copy = mat.block<3, 3>(0, 0);
        mat_copy.col(0).normalize();
        mat_copy.col(1).normalize();
        mat_copy.col(2).normalize();
        return CameraPose(mat_copy, Eigen::Vector3f(mat.block<3, 1>(0, 3)));
    }

    static inline CameraPose FromRt(const RowMajorMatrix34f &mat) {
        return CameraPose(RowMajorMatrix3f(mat.template block<3, 3>(0, 0)),
                          Eigen::Vector3f(mat.template block<3, 1>(0, 3)));
    }
};

// YUCK FIXME
// Maybe use Sophus or define a proper SE3 class.
using Pose = CameraPose;

struct CameraState {
    CameraIntrinsics intrinsics;
    CameraPose pose;
};

using BundleOptions = poselib::BundleOptions;
using RansacOptions = poselib::RansacOptions;
using BundleStats = poselib::BundleStats;
using RansacStats = poselib::RansacStats;
