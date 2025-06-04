#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "eigen_typedefs.h"
#include "quaternion.h"
#include "utils.h"

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

    // We can get rid of this field, and only use negative fx/fy/cx/cy, but
    // adding it might be useful for debugging.
    CameraConvention convention;

    RowMajorMatrix4f To4x4ProjectionMatrix() const {
        // FIXME: I'm completely ignoring the view frustrum near and far
        // clipping planes. The z component due to using this projection matrix
        // is completely bogus This is just the camera intrinsics matrix (K).

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

    Eigen::Vector2f Project(const Eigen::Vector2f &x) const {
        return Eigen::Vector2f(fx * x(0) + cx, fy * x(1) + cy);
    }

    Eigen::Vector2f Project(const Eigen::Vector3f &x) const {
        return Eigen::Vector2f(fx * x(0) / x(2) + cx, fy * x(1) / x(2) + cy);
    }

    void ProjectWithJac(const Eigen::Vector3f &x, Eigen::Vector2f *xp,
                        RowMajorMatrixf<2, 3> *jac_x = nullptr,
                        RowMajorMatrixf<2, 3> *jac_intrin = nullptr) const {
        (*xp)(0) = fx * x(0) / x(2) + cx;
        (*xp)(1) = fy * x(1) / x(2) + cy;

        // We're assuming that fx/fy = aspect_ratio should always be true, so
        // fy is the free parameter, while fx=aspect_ratio*fy

        if (jac_x) {
            // clang-format off
            *jac_x <<
                fx / x(2),  0,          -fx * x(0) / (x(2) * x(2)),
                0,          fy / x(2),  -fy * x(1) / (x(2) * x(2));
            // clang-format on
        }

        if (jac_intrin) {
            // clang-format off
            *jac_intrin <<
                aspect_ratio * x(0) / x(2),     1.0f,   0.0f,
                x(1) / x(2),                    0.0f,   1.0f;
            // clang-format on
        }
    }

    Eigen::Vector3f Unproject(const Eigen::Vector2f &x) const {
        const float s = convention == CameraConvention::OpenCV ? 1.0f : -1.0f;
        return s * Eigen::Vector3f((x(0) - cx) / fx, (x(1) - cy) / fy, 1.0f);
    }

    void UnprojectWithJac(const Eigen::Vector2f &x, Eigen::Vector3f *xup,
                          RowMajorMatrixf<3, 3> *jac_x = nullptr,
                          RowMajorMatrixf<3, 3> *jac_intrin = nullptr) const {
        const float s = convention == CameraConvention::OpenCV ? 1.0f : -1.0f;

        (*xup)(0) = s * (x(0) - cx) / fx;
        (*xup)(1) = s * (x(1) - cy) / fy;
        (*xup)(2) = s;

        if (jac_x) {
            // clang-format off
            *jac_x <<
                s / fx,  0.0f,
                0.0f,    s / fy,
                0.0f,    0.0f;
            // clang-format on
        }
        if (jac_intrin) {
            // clang-format off
            *jac_intrin <<
                s * (cx - x(0)) / (fy * fy * aspect_ratio),     -s/fx,    0.0f,
                s * (cy - x(1)) / (fy * fy),                    0.0f,    -s/fy,
                0.0f,                                           0.0f,     0.0f;
            // clang-format on
        }
    }

    float Focal() const { return std::abs((fx + fy) / 2.0f); }

    inline bool IsBehind(const Eigen::Vector3f &p) const {
        return convention == CameraConvention::OpenCV ? p.z() < 0.0f
                                                      : p.z() > 0.0f;
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

    struct Bounds {
        Float f_low;
        Float f_high;
        Float cx_low;
        Float cx_high;
        Float cy_low;
        Float cy_high;
    };

    Bounds GetBounds(Float min_fov_deg = 15, Float max_fov_deg = 160) const {
        const Float min_fov = min_fov_deg * M_PI / 180;
        const Float max_fov = max_fov_deg * M_PI / 180;

        const Float min_tan_fov_2 = std::tan(min_fov / 2);
        const Float max_tan_fov_2 = std::tan(max_fov / 2);

        Float f_low;
        Float f_high;

        if (convention == CameraConvention::OpenGL) {
            f_low = -(width / 2.0f) / min_tan_fov_2;
            f_high = -(width / 2.0f) / max_tan_fov_2;
        } else {
            f_high = (width / 2.0f) / min_tan_fov_2;
            f_low = (width / 2.0f) / max_tan_fov_2;
        }

        const Float cx_low = 0.0f;
        const Float cx_high = width;

        const Float cy_low = 0.0f;
        const Float cy_high = height;

        CHECK(f_low < f_high);
        CHECK(cx_low < cx_high);
        CHECK(cy_low < cy_high);

        return Bounds{
            .f_low = f_low,
            .f_high = f_high,
            .cx_low = cx_low,
            .cx_high = cx_high,
            .cy_low = cy_low,
            .cy_high = cy_high,
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
    CameraPose(const Eigen::Vector4f &qq, const Eigen::Vector3f &tt)
        : q(qq), t(tt) {}
    CameraPose(const RowMajorMatrix3f &R, const Eigen::Vector3f &tt)
        : q(rotmat_to_quat(R)), t(tt) {}

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
    inline Eigen::Vector3f Rotate(const Eigen::Vector3f &p) const {
        return quat_rotate(q, p);
    }
    inline Eigen::Vector3f Derotate(const Eigen::Vector3f &p) const {
        return quat_rotate(quat_conj(q), p);
    }
    inline Eigen::Vector3f Apply(const Eigen::Vector3f &p) const {
        return Rotate(p) + t;
    }

    inline Eigen::Vector3f Center() const { return -Derotate(t); }

    inline void CenterWithJac(Eigen::Vector3f *center,
                              RowMajorMatrixf<3, 3> *jac_R = nullptr,
                              RowMajorMatrixf<3, 3> *jac_t = nullptr) const {
        *center = Center();

        if (jac_R) {
            *jac_R = Skew(*center);
        }
        if (jac_t) {
            *jac_t = -R().transpose();
        }
    }

    inline void ApplyWithJac(const Eigen::Vector3f &p, Eigen::Vector3f *result,
                             RowMajorMatrixf<3, 3> *jac_p = nullptr,
                             RowMajorMatrixf<3, 3> *jac_R = nullptr,
                             RowMajorMatrixf<3, 3> *jac_t = nullptr) const {
        ApplyWithJac(p, R(), t, result, jac_p, jac_R, jac_t);
    }

    static inline void ApplyWithJac(const Eigen::Vector3f &p,
                                    const RowMajorMatrix3f &R,
                                    const Eigen::Vector3f &t,
                                    Eigen::Vector3f *result,
                                    RowMajorMatrixf<3, 3> *jac_p = nullptr,
                                    RowMajorMatrixf<3, 3> *jac_R = nullptr,
                                    RowMajorMatrixf<3, 3> *jac_t = nullptr) {
        *result = R * p + t;

        if (jac_p) {
            *jac_p = R;
        }
        if (jac_R) {
            *jac_R = R * Skew(-p);
        }
        if (jac_t) {
            *jac_t = RowMajorMatrixf<3, 3>::Identity();
        }
    }

    inline void DerotateWithJac(const Eigen::Vector3f &p,
                                Eigen::Vector3f *result,
                                RowMajorMatrixf<3, 3> *jac_p = nullptr,
                                RowMajorMatrixf<3, 3> *jac_R = nullptr) const {
        *result = Derotate(p);
        if (jac_p) {
            *jac_p = R().transpose();
        }
        if (jac_R) {
            *jac_R = Skew(*result);
        }
    }

    static inline void DerotateWithJac(const Eigen::Vector3f &p,
                                       const RowMajorMatrix3f &R,
                                       Eigen::Vector3f *result,
                                       RowMajorMatrixf<3, 3> *jac_p = nullptr,
                                       RowMajorMatrixf<3, 3> *jac_R = nullptr) {
        *result = R.transpose() * p;
        if (jac_p) {
            *jac_p = R.transpose();
        }
        if (jac_R) {
            *jac_R = Skew(*result);
        }
    }

    static inline void CenterWithJac(Eigen::Vector3f *center,
                                     const RowMajorMatrix3f &R,
                                     const Eigen::Vector3f &t,
                                     RowMajorMatrixf<3, 3> *jac_R = nullptr,
                                     RowMajorMatrixf<3, 3> *jac_t = nullptr) {
        *center = -R.transpose() * t;

        if (jac_R) {
            *jac_R = Skew(*center);
        }
        if (jac_t) {
            *jac_t = -R.transpose();
        }
    }

    inline CameraPose Inverse() const {
        return CameraPose(quat_conj(q), -Derotate(t));
    }

    static inline CameraPose FromRt(const RowMajorMatrix4f &mat) {
        return CameraPose(RowMajorMatrix3f(mat.block<3, 3>(0, 0)),
                          Eigen::Vector3f(mat.block<3, 1>(0, 3)));
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

    static inline RowMajorMatrix3f Skew(const Eigen::Vector3f &v) {
        // clang-format off
        return (RowMajorMatrix3f() <<
            0,     -v(2), v(1),
            v(2),  0,     -v(0),
            -v(1), v(0),  0
        ).finished();
        // clang-format on
    }
};

// YUCK FIXME
// Maybe use Sophus or define a proper SE3 class.
using Pose = CameraPose;

struct CameraState {
    CameraIntrinsics intrinsics;
    CameraPose pose;
};

// FIXME: Since we have our custom bundle adjuster, maybe create our custom
// bundle types?
struct BundleOptions {
    size_t max_iterations = 100;
    enum class LossType {
        TRIVIAL,
        TRUNCATED,
        HUBER,
        CAUCHY,
        // This is the TR-IRLS scheme from Le and Zach, 3DV 2021
        TRUNCATED_LE_ZACH
    } loss_type = LossType::HUBER;
    Float loss_scale = 1.0;
    Float gradient_tol = 1e-10;
    Float step_tol = 1e-8;
    Float initial_lambda = 1e-5;
    Float min_lambda = 1e-10;
    Float max_lambda = 1e10;
    bool verbose = false;
};

struct BundleStats {
    size_t iterations = 0;
    Float initial_cost;
    Float cost;
    Float lambda;
    size_t invalid_steps;
    Float step_norm;
    Float grad_norm;
};
