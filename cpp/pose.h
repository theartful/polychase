#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "eigen_typedefs.h"

// FIXME: Maybe use Sophus or define a proper SE3 class.
struct Pose {
    Eigen::Quaternionf q;
    Eigen::Vector3f t;

    // Constructors (Defaults to identity)
    Pose() : q(Eigen::Quaternionf::Identity()), t(0.0, 0.0, 0.0) {}
    Pose(const Eigen::Quaternionf &qq, const Eigen::Vector3f &tt)
        : q(qq), t(tt) {}
    Pose(const RowMajorMatrix3f &R, const Eigen::Vector3f &tt) : q(R), t(tt) {}

    // Helper functions
    inline RowMajorMatrix3f R() const { return q.toRotationMatrix(); }
    inline RowMajorMatrix34f Rt() const {
        RowMajorMatrix34f tmp;
        tmp.block<3, 3>(0, 0) = R();
        tmp.col(3) = t;
        return tmp;
    }
    inline RowMajorMatrix4f Rt4x4() const {
        RowMajorMatrix4f tmp;
        tmp.block<3, 3>(0, 0) = R();
        tmp.block<3, 1>(0, 3) = t;
        tmp(3, 3) = 1.0f;
        tmp(3, 0) = 0.0f;
        tmp(3, 1) = 0.0f;
        tmp(3, 2) = 0.0f;
        return tmp;
    }
    inline Eigen::Vector3f Rotate(const Eigen::Vector3f &p) const {
        return q * p;
    }
    inline Eigen::Vector3f Derotate(const Eigen::Vector3f &p) const {
        return q.conjugate() * p;
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

    inline Pose Inverse() const { return Pose(q.conjugate(), -Derotate(t)); }

    static inline Pose FromRt(const RowMajorMatrix4f &mat) {
        return Pose(RowMajorMatrix3f(mat.block<3, 3>(0, 0)),
                    Eigen::Vector3f(mat.block<3, 1>(0, 3)));
    }

    static inline Pose FromSRt(const RowMajorMatrix4f &mat) {
        RowMajorMatrix3f mat_copy = mat.block<3, 3>(0, 0);
        mat_copy.col(0).normalize();
        mat_copy.col(1).normalize();
        mat_copy.col(2).normalize();
        return Pose(mat_copy, Eigen::Vector3f(mat.block<3, 1>(0, 3)));
    }

    static inline Pose FromRt(const RowMajorMatrix34f &mat) {
        return Pose(RowMajorMatrix3f(mat.template block<3, 3>(0, 0)),
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
