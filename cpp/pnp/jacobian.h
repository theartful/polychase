// Copyright (c) 2021, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <PoseLib/camera_pose.h>
#include <PoseLib/misc/colmap_models.h>
#include <PoseLib/robust/jacobian_impl.h>
#include <PoseLib/types.h>

#include "eigen_typedefs.h"
#include "types.h"
#include "utils.h"

template <typename LossFunction, typename ResidualWeightVector = poselib::UniformWeightVector>
class CameraJacobianAccumulator {
   public:
    using param_t = CameraState;
    static constexpr size_t num_params = 9;

    CameraJacobianAccumulator(const ConstRefRowMajorMatrixX2f &points2D, const ConstRefRowMajorMatrixX3f &points3D,
                              const LossFunction &loss, bool optimize_focal_length = false,
                              bool optimize_principal_point = false, float width = 2.0, float height = 2.0,
                              CameraConvention convention = CameraConvention::OpenGL,
                              const ResidualWeightVector &w = ResidualWeightVector())
        : x(points2D),
          X(points3D),
          loss_fn(loss),
          weights(w),
          optimize_focal_length(optimize_focal_length && x.rows() > 3),
          optimize_principal_point(optimize_principal_point && x.rows() > 3) {
        CHECK_EQ(x.rows(), X.rows());
        constexpr float min_fov = 15 * M_PI / 180;
        constexpr float max_fov = 160 * M_PI / 180;

        // TODO: constexpr
        const float min_tan_fov_2 = std::tan(min_fov / 2);
        const float max_tan_fov_2 = std::tan(max_fov / 2);

        if (convention == CameraConvention::OpenGL) {
            f_low = -(width / 2.0f) / min_tan_fov_2;
            f_high = -(width / 2.0f) / max_tan_fov_2;
        } else {
            f_high = (width / 2.0f) / min_tan_fov_2;
            f_low = (width / 2.0f) / max_tan_fov_2;
        }

        cx_low = 0.0f;
        cx_high = width;

        cy_low = 0.0f;
        cy_high = height;

        CHECK(f_low < f_high);
        CHECK(cx_low < cx_high);
        CHECK(cy_low < cy_high);
    }

    float residual(const CameraState &state) const {
        float cost = 0;
        for (Eigen::Index i = 0; i < x.rows(); ++i) {
            if (weights[i] == 0.0) {
                continue;
            }
            const Eigen::Vector3f Z = state.pose.Apply(X.row(i));
            if (state.intrinsics.IsBehind(Z)) {
                return std::numeric_limits<float>::max();
            }
            const Eigen::Vector2f p = state.intrinsics.Project(Z);
            const float r_squared = (p - Eigen::Vector2f(x.row(i))).squaredNorm();
            cost += weights[i] * loss_fn.loss(r_squared);
        }
        return cost;
    }

    RowMajorMatrix3f skew(const Eigen::Vector3f &v) const {
        // clang-format off
        return (RowMajorMatrix3f() <<
            0,     -v(2), v(1),
            v(2),  0,     -v(0),
            -v(1), v(0),  0
        ).finished();
        // clang-format on
    }

    // computes J.transpose() * J and J.transpose() * res
    // Only computes the lower half of JtJ
    size_t accumulate(const CameraState &camera, RowMajorMatrixf<9, 9> &JtJ, RowMajorMatrixf<9, 1> &Jtr) const {
        const RowMajorMatrix3f R = camera.pose.R();
        RowMajorMatrixf<2, 5> Jcam;
        Jcam.setIdentity();  // we initialize to identity here (this is for the calibrated case)
        size_t num_residuals = 0;

        for (Eigen::Index i = 0; i < x.rows(); ++i) {
            if (weights[i] == 0.0) {
                continue;
            }

            const Eigen::Vector2f p = x.row(i);
            const Eigen::Vector3f Z = X.row(i);
            const Eigen::Vector3f RtZ = R * Z + camera.pose.t;

            const Eigen::Vector2f RtZ_hnorm = RtZ.hnormalized();

            Eigen::Vector2f z;
            camera.intrinsics.ProjectWithJac(RtZ_hnorm, &z, &Jcam);

            const Eigen::Vector2f r = z - p;
            const float r_squared = r.squaredNorm();
            const float weight = weights[i] * loss_fn.weight(r_squared);

            const RowMajorMatrix3f pRtZ_pR = R * skew(-Z);

            // clang-format off
            // Should be divided by RtZ.z, but we will do the division later for numerical stability
            const RowMajorMatrixf<2, 3> pHnorm_pRtZ = (RowMajorMatrixf<2, 3>() <<
                1.0, 0.0, -RtZ_hnorm.x(),
                0.0, 1.0, -RtZ_hnorm.y()
            ).finished();
            // clang-format on

            const RowMajorMatrixf<2, 3> pz_pRtZ = Jcam.block<2, 2>(0, 0) * pHnorm_pRtZ;

            RowMajorMatrixf<2, 9> J;
            J.setZero();

            J.block<2, 3>(0, 0) = (pz_pRtZ * pRtZ_pR) / RtZ.z();
            J.block<2, 3>(0, 3) = pz_pRtZ / RtZ.z();

            if (optimize_focal_length) {
                J.block<2, 1>(0, 6) = Jcam.block<2, 1>(0, 2);
            }
            if (optimize_principal_point) {
                J.block<2, 2>(0, 7) = Jcam.block<2, 2>(0, 3);
            }

            JtJ.selfadjointView<Eigen::Lower>().rankUpdate(J.transpose(), weight);
            Jtr += J.transpose() * (weight * r);

            num_residuals++;
        }

        return num_residuals;
    }

    CameraState step(RowMajorMatrixf<9, 1> dp, const CameraState &camera) const {
        CameraState camera_new = camera;
        camera_new.pose.q = quat_step_post(camera.pose.q, dp.block<3, 1>(0, 0));
        camera_new.pose.t = camera.pose.t + dp.block<3, 1>(3, 0);

        if (optimize_focal_length) {
            camera_new.intrinsics.fy = camera.intrinsics.fy + dp(6, 0);
            camera_new.intrinsics.fx = camera_new.intrinsics.fy * camera_new.intrinsics.aspect_ratio;

            camera_new.intrinsics.fy = std::clamp(camera_new.intrinsics.fy, f_low, f_high);
            camera_new.intrinsics.fx = std::clamp(camera_new.intrinsics.fx, f_low, f_high);
        }
        if (optimize_principal_point) {
            camera_new.intrinsics.cx = camera.intrinsics.cx + dp(7, 0);
            camera_new.intrinsics.cy = camera.intrinsics.cy + dp(8, 0);

            camera_new.intrinsics.cx = std::clamp(camera_new.intrinsics.cx, cx_low, cx_high);
            camera_new.intrinsics.cy = std::clamp(camera_new.intrinsics.cy, cy_low, cy_high);
        }

        return camera_new;
    }

   private:
    const ConstRefRowMajorMatrixX2f &x;
    const ConstRefRowMajorMatrixX3f &X;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;

    const bool optimize_focal_length;
    const bool optimize_principal_point;

    float f_low;
    float f_high;
    float cx_low;
    float cx_high;
    float cy_low;
    float cy_high;
};

class CameraJacobian {
   public:
    using Parameters = CameraState;
    static constexpr int kNumParams = 9;
    static constexpr int kResidualLength = 2;

    CameraJacobian(const ConstRefRowMajorMatrixX2f &points2D, const ConstRefRowMajorMatrixX3f &points3D,
                   bool optimize_focal_length = false, bool optimize_principal_point = false, float width = 2.0,
                   float height = 2.0, CameraConvention convention = CameraConvention::OpenGL)
        : x(points2D),
          X(points3D),
          optimize_focal_length(optimize_focal_length && x.rows() > 3),
          optimize_principal_point(optimize_principal_point && x.rows() > 3) {
        CHECK_EQ(x.rows(), X.rows());
        constexpr float min_fov = 15 * M_PI / 180;
        constexpr float max_fov = 160 * M_PI / 180;

        // TODO: constexpr
        const float min_tan_fov_2 = std::tan(min_fov / 2);
        const float max_tan_fov_2 = std::tan(max_fov / 2);

        if (convention == CameraConvention::OpenGL) {
            f_low = -(width / 2.0f) / min_tan_fov_2;
            f_high = -(width / 2.0f) / max_tan_fov_2;
        } else {
            f_high = (width / 2.0f) / min_tan_fov_2;
            f_low = (width / 2.0f) / max_tan_fov_2;
        }

        cx_low = 0.0f;
        cx_high = width;

        cy_low = 0.0f;
        cy_high = height;

        CHECK(f_low < f_high);
        CHECK(cx_low < cx_high);
        CHECK(cy_low < cy_high);
    }

    size_t NumParams() const { return 9; }
    size_t NumResiduals() const { return x.rows(); }

    Eigen::Vector2f Evaluate(const CameraState &state, size_t idx) const {
        const Eigen::Vector3f Z = state.pose.Apply(X.row(idx));
        if (state.intrinsics.IsBehind(Z)) {
            return Eigen::Vector2f(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
        }
        const Eigen::Vector2f p = state.intrinsics.Project(Z);
        return p - Eigen::Vector2f(x.row(idx));
    }

    void EvaluateWithJacobian(const CameraState &camera, size_t idx, RowMajorMatrixf<2, 9> &J,
                              Eigen::Vector2f &r) const {
        const Eigen::Vector2f p = x.row(idx);
        const Eigen::Vector3f Z = X.row(idx);

        Eigen::Vector3f RtZ;
        RowMajorMatrixf<3, 3> dRtZ_dR;
        // dRtZ_dt is Identity, so we can drop it
        // RowMajorMatrixf<3, 3> dRtZ_dt;

        // FIXME: This makes me a little sad since we're not caching camera.pose.R(),
        // and this causes it to be recalculated for each residual.
        camera.pose.ApplyWithJac(Z, &RtZ, nullptr, &dRtZ_dR, nullptr);

        Eigen::Vector2f z;
        RowMajorMatrixf<2, 3> dp_dRtZ;
        RowMajorMatrixf<2, 3> dp_dIntrin;
        camera.intrinsics.ProjectWithJac(RtZ, &z, &dp_dRtZ, &dp_dIntrin);
        r = z - p;

        J.block<2, 3>(0, 0) = dp_dRtZ * dRtZ_dR;
        J.block<2, 3>(0, 3) = dp_dRtZ;

        J.block<2, 3>(0, 6) = dp_dIntrin;
        if (!optimize_focal_length) {
            J.block<2, 1>(0, 6).setZero();
        }
        if (!optimize_principal_point) {
            J.block<2, 2>(0, 7).setZero();
        }
    }

    CameraState Step(const CameraState &camera, const RowMajorMatrixf<9, 1> &dp) const {
        CameraState camera_new = camera;
        camera_new.pose.q = quat_step_post(camera.pose.q, dp.block<3, 1>(0, 0));
        camera_new.pose.t = camera.pose.t + dp.block<3, 1>(3, 0);

        if (optimize_focal_length) {
            camera_new.intrinsics.fy = camera.intrinsics.fy + dp(6, 0);
            camera_new.intrinsics.fx = camera_new.intrinsics.fy * camera_new.intrinsics.aspect_ratio;

            camera_new.intrinsics.fy = std::clamp(camera_new.intrinsics.fy, f_low, f_high);
            camera_new.intrinsics.fx = std::clamp(camera_new.intrinsics.fx, f_low, f_high);
        }
        if (optimize_principal_point) {
            camera_new.intrinsics.cx = camera.intrinsics.cx + dp(7, 0);
            camera_new.intrinsics.cy = camera.intrinsics.cy + dp(8, 0);

            camera_new.intrinsics.cx = std::clamp(camera_new.intrinsics.cx, cx_low, cx_high);
            camera_new.intrinsics.cy = std::clamp(camera_new.intrinsics.cy, cy_low, cy_high);
        }

        return camera_new;
    }

   private:
    const ConstRefRowMajorMatrixX2f &x;
    const ConstRefRowMajorMatrixX3f &X;

    const bool optimize_focal_length;
    const bool optimize_principal_point;

    float f_low;
    float f_high;
    float cx_low;
    float cx_high;
    float cy_low;
    float cy_high;
};
