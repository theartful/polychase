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
                              bool optimize_principal_point = false,
                              const ResidualWeightVector &w = ResidualWeightVector())
        : x(points2D),
          X(points3D),
          loss_fn(loss),
          weights(w),
          optimize_focal_length(optimize_focal_length && x.rows() > 3),
          optimize_principal_point(optimize_principal_point && x.rows() > 3) {
        CHECK_EQ(x.rows(), X.rows());
    }

    float residual(const CameraState &state) const {
        float cost = 0;
        for (Eigen::Index i = 0; i < x.rows(); ++i) {
            const Eigen::Vector3f Z = state.pose.apply(X.row(i));
            // Note this assumes points that are behind the camera will stay behind the camera
            // during the optimization
            if (state.intrinsics.is_behind(Z)) continue;
            const float inv_z = 1.0 / Z(2);
            const Eigen::Vector2f p = state.intrinsics.project({Z(0) * inv_z, Z(1) * inv_z});
            const float r0 = p(0) - x.row(i)(0);
            const float r1 = p(1) - x.row(i)(1);
            const float r_squared = r0 * r0 + r1 * r1;
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
            const Eigen::Vector2f p = x.row(i);
            const Eigen::Vector3f Z = X.row(i);
            const Eigen::Vector3f RtZ = R * Z + camera.pose.t;

            if (camera.intrinsics.is_behind(RtZ)) {
                continue;
            }

            const Eigen::Vector2f RtZ_hnorm = RtZ.hnormalized();

            Eigen::Vector2f z;
            camera.intrinsics.project_with_jac(RtZ_hnorm, &z, &Jcam);

            const Eigen::Vector2f r = z - p;
            const float r_squared = r.squaredNorm();
            const float weight = weights[i] * loss_fn.weight(r_squared);

            if (weight == 0.0) {
                continue;
            }

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
            J.block<2, 3>(0, 0) = (pz_pRtZ * pRtZ_pR) / RtZ.z();
            J.block<2, 3>(0, 3) = pz_pRtZ / RtZ.z();

            if (optimize_focal_length) {
                J.block<2, 1>(0, 6) = Jcam.block<2, 1>(0, 2);
            } else {
                J.block<2, 1>(0, 6).setZero();
            }
            if (optimize_principal_point) {
                J.block<2, 2>(0, 7) = Jcam.block<2, 2>(0, 3);
            } else {
                J.block<2, 1>(0, 6).setZero();
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

        camera_new.intrinsics = camera.intrinsics;
        if (optimize_focal_length) {
            camera_new.intrinsics.fy += dp(6, 0);
            camera_new.intrinsics.fx = camera_new.intrinsics.fy * camera_new.intrinsics.aspect_ratio;
        }
        if (optimize_principal_point) {
            camera_new.intrinsics.cx += dp(7, 0);
            camera_new.intrinsics.cy += dp(8, 0);
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
};
