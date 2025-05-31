#pragma once

#include <optional>

#include "eigen_typedefs.h"
#include "types.h"
#include "utils.h"

class PnPProblem {
   public:
    struct Parameters {
        CameraState cam;
        RowMajorMatrix3f R;  // Caching the rotation matrix so that we don't have to recalculate it.
    };

    static constexpr bool kShouldNormalize = false;
    static constexpr int kNumParams = 9;
    static constexpr int kResidualLength = 2;

    PnPProblem(const RefConstRowMajorMatrixX2f &points2D, const RefConstRowMajorMatrixX3f &points3D,
               const RefConstArrayXf &weights, bool optimize_focal_length = false,
               bool optimize_principal_point = false, CameraIntrinsics::Bounds bounds = {})
        : x(points2D),
          X(points3D),
          weights(weights),
          optimize_focal_length(optimize_focal_length && x.rows() > 3),
          optimize_principal_point(optimize_principal_point && x.rows() > 3),
          bounds(bounds) {
        CHECK_EQ(x.rows(), X.rows());
        CHECK(weights.rows() == 0 || weights.rows() == x.rows());
    }

    constexpr size_t NumParams() const { return 9; }
    size_t NumResiduals() const { return x.rows(); }

    Float Weight(size_t idx) {
        if (weights.rows() == 0) {
            return 1.0f;
        } else {
            return weights[idx];
        }
    }

    std::optional<Eigen::Vector2f> Evaluate(const Parameters &params, size_t idx) const {
        const Eigen::Vector3f Z = params.cam.pose.Apply(X.row(idx));
        if (params.cam.intrinsics.IsBehind(Z)) {
            return Eigen::Vector2f(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
        }
        const Eigen::Vector2f z = params.cam.intrinsics.Project(Z);
        return z - Eigen::Vector2f(x.row(idx));
    }

    bool EvaluateWithJacobian(const Parameters &params, size_t idx, RowMajorMatrixf<kResidualLength, kNumParams> &J,
                              Eigen::Vector2f &res) const {
        const CameraPose &pose = params.cam.pose;
        const CameraIntrinsics &intrin = params.cam.intrinsics;
        const RowMajorMatrix3f &R = params.R;

        const Eigen::Vector3f Z = X.row(idx);

        Eigen::Vector3f RtZ;
        RowMajorMatrixf<3, 3> dRtZ_dR;
        // dRtZ_dt is Identity, so we can drop it
        // RowMajorMatrixf<3, 3> dRtZ_dt;
        CameraPose::ApplyWithJac(Z, R, pose.t, &RtZ, nullptr, &dRtZ_dR, nullptr);

        Eigen::Vector2f z;
        RowMajorMatrixf<2, 3> dz_dRtZ;
        RowMajorMatrixf<2, 3> dz_dIntrin;
        intrin.ProjectWithJac(RtZ, &z, &dz_dRtZ, &dz_dIntrin);

        // Residual
        res = z - Eigen::Vector2f(x.row(idx));

        // Jacobian
        J.block<2, 3>(0, 0) = dz_dRtZ * dRtZ_dR;
        J.block<2, 3>(0, 3) = dz_dRtZ;

        J.block<2, 3>(0, 6) = dz_dIntrin;
        if (!optimize_focal_length) {
            J.block<2, 1>(0, 6).setZero();
        }
        if (!optimize_principal_point) {
            J.block<2, 2>(0, 7).setZero();
        }

        return true;
    }

    void Step(const Parameters &params, const RowMajorMatrixf<kNumParams, 1> &dp, Parameters &result) const {
        const CameraState &camera = params.cam;
        CameraState &camera_new = result.cam;

        camera_new.pose.q = quat_step_post(camera.pose.q, dp.block<3, 1>(0, 0));
        camera_new.pose.t = camera.pose.t + dp.block<3, 1>(3, 0);

        if (optimize_focal_length) {
            camera_new.intrinsics.fy = camera.intrinsics.fy + dp(6, 0);
            camera_new.intrinsics.fx = camera_new.intrinsics.fy * camera_new.intrinsics.aspect_ratio;

            camera_new.intrinsics.fy = std::clamp(camera_new.intrinsics.fy, bounds.f_low, bounds.f_high);
            camera_new.intrinsics.fx = std::clamp(camera_new.intrinsics.fx, bounds.f_low, bounds.f_high);
        }
        if (optimize_principal_point) {
            camera_new.intrinsics.cx = camera.intrinsics.cx + dp(7, 0);
            camera_new.intrinsics.cy = camera.intrinsics.cy + dp(8, 0);

            camera_new.intrinsics.cx = std::clamp(camera_new.intrinsics.cx, bounds.cx_low, bounds.cx_high);
            camera_new.intrinsics.cy = std::clamp(camera_new.intrinsics.cy, bounds.cy_low, bounds.cy_high);
        }

        result.R = result.cam.pose.R();
    }

   private:
    const RefConstRowMajorMatrixX2f x;
    const RefConstRowMajorMatrixX3f X;
    const RefConstArrayXf weights;

    const bool optimize_focal_length;
    const bool optimize_principal_point;

    const CameraIntrinsics::Bounds bounds;
};
