#pragma once

#include "eigen_typedefs.h"
#include "types.h"
#include "utils.h"

class PnPProblem {
   public:
    struct Parameters {
        CameraState cam;
        RowMajorMatrix3f R;  // Caching the rotation matrix so that we don't have to recalculate it.
    };

    struct Bounds {
        Float f_low;
        Float f_high;
        Float cx_low;
        Float cx_high;
        Float cy_low;
        Float cy_high;
    };

    static constexpr int kNumParams = 9;
    static constexpr int kResidualLength = 2;

    PnPProblem(const ConstRefRowMajorMatrixX2f &points2D, const ConstRefRowMajorMatrixX3f &points3D,
               bool optimize_focal_length = false, bool optimize_principal_point = false, Bounds bounds = {})
        : x(points2D),
          X(points3D),
          optimize_focal_length(optimize_focal_length && x.rows() > 3),
          optimize_principal_point(optimize_principal_point && x.rows() > 3),
          bounds(bounds) {
        CHECK_EQ(x.rows(), X.rows());
    }

    size_t NumParams() const { return 9; }
    size_t NumResiduals() const { return x.rows(); }

    Eigen::Vector2f Evaluate(const Parameters &params, size_t idx) const {
        const Eigen::Vector3f Z = params.cam.pose.Apply(X.row(idx));
        if (params.cam.intrinsics.IsBehind(Z)) {
            return Eigen::Vector2f(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
        }
        const Eigen::Vector2f z = params.cam.intrinsics.Project(Z);
        return z - Eigen::Vector2f(x.row(idx));
    }

    void EvaluateWithJacobian(const Parameters &params, size_t idx, RowMajorMatrixf<kResidualLength, kNumParams> &J,
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
    }

    Parameters Step(const Parameters &params, const RowMajorMatrixf<kNumParams, 1> &dp) const {
        const CameraState &camera = params.cam;
        CameraState camera_new = params.cam;

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

        return Parameters{
            .cam = camera_new,
            // Cache the rotation matrix
            .R = camera_new.pose.R(),
        };
    }

   private:
    const ConstRefRowMajorMatrixX2f &x;
    const ConstRefRowMajorMatrixX3f &X;

    const bool optimize_focal_length;
    const bool optimize_principal_point;

    const Bounds bounds;
};
