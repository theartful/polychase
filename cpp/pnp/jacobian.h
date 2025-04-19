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
    static constexpr size_t num_params = 6;

    CameraJacobianAccumulator(const ConstRefRowMajorMatrixX2f &points2D, const ConstRefRowMajorMatrixX3f &points3D,
                              const LossFunction &loss, const ResidualWeightVector &w = ResidualWeightVector())
        : x(points2D), X(points3D), loss_fn(loss), weights(w) {
        CHECK_EQ(x.rows(), X.rows());
    }

    float residual(const CameraState &state) const {
        float cost = 0;
        for (Eigen::Index i = 0; i < x.rows(); ++i) {
            const Eigen::Vector3f Z = state.pose.apply(X.row(i));
            // Note this assumes points that are behind the camera will stay behind the camera
            // during the optimization
            if (Z(2) < 0) continue;
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
    size_t accumulate(const CameraState &camera, RowMajorMatrixf<6, 6> &JtJ, RowMajorMatrixf<6, 1> &Jtr) const {
        const RowMajorMatrix3f R = camera.pose.R();
        RowMajorMatrix2f Jcam;
        Jcam.setIdentity();  // we initialize to identity here (this is for the calibrated case)
        size_t num_residuals = 0;

        for (Eigen::Index i = 0; i < x.rows(); ++i) {
            const Eigen::Vector2f p = x.row(i);
            const Eigen::Vector3f Z = X.row(i);
            const Eigen::Vector3f RtZ = R * Z + camera.pose.t;

            if (RtZ.z() < 0) {
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
            const RowMajorMatrixf<2, 3> pHnorm_pRtZ = (RowMajorMatrixf<2, 3>() <<
                1.0, 0.0, -RtZ_hnorm.x(),
                0.0, 1.0, -RtZ_hnorm.y()
            ).finished();
            // clang-format on

            const RowMajorMatrixf<2, 3> pz_pRtZ = Jcam * pHnorm_pRtZ;

            RowMajorMatrixf<2, 6> J;
            J.block<2, 3>(0, 0) = (pz_pRtZ * pRtZ_pR) / RtZ.z();
            J.block<2, 3>(0, 3) = pz_pRtZ / RtZ.z();

            JtJ.selfadjointView<Eigen::Lower>().rankUpdate(J.transpose(), weight);
            Jtr += J.transpose() * (weight * r);

            num_residuals++;
        }
        return num_residuals;
    }

    CameraState step(RowMajorMatrixf<6, 1> dp, const CameraState &camera) const {
        CameraPose pose_new;
        pose_new.q = quat_step_post(camera.pose.q, dp.block<3, 1>(0, 0));
        pose_new.t = camera.pose.t + dp.block<3, 1>(3, 0);
        return CameraState{
            .intrinsics = camera.intrinsics,
            .pose = pose_new,
        };
    }

   private:
    const ConstRefRowMajorMatrixX2f &x;
    const ConstRefRowMajorMatrixX3f &X;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
};
