#pragma once

#include <PoseLib/robust/jacobian_impl.h>

#include <vector>

#include "geometry.h"
#include "pnp/types.h"
#include "ray_casting.h"

struct CameraPair {
    CameraState camera_src;
    CameraState camera_tgt;
};

template <typename LossFunction, typename ResidualWeightVector = poselib::UniformWeightVector>
class DifferentPnPJacobianAccumulator {
   public:
    using param_t = CameraPair;
    static constexpr size_t num_params = 18;

    DifferentPnPJacobianAccumulator(const RefConstRowMajorMatrixX2f& points2D_src,
                                    const RefConstRowMajorMatrixX2f& points2D_tgt, const std::vector<Plane>& planes,
                                    bool optimize_focal_length, bool optimize_principal_point, LossFunction loss_fn,
                                    ResidualWeightVector weights)
        : points2D_src(points2D_src),
          points2D_tgt(points2D_tgt),
          planes(planes),
          optimize_focal_length(optimize_focal_length),
          optimize_principal_point(optimize_principal_point),
          loss_fn(loss_fn),
          weights(weights) {
        CHECK_EQ(points2D_src.rows(), points2D_tgt.rows());
        CHECK_EQ(points2D_src.rows(), static_cast<Eigen::Index>(planes.size()));
    }

    float residual(const CameraPair& camera_pair) const {
        const CameraState& camera_src = camera_pair.camera_src;
        const CameraState& camera_tgt = camera_pair.camera_tgt;

        float cost = 0;
        for (Eigen::Index i = 0; i < points2D_src.rows(); ++i) {
            if (weights[i] == 0.0) {
                continue;
            }
            const Eigen::Vector2f point_tgt = points2D_tgt.row(i);
            const Eigen::Vector2f point_src = points2D_src.row(i);
            const Plane& plane = planes[i];

            const Eigen::Vector3f dir = camera_src.intrinsics.Unproject(point_src);
            const Ray ray = {
                .origin = camera_src.pose.Center(),
                .dir = camera_src.pose.Derotate(dir),
            };
            // FIXME
            const std::optional<Eigen::Vector3f> maybe_point_world = *Intersect(ray, plane);
            if (!maybe_point_world) {
                continue;
            }
            const Eigen::Vector3f& point_world = *maybe_point_world;

            const Eigen::Vector3f point_camera = camera_tgt.pose.Apply(point_world);

            if (camera_tgt.intrinsics.IsBehind(point_camera)) {
                return std::numeric_limits<float>::max();
            }

            const Eigen::Vector2f p = camera_tgt.intrinsics.Project(point_camera);

            const float r_squared = (p - point_tgt).squaredNorm();
            cost += weights[i] * loss_fn.Loss(r_squared);
        }
        return cost;
    }

    size_t accumulate(const CameraPair& camera_pair, RowMajorMatrixf<18, 18>& JtJ, RowMajorMatrixf<18, 1>& Jtr) const {
        size_t num_residuals = 0;

        const CameraState& camera_src = camera_pair.camera_src;
        const CameraState& camera_tgt = camera_pair.camera_tgt;

        const RowMajorMatrix3f camera_tgt_R = camera_tgt.pose.R();
        const Eigen::Vector3f& camera_tgt_t = camera_tgt.pose.t;

        const RowMajorMatrix3f camera_src_R = camera_src.pose.R();
        const Eigen::Vector3f& camera_src_t = camera_src.pose.t;

        for (Eigen::Index i = 0; i < points2D_src.rows(); i++) {
            if (weights[i] == 0.0) {
                continue;
            }

            const Eigen::Vector2f point_tgt = points2D_tgt.row(i);
            const Eigen::Vector2f point_src = points2D_src.row(i);
            const Plane& plane = planes[i];

            RowMajorMatrixf<3, 3> dDirCam_dIntrin;
            Eigen::Vector3f dirCam;
            camera_src.intrinsics.UnprojectWithJac(point_src, &dirCam, nullptr, &dDirCam_dIntrin);

            Eigen::Vector3f origin;
            RowMajorMatrixf<3, 3> dOrigin_dR;
            RowMajorMatrixf<3, 3> dOrigin_dt;
            CameraPose::CenterWithJac(&origin, camera_src_R, camera_src_t, &dOrigin_dR, &dOrigin_dt);

            Eigen::Vector3f dirWorld;
            RowMajorMatrixf<3, 3> dDirWorld_dDirCam;
            RowMajorMatrixf<3, 3> dDirWorld_dR;
            CameraPose::DerotateWithJac(dirCam, camera_src_R, &dirWorld, &dDirWorld_dDirCam, &dDirWorld_dR);

            const Ray ray = {
                .origin = origin,
                .dir = dirWorld,
            };

            Eigen::Vector3f X;
            RowMajorMatrixf<3, 3> dX_dOrigin;
            RowMajorMatrixf<3, 3> dX_dDirWorld;
            const bool ok = IntersectWithJac(ray, plane, &X, &dX_dOrigin, &dX_dDirWorld);
            CHECK(ok);

            Eigen::Vector3f XCam;
            RowMajorMatrixf<3, 3> dXCam_dX;
            RowMajorMatrixf<3, 3> dXCam_dR;
            // dXCam_dt is Identity, so we can drop it
            // RowMajorMatrixf<3, 3> dXCam_dt;
            CameraPose::ApplyWithJac(X, camera_tgt_R, camera_tgt_t, &XCam, &dXCam_dX, &dXCam_dR, nullptr);

            Eigen::Vector2f p;
            RowMajorMatrixf<2, 3> dp_dXCam;
            RowMajorMatrixf<2, 3> dp_dIntrin;
            camera_tgt.intrinsics.ProjectWithJac(XCam, &p, &dp_dXCam, &dp_dIntrin);

            const Eigen::Vector2f r = p - point_tgt;
            const float r_squared = r.squaredNorm();
            const float weight = weights[i] * loss_fn.Weight(r_squared);
            if (weight == 0.0f) continue;

            const RowMajorMatrixf<2, 3> dp_dX = dp_dXCam * dXCam_dX;

            RowMajorMatrixf<2, 18> J;
            J.setZero();

            // Source camera
            J.block<2, 3>(0, 0) = dp_dX * (dX_dOrigin * dOrigin_dR + dX_dDirWorld * dDirWorld_dR);
            J.block<2, 3>(0, 3) = dp_dX * dX_dOrigin * dOrigin_dt;

            // Target camera
            J.block<2, 3>(0, 0 + 9) = dp_dXCam * dXCam_dR;
            J.block<2, 3>(0, 3 + 9) = dp_dXCam;

            if (optimize_focal_length | optimize_principal_point) {
                J.block<2, 3>(0, 6) = dp_dX * dX_dDirWorld * dDirWorld_dDirCam * dDirCam_dIntrin;
                J.block<2, 3>(0, 6 + 9) = dp_dIntrin;

                if (!optimize_focal_length) {
                    J.block<2, 1>(0, 6).setZero();
                    J.block<2, 1>(0, 6 + 9).setZero();
                }
                if (!optimize_principal_point) {
                    J.block<2, 2>(0, 7).setZero();
                    J.block<2, 2>(0, 7 + 9).setZero();
                }
            }

            JtJ.selfadjointView<Eigen::Lower>().rankUpdate(J.transpose(), weight);
            Jtr += J.transpose() * (weight * r);

            num_residuals++;
        }

        return num_residuals;
    }

    CameraPair step(RowMajorMatrixf<18, 1> dp, const CameraPair& camera_pair) const {
        const CameraState& camera_src = camera_pair.camera_src;
        const CameraState& camera_tgt = camera_pair.camera_tgt;

        CameraPair camera_pair_new = camera_pair;
        CameraState& camera_src_new = camera_pair_new.camera_src;
        CameraState& camera_tgt_new = camera_pair_new.camera_tgt;

        camera_src_new.pose.q = quat_step_post(camera_src.pose.q, dp.block<3, 1>(0, 0));
        camera_src_new.pose.t = camera_src.pose.t + dp.block<3, 1>(3, 0);

        camera_tgt_new.pose.q = quat_step_post(camera_tgt.pose.q, dp.block<3, 1>(0 + 9, 0));
        camera_tgt_new.pose.t = camera_tgt.pose.t + dp.block<3, 1>(3 + 9, 0);

        if (optimize_focal_length) {
            camera_src_new.intrinsics.fy = camera_src.intrinsics.fy + dp(6, 0);
            camera_src_new.intrinsics.fx = camera_src_new.intrinsics.fy * camera_src_new.intrinsics.aspect_ratio;

            camera_tgt_new.intrinsics.fy = camera_tgt.intrinsics.fy + dp(6 + 9, 0);
            camera_tgt_new.intrinsics.fx = camera_tgt_new.intrinsics.fy * camera_tgt_new.intrinsics.aspect_ratio;
        }
        if (optimize_principal_point) {
            camera_src_new.intrinsics.cx = camera_src.intrinsics.cx + dp(7, 0);
            camera_src_new.intrinsics.cy = camera_src.intrinsics.cy + dp(8, 0);

            camera_tgt_new.intrinsics.cx = camera_tgt.intrinsics.cx + dp(7 + 9, 0);
            camera_tgt_new.intrinsics.cy = camera_tgt.intrinsics.cy + dp(8 + 9, 0);
        }

        return camera_pair_new;
    }

   private:
    const RefConstRowMajorMatrixX2f points2D_src;
    const RefConstRowMajorMatrixX2f points2D_tgt;
    const std::vector<Plane>& planes;

    const bool optimize_focal_length;
    const bool optimize_principal_point;

    const LossFunction& loss_fn;
    const ResidualWeightVector& weights;
};
