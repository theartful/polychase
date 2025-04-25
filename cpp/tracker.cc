#include "tracker.h"

#include <spdlog/spdlog.h>

#include <Eigen/Eigen>
#include <Eigen/LU>

#include "database.h"
#include "pnp/solvers.h"
#include "pnp/types.h"
#include "ray_casting.h"

std::optional<PnPResult> SolveFrame(const Database& database, const std::vector<CameraPose>& camera_poses,
                                    const RowMajorMatrix4f& model_matrix, const CameraIntrinsics& intrinsics,
                                    const int32_t first_frame, const int32_t frame_id,
                                    const AcceleratedMeshSptr& accel_mesh) {
    CHECK(frame_id > first_frame);

    std::vector<Eigen::Vector3f> object_points_worldspace;
    std::vector<Eigen::Vector2f> image_points;

    const std::vector<int32_t> flow_frames_ids = database.FindOpticalFlowsToImage(frame_id);

    for (int32_t flow_frame_id : flow_frames_ids) {
        CHECK(flow_frame_id != frame_id);

        // We don't have a solution for frames in the future yet at this point.
        if (flow_frame_id > frame_id) {
            continue;
        }
        // FIXME: We don't have access to frames before first_frame
        if (flow_frame_id < first_frame) {
            continue;
        }

        const KeypointsMatrix keypoints = database.ReadKeypoints(flow_frame_id);
        const ImagePairFlow flow = database.ReadImagePairFlow(flow_frame_id, frame_id);

        CHECK_EQ(flow.src_kps_indices.rows(), flow.tgt_kps.rows());
        const Eigen::Index num_matches = flow.src_kps_indices.rows();

        // TODO: benchmark / vectorize / parallelize
        for (int32_t i = 0; i < num_matches; i++) {
            const int32_t kp_idx = flow.src_kps_indices[i];
            const Eigen::Vector2f kp = keypoints.row(kp_idx);

            const SceneTransformations scene_transform = {
                .model_matrix = model_matrix,
                .view_matrix = camera_poses.at(flow_frame_id - first_frame).Rt4x4(),
                .intrinsics = intrinsics,
            };
            // Mabye collect rays, and bulk RayCast using embrees optimized rtcIntersect4/8/16
            const std::optional<RayHit> hit = RayCast(accel_mesh, scene_transform, kp);

            if (hit) {
                const Eigen::Vector3f intersection_point_worldspace =
                    model_matrix.block<3, 3>(0, 0) * hit->pos + model_matrix.block<3, 1>(0, 3);

                object_points_worldspace.push_back(intersection_point_worldspace);
                image_points.push_back(flow.tgt_kps.row(i));
            }
        }
    }

    if (object_points_worldspace.size() < 3) {
        return std::nullopt;
    }

    const Eigen::Map<const RowMajorMatrixX3f> object_points_eigen{
        reinterpret_cast<const float*>(object_points_worldspace.data()),
        static_cast<Eigen::Index>(object_points_worldspace.size()), 3};

    const Eigen::Map<const RowMajorMatrixX2f> image_points_eigen{reinterpret_cast<const float*>(image_points.data()),
                                                                 static_cast<Eigen::Index>(image_points.size()), 2};

    // The solution should be very close to the previous pose
    PnPResult result = {
        .camera =
            {
                .intrinsics = intrinsics,
                .pose = camera_poses.at(frame_id - first_frame - 1),
            },
        .bundle_stats = {},
        .ransac_stats = {},
    };

    RansacOptions ransac_opts = {};
    ransac_opts.score_initial_model = true;
    SolvePnPRansac(object_points_eigen, image_points_eigen, ransac_opts, {}, result);

    return result;
}

static Pose GetTargetPose(const SceneTransformations& scene_transform, const PnPResult& pnp_result,
                          TransformationType trans_type) {
    switch (trans_type) {
        case TransformationType::Camera: {
            return pnp_result.camera.pose;
        }
        case TransformationType::Model: {
            RowMajorMatrix4f model_matrix =
                scene_transform.view_matrix.inverse() * pnp_result.camera.pose.Rt4x4() * scene_transform.model_matrix;

            // Orthogonalize to remove scaling, since this matrix is [SR|t], where S is a diagonal scaling matrix
            model_matrix.col(0).normalize();
            model_matrix.col(1).normalize();
            model_matrix.col(2).normalize();

            return CameraPose::FromRt(model_matrix);
        }
        default: {
            throw std::runtime_error(fmt::format("Invalid trans_type value: {}", static_cast<int>(trans_type)));
        }
    }
}

bool TrackForwards(const std::string& database_path, int32_t frame_from, size_t num_frames,
                   const SceneTransformations& scene_transform, const AcceleratedMeshSptr& accel_mesh,
                   TransformationType trans_type, TrackingCallback callback) {
    SPDLOG_INFO("Tracking forwards {} frames from frame #{}", num_frames, frame_from);

    const Database database{database_path};

    // I'm assuming that we're solving for the view matrix, since it's easier to reason about.
    // So, both the model matrix, and the projection matrix are constant.
    std::vector<CameraPose> camera_poses;
    camera_poses.reserve(num_frames);

    camera_poses.push_back(CameraPose::FromRt(scene_transform.view_matrix));
    for (size_t i = 1; i < num_frames; i++) {
        int32_t frame_id = static_cast<int32_t>(frame_from + i);
        CHECK(frame_id > frame_from);

        const std::optional<PnPResult> maybe_result =
            SolveFrame(database, camera_poses, scene_transform.model_matrix, scene_transform.intrinsics, frame_from,
                       frame_id, accel_mesh);

        if (!maybe_result) {
            SPDLOG_INFO("Could not tarck to frame: {}", frame_id);
            return false;
        }

        const PnPResult& pnp_result = *maybe_result;
        FrameTrackingResult result = {
            .frame = frame_id,
            .pose = GetTargetPose(scene_transform, pnp_result, trans_type),
            .intrinsics = pnp_result.camera.intrinsics,
            .ransac_stats = pnp_result.ransac_stats,
            .bundle_stats = pnp_result.bundle_stats,
        };

        const bool ok = callback(result);
        if (!ok) {
            SPDLOG_INFO("User requested to stop");
            return false;
        }

        camera_poses.push_back(pnp_result.camera.pose);
    }

    return true;
}
