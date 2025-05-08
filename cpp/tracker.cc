#include "tracker.h"

#include <spdlog/spdlog.h>

#include <Eigen/Eigen>
#include <Eigen/LU>
#include <vector>

#include "camera_trajectory.h"
#include "database.h"
#include "pnp/solvers.h"
#include "pnp/types.h"
#include "ray_casting.h"

struct SolveFrameCache {
    std::vector<Eigen::Vector3f> object_points_worldspace;
    std::vector<Eigen::Vector2f> image_points;
    std::vector<int32_t> flow_frames_ids;
    Keypoints keypoints;
    ImagePairFlow flow;
    SolvePnPRansacCache pnp_cache;

    void Clear() {
        object_points_worldspace.clear();
        image_points.clear();
        flow_frames_ids.clear();
        keypoints.clear();
        flow.Clear();
        pnp_cache.Clear();
    }
};

static std::optional<PnPResult> SolveFrame(const Database& database, const CameraTrajectory& camera_traj,
                                           const RowMajorMatrix4f& model_matrix, const int32_t frame_id,
                                           const AcceleratedMeshSptr& accel_mesh, bool optimize_focal_length,
                                           bool optimize_principal_point, SolveFrameCache& cache) {
    cache.Clear();
    database.FindOpticalFlowsToImage(frame_id, cache.flow_frames_ids);

    for (int32_t flow_frame_id : cache.flow_frames_ids) {
        CHECK(flow_frame_id != frame_id);

        if (!camera_traj.IsFrameFilled(flow_frame_id)) {
            continue;
        }

        database.ReadKeypoints(flow_frame_id, cache.keypoints);
        database.ReadImagePairFlow(flow_frame_id, frame_id, cache.flow);

        CHECK_EQ(cache.flow.src_kps_indices.size(), cache.flow.tgt_kps.size());
        const size_t num_matches = cache.flow.src_kps_indices.size();

        const std::optional<CameraState>& maybe_camera_state = camera_traj.Get(flow_frame_id);
        CHECK(maybe_camera_state.has_value());
        const CameraState& camera_state = *maybe_camera_state;

        // TODO: benchmark / vectorize / parallelize
        for (size_t i = 0; i < num_matches; i++) {
            const uint32_t kp_idx = cache.flow.src_kps_indices[i];
            const Eigen::Vector2f kp = cache.keypoints[kp_idx];

            const SceneTransformations scene_transform = {
                .model_matrix = model_matrix,
                .view_matrix = camera_state.pose.Rt4x4(),
                .intrinsics = camera_state.intrinsics,
            };
            // Mabye collect rays, and bulk RayCast using embrees optimized rtcIntersect4/8/16
            const std::optional<RayHit> hit = RayCast(accel_mesh, scene_transform, kp);

            if (hit) {
                const Eigen::Vector3f intersection_point_worldspace =
                    model_matrix.block<3, 3>(0, 0) * hit->pos + model_matrix.block<3, 1>(0, 3);

                cache.object_points_worldspace.push_back(intersection_point_worldspace);
                cache.image_points.push_back(cache.flow.tgt_kps[i]);
            }
        }
    }

    if (cache.object_points_worldspace.size() < 3) {
        return std::nullopt;
    }

    const Eigen::Map<const RowMajorMatrixX3f> object_points_eigen{
        reinterpret_cast<const float*>(cache.object_points_worldspace.data()),
        static_cast<Eigen::Index>(cache.object_points_worldspace.size()), 3};

    const Eigen::Map<const RowMajorMatrixX2f> image_points_eigen{
        reinterpret_cast<const float*>(cache.image_points.data()),
        static_cast<Eigen::Index>(cache.image_points.size()), 2};

    PnPResult result;
    // The solution should be very close to the previous/next pose
    if (camera_traj.IsFrameFilled(frame_id - 1)) {
        result.camera = *camera_traj.Get(frame_id - 1);
    } else if (camera_traj.IsFrameFilled(frame_id + 1)) {
        result.camera = *camera_traj.Get(frame_id + 1);
    }

    RansacOptions ransac_opts = {};
    ransac_opts.score_initial_model = true;

    SolvePnPRansac(object_points_eigen, image_points_eigen, ransac_opts, {}, optimize_focal_length,
                   optimize_principal_point, cache.pnp_cache, result);

    return result;
}

static Pose GetTargetPose(const SceneTransformations& scene_transform, const PnPResult& pnp_result,
                          TransformationType trans_type) {
    switch (trans_type) {
        case TransformationType::Camera:
            return pnp_result.camera.pose;
        case TransformationType::Model:
            return CameraPose::FromSRt(scene_transform.view_matrix.inverse() * pnp_result.camera.pose.Rt4x4() *
                                       scene_transform.model_matrix);
        default:
            throw std::runtime_error(fmt::format("Invalid trans_type value: {}", static_cast<int>(trans_type)));
    }
}

bool TrackSequence(const std::string& database_path, int32_t frame_from, int32_t frame_to_inclusive,
                   const SceneTransformations& scene_transform, const AcceleratedMeshSptr& accel_mesh,
                   TransformationType trans_type, TrackingCallback callback, bool optimize_focal_length,
                   bool optimize_principal_point) {
    SPDLOG_INFO("Tracking from frame #{} to frame #{}", frame_from, frame_to_inclusive);

    const Database database{database_path};

    const int32_t first_frame = std::min(frame_from, frame_to_inclusive);
    const int32_t last_frame = std::max(frame_from, frame_to_inclusive);
    const int32_t dir = (frame_from < frame_to_inclusive) ? 1 : -1;
    const size_t num_frames = static_cast<size_t>(last_frame - first_frame + 1);

    CameraTrajectory camera_traj{first_frame, num_frames};

    // Sanity checks
    CHECK(camera_traj.IsValidFrame(first_frame));
    CHECK(camera_traj.IsValidFrame(last_frame));
    CHECK(!camera_traj.IsValidFrame(first_frame - 1));
    CHECK(!camera_traj.IsValidFrame(last_frame + 1));

    camera_traj.Set(frame_from,
                    CameraState{scene_transform.intrinsics, CameraPose::FromRt(scene_transform.view_matrix)});

    // To avoid unnecessary re-alloactions
    SolveFrameCache cache;

    for (int32_t frame_id = frame_from + dir; frame_id != frame_to_inclusive + dir; frame_id += dir) {
        SPDLOG_DEBUG("Tracking frame {}", frame_id);

        const std::optional<PnPResult> maybe_result =
            SolveFrame(database, camera_traj, scene_transform.model_matrix, frame_id, accel_mesh,
                       optimize_focal_length, optimize_principal_point, cache);

        if (!maybe_result) {
            SPDLOG_WARN("Could not track backwards to frame: {}. Stopping tracking.", frame_id);
            return false;
        }

        const PnPResult& pnp_result = *maybe_result;

        const FrameTrackingResult result = {
            .frame = frame_id,
            .pose = GetTargetPose(scene_transform, pnp_result, trans_type),
            .intrinsics = pnp_result.camera.intrinsics,
            .ransac_stats = pnp_result.ransac_stats,
            .bundle_stats = pnp_result.bundle_stats,
        };

        const bool ok = callback(result);
        if (!ok) {
            SPDLOG_INFO("User requested to stop at frame {}", frame_id);
            return false;
        }

        camera_traj.Set(frame_id, pnp_result.camera);
    }

    SPDLOG_INFO("Successfully tracked backwards {} frames starting after frame #{}", num_frames - 1, frame_from);
    return true;
}
