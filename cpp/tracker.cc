#include "tracker.h"

#include <spdlog/spdlog.h>

#include <Eigen/Eigen>
#include <Eigen/LU>
#include <vector>

#include "database.h"
#include "pnp/solvers.h"
#include "pnp/types.h"
#include "ray_casting.h"

struct CameraStateBuffer {
    std::vector<std::optional<CameraState>> states;
    int32_t first_frame_id = 0;

    CameraStateBuffer(int32_t first_frame_id, size_t count) : states(count), first_frame_id(first_frame_id) {}

    size_t Index(int32_t frame_id) const { return static_cast<size_t>(frame_id - first_frame_id); }

    bool IsValidFrame(int32_t frame_id) const { return Index(frame_id) < Count(); }

    bool IsFrameFilled(int32_t frame_id) const { return IsValidFrame(frame_id) && Get(frame_id).has_value(); }

    const std::optional<CameraState>& Get(int32_t frame_id) const {
        const size_t index = Index(frame_id);
        CHECK(index < Count());

        return states[index];
    }

    void Set(int32_t frame_id, const CameraState& state) {
        const size_t index = Index(frame_id);
        CHECK(index < Count());
        CHECK_EQ(states[index], std::nullopt);

        states[index] = state;
    }

    size_t Count() const { return states.size(); }
};

std::optional<PnPResult> SolveFrame(const Database& database, const CameraStateBuffer& camera_states,
                                    const RowMajorMatrix4f& model_matrix, const int32_t frame_id,
                                    const AcceleratedMeshSptr& accel_mesh, bool optimize_focal_length,
                                    bool optimize_principal_point) {
    std::vector<Eigen::Vector3f> object_points_worldspace;
    std::vector<Eigen::Vector2f> image_points;

    const std::vector<int32_t> flow_frames_ids = database.FindOpticalFlowsToImage(frame_id);

    for (int32_t flow_frame_id : flow_frames_ids) {
        CHECK(flow_frame_id != frame_id);

        if (!camera_states.IsFrameFilled(flow_frame_id)) {
            continue;
        }

        const KeypointsMatrix keypoints = database.ReadKeypoints(flow_frame_id);
        const ImagePairFlow flow = database.ReadImagePairFlow(flow_frame_id, frame_id);

        CHECK_EQ(flow.src_kps_indices.rows(), flow.tgt_kps.rows());
        const Eigen::Index num_matches = flow.src_kps_indices.rows();

        const std::optional<CameraState>& maybe_camera_state = camera_states.Get(flow_frame_id);
        CHECK(maybe_camera_state.has_value());
        const CameraState& camera_state = *maybe_camera_state;

        // TODO: benchmark / vectorize / parallelize
        for (int32_t i = 0; i < num_matches; i++) {
            const int32_t kp_idx = flow.src_kps_indices[i];
            const Eigen::Vector2f kp = keypoints.row(kp_idx);

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

    PnPResult result;
    // The solution should be very close to the previous/next pose
    if (camera_states.IsFrameFilled(frame_id - 1)) {
        result.camera = *camera_states.Get(frame_id - 1);
    } else if (camera_states.IsFrameFilled(frame_id + 1)) {
        result.camera = *camera_states.Get(frame_id + 1);
    }

    RansacOptions ransac_opts = {};
    ransac_opts.score_initial_model = true;

    SolvePnPRansac(object_points_eigen, image_points_eigen, ransac_opts, {}, optimize_focal_length,
                   optimize_principal_point, result);

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

    CameraStateBuffer camera_states{first_frame, num_frames};

    // Sanity checks
    CHECK(camera_states.IsValidFrame(first_frame));
    CHECK(camera_states.IsValidFrame(last_frame));
    CHECK(!camera_states.IsValidFrame(first_frame - 1));
    CHECK(!camera_states.IsValidFrame(last_frame + 1));

    camera_states.Set(frame_from,
                      CameraState{scene_transform.intrinsics, CameraPose::FromRt(scene_transform.view_matrix)});

    for (int32_t frame_id = frame_from + dir; frame_id != frame_to_inclusive + dir; frame_id += dir) {
        SPDLOG_DEBUG("Tracking frame {}", frame_id);

        const std::optional<PnPResult> maybe_result =
            SolveFrame(database, camera_states, scene_transform.model_matrix, frame_id, accel_mesh,
                       optimize_focal_length, optimize_principal_point);

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

        camera_states.Set(frame_id, pnp_result.camera);
    }

    SPDLOG_INFO("Successfully tracked backwards {} frames starting after frame #{}", num_frames - 1, frame_from);
    return true;
}
