#include "refiner.h"

#include <PoseLib/robust/jacobian_impl.h>
#include <fmt/chrono.h>
#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <Eigen/SparseCore>

#include "database.h"
#include "pnp/lev_marq.h"
#include "pnp/robust_loss.h"
#include "tracker.h"

class CachedDatabase {
   public:
    CachedDatabase(const Database& database, int32_t first_frame_id, size_t num_frames)
        : first_frame_id(first_frame_id), num_frames(num_frames) {
        SPDLOG_INFO("Loading database into memory...");
        // Read keypoints
        frames_keypoints.resize(num_frames);
        flows.reserve(num_frames * 8);

        std::vector<int32_t> ids;

        const int32_t last_frame_id_inclusive = first_frame_id + num_frames - 1;

        for (size_t idx = 0; idx < num_frames; idx++) {
            int32_t frame_id = first_frame_id + idx;
            database.ReadKeypoints(frame_id, frames_keypoints[idx]);

            ids.clear();
            database.FindOpticalFlowsFromImage(frame_id, ids);

            for (int32_t frame_id_to : ids) {
                if (frame_id_to < first_frame_id || frame_id_to > last_frame_id_inclusive) {
                    continue;
                }
                ImagePairFlow& flow = flows.emplace_back();
                database.ReadImagePairFlow(frame_id, frame_id_to, flow);
            }
        }
        SPDLOG_INFO("Done loading database into memory...");
    }

    inline size_t NumFlows() const { return flows.size(); }

    inline int32_t FirstFrameId() const { return first_frame_id; }

    inline int32_t LastFrameId() const { return first_frame_id + num_frames - 1; }

    const Keypoints& ReadKeypoints(int32_t frame_id) const {
        CHECK_GE(frame_id, FirstFrameId());
        CHECK_LE(frame_id, LastFrameId());

        return frames_keypoints[frame_id - first_frame_id];
    }

    const ImagePairFlow& ReadFlow(size_t idx) const { return flows[idx]; }

   private:
    int32_t first_frame_id;
    size_t num_frames;

    std::vector<Keypoints> frames_keypoints;
    std::vector<ImagePairFlow> flows;
};

class CameraTrajectoryRefineProblem {
   public:
    using Parameters = CameraTrajectory;

    static constexpr int kNumParams = Eigen::Dynamic;
    static constexpr int kResidualLength = 2;
    static constexpr bool kShouldNormalize = true;

    struct Bounds {
        Float f_low;
        Float f_high;
        Float cx_low;
        Float cx_high;
        Float cy_low;
        Float cy_high;
    };

    CameraTrajectoryRefineProblem(const CachedDatabase& cached_database, const AcceleratedMeshSptr& mesh,
                                  const RowMajorMatrix4f& model_matrix, const CameraTrajectory& initial_camera_traj,
                                  bool optimize_focal_length = false, bool optimize_principal_point = false,
                                  const Bounds& bounds = {})
        : cached_database(cached_database),
          mesh(mesh),
          model_matrix(model_matrix),
          model_matrix_inv(model_matrix.inverse()),
          optimize_focal_length(optimize_focal_length),
          optimize_principal_point(optimize_principal_point),
          bounds(bounds) {
        first_frame_id = initial_camera_traj.FirstFrame();
        num_cameras = initial_camera_traj.Count();
        // 6 parameters for extrinsics, and potentially 3 for intrinsics.
        // FIXME
        if (optimize_focal_length || optimize_principal_point) {
            num_params_per_camera = 9;
        } else {
            num_params_per_camera = 6;
        }
        num_params = num_params_per_camera * num_cameras;

        num_residuals = 0;
        const size_t num_flows = cached_database.NumFlows();
        for (size_t i = 0; i < num_flows; i++) {
            const size_t flow_residuals = cached_database.ReadFlow(i).tgt_kps.size();
            num_residuals += flow_residuals;
            accum_residuals.push_back(num_residuals);
        }
    }

    size_t NumParams() const { return num_params; }

    size_t NumResiduals(size_t edge_idx) const { return cached_database.ReadFlow(edge_idx).tgt_kps.size(); }

    size_t NumParamBlocks() const { return num_cameras; }

    size_t ParamBlockLength() const { return num_params_per_camera; }

    size_t NumEdges() const { return cached_database.NumFlows(); }

    std::pair<size_t, size_t> GetEdge(size_t edge_idx) const {
        const ImagePairFlow& flow = cached_database.ReadFlow(edge_idx);
        return {flow.image_id_from - first_frame_id, flow.image_id_to - first_frame_id};
    }

    std::optional<Eigen::Vector2f> Evaluate(const CameraTrajectory& traj, size_t flow_idx, size_t kp_idx) const {
        const ImagePairFlow& flow = cached_database.ReadFlow(flow_idx);
        const int32_t src_frame = flow.image_id_from;
        const int32_t tgt_frame = flow.image_id_to;

        const Keypoints& src_kps = cached_database.ReadKeypoints(src_frame);
        const KeypointsIndices& src_kps_indices = flow.src_kps_indices;
        const Keypoints& tgt_kps = flow.tgt_kps;

        const auto& maybe_src_camera = traj.Get(src_frame);
        CHECK(maybe_src_camera.has_value());
        const auto& maybe_tgt_camera = traj.Get(tgt_frame);
        CHECK(maybe_tgt_camera.has_value());

        const CameraState& src_camera = *maybe_src_camera;
        const CameraState& tgt_camera = *maybe_tgt_camera;

        CHECK_LT(kp_idx, src_kps_indices.size());
        CHECK_LT(src_kps_indices[kp_idx], src_kps.size());
        CHECK_LT(kp_idx, tgt_kps.size());

        const Eigen::Vector2f src_point = src_kps[src_kps_indices[kp_idx]];
        const Eigen::Vector2f tgt_point = tgt_kps[kp_idx];

        const Eigen::Vector3f dir_camspace = src_camera.intrinsics.Unproject(src_point);
        const Ray ray_worldspace = {
            .origin = src_camera.pose.Center(),
            .dir = src_camera.pose.Derotate(dir_camspace),
        };
        const Ray ray_objectspace = {
            .origin = (model_matrix_inv * ray_worldspace.origin.homogeneous()).hnormalized(),
            .dir = model_matrix_inv.block<3, 3>(0, 0) * ray_worldspace.dir,
        };

        const std::optional<RayHit> maybe_rayhit = RayCast(mesh, ray_objectspace.origin, ray_objectspace.dir);
        if (!maybe_rayhit.has_value()) {
            return std::nullopt;
        }

        const RayHit& rayhit = *maybe_rayhit;
        const Eigen::Vector3f point_world = (model_matrix * rayhit.pos.homogeneous()).hnormalized();
        const Eigen::Vector3f point_camera = tgt_camera.pose.Apply(point_world);

        if (tgt_camera.intrinsics.IsBehind(point_camera)) {
            return Eigen::Vector2f(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
        }

        return tgt_camera.intrinsics.Project(point_camera) - tgt_point;
    }

    bool EvaluateWithJacobian(const CameraTrajectory& traj, size_t flow_idx, size_t kp_idx,
                              RowMajorMatrixf<kResidualLength, Eigen::Dynamic>& J, Eigen::Vector2f& res) const {
        const ImagePairFlow& flow = cached_database.ReadFlow(flow_idx);
        const int32_t src_frame = flow.image_id_from;
        const int32_t tgt_frame = flow.image_id_to;

        const Keypoints& src_kps = cached_database.ReadKeypoints(src_frame);
        const KeypointsIndices& src_kps_indices = flow.src_kps_indices;
        const Keypoints& tgt_kps = flow.tgt_kps;

        const auto& maybe_src_camera = traj.Get(src_frame);
        CHECK(maybe_src_camera.has_value());
        const auto& maybe_tgt_camera = traj.Get(tgt_frame);
        CHECK(maybe_tgt_camera.has_value());

        const CameraState& src_camera = *maybe_src_camera;
        const CameraState& tgt_camera = *maybe_tgt_camera;

        CHECK_LT(kp_idx, src_kps_indices.size());
        CHECK_LT(src_kps_indices[kp_idx], src_kps.size());
        CHECK_LT(kp_idx, tgt_kps.size());

        const Eigen::Vector2f src_point = src_kps[src_kps_indices[kp_idx]];
        const Eigen::Vector2f tgt_point = tgt_kps[kp_idx];

        RowMajorMatrixf<3, 3> dDirCam_dIntrin;
        Eigen::Vector3f dirCam;
        src_camera.intrinsics.UnprojectWithJac(src_point, &dirCam, nullptr, &dDirCam_dIntrin);

        Eigen::Vector3f origin;
        RowMajorMatrixf<3, 3> dOrigin_dR;
        RowMajorMatrixf<3, 3> dOrigin_dt;
        src_camera.pose.CenterWithJac(&origin, &dOrigin_dR, &dOrigin_dt);

        Eigen::Vector3f dirWorld;
        RowMajorMatrixf<3, 3> dDirWorld_dDirCam;
        RowMajorMatrixf<3, 3> dDirWorld_dR;
        src_camera.pose.DerotateWithJac(dirCam, &dirWorld, &dDirWorld_dDirCam, &dDirWorld_dR);

        const Ray ray_worldspace = {
            .origin = origin,
            .dir = dirWorld,
        };
        const Ray ray_objectspace = {
            .origin = (model_matrix_inv * ray_worldspace.origin.homogeneous()).hnormalized(),
            .dir = model_matrix_inv.block<3, 3>(0, 0) * ray_worldspace.dir,
        };

        const std::optional<RayHit> maybe_rayhit = RayCast(mesh, ray_objectspace.origin, ray_objectspace.dir);
        if (!maybe_rayhit.has_value()) {
            return false;
        }

        const RayHit& rayhit = *maybe_rayhit;
        // fmt::println("rayhit triangle index = {}", rayhit.primitive_id);
        const Plane plane_worldspace = {
            .point = (model_matrix * rayhit.pos.homogeneous()).hnormalized(),
            .normal = model_matrix_inv.transpose().block<3, 3>(0, 0) * rayhit.normal,
        };

        Eigen::Vector3f X;
        RowMajorMatrixf<3, 3> dX_dOrigin;
        RowMajorMatrixf<3, 3> dX_dDirWorld;

        const bool ok = IntersectWithJac(ray_worldspace, plane_worldspace, &X, &dX_dOrigin, &dX_dDirWorld);
        CHECK(ok);

        Eigen::Vector3f XCam;
        RowMajorMatrixf<3, 3> dXCam_dX;
        RowMajorMatrixf<3, 3> dXCam_dR;
        // dXCam_dt is Identity, so we can drop it
        // RowMajorMatrixf<3, 3> dXCam_dt;
        tgt_camera.pose.ApplyWithJac(X, &XCam, &dXCam_dX, &dXCam_dR, nullptr);
        if (tgt_camera.intrinsics.IsBehind(XCam)) {
            return false;
        }

        Eigen::Vector2f p;
        RowMajorMatrixf<2, 3> dp_dXCam;
        RowMajorMatrixf<2, 3> dp_dIntrin;
        tgt_camera.intrinsics.ProjectWithJac(XCam, &p, &dp_dXCam, &dp_dIntrin);

        // Copmute residual
        res = p - tgt_point;

        // Compute jacboian
        const RowMajorMatrixf<2, 3> dp_dX = dp_dXCam * dXCam_dX;

        // Source camera
        J.block<2, 3>(0, 0) = dp_dX * (dX_dOrigin * dOrigin_dR + dX_dDirWorld * dDirWorld_dR);
        J.block<2, 3>(0, 3) = dp_dX * dX_dOrigin * dOrigin_dt;

        if (optimize_focal_length || optimize_principal_point) {
            J.block<2, 3>(0, 6) = dp_dX * dX_dDirWorld * dDirWorld_dDirCam * dDirCam_dIntrin;

            if (!optimize_focal_length) {
                J.block<2, 1>(0, 6).setZero();
            }
            if (!optimize_principal_point) {
                J.block<2, 2>(0, 7).setZero();
            }
        }

        // Target camera
        J.block<2, 3>(0, num_params_per_camera + 0) = dp_dXCam * dXCam_dR;
        J.block<2, 3>(0, num_params_per_camera + 3) = dp_dXCam;

        if (optimize_focal_length || optimize_principal_point) {
            J.block<2, 3>(0, num_params_per_camera + 6) = dp_dIntrin;

            if (!optimize_focal_length) {
                J.block<2, 1>(0, num_params_per_camera + 6).setZero();
            }
            if (!optimize_principal_point) {
                J.block<2, 2>(0, num_params_per_camera + 7).setZero();
            }
        }

        return true;
    }

    void Step(const CameraTrajectory& traj, const RowMajorMatrixf<kNumParams, 1>& dp, CameraTrajectory& result) const {
        if (result.FirstFrame() != traj.FirstFrame() || result.LastFrame() != traj.LastFrame()) {
            result = traj;
        }

        const size_t num_cameras = traj.Count();

        CHECK_EQ(dp.rows(), static_cast<Eigen::Index>(num_params_per_camera * num_cameras));
        CHECK_EQ(first_frame_id, traj.FirstFrame());

        // The first and last cameras are constant.
        // TODO: Maybe don't add them to the problem to begin with?
        for (size_t i = 1; i < num_cameras - 1; i++) {
            const int32_t frame = first_frame_id + i;
            const CameraState& camera = *traj.Get(frame);
            CameraState camera_new = camera;

            const Eigen::Index J_idx = num_params_per_camera * i;

            camera_new.pose.q = quat_step_post(camera.pose.q, dp.block<3, 1>(J_idx + 0, 0));
            camera_new.pose.t = camera.pose.t + dp.block<3, 1>(J_idx + 3, 0);

            if (optimize_focal_length) {
                camera_new.intrinsics.fy = camera.intrinsics.fy + dp(J_idx + 6, 0);
                camera_new.intrinsics.fx = camera_new.intrinsics.fy * camera_new.intrinsics.aspect_ratio;

                camera_new.intrinsics.fy = std::clamp(camera_new.intrinsics.fy, bounds.f_low, bounds.f_high);
                camera_new.intrinsics.fx = std::clamp(camera_new.intrinsics.fx, bounds.f_low, bounds.f_high);
            }
            if (optimize_principal_point) {
                camera_new.intrinsics.cx = camera.intrinsics.cx + dp(J_idx + 7, 0);
                camera_new.intrinsics.cy = camera.intrinsics.cy + dp(J_idx + 8, 0);

                camera_new.intrinsics.cx = std::clamp(camera_new.intrinsics.cx, bounds.cx_low, bounds.cx_high);
                camera_new.intrinsics.cy = std::clamp(camera_new.intrinsics.cy, bounds.cy_low, bounds.cy_high);
            }

            result.Set(frame, camera_new);
        }
    }

   private:
    std::pair<size_t, size_t> DecomposeResidualIndex(size_t residual_index) const {
        const size_t flow_idx = std::distance(
            accum_residuals.begin(), std::upper_bound(accum_residuals.begin(), accum_residuals.end(), residual_index));
        const size_t kp_idx = (flow_idx == 0) ? residual_index : residual_index - accum_residuals[flow_idx - 1];

        return {flow_idx, kp_idx};
    }

   private:
    const CachedDatabase& cached_database;
    const AcceleratedMeshSptr mesh;
    const RowMajorMatrix4f model_matrix;
    const RowMajorMatrix4f model_matrix_inv;
    const bool optimize_focal_length;
    const bool optimize_principal_point;
    const Bounds bounds;

    int32_t first_frame_id;
    size_t num_cameras;
    size_t num_params_per_camera;
    size_t num_params;
    size_t num_residuals;

    std::vector<uint32_t> accum_residuals;
};

static bool RefineTrajectory(const Database& database, CameraTrajectory& traj, const RowMajorMatrix4f& model_matrix,
                             AcceleratedMeshSptr mesh, bool optimize_focal_length, bool optimize_principal_point,
                             RefineTrajectoryCallback callback, const BundleOptions& bundle_opts) {
    for (int32_t frame = traj.FirstFrame() + 1; frame < traj.LastFrame(); frame++) {
        traj.Clear(frame);
    }
    RefineTrajectoryUpdate update = {};
    const int32_t middle_frame = (traj.FirstFrame() + traj.LastFrame()) / 2;

    update.message = fmt::format("Tracking forward from {} to {}", traj.FirstFrame(), middle_frame);
    callback(update);

    TrackCameraSequence(database, traj, traj.FirstFrame(), middle_frame, model_matrix, mesh, nullptr,
                        optimize_focal_length, optimize_principal_point, bundle_opts);

    update.message = fmt::format("Tracking backwards from {} to {}", traj.LastFrame(), middle_frame + 1);
    callback(update);

    TrackCameraSequence(database, traj, traj.LastFrame(), middle_frame + 1, model_matrix, mesh, nullptr,
                        optimize_focal_length, optimize_principal_point, bundle_opts);

    CachedDatabase cached_database{database, traj.FirstFrame(), traj.Count()};

    CameraTrajectoryRefineProblem problem{cached_database,         mesh, model_matrix, traj, optimize_focal_length,
                                          optimize_principal_point};

    auto cb = [&](const BundleStats& stats) {
        fmt::println("BundleStats:");
        fmt::println("\titerations = {}", stats.iterations);
        fmt::println("\tinitial_cost = {}", stats.initial_cost);
        fmt::println("\tcost = {}", stats.cost);
        fmt::println("\tlambda = {}", stats.lambda);
        fmt::println("\tinvalid_steps = {}", stats.invalid_steps);
        fmt::println("\tstep_norm = {}", stats.step_norm);
        fmt::println("\tgrad_norm = {}", stats.grad_norm);
        std::fflush(stdout);

        update.progress = static_cast<float>(stats.iterations) / bundle_opts.max_iterations;
        update.message = fmt::format("Cost: {:.04f} (Initial: {:.04f})", stats.cost, stats.initial_cost);
        update.stats = stats;

        return callback(update);
    };

    CauchyLoss loss(1.0);
    poselib::UniformWeightVector weights;
    auto stats = LevMarqSparseSolve(problem, loss, weights, &traj, bundle_opts, cb);

    cb(stats);

    return true;
}

bool RefineTrajectory(const std::string& database_path, CameraTrajectory& traj, const RowMajorMatrix4f& model_matrix,
                      AcceleratedMeshSptr mesh, bool optimize_focal_length, bool optimize_principal_point,
                      RefineTrajectoryCallback callback, BundleOptions bundle_opts) {
    Database database{database_path};
    optimize_focal_length = false;
    optimize_principal_point = false;
    return RefineTrajectory(database, traj, model_matrix, mesh, optimize_focal_length, optimize_principal_point,
                            callback, bundle_opts);
}
