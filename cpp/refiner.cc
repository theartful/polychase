// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

#include "refiner.h"

#include <fmt/chrono.h>
#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <Eigen/SparseCore>
#include <atomic>

#include "database.h"
#include "pnp/lev_marq.h"
#include "pnp/robust_loss.h"

static Bbox2 TransformBbox(const Bbox3& bbox,
                           const RowMajorMatrix4f& transform) {
    const std::array<Eigen::Vector3f, 8> points = {
        Eigen::Vector3f{bbox.pmin.x(), bbox.pmin.y(), bbox.pmin.z()},  // 0 0 0
        Eigen::Vector3f{bbox.pmin.x(), bbox.pmin.y(), bbox.pmax.z()},  // 0 0 1
        Eigen::Vector3f{bbox.pmin.x(), bbox.pmax.y(), bbox.pmin.z()},  // 0 1 0
        Eigen::Vector3f{bbox.pmin.x(), bbox.pmax.y(), bbox.pmax.z()},  // 0 1 1
        Eigen::Vector3f{bbox.pmax.x(), bbox.pmin.y(), bbox.pmin.z()},  // 1 0 0
        Eigen::Vector3f{bbox.pmax.x(), bbox.pmin.y(), bbox.pmax.z()},  // 1 0 1
        Eigen::Vector3f{bbox.pmax.x(), bbox.pmax.y(), bbox.pmin.z()},  // 1 1 0
        Eigen::Vector3f{bbox.pmax.x(), bbox.pmax.y(), bbox.pmax.z()},  // 1 1 1
    };

    Eigen::Vector2f transformed_pmin{std::numeric_limits<float>::max(),
                                     std::numeric_limits<float>::max()};
    Eigen::Vector2f transformed_pmax{std::numeric_limits<float>::lowest(),
                                     std::numeric_limits<float>::lowest()};

    for (const auto& point : points) {
        Eigen::Vector3f transformed =
            (transform * point.homogeneous()).hnormalized();

        transformed_pmin.x() = std::min(transformed.x(), transformed_pmin.x());
        transformed_pmax.x() = std::max(transformed.x(), transformed_pmax.x());

        transformed_pmin.y() = std::min(transformed.y(), transformed_pmin.y());
        transformed_pmax.y() = std::max(transformed.y(), transformed_pmax.y());
    }

    return Bbox2{
        .pmin = transformed_pmin,
        .pmax = transformed_pmax,
    };
}

static Bbox2 ComputeBbox(const CameraTrajectory& traj, const Mesh& mesh,
                         const RowMajorMatrix4f& model_matrix,
                         int32_t frame_id) {
    CHECK(traj.IsFrameFilled(frame_id));
    const CameraState& state = *traj.Get(frame_id);
    const RowMajorMatrix4f mvp = state.intrinsics.To4x4ProjectionMatrix() *
                                 state.pose.Rt4x4() * model_matrix;

    Bbox2 bbox = TransformBbox(mesh.bbox, mvp);

    // Add padding
    constexpr float kPadding = 20.0f;
    bbox.pmin -= Eigen::Vector2f(kPadding, kPadding);
    bbox.pmax += Eigen::Vector2f(kPadding, kPadding);

    return bbox;
}

class CachedDatabase {
   public:
    CachedDatabase(const Database& database, const CameraTrajectory& traj,
                   const Mesh& mesh, const RowMajorMatrix4f& model_matrix)
        : first_frame_id(traj.FirstFrame()), num_frames(traj.Count()) {
        SPDLOG_INFO("Loading database into memory...");
        // Read keypoints
        frames_keypoints.resize(num_frames);
        flows.reserve(num_frames * 8);

        std::vector<int32_t> ids;
        std::vector<size_t> to_filtered_idx;

        const int32_t last_frame_id_inclusive = first_frame_id + num_frames - 1;

        for (size_t idx = 0; idx < num_frames; idx++) {
            const int32_t frame_id = first_frame_id + idx;

            Keypoints& keypoints = frames_keypoints[idx];
            database.ReadKeypoints(frame_id, keypoints);

            const Bbox2 bbox_imagespace =
                ComputeBbox(traj, mesh, model_matrix, frame_id);

            // Remove all keypoints that are not in the bounding box of the
            // model And store the permutation done in "to_filtered_idx", which
            // maps the old index of a keypoint, to its new index in the mapped
            // keypoints.
            constexpr size_t kInvalidIdx = std::numeric_limits<size_t>::max();
            to_filtered_idx.clear();
            to_filtered_idx.resize(keypoints.size());

            size_t k = 0;
            for (size_t j = 0; j < keypoints.size(); j++) {
                if (bbox_imagespace.Contains(keypoints[j])) {
                    keypoints[k] = keypoints[j];
                    to_filtered_idx[j] = k;

                    k++;
                } else {
                    to_filtered_idx[j] = kInvalidIdx;
                }
            }
            keypoints.resize(k);

            ids.clear();
            database.FindOpticalFlowsFromImage(frame_id, ids);

            for (int32_t frame_id_to : ids) {
                if (frame_id_to < first_frame_id ||
                    frame_id_to > last_frame_id_inclusive) {
                    continue;
                }

                ImagePairFlow flow;
                database.ReadImagePairFlow(frame_id, frame_id_to, flow);

                // Filter image pair flow keypoints
                size_t k = 0;
                for (size_t j = 0; j < flow.tgt_kps.size(); j++) {
                    const size_t src_kp_idx =
                        to_filtered_idx[flow.src_kps_indices[j]];

                    if (src_kp_idx != kInvalidIdx) {
                        flow.src_kps_indices[k] = src_kp_idx;
                        flow.tgt_kps[k] = flow.tgt_kps[j];
                        flow.flow_errors[k] = flow.flow_errors[j];
                        k++;
                    }
                }
                if (k != 0) {
                    flow.src_kps_indices.resize(k);
                    flow.tgt_kps.resize(k);
                    flow.flow_errors.resize(k);

                    flows.push_back(std::move(flow));
                }
            }
        }
        SPDLOG_INFO("Done loading database into memory...");
    }

    inline size_t NumFrames() const { return num_frames; }

    inline size_t NumFlows() const { return flows.size(); }

    inline int32_t FirstFrameId() const { return first_frame_id; }

    inline int32_t LastFrameId() const {
        return first_frame_id + num_frames - 1;
    }

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

class RefinementProblemBase {
   public:
    using Parameters = CameraTrajectory;

    static constexpr int kResidualLength = 2;
    static constexpr bool kShouldNormalize = true;

    RefinementProblemBase(const CachedDatabase& cached_database,
                          const AcceleratedMesh& mesh,
                          const RowMajorMatrix4f& model_matrix,
                          const CameraTrajectory& initial_camera_traj,
                          bool optimize_focal_length,
                          bool optimize_principal_point,
                          const CameraIntrinsics::Bounds& bounds)
        : cached_database(cached_database),
          mesh(mesh),
          model_matrix(model_matrix),
          model_matrix_inv(model_matrix.inverse()),
          optimize_focal_length(optimize_focal_length),
          optimize_principal_point(optimize_principal_point),
          bounds(bounds),
          first_frame_id(initial_camera_traj.FirstFrame()),
          num_cameras(initial_camera_traj.Count()) {
        CHECK_EQ(cached_database.NumFrames(), num_cameras);
        // 6 parameters for extrinsics, and potentially 3 for intrinsics.
        if (optimize_focal_length || optimize_principal_point) {
            num_params_per_camera = 9;
        } else {
            num_params_per_camera = 6;
        }

        intersections.reserve(num_cameras);
        for (int32_t frame = cached_database.FirstFrameId();
             frame <= cached_database.LastFrameId(); frame++) {
            const size_t num_keypoints =
                cached_database.ReadKeypoints(frame).size();
            intersections.emplace_back(num_keypoints, kInvalidIndex);
        }
    }

    std::pair<size_t, size_t> GetEdge(size_t edge_idx) const {
        const ImagePairFlow& flow = cached_database.ReadFlow(edge_idx);
        return {flow.image_id_from - first_frame_id,
                flow.image_id_to - first_frame_id};
    }

    Float FrameWeight(int32_t frame_id) const {
        const int32_t last_frame_id = first_frame_id + num_cameras - 1;
        const int32_t distance =
            std::min(frame_id - first_frame_id, last_frame_id - frame_id);

        return Float(1.0) / (distance + Float(1.0));
    }

    Float ResidualWeight([[maybe_unused]] size_t edge_idx,
                         [[maybe_unused]] size_t res_idx) const {
#if 0
        const ImagePairFlow& flow = cached_database.ReadFlow(edge_idx);
        return std::min(1.0 / flow.flow_errors[res_idx], 1e3);
#else
        return 1.0f;
#endif
    }

    bool IsGroundTruth(int32_t frame_id) const {
        return frame_id == cached_database.FirstFrameId() ||
               frame_id == cached_database.LastFrameId();
    }

    std::optional<Eigen::Vector2f> Evaluate(const CameraTrajectory& traj,
                                            size_t flow_idx,
                                            size_t kp_idx) const {
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

        const uint32_t src_kp_idx = src_kps_indices[kp_idx];

        CHECK_LT(kp_idx, src_kps_indices.size());
        CHECK_LT(src_kp_idx, src_kps.size());
        CHECK_LT(kp_idx, tgt_kps.size());

        const Eigen::Vector2f src_point = src_kps[src_kp_idx];
        const Eigen::Vector2f tgt_point = tgt_kps[kp_idx];

        const Eigen::Vector3f dir_camspace =
            src_camera.intrinsics.Unproject(src_point);
        const Ray ray_worldspace = {
            .origin = src_camera.pose.Center(),
            .dir = src_camera.pose.Derotate(dir_camspace),
        };

        const Ray ray_objectspace = {
            .origin = (model_matrix_inv * ray_worldspace.origin.homogeneous())
                          .hnormalized(),
            .dir = model_matrix_inv.block<3, 3>(0, 0) * ray_worldspace.dir,
        };

        bool found_intersection = false;
        Eigen::Vector3f point_object;

        const uint32_t primitive_id =
            GetIntersectionPrimitiveId(src_frame, src_kp_idx);

        if (primitive_id != kInvalidIndex) {
            const Triangle triangle = mesh.Inner().GetTriangle(primitive_id);
            const auto maybe_point = Intersect(ray_objectspace, triangle);
            if (maybe_point) {
                point_object = *maybe_point;
                found_intersection = true;
            }
        }

        if (!found_intersection) {
            const std::optional<RayHit> maybe_rayhit =
                mesh.RayCast(ray_objectspace.origin, ray_objectspace.dir, true);
            if (maybe_rayhit.has_value()) {
                SetIntersectionPrimitiveId(src_frame, src_kp_idx,
                                           maybe_rayhit->primitive_id);
                point_object = maybe_rayhit->pos;
                found_intersection = true;
            }
        }

        if (!found_intersection) {
            SetIntersectionPrimitiveId(src_frame, src_kp_idx, kInvalidIndex);
            return std::nullopt;
        }

        const Eigen::Vector3f point_world =
            (model_matrix * point_object.homogeneous()).hnormalized();
        const Eigen::Vector3f point_camera = tgt_camera.pose.Apply(point_world);

        if (tgt_camera.intrinsics.IsBehind(point_camera)) {
            return std::nullopt;
        }

        return tgt_camera.intrinsics.Project(point_camera) - tgt_point;
    }

    bool EvaluateWithJacobian(
        const CameraTrajectory& traj, size_t flow_idx, size_t kp_idx,
        RefRowMajorMatrixf<kResidualLength, Eigen::Dynamic>* J_src,
        RefRowMajorMatrixf<kResidualLength, Eigen::Dynamic>* J_tgt,
        Eigen::Vector2f& res) const {
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

        const uint32_t src_kp_idx = src_kps_indices[kp_idx];
        const uint32_t primitive_id =
            GetIntersectionPrimitiveId(src_frame, src_kp_idx);

        if (primitive_id == kInvalidIndex) {
            return false;
        }

        const Eigen::Vector2f src_point = src_kps[src_kp_idx];
        const Eigen::Vector2f tgt_point = tgt_kps[kp_idx];

        RowMajorMatrixf<3, 3> dDirCam_dIntrin;
        Eigen::Vector3f dirCam;
        src_camera.intrinsics.UnprojectWithJac(src_point, &dirCam, nullptr,
                                               &dDirCam_dIntrin);

        Eigen::Vector3f origin;
        RowMajorMatrixf<3, 3> dOrigin_dR;
        RowMajorMatrixf<3, 3> dOrigin_dt;
        src_camera.pose.CenterWithJac(&origin, &dOrigin_dR, &dOrigin_dt);

        Eigen::Vector3f dirWorld;
        RowMajorMatrixf<3, 3> dDirWorld_dDirCam;
        RowMajorMatrixf<3, 3> dDirWorld_dR;
        src_camera.pose.DerotateWithJac(dirCam, &dirWorld, &dDirWorld_dDirCam,
                                        &dDirWorld_dR);

        const Ray ray_worldspace = {
            .origin = origin,
            .dir = dirWorld,
        };

        const Triangle triangle_objectspace =
            mesh.Inner().GetTriangle(primitive_id);
        const Plane plane_worldspace = {
            .point = (model_matrix * triangle_objectspace.p1.homogeneous())
                         .head<3>(),
            .normal =
                model_matrix_inv.transpose().block<3, 3>(0, 0) *
                (triangle_objectspace.p2 - triangle_objectspace.p1)
                    .cross(triangle_objectspace.p3 - triangle_objectspace.p1),
        };

        Eigen::Vector3f X;
        RowMajorMatrixf<3, 3> dX_dOrigin;
        RowMajorMatrixf<3, 3> dX_dDirWorld;

        const bool ok = IntersectWithJac(ray_worldspace, plane_worldspace, &X,
                                         &dX_dOrigin, &dX_dDirWorld);
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
        if (J_src) {
            CHECK_EQ(J_src->rows(), 2);
            CHECK_GE(J_src->cols(), 6);

            J_src->block<2, 3>(0, 0) =
                dp_dX * (dX_dOrigin * dOrigin_dR + dX_dDirWorld * dDirWorld_dR);
            J_src->block<2, 3>(0, 3) = dp_dX * dX_dOrigin * dOrigin_dt;

            if (optimize_focal_length || optimize_principal_point) {
                CHECK_GE(J_src->cols(), 9);

                J_src->block<2, 3>(0, 6) =
                    dp_dX * dX_dDirWorld * dDirWorld_dDirCam * dDirCam_dIntrin;

                if (!optimize_focal_length) {
                    J_src->block<2, 1>(0, 6).setZero();
                }
                if (!optimize_principal_point) {
                    J_src->block<2, 2>(0, 7).setZero();
                }
            }
        }

        // Target camera
        if (J_tgt) {
            CHECK_EQ(J_tgt->rows(), 2);
            CHECK_GE(J_tgt->cols(), 6);

            J_tgt->block<2, 3>(0, 0) = dp_dXCam * dXCam_dR;
            J_tgt->block<2, 3>(0, 3) = dp_dXCam;

            if (optimize_focal_length || optimize_principal_point) {
                CHECK_GE(J_tgt->cols(), 9);

                J_tgt->block<2, 3>(0, 6) = dp_dIntrin;

                if (!optimize_focal_length) {
                    J_tgt->block<2, 1>(0, 6).setZero();
                }
                if (!optimize_principal_point) {
                    J_tgt->block<2, 2>(0, 7).setZero();
                }
            }
        }

        return true;
    }

    void Step(CameraState& state, const RefConstVectorXf& dp) const {
        CHECK_GE(dp.rows(), 6);

        state.pose.q = quat_step_post(state.pose.q, dp.block<3, 1>(0, 0));
        state.pose.t = state.pose.t + dp.block<3, 1>(3, 0);

        if (optimize_focal_length) {
            CHECK_GE(dp.rows(), 7);

            state.intrinsics.fy = state.intrinsics.fy + dp(6, 0);
            state.intrinsics.fx =
                state.intrinsics.fy * state.intrinsics.aspect_ratio;

            state.intrinsics.fy =
                std::clamp(state.intrinsics.fy, bounds.f_low, bounds.f_high);
            state.intrinsics.fx =
                std::clamp(state.intrinsics.fx, bounds.f_low, bounds.f_high);
        }
        if (optimize_principal_point) {
            CHECK_GE(dp.rows(), 9);

            state.intrinsics.cx = state.intrinsics.cx + dp(7, 0);
            state.intrinsics.cy = state.intrinsics.cy + dp(8, 0);

            state.intrinsics.cx =
                std::clamp(state.intrinsics.cx, bounds.cx_low, bounds.cx_high);
            state.intrinsics.cy =
                std::clamp(state.intrinsics.cy, bounds.cy_low, bounds.cy_high);
        }
    }

   protected:
    uint32_t GetIntersectionPrimitiveId(int32_t frame_id,
                                        uint32_t kp_idx) const {
        std::atomic_ref<uint32_t> primitive_atomic{
            intersections[frame_id - first_frame_id][kp_idx]};
        return primitive_atomic.load(std::memory_order_relaxed);
    }

    void SetIntersectionPrimitiveId(int32_t frame_id, uint32_t kp_idx,
                                    uint32_t primitive_id) const {
        std::atomic_ref<uint32_t> primitive_atomic{
            intersections[frame_id - first_frame_id][kp_idx]};
        primitive_atomic.store(primitive_id, std::memory_order_relaxed);
    }

   protected:
    static constexpr uint32_t kInvalidIndex =
        std::numeric_limits<uint32_t>::max();

    const CachedDatabase& cached_database;
    const AcceleratedMesh& mesh;
    const RowMajorMatrix4f model_matrix;
    const RowMajorMatrix4f model_matrix_inv;
    const bool optimize_focal_length;
    const bool optimize_principal_point;
    const CameraIntrinsics::Bounds bounds;

    int32_t first_frame_id;
    size_t num_cameras;
    size_t num_params_per_camera;

    mutable std::vector<std::vector<uint32_t>> intersections;
};

class GlobalRefinementProblem : public RefinementProblemBase {
   public:
    using RefinementProblemBase::RefinementProblemBase;

    size_t NumParams() const { return num_params_per_camera * num_cameras; }

    size_t NumResiduals(size_t edge_idx) const {
        return cached_database.ReadFlow(edge_idx).tgt_kps.size();
    }

    size_t NumParamBlocks() const { return num_cameras; }

    size_t ParamBlockLength() const { return num_params_per_camera; }

    size_t NumEdges() const { return cached_database.NumFlows(); }

    Float EdgeWeight(size_t edge_idx) const {
        const ImagePairFlow& flow = cached_database.ReadFlow(edge_idx);
        return FrameWeight(flow.image_id_from);
    }

    bool EvaluateWithJacobian(
        const CameraTrajectory& traj, size_t flow_idx, size_t kp_idx,
        RowMajorMatrixf<kResidualLength, Eigen::Dynamic>& J,
        Eigen::Vector2f& res) const {
        RefRowMajorMatrixf<kResidualLength, Eigen::Dynamic> J_src =
            J.block(0, 0, kResidualLength, ParamBlockLength());
        RefRowMajorMatrixf<kResidualLength, Eigen::Dynamic> J_tgt =
            J.block(0, ParamBlockLength(), kResidualLength, ParamBlockLength());

        const ImagePairFlow& flow = cached_database.ReadFlow(flow_idx);
        auto* J_src_ptr = IsGroundTruth(flow.image_id_from) ? nullptr : &J_src;
        auto* J_tgt_ptr = IsGroundTruth(flow.image_id_to) ? nullptr : &J_tgt;

        const bool result = RefinementProblemBase::EvaluateWithJacobian(
            traj, flow_idx, kp_idx, J_src_ptr, J_tgt_ptr, res);

#if 0
        // This is a hack, but it sometimes yields better results in my
        // experiments. Each residual/edge has two jacobian components: source
        // pose, and target pose. Here we dampen the jacobian of the pose with
        // the highest weight, which acts as some sort of prior regularization.
        //
        // Needs more investigation
        if (result) {
            const ImagePairFlow& flow = cached_database.ReadFlow(flow_idx);
            const Float src_weight = FrameWeight(flow.image_id_from);
            const Float tgt_weight = FrameWeight(flow.image_id_to);

            if (src_weight > tgt_weight) {
                J_src *= tgt_weight / src_weight;
            } else {
                J_tgt *= src_weight / tgt_weight;
            }
        }
#endif

        return result;
    }

    void Step(const CameraTrajectory& traj,
              const RowMajorMatrixf<Eigen::Dynamic, 1>& dp,
              CameraTrajectory& result) const {
        CHECK_EQ(result.FirstFrame(), traj.FirstFrame());
        CHECK_EQ(result.LastFrame(), traj.LastFrame());
        CHECK_EQ(dp.rows(), static_cast<Eigen::Index>(num_params_per_camera *
                                                      num_cameras));
        CHECK_EQ(first_frame_id, traj.FirstFrame());

        const size_t num_cameras = traj.Count();

        // The first and last cameras are constant.
        // TODO: Maybe don't add them to the problem to begin with?
        for (size_t i = 1; i < num_cameras - 1; i++) {
            const int32_t frame = first_frame_id + i;
            CameraState camera = *traj.Get(frame);

            const Eigen::Index J_idx = num_params_per_camera * i;
            const RefConstVectorXf dp_cam =
                dp.block(J_idx, 0, num_params_per_camera, 1);

            RefinementProblemBase::Step(camera, dp_cam);
            result.Set(frame, camera);
        }
        result.Set(traj.FirstFrame(), *traj.Get(traj.FirstFrame()));
        result.Set(traj.LastFrame(), *traj.Get(traj.LastFrame()));
    }
};

class LocalRefinementProblem : public RefinementProblemBase {
   public:
    static constexpr int kNumParams = 9;

    using RefinementProblemBase::RefinementProblemBase;

    constexpr size_t NumParams() const { return kNumParams; }
    size_t NumResiduals() const { return num_residuals; }

    void SetTargetFrame(int32_t frame_id) {
        target_frame_id = frame_id;
        num_residuals = 0;

        edges.clear();
        accum_num_residuals.clear();

        edges.reserve(16);
        accum_num_residuals.reserve(16);

        const size_t num_edges = cached_database.NumFlows();
        for (size_t edge_idx = 0; edge_idx < num_edges; edge_idx++) {
            const ImagePairFlow& flow = cached_database.ReadFlow(edge_idx);

            if (flow.image_id_from == target_frame_id ||
                flow.image_id_to == target_frame_id) {
                edges.push_back(edge_idx);
                num_residuals += flow.tgt_kps.size();
                accum_num_residuals.push_back(num_residuals);
            }
        }
    }

    std::pair<size_t, size_t> DecomposeResidualIndex(size_t idx) const {
        const size_t edge_idx =
            std::distance(accum_num_residuals.begin(),
                          std::upper_bound(accum_num_residuals.begin(),
                                           accum_num_residuals.end(), idx));
        if (edge_idx == 0) {
            return {edges[edge_idx], idx};
        } else {
            CHECK_GE(idx, accum_num_residuals[edge_idx - 1]);
            return {edges[edge_idx], idx - accum_num_residuals[edge_idx - 1]};
        }
    }

    Float Weight(size_t idx) const {
        const auto [flow_idx, kp_idx] = DecomposeResidualIndex(idx);
        const ImagePairFlow& flow = cached_database.ReadFlow(flow_idx);
        const int32_t other_frame = flow.image_id_from == target_frame_id
                                        ? flow.image_id_to
                                        : flow.image_id_from;
        CHECK_NE(other_frame, target_frame_id);
        return FrameWeight(other_frame) * ResidualWeight(flow_idx, kp_idx);
    }

    std::optional<Eigen::Vector2f> Evaluate(const CameraTrajectory& traj,
                                            size_t idx) const {
        const auto [flow_idx, kp_idx] = DecomposeResidualIndex(idx);
        return RefinementProblemBase::Evaluate(traj, flow_idx, kp_idx);
    }

    bool EvaluateWithJacobian(const CameraTrajectory& traj, size_t idx,
                              RowMajorMatrixf<kResidualLength, kNumParams>& J,
                              Eigen::Vector2f& res) const {
        const auto [flow_idx, kp_idx] = DecomposeResidualIndex(idx);
        const ImagePairFlow& flow = cached_database.ReadFlow(flow_idx);

        RefRowMajorMatrixf<kResidualLength, Eigen::Dynamic> J_ref = J;

        if (flow.image_id_from == target_frame_id) {
            return RefinementProblemBase::EvaluateWithJacobian(
                traj, flow_idx, kp_idx, &J_ref, nullptr, res);
        } else {
            CHECK_EQ(flow.image_id_to, target_frame_id);
            return RefinementProblemBase::EvaluateWithJacobian(
                traj, flow_idx, kp_idx, nullptr, &J_ref, res);
        }
    }

    void Step(const CameraTrajectory& traj,
              const RowMajorMatrixf<kNumParams, 1>& dp,
              CameraTrajectory& result) const {
        CHECK_EQ(result.FirstFrame(), traj.FirstFrame());
        CHECK_EQ(result.LastFrame(), traj.LastFrame());
        CHECK_EQ(dp.rows(), static_cast<Eigen::Index>(kNumParams));
        CHECK_EQ(first_frame_id, traj.FirstFrame());

        CameraState camera = *traj.Get(target_frame_id);
        RefinementProblemBase::Step(camera, dp);
        result.Set(target_frame_id, camera);
    }

   private:
    int32_t target_frame_id = kInvalidId;
    size_t num_residuals = 0;

    std::vector<size_t> edges;
    std::vector<size_t> accum_num_residuals;
};

template <typename LossFunction>
static void RefineTrajectory(const Database& database, CameraTrajectory& traj,
                             const RowMajorMatrix4f& model_matrix,
                             const AcceleratedMesh& mesh,
                             const LossFunction& loss_fn,
                             bool optimize_focal_length,
                             bool optimize_principal_point,
                             RefineTrajectoryCallback callback,
                             const BundleOptions& bundle_opts) {
    SPDLOG_INFO("Refining trajectory from frame #{} to frame #{} inclusive",
                traj.FirstFrame(), traj.LastFrame());

    CHECK(traj.Count() > 2);
    for (int32_t frame = traj.FirstFrame(); frame <= traj.LastFrame();
         frame++) {
        CHECK(traj.IsFrameFilled(frame));
    }

    CachedDatabase cached_database{database, traj, mesh.Inner(), model_matrix};
    RefineTrajectoryUpdate update = {};

#if 0
    LocalRefinementProblem local_problem{
        cached_database,
        mesh,
        model_matrix,
        traj,
        optimize_focal_length,
        optimize_principal_point,
        traj.Get(traj.FirstFrame())->intrinsics.GetBounds()};

    const int32_t middle_frame = (traj.FirstFrame() + traj.LastFrame()) / 2;

    for (int32_t frame_id = traj.FirstFrame() + 1; frame_id < middle_frame;
         frame_id++) {
        update.progress = 0.25 * (frame_id - traj.FirstFrame()) /
                          (middle_frame - traj.FirstFrame());
        update.message = fmt::format("Optimizing frame #{}", frame_id);
        if (!callback(update)) {
            return true;
        }

        local_problem.SetTargetFrame(frame_id);
        LevMarqDenseSolve(local_problem, loss_fn, &traj, bundle_opts);
    }

    for (int32_t frame_id = traj.LastFrame() - 1; frame_id >= middle_frame;
         frame_id--) {
        update.progress = 0.25 + 0.25 * (traj.LastFrame() - frame_id) /
                                     (traj.LastFrame() - middle_frame);
        update.message = fmt::format("Optimizing frame #{}", frame_id);
        if (!callback(update)) {
            return true;
        }

        local_problem.SetTargetFrame(frame_id);
        LevMarqDenseSolve(local_problem, loss_fn, &traj, bundle_opts);
    }
#endif

#if 1
    auto callback_wrapper = [&](const BundleStats& stats) {
        update.progress =
            static_cast<float>(stats.iterations) / bundle_opts.max_iterations;
        update.message = fmt::format("Cost: {:.02f} (Initial: {:.02f})",
                                     stats.cost, stats.initial_cost);
        update.stats = stats;

        return callback(update);
    };

    GlobalRefinementProblem problem{
        cached_database,
        mesh,
        model_matrix,
        traj,
        optimize_focal_length,
        optimize_principal_point,
        traj.Get(traj.FirstFrame())->intrinsics.GetBounds()};

    LevMarqSparseSolve(problem, loss_fn, &traj, bundle_opts, callback_wrapper);
#endif
}

static void RefineTrajectory(const Database& database, CameraTrajectory& traj,
                             const RowMajorMatrix4f& model_matrix,
                             const AcceleratedMesh& mesh,
                             bool optimize_focal_length,
                             bool optimize_principal_point,
                             RefineTrajectoryCallback callback,
                             const BundleOptions& bundle_opts) {
    switch (bundle_opts.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                           \
    {                                                                     \
        LossFunction loss_fn(bundle_opts.loss_scale);                     \
        RefineTrajectory(database, traj, model_matrix, mesh, loss_fn,     \
                         optimize_focal_length, optimize_principal_point, \
                         callback, bundle_opts);                          \
    }
        SWITCH_LOSS_FUNCTIONS
#undef SWITCH_LOSS_FUNCTION_CASE
        default:
            throw std::runtime_error(
                fmt::format("Unknown loss type: {}",
                            static_cast<int>(bundle_opts.loss_type)));
    }
}

void RefineTrajectory(const std::string& database_path, CameraTrajectory& traj,
                      const RowMajorMatrix4f& model_matrix,
                      const AcceleratedMesh& mesh, bool optimize_focal_length,
                      bool optimize_principal_point,
                      RefineTrajectoryCallback callback,
                      BundleOptions bundle_opts) {
    Database database{database_path};
    RefineTrajectory(database, traj, model_matrix, mesh, optimize_focal_length,
                     optimize_principal_point, callback, bundle_opts);
}
