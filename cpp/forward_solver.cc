#include "forward_solver.h"

#include <spdlog/spdlog.h>

#include <Eigen/Eigen>
#include <Eigen/LU>
#include <iostream>

#include "database.h"
#include "pnp.h"
#include "ray_casting.h"

std::optional<RowMajorMatrix4f> SolveFrame(const Database& database,
                                           const std::vector<RowMajorMatrix4f>& view_matrices,
                                           const RowMajorMatrix4f& model_matrix,
                                           const RowMajorMatrix4f& projection_matrix, uint32_t first_frame,
                                           uint32_t frame_id, const AcceleratedMeshSptr& accel_mesh) {
    std::vector<Eigen::Vector3f> object_points_worldspace;
    std::vector<Eigen::Vector2f> image_points;

    CHECK(frame_id > first_frame);
    const RowMajorMatrix4f& prev_view_matrix = view_matrices.at(frame_id - first_frame - 1);

    const std::vector<uint32_t> flow_frames_ids = database.FindOpticalFlowsToImage(frame_id);
    for (uint32_t flow_frame_id : flow_frames_ids) {
        CHECK(flow_frame_id != frame_id);
        // We don't have a solution for frames in the future yet at this point.
        if (flow_frame_id > frame_id) {
            continue;
        }
        // FIXME: We don't have access to frames before first_frame
        if (flow_frame_id < first_frame) {
            continue;
        }

        // if (frame_id - flow_frame_id != 1) {
        //     continue;
        // }

        const KeypointsMatrix keypoints = database.ReadKeypoints(flow_frame_id);
        const ImagePairFlow flow = database.ReadImagePairFlow(flow_frame_id, frame_id);

        CHECK_EQ(flow.src_kps_indices.rows(), flow.tgt_kps.rows());
        const Eigen::Index num_matches = flow.src_kps_indices.rows();

        // TODO: benchmark / vectorize / parallelize
        for (uint32_t i = 0; i < num_matches; i++) {
            const uint32_t kp_idx = flow.src_kps_indices[i];
            const Eigen::Vector2f kp = keypoints.row(kp_idx);

            // Mabye collect rays, and bulk RayCast using embrees optimized rtcIntersect4/8/16
            const std::optional<RayHit> hit =
                RayCast(accel_mesh,
                        SceneTransformations{.model_matrix = model_matrix,
                                             .view_matrix = view_matrices.at(flow_frame_id - first_frame),
                                             .projection_matrix = projection_matrix},
                        kp);
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

    // for (size_t i = 0; i < object_points_worldspace.size(); i++) {
    //     const Eigen::Vector4f p{object_points_worldspace[i].x(), object_points_worldspace[i].y(),
    //                             object_points_worldspace[i].z(), 1.0};
    //     const Eigen::Vector4f pc = prev_view_matrix * p;
    //     Eigen::Vector4f estimated_image_point = projection_matrix * pc;
    //     estimated_image_point /= estimated_image_point[3];

    //     const Eigen::Vector2f estimated{estimated_image_point.x(), estimated_image_point.y()};

    //     std::cout << "ESTIMATED_IMAGE_POINT = \t" << estimated.x() << '\t' << estimated.y() << '\n';
    //     std::cout << "ACTUAL_IMAGE_POINT = \t" << image_points[i].x() << '\t' << image_points[i].y() << '\n';
    //     std::cout << "ERROR_BETWEEN = \t" << (image_points[i] - estimated).norm() << '\n';
    // }

    const Eigen::Map<const RowMajorMatrixX3f> object_points_eigen{
        reinterpret_cast<const float*>(object_points_worldspace.data()),
        static_cast<Eigen::Index>(object_points_worldspace.size()), 3};

    const Eigen::Map<const RowMajorMatrixX2f> image_points_eigen{reinterpret_cast<const float*>(image_points.data()),
                                                                 static_cast<Eigen::Index>(image_points.size()), 2};

    // The solution should be very close to the previous view matrix
    PnPResult result = {.translation = prev_view_matrix.block<3, 1>(0, 3),
                        .rotation = prev_view_matrix.block<3, 3>(0, 0)};

    const bool ok =
        SolvePnP(object_points_eigen, image_points_eigen, projection_matrix, PnPSolveMethod::Ransac, result);
    if (ok) {
        RowMajorMatrix4f new_view_matrix = RowMajorMatrix4f::Identity();
        new_view_matrix.block<3, 3>(0, 0) = result.rotation;
        new_view_matrix.block<3, 1>(0, 3) = result.translation;

        return new_view_matrix;
    } else {
        return std::nullopt;
    }
}

bool SolveForwards(const std::string& database_path, uint32_t frame_from, size_t num_frames,
                   const SceneTransformations& scene_transform, const AcceleratedMeshSptr& accel_mesh,
                   TransformationType trans_type, SolveForwardsCallback callback) {
    const Database database{database_path};
    // I'm assuming that we're solving for the view matrix, since it's easier to reason about.
    // So, both the model matrix, and the projection matrix are constant.
    std::vector<RowMajorMatrix4f> view_matrices;
    view_matrices.reserve(num_frames);

    view_matrices.push_back(scene_transform.view_matrix);

    // std::cout << "PROJECTION_MATRIX = \n" << scene_transform.projection_matrix << '\n';

    for (uint32_t frame_id = frame_from + 1; frame_id < frame_from + num_frames; frame_id++) {
        std::optional<RowMajorMatrix4f> maybe_view =
            SolveFrame(database, view_matrices, scene_transform.model_matrix, scene_transform.projection_matrix,
                       frame_from, frame_id, accel_mesh);

        if (!maybe_view) {
            return false;
        }

        view_matrices.push_back(*maybe_view);

        bool ok;
        switch (trans_type) {
            case TransformationType::Camera: {
                ok = callback(frame_id, *maybe_view);
                break;
            }
            case TransformationType::Model: {
                ok = callback(frame_id,
                              scene_transform.view_matrix.inverse() * (*maybe_view) * scene_transform.model_matrix);
                break;
            }
        }

        if (!ok) {
            return false;
        }
    }

    return true;
}
