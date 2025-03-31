#pragma once

#include <Eigen/Core>
#include <memory>
#include <optional>

#include "geometry.h"

class AcceleratedMesh;
using AcceleratedMeshSptr = std::shared_ptr<AcceleratedMesh>;

struct Ray {
    Eigen::Vector3f origin;
    Eigen::Vector3f dir; // Doesn't have to be normalized
};

struct RayHit {
    Eigen::Vector3f pos;
    Eigen::Vector3f normal;
    Eigen::Vector2f barycentric_coordinate;
    float t;
    uint32_t primitive_id;
};

AcceleratedMeshSptr CreateAcceleratedMesh(MeshSptr mesh);
AcceleratedMeshSptr CreateAcceleratedMesh(RowMajorArrayX3f vertices, RowMajorArrayX3u indices);

std::optional<RayHit> RayCast(const AcceleratedMeshSptr& accel_mesh, Eigen::Vector3f origin,
                              Eigen::Vector3f direction);

std::optional<RayHit> RayCast(const AcceleratedMeshSptr& accel_mesh, const SceneTransformations& scene_transform,
                              Eigen::Vector2f ndc_pos);

// YUCK: Do it by calling GetRayWorldSpace!
static inline Ray GetRayObjectSpace(const SceneTransformations& scene_transform, Eigen::Vector2f ndc_pos) {
    // NDC is from -1 to 1
    const Eigen::Matrix4f inverse_model_view_mat =
        (scene_transform.view_matrix * scene_transform.model_matrix).inverse();
    const Eigen::Matrix4f inverse_model_view_proj_mat =
        (scene_transform.projection_matrix * scene_transform.view_matrix * scene_transform.model_matrix).inverse();
    const Eigen::Vector3f camera_center_objspace = inverse_model_view_mat.col(3).head<3>();

    // Convert from pixel coordinates to NDC coordinates
    const Eigen::Vector4f ndc_pos_homo = {ndc_pos.x(), ndc_pos.y(), -0.5f, 1.0f};

    // Calculate ray direction
    const Eigen::Vector4f target_point_homo = inverse_model_view_proj_mat * ndc_pos_homo;
    const Eigen::Vector3f ray_dir = (target_point_homo.head<3>() / target_point_homo.w()) - camera_center_objspace;

    // Calculate ray origin
    const Eigen::Vector3f ray_origin = inverse_model_view_mat.col(3).head<3>();

    return Ray{ray_origin, ray_dir};
}

static inline Ray GetRayWorldSpace(const SceneTransformations& scene_transform, Eigen::Vector2f ndc_pos) {
    // NDC is from -1 to 1
    const Eigen::Matrix4f inverse_view_mat = scene_transform.view_matrix.inverse();
    const Eigen::Matrix4f inverse_view_proj_mat =
        (scene_transform.projection_matrix * scene_transform.view_matrix).inverse();
    const Eigen::Vector3f camera_center_worldspace = inverse_view_mat.col(3).head<3>();

    // Convert from pixel coordinates to NDC coordinates
    const Eigen::Vector4f ndc_pos_homo = {ndc_pos.x(), ndc_pos.y(), -0.5f, 1.0f};

    // Calculate ray direction
    const Eigen::Vector4f target_point_homo = inverse_view_proj_mat * ndc_pos_homo;
    const Eigen::Vector3f ray_dir = (target_point_homo.head<3>() / target_point_homo.w()) - camera_center_worldspace;

    // Calculate ray origin
    const Eigen::Vector3f ray_origin = inverse_view_mat.col(3).head<3>();

    return Ray{ray_origin, ray_dir};
}
