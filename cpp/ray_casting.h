#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include <optional>

#include "eigen_typedefs.h"
#include "geometry.h"

class AcceleratedMesh;
using AcceleratedMeshSptr = std::shared_ptr<AcceleratedMesh>;

struct Ray {
    Eigen::Vector3f origin;
    Eigen::Vector3f dir;  // Doesn't have to be normalized
};

struct RaysSameOrigin {
    Eigen::Vector3f origin;
    RowMajorMatrixX3f dirs;
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
                              Eigen::Vector2f pos);

std::optional<RayHit> RayCast(const AcceleratedMeshSptr& accel_mesh, const SceneTransformations& scene_transform,
                              RowMajorMatrixX2f pos);

static inline Ray GetRayObjectSpace(const SceneTransformations& scene_transform, Eigen::Vector2f pos) {
    const Eigen::Matrix4f inverse_model_view_proj_mat = (scene_transform.intrinsics.To4x4ProjectionMatrix() *
                                                         scene_transform.view_matrix * scene_transform.model_matrix)
                                                            .inverse();
    const Eigen::Vector3f ray_origin =
        (scene_transform.view_matrix * scene_transform.model_matrix).inverse().col(3).head<3>();

    const Eigen::Vector4f pos_homo = {pos.x(), pos.y(), -0.5f, 1.0f};

    // Calculate ray direction
    const Eigen::Vector3f target_point = (inverse_model_view_proj_mat * pos_homo).hnormalized();
    const Eigen::Vector3f ray_dir = target_point - ray_origin;

    return Ray{ray_origin, ray_dir};
}

static inline Ray GetRayWorldSpace(const SceneTransformations& scene_transform, Eigen::Vector2f pos) {
    const Eigen::Matrix4f inverse_view_proj_mat =
        (scene_transform.intrinsics.To4x4ProjectionMatrix() * scene_transform.view_matrix).inverse();
    const Eigen::Vector3f ray_origin =
        -scene_transform.view_matrix.block<3, 3>(0, 0).transpose() * scene_transform.view_matrix.block<3, 1>(0, 3);

    const Eigen::Vector4f pos_homo = {pos.x(), pos.y(), -1.0f, 1.0f};

    // Calculate ray direction
    const Eigen::Vector3f target_point = (inverse_view_proj_mat * pos_homo).hnormalized();
    const Eigen::Vector3f ray_dir = target_point - ray_origin;

    return Ray{ray_origin, ray_dir};
}

static inline RaysSameOrigin GetRaysObjectSpace(const SceneTransformations& scene_transform,
                                                RowMajorMatrixX2f positions) {
    const Eigen::Matrix4f inverse_model_view_proj_mat = (scene_transform.intrinsics.To4x4ProjectionMatrix() *
                                                         scene_transform.view_matrix * scene_transform.model_matrix)
                                                            .inverse();
    const Eigen::Vector3f ray_origin =
        (scene_transform.view_matrix * scene_transform.model_matrix).inverse().col(3).head<3>();

    RaysSameOrigin result = {.origin = ray_origin, .dirs = RowMajorMatrixX3f{positions.rows(), 3}};

    // TODO: benchmark / vectorize / parallelize
    for (Eigen::Index row = 0; row < positions.rows(); row++) {
        const Eigen::Vector4f pos_homo = {positions.row(row).x(), positions.row(row).y(), -0.5f, 1.0f};

        // Calculate ray direction
        const Eigen::Vector3f target_point = (inverse_model_view_proj_mat * pos_homo).hnormalized();
        result.dirs.row(row) = target_point - ray_origin;
    }

    return result;
}
