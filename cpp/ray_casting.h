#pragma once

#include <embree4/rtcore.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <optional>

#include "eigen_typedefs.h"
#include "geometry.h"

struct RayHit {
    Eigen::Vector3f pos;
    Eigen::Vector3f normal;
    Eigen::Vector2f barycentric_coordinate;
    float t;
    uint32_t primitive_id;
};

class AcceleratedMesh {
   public:
    AcceleratedMesh(RowMajorArrayX3f vertices, RowMajorArrayX3u indices);

    AcceleratedMesh(const AcceleratedMesh&) = delete;
    AcceleratedMesh(AcceleratedMesh&&) = delete;
    AcceleratedMesh& operator=(const AcceleratedMesh&) = delete;
    AcceleratedMesh& operator=(AcceleratedMesh&&) = delete;

    ~AcceleratedMesh();

    std::optional<RayHit> RayCast(Eigen::Vector3f origin,
                                  Eigen::Vector3f direction) const;
    const Mesh& Inner() const { return mesh_; }

   private:
    Mesh mesh_;
    RTCDevice rtc_device_;
    RTCScene rtc_scene_;
};

std::optional<RayHit> RayCast(const AcceleratedMesh& accel_mesh,
                              Eigen::Vector3f origin,
                              Eigen::Vector3f direction);

std::optional<RayHit> RayCast(const AcceleratedMesh& accel_mesh,
                              const SceneTransformations& scene_transform,
                              Eigen::Vector2f pos);

std::optional<RayHit> RayCast(const AcceleratedMesh& accel_mesh,
                              const SceneTransformations& scene_transform,
                              RowMajorMatrixX2f pos);

static inline Ray GetRayObjectSpace(const SceneTransformations& scene_transform,
                                    Eigen::Vector2f pos) {
    const RowMajorMatrix4f mat =
        (scene_transform.view_matrix * scene_transform.model_matrix).inverse();

    const Eigen::Vector3f ray_origin = mat.col(3).head<3>();
    const Eigen::Vector3f ray_dir =
        mat.block<3, 3>(0, 0) * scene_transform.intrinsics.Unproject(pos);

    return Ray{ray_origin, ray_dir};
}

static inline Ray GetRayWorldSpace(const SceneTransformations& scene_transform,
                                   Eigen::Vector2f pos) {
    const RowMajorMatrix4f mat = scene_transform.view_matrix.inverse();

    const Eigen::Vector3f ray_origin = mat.col(3).head<3>();
    const Eigen::Vector3f ray_dir =
        mat.block<3, 3>(0, 0) * scene_transform.intrinsics.Unproject(pos);

    return Ray{ray_origin, ray_dir};
}

[[nodiscard]] static inline bool IntersectWithJac(
    const Ray& ray, const Plane& plane, Eigen::Vector3f* intersection,
    RowMajorMatrixf<3, 3>* jac_origin = nullptr,
    RowMajorMatrixf<3, 3>* jac_dir = nullptr) {
    // Plane equation:
    //      (p - p0).dot(n) = 0
    // Ray equation:
    //      p = o + td
    //
    // Substituting:
    //      (o + td - p0).dot(n) = 0
    //      (o - p0).dot(n) + t d.dot(n) = 0
    //      t = (p0 - o).dot(n) / d.dot(n)

    const double d_dot_n = ray.dir.dot(plane.normal);
    if (d_dot_n > -1e-10 && d_dot_n < 1e-10) {
        return false;
    }

    const double p0_minus_o_dot_n =
        (plane.point - ray.origin).dot(plane.normal);

    const double t = p0_minus_o_dot_n / d_dot_n;
    *intersection = ray.origin + ray.dir * t;

    if (jac_origin) {
        *jac_origin = RowMajorMatrix3f::Identity() -
                      ray.dir * plane.normal.transpose() / d_dot_n;
    }
    if (jac_dir) {
        *jac_dir = (RowMajorMatrix3f::Identity() -
                    ray.dir * plane.normal.transpose() / d_dot_n) *
                   t;
    }

    return true;
}

static inline std::optional<Eigen::Vector3f> Intersect(const Ray& ray,
                                                       const Plane& plane) {
    Eigen::Vector3f result;
    const bool ok = IntersectWithJac(ray, plane, &result);
    if (!ok) {
        return {};
    }

    return result;
}

[[nodiscard]] static inline bool IntersectWithJac(
    const Ray& ray, const Triangle& tri, Eigen::Vector3f* intersection,
    RowMajorMatrixf<3, 3>* jac_origin = nullptr,
    RowMajorMatrixf<3, 3>* jac_dir = nullptr) {
    // https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    constexpr float epsilon = 1e-10;

    const Eigen::Vector3f edge1 = tri.p2 - tri.p1;
    const Eigen::Vector3f edge2 = tri.p3 - tri.p1;
    const Eigen::Vector3f ray_cross_e2 = ray.dir.cross(edge2);
    const float det = edge1.dot(ray_cross_e2);

    if (det > -epsilon && det < epsilon) {
        return false;
    }

    const float inv_det = 1.0 / det;
    const Eigen::Vector3f s = ray.origin - tri.p1;
    const float u = inv_det * s.dot(ray_cross_e2);

    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    const Eigen::Vector3f s_cross_e1 = s.cross(edge1);
    const float v = inv_det * ray.dir.dot(s_cross_e1);

    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }

    const float edge2_dot_s_cross_e1 = edge2.dot(s_cross_e1);
    const float t = inv_det * edge2_dot_s_cross_e1;
    if (t < 0.0f) {
        return false;
    }

    *intersection = ray.origin + ray.dir * t;

    if (jac_origin || jac_dir) {
        const Eigen::Vector3f plane_normal = edge2.cross(edge1);

        if (jac_origin) {
            *jac_origin = RowMajorMatrix3f::Identity() -
                          inv_det * ray.dir * plane_normal.transpose();
        }
        if (jac_dir) {
            *jac_dir = (RowMajorMatrix3f::Identity() -
                        ray.dir * plane_normal.transpose() * inv_det) *
                       t;
        }
    }

    return true;
}

static inline std::optional<Eigen::Vector3f> Intersect(const Ray& ray,
                                                       const Triangle& tri) {
    Eigen::Vector3f result;
    const bool ok = IntersectWithJac(ray, tri, &result);
    if (!ok) {
        return {};
    }

    return result;
}
