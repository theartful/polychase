#pragma once

#include <Eigen/Core>
#include <memory>
#include <optional>

#include "geometry.h"

class AcceleratedMesh;
using AcceleratedMeshSptr = std::shared_ptr<AcceleratedMesh>;

struct RayHit {
    Eigen::Vector3f pos;
    Eigen::Vector3f normal;
    Eigen::Vector2f barycentric_coordinate;
    float t;
    uint32_t primitive_id;
};

AcceleratedMeshSptr CreateAcceleratedMesh(MeshSptr mesh);
std::optional<RayHit> RayCast(AcceleratedMeshSptr& accel_mesh, Eigen::Vector3f origin, Eigen::Vector3f direction);
