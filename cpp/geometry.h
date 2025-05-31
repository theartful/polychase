#pragma once

#include <Eigen/Core>
#include <memory>

#include "eigen_typedefs.h"
#include "pnp/types.h"
#include "utils.h"

struct Triangle {
    Eigen::Vector3f p1;
    Eigen::Vector3f p2;
    Eigen::Vector3f p3;

    inline Eigen::Vector3f Barycentric(float u, float v) const { return (1.0 - u - v) * p1 + u * p2 + v * p3; }
};

struct Plane {
    Eigen::Vector3f point;
    Eigen::Vector3f normal;
};

struct Ray {
    Eigen::Vector3f origin;
    Eigen::Vector3f dir;  // Doesn't have to be normalized
};

struct Bbox3 {
    Eigen::Vector3f pmin;
    Eigen::Vector3f pmax;

    bool Contains(Eigen::Vector3f p) const {
        return p.x() > pmin.x() && p.y() > pmin.y() && p.z() > pmin.z() &&  //
               p.x() < pmax.x() && p.y() < pmax.y() && p.z() < pmax.z();
    }
};

struct Bbox2 {
    Eigen::Vector2f pmin;
    Eigen::Vector2f pmax;

    bool Contains(Eigen::Vector2f p) const {
        return p.x() > pmin.x() && p.y() > pmin.y() &&  //
               p.x() < pmax.x() && p.y() < pmax.y();
    }
};

struct Mesh {
    RowMajorArrayX3f vertices;
    RowMajorArrayX3u indices;
    Bbox3 bbox;

    Mesh(RowMajorArrayX3f vertices_, RowMajorArrayX3u indices_)
        : vertices{std::move(vertices_)}, indices{std::move(indices_)} {
        Eigen::Vector3f pmin = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                                std::numeric_limits<float>::max()};
        Eigen::Vector3f pmax = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                                std::numeric_limits<float>::lowest()};

        const Eigen::Index num_vertices = vertices.rows();
        for (Eigen::Index i = 0; i < num_vertices; i++) {
            const Eigen::Vector3f vertex = vertices.row(i);

            pmin[0] = std::min(pmin[0], vertex[0]);
            pmin[1] = std::min(pmin[1], vertex[1]);
            pmin[2] = std::min(pmin[2], vertex[2]);

            pmax[0] = std::max(pmax[0], vertex[0]);
            pmax[1] = std::max(pmax[1], vertex[1]);
            pmax[2] = std::max(pmax[2], vertex[2]);
        }

        bbox = {.pmin = pmin, .pmax = pmax};
    }

    inline Eigen::Vector3f GetVertex(uint32_t idx) const {
        CHECK(idx < vertices.rows());

        return vertices.row(static_cast<Eigen::Index>(idx));
    };

    inline Triangle GetTriangle(uint32_t triangle_index) const {
        CHECK(triangle_index < indices.rows());

        return Triangle{
            .p1 = GetVertex(indices.row(triangle_index)[0]),
            .p2 = GetVertex(indices.row(triangle_index)[1]),
            .p3 = GetVertex(indices.row(triangle_index)[2]),
        };
    }
};

using MeshSptr = std::shared_ptr<Mesh>;

static inline MeshSptr CreateMesh(RowMajorArrayX3f vertices, RowMajorArrayX3u indices) {
    return std::make_shared<Mesh>(std::move(vertices), std::move(indices));
}

// Maybe use Sophus types for lie-groups instead of RowMajorMatrix4f. For example view_matrix should be SE3.
struct SceneTransformations {
    // Object to world matrix
    RowMajorMatrix4f model_matrix;
    // World to camera matrix
    RowMajorMatrix4f view_matrix;
    // Camera intrinsics.
    CameraIntrinsics intrinsics;
};

enum class TransformationType {
    Camera,
    Model,
};
