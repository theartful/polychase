#pragma once

#include <Eigen/Core>
#include <memory>

#include "eigen_typedefs.h"
#include "utils.h"

struct Triangle {
    Eigen::Vector3f p1;
    Eigen::Vector3f p2;
    Eigen::Vector3f p3;

    inline Eigen::Vector3f Barycentric(float u, float v) const { return (1.0 - u - v) * p1 + u * p2 + v * p3; }
};

struct Mesh {
    RowMajorArrayX3f vertices;
    RowMajorArrayX3u indices;

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
    auto mesh_sptr = std::make_shared<Mesh>();
    mesh_sptr->vertices = std::move(vertices);
    mesh_sptr->indices = std::move(indices);

    return mesh_sptr;
}

struct SceneTransformations {
    // Object to world matrix
    RowMajorMatrix4f model_matrix;
    // World to camera matrix
    RowMajorMatrix4f view_matrix;
    // Camera to NDC matrix
    RowMajorMatrix4f projection_matrix;

    int window_width;
    int window_height;
};
