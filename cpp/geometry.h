// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

#pragma once

#include <Eigen/Core>

#include "eigen_typedefs.h"
#include "pnp/types.h"
#include "utils.h"

struct Triangle {
    Eigen::Vector3f p1;
    Eigen::Vector3f p2;
    Eigen::Vector3f p3;

    inline Eigen::Vector3f Barycentric(float u, float v) const {
        return (1.0 - u - v) * p1 + u * p2 + v * p3;
    }
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
    RowMajorArrayX3u triangles;
    ArrayXu masked_triangles;
    Bbox3 bbox;

    Mesh(RowMajorArrayX3f vertices_, RowMajorArrayX3u triangles_,
         ArrayXu masked_triangles_)
        : vertices{std::move(vertices_)},
          triangles{std::move(triangles_)},
          masked_triangles{masked_triangles_} {
        const int mask_num_ints = (triangles.rows() + 31) / 32;
        const int mask_num_ints_padded =
            mask_num_ints + (4 - mask_num_ints % 4) % 4;

        if (masked_triangles.rows() == 0) {
            masked_triangles = ArrayXu(mask_num_ints_padded);
            masked_triangles.setZero();
        }

        CHECK_GE(masked_triangles.rows(), mask_num_ints_padded);

        // Compute Bbox
        Eigen::Vector3f pmin = {std::numeric_limits<float>::max(),
                                std::numeric_limits<float>::max(),
                                std::numeric_limits<float>::max()};
        Eigen::Vector3f pmax = {std::numeric_limits<float>::lowest(),
                                std::numeric_limits<float>::lowest(),
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
        CHECK(triangle_index < triangles.rows());

        return Triangle{
            .p1 = GetVertex(triangles.row(triangle_index)[0]),
            .p2 = GetVertex(triangles.row(triangle_index)[1]),
            .p3 = GetVertex(triangles.row(triangle_index)[2]),
        };
    }

    inline bool IsTriangleMasked(uint32_t tri_idx) const {
        const uint32_t elem_idx = tri_idx / 32;
        const uint32_t bit_idx = tri_idx % 32;

        CHECK_LT(elem_idx, masked_triangles.rows());

        return (masked_triangles[elem_idx] & (1u << bit_idx)) != 0;
    }

    inline void MaskTriangle(uint32_t tri_idx) {
        const uint32_t elem_idx = tri_idx / 32;
        const uint32_t bit_idx = tri_idx % 32;

        CHECK_LT(elem_idx, masked_triangles.rows());

        uint32_t& mask_element = masked_triangles[elem_idx];
        mask_element |= (1u << bit_idx);
    }

    inline void UnmaskTriangle(uint32_t tri_idx) {
        const uint32_t elem_idx = tri_idx / 32;
        const uint32_t bit_idx = tri_idx % 32;

        CHECK_LT(elem_idx, masked_triangles.rows());

        uint32_t& mask_element = masked_triangles[elem_idx];
        mask_element &= ~(1u << bit_idx);
    }

    inline void ToggleMaskTriangle(uint32_t tri_idx) {
        const uint32_t elem_idx = tri_idx / 32;
        const uint32_t bit_idx = tri_idx % 32;

        CHECK_LT(elem_idx, masked_triangles.rows());

        uint32_t& mask_element = masked_triangles[elem_idx];
        mask_element ^= (1u << bit_idx);
    }
};

// Maybe use Sophus types for lie-groups instead of RowMajorMatrix4f. For
// example view_matrix should be SE3.
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
