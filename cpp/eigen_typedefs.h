#pragma once

#include <Eigen/Core>
#include <cstdint>

using RowMajorArrayX4f = Eigen::Array<float, Eigen::Dynamic, 4, Eigen::RowMajor>;
using RowMajorArrayX3f = Eigen::Array<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
using RowMajorArrayX3u = Eigen::Array<uint32_t, Eigen::Dynamic, 3, Eigen::RowMajor>;
using RowMajorArrayX2u = Eigen::Array<uint32_t, Eigen::Dynamic, 2, Eigen::RowMajor>;

using RowMajorMatrixX4f = Eigen::Matrix<float, Eigen::Dynamic, 4, Eigen::RowMajor>;
using RowMajorMatrixX3f = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
using RowMajorMatrixX2f = Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>;
using RowMajorMatrix4f = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;
using RowMajorMatrix3f = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;

// Reference types

using RefRowMajorArrayX4f = Eigen::Ref<RowMajorArrayX4f>;
using RefRowMajorArrayX3f = Eigen::Ref<RowMajorArrayX3f>;
using RefRowMajorArrayX3u = Eigen::Ref<RowMajorArrayX3u>;
using RefRowMajorArrayX2u = Eigen::Ref<RowMajorArrayX2u>;

using RefRowMajorMatrixX4f = Eigen::Ref<RowMajorMatrixX4f>;
using RefRowMajorMatrixX3f = Eigen::Ref<RowMajorMatrixX3f>;
using RefRowMajorMatrixX2f = Eigen::Ref<RowMajorMatrixX2f>;
using RefRowMajorMatrix4f = Eigen::Ref<RowMajorMatrix4f>;
using RefRowMajorMatrix3f = Eigen::Ref<RowMajorMatrix3f>;
