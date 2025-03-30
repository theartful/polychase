#pragma once

#include <Eigen/Core>
#include <cstdint>

using RowMajorArrayX3f = Eigen::Array<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
using RowMajorArrayX3u = Eigen::Array<uint32_t, Eigen::Dynamic, 3, Eigen::RowMajor>;
using RowMajorArrayX2u = Eigen::Array<uint32_t, Eigen::Dynamic, 2, Eigen::RowMajor>;
using RowMajorMatrix4f = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;
