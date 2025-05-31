#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <type_traits>

// In case I want to start using double precision.
// This currently won't work though, since embree for example only supports floats
using Float = float;

// Row major types

using RowMajorArrayX4f = Eigen::Array<Float, Eigen::Dynamic, 4, Eigen::RowMajor>;
using RowMajorArrayX3f = Eigen::Array<Float, Eigen::Dynamic, 3, Eigen::RowMajor>;
using RowMajorArrayX2f = Eigen::Array<Float, Eigen::Dynamic, 2, Eigen::RowMajor>;
using RowMajorArrayX3u = Eigen::Array<uint32_t, Eigen::Dynamic, 3, Eigen::RowMajor>;
using RowMajorArrayX2u = Eigen::Array<uint32_t, Eigen::Dynamic, 2, Eigen::RowMajor>;

using RowMajorMatrixX4f = Eigen::Matrix<Float, Eigen::Dynamic, 4, Eigen::RowMajor>;
using RowMajorMatrixX3f = Eigen::Matrix<Float, Eigen::Dynamic, 3, Eigen::RowMajor>;
using RowMajorMatrixX2f = Eigen::Matrix<Float, Eigen::Dynamic, 2, Eigen::RowMajor>;
using RowMajorMatrix4f = Eigen::Matrix<Float, 4, 4, Eigen::RowMajor>;
using RowMajorMatrix3f = Eigen::Matrix<Float, 3, 3, Eigen::RowMajor>;
using RowMajorMatrix34f = Eigen::Matrix<Float, 3, 4, Eigen::RowMajor>;
using RowMajorMatrix2f = Eigen::Matrix<Float, 2, 2, Eigen::RowMajor>;
using RowMajorMatrixXf = Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using ArrayXu = Eigen::Array<uint32_t, Eigen::Dynamic, 1>;
using ArrayXf = Eigen::ArrayXf;
using VectorXf = Eigen::VectorXf;

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

using ConstRefRowMajorMatrixX3f = Eigen::Ref<const RowMajorMatrixX3f>;
using ConstRefRowMajorMatrixX4f = Eigen::Ref<const RowMajorMatrixX4f>;
using ConstRefRowMajorMatrixX3f = Eigen::Ref<const RowMajorMatrixX3f>;
using ConstRefRowMajorMatrixX2f = Eigen::Ref<const RowMajorMatrixX2f>;
using ConstRefRowMajorMatrix4f = Eigen::Ref<const RowMajorMatrix4f>;
using ConstRefRowMajorMatrix3f = Eigen::Ref<const RowMajorMatrix3f>;

using ConstRefArrayXf = Eigen::Ref<const ArrayXf>;
using ConstRefVectorXf = Eigen::Ref<const VectorXf>;

template <typename T, int R, int C>
using RowMajorMatrix =
    std::conditional_t<R != 1 && C != 1, Eigen::Matrix<T, R, C, Eigen::RowMajor>, Eigen::Matrix<T, R, C>>;

template <int R, int C>
using RowMajorMatrixf = RowMajorMatrix<Float, R, C>;

template <int R, int C>
using RowMajorMatrixd = RowMajorMatrix<double, R, C>;

template <int R, int C>
using RefRowMajorMatrixf = Eigen::Ref<RowMajorMatrix<Float, R, C>>;

template <int R, int C>
using RefRowMajorMatrixd = Eigen::Ref<RowMajorMatrix<double, R, C>>;
