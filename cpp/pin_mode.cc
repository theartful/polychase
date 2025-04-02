#include "pin_mode.h"

#include <spdlog/spdlog.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <cmath>
#include <opencv2/calib3d.hpp>

#include "pnp.h"
#include "ray_casting.h"

static std::optional<RowMajorMatrix4f> FindTransformationN(const RefRowMajorMatrixX3f& object_points,
                                                           const SceneTransformations& scene_transform,
                                                           const PinUpdate& update, TransformationType trans_type) {
    CHECK(object_points.rows() > 2);

    const CameraIntrinsics camera = {
        .fx = scene_transform.projection_matrix.coeff(0, 0),
        .fy = scene_transform.projection_matrix.coeff(1, 1),
        .cx = scene_transform.projection_matrix.coeff(2, 0),
        .cy = scene_transform.projection_matrix.coeff(2, 1),
    };

    const RowMajorMatrix3f K_transpose = CreateOpenGLCameraIntrinsicsMatrix(camera).transpose();

    // Project points
    const RowMajorMatrix4f model_view = scene_transform.view_matrix * scene_transform.model_matrix;

    // We will store the final result in these variables as well. Hence they're not const
    PnPResult result = {
        .translation = model_view.block<3, 1>(0, 3),
        .rotation = model_view.block<3, 3>(0, 0),
    };

    // Project points
    const RowMajorMatrixX3f image_points_3d =
        ((object_points * result.rotation.transpose()).rowwise() + result.translation.transpose()) * K_transpose;

    // FIXME: Maybe find some way that doesn't require allocating a new array
    // the problem is cv::solvePnP requires the data to be contiguous, and I'm too lazy to do it cleverly!
    RowMajorMatrixX2f image_points =
        image_points_3d.block(0, 0, image_points_3d.rows(), 2).array().colwise() / image_points_3d.col(2).array();

    // Apply the update
    image_points.row(update.pin_idx) = update.pos;

    const bool ok = SolvePnP(object_points, image_points, camera, result);
    if (!ok) {
        return std::nullopt;
    }

    RowMajorMatrix4f result_model_view = RowMajorMatrix4f::Identity();
    result_model_view.block<3, 3>(0, 0) = result.rotation;
    result_model_view.block<3, 1>(0, 3) = result.translation;

    switch (trans_type) {
        case TransformationType::Model:
            return scene_transform.view_matrix.inverse() * result_model_view;
        case TransformationType::Camera:
            return result_model_view * scene_transform.model_matrix.inverse();
        default:
            throw std::runtime_error(fmt::format("Invalid trans_type value: {}", static_cast<int>(trans_type)));
    }
}

static std::optional<RowMajorMatrix4f> FindTransformation1(const RefRowMajorMatrixX3f& object_points,
                                                           const SceneTransformations& scene_transform,
                                                           const PinUpdate& update, TransformationType trans_type) {
    CHECK(object_points.rows() == 1);
    // TODO
    CHECK(trans_type == TransformationType::Model);

    const Ray ray = GetRayWorldSpace(scene_transform, update.pos);

    const Eigen::Vector3f point_object_space = object_points.row(0);
    const Eigen::Vector3f point_world_space = Eigen::Affine3f(scene_transform.model_matrix) * point_object_space;

    const float depth = (point_world_space - ray.origin).norm();
    const Eigen::Vector3f translated_pos = ray.origin + depth * ray.dir.normalized();
    const Eigen::Vector3f translation = translated_pos - point_world_space;

    RowMajorMatrix4f result = scene_transform.model_matrix;
    result.block<3, 1>(0, 3) += translation;

    return result;
}

static std::optional<RowMajorMatrix4f> FindTransformation2(const RefRowMajorMatrixX3f& object_points,
                                                           const SceneTransformations& scene_transform,
                                                           const PinUpdate& update, TransformationType trans_type) {
    // FIXME: This entire function is just wrong!
    CHECK(object_points.rows() == 2);
    // TODO
    CHECK(trans_type == TransformationType::Model);

    const Ray ray = GetRayWorldSpace(scene_transform, update.pos);

    const Eigen::Vector3f moving_point_object_space = object_points.row(update.pin_idx);
    const Eigen::Vector3f anchor_point_object_space = object_points.row(1 - update.pin_idx);

    const Eigen::Vector3f moving_point_world_space =
        Eigen::Affine3f(scene_transform.model_matrix) * moving_point_object_space;

    const Eigen::Vector3f anchor_point_world_space =
        Eigen::Affine3f(scene_transform.model_matrix) * anchor_point_object_space;

    const float depth = (moving_point_world_space - ray.origin).norm();
    const Eigen::Vector3f translated_moving_point = ray.origin + depth * ray.dir.normalized();

    const Eigen::Vector3f du = (moving_point_world_space - anchor_point_world_space);
    const Eigen::Vector3f dv = (translated_moving_point - anchor_point_world_space);

    const float scale = dv.norm() / du.norm();

    const float norms_multiplied = du.norm() * dv.norm();
    const Eigen::Vector3f dn = du.cross(dv) / norms_multiplied;
    const float angle = std::atan2(dn.norm(), du.dot(dv) / norms_multiplied);

    const Eigen::AngleAxisf rot{angle, dn};

    return (Eigen::Translation3f(anchor_point_world_space) * Eigen::UniformScaling(scale) * rot *
            Eigen::Translation3f(-anchor_point_world_space)) *
           scene_transform.model_matrix;
}

std::optional<RowMajorMatrix4f> FindTransformation(const RefRowMajorMatrixX3f& object_points,
                                                   const SceneTransformations& scene_transform,
                                                   const PinUpdate& update, TransformationType trans_type) {
    CHECK(update.pin_idx < object_points.rows());

    switch (object_points.rows()) {
        case 1:
            return FindTransformation1(object_points, scene_transform, update, trans_type);
        case 2:
            return FindTransformation2(object_points, scene_transform, update, trans_type);
        default:
            return FindTransformationN(object_points, scene_transform, update, trans_type);
    }
}
