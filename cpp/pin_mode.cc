#include "pin_mode.h"

#include <spdlog/spdlog.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <cmath>
#include <opencv2/calib3d.hpp>

#include "pnp.h"
#include "ray_casting.h"

static std::optional<RowMajorMatrix4f> FindTransformationN(const ConstRefRowMajorMatrixX3f& object_points,
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
    const RowMajorMatrix3f model_view_rotation = model_view.block<3, 3>(0, 0);
    const Eigen::Vector3f model_view_translation = model_view.block<3, 1>(0, 3);

    // Project points
    const RowMajorMatrixX3f transformed_object_points =
        (object_points * model_view_rotation.transpose()).rowwise() + model_view_translation.transpose();

    // Project points
    const RowMajorMatrixX3f image_points_3d = transformed_object_points * K_transpose;

    // FIXME: Maybe find some way that doesn't require allocating a new array
    // the problem is cv::solvePnP requires the data to be contiguous, and I'm too lazy to do it cleverly!
    RowMajorMatrixX2f image_points =
        image_points_3d.block(0, 0, image_points_3d.rows(), 2).array().colwise() / image_points_3d.col(2).array();

    // Apply the update
    image_points.row(update.pin_idx) = update.pos;

    PnPResult result = {.translation = Eigen::Vector3f(0.0, 0.0, 0.0), .rotation = RowMajorMatrix3f::Identity()};
    const bool ok = SolvePnP(transformed_object_points, image_points, camera, result);
    if (!ok) {
        return std::nullopt;
    }

    switch (trans_type) {
        case TransformationType::Model: {
            RowMajorMatrix4f new_model_view = RowMajorMatrix4f::Identity();
            new_model_view.block<3, 3>(0, 0) = result.rotation * model_view_rotation;
            new_model_view.block<3, 1>(0, 3) = result.rotation * model_view_translation + result.translation;
            return scene_transform.view_matrix.inverse() * new_model_view;
        }
        case TransformationType::Camera: {
            RowMajorMatrix4f view_matrix_update = RowMajorMatrix4f::Identity();
            view_matrix_update.block<3, 3>(0, 0) = result.rotation;
            view_matrix_update.block<3, 1>(0, 3) = result.translation;
            return view_matrix_update * scene_transform.view_matrix;
        }
        default:
            throw std::runtime_error(fmt::format("Invalid trans_type value: {}", static_cast<int>(trans_type)));
    }
}

static std::optional<RowMajorMatrix4f> FindTransformation1(const ConstRefRowMajorMatrixX3f& object_points,
                                                           const SceneTransformations& scene_transform,
                                                           const PinUpdate& update, TransformationType trans_type) {
    CHECK(object_points.rows() == 1);

    const Ray ray = GetRayWorldSpace(scene_transform, update.pos);

    const Eigen::Vector3f point_object_space = object_points.row(0);
    const Eigen::Vector3f point_world_space = Eigen::Affine3f(scene_transform.model_matrix) * point_object_space;

    const float depth = (point_world_space - ray.origin).norm();
    const Eigen::Vector3f translated_pos = ray.origin + depth * ray.dir.normalized();
    const Eigen::Vector3f translation = translated_pos - point_world_space;

    RowMajorMatrix4f new_model_matrix = scene_transform.model_matrix;
    new_model_matrix.block<3, 1>(0, 3) += translation;

    switch (trans_type) {
        case TransformationType::Model:
            return new_model_matrix;
        case TransformationType::Camera:
            return scene_transform.view_matrix * (new_model_matrix * scene_transform.model_matrix.inverse());
        default:
            throw std::runtime_error(fmt::format("Invalid trans_type value: {}", static_cast<int>(trans_type)));
    }
}

static std::optional<RowMajorMatrix4f> FindTransformation2(const ConstRefRowMajorMatrixX3f& object_points,
                                                           const SceneTransformations& scene_transform,
                                                           const PinUpdate& update, TransformationType trans_type) {
    CHECK(object_points.rows() == 2);

    const Ray ray = GetRayWorldSpace(scene_transform, update.pos);

    const RowMajorMatrix4f view_matrix_inverse = scene_transform.view_matrix.inverse();
    const Eigen::Vector3f camera_center = view_matrix_inverse.block<3, 1>(0, 3);

    const Eigen::Vector3f moving_point =
        (scene_transform.model_matrix * object_points.row(update.pin_idx).transpose().homogeneous()).hnormalized();
    const Eigen::Vector3f anchor_point =
        (scene_transform.model_matrix * object_points.row(1 - update.pin_idx).transpose().homogeneous()).hnormalized();

    const float depth = (moving_point - ray.origin).norm();
    const Eigen::Vector3f translated_moving_point = ray.origin + depth * ray.dir.normalized();

    const Eigen::Vector3f du = moving_point - anchor_point;
    const Eigen::Vector3f dv = translated_moving_point - anchor_point;
    const Eigen::Vector3f dn = view_matrix_inverse.block<3, 1>(0, 2);
    const Eigen::Vector3f du_unit = du.normalized();
    const Eigen::Vector3f dv_unit = dv.normalized();
    const float angle = std::atan2(du_unit.cross(dv_unit).dot(dn), du.dot(dv_unit));

    const Eigen::AngleAxisf rot{angle, dn.normalized()};

    // We want to scale around the anchor point by "scale" factor.
    // This is equivalent to multiplying the distance between the camera center and the anchor point by 1/s
    const float scale_inv = du.norm() / dv.norm();
    const Eigen::Vector3f new_anchor_point = camera_center + (anchor_point - camera_center) * scale_inv;

    switch (trans_type) {
        case TransformationType::Model:
            return (Eigen::Translation3f(new_anchor_point) * rot * Eigen::Translation3f(-anchor_point)) *
                   scene_transform.model_matrix;
        case TransformationType::Camera:
            return scene_transform.view_matrix *
                   (Eigen::Translation3f(new_anchor_point) * rot * Eigen::Translation3f(-anchor_point)).matrix();
        default:
            throw std::runtime_error(fmt::format("Invalid trans_type value: {}", static_cast<int>(trans_type)));
    }
}

std::optional<RowMajorMatrix4f> FindTransformation(const ConstRefRowMajorMatrixX3f& object_points,
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
