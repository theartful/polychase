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
                                                           const SceneTransformations& initial_scene_transform,
                                                           const SceneTransformations& current_scene_transform,
                                                           const PinUpdate& update, TransformationType trans_type) {
    CHECK(object_points.rows() > 2);

    // Step 1: Project points
    // FIXME: This step can be cached across invocations of this function.
    const CameraIntrinsics camera = {
        .fx = current_scene_transform.projection_matrix.coeff(0, 0),
        .fy = current_scene_transform.projection_matrix.coeff(1, 1),
        .cx = current_scene_transform.projection_matrix.coeff(2, 0),
        .cy = current_scene_transform.projection_matrix.coeff(2, 1),
    };
    const RowMajorMatrix3f K_transpose = CreateOpenGLCameraIntrinsicsMatrix(camera).transpose();

    const RowMajorMatrix4f model_view = initial_scene_transform.view_matrix * initial_scene_transform.model_matrix;
    const RowMajorMatrix3f model_view_rotation = model_view.block<3, 3>(0, 0);
    const Eigen::Vector3f model_view_translation = model_view.block<3, 1>(0, 3);

    const RowMajorMatrixX3f object_points_worldspace =
        (object_points * model_view_rotation.transpose()).rowwise() + model_view_translation.transpose();

    const RowMajorMatrixX3f image_points_3d = object_points_worldspace * K_transpose;
    RowMajorMatrixX2f image_points =
        image_points_3d.block(0, 0, image_points_3d.rows(), 2).array().colwise() / image_points_3d.col(2).array();

    // Apply the update
    image_points.row(update.pin_idx) = update.pos;

    // Step 2: Initialize the solution with current_scene_transform so that we get smooth transitions.
    const RowMajorMatrix4f initial_solution =
        (current_scene_transform.view_matrix * current_scene_transform.model_matrix) * model_view.inverse();

    PnPResult result = {.translation = initial_solution.block<3, 1>(0, 3),
                        .rotation = initial_solution.block<3, 3>(0, 0)};
    const bool ok = SolvePnP(object_points_worldspace, image_points, camera, PnPSolveMethod::Iterative, result);
    if (!ok) {
        return std::nullopt;
    }

    switch (trans_type) {
        case TransformationType::Model: {
            RowMajorMatrix4f new_model_view = RowMajorMatrix4f::Identity();
            new_model_view.block<3, 3>(0, 0) = result.rotation * model_view_rotation;
            new_model_view.block<3, 1>(0, 3) = result.rotation * model_view_translation + result.translation;
            return initial_scene_transform.view_matrix.inverse() * new_model_view;
        }
        case TransformationType::Camera: {
            RowMajorMatrix4f view_matrix_update = RowMajorMatrix4f::Identity();
            view_matrix_update.block<3, 3>(0, 0) = result.rotation;
            view_matrix_update.block<3, 1>(0, 3) = result.translation;
            return view_matrix_update * initial_scene_transform.view_matrix;
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
                                                   const SceneTransformations& initial_scene_transform,
                                                   const SceneTransformations& current_scene_transform,
                                                   const PinUpdate& update, TransformationType trans_type) {
    CHECK(update.pin_idx < object_points.rows());

    switch (object_points.rows()) {
        case 1:
            return FindTransformation1(object_points, current_scene_transform, update, trans_type);
        case 2:
            return FindTransformation2(
                object_points,
                // current_scene_transform would work, but error would accumulate, so the anchor point might move.
                // initial_scene_transform would work, but if the user changed the projection for some reason while
                // dragging (by panning/scaling for example), the results would be incorrect.
                SceneTransformations{.model_matrix = initial_scene_transform.model_matrix,
                                     .view_matrix = initial_scene_transform.view_matrix,
                                     .projection_matrix = current_scene_transform.projection_matrix},
                update, trans_type);
        default:
            return FindTransformationN(object_points, initial_scene_transform, current_scene_transform, update,
                                       trans_type);
    }
}
