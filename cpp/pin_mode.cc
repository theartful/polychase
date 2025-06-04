#include "pin_mode.h"

#include <spdlog/spdlog.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <cmath>

#include "pnp/solvers.h"
#include "ray_casting.h"

static SceneTransformations FindTransformationN(
    const RefConstRowMajorMatrixX3f& object_points,
    const SceneTransformations& initial_scene_transform,
    const SceneTransformations& current_scene_transform,
    const PinUpdate& update, TransformationType trans_type,
    bool optimize_focal_length, bool optimize_principal_point) {
    CHECK_GT(object_points.rows(), 2);

    // Step 1: Project points
    // FIXME: This step can be cached across invocations of this function.
    const RowMajorMatrix3f proj3x3_transpose =
        initial_scene_transform.intrinsics.To3x3ProjectionMatrix().transpose();

    const RowMajorMatrix4f model_view = initial_scene_transform.view_matrix *
                                        initial_scene_transform.model_matrix;
    const RowMajorMatrix3f model_view_rotation = model_view.block<3, 3>(0, 0);
    const Eigen::Vector3f model_view_translation = model_view.block<3, 1>(0, 3);

    const RowMajorMatrixX3f object_points_cameraspace =
        (object_points * model_view_rotation.transpose()).rowwise() +
        model_view_translation.transpose();

    const RowMajorMatrixX3f image_points_3d =
        object_points_cameraspace * proj3x3_transpose;
    RowMajorMatrixX2f image_points =
        image_points_3d.block(0, 0, image_points_3d.rows(), 2)
            .array()
            .colwise() /
        image_points_3d.col(2).array();

    // Apply the update
    image_points.row(update.pin_idx) = update.pos;

    // Step 2: Initialize the solution with current_scene_transform so that we
    // get smooth transitions.
    const RowMajorMatrix4f initial_pose =
        (current_scene_transform.view_matrix *
         current_scene_transform.model_matrix) *
        model_view.inverse();

    PnPResult result = {
        .camera =
            {
                .intrinsics = current_scene_transform.intrinsics,
                .pose = CameraPose::FromRt(initial_pose),
            },
        .bundle_stats = {},
    };

    BundleOptions bundle_opts = {};
    bundle_opts.loss_type = BundleOptions::LossType::TRIVIAL;

    const PnPOptions opts = {
        .bundle_opts = bundle_opts,
        .max_inlier_error = 0.0f,  // We don't care about inlier ratio here
        .optimize_focal_length = optimize_focal_length,
        .optimize_principal_point = optimize_principal_point,
    };

    SolvePnPIterative(object_points_cameraspace, image_points, opts, result);

    const RowMajorMatrix3f result_R = result.camera.pose.R();
    const Eigen::Vector3f result_t = result.camera.pose.t;

    switch (trans_type) {
        case TransformationType::Model: {
            RowMajorMatrix4f new_model_view = RowMajorMatrix4f::Identity();
            new_model_view.block<3, 3>(0, 0) = result_R * model_view_rotation;
            new_model_view.block<3, 1>(0, 3) =
                result_R * model_view_translation + result_t;
            return SceneTransformations{
                .model_matrix = initial_scene_transform.view_matrix.inverse() *
                                new_model_view,
                .view_matrix = current_scene_transform.view_matrix,
                .intrinsics = result.camera.intrinsics,
            };
        }
        case TransformationType::Camera: {
            RowMajorMatrix4f view_matrix_update = RowMajorMatrix4f::Identity();
            view_matrix_update.block<3, 3>(0, 0) = result_R;
            view_matrix_update.block<3, 1>(0, 3) = result_t;
            return SceneTransformations{
                .model_matrix = current_scene_transform.model_matrix,
                .view_matrix =
                    view_matrix_update * initial_scene_transform.view_matrix,
                .intrinsics = result.camera.intrinsics,
            };
        }
        default:
            throw std::runtime_error(fmt::format("Invalid trans_type value: {}",
                                                 static_cast<int>(trans_type)));
    }
}

static SceneTransformations FindTransformation1(
    const RefConstRowMajorMatrixX3f& object_points,
    const SceneTransformations& scene_transform, const PinUpdate& update,
    TransformationType trans_type) {
    CHECK_EQ(object_points.rows(), 1);

    const Ray ray = GetRayWorldSpace(scene_transform, update.pos);

    const Eigen::Vector3f point_object_space = object_points.row(0);
    const Eigen::Vector3f point_world_space =
        Eigen::Affine3f(scene_transform.model_matrix) * point_object_space;

    const float depth = (point_world_space - ray.origin).norm();
    const Eigen::Vector3f translated_pos =
        ray.origin + depth * ray.dir.normalized();
    const Eigen::Vector3f translation = translated_pos - point_world_space;

    RowMajorMatrix4f new_model_matrix = scene_transform.model_matrix;
    new_model_matrix.block<3, 1>(0, 3) += translation;

    switch (trans_type) {
        case TransformationType::Model:
            return SceneTransformations{
                .model_matrix = new_model_matrix,
                .view_matrix = scene_transform.view_matrix,
                .intrinsics = scene_transform.intrinsics,
            };
        case TransformationType::Camera:
            return SceneTransformations{
                .model_matrix = scene_transform.model_matrix,
                .view_matrix =
                    scene_transform.view_matrix *
                    (new_model_matrix * scene_transform.model_matrix.inverse()),
                .intrinsics = scene_transform.intrinsics,
            };
        default:
            throw std::runtime_error(fmt::format("Invalid trans_type value: {}",
                                                 static_cast<int>(trans_type)));
    }
}

static SceneTransformations FindTransformation2(
    const RefConstRowMajorMatrixX3f& object_points,
    const SceneTransformations& scene_transform, const PinUpdate& update,
    TransformationType trans_type) {
    CHECK_EQ(object_points.rows(), 2);

    const Ray ray = GetRayWorldSpace(scene_transform, update.pos);

    const RowMajorMatrix4f view_matrix_inverse =
        scene_transform.view_matrix.inverse();
    const Eigen::Vector3f camera_center = view_matrix_inverse.block<3, 1>(0, 3);

    const Eigen::Affine3f model_transform{scene_transform.model_matrix};

    const Eigen::Vector3f moving_point =
        model_transform * object_points.row(update.pin_idx).transpose();
    const Eigen::Vector3f anchor_point =
        model_transform * object_points.row(1 - update.pin_idx).transpose();

    const float depth = (moving_point - ray.origin).norm();
    const Eigen::Vector3f translated_moving_point =
        ray.origin + depth * ray.dir.normalized();

    const Eigen::Vector3f du = moving_point - anchor_point;
    const Eigen::Vector3f dv = translated_moving_point - anchor_point;
    const Eigen::Vector3f dn_unit =
        view_matrix_inverse.block<3, 1>(0, 2).normalized();
    const Eigen::Vector3f du_unit = du.normalized();
    const Eigen::Vector3f dv_unit = dv.normalized();
    const float angle =
        std::atan2(du_unit.cross(dv_unit).dot(dn_unit), du_unit.dot(dv_unit));

    const Eigen::AngleAxisf rot{angle, dn_unit};

    // We want to scale around the anchor point by "scale" factor.
    // This is equivalent to multiplying the distance between the camera center
    // and the anchor point by 1/s
    //
    // FIXME: Strictly speaking, this isn't exactly correct. You can see this by
    // passing "initial_scene_transform" instead of "current_scene_transform" to
    // this function
    const float scale_inv = du.norm() / dv.norm();
    const Eigen::Vector3f new_anchor_point =
        camera_center + (anchor_point - camera_center) * scale_inv;

    const auto update_transform = (Eigen::Translation3f(new_anchor_point) *
                                   rot * Eigen::Translation3f(-anchor_point));

    switch (trans_type) {
        case TransformationType::Model:
            return SceneTransformations{
                .model_matrix = (update_transform * model_transform).matrix(),
                .view_matrix = scene_transform.view_matrix,
                .intrinsics = scene_transform.intrinsics,
            };
        case TransformationType::Camera:
            return SceneTransformations{
                .model_matrix = scene_transform.model_matrix,
                .view_matrix =
                    (scene_transform.view_matrix * update_transform).matrix(),
                .intrinsics = scene_transform.intrinsics,
            };
        default:
            throw std::runtime_error(fmt::format("Invalid trans_type value: {}",
                                                 static_cast<int>(trans_type)));
    }
}

SceneTransformations FindTransformation(
    const RefConstRowMajorMatrixX3f& object_points,
    const SceneTransformations& initial_scene_transform,
    const SceneTransformations& current_scene_transform,
    const PinUpdate& update, TransformationType trans_type,
    bool optimize_focal_length, bool optimize_principal_point) {
    CHECK_LT(update.pin_idx, object_points.rows());

    switch (object_points.rows()) {
        case 1:
            return FindTransformation1(object_points, initial_scene_transform,
                                       update, trans_type);
        case 2:
            // This function is not entirely correct, so we're starting from
            // current_scene_transform instead of initial_scene_transform
            return FindTransformation2(object_points, current_scene_transform,
                                       update, trans_type);
        // TODO: Use P3P directly instead of the general iterative approach when
        // n = 3. case 3:
        //     return FindTransformation3(object_points,
        //     initial_scene_transform, update, trans_type);
        default:
            return FindTransformationN(object_points, initial_scene_transform,
                                       current_scene_transform, update,
                                       trans_type, optimize_focal_length,
                                       optimize_principal_point);
    }
}
