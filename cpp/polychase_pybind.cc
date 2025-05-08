#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "camera_trajectory.h"
#include "cvnp/cvnp.h"  // IWYU pragma: keep
#include "database.h"
#include "geometry.h"
#include "opticalflow.h"
#include "pin_mode.h"
#include "pnp/solvers.h"
#include "pnp/types.h"
#include "pybind11_extension.h"  // IWYU pragma: keep
#include "ray_casting.h"
#include "tracker.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(AcceleratedMeshSptr)

template <size_t CacheSize>
struct SequentialWrapper {
    SequentialWrapper(FrameAccessorFunction accessor) : accessor{std::move(accessor)} {}

    std::optional<cv::Mat> RequestFrame(uint32_t frame_id) {
        if (invalid) {
            return std::nullopt;
        }
        std::optional<cv::Mat> maybe_frame = accessor(frame_id);
        if (!maybe_frame) {
            invalid = true;
        }
        return maybe_frame;
    }

    std::optional<cv::Mat> operator()(uint32_t frame_id) {
        const size_t frame_idx = frame_id % CacheSize;

        if (highest_frame_id == INVALID_ID) {
            highest_frame_id = frame_id;
            frames[frame_idx] = accessor(frame_id);
            return frames[frame_idx];
        }

        if (frame_id <= highest_frame_id) {
            CHECK(highest_frame_id - frame_id < CacheSize);
            return frames[frame_idx];
        }

        CHECK(frame_id - highest_frame_id < CacheSize);

        for (uint32_t id = highest_frame_id + 1; id <= frame_id; id++) {
            const size_t idx = id % CacheSize;
            frames[idx] = RequestFrame(id);
        }

        highest_frame_id = frame_id;
        return frames[frame_idx];
    }

    FrameAccessorFunction accessor;
    uint32_t highest_frame_id = INVALID_ID;
    bool invalid = false;
    std::optional<cv::Mat> frames[CacheSize];
};

void GenerateOpticalFlowDatabaseWrapper(const VideoInfo& video_info, FrameAccessorFunction frame_accessor,
                                        ProgressCallback callback, const std::string& database_path,
                                        const FeatureDetectorOptions& detector_options,
                                        const OpticalFlowOptions& flow_options, bool write_images) {
    GenerateOpticalFlowDatabase(video_info, SequentialWrapper<17>(std::move(frame_accessor)), std::move(callback),
                                database_path, detector_options, flow_options, write_images);
}

PYBIND11_MODULE(polychase_core, m) {
    py::class_<Mesh>(m, "Mesh")
        .def_readwrite("vertices", &Mesh::vertices)  //
        .def_readwrite("indices", &Mesh::indices);

    py::class_<AcceleratedMeshSptr> _(m, "AcceleratedMesh");

    py::class_<SceneTransformations>(m, "SceneTransformations")
        .def(py::init<RowMajorMatrix4f, RowMajorMatrix4f, CameraIntrinsics>(), py::arg("model_matrix"),
             py::arg("view_matrix"), py::arg("intrinsics"))
        .def_readwrite("model_matrix", &SceneTransformations::model_matrix)
        .def_readwrite("view_matrix", &SceneTransformations::view_matrix)
        .def_readwrite("intrinsics", &SceneTransformations::intrinsics);

    py::class_<RayHit>(m, "RayHit")
        .def_readwrite("pos", &RayHit::pos)
        .def_readwrite("normal", &RayHit::normal)
        .def_readwrite("barycentric_coordinate", &RayHit::barycentric_coordinate)
        .def_readwrite("t", &RayHit::t)
        .def_readwrite("primitive_id", &RayHit::primitive_id);

    py::class_<PinUpdate>(m, "PinUpdate")
        .def(py::init<uint32_t, Eigen::Vector2f>(), py::arg("pin_idx"), py::arg("pin_pos"))
        .def_readwrite("pin_idx", &PinUpdate::pin_idx)
        .def_readwrite("pos", &PinUpdate::pos);

    py::class_<Database>(m, "Database")
        .def(py::init<const std::string&>(), py::arg("path"))
        .def("open", &Database::Open, py::arg("path"))
        .def("close", &Database::Close)
        .def("read_keypoints", py::overload_cast<int32_t>(&Database::ReadKeypoints, py::const_), py::arg("image_id"))
        .def("write_keypoints", &Database::WriteKeypoints, py::arg("image_id"), py::arg("keypoints"))
        .def("read_image_pair_flow", py::overload_cast<int32_t, int32_t>(&Database::ReadImagePairFlow, py::const_),
             py::arg("image_id_from"), py::arg("image_id_to"))
        .def("write_image_pair_flow",
             py::overload_cast<int32_t, int32_t, const KeypointsIndices&, const Keypoints&, const FlowErrors&>(
                 &Database::WriteImagePairFlow),
             py::arg("image_id_from"), py::arg("image_id_to"), py::arg("src_kps_indices"), py::arg("tgt_kps"),
             py::arg("flow_errors"))
        .def("write_image_pair_flow", py::overload_cast<const ImagePairFlow&>(&Database::WriteImagePairFlow),
             py::arg("image_pair_flow"))
        .def("find_optical_flows_from_image",
             py::overload_cast<int32_t>(&Database::FindOpticalFlowsFromImage, py::const_), py::arg("image_id_from"))
        .def("find_optical_flows_to_image", py::overload_cast<int32_t>(&Database::FindOpticalFlowsToImage, py::const_),
             py::arg("image_id_to"))
        .def("keypoints_exist", &Database::KeypointsExist, py::arg("image_id"))
        .def("image_pair_flow_exists", &Database::ImagePairFlowExists, py::arg("image_id_from"),
             py::arg("image_id_to"))
        .def("get_min_image_id_with_keypoints", &Database::GetMinImageIdWithKeypoints)
        .def("get_max_image_id_with_keypoints", &Database::GetMaxImageIdWithKeypoints);

    py::class_<ImagePairFlow>(m, "ImagePairFlow")
        .def(py::init<>())
        .def_readwrite("image_id_from", &ImagePairFlow::image_id_from)
        .def_readwrite("image_id_to", &ImagePairFlow::image_id_to)
        .def_readwrite("src_kps_indices", &ImagePairFlow::src_kps_indices)
        .def_readwrite("tgt_kps", &ImagePairFlow::tgt_kps)
        .def_readwrite("flow_errors", &ImagePairFlow::flow_errors);

    py::class_<VideoInfo>(m, "VideoInfo")
        .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t>(), py::arg("width"), py::arg("height"),
             py::arg("first_frame"), py::arg("num_frames"))
        .def_readwrite("width", &VideoInfo::width)
        .def_readwrite("height", &VideoInfo::height)
        .def_readwrite("first_frame", &VideoInfo::first_frame)
        .def_readwrite("num_frames", &VideoInfo::num_frames);

    py::class_<FeatureDetectorOptions>(m, "FeatureDetectorOptions")
        .def(py::init<>())
        .def_readwrite("quality_level", &FeatureDetectorOptions::quality_level)
        .def_readwrite("min_distance", &FeatureDetectorOptions::min_distance)
        .def_readwrite("block_size", &FeatureDetectorOptions::block_size)
        .def_readwrite("gradient_size", &FeatureDetectorOptions::gradient_size)
        .def_readwrite("max_corners", &FeatureDetectorOptions::max_corners)
        .def_readwrite("use_harris", &FeatureDetectorOptions::use_harris)
        .def_readwrite("k", &FeatureDetectorOptions::k);

    py::class_<OpticalFlowOptions>(m, "OpticalFlowOptions")
        .def(py::init<>())
        .def_readwrite("window_size", &OpticalFlowOptions::window_size)
        .def_readwrite("max_level", &OpticalFlowOptions::max_level)
        .def_readwrite("term_epsilon", &OpticalFlowOptions::term_epsilon)
        .def_readwrite("term_max_iters", &OpticalFlowOptions::term_max_iters)
        .def_readwrite("min_eigen_threshold", &OpticalFlowOptions::min_eigen_threshold);

    py::enum_<TransformationType>(m, "TransformationType")
        .value("Camera", TransformationType::Camera)
        .value("Model", TransformationType::Model);

    py::enum_<CameraConvention>(m, "CameraConvention")
        .value("OpenGL", CameraConvention::OpenGL)
        .value("OpenCV", CameraConvention::OpenCV);

    py::class_<CameraIntrinsics>(m, "CameraIntrinsics")
        .def(py::init<float, float, float, float, float, float, float, CameraConvention>(), py::arg("fx"),
             py::arg("fy"), py::arg("cx"), py::arg("cy"), py::arg("aspect_ratio"), py::arg("width"), py::arg("height"),
             py::arg("convention") = CameraConvention::OpenGL)
        .def_readwrite("fx", &CameraIntrinsics::fx)
        .def_readwrite("fy", &CameraIntrinsics::fy)
        .def_readwrite("cx", &CameraIntrinsics::cx)
        .def_readwrite("cy", &CameraIntrinsics::cy)
        .def_readwrite("aspect_ratio", &CameraIntrinsics::aspect_ratio)
        .def_readwrite("width", &CameraIntrinsics::width)
        .def_readwrite("height", &CameraIntrinsics::height)
        .def_readwrite("convention", &CameraIntrinsics::convention);

    py::class_<CameraPose>(m, "CameraPose")
        .def(py::init<>())
        .def("inverse", &CameraPose::Inverse)
        .def_readwrite("q", &CameraPose::q)
        .def_readwrite("t", &CameraPose::t);

    py::class_<CameraState>(m, "CameraState")
        .def(py::init<CameraIntrinsics, CameraPose>(), py::arg("intrinsics"), py::arg("pose"))
        .def_readwrite("intrinsics", &CameraState::intrinsics)
        .def_readwrite("pose", &CameraState::pose);

    py::class_<BundleStats>(m, "BundleStats")
        .def(py::init<>())
        .def_readwrite("iterations", &BundleStats::iterations)
        .def_readwrite("initial_cost", &BundleStats::initial_cost)
        .def_readwrite("cost", &BundleStats::cost)
        .def_readwrite("lambda", &BundleStats::lambda)
        .def_readwrite("invalid_steps", &BundleStats::invalid_steps)
        .def_readwrite("step_norm", &BundleStats::step_norm)
        .def_readwrite("grad_norm", &BundleStats::grad_norm);

    py::class_<RansacStats>(m, "RansacStats")
        .def(py::init<>())
        .def_readwrite("refinements", &RansacStats::refinements)
        .def_readwrite("iterations", &RansacStats::iterations)
        .def_readwrite("num_inliers", &RansacStats::num_inliers)
        .def_readwrite("inlier_ratio", &RansacStats::inlier_ratio)
        .def_readwrite("model_score", &RansacStats::model_score);

    py::class_<PnPResult>(m, "PnPResult")
        .def_readwrite("camera", &PnPResult::camera)
        .def_readwrite("bundle_stats", &PnPResult::bundle_stats)
        .def_readwrite("ransac_stats", &PnPResult::ransac_stats);

    py::class_<FrameTrackingResult>(m, "FrameTrackingResult")
        .def_readwrite("frame", &FrameTrackingResult::frame)
        .def_readwrite("pose", &FrameTrackingResult::pose)
        .def_readwrite("intrinsics", &FrameTrackingResult::intrinsics)
        .def_readwrite("ransac_stats", &FrameTrackingResult::ransac_stats)
        .def_readwrite("bundle_stats", &FrameTrackingResult::bundle_stats);

    py::class_<CameraTrajectory>(m, "CameraTrajectory")
        .def(py::init<int32_t, size_t>(), py::arg("first_frame_id"), py::arg("count"))
        .def("is_valid_frame", &CameraTrajectory::IsValidFrame, py::arg("frame_id"))
        .def("is_frame_filled", &CameraTrajectory::IsFrameFilled, py::arg("frame_id"))
        .def("get", &CameraTrajectory::Get, py::arg("frame_id"))
        .def("set", &CameraTrajectory::Set, py::arg("frame_id"), py::arg("state"))
        .def("count", &CameraTrajectory::Count)
        .def("first_frame", &CameraTrajectory::FirstFrame)
        .def("last_frame", &CameraTrajectory::LastFrame);

    m.def("create_mesh", CreateMesh);

    m.def("create_accelerated_mesh", py::overload_cast<MeshSptr>(&CreateAcceleratedMesh), py::arg("mesh"));
    m.def("create_accelerated_mesh", py::overload_cast<RowMajorArrayX3f, RowMajorArrayX3u>(&CreateAcceleratedMesh),
          py::arg("vertices"), py::arg("indices"));

    m.def("ray_cast", py::overload_cast<const AcceleratedMeshSptr&, Eigen::Vector3f, Eigen::Vector3f>(RayCast),
          py::arg("accel_mesh"), py::arg("ray_origin"), py::arg("ray_direction"));
    m.def("ray_cast",
          py::overload_cast<const AcceleratedMeshSptr&, const SceneTransformations&, Eigen::Vector2f>(RayCast),
          py::arg("accel_mesh"), py::arg("scene_transform"), py::arg("pos"));

    m.def("find_transformation", FindTransformation, py::arg("object_points").noconvert(),
          py::arg("initial_scene_transform"), py::arg("current_scene_transform"), py::arg("update"),
          py::arg("trans_type"), py::arg("optimize_focal_length") = false,
          py::arg("optimize_principal_point") = false);

    m.def("generate_optical_flow_database", GenerateOpticalFlowDatabaseWrapper, py::arg("video_info"),
          py::arg("frame_accessor_function"), py::arg("callback"), py::arg("database_path"),
          py::arg("detector_options") = FeatureDetectorOptions{}, py::arg("flow_options") = OpticalFlowOptions{},
          py::arg("write_images") = false, py::call_guard<py::gil_scoped_release>());

    m.def("track_sequence", TrackSequence, py::arg("database_path"), py::arg("frame_from"),
          py::arg("frame_to_inclusive"), py::arg("scene_transform"), py::arg("accel_mesh"), py::arg("trans_type"),
          py::arg("callback"), py::arg("optimize_focal_length") = false, py::arg("optimize_principal_point") = false,
          py::call_guard<py::gil_scoped_release>());
}
