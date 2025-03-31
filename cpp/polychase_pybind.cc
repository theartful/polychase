#include "geometry.h"
#include "pin_mode.h"
#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "ray_casting.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(AcceleratedMeshSptr)

PYBIND11_MODULE(polychase_core, m) {
    py::class_<Mesh>(m, "Mesh")
        .def_readwrite("vertices", &Mesh::vertices)  //
        .def_readwrite("indices", &Mesh::indices);

    py::class_<AcceleratedMeshSptr> _(m, "AcceleratedMesh");

    py::class_<SceneTransformations>(m, "SceneTransformations")
        .def(py::init<RowMajorMatrix4f, RowMajorMatrix4f, RowMajorMatrix4f>(), py::arg("model_matrix"),
             py::arg("view_matrix"), py::arg("projection_matrix"))
        .def_readwrite("model_matrix", &SceneTransformations::model_matrix)
        .def_readwrite("view_matrix", &SceneTransformations::view_matrix)
        .def_readwrite("projection_matrix", &SceneTransformations::projection_matrix);

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

    py::enum_<TransformationType>(m, "TransformationType")
        .value("Camera", TransformationType::Camera)
        .value("Model", TransformationType::Model);

    m.def("create_mesh", CreateMesh);

    m.def("create_accelerated_mesh", py::overload_cast<MeshSptr>(&CreateAcceleratedMesh));
    m.def("create_accelerated_mesh", py::overload_cast<RowMajorArrayX3f, RowMajorArrayX3u>(&CreateAcceleratedMesh));

    m.def("ray_cast", py::overload_cast<const AcceleratedMeshSptr&, Eigen::Vector3f, Eigen::Vector3f>(RayCast),
          py::arg("accel_mesh"), py::arg("ray_origin"), py::arg("ray_direction"));
    m.def("ray_cast",
          py::overload_cast<const AcceleratedMeshSptr&, const SceneTransformations&, Eigen::Vector2f>(RayCast),
          py::arg("accel_mesh"), py::arg("scene_transform"), py::arg("ndc_pos"));

    m.def("find_transformation", FindTransformation, py::arg("object_points").noconvert(), py::arg("scene_transform"),
          py::arg("update"), py::arg("trans_type"));
}
