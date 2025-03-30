#include "geometry.h"
#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "raycasting.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(AcceleratedMeshSptr)

PYBIND11_MODULE(polychase_core, m) {
    py::class_<Mesh>(m, "Mesh")
        .def_readwrite("vertices", &Mesh::vertices)  //
        .def_readwrite("indices", &Mesh::indices);

    py::class_<SceneTransformations>(m, "SceneTransformations")
        .def(py::init<>())
        .def_readwrite("model_matrix", &SceneTransformations::model_matrix)
        .def_readwrite("view_matrix", &SceneTransformations::view_matrix)
        .def_readwrite("projection_matrix", &SceneTransformations::projection_matrix)
        .def_readwrite("window_width", &SceneTransformations::window_width)
        .def_readwrite("window_height", &SceneTransformations::window_height);

    py::class_<Mesh>(m, "Mesh")
        .def_readwrite("vertices", &Mesh::vertices)  //
        .def_readwrite("indices", &Mesh::indices);

    py::class_<RayHit>(m, "RayHit")
        .def_readwrite("pos", &RayHit::pos)
        .def_readwrite("normal", &RayHit::normal)
        .def_readwrite("barycentric_coordinate", &RayHit::barycentric_coordinate)
        .def_readwrite("t", &RayHit::t)
        .def_readwrite("primitive_id", &RayHit::primitive_id);

    m.def("create_mesh", CreateMesh);

    m.def("create_accelerated_mesh", CreateAcceleratedMesh);

    m.def("ray_cast", RayCast);
}
