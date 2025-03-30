find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(glm REQUIRED)
find_package(embree 4 REQUIRED)
find_package(pybind11 REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)

add_library(
polychase
    geometry.h
    geometry.cc
    raycasting.h
    raycasting.cc
    utils.h
)

set_property(TARGET polychase PROPERTY CXX_STANDARD 20)
set_property(TARGET polychase PROPERTY POSITION_INDEPENDENT_CODE TRUE)

target_link_libraries(
polychase
PRIVATE
    "${OpenCV_LIBS}"
    embree
    spdlog::spdlog
    fmt::fmt
PUBLIC
    Eigen3::Eigen
    glm::glm
)

target_include_directories(
polychase
PRIVATE
    "${OpenCV_INCLUDE_DIRS}"
)

add_executable(main main.cc)
target_link_libraries(main PRIVATE polychase)

pybind11_add_module(polychase_core polychase_pybind.cc)
target_link_libraries(polychase_core PRIVATE polychase)
