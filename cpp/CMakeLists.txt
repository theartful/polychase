find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(glm REQUIRED)
find_package(embree 4 REQUIRED)
find_package(pybind11 REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(PoseLib REQUIRED)

set(CMAKE_CXX_STANDARD 20)

if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
    add_compile_options(-Wall -Wextra -pedantic)
endif()

add_subdirectory(external/cvnp)

add_library(
polychase
    utils.h
    geometry.h
    ray_casting.h
    ray_casting.cc
    pin_mode.h
    pin_mode.cc
    database.h
    database.cc
    opticalflow.h
    opticalflow.cc
    forward_solver.h
    forward_solver.cc
    pnp/pnp_opengl.h
    pnp/pnp_opengl.cc
    pnp/solvers.h
    pnp/solvers.cc
)

set_property(TARGET polychase PROPERTY POSITION_INDEPENDENT_CODE TRUE)

target_link_libraries(
polychase
PRIVATE
    "${OpenCV_LIBS}"
    embree
    spdlog::spdlog
    fmt::fmt
    SQLite::SQLite3
    PoseLib::PoseLib
PUBLIC
    Eigen3::Eigen
    glm::glm
)

target_include_directories(
polychase
PRIVATE
    "${OpenCV_INCLUDE_DIRS}"
PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_executable(main main.cc)
target_link_libraries(main PRIVATE polychase)

pybind11_add_module(polychase_core polychase_pybind.cc)
target_link_libraries(
polychase_core
PRIVATE
    polychase
    embree
    cvnp
)
target_include_directories(
polychase_core
PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/external/cvnp"
)

add_subdirectory(examples)
