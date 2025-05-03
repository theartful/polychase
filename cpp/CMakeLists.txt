find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs tracking)
find_package(Eigen3 REQUIRED)
find_package(embree 4 REQUIRED)
find_package(pybind11 REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(PoseLib REQUIRED)
find_package(CHOLMOD REQUIRED)
find_package(Ceres REQUIRED)

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
    tracker.h
    tracker.cc
    pnp/pnp_opengl.h
    pnp/pnp_opengl.cc
    pnp/solvers.h
    pnp/solvers.cc
)

set_property(TARGET polychase PROPERTY POSITION_INDEPENDENT_CODE TRUE)

target_link_libraries(
polychase
PRIVATE
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_tracking
    embree
    spdlog::spdlog
    fmt::fmt
    SQLite::SQLite3
    PoseLib::PoseLib
    Ceres::ceres
    SuiteSparse::CHOLMOD
PUBLIC
    Eigen3::Eigen
)

target_include_directories(
polychase
PUBLIC
    "${OpenCV_INCLUDE_DIRS}"
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
    opencv_core
    cvnp
)
target_include_directories(
polychase_core
PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/external/cvnp"
)

add_subdirectory(examples)
