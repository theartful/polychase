find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs video)
find_package(Eigen3 REQUIRED)
find_package(embree 4 REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(TBB REQUIRED)
find_package(pybind11 REQUIRED)

set(CMAKE_CXX_STANDARD 20)

if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
    add_compile_options(-Wall -Wextra -pedantic)
endif()

if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    add_compile_options(/D_USE_MATH_DEFINES /Gy)
    add_link_options(/OPT:REF /OPT:ICF)
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
    refiner.h
    refiner.cc
    pnp/pnp_opengl.h
    pnp/pnp_opengl.cc
    pnp/solvers.h
    pnp/solvers.cc
)

set_property(TARGET polychase PROPERTY POSITION_INDEPENDENT_CODE TRUE)

target_link_libraries(
polychase
PUBLIC
    TBB::tbb
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_video
    embree
    spdlog::spdlog
    fmt::fmt
    SQLite::SQLite3
    Eigen3::Eigen
)

target_include_directories(
polychase
PUBLIC
    "${OpenCV_INCLUDE_DIRS}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

pybind11_add_module(
    polychase_core
        polychase_pybind.cc
        tracker_thread.h
        refiner_thread.h
        opticalflow_thread.h
)

target_link_libraries(
polychase_core
PRIVATE
    polychase
    cvnp
)
target_include_directories(
polychase_core
PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/external/cvnp"
)

option(BUILD_EXAMPLES "Build examples" OFF)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
