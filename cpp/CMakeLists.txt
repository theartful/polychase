find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs video)
find_package(Eigen3 REQUIRED)
find_package(embree 4 REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(TBB REQUIRED)

set(CMAKE_CXX_STANDARD 20)

if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
    add_compile_options(-Wall -Wextra -pedantic)
    add_link_options(-Wl,--no-whole-archive -Wl,--exclude-libs=ALL)
elseif(MSVC)
    add_compile_options(/D_USE_MATH_DEFINES /Gy)
    add_link_options(/OPT:REF /OPT:ICF)
endif()

add_subdirectory(external/pybind11)
add_subdirectory(external/cvnp)

add_library(
polychase
    utils.h
    utils.cc
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
    pnp/solvers.h
    pnp/solvers.cc
    feature_detection/gftt.h
    feature_detection/gftt.cc
)

set_property(TARGET polychase PROPERTY POSITION_INDEPENDENT_CODE TRUE)

target_link_libraries(
polychase
PUBLIC
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_video
    TBB::tbb
    embree
    SQLite::SQLite3
    Eigen3::Eigen
    spdlog::spdlog
    fmt::fmt
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
