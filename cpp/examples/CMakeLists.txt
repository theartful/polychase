find_package(gflags REQUIRED)

add_executable(generate_database generate_database.cc)
target_link_libraries(generate_database PRIVATE gflags polychase spdlog::spdlog)
target_include_directories(generate_database PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../")

add_executable(visualize_flow visualize_flow.cc)
target_link_libraries(visualize_flow PRIVATE gflags polychase spdlog::spdlog)
target_include_directories(visualize_flow PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../")

add_executable(levmarq_ill_conditioned_float32_issue levmarq_ill_conditioned_float32_issue)
target_link_libraries(levmarq_ill_conditioned_float32_issue Eigen3::Eigen)
