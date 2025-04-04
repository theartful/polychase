find_package(gflags REQUIRED)

add_executable(generate_database generate_database.cc)
target_link_libraries(generate_database PRIVATE gflags polychase)
target_include_directories(generate_database PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../")

add_executable(visualize_flow visualize_flow.cc)
target_link_libraries(visualize_flow PRIVATE gflags polychase)
target_include_directories(visualize_flow PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../")
