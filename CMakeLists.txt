cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0087 NEW)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

project(polychase)

add_subdirectory(cpp)

# Change the behavior of install when using vcpkg to install vcpkg dependencies as well
if(DEFINED VCPKG_INSTALLED_DIR)
    include(PolychaseVcpkg)
endif()

install(TARGETS polychase_core EXPORT polychase_core-targets LIBRARY DESTINATION $<IF:$<PLATFORM_ID:Windows>,bin,lib>)
install(DIRECTORY blender_addon/ DESTINATION .)
