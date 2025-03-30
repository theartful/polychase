cmake_minimum_required(VERSION 3.10)
project(polychase)

add_subdirectory(cpp)

install(TARGETS polychase_core EXPORT polychase_core-targets LIBRARY DESTINATION $<IF:$<PLATFORM_ID:Windows>,bin,lib>)
install(DIRECTORY blender_addon/ DESTINATION .)
