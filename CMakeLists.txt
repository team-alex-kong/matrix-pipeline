cmake_minimum_required(VERSION 3.24)

# Supports RTX 30-series (86), RTX 40-series (89), and A100 (80)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89)
project(matrix-pipeline LANGUAGES CXX CUDA)
add_definitions(-DPROJECT_NAME="${PROJECT_NAME}")

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Always generate compile_commands.json for clangd, etc.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/compiler-flags.cmake)
include(cmake/sanitizers.cmake)
include(cmake/git.cmake)

find_package(OpenCV REQUIRED COMPONENTS)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV found at: ${OpenCV_DIR}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")

add_subdirectory(tests)
add_subdirectory(src/matrix-pipeline)
add_subdirectory(src/tools/yunet)
add_subdirectory(src/tools/rtsp)
add_subdirectory(src/tools/yunet_leak_test)