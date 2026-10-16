cmake_minimum_required(VERSION 3.24)
project(apex LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   

# metal cpp header. 
add_library(metal_cpp INTERFACE)
target_include_directories(metal_cpp INTERFACE third_party/metal-cpp)
target_link_libraries(metal_cpp INTERFACE
  "-framework Foundation" "-framework Metal" "-framework QuartzCore")

# Header-only library: there is no separate .cpp file to compile.
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE third_party)

include(cmake/MetalShaders.cmake)
add_metal_library(apex_shaders OUTPUT apex.metallib
  SOURCES shaders/norm.metal shaders/matmul.metal shaders/geglu.metal
  DEPENDS shaders/quant.h)

# main stuff
add_executable(apex src/main.cpp src/metal_impl.cpp src/backend/metal_backend.cpp
  src/tests/test_norm.cpp src/tests/dense_mlp.cpp)
target_link_libraries(apex PRIVATE metal_cpp nlohmann_json)
add_dependencies(apex apex_shaders)

# Dev only
target_compile_definitions(apex PRIVATE APEX_METALLIB="${CMAKE_BINARY_DIR}/apex.metallib")
