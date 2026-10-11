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

include(cmake/MetalShaders.cmake)
add_metal_library(apex_shaders OUTPUT apex.metallib SOURCES shaders/hello.metal)

# main stuff
add_executable(apex src/main.cpp src/metal_impl.cpp)
target_link_libraries(apex PRIVATE metal_cpp)
add_dependencies(apex apex_shaders)

# Dev only
target_compile_definitions(apex PRIVATE APEX_METALLIB="${CMAKE_BINARY_DIR}/apex.metallib")
