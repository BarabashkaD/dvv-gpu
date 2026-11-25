# Consumer of dvv_gpu; built by run-package-test.cmake. With DVV_GPU_SOURCE_DIR it takes dvv_gpu
# in-tree through FetchContent (as a dependent library or the app takes it; then
# DVV_BUILD_CMAKE_DIR names dvv-build's cmake folder), otherwise the installed package.
cmake_minimum_required(VERSION 4.2)
project(dvv_gpu_package_consumer LANGUAGES CXX)

if(DVV_GPU_SOURCE_DIR)
    include(FetchContent)
    FetchContent_Declare(dvv_gpu SOURCE_DIR "${DVV_GPU_SOURCE_DIR}")
    FetchContent_MakeAvailable(dvv_gpu)
else()
    # dvv::gpu alone needs no OS text library: the text component brings dvv::gpu_text.
    find_package(dvv_gpu 0.6 REQUIRED)
    if(TARGET dvv::gpu_text)
        message(FATAL_ERROR "dvv::gpu_text exists without the text component")
    endif()
    find_package(dvv_gpu 0.6 REQUIRED COMPONENTS text)
endif()
if(NOT DVV_CMAKE_DIR)
    message(FATAL_ERROR "dvv_base did not define DVV_CMAKE_DIR")
endif()
include("${DVV_CMAKE_DIR}/DvvCompilerOptions.cmake")

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE dvv::gpu dvv::gpu_text)
# dvv_add_shaders() comes with the package (installed) or with dvv_gpu's build (in-tree).
dvv_add_shaders(consumer NAMESPACE consumer_shaders SHADERS ../shaders/add_one.slang)
dvv_set_compiler_options(consumer)
