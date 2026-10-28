cmake_minimum_required(VERSION 4.2)
project(dvv_gpu VERSION 0.1.0 DESCRIPTION "dvv GPU core" LANGUAGES CXX)
# Version string; a pre-release build adds its release-stage label (user decision U-M4ORDER).
set(dvv_gpu_VERSION_STRING "${PROJECT_VERSION}")

# dvv-build provides the shared CMake modules and the dependency check (dvv-deps.txt); the dvv
# presets set DVV_BUILD_CMAKE_DIR to its cmake folder.
if(NOT DVV_BUILD_CMAKE_DIR OR NOT IS_ABSOLUTE "${DVV_BUILD_CMAKE_DIR}"
   OR NOT EXISTS "${DVV_BUILD_CMAKE_DIR}/DvvDeps.cmake")
    message(FATAL_ERROR "dvv_gpu: DVV_BUILD_CMAKE_DIR is not set, not absolute or "
                        "has no DvvDeps.cmake (it must be dvv-build's cmake folder); configure "
                        "with a dvv-build preset (e.g. `cmake --preset vs2026`, see README.md) "
                        "or pass -DDVV_BUILD_CMAKE_DIR=<absolute path of dvv-build>/cmake")
endif()
include("${DVV_BUILD_CMAKE_DIR}/DvvDeps.cmake")
# Adds dvv-base and dvv-platform (native windows, Vulkan surfaces, run-time libraries; dvv-deps.txt)
# and checks their versions; a stand-alone build tests only this repository.
dvv_dependencies()

option(DVV_BUILD_TESTS "Build the dvv tests" ${PROJECT_IS_TOP_LEVEL})

# dvv-base sets DVV_CMAKE_DIR to the shared modules (in-tree: dvv-build's cmake folder).
if(NOT DVV_CMAKE_DIR)
    message(FATAL_ERROR "dvv_base did not define DVV_CMAKE_DIR")
endif()
include("${DVV_CMAKE_DIR}/DvvCompilerOptions.cmake")
# dvv_add_shaders() for this repository and, in-tree (a dependent library or the app), for its
# users.
include(cmake/DvvShaders.cmake)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Vulkan comes only from the pinned packages in the third-party prefix, never from CMake's FindVulkan
# module, which would pick up an installed Vulkan SDK through VULKAN_SDK.
set(CMAKE_DISABLE_FIND_PACKAGE_Vulkan ON)
find_package(VulkanHeaders 1.4.365 EXACT CONFIG REQUIRED)
find_package(VulkanMemoryAllocator 3.4.0 CONFIG REQUIRED)
# vk-bootstrap is compiled into dvv_gpu from the source the superbuild installs: its prebuilt
# Release library cannot be linked into MSVC Debug builds (different runtime library).
find_file(DVV_VK_BOOTSTRAP_SOURCE VkBootstrap.cpp
    PATHS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES src/vk-bootstrap NO_DEFAULT_PATH REQUIRED)
find_path(DVV_VK_BOOTSTRAP_INCLUDE_DIR VkBootstrap.h
    PATHS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES include NO_DEFAULT_PATH REQUIRED)

add_library(dvv_gpu STATIC
    src/command_recorder.cpp
    src/compute_pipeline.cpp
    src/device_core.cpp
    src/devices.cpp
    src/execution_context.cpp
    src/fault_injection.cpp
    src/gpu_info.cpp
    src/gpu_registry.cpp
    src/instance.cpp
    src/loader.cpp
    src/pipeline_cache.cpp
    src/presenter.cpp
    src/presenter_core.cpp
    src/presenter_logic.cpp
    src/resource_pool.cpp
    src/resources.cpp
    src/submitter.cpp
    src/vk_result.cpp
    src/vma.cpp
    "${DVV_VK_BOOTSTRAP_SOURCE}")
add_library(dvv::gpu ALIAS dvv_gpu)
# VkBootstrap.cpp is third-party code: its warnings stay visible but never fail our build; our own
# warning rules apply to our code only. Source options follow the target's /WX or -Werror.
set_source_files_properties("${DVV_VK_BOOTSTRAP_SOURCE}" PROPERTIES
    COMPILE_OPTIONS "$<IF:$<CXX_COMPILER_ID:MSVC>,/WX-,-Wno-error>")
set_target_properties(dvv_gpu PROPERTIES EXPORT_NAME gpu)
target_include_directories(dvv_gpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(dvv_gpu SYSTEM PRIVATE "${DVV_VK_BOOTSTRAP_INCLUDE_DIR}")
# Vulkan's C headers are part of the interface, always without prototypes: nothing links the
# Vulkan loader, it is loaded at run time. VMA is header-only and compiled into dvv_gpu (src/vma.cpp),
# so installed consumers do not need it.
target_compile_definitions(dvv_gpu PUBLIC VK_NO_PROTOTYPES)
# The Vulkan loader is opened through dvv-platform's DynamicLibrary, which links the dl library.
target_link_libraries(dvv_gpu
    PUBLIC dvv::base dvv::platform Vulkan::Headers
    PRIVATE $<BUILD_INTERFACE:GPUOpen::VulkanMemoryAllocator>)
dvv_set_compiler_options(dvv_gpu)
# The presenter's stop waits while dispatching sent messages on Windows (design M1b §5.5, R15).
if(WIN32)
    target_sources(dvv_gpu PRIVATE src/thread_wait_win32.cpp)
    target_link_libraries(dvv_gpu PRIVATE user32)
endif()

# dvv-gpu-info: prints every Vulkan device the registry finds (text or --json).
add_executable(dvv_gpu_info tools/dvv_gpu_info.cpp)
set_target_properties(dvv_gpu_info PROPERTIES OUTPUT_NAME dvv-gpu-info)
target_link_libraries(dvv_gpu_info PRIVATE dvv::gpu)
dvv_set_compiler_options(dvv_gpu_info)

install(TARGETS dvv_gpu EXPORT dvv_gpuTargets)
install(TARGETS dvv_gpu_info)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
set(config_dir ${CMAKE_INSTALL_LIBDIR}/cmake/dvv_gpu)
install(EXPORT dvv_gpuTargets NAMESPACE dvv:: DESTINATION ${config_dir})
configure_package_config_file(cmake/dvv_gpuConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/dvv_gpuConfig.cmake INSTALL_DESTINATION ${config_dir})
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/dvv_gpuConfigVersion.cmake
    COMPATIBILITY SameMajorVersion)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/dvv_gpuConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/dvv_gpuConfigVersion.cmake
    cmake/DvvShaders.cmake
    cmake/DvvEmbedShader.cmake
    DESTINATION ${config_dir})

if(DVV_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
