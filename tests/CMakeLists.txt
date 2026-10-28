find_package(doctest 2.5 REQUIRED CONFIG)

# The test shaders, compiled by dvv_add_shaders() and shared by the unit and the GPU tests.
add_library(dvv_gpu_test_shaders STATIC)
dvv_add_shaders(dvv_gpu_test_shaders NAMESPACE test_shaders
                SHADERS shaders/add_one.slang shaders/write_image.slang
                        shaders/write_two_images.slang)
dvv_set_compiler_options(dvv_gpu_test_shaders)

add_executable(dvv_gpu_tests
    main.cpp
    test_device_id.cpp
    test_free_list.cpp
    test_gpu_info.cpp
    test_loader.cpp
    test_pipeline_cache.cpp
    test_present_blit.cpp
    test_presenter_core.cpp
    test_presenter_logic.cpp
    test_shaders.cpp
    test_validation.cpp
    test_vk_result.cpp
    test_window_system.cpp)
target_link_libraries(dvv_gpu_tests PRIVATE dvv::gpu dvv_gpu_test_shaders doctest::doctest)
# The unit tests also check private parts (src/), e.g. the Vulkan result conversion.
target_include_directories(dvv_gpu_tests PRIVATE ${dvv_gpu_SOURCE_DIR}/src)
# doctest otherwise only forward-declares std::ostream, which breaks printing std types.
target_compile_definitions(dvv_gpu_tests PRIVATE DOCTEST_CONFIG_USE_STD_HEADERS)
dvv_set_compiler_options(dvv_gpu_tests)
add_test(NAME dvv_gpu.unit COMMAND dvv_gpu_tests)
# The timeout turns a hang into a failure long before the CI job limit.
set_tests_properties(dvv_gpu.unit PROPERTIES
    TIMEOUT 300
    ENVIRONMENT DVV_LEAK_CHECK=fail
    ENVIRONMENT_MODIFICATION "${DVV_TEST_ENVIRONMENT}")
# Cycle labels: `sanity` marks the fast inner-loop subset (ctest --preset vs2026-sanity); `package`
# marks install and consumer tests, which only the full cycle runs (vs2026-small excludes them).
set_property(TEST dvv_gpu.unit APPEND PROPERTY LABELS sanity)

# dvv_add_shaders() must refuse a missing or wrong slangc at configure time with a clear message.
string(REPLACE ";" "," slangc_prefix_path "${CMAKE_PREFIX_PATH}")
add_test(NAME dvv_gpu.slangc_check
         COMMAND ${CMAKE_COMMAND}
                 -DSHADERS_MODULE=${dvv_gpu_SOURCE_DIR}/cmake/DvvShaders.cmake
                 -DPREFIX_PATH=${slangc_prefix_path}
                 -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/slangc-check
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/shaders/run-slangc-check.cmake)
# ... and name shaders in camelCase, refusing duplicates and C++ keywords.
add_test(NAME dvv_gpu.shader_names
         COMMAND ${CMAKE_COMMAND}
                 -DSHADERS_MODULE=${dvv_gpu_SOURCE_DIR}/cmake/DvvShaders.cmake
                 -DPREFIX_PATH=${slangc_prefix_path}
                 -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/shader-names-check
                 "-DGENERATOR=${CMAKE_GENERATOR}"
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/shaders/run-shader-names-check.cmake)
set_tests_properties(dvv_gpu.slangc_check dvv_gpu.shader_names PROPERTIES TIMEOUT 120)

# GPU tests (spec §12): a separate executable, CTest label gpu. It exits with 77 (reported as
# skipped) when no usable Vulkan 1.4 device exists, unless DVV_REQUIRE_GPU=1. Every test checks at
# its end that no validation message was counted. The loader is pointed at the validation layer of
# the third-party prefix (VK_ADD_LAYER_PATH); a system's layer may be too old for Vulkan 1.4.
find_file(DVV_VK_VALIDATION_LAYER_MANIFEST VkLayer_khronos_validation.json
    PATHS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES bin share/vulkan/explicit_layer.d NO_DEFAULT_PATH)
set(gpu_test_environment DVV_LEAK_CHECK=fail)
if(DVV_VK_VALIDATION_LAYER_MANIFEST)
    cmake_path(GET DVV_VK_VALIDATION_LAYER_MANIFEST PARENT_PATH layer_dir)
    list(APPEND gpu_test_environment "VK_ADD_LAYER_PATH=${layer_dir}")
else()
    message(STATUS "No Vulkan validation layer in the prefix; GPU tests run without it")
endif()
# The software driver is not instrumented; see tsan.supp.
if(DVV_SANITIZE STREQUAL "thread")
    list(APPEND gpu_test_environment
         "TSAN_OPTIONS=suppressions=${CMAKE_CURRENT_SOURCE_DIR}/tsan.supp")
endif()

add_executable(dvv_gpu_device_tests
    gpu_fixture.cpp
    gpu_main.cpp
    gpu_test_compute.cpp
    gpu_test_context.cpp
    gpu_test_device_lost.cpp
    gpu_test_images.cpp
    gpu_test_pipeline.cpp
    gpu_test_present_blit.cpp
    gpu_test_registry.cpp
    gpu_test_resource_pool.cpp
    gpu_test_resources.cpp
    gpu_test_submission.cpp)
# White-box tests reach VMA through the private headers.
target_link_libraries(dvv_gpu_device_tests PRIVATE dvv::gpu dvv_gpu_test_shaders doctest::doctest
                                                   GPUOpen::VulkanMemoryAllocator)
target_include_directories(dvv_gpu_device_tests PRIVATE ${dvv_gpu_SOURCE_DIR}/src)
target_compile_definitions(dvv_gpu_device_tests PRIVATE DOCTEST_CONFIG_USE_STD_HEADERS)
dvv_set_compiler_options(dvv_gpu_device_tests)
add_test(NAME dvv_gpu.gpu COMMAND dvv_gpu_device_tests)
add_test(NAME dvv_gpu.gpu_info
         COMMAND ${CMAKE_COMMAND} -DTOOL=$<TARGET_FILE:dvv_gpu_info>
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/run-gpu-info.cmake)
set_tests_properties(dvv_gpu.gpu dvv_gpu.gpu_info PROPERTIES
    LABELS gpu
    SKIP_RETURN_CODE 77
    TIMEOUT 300
    ENVIRONMENT "${gpu_test_environment}"
    ENVIRONMENT_MODIFICATION "${DVV_TEST_ENVIRONMENT}")
set_property(TEST dvv_gpu.gpu_info APPEND PROPERTY LABELS sanity)
# Presentation tests (design M1b §8.2): raw native windows from dvv::platform_testing, no wx. They
# exit with 77 without a usable device that can present or without a display, unless
# DVV_REQUIRE_GPU=1. Linux CI runs them under xvfb-run.
add_executable(dvv_gpu_present_tests
    present_fixture.cpp
    present_main.cpp
    present_test_device_lost.cpp
    present_test_events.cpp
    present_test_presenter.cpp)
target_link_libraries(dvv_gpu_present_tests PRIVATE dvv::gpu dvv::platform_testing
                                                    doctest::doctest GPUOpen::VulkanMemoryAllocator)
# The device-loss cases stub the context's dispatch table through the private headers.
target_include_directories(dvv_gpu_present_tests PRIVATE ${dvv_gpu_SOURCE_DIR}/src)
target_compile_definitions(dvv_gpu_present_tests PRIVATE DOCTEST_CONFIG_USE_STD_HEADERS)
dvv_set_compiler_options(dvv_gpu_present_tests)
add_test(NAME dvv_gpu.present COMMAND dvv_gpu_present_tests)
set_tests_properties(dvv_gpu.present PROPERTIES
    LABELS "gpu;display"
    SKIP_RETURN_CODE 77
    TIMEOUT 300
    ENVIRONMENT "${gpu_test_environment}"
    ENVIRONMENT_MODIFICATION "${DVV_TEST_ENVIRONMENT}")
set_property(TEST dvv_gpu.present APPEND PROPERTY LABELS sanity)
# Package test (spec §3.1): install, then build a consumer with find_package(dvv_gpu) and another
# with dvv_gpu in-tree. The whole build tree is installed (CMAKE_BINARY_DIR), so the prefix also
# holds dvv_base and dvv_platform, which dvv_dependencies() always adds as subprojects. Skipped for
# sanitizer builds, whose consumers would need the same runtime flags.
if(NOT DVV_SANITIZE)
    string(REPLACE ";" "," deps_prefix_path "${CMAKE_PREFIX_PATH}")
    add_test(NAME dvv_gpu.package
             COMMAND ${CMAKE_COMMAND}
                     -DBUILD_DIR=${CMAKE_BINARY_DIR}
                     -DCONFIG=$<CONFIG>
                     -DPREFIX=${CMAKE_CURRENT_BINARY_DIR}/package-prefix
                     -DSOURCE_DIR=${dvv_gpu_SOURCE_DIR}
                     -DCONSUMER_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/package
                     -DCONSUMER_BUILD=${CMAKE_CURRENT_BINARY_DIR}/package-consumer
                     -DCONSUMER_TREE_BUILD=${CMAKE_CURRENT_BINARY_DIR}/pkg-tree
                     -DGENERATOR=${CMAKE_GENERATOR}
                     -DPLATFORM=${CMAKE_GENERATOR_PLATFORM}
                     -DCXX_COMPILER=${CMAKE_CXX_COMPILER}
                     -DDEPS_PREFIX_PATH=${deps_prefix_path}
                     -DDVV_BUILD_CMAKE_DIR=${DVV_BUILD_CMAKE_DIR}
                     -P ${CMAKE_CURRENT_SOURCE_DIR}/package/run-package-test.cmake)
    # Package and compile-fail tests of every dvv repository drive the same build tree in a
    # dependent build with DVV_TEST_DEPENDENCIES, so they share one lock name.
    set_tests_properties(dvv_gpu.package PROPERTIES
        RESOURCE_LOCK dvv_build_tree
        LABELS package)
endif()

# Compile-fail tests (spec §9.4), run by dvv-base's shared runner. Case 0 is the control and must
# build; cases 1-8 must fail to compile with the expected diagnostic: a forged registry key, and a
# registry, a context, a buffer, an image, a resource pool, a compute pipeline or a presenter
# constructed without a key.
set(compile_fail_source ${CMAKE_CURRENT_SOURCE_DIR}/compile_fail/direct_construction.cpp)
add_executable(dvv_gpu_compile_ok ${compile_fail_source})
target_compile_definitions(dvv_gpu_compile_ok PRIVATE DVV_COMPILE_FAIL_CASE=0)
target_link_libraries(dvv_gpu_compile_ok PRIVATE dvv::gpu)
dvv_set_compiler_options(dvv_gpu_compile_ok)
if(MSVC)
    set(compile_fail_patterns "C2248" "C2665" "C2661" "C2665: 'dvv::gpu::GpuBuffer::GpuBuffer'"
        "C2665: 'dvv::gpu::GpuImage::GpuImage'" "C2661: 'dvv::gpu::ResourcePool::ResourcePool'"
        "C2661: 'dvv::gpu::ComputePipeline::ComputePipeline'"
        "C2661: 'dvv::gpu::Presenter::Presenter'")
else()
    set(compile_fail_patterns "private" "no matching constructor" "no matching constructor"
        "no matching constructor for initialization of 'dvv::gpu::GpuBuffer'"
        "no matching constructor for initialization of 'dvv::gpu::GpuImage'"
        "no matching constructor for initialization of 'dvv::gpu::ResourcePool'"
        "no matching constructor for initialization of 'dvv::gpu::ComputePipeline'"
        "no matching constructor for initialization of 'dvv::gpu::Presenter'")
endif()
foreach(case IN ITEMS 1 2 3 4 5 6 7 8)
    set(target dvv_gpu_compile_fail_${case})
    add_executable(${target} EXCLUDE_FROM_ALL ${compile_fail_source})
    target_compile_definitions(${target} PRIVATE DVV_COMPILE_FAIL_CASE=${case})
    target_link_libraries(${target} PRIVATE dvv::gpu)
    dvv_set_compiler_options(${target})
    math(EXPR pattern_index "${case} - 1")
    list(GET compile_fail_patterns ${pattern_index} pattern)
    # cmake --build needs the top-level tree; in a dependent build (another dvv library or the app)
    # dvv-gpu is a subproject.
    add_test(NAME dvv_gpu.compile_fail.${case}
             COMMAND ${CMAKE_COMMAND}
                     -DBUILD_DIR=${CMAKE_BINARY_DIR}
                     -DCONFIG=$<CONFIG>
                     -DTARGET=${target}
                     "-DPATTERN=${pattern}"
                     -P ${DVV_CMAKE_DIR}/run-compile-fail.cmake)
    set_tests_properties(dvv_gpu.compile_fail.${case} PROPERTIES RESOURCE_LOCK dvv_build_tree)
endforeach()
