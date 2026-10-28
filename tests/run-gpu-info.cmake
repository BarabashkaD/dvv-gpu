# Runs dvv-gpu-info and checks its text and JSON output, its exit codes and that no Vulkan
# validation message was logged. Exits with 77 (CTest: skipped) when the tool finds no usable
# device, unless DVV_REQUIRE_GPU=1. Called by the dvv_gpu.gpu_info test with -DTOOL=<executable>.
cmake_minimum_required(VERSION 4.2)

if(NOT DEFINED TOOL)
    message(FATAL_ERROR "run-gpu-info.cmake needs -DTOOL")
endif()

# run(<expected exit code> <output variable> <arguments>...)
function(run expected output_variable)
    execute_process(COMMAND "${TOOL}" ${ARGN}
                    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE errors)
    if(status EQUAL 4 AND NOT expected EQUAL 4)
        if("$ENV{DVV_REQUIRE_GPU}" STREQUAL "1")
            message(FATAL_ERROR "dvv-gpu-info found no usable device (DVV_REQUIRE_GPU=1):\n"
                                "${output}${errors}")
        endif()
        message(STATUS "dvv-gpu-info found no usable device; skipped\n${output}${errors}")
        cmake_language(EXIT 77)
    endif()
    if(NOT status EQUAL expected)
        message(FATAL_ERROR "dvv-gpu-info ${ARGN}: exit ${status}, expected ${expected}\n"
                            "${output}${errors}")
    endif()
    # Counted debug messages are logged as warnings or errors, which the tool prints to stderr.
    if(errors MATCHES "Vulkan [a-z|-]+ (warning|error)")
        message(FATAL_ERROR "dvv-gpu-info ${ARGN} logged a Vulkan validation message:\n${errors}")
    endif()
    set(${output_variable} "${output}" PARENT_SCOPE)
endfunction()

run(0 json --json)
string(JSON selected GET "${json}" selectedDevice)
string(JSON first GET "${json}" devices 0 id)
string(JSON usable GET "${json}" devices 0 usable)
string(JSON api_version GET "${json}" devices 0 apiVersion)
string(JSON validation GET "${json}" validation)
string(JSON swapchain_type TYPE "${json}" devices 0 optionalFeatures swapchain)
if(NOT swapchain_type STREQUAL "BOOLEAN")
    message(FATAL_ERROR "optionalFeatures.swapchain must be a boolean, got ${swapchain_type}:\n"
                        "${json}")
endif()
if(NOT selected STREQUAL first OR NOT usable)
    message(FATAL_ERROR "the selected device must be usable and ranked first:\n${json}")
endif()
if(NOT api_version MATCHES "^1\\.([4-9]|[1-9][0-9])\\.")
    message(FATAL_ERROR "the selected device reports Vulkan ${api_version}:\n${json}")
endif()
if(DEFINED ENV{DVV_VK_VALIDATION} AND NOT "$ENV{DVV_VK_VALIDATION}" STREQUAL ""
   AND NOT validation STREQUAL "$ENV{DVV_VK_VALIDATION}")
    message(FATAL_ERROR "validation is '${validation}', DVV_VK_VALIDATION asks for "
                        "'$ENV{DVV_VK_VALIDATION}'")
endif()

run(0 text)
if(NOT text MATCHES "Device ${selected}: [^\n]+, selected\n")
    message(FATAL_ERROR "the text output does not mark device ${selected} as selected:\n${text}")
endif()
if(NOT text MATCHES "VK_KHR_swapchain (yes|no)")
    message(FATAL_ERROR "the text output does not report VK_KHR_swapchain:\n${text}")
endif()

run(1 usage --no-such-option)
run(0 help --help)
if(NOT help MATCHES "usage: dvv-gpu-info")
    message(FATAL_ERROR "--help prints no usage:\n${help}")
endif()
if(NOT help MATCHES "exit codes: 0 [^\n]+, 1 [^\n]+, 4 GPU error")
    message(FATAL_ERROR "--help does not name the exit codes:\n${help}")
endif()
