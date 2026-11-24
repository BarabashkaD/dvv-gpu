# Checks shader imports: dvv_export_shader_sources() on one library and dvv_add_shaders(...
# IMPORTS ...) on another, whose shader does `import lines.probe_lines;`. A project is built for
# an in-tree target, an alias of it, the installed package of it (dvv_import_shader_sources() on
# its exported target), exports that move between targets and exports from two directories (the
# first with another data folder); a changed imported source rebuilds the shader (depfile).
# Further cases only configure (no compiler) and check the error messages. dvv::gpu is a stub
# that gives only the include directory of dvv/gpu/embedded_shader.h.
#   cmake -DSHADERS_MODULE=<DvvShaders.cmake> -DPREFIX_PATH=<prefix,...> -DWORK_DIR=<dir>
#         -DGENERATOR=<generator> [-DPLATFORM=<platform>] [-DTOOLSET=<toolset>]
#         -DCXX_COMPILER=<compiler> -DGPU_INCLUDE_DIR=<dvv-gpu include>
#         -P run-shader-imports-check.cmake
cmake_minimum_required(VERSION 4.2)

file(TO_CMAKE_PATH "${SHADERS_MODULE}" SHADERS_MODULE)
file(TO_CMAKE_PATH "${WORK_DIR}" WORK_DIR)
file(TO_CMAKE_PATH "${GPU_INCLUDE_DIR}" GPU_INCLUDE_DIR)
file(REMOVE_RECURSE "${WORK_DIR}")
set(source_dir "${WORK_DIR}/source")
set(prefix "${WORK_DIR}/prefix")

file(WRITE "${source_dir}/CMakeLists.txt" "cmake_minimum_required(VERSION 4.2)
project(shader_imports LANGUAGES NONE)
# The configure-only cases need no compiler (each detection takes time, most with Visual Studio).
if(NOT CONFIGURE_ONLY)
    enable_language(CXX)
endif()
set(CMAKE_CXX_STANDARD 23)
# No C++20 module scanning, as in dvv's own targets: not every toolchain has a scanner.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
string(REPLACE \",\" \";\" CMAKE_PREFIX_PATH \"\${PREFIX_PATH}\")
include(\"${SHADERS_MODULE}\")
add_library(stub_gpu INTERFACE)
target_include_directories(stub_gpu INTERFACE \"${GPU_INCLUDE_DIR}\")
add_library(dvv::gpu ALIAS stub_gpu)

add_library(lines STATIC empty.cpp)
if(CASE STREQUAL \"not_slang\")
    dvv_export_shader_sources(lines SOURCES probe.txt)
elseif(CASE STREQUAL \"bad_namespace\")
    dvv_export_shader_sources(lines NAMESPACE a.b SOURCES probe_lines.slang)
elseif(CASE STREQUAL \"two_calls\")
    # The first call comes from a subdirectory with another data folder, the second from here.
    add_subdirectory(sub)
    dvv_export_shader_sources(lines SOURCES probe_lines.slang)
else()
    dvv_export_shader_sources(lines SOURCES probe_lines.slang)
endif()
add_library(test::lines ALIAS lines)
install(TARGETS lines EXPORT lines_targets)
install(EXPORT lines_targets NAMESPACE installed:: DESTINATION lib/cmake/lines)

set(imports lines)
if(CASE STREQUAL \"alias\")
    set(imports lines test::lines)
elseif(CASE STREQUAL \"imported\" OR CASE STREQUAL \"imported_clash\")
    # What the package's config file does.
    include(\"${prefix}/lib/cmake/lines/lines_targets.cmake\")
    dvv_import_shader_sources(installed::lines PREFIX \"${prefix}\")
    if(CASE STREQUAL \"imported\")
        set(imports installed::lines)
    else()
        set(imports lines installed::lines)
    endif()
elseif(CASE STREQUAL \"imported_empty\")
    file(MAKE_DIRECTORY \"\${CMAKE_BINARY_DIR}/empty\")
    add_library(empty_lines INTERFACE IMPORTED)
    set_property(TARGET empty_lines PROPERTY DVV_SHADER_IMPORT_DIR \"\${CMAKE_BINARY_DIR}/empty\")
    set(imports empty_lines)
elseif(CASE STREQUAL \"not_a_target\")
    set(imports nothing_here)
elseif(CASE STREQUAL \"no_exports\")
    set(imports stub_gpu)
elseif(CASE STREQUAL \"clash\")
    add_library(lines2 STATIC empty.cpp)
    dvv_export_shader_sources(lines2 NAMESPACE lines SOURCES probe_lines.slang)
    set(imports lines lines2)
elseif(CASE STREQUAL \"moved\")
    # Step 2 moves lines/moved.slang from lines2 to lines; lines2's old copy is still there.
    add_library(lines2 STATIC empty.cpp)
    if(STEP STREQUAL \"2\")
        dvv_export_shader_sources(lines SOURCES moved.slang)
        dvv_export_shader_sources(lines2 NAMESPACE lines SOURCES other.slang)
    else()
        dvv_export_shader_sources(lines2 NAMESPACE lines SOURCES moved.slang)
    endif()
    set(imports lines lines2)
elseif(CASE STREQUAL \"export_alias\")
    dvv_export_shader_sources(test::lines SOURCES probe_lines.slang)
elseif(CASE STREQUAL \"export_not_a_target\")
    dvv_export_shader_sources(nothing_here SOURCES probe_lines.slang)
elseif(CASE STREQUAL \"exported_twice\")
    dvv_export_shader_sources(lines SOURCES probe_lines.slang)
elseif(CASE STREQUAL \"import_not_a_target\")
    dvv_import_shader_sources(nothing_here PREFIX \"${prefix}\")
elseif(CASE STREQUAL \"import_not_imported\")
    dvv_import_shader_sources(lines PREFIX \"${prefix}\")
elseif(CASE STREQUAL \"import_no_install_dir\")
    add_library(bare_lines INTERFACE IMPORTED)
    dvv_import_shader_sources(bare_lines PREFIX \"${prefix}\")
endif()

add_library(user STATIC empty.cpp)
dvv_add_shaders(user SHADERS user_probe.slang IMPORTS \${imports})
")
file(WRITE "${source_dir}/sub/CMakeLists.txt" "set(CMAKE_INSTALL_DATADIR sub_data)
dvv_export_shader_sources(lines SOURCES extra.slang)
")
file(WRITE "${source_dir}/empty.cpp" "")
file(WRITE "${source_dir}/probe.txt" "not a shader\n")
foreach(unused moved.slang other.slang sub/extra.slang)
    file(WRITE "${source_dir}/${unused}" "// Exported, not imported.\n")
endforeach()
file(WRITE "${source_dir}/user_probe.slang" "// Writes probeValue(x) into a uint storage image.
import lines.probe_lines;

[[vk::binding(0, 0)]] [format(\"unknown\")] RWTexture2D<uint> image;

[shader(\"compute\")]
[numthreads(8, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    image[id.xy] = probeValue(id.x);
}
")

# write_lines(<file> <added constant>): the imported module.
function(write_lines file constant)
    file(WRITE "${file}" "// The value the probe shader writes.
uint probeValue(uint x)
{
    return x + ${constant};
}
")
endfunction()
write_lines("${source_dir}/probe_lines.slang" 7)

# configure(<case> <status out> <output out> [<argument>...])
function(configure case status_out output_out)
    set(command "${CMAKE_COMMAND}" -S "${source_dir}" -B "${WORK_DIR}/build-${case}"
        -G "${GENERATOR}" "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}" "-DCMAKE_BUILD_TYPE=Release"
        "-DPREFIX_PATH=${PREFIX_PATH}" "-DCASE=${case}" ${ARGN})
    if(PLATFORM)
        list(APPEND command -A "${PLATFORM}")
    endif()
    if(TOOLSET)
        list(APPEND command -T "${TOOLSET}")
    endif()
    execute_process(COMMAND ${command}
        RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE output)
    set(${status_out} "${status}" PARENT_SCOPE)
    set(${output_out} "${output}" PARENT_SCOPE)
endfunction()

# build(<case>): builds the configured case, which must give the shader's SPIR-V, or fails.
function(build case)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" --build "${WORK_DIR}/build-${case}" --config Release
        RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE output)
    set(spirv "${WORK_DIR}/build-${case}/dvv_shaders/user/user_probe.spv")
    if(NOT status EQUAL 0 OR NOT EXISTS "${spirv}")
        message(FATAL_ERROR "${case}: the build fails or gives no ${spirv}:\n${output}")
    endif()
endfunction()

# configure_and_build(<case> [<argument>...]): configuring and building succeed.
function(configure_and_build case)
    configure(${case} status output ${ARGN})
    if(NOT status EQUAL 0)
        message(FATAL_ERROR "${case}: configuring fails:\n${output}")
    endif()
    build(${case})
endfunction()

# expect_build(<case> [<argument>...]): configuring and building succeed.
function(expect_build case)
    configure_and_build(${case} ${ARGN})
    message(STATUS "${case}: ok")
endfunction()

# expect_error(<case> <regular expression> [<rest of it>]): configuring fails with a matching
# message.
function(expect_error case pattern)
    if(ARGC GREATER 2)
        string(APPEND pattern "${ARGV2}")
    endif()
    configure(${case} status output -DCONFIGURE_ONLY=ON)
    string(REGEX REPLACE "[ \t\r\n]+" " " folded "${output}")
    if(status EQUAL 0 OR NOT folded MATCHES "${pattern}")
        message(FATAL_ERROR "${case}: the output does not match '${pattern}':\n${output}")
    endif()
    message(STATUS "${case}: ok")
endfunction()

# A data folder other than the default, which the installed package must follow.
expect_build(builds -DCMAKE_INSTALL_DATADIR=data)

# A copy the target no longer exports is removed when configuring again.
set(stale "${WORK_DIR}/build-builds/dvv_shader_exports/lines/lines/stale.slang")
file(WRITE "${stale}" "")
configure(builds status output)
if(NOT status EQUAL 0 OR EXISTS "${stale}")
    message(FATAL_ERROR "pruned: configuring again fails or keeps ${stale}:\n${output}")
endif()
message(STATUS "pruned: ok")

# The exported sources and the target are installed; the "imported" cases use the package.
execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${WORK_DIR}/build-builds" --config Release
            --prefix "${prefix}"
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE output)
set(installed "${prefix}/data/lines/shaders/lines/probe_lines.slang")
if(NOT status EQUAL 0 OR NOT EXISTS "${installed}")
    message(FATAL_ERROR "installed: the install fails or gives no ${installed}:\n${output}")
endif()
message(STATUS "installed: ok")

# A changed imported source is copied again when the build re-runs configure, and slangc's
# depfile then rebuilds the shader.
set(depfile_spirv "${WORK_DIR}/build-builds/dvv_shaders/user/user_probe.spv")
file(SHA256 "${depfile_spirv}" before)
write_lines("${source_dir}/probe_lines.slang" 8)
build(builds)
file(SHA256 "${depfile_spirv}" after)
write_lines("${source_dir}/probe_lines.slang" 7)
if(before STREQUAL after)
    message(FATAL_ERROR "depfile: the shader was not rebuilt after its import changed")
endif()
message(STATUS "depfile: rebuilt")

expect_build(alias)
expect_build(imported)
configure_and_build(moved -DSTEP=1)
expect_build(moved -DSTEP=2)
expect_build(two_calls)
set(exports "${WORK_DIR}/build-two_calls/dvv_shader_exports/lines/lines")
foreach(file IN ITEMS extra.slang probe_lines.slang)
    if(NOT EXISTS "${exports}/${file}")
        message(FATAL_ERROR "two_calls: an export is not in the folder: ${exports}/${file}")
    endif()
endforeach()
# The prune runs after both calls, so configuring again leaves the second call's copy alone.
file(TIMESTAMP "${exports}/probe_lines.slang" before "%s%f")
configure(two_calls status output)
file(TIMESTAMP "${exports}/probe_lines.slang" after "%s%f")
if(NOT status EQUAL 0 OR NOT before STREQUAL after)
    message(FATAL_ERROR "two_calls: configuring again fails or rewrites "
                        "${exports}/probe_lines.slang:\n${output}")
endif()
message(STATUS "two_calls kept: ok")
# Both calls install to the first call's data folder, which the exported target names.
execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${WORK_DIR}/build-two_calls" --config Release
            --prefix "${WORK_DIR}/prefix-two_calls"
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE output)
set(installed "${WORK_DIR}/prefix-two_calls/sub_data/lines/shaders/lines")
foreach(file IN ITEMS extra.slang probe_lines.slang)
    if(NOT status EQUAL 0 OR NOT EXISTS "${installed}/${file}")
        message(FATAL_ERROR "two_calls: the install fails or gives no ${installed}/${file}:\n"
                            "${output}")
    endif()
endforeach()
message(STATUS "two_calls installed: ok")
expect_error(not_a_target "dvv_add_shaders: 'nothing_here' is not a target")
expect_error(no_exports
    "dvv_add_shaders: 'stub_gpu' exports no shader sources \\(DVV_SHADER_IMPORT_DIR is not set\\)")
expect_error(imported_empty
    "dvv_add_shaders: 'empty_lines' exports no shader sources "
    "\\('[^']*/empty' holds no .slang file\\)")
expect_error(clash
    "dvv_add_shaders: 'lines/probe_lines.slang' is exported by both 'lines' and 'lines2'")
expect_error(imported_clash
    "dvv_add_shaders: 'lines/probe_lines.slang' is exported by both 'lines' and 'installed::lines'")
expect_error(not_slang "dvv_export_shader_sources: 'probe.txt' is not a .slang file")
expect_error(export_alias
    "dvv_export_shader_sources: 'test::lines' is an alias; export from 'lines'")
expect_error(export_not_a_target "dvv_export_shader_sources: 'nothing_here' is not a target")
expect_error(bad_namespace "dvv_export_shader_sources: 'a.b' is not a Slang module namespace")
expect_error(exported_twice
    "dvv_export_shader_sources: 'lines/probe_lines.slang' is exported twice by 'lines'")
expect_error(import_not_a_target "dvv_import_shader_sources: 'nothing_here' is not a target")
expect_error(import_not_imported
    "dvv_import_shader_sources: 'lines' is not an imported target")
expect_error(import_no_install_dir
    "dvv_import_shader_sources: 'bare_lines' exports no shader sources "
    "\\(DVV_SHADER_INSTALL_DIR is not set\\)")
