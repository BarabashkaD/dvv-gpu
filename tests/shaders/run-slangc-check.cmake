# Checks that dvv_add_shaders refuses, at configure time and with a clear message, a slangc that
# is missing from the prefix (even when one is on PATH), one that cannot run and one of another
# version, and accepts the pinned one. Each case runs a child `cmake -P` that includes
# DvvShaders.cmake and calls dvv_find_slangc().
#   cmake -DSHADERS_MODULE=<DvvShaders.cmake> -DPREFIX_PATH=<prefix,...> -DWORK_DIR=<dir>
#         -P run-slangc-check.cmake
cmake_minimum_required(VERSION 4.2)

# The paths go into a generated script, so they must not hold backslashes.
file(TO_CMAKE_PATH "${SHADERS_MODULE}" SHADERS_MODULE)
file(TO_CMAKE_PATH "${WORK_DIR}" WORK_DIR)

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}/empty-prefix")
set(probe "${WORK_DIR}/probe-slangc.cmake")
file(WRITE "${probe}" "include(\"${SHADERS_MODULE}\")
string(REPLACE \",\" \";\" CMAKE_PREFIX_PATH \"\${PREFIX_PATH}\")
if(DEFINED REQUIRED_VERSION)
    set_property(GLOBAL PROPERTY DVV_SLANGC_VERSION \"\${REQUIRED_VERSION}\")
endif()
dvv_find_slangc()
message(STATUS \"slangc: \${DVV_SLANGC}\")
")

# A prefix whose slangc is not a program.
set(broken "${WORK_DIR}/broken-prefix/slang/bin/slangc")
if(CMAKE_HOST_WIN32)
    string(APPEND broken ".exe")
endif()
file(WRITE "${broken}" "not a program\n")
file(CHMOD "${broken}" FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)

# expect(<name> <0 = must succeed, 1 = must fail> <regular expression> [-D...])
function(expect name must_fail pattern)
    execute_process(COMMAND "${CMAKE_COMMAND}" ${ARGN} -P "${probe}"
        RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE output)
    # CMake wraps long messages; compare with the white space folded.
    string(REGEX REPLACE "[ \t\r\n]+" " " folded "${output}")
    if(must_fail AND status EQUAL 0)
        message(FATAL_ERROR "${name}: configuring succeeded, a failure was expected:\n${output}")
    elseif(NOT must_fail AND NOT status EQUAL 0)
        message(FATAL_ERROR "${name}: configuring failed:\n${output}")
    endif()
    if(NOT folded MATCHES "${pattern}")
        message(FATAL_ERROR "${name}: the output does not match '${pattern}':\n${output}")
    endif()
    message(STATUS "${name}: ok")
endfunction()

expect("pinned slangc" 0 "slangc: .*slangc"
       "-DPREFIX_PATH=${PREFIX_PATH}")
string(CONCAT mismatch "reports version '[0-9.]+', but dvv-gpu pins slangc 1\\.2\\.3\\. "
       "Rebuild the third-party prefix, or update DVV_SLANGC_VERSION")
expect("wrong slangc version" 1 "${mismatch}"
       "-DPREFIX_PATH=${PREFIX_PATH}" -DREQUIRED_VERSION=1.2.3)
expect("slangc that cannot run" 1 "slangc[.a-z]* -version failed \\("
       "-DPREFIX_PATH=${WORK_DIR}/broken-prefix")

# The real slangc on PATH must not be found: only the prefix counts.
string(REPLACE "," ";" prefixes "${PREFIX_PATH}")
set(slang_dirs "")
foreach(prefix IN LISTS prefixes)
    list(APPEND slang_dirs "${prefix}/slang/bin")
endforeach()
cmake_path(CONVERT "${slang_dirs}" TO_NATIVE_PATH_LIST native_dirs)
if(CMAKE_HOST_WIN32)
    set(ENV{PATH} "${native_dirs};$ENV{PATH}")
else()
    set(ENV{PATH} "${native_dirs}:$ENV{PATH}")
endif()
expect("no slangc in the prefix, one on PATH" 1
       "slangc [0-9.]+ was not found in the third-party prefix"
       "-DPREFIX_PATH=${WORK_DIR}/empty-prefix")
