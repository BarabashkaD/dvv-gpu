# Checks the C++ names dvv_add_shaders gives shaders, and its configure errors for two shaders
# with the same name and for a name that is a C++ keyword. Each case configures a small project
# (no languages, nothing is built) that calls dvv_add_shaders on a static library.
#   cmake -DSHADERS_MODULE=<DvvShaders.cmake> -DPREFIX_PATH=<prefix,...> -DWORK_DIR=<dir>
#         -DGENERATOR=<generator> -P run-shader-names-check.cmake
cmake_minimum_required(VERSION 4.2)

file(TO_CMAKE_PATH "${SHADERS_MODULE}" SHADERS_MODULE)
file(TO_CMAKE_PATH "${WORK_DIR}" WORK_DIR)
file(REMOVE_RECURSE "${WORK_DIR}")
file(WRITE "${WORK_DIR}/source/CMakeLists.txt" "cmake_minimum_required(VERSION 4.2)
project(shader_names LANGUAGES NONE)
string(REPLACE \",\" \";\" CMAKE_PREFIX_PATH \"\${PREFIX_PATH}\")
include(\"${SHADERS_MODULE}\")
add_library(names STATIC)
string(REPLACE \",\" \";\" shaders \"\${SHADERS}\")
dvv_add_shaders(names SHADERS \${shaders})
get_property(functions TARGET names PROPERTY DVV_SHADER_FUNCTIONS)
# Stops before generating: the project has no language to build the generated sources with.
message(FATAL_ERROR \"functions: \${functions}\")
")

# expect(<name> <shaders, comma separated> <regular expression>); configuring always fails.
function(expect name shaders pattern)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -S "${WORK_DIR}/source" -B "${WORK_DIR}/build" -G "${GENERATOR}"
                "-DPREFIX_PATH=${PREFIX_PATH}" "-DSHADERS=${shaders}"
        RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE output)
    file(REMOVE_RECURSE "${WORK_DIR}/build")
    string(REGEX REPLACE "[ \t\r\n]+" " " folded "${output}")
    if(status EQUAL 0 OR NOT folded MATCHES "${pattern}")
        message(FATAL_ERROR "${name}: the output does not match '${pattern}':\n${output}")
    endif()
    message(STATUS "${name}: ok")
endfunction()

expect(camel_case "add_one.slang,blurH.slang,YCbCr.slang,two-step.v2.slang"
       "functions: addOne;blurH;yCbCr;twoStepV2")
expect(same_name "a/add_one.slang,b/add-one.slang"
       "'[^']*/b/add-one.slang' gives the same name as '[^']*/a/add_one.slang' \\(addOne\\)")
expect(keyword "new.slang" "'new.slang' does not make a C\\+\\+ function name \\('new'\\)")
