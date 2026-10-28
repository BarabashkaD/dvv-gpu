# dvv_add_shaders(<target> [NAMESPACE <namespace>] SHADERS <file.slang>...)
#
# Compiles Slang compute shaders at build time with the pinned slangc of the third-party prefix
# (spec §5.4, §13.2) and embeds them in <target>. For each <name>.slang the generated header
# "<namespace as a path>/<name>.h" (e.g. "my_shaders/add_one.h") declares
#
#     const dvv::gpu::EmbeddedShader& <namespace>::<name in camelCase>() noexcept;
#
# (add_one.slang -> addOne(), blurH.slang -> blurH()) with the SPIR-V words and slangc's
# reflection JSON. The default namespace is "<target>_shaders", so two libraries never share a
# header path or a symbol. Each shader has one compute entry point named main. Two shaders of a
# target that give the same function name, or a name that is a C++ keyword, fail configuring.
# The target gets the generated sources, their include directory (build tree only) and dvv::gpu,
# PUBLIC, so a library can pass the headers on; the target must use the keyword form of
# target_link_libraries().
#
# slangc is taken only from <prefix>/slang/bin of CMAKE_PREFIX_PATH, never from PATH (an installed
# Vulkan SDK may bring another version), and its version must be the pinned one (the global
# property DVV_SLANGC_VERSION); otherwise configuring fails with a message that says what is
# wrong. slangc also writes a depfile, so a shader is rebuilt when a file it imports changes.
include_guard(GLOBAL)

# The superbuild's Slang pin (dvv-build deps/versions.cmake); keep both in step. A global
# property, so it is visible wherever dvv_add_shaders() is called (in-tree this file is included
# in dvv_gpu's directory).
set_property(GLOBAL PROPERTY DVV_SLANGC_VERSION "2026.20")

# Sets DVV_SLANGC (cache) to the pinned slangc, or fails.
function(dvv_find_slangc)
    get_property(required GLOBAL PROPERTY DVV_SLANGC_VERSION)
    find_program(DVV_SLANGC NAMES slangc
        PATHS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES slang/bin NO_DEFAULT_PATH)
    if(NOT DVV_SLANGC)
        message(FATAL_ERROR
            "dvv_add_shaders: slangc ${required} was not found in the third-party "
            "prefix (looked in <prefix>/slang/bin for CMAKE_PREFIX_PATH '${CMAKE_PREFIX_PATH}'). "
            "Build the dvv-build third-party superbuild and pass its prefix in "
            "CMAKE_PREFIX_PATH.")
    endif()
    execute_process(COMMAND "${DVV_SLANGC}" -version
        RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE output)
    string(STRIP "${output}" version)
    if(NOT status EQUAL 0)
        message(FATAL_ERROR
            "dvv_add_shaders: ${DVV_SLANGC} -version failed (${status}): ${version}")
    endif()
    if(NOT version STREQUAL required)
        message(FATAL_ERROR
            "dvv_add_shaders: ${DVV_SLANGC} reports version '${version}', but dvv-gpu pins "
            "slangc ${required}. Rebuild the third-party prefix, or update DVV_SLANGC_VERSION in "
            "dvv-gpu's cmake/DvvShaders.cmake together with dvv-build's "
            "deps/versions.cmake.")
    endif()
endfunction()

# Sets <out> to the C++ function name of a shader file stem: the parts between '-', '_' and '.'
# in camelCase (the first letter lower case, the first letter of each further part upper case).
function(_dvv_shader_function stem out)
    string(REGEX REPLACE "[-_.]+" ";" parts "${stem}")
    set(function "")
    foreach(part IN LISTS parts)
        string(SUBSTRING "${part}" 0 1 first)
        string(SUBSTRING "${part}" 1 -1 rest)
        if(function STREQUAL "")
            string(TOLOWER "${first}" first)
        else()
            string(TOUPPER "${first}" first)
        endif()
        string(APPEND function "${first}${rest}")
    endforeach()
    set(${out} "${function}" PARENT_SCOPE)
endfunction()

function(dvv_add_shaders target)
    cmake_parse_arguments(PARSE_ARGV 1 arg "" "NAMESPACE" "SHADERS")
    if(arg_UNPARSED_ARGUMENTS OR NOT arg_SHADERS)
        message(FATAL_ERROR
            "usage: dvv_add_shaders(<target> [NAMESPACE <namespace>] SHADERS <file.slang>...)")
    endif()
    if(NOT TARGET ${target})
        message(FATAL_ERROR "dvv_add_shaders: ${target} is not a target")
    endif()
    if(NOT arg_NAMESPACE)
        string(MAKE_C_IDENTIFIER "${target}_shaders" arg_NAMESPACE)
    endif()
    if(NOT arg_NAMESPACE MATCHES "^[A-Za-z_][A-Za-z0-9_]*(::[A-Za-z_][A-Za-z0-9_]*)*$")
        message(FATAL_ERROR "dvv_add_shaders: '${arg_NAMESPACE}' is not a C++ namespace")
    endif()
    dvv_find_slangc()

    # C++ keywords (and alternative tokens) that a shader name must not become.
    set(keywords
        alignas alignof and and_eq asm auto bitand bitor bool break case catch char char8_t char16_t
        char32_t class compl concept const consteval constexpr constinit const_cast continue co_await
        co_return co_yield decltype default delete do double dynamic_cast else enum explicit export
        extern false float for friend goto if inline int long mutable namespace new noexcept not
        not_eq nullptr operator or or_eq private protected public register reinterpret_cast requires
        return short signed sizeof static static_assert static_cast struct switch template this
        thread_local throw true try typedef typeid typename union unsigned using virtual void
        volatile wchar_t while xor xor_eq)

    string(REPLACE "::" "/" namespace_dir "${arg_NAMESPACE}")
    set(generated "${CMAKE_CURRENT_BINARY_DIR}/dvv_shaders/${target}")
    set(embed_script "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/DvvEmbedShader.cmake")
    # Function names and their files, also across several calls for one target.
    get_property(functions TARGET ${target} PROPERTY DVV_SHADER_FUNCTIONS)
    get_property(sources TARGET ${target} PROPERTY DVV_SHADER_SOURCES)
    foreach(shader IN LISTS arg_SHADERS)
        cmake_path(ABSOLUTE_PATH shader BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                   NORMALIZE OUTPUT_VARIABLE source)
        cmake_path(GET source STEM LAST_ONLY name)
        _dvv_shader_function("${name}" function)
        if(NOT function MATCHES "^[a-z][A-Za-z0-9]*$" OR function IN_LIST keywords)
            message(FATAL_ERROR "dvv_add_shaders: '${shader}' does not make a C++ function name "
                                "('${function}')")
        endif()
        list(FIND functions "${function}" duplicate)
        if(NOT duplicate EQUAL -1)
            list(GET sources ${duplicate} other)
            message(FATAL_ERROR
                "dvv_add_shaders: '${source}' gives the same name as '${other}' (${function})")
        endif()
        list(APPEND functions "${function}")
        list(APPEND sources "${source}")

        set(spirv "${generated}/${name}.spv")
        set(reflection "${generated}/${name}.json")
        set(depfile "${generated}/${name}.d")
        set(header "${generated}/${namespace_dir}/${name}.h")
        set(cpp "${generated}/${name}.cpp")
        set(stamp "${generated}/${name}.stamp")
        add_custom_command(
            OUTPUT "${spirv}" "${reflection}"
            COMMAND "${DVV_SLANGC}" "${source}" -target spirv -profile spirv_1_6 -entry main
                    -stage compute -O2 -warnings-as-errors all
                    -reflection-json "${reflection}" -depfile "${depfile}" -o "${spirv}"
            DEPENDS "${source}"
            DEPFILE "${depfile}"
            COMMENT "slangc ${name}.slang"
            VERBATIM)
        # The script rewrites the source and the header only when they change, and always touches
        # the stamp, so the rule is up to date afterwards on every generator.
        add_custom_command(
            OUTPUT "${stamp}"
            BYPRODUCTS "${cpp}" "${header}"
            COMMAND "${CMAKE_COMMAND}" "-DNAME=${name}" "-DFUNCTION=${function}"
                    "-DNAMESPACE=${arg_NAMESPACE}" "-DHEADER_INCLUDE=${namespace_dir}/${name}.h"
                    "-DSPIRV=${spirv}" "-DREFLECTION=${reflection}" "-DHEADER=${header}"
                    "-DSOURCE=${cpp}" "-DSTAMP=${stamp}" -P "${embed_script}"
            DEPENDS "${spirv}" "${reflection}" "${embed_script}"
            COMMENT "embed ${name}.slang"
            VERBATIM)
        # The arrays in every generated source have the same names.
        set_source_files_properties("${cpp}" PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
        target_sources(${target} PRIVATE "${cpp}" "${header}" "${stamp}")
    endforeach()
    set_property(TARGET ${target} PROPERTY DVV_SHADER_FUNCTIONS "${functions}")
    set_property(TARGET ${target} PROPERTY DVV_SHADER_SOURCES "${sources}")
    target_include_directories(${target} PUBLIC "$<BUILD_INTERFACE:${generated}>")
    target_link_libraries(${target} PUBLIC dvv::gpu)
endfunction()
