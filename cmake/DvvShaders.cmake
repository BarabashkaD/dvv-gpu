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
#
# With IMPORTS, the shaders import Slang sources that other targets export with
#
# dvv_export_shader_sources(<target> [NAMESPACE <namespace>] SOURCES <file.slang>...)
#
# which copies the sources at configure time (a changed source re-runs configure) to
# "dvv_shader_exports/<target>/<namespace>/" in the target's binary directory, sets the target
# property DVV_SHADER_IMPORT_DIR to "dvv_shader_exports/<target>" there and installs the sources
# to "<datadir>/<target>/shaders/<namespace>" (<datadir> is CMAKE_INSTALL_DATADIR at the target's
# first call). The namespace defaults to the target name as a C identifier. A shader imports a
# source as `import <namespace>.<stem>;`. <target> is a real target, not an alias. Several calls
# for one target add to its exports, also from other directories; make them all before the
# dvv_add_shaders() call that imports the target.
#
# This file owns the install folder: the target property DVV_SHADER_INSTALL_DIR holds
# "<datadir>/<target>/shaders" and is exported with the target (EXPORT_PROPERTIES). An installed
# package's config file turns it into DVV_SHADER_IMPORT_DIR on its imported target:
#
#     find_dependency(dvv_gpu)   # gives dvv_import_shader_sources()
#     include("${CMAKE_CURRENT_LIST_DIR}/<package>Targets.cmake")
#     dvv_import_shader_sources(<ns>::<name> PREFIX "${PACKAGE_PREFIX_DIR}")
#
# Each import (an alias is resolved) adds its DVV_SHADER_IMPORT_DIR to slangc's search path. An
# import without the property or without sources in its folder, and two imports that export the
# same "<namespace>/<file>", fail configuring.
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
    cmake_parse_arguments(PARSE_ARGV 1 arg "" "NAMESPACE" "SHADERS;IMPORTS")
    if(arg_UNPARSED_ARGUMENTS OR NOT arg_SHADERS)
        message(FATAL_ERROR
            "usage: dvv_add_shaders(<target> [NAMESPACE <namespace>] SHADERS <file.slang>... "
            "[IMPORTS <target>...])")
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

    # slangc's search path for the imports. Each exported "<namespace>/<file>" has one owner.
    set(import_arguments "")
    set(import_targets "")
    set(exported_files "")
    set(exported_by "")
    foreach(import IN LISTS arg_IMPORTS)
        if(NOT TARGET ${import})
            message(FATAL_ERROR "dvv_add_shaders: '${import}' is not a target")
        endif()
        get_target_property(real ${import} ALIASED_TARGET)
        if(NOT real)
            set(real ${import})
        endif()
        if(real IN_LIST import_targets)
            continue()
        endif()
        list(APPEND import_targets ${real})
        get_property(import_dir TARGET ${real} PROPERTY DVV_SHADER_IMPORT_DIR)
        if(NOT import_dir)
            message(FATAL_ERROR "dvv_add_shaders: '${import}' exports no shader sources "
                                "(DVV_SHADER_IMPORT_DIR is not set)")
        endif()
        # The exports of a target of this build are known (its folder may still hold copies the
        # deferred prune removes); the folder of an installed package is read.
        get_property(files TARGET ${real} PROPERTY DVV_SHADER_EXPORTS)
        if(NOT files)
            file(GLOB_RECURSE files RELATIVE "${import_dir}" "${import_dir}/*.slang")
        endif()
        if(NOT files)
            message(FATAL_ERROR "dvv_add_shaders: '${import}' exports no shader sources "
                                "('${import_dir}' holds no .slang file)")
        endif()
        foreach(file IN LISTS files)
            list(FIND exported_files "${file}" owner)
            if(NOT owner EQUAL -1)
                list(GET exported_by ${owner} other)
                message(FATAL_ERROR "dvv_add_shaders: '${file}' is exported by both '${other}' "
                                    "and '${import}'")
            endif()
            list(APPEND exported_files "${file}")
            list(APPEND exported_by "${import}")
        endforeach()
        list(APPEND import_arguments -I "${import_dir}")
    endforeach()

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
                    -reflection-json "${reflection}" -depfile "${depfile}" ${import_arguments}
                    -o "${spirv}"
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

# Removes the copies in <dir> that <target> no longer exports. Deferred to the end of the
# top-level directory, when every dvv_export_shader_sources() call for the target is done.
function(_dvv_prune_shader_exports target dir)
    get_property(exported TARGET ${target} PROPERTY DVV_SHADER_EXPORTS)
    file(GLOB_RECURSE copies RELATIVE "${dir}" "${dir}/*")
    foreach(copy IN LISTS copies)
        if(NOT copy IN_LIST exported)
            file(REMOVE "${dir}/${copy}")
        endif()
    endforeach()
endfunction()

function(dvv_export_shader_sources target)
    cmake_parse_arguments(PARSE_ARGV 1 arg "" "NAMESPACE" "SOURCES")
    if(arg_UNPARSED_ARGUMENTS OR NOT arg_SOURCES)
        message(FATAL_ERROR "usage: dvv_export_shader_sources(<target> [NAMESPACE <namespace>] "
                            "SOURCES <file.slang>...)")
    endif()
    if(NOT TARGET ${target})
        message(FATAL_ERROR "dvv_export_shader_sources: '${target}' is not a target")
    endif()
    get_target_property(aliased ${target} ALIASED_TARGET)
    if(aliased)
        message(FATAL_ERROR
            "dvv_export_shader_sources: '${target}' is an alias; export from '${aliased}'")
    endif()
    if(NOT arg_NAMESPACE)
        string(MAKE_C_IDENTIFIER "${target}" arg_NAMESPACE)
    endif()
    if(NOT arg_NAMESPACE MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
        message(FATAL_ERROR
            "dvv_export_shader_sources: '${arg_NAMESPACE}' is not a Slang module namespace")
    endif()
    include(GNUInstallDirs)

    # One folder per target, whichever directory a call comes from.
    get_target_property(binary_dir ${target} BINARY_DIR)
    set(dir "${binary_dir}/dvv_shader_exports/${target}")
    # The exported "<namespace>/<file>" names, also across several calls for one target.
    get_property(exported TARGET ${target} PROPERTY DVV_SHADER_EXPORTS)
    if(exported)
        # One install folder per target too: the first call's.
        get_property(install_dir TARGET ${target} PROPERTY DVV_SHADER_INSTALL_DIR)
    else()
        set(install_dir "${CMAKE_INSTALL_DATADIR}/${target}/shaders")
        # The end of the top-level directory comes after every call, wherever it is made.
        cmake_language(EVAL CODE
            "cmake_language(DEFER DIRECTORY [==[${CMAKE_SOURCE_DIR}]==] CALL "
            "_dvv_prune_shader_exports [==[${target}]==] [==[${dir}]==])")
        set_property(TARGET ${target} PROPERTY DVV_SHADER_INSTALL_DIR "${install_dir}")
        set_property(TARGET ${target} APPEND PROPERTY EXPORT_PROPERTIES DVV_SHADER_INSTALL_DIR)
    endif()
    set(sources "")
    foreach(file IN LISTS arg_SOURCES)
        cmake_path(ABSOLUTE_PATH file BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" NORMALIZE
                   OUTPUT_VARIABLE source)
        cmake_path(GET source EXTENSION LAST_ONLY extension)
        if(NOT extension STREQUAL ".slang")
            message(FATAL_ERROR "dvv_export_shader_sources: '${file}' is not a .slang file")
        endif()
        cmake_path(GET source FILENAME name)
        if("${arg_NAMESPACE}/${name}" IN_LIST exported)
            message(FATAL_ERROR "dvv_export_shader_sources: '${arg_NAMESPACE}/${name}' is "
                                "exported twice by '${target}'")
        endif()
        list(APPEND exported "${arg_NAMESPACE}/${name}")
        list(APPEND sources "${source}")
        configure_file("${source}" "${dir}/${arg_NAMESPACE}/${name}" COPYONLY)
    endforeach()
    set_property(TARGET ${target} PROPERTY DVV_SHADER_EXPORTS "${exported}")
    set_property(TARGET ${target} PROPERTY DVV_SHADER_IMPORT_DIR "${dir}")
    install(FILES ${sources} DESTINATION "${install_dir}/${arg_NAMESPACE}")
endfunction()

# dvv_import_shader_sources(<target> PREFIX <package prefix>): for an installed package's config
# file. Sets DVV_SHADER_IMPORT_DIR of the imported <target> to its exported DVV_SHADER_INSTALL_DIR
# below <package prefix>.
function(dvv_import_shader_sources target)
    cmake_parse_arguments(PARSE_ARGV 1 arg "" "PREFIX" "")
    if(arg_UNPARSED_ARGUMENTS OR NOT arg_PREFIX)
        message(FATAL_ERROR "usage: dvv_import_shader_sources(<target> PREFIX <package prefix>)")
    endif()
    if(NOT TARGET ${target})
        message(FATAL_ERROR "dvv_import_shader_sources: '${target}' is not a target")
    endif()
    get_target_property(imported ${target} IMPORTED)
    if(NOT imported)
        message(FATAL_ERROR "dvv_import_shader_sources: '${target}' is not an imported target")
    endif()
    get_property(install_dir TARGET ${target} PROPERTY DVV_SHADER_INSTALL_DIR)
    if(NOT install_dir)
        message(FATAL_ERROR "dvv_import_shader_sources: '${target}' exports no shader sources "
                            "(DVV_SHADER_INSTALL_DIR is not set)")
    endif()
    cmake_path(ABSOLUTE_PATH install_dir BASE_DIRECTORY "${arg_PREFIX}" NORMALIZE
               OUTPUT_VARIABLE import_dir)
    set_property(TARGET ${target} PROPERTY DVV_SHADER_IMPORT_DIR "${import_dir}")
endfunction()
