#include "dvv/platform/dynamic_library.h"
#include "loader.h"

namespace dvv::gpu::detail {

LoaderLibrary systemLoaderLibrary() {
    // Only through the executable's LC_RPATH (no system folder holds a Vulkan loader on macOS):
    // CMAKE_BUILD_RPATH in a build tree, @executable_path/../Frameworks in the app. The "@rpath/"
    // prefix is dvv-gpu's loader rule; dvv-platform's libraryFileName gives the bare name.
    return {.name = "@rpath/" + platform::libraryFileName("vulkan", 1),
            .search = platform::LibrarySearch::Default};
}

}  // namespace dvv::gpu::detail
