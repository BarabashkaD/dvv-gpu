#include "dvv/platform/dynamic_library.h"
#include "loader.h"

namespace dvv::gpu::detail {

LoaderLibrary systemLoaderLibrary() {
    // Only from the system directory: a vulkan-1.dll next to the executable, in the current
    // directory or on PATH is not found (an already loaded module of that name is used as is).
    return {.name = platform::libraryFileName("vulkan", 1),
            .search = platform::LibrarySearch::SystemDirectory};
}

}  // namespace dvv::gpu::detail
