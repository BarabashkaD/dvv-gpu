#include "dvv/platform/dynamic_library.h"
#include "loader.h"

namespace dvv::gpu::detail {

LoaderLibrary systemLoaderLibrary() {
    // The loader's soname; on Linux the system-directory search is the default library search.
    return {.name = platform::libraryFileName("vulkan", 1),
            .search = platform::LibrarySearch::SystemDirectory};
}

}  // namespace dvv::gpu::detail
