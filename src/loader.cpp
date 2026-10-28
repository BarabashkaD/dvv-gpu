#include "loader.h"

#include <format>
#include <mutex>
#include <utility>
#include <vector>

#include "dvv/platform/dynamic_library.h"

namespace dvv::gpu::detail {

namespace {

// Process infrastructure: the loaded libraries, intentionally never destroyed (see loader.h).
struct LoadedLibraries {
    std::mutex mutex;
    std::vector<platform::DynamicLibrary> libraries;
    PFN_vkGetInstanceProcAddr defaultLoader = nullptr;
};

LoadedLibraries& loadedLibraries() {
    static auto* const LIBRARIES = new LoadedLibraries();
    return *LIBRARIES;
}

std::string describe(const std::string& library_name) {
    return library_name.empty() ? std::string("the default library")
                                : std::format("'{}'", library_name);
}

}  // namespace

LoaderLibrary loaderLibrary(const std::string& library_name) {
    if (!library_name.empty()) {
        return {.name = library_name, .search = platform::LibrarySearch::Default};
    }
#if defined(_WIN32)
    // Only from the system directory: a vulkan-1.dll next to the executable, in the current
    // directory or on PATH is not found (an already loaded module of that name is used as is).
    return {.name = "vulkan-1.dll", .search = platform::LibrarySearch::SystemDirectory};
#else
    return {.name = "libvulkan.so.1", .search = platform::LibrarySearch::SystemDirectory};
#endif
}

Result<PFN_vkGetInstanceProcAddr> loadVulkanLoader(const std::string& library_name) {
    const LoaderLibrary wanted = loaderLibrary(library_name);
    auto library = platform::DynamicLibrary::open(wanted.name, wanted.search);
    if (!library) {
        return makeError(Errc::Unsupported,
                         std::format("Vulkan loader not found ({})", describe(library_name)));
    }
    const auto get_instance_proc_addr =
        library->function<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    if (get_instance_proc_addr == nullptr) {
        return makeError(Errc::Unsupported,
                         std::format("Vulkan loader not found ({} has no vkGetInstanceProcAddr)",
                                     describe(library_name)));
    }
    library->keepLoaded();
    auto& loaded = loadedLibraries();
    const std::scoped_lock lock(loaded.mutex);
    loaded.libraries.push_back(std::move(*library));
    return get_instance_proc_addr;
}

Result<PFN_vkGetInstanceProcAddr> vulkanLoader() {
    auto& loaded = loadedLibraries();
    {
        const std::scoped_lock lock(loaded.mutex);
        if (loaded.defaultLoader != nullptr) {
            return loaded.defaultLoader;
        }
    }
    DVV_TRY(const auto get_instance_proc_addr, loadVulkanLoader({}));
    const std::scoped_lock lock(loaded.mutex);
    if (loaded.defaultLoader == nullptr) {
        loaded.defaultLoader = get_instance_proc_addr;
    }
    return loaded.defaultLoader;
}

}  // namespace dvv::gpu::detail
