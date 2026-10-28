#pragma once

// The Vulkan loader library, loaded at run time (nothing links it).

#include <string>

#include "dvv/base/error.h"
#include "dvv/platform/dynamic_library.h"
#include "vk.h"

namespace dvv::gpu::detail {

// The library loadVulkanLoader() opens and how it is searched.
struct LoaderLibrary {
    std::string name;
    platform::LibrarySearch search = platform::LibrarySearch::Default;
};

// Empty library_name: the platform's loader from the system directory ("vulkan-1.dll" on Windows,
// so a vulkan-1.dll next to the executable, in the current directory or on PATH is not found by the
// search (a module of that name already loaded in the process is used as is); "libvulkan.so.1" on
// Linux, where that search is the default one). Otherwise (tests) the given name, searched the
// usual way.
[[nodiscard]] LoaderLibrary loaderLibrary(const std::string& library_name);

// Loads the loader library (loaderLibrary(library_name)) and returns its vkGetInstanceProcAddr.
// Errc::Unsupported "Vulkan loader not found" when the library or the function is missing. A loaded
// library is never unloaded: vk-bootstrap keeps the first vkGetInstanceProcAddr it is given for the
// rest of the process (see vkBootstrapMutex() in instance.h for the rest of its process-wide
// state).
[[nodiscard]] Result<PFN_vkGetInstanceProcAddr> loadVulkanLoader(const std::string& library_name);

// The default loader: loaded on first success, then cached. A failure is retried on the next call.
[[nodiscard]] Result<PFN_vkGetInstanceProcAddr> vulkanLoader();

}  // namespace dvv::gpu::detail
