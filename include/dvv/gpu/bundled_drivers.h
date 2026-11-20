#pragma once

#include <cstdint>
#include <filesystem>
#include <span>

#include "dvv/base/error.h"

namespace dvv::gpu {

// What useBundledDrivers decided.
enum class DriverChoice : std::uint8_t {
    NoManifest,  // none of the manifests is a regular file: nothing was set
    UserChoice,  // VK_DRIVER_FILES or VK_ICD_FILENAMES is set and not empty: left as it is
    Bundled,     // VK_DRIVER_FILES now names the existing manifests, in the given order
};

// Makes the GPU backend use only the given bundled driver manifests (Vulkan ICD JSON files),
// unless the user has chosen drivers (VK_DRIVER_FILES or VK_ICD_FILENAMES set and not empty; an
// empty value counts as unset, as for the loader). Manifests that are not regular files are
// skipped. Call it before any other thread starts and before the first GpuInstance::create: it
// sets a process environment variable, which is not synchronised (dvv/base/environment.h).
// Errc::Unsupported, and nothing set, when an existing manifest's path contains the OS list
// separator (';' on Windows, ':' elsewhere; on macOS a '/' typed in a Finder name is a ':' in the
// path): the loader would split the value there. Bundle the drivers under a path without one.
// Errc::Internal when the variable cannot be set: on Windows when the joined value is 32767 UTF-16
// units or longer, on Linux when a manifest's path text is not valid UTF-8. The Vulkan loader
// ignores the variable in an elevated process (a Windows administrator, a setuid program on Linux).
[[nodiscard]] Result<DriverChoice> useBundledDrivers(
    std::span<const std::filesystem::path> manifests);

}  // namespace dvv::gpu
