#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "dvv/gpu/bundled_drivers.h"

namespace dvv::gpu::detail {
struct DriverSetting {
    DriverChoice choice = DriverChoice::NoManifest;
    std::string value{};  // the VK_DRIVER_FILES value when choice is Bundled, else empty
};
// The decision and value for `manifests`, with the environment read through `get_env`. The value
// is the existing manifests' path text (dvv::pathText), joined by ';' on Windows and ':' elsewhere.
// Errc::Unsupported when an existing manifest's path text contains that separator.
[[nodiscard]] Result<DriverSetting> bundledDriverSetting(
    std::span<const std::filesystem::path> manifests,
    const std::function<std::optional<std::string>(std::string_view)>& get_env);
}  // namespace dvv::gpu::detail
