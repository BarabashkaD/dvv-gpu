#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <span>
#include <string_view>
#include <vector>

#include "dvv/base/error.h"
#include "dvv/gpu/gpu_registry.h"
#include "dvv/platform/native_window.h"
#include "instance.h"

using dvv::gpu::GpuRegistryOptions;
using dvv::gpu::detail::requireInstanceExtensions;
using dvv::gpu::detail::windowSystemExtensions;
using dvv::platform::WindowSystem;

namespace {

std::vector<std::string_view> names(std::span<const char* const> extensions) {
    return {extensions.begin(), extensions.end()};
}

vk::ExtensionProperties extension(std::string_view name) {
    vk::ExtensionProperties properties{};
    std::ranges::copy(name, properties.extensionName.begin());
    return properties;
}

}  // namespace

TEST_CASE("a headless registry asks for no surface extensions") {
    CHECK(windowSystemExtensions(GpuRegistryOptions{}).empty());
}

TEST_CASE("a window system adds VK_KHR_surface and its platform extension") {
    const std::vector<std::string_view> win32{"VK_KHR_surface", "VK_KHR_win32_surface"};
    const std::vector<std::string_view> xlib{"VK_KHR_surface", "VK_KHR_xlib_surface"};
    CHECK(names(windowSystemExtensions({.windowSystem = WindowSystem::Win32})) == win32);
    CHECK(names(windowSystemExtensions({.windowSystem = WindowSystem::Xlib})) == xlib);
}

TEST_CASE("missing instance extensions are unsupported") {
    const std::array<const char*, 2> required{"VK_KHR_surface", "VK_KHR_xlib_surface"};
    const std::vector<vk::ExtensionProperties> surface_only{extension("VK_KHR_surface")};
    const auto missing = requireInstanceExtensions(required, surface_only, "Xlib");
    REQUIRE_FALSE(missing.has_value());
    CHECK(missing.error().code == dvv::Errc::Unsupported);
    CHECK(missing.error().message ==
          "the Vulkan loader does not offer VK_KHR_xlib_surface, which the Xlib window system "
          "needs");

    const std::vector<vk::ExtensionProperties> both{extension("VK_KHR_xlib_surface"),
                                                    extension("VK_KHR_surface")};
    CHECK(requireInstanceExtensions(required, both, "Xlib").has_value());
    CHECK(requireInstanceExtensions({}, {}, "Win32").has_value());
}
