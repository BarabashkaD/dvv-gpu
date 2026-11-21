#include <doctest/doctest.h>

#include <cstdint>
#include <string_view>

#include <vulkan/vulkan_core.h>

#include "dvv/platform/window_handle.h"
#include "surface_vulkan.h"

#if defined(__APPLE__)
#include "dvv/platform/window_handle_metal.h"
#endif

using dvv::gpu::detail::createSurface;
using dvv::gpu::detail::surfaceInstanceExtensions;
using dvv::platform::WindowHandle;
using dvv::platform::WindowSystem;

TEST_CASE("each window system needs VK_KHR_surface and its own extension") {
    const auto win32 = surfaceInstanceExtensions(WindowSystem::Win32);
    REQUIRE(win32.size() == 2);
    CHECK(std::string_view(win32[0]) == "VK_KHR_surface");
    CHECK(std::string_view(win32[1]) == "VK_KHR_win32_surface");
    const auto xlib = surfaceInstanceExtensions(WindowSystem::Xlib);
    REQUIRE(xlib.size() == 2);
    CHECK(std::string_view(xlib[0]) == "VK_KHR_surface");
    CHECK(std::string_view(xlib[1]) == "VK_KHR_xlib_surface");
    const auto metal = surfaceInstanceExtensions(WindowSystem::Metal);
    REQUIRE(metal.size() == 2);
    CHECK(std::string_view(metal[0]) == "VK_KHR_surface");
    CHECK(std::string_view(metal[1]) == "VK_EXT_metal_surface");
}

TEST_CASE("createSurface rejects an empty window and a window system this build lacks") {
    const auto empty = createSurface(VK_NULL_HANDLE, nullptr, WindowHandle{});
    REQUIRE_FALSE(empty.has_value());
    if (!empty) {
        CHECK(empty.error() == VK_ERROR_INITIALIZATION_FAILED);
    }

    // Non-null fake words: only the window system decides, the words are never used.
    const WindowSystem other = dvv::platform::defaultWindowSystem() == WindowSystem::Win32
                                   ? WindowSystem::Xlib
                                   : WindowSystem::Win32;
    const auto lacking = createSurface(VK_NULL_HANDLE, nullptr, WindowHandle::make(other, 1, 1));
    REQUIRE_FALSE(lacking.has_value());
    if (!lacking) {
        CHECK(lacking.error() == VK_ERROR_EXTENSION_NOT_PRESENT);
    }

    const auto no_instance = createSurface(
        VK_NULL_HANDLE, nullptr, WindowHandle::make(dvv::platform::defaultWindowSystem(), 1, 1));
    REQUIRE_FALSE(no_instance.has_value());
    if (!no_instance) {
        CHECK(no_instance.error() == VK_ERROR_INITIALIZATION_FAILED);
    }
}

#if defined(__APPLE__)
namespace {

// Fake instance handle and function loader: the layer check and the missing-function check come
// before any Vulkan call, so no loader is needed.
// NOLINTNEXTLINE(performance-no-int-to-ptr): a fake instance handle, never dereferenced
const VkInstance FAKE_INSTANCE = reinterpret_cast<VkInstance>(std::uintptr_t{1});
PFN_vkVoidFunction noFunctions(VkInstance, const char*) {
    return nullptr;
}

}  // namespace

TEST_CASE("a Metal window without a layer is rejected") {
    const auto surface =
        createSurface(FAKE_INSTANCE, noFunctions,
                      dvv::platform::windowHandleOf(dvv::platform::MetalWindow{.layer = nullptr}));
    REQUIRE_FALSE(surface.has_value());
    if (!surface) {
        CHECK(surface.error() == VK_ERROR_INITIALIZATION_FAILED);
    }
}

TEST_CASE("an instance without vkCreateMetalSurfaceEXT cannot create Metal surfaces") {
    // NOLINTNEXTLINE(performance-no-int-to-ptr): a fake native handle, never dereferenced
    const dvv::platform::MetalWindow window{.layer = reinterpret_cast<void*>(std::uintptr_t{1})};
    const auto surface =
        createSurface(FAKE_INSTANCE, noFunctions, dvv::platform::windowHandleOf(window));
    REQUIRE_FALSE(surface.has_value());
    if (!surface) {
        CHECK(surface.error() == VK_ERROR_EXTENSION_NOT_PRESENT);
    }
}
#endif
