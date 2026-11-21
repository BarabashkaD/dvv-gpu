#include "surface_vulkan.h"

#include <array>

namespace dvv::gpu::detail {

namespace {

constexpr std::array<const char*, 2> WIN32_EXTENSIONS{VK_KHR_SURFACE_EXTENSION_NAME,
                                                      "VK_KHR_win32_surface"};
constexpr std::array<const char*, 2> XLIB_EXTENSIONS{VK_KHR_SURFACE_EXTENSION_NAME,
                                                     "VK_KHR_xlib_surface"};
constexpr std::array<const char*, 2> METAL_EXTENSIONS{VK_KHR_SURFACE_EXTENSION_NAME,
                                                      "VK_EXT_metal_surface"};

}  // namespace

std::span<const char* const> surfaceInstanceExtensions(platform::WindowSystem system) noexcept {
    switch (system) {
        case platform::WindowSystem::Win32: return WIN32_EXTENSIONS;
        case platform::WindowSystem::Xlib: return XLIB_EXTENSIONS;
        case platform::WindowSystem::Metal: return METAL_EXTENSIONS;
    }
    return {};
}

std::expected<VkSurfaceKHR, VkResult> createSurface(
    VkInstance instance, PFN_vkGetInstanceProcAddr get_instance_proc_addr,
    const platform::WindowHandle& window) noexcept {
    if (window.empty()) {
        return std::unexpected(VK_ERROR_INITIALIZATION_FAILED);
    }
    if (window.system() != platform::defaultWindowSystem()) {
        return std::unexpected(VK_ERROR_EXTENSION_NOT_PRESENT);  // this build lacks it
    }
    if (instance == VK_NULL_HANDLE || get_instance_proc_addr == nullptr) {
        return std::unexpected(VK_ERROR_INITIALIZATION_FAILED);
    }
    return createPlatformSurface(instance, get_instance_proc_addr, window);
}

}  // namespace dvv::gpu::detail
