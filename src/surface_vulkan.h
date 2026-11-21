// src/surface_vulkan.h (private). Vulkan's C header only, never vk.h (see "Vulkan headers" below).
#pragma once

#include <expected>
#include <span>

#include <vulkan/vulkan_core.h>

#include "dvv/platform/window_handle.h"

// Vulkan headers: vk.h requires one configuration of Vulkan-Hpp in every translation unit; a Vulkan
// platform macro next to it would declare more of its dispatch table there than elsewhere (an ODR
// violation). So this header and the surface_vulkan files include only <vulkan/vulkan_core.h> and
// their window system's C header (vulkan_win32.h, vulkan_xlib.h, vulkan_metal.h; none needs a
// platform macro), never vk.h. Files that include vk.h include this header after it.

namespace dvv::gpu::detail {
// {VK_KHR_surface, the window system's own extension}; empty for an unknown value.
[[nodiscard]] std::span<const char* const> surfaceInstanceExtensions(
    platform::WindowSystem system) noexcept;
// Through vkGetInstanceProcAddr(instance, "vkCreate...Surface..."). Errors, unchanged:
// VK_ERROR_INITIALIZATION_FAILED for an empty handle, a null instance or a null
// get_instance_proc_addr, or null native handles; VK_ERROR_EXTENSION_NOT_PRESENT for a window
// system this build lacks or a creator the instance does not expose; else the driver's result.
// The caller destroys the surface with vkDestroySurfaceKHR before the window.
[[nodiscard]] std::expected<VkSurfaceKHR, VkResult> createSurface(
    VkInstance instance, PFN_vkGetInstanceProcAddr get_instance_proc_addr,
    const platform::WindowHandle& window) noexcept;
// Defined once per build, in surface_vulkan_<win32|metal|xlib>.cpp; called only by
// createSurface after the shared checks (the handle's system is defaultWindowSystem(), the
// instance and get_instance_proc_addr are set).
[[nodiscard]] std::expected<VkSurfaceKHR, VkResult> createPlatformSurface(
    VkInstance instance, PFN_vkGetInstanceProcAddr get_instance_proc_addr,
    const platform::WindowHandle& window) noexcept;
}  // namespace dvv::gpu::detail
