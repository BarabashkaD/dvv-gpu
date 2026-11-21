// vulkan_metal.h declares CAMetalLayer as an opaque type, so this file stays plain C++.
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_metal.h>

#include "dvv/platform/window_handle_metal.h"
#include "surface_vulkan.h"

namespace dvv::gpu::detail {

std::expected<VkSurfaceKHR, VkResult> createPlatformSurface(
    VkInstance instance, PFN_vkGetInstanceProcAddr get_instance_proc_addr,
    const platform::WindowHandle& window) noexcept {
    const auto native = platform::metalWindowOf(window);
    if (!native) {
        return std::unexpected(VK_ERROR_INITIALIZATION_FAILED);  // createSurface checked the system
    }
    if (native->layer == nullptr) {
        return std::unexpected(VK_ERROR_INITIALIZATION_FAILED);
    }
    const auto create = reinterpret_cast<PFN_vkCreateMetalSurfaceEXT>(
        get_instance_proc_addr(instance, "vkCreateMetalSurfaceEXT"));
    if (create == nullptr) {
        return std::unexpected(VK_ERROR_EXTENSION_NOT_PRESENT);
    }
    const VkMetalSurfaceCreateInfoEXT info{
        .sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT,
        .pNext = nullptr,
        .flags = 0,
        .pLayer = static_cast<const CAMetalLayer*>(native->layer)};
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    const VkResult result = create(instance, &info, nullptr, &surface);
    if (result != VK_SUCCESS) {
        return std::unexpected(result);
    }
    return surface;
}

}  // namespace dvv::gpu::detail
