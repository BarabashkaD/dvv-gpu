#include <X11/Xlib.h>
// vulkan_xlib.h needs the Xlib types (Display, Window, VisualID). X11's macros (None, Bool, Status,
// True, False, ...) stay in this file: no header includes X11.
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_xlib.h>

#include "dvv/platform/window_handle_xlib.h"
#include "surface_vulkan.h"

namespace dvv::gpu::detail {

std::expected<VkSurfaceKHR, VkResult> createPlatformSurface(
    VkInstance instance, PFN_vkGetInstanceProcAddr get_instance_proc_addr,
    const platform::WindowHandle& window) noexcept {
    const auto native = platform::xlibWindowOf(window);
    if (!native) {
        return std::unexpected(VK_ERROR_INITIALIZATION_FAILED);  // createSurface checked the system
    }
    if (native->display == nullptr || native->window == 0) {
        return std::unexpected(VK_ERROR_INITIALIZATION_FAILED);
    }
    const auto create = reinterpret_cast<PFN_vkCreateXlibSurfaceKHR>(
        get_instance_proc_addr(instance, "vkCreateXlibSurfaceKHR"));
    if (create == nullptr) {
        return std::unexpected(VK_ERROR_EXTENSION_NOT_PRESENT);
    }
    const VkXlibSurfaceCreateInfoKHR info{.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
                                          .pNext = nullptr,
                                          .flags = 0,
                                          .dpy = static_cast<Display*>(native->display),
                                          .window = static_cast<Window>(native->window)};
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    const VkResult result = create(instance, &info, nullptr, &surface);
    if (result != VK_SUCCESS) {
        return std::unexpected(result);
    }
    return surface;
}

}  // namespace dvv::gpu::detail
