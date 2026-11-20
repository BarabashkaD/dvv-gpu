#include <doctest/doctest.h>

#include <memory>
#include <utility>

#include <vulkan/vulkan_core.h>

#include "access.h"
#include "dvv/gpu/gpu_instance.h"
#include "dvv/platform/testing/test_window.h"
#include "dvv/platform/window_handle.h"
#include "surface_vulkan.h"

namespace dvv::gpu::test {

namespace {

using platform::WindowHandle;
using platform::WindowSystem;
using platform::testing::TestWindow;

std::shared_ptr<GpuInstance> createGpuInstance(const GpuInstanceOptions& options) {
    auto gpu_instance = GpuInstance::create(options);
    if (!gpu_instance) {
        FAIL(gpu_instance.error().describe());
    }
    return std::move(*gpu_instance);
}

TestWindow testWindow() {
    auto window = TestWindow::create({.width = 320, .height = 240});
    if (!window) {
        FAIL(window.error().describe());
    }
    return std::move(*window);
}

}  // namespace

TEST_CASE("a surface is created for a test window and destroyed") {
    const auto gpu_instance = createGpuInstance({.windowSystem = platform::defaultWindowSystem()});
    // As PresenterCore's constructor reads them.
    const detail::InstanceCore& core = detail::Access::instance(*gpu_instance);
    const auto instance = static_cast<VkInstance>(core.instance());
    const PFN_vkGetInstanceProcAddr get_instance_proc_addr = core.dispatch().vkGetInstanceProcAddr;
    const TestWindow window = testWindow();
    const auto surface = detail::createSurface(instance, get_instance_proc_addr, window.handle());
    REQUIRE(surface.has_value());
    if (surface) {
        CHECK(*surface != VK_NULL_HANDLE);
        core.dispatch().vkDestroySurfaceKHR(instance, *surface, nullptr);  // before the window goes
    }
    CHECK(gpu_instance->validationMessageCount() == 0);
}

TEST_CASE("a window with null handles is rejected") {
    const auto gpu_instance = createGpuInstance({.windowSystem = platform::defaultWindowSystem()});
    const detail::InstanceCore& core = detail::Access::instance(*gpu_instance);
    const auto instance = static_cast<VkInstance>(core.instance());
    const PFN_vkGetInstanceProcAddr get_instance_proc_addr = core.dispatch().vkGetInstanceProcAddr;
    const TestWindow window = testWindow();
    const WindowHandle real = window.handle();
    const auto words = real.words();
#if defined(_WIN32)
    // Words: HINSTANCE, HWND.
    const WindowHandle no_hwnd = WindowHandle::make(WindowSystem::Win32, words[0], 0);
    const auto surface = detail::createSurface(instance, get_instance_proc_addr, no_hwnd);
    REQUIRE_FALSE(surface.has_value());
    if (!surface) {
        CHECK(surface.error() == VK_ERROR_INITIALIZATION_FAILED);
    }
#elif defined(__APPLE__)
    // Words: the CAMetalLayer, 0. A handle with a null layer is not empty (empty() reads only the
    // window system), so it reaches the Metal file's null check.
    const WindowHandle no_layer = WindowHandle::make(WindowSystem::Metal, 0, 0);
    CHECK_FALSE(no_layer.empty());
    const auto surface = detail::createSurface(instance, get_instance_proc_addr, no_layer);
    REQUIRE_FALSE(surface.has_value());
    if (!surface) {
        CHECK(surface.error() == VK_ERROR_INITIALIZATION_FAILED);
    }
    CHECK(words[0] != 0);
#else
    // Words: Display*, Window.
    const WindowHandle no_window = WindowHandle::make(WindowSystem::Xlib, words[0], 0);
    const auto without_window = detail::createSurface(instance, get_instance_proc_addr, no_window);
    REQUIRE_FALSE(without_window.has_value());
    if (!without_window) {
        CHECK(without_window.error() == VK_ERROR_INITIALIZATION_FAILED);
    }
    const WindowHandle no_display = WindowHandle::make(WindowSystem::Xlib, 0, words[1]);
    const auto without_display =
        detail::createSurface(instance, get_instance_proc_addr, no_display);
    REQUIRE_FALSE(without_display.has_value());
    if (!without_display) {
        CHECK(without_display.error() == VK_ERROR_INITIALIZATION_FAILED);
    }
#endif
    CHECK(gpu_instance->validationMessageCount() == 0);
}

TEST_CASE("an instance without the surface extensions cannot create surfaces") {
    const auto gpu_instance = createGpuInstance({});
    const detail::InstanceCore& core = detail::Access::instance(*gpu_instance);
    const auto instance = static_cast<VkInstance>(core.instance());
    const PFN_vkGetInstanceProcAddr get_instance_proc_addr = core.dispatch().vkGetInstanceProcAddr;
    const TestWindow window = testWindow();
    const auto surface = detail::createSurface(instance, get_instance_proc_addr, window.handle());
    REQUIRE_FALSE(surface.has_value());
    if (!surface) {
        CHECK(surface.error() == VK_ERROR_EXTENSION_NOT_PRESENT);
    }
    CHECK(gpu_instance->validationMessageCount() == 0);
}

}  // namespace dvv::gpu::test
