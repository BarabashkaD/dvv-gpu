#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "access.h"
#include "dvv/base/environment.h"
#include "dvv/base/log.h"
#include "dvv/gpu/gpu_info.h"
#include "dvv/gpu/gpu_instance.h"
#include "dvv/platform/window_handle.h"
#include "gpu_fixture.h"
#include "instance.h"

using dvv::gpu::DeviceId;
using dvv::gpu::GpuInfo;
using dvv::gpu::GpuInstance;
using dvv::gpu::detail::Access;
using dvv::gpu::test::CapturedLog;
using dvv::gpu::test::GpuTest;
using dvv::gpu::test::TemporaryDirectory;
using dvv::platform::WindowSystem;

namespace {

// Sets (or, with an empty value, removes) an environment variable for one scope.
class ScopedEnvironment {
public:
    ScopedEnvironment(std::string name, const std::string& value)
        : name_(std::move(name)), old_(dvv::getEnvironment(name_)) {
        REQUIRE(dvv::setEnvironment(name_, value));
    }
    ScopedEnvironment(const ScopedEnvironment&) = delete;
    ScopedEnvironment& operator=(const ScopedEnvironment&) = delete;
    ScopedEnvironment(ScopedEnvironment&&) = delete;
    ScopedEnvironment& operator=(ScopedEnvironment&&) = delete;
    ~ScopedEnvironment() { static_cast<void>(dvv::setEnvironment(name_, old_.value_or(""))); }

private:
    std::string name_;
    std::optional<std::string> old_;
};

#if defined(_WIN32)
constexpr char MISSING_LAYER_LIBRARY[] = "./dvv_missing_validation_layer.dll";
constexpr char PLATFORM_SURFACE_FUNCTION[] = "vkCreateWin32SurfaceKHR";
constexpr WindowSystem OTHER_WINDOW_SYSTEM = WindowSystem::Xlib;
constexpr char OTHER_SURFACE_EXTENSION[] = "VK_KHR_xlib_surface";
#elif defined(__APPLE__)
constexpr char MISSING_LAYER_LIBRARY[] = "./libdvv_missing_validation_layer.dylib";
constexpr char PLATFORM_SURFACE_FUNCTION[] = "vkCreateMetalSurfaceEXT";
constexpr WindowSystem OTHER_WINDOW_SYSTEM = WindowSystem::Xlib;
constexpr char OTHER_SURFACE_EXTENSION[] = "VK_KHR_xlib_surface";
#else
constexpr char MISSING_LAYER_LIBRARY[] = "./libdvv_missing_validation_layer.so";
constexpr char PLATFORM_SURFACE_FUNCTION[] = "vkCreateXlibSurfaceKHR";
constexpr WindowSystem OTHER_WINDOW_SYSTEM = WindowSystem::Win32;
constexpr char OTHER_SURFACE_EXTENSION[] = "VK_KHR_win32_surface";
#endif

}  // namespace

TEST_CASE_FIXTURE(GpuTest, "leak reports name the GPU instance GpuInstance") {
    CHECK(gpuInstance().typeName() == "dvv::gpu::GpuInstance");
}

TEST_CASE_FIXTURE(GpuTest, "the GPU instance finds a usable Vulkan 1.4 device and ranks it first") {
    CHECK(gpuInstance().loaderVersion() >= VK_API_VERSION_1_3);
    REQUIRE_FALSE(gpuInstance().devices().empty());
    REQUIRE(gpuInstance().rankedDevices().size() == gpuInstance().devices().size());
    CHECK(gpuInstance().rankedDevices().front() == device());

    for (std::uint32_t index = 0; index < gpuInstance().devices().size(); ++index) {
        CHECK(gpuInstance().devices()[index].id == DeviceId(index));
    }
    const GpuInfo& best = gpuInstance().devices()[device().index()];
    INFO(best.name);
    CHECK(best.usable());
    CHECK(best.apiVersion >= VK_API_VERSION_1_4);
    CHECK_FALSE(best.name.empty());
    CHECK_FALSE(best.driverName.empty());
    CHECK(best.deviceLocalMemoryBytes > 0);
    CHECK(std::ranges::any_of(best.deviceUuid, [](std::uint8_t byte) { return byte != 0; }));
}

TEST_CASE_FIXTURE(GpuTest, "the default GPU instance is headless") {
    CHECK(gpuInstance().windowSystem() == std::nullopt);
    const auto& instance = Access::instance(gpuInstance());
    CHECK(instance.dispatch().vkGetInstanceProcAddr(instance.instance(), "vkDestroySurfaceKHR") ==
          nullptr);
}

TEST_CASE("a GPU instance with the platform's window system enables the surface extensions") {
    const WindowSystem system = dvv::platform::defaultWindowSystem();
    const auto gpu_instance = GpuInstance::create({.windowSystem = system});
    if (!gpu_instance) {
        FAIL(gpu_instance.error().describe());
    }
    CHECK((*gpu_instance)->windowSystem() == system);
    const auto& instance = Access::instance(**gpu_instance);
    const auto get_instance_proc_addr = instance.dispatch().vkGetInstanceProcAddr;
    CHECK(get_instance_proc_addr(instance.instance(), "vkDestroySurfaceKHR") != nullptr);
    CHECK(get_instance_proc_addr(instance.instance(), PLATFORM_SURFACE_FUNCTION) != nullptr);
    const auto best = (*gpu_instance)->bestDevice();
    REQUIRE(best.has_value());
    CHECK((*gpu_instance)->devices()[best.value_or(DeviceId(0)).index()].optional.swapchain);
    CHECK((*gpu_instance)->validationMessageCount() == 0);
}

TEST_CASE("a window system the loader lacks is unsupported") {
    const auto gpu_instance = GpuInstance::create({.windowSystem = OTHER_WINDOW_SYSTEM});
    REQUIRE_FALSE(gpu_instance.has_value());
    CHECK(gpu_instance.error().code == dvv::Errc::Unsupported);
    CHECK(gpu_instance.error().message.find(OTHER_SURFACE_EXTENSION) != std::string::npos);
}

TEST_CASE_FIXTURE(GpuTest, "the GPU instance validates as DVV_VK_VALIDATION asks") {
    const auto requested = dvv::getEnvironment("DVV_VK_VALIDATION");
    if (!requested || requested->empty()) {
        MESSAGE("DVV_VK_VALIDATION is not set");
        return;
    }
    // The test environment points the loader at the prefix's layer, so it is always found.
    CHECK(dvv::gpu::validationModeName(gpuInstance().validation()) == *requested);
}

TEST_CASE("debug messages are counted by the messenger policy and logged") {
    using Severity = vk::DebugUtilsMessageSeverityFlagBitsEXT;
    using Type = vk::DebugUtilsMessageTypeFlagBitsEXT;

    const auto gpu_instance = GpuInstance::create().value();
    const auto& instance = dvv::gpu::detail::Access::instance(*gpu_instance);
    REQUIRE(instance.debugUtils());
    const CapturedLog log(dvv::LogLevel::Info);

    const auto submit = [&instance](Severity severity, Type type, const char* text) {
        const vk::DebugUtilsMessengerCallbackDataEXT data{.pMessageIdName = "dvv-test",
                                                          .pMessage = text};
        instance.instance().submitDebugUtilsMessageEXT(
            severity, vk::DebugUtilsMessageTypeFlagsEXT(type), data, instance.dispatch());
    };
    const std::uint64_t before = gpu_instance->validationMessageCount();
    submit(Severity::eError, Type::eGeneral, "counted general error");
    submit(Severity::eWarning, Type::eValidation, "counted validation warning");
    submit(Severity::eWarning, Type::eGeneral, "uncounted general warning");
    submit(Severity::eInfo, Type::eValidation, "uncounted validation info");
    CHECK(gpu_instance->validationMessageCount() - before == 2);
    CHECK(log.contains("Vulkan general error [dvv-test]: counted general error"));
    CHECK(log.contains("Vulkan validation warning [dvv-test]: counted validation warning"));
    CHECK(log.contains("Vulkan general warning [dvv-test]: uncounted general warning"));
    // Logged at Debug, below the Info level set here.
    CHECK_FALSE(log.contains("uncounted validation info"));
}

TEST_CASE("a deliberate validation error is counted") {
    const auto gpu_instance = GpuInstance::create().value();
    if (gpu_instance->validation() == dvv::gpu::ValidationMode::Off) {
        MESSAGE("validation is off; nothing to check");
        return;
    }
    const auto& instance = dvv::gpu::detail::Access::instance(*gpu_instance);
    const vk::PhysicalDevice physical =
        instance.physicalDevices()
            .at(gpu_instance->bestDevice().value_or(DeviceId(0)).index())
            .handle;
    const CapturedLog log(dvv::LogLevel::Error);
    // An image usage of 0 is invalid usage (VkPhysicalDeviceImageFormatInfo2::usage must not be 0).
    const vk::PhysicalDeviceImageFormatInfo2 info{.format = vk::Format::eR8G8B8A8Unorm,
                                                  .type = vk::ImageType::e2D,
                                                  .tiling = vk::ImageTiling::eOptimal};
    static_cast<void>(physical.getImageFormatProperties2(info, instance.dispatch()));
    CHECK(gpu_instance->validationMessageCount() > 0);
    CHECK(log.contains("Vulkan validation error"));
}

TEST_CASE("a validation layer that cannot be loaded turns validation off without a count") {
    // A manifest for the validation layer whose library does not exist, as the only layer the
    // loader sees. vkCreateInstance fails (VK_ERROR_LAYER_NOT_PRESENT from the Linux loader 1.3,
    // VK_ERROR_OUT_OF_HOST_MEMORY from the Windows loader 1.4), and the loader reports the failed
    // library load at ERROR through the instance-creation messenger, which counts it.
    const TemporaryDirectory folder;
    {
        std::ofstream manifest(folder.path() / "VkLayer_khronos_validation.json");
        manifest << R"({"file_format_version": "1.2.0", "layer": {)"
                 << R"("name": "VK_LAYER_KHRONOS_validation", "type": "GLOBAL", )"
                 << R"("library_path": ")" << MISSING_LAYER_LIBRARY << R"(", )"
                 << R"("api_version": "1.4.365", "implementation_version": "1", )"
                 << R"("description": "dvv-gpu test layer without a library"}})";
    }
    const ScopedEnvironment layer_path("VK_LAYER_PATH", folder.path().string());
    const ScopedEnvironment add_layer_path("VK_ADD_LAYER_PATH", "");

    {
        const CapturedLog log(dvv::LogLevel::Warn);
        const auto gpu_instance = GpuInstance::create(
            {.validation = dvv::gpu::ValidationMode::On, .validationRequired = false});
        if (!gpu_instance) {
            FAIL(gpu_instance.error().describe());
        }
        CHECK((*gpu_instance)->validation() == dvv::gpu::ValidationMode::Off);
        CHECK((*gpu_instance)->validationMessageCount() == 0);
        // With the loader's result: ErrorLayerNotPresent (Linux) or ErrorOutOfHostMemory (Windows).
        CHECK(log.contains("VK_LAYER_KHRONOS_validation could not be loaded (Error"));
        CHECK(log.contains("); validation is off"));
    }

    const auto required = GpuInstance::create(
        {.validation = dvv::gpu::ValidationMode::On, .validationRequired = true});
    REQUIRE_FALSE(required.has_value());
    CHECK((required.error().code == dvv::Errc::Unsupported ||
           required.error().code == dvv::Errc::OutOfMemory));
}
