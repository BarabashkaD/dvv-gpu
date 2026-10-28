#pragma once

// The Vulkan instance behind a GpuRegistry: validation settings, the instance and its dispatcher,
// and the debug messenger that counts and logs messages.

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "devices.h"
#include "dvv/base/error.h"
#include "dvv/base/log.h"
#include "dvv/gpu/gpu_registry.h"
#include "dvv/platform/native_window.h"
#include "vk.h"

namespace dvv::gpu::detail {

inline constexpr char VALIDATION_LAYER[] = "VK_LAYER_KHRONOS_validation";

// What GpuRegistry::create was asked for, before the layer's presence is known.
struct ValidationRequest {
    ValidationMode mode = ValidationMode::Off;
    bool required = false;
    bool explicitlyRequested = false;  // by the options or DVV_VK_VALIDATION, not by default
};

// "off", "on" or "sync"; Errc::Format otherwise.
[[nodiscard]] Result<ValidationMode> parseValidationMode(std::string_view text);

// The options win; unset options fall back to the environment values (nullopt or empty = unset),
// then to the default (On in Debug builds, Off in Release builds; not required).
[[nodiscard]] Result<ValidationRequest> resolveValidation(
    const GpuRegistryOptions& options, const std::optional<std::string>& environment_mode,
    const std::optional<std::string>& environment_required, bool debug_build);

// The mode to enable: the requested one when the layer is installed (in version 1.4 or newer);
// otherwise Off, or Errc::Unsupported when validation is required.
[[nodiscard]] Result<ValidationMode> effectiveValidation(const ValidationRequest& request,
                                                         bool layer_installed);

// Messenger policy: errors of any type count, and so do validation or performance warnings.
// General warnings come from the loader and depend on the machine, so they do not count.
[[nodiscard]] bool countsAsFailure(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
                                   vk::DebugUtilsMessageTypeFlagsEXT types) noexcept;

// Counted messages are logged at their severity (Error, Warn); the others one level lower
// (warning -> Info, info -> Debug, verbose -> Trace).
[[nodiscard]] LogLevel debugMessageLevel(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
                                         bool counted) noexcept;

// "Vulkan validation error [VUID-...]: <message>"; types are joined with '|'.
[[nodiscard]] std::string formatDebugMessage(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
                                             vk::DebugUtilsMessageTypeFlagsEXT types,
                                             std::string_view id_name, std::string_view message);

// The instance extensions the registry's window system needs: none for a headless registry,
// otherwise platform::surfaceInstanceExtensions(*options.windowSystem).
[[nodiscard]] std::span<const char* const> windowSystemExtensions(
    const GpuRegistryOptions& options) noexcept;

// Errc::Unsupported "the Vulkan loader does not offer <extension>, which the <name> window system
// needs" for the first required extension that is not available.
[[nodiscard]] Result<void> requireInstanceExtensions(
    std::span<const char* const> required, std::span<const vk::ExtensionProperties> available,
    std::string_view window_system_name);

// vk-bootstrap keeps process-wide state: the first vkGetInstanceProcAddr it is given (loader.h)
// and the instance-level function pointers of the first instance, which stay in use after that
// instance is gone. The core pointers are loader trampolines and safe to share. The surface
// pointers come from that first instance and are null when it was headless. Use vk-bootstrap only
// through InstanceBuilder constructed with vulkanLoader()'s vkGetInstanceProcAddr, and through its
// physical-device and device builders without a surface. Never use vkb::destroy_surface,
// SwapchainBuilder, a selector with a surface, or an entry point without that pointer: it would
// load vulkan-1.dll with the default search. Surfaces come from dvv::platform::createSurface, and
// swapchains from Vulkan-Hpp with the device dispatcher. Its tables are not synchronised, so every
// vk-bootstrap call (instance, physical device or device builders) must hold this mutex.
[[nodiscard]] std::mutex& vkBootstrapMutex();

// Whether a failed instance build is retried without the validation layer: vkCreateInstance itself
// failed (`create_instance_failed`; a later step such as creating the messenger leaves an instance
// behind), the layer was listed but could not be loaded, and validation is not required. Loaders
// report that as VK_ERROR_LAYER_NOT_PRESENT (Linux loader 1.3.275) or VK_ERROR_OUT_OF_HOST_MEMORY
// (Windows loader 1.4.341); a real out-of-memory failure then fails again in the retry.
[[nodiscard]] bool retryWithoutValidation(bool create_instance_failed, VkResult result,
                                          ValidationMode validation, bool required) noexcept;

// Implementation part of GpuRegistry (exempt from spec §9.4 registration: owned only by its
// registry, with the same lifetime). Created only by create(); the public default constructor
// serves create() and white-box tests.
class InstanceCore {
public:
    [[nodiscard]] static Result<std::unique_ptr<InstanceCore>> create(
        const GpuRegistryOptions& options);

    InstanceCore() = default;
    InstanceCore(const InstanceCore&) = delete;
    InstanceCore& operator=(const InstanceCore&) = delete;
    InstanceCore(InstanceCore&&) = delete;
    InstanceCore& operator=(InstanceCore&&) = delete;
    // Destroys the messenger, then the instance.
    ~InstanceCore();

    [[nodiscard]] vk::Instance instance() const noexcept { return instance_; }
    [[nodiscard]] const vk::detail::DispatchLoaderDynamic& dispatch() const noexcept {
        return dispatch_;
    }
    [[nodiscard]] ValidationMode validation() const noexcept { return validation_; }
    // VK_EXT_debug_utils is enabled: messages are routed and objects can carry debug names.
    [[nodiscard]] bool debugUtils() const noexcept { return debug_utils_; }
    [[nodiscard]] std::uint32_t loaderVersion() const noexcept { return loader_version_; }
    // The window system whose surface extensions are enabled; nullopt for a headless instance.
    [[nodiscard]] std::optional<platform::WindowSystem> windowSystem() const noexcept {
        return window_system_;
    }
    // The instance's physical devices, indexed by DeviceId.
    [[nodiscard]] const std::vector<PhysicalDevice>& physicalDevices() const noexcept {
        return physical_devices_;
    }
    void setPhysicalDevices(std::vector<PhysicalDevice> devices) noexcept {
        physical_devices_ = std::move(devices);
    }
    [[nodiscard]] std::uint64_t validationMessageCount() const noexcept {
        return validation_messages_.load(std::memory_order_relaxed);
    }

    // The messenger callback: counts the message by the policy above and logs it. Any thread.
    void onMessage(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
                   vk::DebugUtilsMessageTypeFlagsEXT types,
                   const vk::DebugUtilsMessengerCallbackDataEXT& data) noexcept;

private:
    vk::detail::DispatchLoaderDynamic dispatch_;
    vk::Instance instance_;
    vk::DebugUtilsMessengerEXT messenger_;
    ValidationMode validation_ = ValidationMode::Off;
    bool debug_utils_ = false;
    std::uint32_t loader_version_ = 0;
    std::optional<platform::WindowSystem> window_system_;
    std::atomic<std::uint64_t> validation_messages_{0};
    std::vector<PhysicalDevice> physical_devices_;
};

}  // namespace dvv::gpu::detail
