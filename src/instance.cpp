#include "instance.h"

#include <algorithm>
#include <cstdio>
#include <format>
#include <mutex>
#include <source_location>
#include <span>
#include <string>
#include <utility>

#include <VkBootstrap.h>

#include "dvv/base/environment.h"
#include "dvv/platform/surface.h"
#include "loader.h"
#include "vk_result.h"

namespace dvv::gpu::detail {

namespace {

#if defined(NDEBUG)
constexpr bool DEBUG_BUILD = false;
#else
constexpr bool DEBUG_BUILD = true;
#endif

using Severity = vk::DebugUtilsMessageSeverityFlagBitsEXT;
using Type = vk::DebugUtilsMessageTypeFlagBitsEXT;

VKAPI_ATTR VkBool32 VKAPI_CALL onDebugMessage(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                              VkDebugUtilsMessageTypeFlagsEXT types,
                                              const VkDebugUtilsMessengerCallbackDataEXT* data,
                                              void* user_data) noexcept {
    if (user_data != nullptr && data != nullptr) {
        static_cast<InstanceCore*>(user_data)->onMessage(
            static_cast<Severity>(severity), vk::DebugUtilsMessageTypeFlagsEXT(types),
            *reinterpret_cast<const vk::DebugUtilsMessengerCallbackDataEXT*>(data));
    }
    return VK_FALSE;  // never abort the Vulkan call
}

// Errc from the Vulkan result when there is one, otherwise Unsupported (vk-bootstrap's own checks:
// versions, layers, extensions).
template <class T>
std::unexpected<Error> vkBootstrapError(
    const vkb::Result<T>& result, std::string_view what,
    std::source_location where = std::source_location::current()) {
    const VkResult vk_result = result.vk_result();
    const Errc code =
        vk_result == VK_SUCCESS ? Errc::Unsupported : errcOf(static_cast<vk::Result>(vk_result));
    std::string message = std::format("{} failed: {}", what, result.error().message());
    if (vk_result != VK_SUCCESS) {
        message += std::format(" ({})", vk::to_string(static_cast<vk::Result>(vk_result)));
    }
    return makeError(code, std::move(message), where);
}

// Builds the instance with vk-bootstrap and the given extensions (the window system's); the
// caller holds vkBootstrapMutex().
vkb::Result<vkb::Instance> buildInstance(PFN_vkGetInstanceProcAddr get_instance_proc_addr,
                                         const GpuRegistryOptions& options, InstanceCore* core,
                                         std::span<const char* const> extensions, bool debug_utils,
                                         ValidationMode validation) {
    static constexpr VkBool32 ENABLED = VK_TRUE;
    vkb::InstanceBuilder builder(get_instance_proc_addr);
    // The instance asks for Vulkan 1.4 (the devices' API) but accepts a 1.3 loader: Ubuntu 24.04
    // ships loader 1.3.275, and the 1.4 lavapipe device works through it.
    // vk-bootstrap 1.4.365 drops both names (its empty() check is inverted), so the driver sees
    // empty names; they are set for the fixed versions.
    builder.set_app_name(options.applicationName)
        .set_engine_name("dvv")
        .set_minimum_instance_version(1, 3, 0)
        .require_api_version(1, 4, 0)
        .set_headless(true);
    // Headless keeps vk-bootstrap from adding its own choice of surface extensions; the window
    // system's are enabled here.
    for (const char* extension : extensions) {
        builder.enable_extension(extension);
    }
    if (debug_utils) {
        builder.set_debug_callback(&onDebugMessage)
            .set_debug_callback_user_data_pointer(core)
            .set_debug_messenger_severity(VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                          VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                                          VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                          VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
            .set_debug_messenger_type(VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                      VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                      VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT);
    }
    if (validation != ValidationMode::Off) {
        builder.enable_validation_layers(true);
    }
    if (validation == ValidationMode::Sync) {
        builder.add_layer_setting({.pLayerName = VALIDATION_LAYER,
                                   .pSettingName = "validate_sync",
                                   .type = VK_LAYER_SETTING_TYPE_BOOL32_EXT,
                                   .valueCount = 1,
                                   .pValues = &ENABLED});
    }
    return builder.build();
}

}  // namespace

std::span<const char* const> windowSystemExtensions(const GpuRegistryOptions& options) noexcept {
    if (!options.windowSystem) {
        return {};
    }
    return platform::surfaceInstanceExtensions(*options.windowSystem);
}

Result<void> requireInstanceExtensions(std::span<const char* const> required,
                                       std::span<const vk::ExtensionProperties> available,
                                       std::string_view window_system_name) {
    for (const char* name : required) {
        const bool offered =
            std::ranges::any_of(available, [name](const vk::ExtensionProperties& extension) {
                return std::string_view(extension.extensionName) == name;
            });
        if (!offered) {
            return makeError(Errc::Unsupported,
                             std::format("the Vulkan loader does not offer {}, which the {} window "
                                         "system needs",
                                         name, window_system_name));
        }
    }
    return {};
}

std::mutex& vkBootstrapMutex() {
    static std::mutex mutex;
    return mutex;
}

bool retryWithoutValidation(bool create_instance_failed, VkResult result, ValidationMode validation,
                            bool required) noexcept {
    return create_instance_failed &&
           (result == VK_ERROR_LAYER_NOT_PRESENT || result == VK_ERROR_OUT_OF_HOST_MEMORY) &&
           validation != ValidationMode::Off && !required;
}

Result<ValidationMode> parseValidationMode(std::string_view text) {
    if (text == "off") {
        return ValidationMode::Off;
    }
    if (text == "on") {
        return ValidationMode::On;
    }
    if (text == "sync") {
        return ValidationMode::Sync;
    }
    return makeError(Errc::Format,
                     std::format("Vulkan validation mode must be off, on or sync, got '{}'", text));
}

Result<ValidationRequest> resolveValidation(const GpuRegistryOptions& options,
                                            const std::optional<std::string>& environment_mode,
                                            const std::optional<std::string>& environment_required,
                                            bool debug_build) {
    ValidationRequest request{.mode = debug_build ? ValidationMode::On : ValidationMode::Off};
    if (options.validation) {
        request.mode = *options.validation;
        request.explicitlyRequested = true;
    } else if (environment_mode && !environment_mode->empty()) {
        const auto mode = parseValidationMode(*environment_mode);
        if (!mode) {
            return makeError(Errc::Format,
                             std::format("DVV_VK_VALIDATION must be off, on or sync, got '{}'",
                                         *environment_mode));
        }
        request.mode = *mode;
        request.explicitlyRequested = true;
    }
    if (options.validationRequired) {
        request.required = *options.validationRequired;
    } else if (environment_required && !environment_required->empty()) {
        if (*environment_required != "0" && *environment_required != "1") {
            return makeError(Errc::Format,
                             std::format("DVV_VK_VALIDATION_REQUIRED must be 0 or 1, got '{}'",
                                         *environment_required));
        }
        request.required = *environment_required == "1";
    }
    return request;
}

Result<ValidationMode> effectiveValidation(const ValidationRequest& request, bool layer_installed) {
    if (request.mode == ValidationMode::Off || layer_installed) {
        return request.mode;
    }
    if (request.required) {
        return makeError(
            Errc::Unsupported,
            std::format("Vulkan validation '{}' is required, but {} 1.4 or newer is not installed",
                        validationModeName(request.mode), VALIDATION_LAYER));
    }
    return ValidationMode::Off;
}

bool countsAsFailure(Severity severity, vk::DebugUtilsMessageTypeFlagsEXT types) noexcept {
    if (severity == Severity::eError) {
        return true;
    }
    const vk::DebugUtilsMessageTypeFlagsEXT checked = Type::eValidation | Type::ePerformance;
    return severity == Severity::eWarning &&
           (types & checked) != vk::DebugUtilsMessageTypeFlagsEXT{};
}

LogLevel debugMessageLevel(Severity severity, bool counted) noexcept {
    switch (severity) {
        case Severity::eError: return LogLevel::Error;
        case Severity::eWarning: return counted ? LogLevel::Warn : LogLevel::Info;
        case Severity::eInfo: return LogLevel::Debug;
        default: return LogLevel::Trace;
    }
}

std::string formatDebugMessage(Severity severity, vk::DebugUtilsMessageTypeFlagsEXT types,
                               std::string_view id_name, std::string_view message) {
    std::string type_names;
    const auto add = [&type_names](std::string_view name) {
        type_names += type_names.empty() ? "" : "|";
        type_names += name;
    };
    if (types & Type::eGeneral) {
        add("general");
    }
    if (types & Type::eValidation) {
        add("validation");
    }
    if (types & Type::ePerformance) {
        add("performance");
    }
    if (types & Type::eDeviceAddressBinding) {
        add("device-address-binding");
    }
    if (type_names.empty()) {
        add("other");
    }
    std::string_view severity_name = "verbose";
    if (severity == Severity::eError) {
        severity_name = "error";
    } else if (severity == Severity::eWarning) {
        severity_name = "warning";
    } else if (severity == Severity::eInfo) {
        severity_name = "info";
    }
    if (id_name.empty()) {
        return std::format("Vulkan {} {}: {}", type_names, severity_name, message);
    }
    return std::format("Vulkan {} {} [{}]: {}", type_names, severity_name, id_name, message);
}

Result<std::unique_ptr<InstanceCore>> InstanceCore::create(const GpuRegistryOptions& options) {
    DVV_TRY(const ValidationRequest request,
            resolveValidation(options, getEnvironment("DVV_VK_VALIDATION"),
                              getEnvironment("DVV_VK_VALIDATION_REQUIRED"), DEBUG_BUILD));
    DVV_TRY(const PFN_vkGetInstanceProcAddr get_instance_proc_addr, vulkanLoader());

    auto core = std::make_unique<InstanceCore>();
    core->dispatch_.init(get_instance_proc_addr);
    const auto& dispatch = core->dispatch_;

    // vkEnumerateInstanceVersion exists from loader 1.1 on.
    if (dispatch.vkEnumerateInstanceVersion == nullptr) {
        return makeError(Errc::Unsupported, "the Vulkan loader is 1.0; 1.3 or newer is required");
    }
    const auto loader_version = vk::enumerateInstanceVersion(dispatch);
    DVV_TRY_VOID(vkCheck(loader_version.result, "vkEnumerateInstanceVersion"));
    core->loader_version_ = loader_version.value;
    if (loader_version.value < VK_API_VERSION_1_3) {
        return makeError(Errc::Unsupported,
                         std::format("the Vulkan loader is {}; 1.3 or newer is required",
                                     formatVulkanVersion(loader_version.value)));
    }

    const auto layers = vk::enumerateInstanceLayerProperties(dispatch);
    DVV_TRY_VOID(vkCheck(layers.result, "vkEnumerateInstanceLayerProperties"));
    const auto layer =
        std::ranges::find_if(layers.value, [](const vk::LayerProperties& properties) {
            return std::string_view(properties.layerName) == VALIDATION_LAYER;
        });
    // An older layer (e.g. a system's 1.3) does not know Vulkan 1.4 and reports false errors.
    const bool layer_installed =
        layer != layers.value.end() && layer->specVersion >= VK_API_VERSION_1_4;
    if (layer != layers.value.end() && !layer_installed) {
        DVV_LOG_INFO("{} {} is too old for Vulkan 1.4 and is not used", VALIDATION_LAYER,
                     formatVulkanVersion(layer->specVersion));
    }
    DVV_TRY(core->validation_, effectiveValidation(request, layer_installed));
    if (core->validation_ != request.mode) {
        DVV_LOG(request.explicitlyRequested ? LogLevel::Warn : LogLevel::Info,
                "Vulkan validation '{}' was requested, but {} 1.4 or newer is not installed; "
                "validation is off",
                validationModeName(request.mode), VALIDATION_LAYER);
    }

    const auto extensions = vk::enumerateInstanceExtensionProperties(nullptr, dispatch);
    DVV_TRY_VOID(vkCheck(extensions.result, "vkEnumerateInstanceExtensionProperties"));
    core->debug_utils_ =
        std::ranges::any_of(extensions.value, [](const vk::ExtensionProperties& extension) {
            return std::string_view(extension.extensionName) == vk::EXTDebugUtilsExtensionName;
        });
    const std::span<const char* const> window_extensions = windowSystemExtensions(options);
    if (options.windowSystem) {
        DVV_TRY_VOID(requireInstanceExtensions(window_extensions, extensions.value,
                                               platform::windowSystemName(*options.windowSystem)));
    }
    core->window_system_ = options.windowSystem;

    const std::scoped_lock lock(vkBootstrapMutex());
    auto built = buildInstance(get_instance_proc_addr, options, core.get(), window_extensions,
                               core->debug_utils_, core->validation_);
    // A layer whose manifest is listed but whose library does not load fails vkCreateInstance;
    // unless validation is required, the instance is built once more without it.
    if (!built &&
        retryWithoutValidation(built.error() == vkb::InstanceError::failed_create_instance,
                               built.vk_result(), core->validation_, request.required)) {
        DVV_LOG_WARN("{} could not be loaded ({}); validation is off", VALIDATION_LAYER,
                     vk::to_string(static_cast<vk::Result>(built.vk_result())));
        core->validation_ = ValidationMode::Off;
        // The failed attempt's loader errors (the layer library did not load, reported at ERROR
        // through the instance-creation messenger) were logged and are not validation findings.
        core->validation_messages_.store(0, std::memory_order_relaxed);
        built = buildInstance(get_instance_proc_addr, options, core.get(), window_extensions,
                              core->debug_utils_, core->validation_);
    }
    if (!built) {
        // When vk-bootstrap fails after vkCreateInstance (only when creating the messenger fails)
        // it leaks the instance; that is practically unreachable here because debug_utils_ was
        // checked first.
        return vkBootstrapError(built, "creating the Vulkan instance");
    }
    core->instance_ = vk::Instance(built->instance);
    core->messenger_ = vk::DebugUtilsMessengerEXT(built->debug_messenger);
    core->dispatch_.init(core->instance_);
    const std::optional<platform::WindowSystem> window_system = core->window_system_;
    DVV_LOG_INFO("Vulkan instance created: loader {}, API 1.4, validation {}, {}",
                 formatVulkanVersion(core->loader_version_), validationModeName(core->validation_),
                 window_system
                     ? std::format("window system {}", platform::windowSystemName(*window_system))
                     : std::string("headless"));
    return core;
}

InstanceCore::~InstanceCore() {
    if (messenger_) {
        instance_.destroyDebugUtilsMessengerEXT(messenger_, nullptr, dispatch_);
    }
    if (instance_) {
        instance_.destroy(nullptr, dispatch_);
    }
}

void InstanceCore::onMessage(Severity severity, vk::DebugUtilsMessageTypeFlagsEXT types,
                             const vk::DebugUtilsMessengerCallbackDataEXT& data) noexcept {
    const bool counted = countsAsFailure(severity, types);
    if (counted) {
        validation_messages_.fetch_add(1, std::memory_order_relaxed);
    }
    const LogLevel level = debugMessageLevel(severity, counted);
    auto& logger = Logger::instance();
    if (!logger.enabled(level)) {
        return;
    }
    try {
        logger.write(level,
                     formatDebugMessage(severity, types,
                                        data.pMessageIdName != nullptr ? data.pMessageIdName : "",
                                        data.pMessage != nullptr ? data.pMessage : ""));
    } catch (...) {
        // The callback runs inside the Vulkan driver; nothing may propagate into it.
        std::fputs("dvv: a Vulkan debug message could not be logged\n", stderr);
    }
}

}  // namespace dvv::gpu::detail
