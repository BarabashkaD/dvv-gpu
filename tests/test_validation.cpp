#include <doctest/doctest.h>

#include <optional>
#include <string>

#include "dvv/base/error.h"
#include "dvv/base/log.h"
#include "dvv/gpu/gpu_info.h"
#include "dvv/gpu/gpu_registry.h"
#include "instance.h"

using dvv::Errc;
using dvv::LogLevel;
using dvv::gpu::GpuRegistryOptions;
using dvv::gpu::ValidationMode;
using dvv::gpu::detail::countsAsFailure;
using dvv::gpu::detail::debugMessageLevel;
using dvv::gpu::detail::effectiveValidation;
using dvv::gpu::detail::formatDebugMessage;
using dvv::gpu::detail::parseValidationMode;
using dvv::gpu::detail::resolveValidation;
using dvv::gpu::detail::retryWithoutValidation;
using dvv::gpu::detail::ValidationRequest;
using Severity = vk::DebugUtilsMessageSeverityFlagBitsEXT;
using Type = vk::DebugUtilsMessageTypeFlagBitsEXT;
using Types = vk::DebugUtilsMessageTypeFlagsEXT;

namespace {

const std::optional<std::string> UNSET;

}  // namespace

TEST_CASE("validation modes parse from off, on and sync") {
    CHECK(parseValidationMode("off").value() == ValidationMode::Off);
    CHECK(parseValidationMode("on").value() == ValidationMode::On);
    CHECK(parseValidationMode("sync").value() == ValidationMode::Sync);
    for (const char* text : {"", "ON", "1", "sync "}) {
        const auto mode = parseValidationMode(text);
        REQUIRE_FALSE(mode.has_value());
        CHECK(mode.error().code == Errc::Format);
    }
    CHECK(dvv::gpu::validationModeName(ValidationMode::Sync) == "sync");
}

TEST_CASE("validation defaults to on in Debug builds and off in Release builds") {
    const auto debug = resolveValidation({}, UNSET, UNSET, true).value();
    CHECK(debug.mode == ValidationMode::On);
    CHECK_FALSE(debug.required);
    CHECK_FALSE(debug.explicitlyRequested);
    CHECK(resolveValidation({}, UNSET, UNSET, false).value().mode == ValidationMode::Off);
    // An empty variable counts as unset.
    CHECK(resolveValidation({}, std::string(), std::string(), false).value().mode ==
          ValidationMode::Off);
}

TEST_CASE("the environment sets validation unless the options do") {
    const auto from_environment = resolveValidation({}, "sync", "1", false).value();
    CHECK(from_environment.mode == ValidationMode::Sync);
    CHECK(from_environment.required);
    CHECK(from_environment.explicitlyRequested);
    CHECK_FALSE(resolveValidation({}, "off", "0", true).value().required);

    const GpuRegistryOptions options{.validation = ValidationMode::Off,
                                     .validationRequired = false};
    const auto from_options = resolveValidation(options, "sync", "1", true).value();
    CHECK(from_options.mode == ValidationMode::Off);
    CHECK_FALSE(from_options.required);
    CHECK(from_options.explicitlyRequested);
}

TEST_CASE("bad validation variables are format errors") {
    const auto mode = resolveValidation({}, "full", UNSET, true);
    REQUIRE_FALSE(mode.has_value());
    CHECK(mode.error().code == Errc::Format);
    CHECK(mode.error().message == "DVV_VK_VALIDATION must be off, on or sync, got 'full'");

    const auto required = resolveValidation({}, UNSET, "yes", true);
    REQUIRE_FALSE(required.has_value());
    CHECK(required.error().code == Errc::Format);
    CHECK(required.error().message == "DVV_VK_VALIDATION_REQUIRED must be 0 or 1, got 'yes'");

    // Variables that the options override are not read at all.
    CHECK(resolveValidation({.validation = ValidationMode::On, .validationRequired = true}, "full",
                            "yes", true)
              .has_value());
}

TEST_CASE("a missing validation layer turns validation off unless it is required") {
    const ValidationRequest sync{.mode = ValidationMode::Sync};
    CHECK(effectiveValidation(sync, true).value() == ValidationMode::Sync);
    CHECK(effectiveValidation(sync, false).value() == ValidationMode::Off);

    const ValidationRequest required{.mode = ValidationMode::On, .required = true};
    CHECK(effectiveValidation(required, true).value() == ValidationMode::On);
    const auto missing = effectiveValidation(required, false);
    REQUIRE_FALSE(missing.has_value());
    CHECK(missing.error().code == Errc::Unsupported);
    CHECK(missing.error().message ==
          "Vulkan validation 'on' is required, but VK_LAYER_KHRONOS_validation 1.4 or newer is not "
          "installed");

    // Requiring validation while switching it off asks for nothing.
    const ValidationRequest off{.mode = ValidationMode::Off, .required = true};
    CHECK(effectiveValidation(off, false).value() == ValidationMode::Off);
}

TEST_CASE("errors and validation or performance warnings count as failures") {
    const Types general = Type::eGeneral;
    const Types validation = Type::eValidation;
    const Types performance = Type::ePerformance;

    CHECK(countsAsFailure(Severity::eError, general));
    CHECK(countsAsFailure(Severity::eError, validation));
    CHECK(countsAsFailure(Severity::eWarning, validation));
    CHECK(countsAsFailure(Severity::eWarning, performance));
    CHECK(countsAsFailure(Severity::eWarning, general | validation));

    CHECK_FALSE(countsAsFailure(Severity::eWarning, general));
    CHECK_FALSE(countsAsFailure(Severity::eInfo, validation));
    CHECK_FALSE(countsAsFailure(Severity::eVerbose, performance));
}

TEST_CASE("debug messages are logged at their severity, uncounted ones one level lower") {
    CHECK(debugMessageLevel(Severity::eError, true) == LogLevel::Error);
    CHECK(debugMessageLevel(Severity::eWarning, true) == LogLevel::Warn);
    CHECK(debugMessageLevel(Severity::eWarning, false) == LogLevel::Info);
    CHECK(debugMessageLevel(Severity::eInfo, false) == LogLevel::Debug);
    CHECK(debugMessageLevel(Severity::eVerbose, false) == LogLevel::Trace);
}

TEST_CASE("debug messages name their types, severity and id") {
    CHECK(formatDebugMessage(Severity::eError, Type::eValidation, "VUID-x", "bad call") ==
          "Vulkan validation error [VUID-x]: bad call");
    CHECK(formatDebugMessage(Severity::eWarning, Type::eGeneral | Type::ePerformance, "", "slow") ==
          "Vulkan general|performance warning: slow");
    CHECK(formatDebugMessage(Severity::eInfo, Types{}, "", "loader") ==
          "Vulkan other info: loader");
    CHECK(formatDebugMessage(Severity::eVerbose, Type::eGeneral, "", "") ==
          "Vulkan general verbose: ");
}

TEST_CASE("Vulkan versions format as major.minor.patch") {
    CHECK(dvv::gpu::formatVulkanVersion(VK_MAKE_API_VERSION(0, 1, 4, 318)) == "1.4.318");
    CHECK(dvv::gpu::formatVulkanVersion(VK_API_VERSION_1_3) == "1.3.0");
}

TEST_CASE("the messenger callback counts by the policy and survives missing strings") {
    dvv::gpu::detail::InstanceCore core;
    const vk::DebugUtilsMessengerCallbackDataEXT data{};  // pMessageIdName and pMessage are null
    core.onMessage(Severity::eWarning, Type::eGeneral, data);
    CHECK(core.validationMessageCount() == 0);
    core.onMessage(Severity::eError, Type::eGeneral, data);
    core.onMessage(Severity::eWarning, Type::eValidation, data);
    CHECK(core.validationMessageCount() == 2);
}

TEST_CASE("a validation layer that fails to load is dropped unless validation is required") {
    constexpr bool CREATE_INSTANCE = true;  // vkCreateInstance itself failed
    CHECK(retryWithoutValidation(CREATE_INSTANCE, VK_ERROR_LAYER_NOT_PRESENT, ValidationMode::On,
                                 false));
    CHECK(retryWithoutValidation(CREATE_INSTANCE, VK_ERROR_LAYER_NOT_PRESENT, ValidationMode::Sync,
                                 false));
    CHECK(retryWithoutValidation(CREATE_INSTANCE, VK_ERROR_OUT_OF_HOST_MEMORY, ValidationMode::On,
                                 false));
    CHECK_FALSE(retryWithoutValidation(CREATE_INSTANCE, VK_ERROR_OUT_OF_HOST_MEMORY,
                                       ValidationMode::On, true));
    CHECK_FALSE(retryWithoutValidation(CREATE_INSTANCE, VK_ERROR_LAYER_NOT_PRESENT,
                                       ValidationMode::Sync, true));
    CHECK_FALSE(retryWithoutValidation(CREATE_INSTANCE, VK_ERROR_LAYER_NOT_PRESENT,
                                       ValidationMode::Off, false));
    CHECK_FALSE(retryWithoutValidation(CREATE_INSTANCE, VK_ERROR_INCOMPATIBLE_DRIVER,
                                       ValidationMode::On, false));
    // A later step failed (creating the messenger): the instance exists, so it is not built again.
    CHECK_FALSE(retryWithoutValidation(!CREATE_INSTANCE, VK_ERROR_OUT_OF_HOST_MEMORY,
                                       ValidationMode::On, false));
}
