#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "dvv/base/error.h"
#include "dvv/base/lifetime.h"
#include "dvv/gpu/device_id.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/gpu_info.h"
#include "dvv/platform/native_window.h"

namespace dvv::gpu {

namespace detail {
class InstanceCore;
struct Access;
}  // namespace detail

// Vulkan validation (VK_LAYER_KHRONOS_validation). Sync adds synchronization validation.
enum class ValidationMode : std::uint8_t { Off, On, Sync };

[[nodiscard]] std::string_view validationModeName(ValidationMode mode) noexcept;

struct GpuRegistryOptions {
    std::string applicationName = "dvv";
    // Unset: the environment variable DVV_VK_VALIDATION (off, on or sync); without it On in Debug
    // builds and Off in Release builds. A requested layer that is not installed turns validation
    // off with a log line, unless validation is required.
    std::optional<ValidationMode> validation = std::nullopt;
    // Unset: DVV_VK_VALIDATION_REQUIRED (1 or 0, default 0). When true, a requested validation
    // layer that is not installed is an error (Errc::Unsupported); used by CI.
    std::optional<bool> validationRequired = std::nullopt;
    // Set: the instance also enables platform::surfaceInstanceExtensions(*windowSystem), so
    // contexts can present (ContextOptions::presentation); Errc::Unsupported when the loader lacks
    // them. Unset: headless.
    std::optional<platform::WindowSystem> windowSystem = std::nullopt;
};

// The process's Vulkan instance (spec §5.2): loads the Vulkan loader at run time, creates an
// instance (Vulkan 1.4 API on a loader of at least 1.3; headless unless options.windowSystem is
// set), routes debug messages to the dvv Logger, and records and ranks the physical devices. Its
// own creation authority; thread-safe.
class GpuRegistry final : public Managed<GpuRegistry>,
                          public std::enable_shared_from_this<GpuRegistry> {
public:
    // Errc::Unsupported "Vulkan loader not found" without a loader; Errc::Unsupported when the
    // loader is older than 1.3, lacks a surface extension of options.windowSystem, or a required
    // validation layer is missing; Errc::Format for a bad DVV_VK_VALIDATION or
    // DVV_VK_VALIDATION_REQUIRED value. A required layer that is listed but whose library does not
    // load fails as the loader reports it: Errc::Unsupported on Linux, Errc::OutOfMemory on Windows
    // (spec §11).
    [[nodiscard]] static Result<std::shared_ptr<GpuRegistry>> create(
        const GpuRegistryOptions& options = {});

    GpuRegistry(CreationKey<GpuRegistry> key, std::unique_ptr<detail::InstanceCore> instance,
                std::vector<GpuInfo> devices, std::vector<DeviceId> ranked) noexcept;
    ~GpuRegistry();

    // The validation actually enabled (Off when the layer was requested but is not installed).
    [[nodiscard]] ValidationMode validation() const noexcept;
    // Debug messages that count as failures: errors of any type, and validation or performance
    // warnings. Every message is logged; tests require this count to stay zero.
    [[nodiscard]] std::uint64_t validationMessageCount() const noexcept;
    // Version of the Vulkan loader (VK_MAKE_API_VERSION encoding).
    [[nodiscard]] std::uint32_t loaderVersion() const noexcept;
    // The window system the instance was created for; nullopt for a headless registry.
    [[nodiscard]] std::optional<platform::WindowSystem> windowSystem() const noexcept;

    // Every physical device in enumeration order: devices()[id.index()].
    [[nodiscard]] std::span<const GpuInfo> devices() const noexcept { return devices_; }
    // Every device id, best first (rankDevices()).
    [[nodiscard]] std::span<const DeviceId> rankedDevices() const noexcept { return ranked_; }
    // The best device that meets the requirements; nullopt when none does.
    [[nodiscard]] std::optional<DeviceId> bestDevice() const noexcept;

    // A context on one device (spec §5.2). Errc::Internal for an id that does not exist or for
    // ContextOptions::presentation on a headless registry, Errc::Unsupported for a device that
    // misses requirements (or VK_KHR_swapchain for presentation); Vulkan errors as converted by
    // dvv-gpu (e.g. OutOfMemory).
    [[nodiscard]] Result<std::shared_ptr<ExecutionContext>> createContext(
        DeviceId id, const ContextOptions& options = {});

private:
    friend struct detail::Access;

    std::unique_ptr<detail::InstanceCore> instance_;
    std::vector<GpuInfo> devices_;
    std::vector<DeviceId> ranked_;
};

}  // namespace dvv::gpu
