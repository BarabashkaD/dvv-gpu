#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "dvv/gpu/device_id.h"

namespace dvv::gpu {

enum class DeviceType : std::uint8_t { Other, IntegratedGpu, DiscreteGpu, VirtualGpu, Cpu };

// "discrete GPU", "integrated GPU", "virtual GPU", "CPU" or "other".
[[nodiscard]] std::string_view deviceTypeName(DeviceType type) noexcept;

// Used when present (spec §5.1); a device without them is still usable.
struct OptionalFeatures {
    bool shaderInt16 = false;
    bool shaderFloat16 = false;
    bool hostImageCopy = false;
    bool memoryBudget = false;  // VK_EXT_memory_budget
    bool swapchain = false;     // VK_KHR_swapchain (device extension; ContextOptions::presentation)
};

// One physical device as the GpuRegistry found it.
struct GpuInfo {
    DeviceId id{0};
    std::string name{};
    DeviceType type = DeviceType::Other;
    std::uint32_t vendorId = 0;
    std::uint32_t productId = 0;      // VkPhysicalDeviceProperties::deviceID
    std::uint32_t apiVersion = 0;     // VK_MAKE_API_VERSION encoding
    std::uint32_t driverVersion = 0;  // vendor-specific encoding
    std::string driverName{};         // VkPhysicalDeviceDriverProperties
    std::string driverInfo{};         // usually the driver's own version text
    std::array<std::uint8_t, 16> deviceUuid{};
    std::uint64_t deviceLocalMemoryBytes = 0;  // sum of the device-local heaps
    // Requirements of spec §5.1 the device does not meet ("Vulkan 1.4", feature names, ...); empty
    // when the device is usable.
    std::vector<std::string> missingRequirements{};
    OptionalFeatures optional{};

    [[nodiscard]] bool usable() const noexcept { return missingRequirements.empty(); }
};

// "major.minor.patch" of a Vulkan version number (VK_MAKE_API_VERSION encoding; the variant bits
// are ignored).
[[nodiscard]] std::string formatVulkanVersion(std::uint32_t version);

// Lower case hexadecimal in the 8-4-4-4-12 grouping.
[[nodiscard]] std::string formatUuid(const std::array<std::uint8_t, 16>& uuid);

// toJson(const GpuInfo&) is in gpu_info_json.h, so that users of this header do not compile the
// JSON library.

// Best first: devices that meet the requirements, then discrete > integrated > virtual > CPU >
// other, then more device-local memory; ties keep the enumeration order.
[[nodiscard]] std::vector<DeviceId> rankDevices(std::span<const GpuInfo> devices);

}  // namespace dvv::gpu
