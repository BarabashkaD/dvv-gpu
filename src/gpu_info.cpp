#include "dvv/gpu/gpu_info.h"

#include <algorithm>
#include <cstddef>
#include <format>

#include <vulkan/vulkan_core.h>

#include "dvv/gpu/gpu_info_json.h"

namespace dvv::gpu {

namespace {

// Smaller is better.
int typeRank(DeviceType type) noexcept {
    switch (type) {
        case DeviceType::DiscreteGpu: return 0;
        case DeviceType::IntegratedGpu: return 1;
        case DeviceType::VirtualGpu: return 2;
        case DeviceType::Cpu: return 3;
        case DeviceType::Other: break;
    }
    return 4;
}

}  // namespace

std::string_view deviceTypeName(DeviceType type) noexcept {
    switch (type) {
        case DeviceType::IntegratedGpu: return "integrated GPU";
        case DeviceType::DiscreteGpu: return "discrete GPU";
        case DeviceType::VirtualGpu: return "virtual GPU";
        case DeviceType::Cpu: return "CPU";
        case DeviceType::Other: break;
    }
    return "other";
}

std::string formatVulkanVersion(std::uint32_t version) {
    return std::format("{}.{}.{}", VK_API_VERSION_MAJOR(version), VK_API_VERSION_MINOR(version),
                       VK_API_VERSION_PATCH(version));
}

std::string formatUuid(const std::array<std::uint8_t, 16>& uuid) {
    std::string text;
    for (std::size_t i = 0; i < uuid.size(); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) {
            text += '-';
        }
        text += std::format("{:02x}", uuid[i]);
    }
    return text;
}

json::Value toJson(const GpuInfo& info) {
    return json::Value{
        {"id", info.id.index()},
        {"name", info.name},
        {"type", deviceTypeName(info.type)},
        {"vendorId", info.vendorId},
        {"productId", info.productId},
        {"apiVersion", formatVulkanVersion(info.apiVersion)},
        {"driverVersion", info.driverVersion},
        {"driverName", info.driverName},
        {"driverInfo", info.driverInfo},
        {"deviceUuid", formatUuid(info.deviceUuid)},
        {"deviceLocalMemoryBytes", info.deviceLocalMemoryBytes},
        {"usable", info.usable()},
        {"missingRequirements", info.missingRequirements},
        {"optionalFeatures",
         {{"shaderInt16", info.optional.shaderInt16},
          {"shaderFloat16", info.optional.shaderFloat16},
          {"hostImageCopy", info.optional.hostImageCopy},
          {"memoryBudget", info.optional.memoryBudget},
          {"swapchain", info.optional.swapchain}}},
    };
}

std::vector<DeviceId> rankDevices(std::span<const GpuInfo> devices) {
    std::vector<const GpuInfo*> order;
    order.reserve(devices.size());
    for (const GpuInfo& device : devices) {
        order.push_back(&device);
    }
    std::ranges::stable_sort(order, [](const GpuInfo* left, const GpuInfo* right) {
        if (left->usable() != right->usable()) {
            return left->usable();
        }
        if (typeRank(left->type) != typeRank(right->type)) {
            return typeRank(left->type) < typeRank(right->type);
        }
        return left->deviceLocalMemoryBytes > right->deviceLocalMemoryBytes;
    });
    std::vector<DeviceId> ranked;
    ranked.reserve(order.size());
    for (const GpuInfo* device : order) {
        ranked.push_back(device->id);
    }
    return ranked;
}

}  // namespace dvv::gpu
