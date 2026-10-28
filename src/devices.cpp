#include "devices.h"

#include <algorithm>
#include <format>
#include <string_view>
#include <utility>

#include "dvv/gpu/gpu_registry.h"
#include "vk_result.h"

namespace dvv::gpu::detail {

namespace {

using Dispatch = vk::detail::DispatchLoaderDynamic;

DeviceType toDeviceType(vk::PhysicalDeviceType type) noexcept {
    switch (type) {
        case vk::PhysicalDeviceType::eIntegratedGpu: return DeviceType::IntegratedGpu;
        case vk::PhysicalDeviceType::eDiscreteGpu: return DeviceType::DiscreteGpu;
        case vk::PhysicalDeviceType::eVirtualGpu: return DeviceType::VirtualGpu;
        case vk::PhysicalDeviceType::eCpu: return DeviceType::Cpu;
        default: return DeviceType::Other;
    }
}

Result<DeviceSupport> querySupport(vk::PhysicalDevice device, std::uint32_t api_version,
                                   const Dispatch& dispatch) {
    DeviceSupport support{.apiVersion = api_version};

    const auto families = device.getQueueFamilyProperties(dispatch);
    constexpr auto UNIVERSAL = vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute;
    for (std::uint32_t index = 0; index < families.size(); ++index) {
        if ((families[index].queueFlags & UNIVERSAL) == UNIVERSAL) {
            support.universalQueueFamily = index;
            break;
        }
    }

    // Chaining a structure of a newer Vulkan version than the device's is invalid usage, so those
    // stay unlinked and all false.
    auto& features = support.features;
    if (api_version < VK_API_VERSION_1_4) {
        features.unlink<vk::PhysicalDeviceVulkan14Features>();
    }
    if (api_version < VK_API_VERSION_1_3) {
        features.unlink<vk::PhysicalDeviceVulkan13Features>();
    }
    if (api_version < VK_API_VERSION_1_2) {
        features.unlink<vk::PhysicalDeviceVulkan12Features>();
        features.unlink<vk::PhysicalDeviceVulkan11Features>();
    }
    device.getFeatures2(&features.get<vk::PhysicalDeviceFeatures2>(), dispatch);
    support.optional.shaderInt16 =
        features.get<vk::PhysicalDeviceFeatures2>().features.shaderInt16 == VK_TRUE;
    support.optional.shaderFloat16 =
        features.get<vk::PhysicalDeviceVulkan12Features>().shaderFloat16 == VK_TRUE;
    support.optional.hostImageCopy =
        features.get<vk::PhysicalDeviceVulkan14Features>().hostImageCopy == VK_TRUE;

    const auto extensions = device.enumerateDeviceExtensionProperties(nullptr, dispatch);
    DVV_TRY_VOID(vkCheck(extensions.result, "vkEnumerateDeviceExtensionProperties"));
    const auto has = [&extensions](std::string_view name) {
        return std::ranges::any_of(extensions.value,
                                   [name](const vk::ExtensionProperties& extension) {
                                       return std::string_view(extension.extensionName) == name;
                                   });
    };
    support.optional.memoryBudget = has(vk::EXTMemoryBudgetExtensionName);
    // Reported for every registry, also a headless one (dvv-gpu-info).
    support.optional.swapchain = has(vk::KHRSwapchainExtensionName);
    return support;
}

}  // namespace

std::vector<std::string> missingRequirements(const DeviceSupport& support) {
    std::vector<std::string> missing;
    if (support.apiVersion < VK_API_VERSION_1_4) {
        missing.push_back(
            std::format("Vulkan 1.4 (the device has {})", formatVulkanVersion(support.apiVersion)));
    }
    if (!support.universalQueueFamily) {
        missing.emplace_back("a queue family with graphics and compute");
    }
    forEachRequiredFeature(support.features, [&missing](std::string_view name, vk::Bool32 flag) {
        if (flag != VK_TRUE) {
            missing.emplace_back(name);
        }
    });
    return missing;
}

FeatureChain enabledFeatures(const OptionalFeatures& optional) {
    FeatureChain chain;
    forEachRequiredFeature(chain,
                           [](std::string_view /*name*/, vk::Bool32& flag) { flag = VK_TRUE; });
    const auto flag = [](bool enabled) { return enabled ? VK_TRUE : VK_FALSE; };
    chain.get<vk::PhysicalDeviceFeatures2>().features.shaderInt16 = flag(optional.shaderInt16);
    chain.get<vk::PhysicalDeviceVulkan12Features>().shaderFloat16 = flag(optional.shaderFloat16);
    chain.get<vk::PhysicalDeviceVulkan14Features>().hostImageCopy = flag(optional.hostImageCopy);
    return chain;
}

Result<std::vector<const char*>> deviceExtensions(const GpuInfo& info, bool presentation) {
    std::vector<const char*> extensions;
    if (info.optional.memoryBudget) {
        extensions.push_back(vk::EXTMemoryBudgetExtensionName);
    }
    if (presentation) {
        if (!info.optional.swapchain) {
            return makeError(Errc::Unsupported,
                             std::format("device {} ({}) does not support {}, which presentation "
                                         "needs",
                                         info.id, info.name, vk::KHRSwapchainExtensionName));
        }
        extensions.push_back(vk::KHRSwapchainExtensionName);
    }
    return extensions;
}

Result<Enumeration> enumerateDevices(vk::Instance instance, const Dispatch& dispatch) {
    const auto handles = instance.enumeratePhysicalDevices(dispatch);
    DVV_TRY_VOID(vkCheck(handles.result, "vkEnumeratePhysicalDevices"));

    Enumeration enumeration;
    for (std::uint32_t index = 0; index < handles.value.size(); ++index) {
        const vk::PhysicalDevice device = handles.value[index];
        const std::uint32_t api_version = device.getProperties(dispatch).apiVersion;

        vk::StructureChain<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceIDProperties,
                           vk::PhysicalDeviceDriverProperties>
            properties;
        if (api_version < VK_API_VERSION_1_2) {
            properties.unlink<vk::PhysicalDeviceDriverProperties>();
        }
        if (api_version < VK_API_VERSION_1_1) {
            properties.unlink<vk::PhysicalDeviceIDProperties>();  // the UUID then stays zero
        }
        device.getProperties2(&properties.get<vk::PhysicalDeviceProperties2>(), dispatch);
        const auto& basic = properties.get<vk::PhysicalDeviceProperties2>().properties;
        const auto& ids = properties.get<vk::PhysicalDeviceIDProperties>();
        const auto& driver = properties.get<vk::PhysicalDeviceDriverProperties>();

        GpuInfo info{.id = DeviceId(index),
                     .name = std::string(basic.deviceName.data()),
                     .type = toDeviceType(basic.deviceType),
                     .vendorId = basic.vendorID,
                     .productId = basic.deviceID,
                     .apiVersion = api_version,
                     .driverVersion = basic.driverVersion,
                     .driverName = std::string(driver.driverName.data()),
                     .driverInfo = std::string(driver.driverInfo.data())};
        std::ranges::copy(ids.deviceUUID, info.deviceUuid.begin());
        const auto memory = device.getMemoryProperties(dispatch);
        for (std::uint32_t heap = 0; heap < memory.memoryHeapCount; ++heap) {
            if (memory.memoryHeaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
                info.deviceLocalMemoryBytes += memory.memoryHeaps[heap].size;
            }
        }

        DVV_TRY(const DeviceSupport support, querySupport(device, api_version, dispatch));
        info.missingRequirements = missingRequirements(support);
        info.optional = support.optional;
        enumeration.infos.push_back(std::move(info));
        enumeration.devices.push_back(
            {.handle = device, .universalQueueFamily = support.universalQueueFamily});
    }
    return enumeration;
}

}  // namespace dvv::gpu::detail
