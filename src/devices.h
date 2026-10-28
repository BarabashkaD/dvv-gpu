#pragma once

// Physical devices: what they support, which of the required features (spec §5.1) they lack, and
// the feature set a device is created with.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "dvv/base/error.h"
#include "dvv/gpu/gpu_info.h"
#include "vk.h"

namespace dvv::gpu::detail {

using FeatureChain =
    vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,
                       vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features,
                       vk::PhysicalDeviceVulkan14Features>;

// The required features, in one list for checking and enabling: calls visit(name, flag) for each,
// where flag is the vk::Bool32 member of the chain.
template <class Chain, class Visit>
void forEachRequiredFeature(Chain& chain, Visit&& visit) {
    auto& core = chain.template get<vk::PhysicalDeviceFeatures2>().features;
    auto& v11 = chain.template get<vk::PhysicalDeviceVulkan11Features>();
    auto& v12 = chain.template get<vk::PhysicalDeviceVulkan12Features>();
    auto& v13 = chain.template get<vk::PhysicalDeviceVulkan13Features>();
    auto& v14 = chain.template get<vk::PhysicalDeviceVulkan14Features>();
    visit("shaderInt64", core.shaderInt64);  // buffer device addresses in shaders
    // Storage images of R8_UINT, R16_UINT, ... written by shaders declared without a format.
    visit("shaderStorageImageExtendedFormats", core.shaderStorageImageExtendedFormats);
    visit("shaderStorageImageWriteWithoutFormat", core.shaderStorageImageWriteWithoutFormat);
    // ... and read without a format (ToDisplay reads R8_UINT/R16_UINT planes, design M3 D5).
    visit("shaderStorageImageReadWithoutFormat", core.shaderStorageImageReadWithoutFormat);
    visit("storageBuffer16BitAccess", v11.storageBuffer16BitAccess);
    visit("uniformAndStorageBuffer16BitAccess", v11.uniformAndStorageBuffer16BitAccess);
    visit("timelineSemaphore", v12.timelineSemaphore);
    visit("bufferDeviceAddress", v12.bufferDeviceAddress);
    visit("descriptorIndexing", v12.descriptorIndexing);
    visit("runtimeDescriptorArray", v12.runtimeDescriptorArray);
    visit("descriptorBindingPartiallyBound", v12.descriptorBindingPartiallyBound);
    visit("descriptorBindingVariableDescriptorCount", v12.descriptorBindingVariableDescriptorCount);
    visit("shaderSampledImageArrayNonUniformIndexing",
          v12.shaderSampledImageArrayNonUniformIndexing);
    visit("descriptorBindingSampledImageUpdateAfterBind",
          v12.descriptorBindingSampledImageUpdateAfterBind);
    visit("synchronization2", v13.synchronization2);
    visit("dynamicRendering", v13.dynamicRendering);
    visit("maintenance5", v14.maintenance5);
    visit("maintenance6", v14.maintenance6);
    visit("pushDescriptor", v14.pushDescriptor);  // storage images bound per dispatch
}

// What a physical device offers, as far as the GpuRegistry needs it.
struct DeviceSupport {
    std::uint32_t apiVersion = 0;
    std::optional<std::uint32_t> universalQueueFamily = std::nullopt;  // graphics + compute
    FeatureChain features{};  // structures above the device's API version stay all false
    OptionalFeatures optional{};
};

// The requirements of spec §5.1 the device misses: "Vulkan 1.4 (the device has 1.3.280)", "a queue
// family with graphics and compute", then the missing feature names in list order.
[[nodiscard]] std::vector<std::string> missingRequirements(const DeviceSupport& support);

// The features a context's device is created with: every required one, plus the optional ones the
// device supports.
[[nodiscard]] FeatureChain enabledFeatures(const OptionalFeatures& optional);

// The device extensions a context's device is created with: VK_EXT_memory_budget when the device
// has it, VK_KHR_swapchain for presentation. Errc::Unsupported "device <id> (<name>) does not
// support VK_KHR_swapchain, which presentation needs" when presentation is asked for without it.
[[nodiscard]] Result<std::vector<const char*>> deviceExtensions(const GpuInfo& info,
                                                                bool presentation);

// A physical device of the instance, indexed by DeviceId.
struct PhysicalDevice {
    vk::PhysicalDevice handle{};
    std::optional<std::uint32_t> universalQueueFamily = std::nullopt;
};

struct Enumeration {
    std::vector<GpuInfo> infos{};
    std::vector<PhysicalDevice> devices{};  // same order as infos
};

[[nodiscard]] Result<Enumeration> enumerateDevices(
    vk::Instance instance, const vk::detail::DispatchLoaderDynamic& dispatch);

}  // namespace dvv::gpu::detail
