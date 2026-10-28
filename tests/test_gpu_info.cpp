#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include "devices.h"
#include "dvv/base/error.h"
#include "dvv/base/json.h"
#include "dvv/gpu/gpu_info.h"
#include "dvv/gpu/gpu_info_json.h"

using dvv::gpu::DeviceId;
using dvv::gpu::DeviceType;
using dvv::gpu::GpuInfo;
using dvv::gpu::detail::deviceExtensions;
using dvv::gpu::detail::DeviceSupport;
using dvv::gpu::detail::enabledFeatures;
using dvv::gpu::detail::forEachRequiredFeature;
using dvv::gpu::detail::missingRequirements;

namespace {

GpuInfo device(std::uint32_t index, DeviceType type, std::uint64_t memory, bool usable = true) {
    GpuInfo info{.id = DeviceId(index), .type = type, .deviceLocalMemoryBytes = memory};
    if (!usable) {
        info.missingRequirements = {"maintenance5"};
    }
    return info;
}

// A Vulkan 1.4 device with a universal queue and every required feature.
DeviceSupport complete() {
    DeviceSupport support{.apiVersion = VK_API_VERSION_1_4, .universalQueueFamily = 0};
    forEachRequiredFeature(support.features,
                           [](std::string_view /*name*/, vk::Bool32& flag) { flag = VK_TRUE; });
    return support;
}

}  // namespace

TEST_CASE("devices rank usable first, then by type, then by device-local memory") {
    const std::vector<GpuInfo> devices{
        device(0, DeviceType::Cpu, 64),          device(1, DeviceType::DiscreteGpu, 8, false),
        device(2, DeviceType::IntegratedGpu, 2), device(3, DeviceType::DiscreteGpu, 4),
        device(4, DeviceType::DiscreteGpu, 12),  device(5, DeviceType::VirtualGpu, 1),
        device(6, DeviceType::Other, 1),         device(7, DeviceType::DiscreteGpu, 12),
    };
    const std::vector<DeviceId> expected{DeviceId(4), DeviceId(7), DeviceId(3), DeviceId(2),
                                         DeviceId(5), DeviceId(0), DeviceId(6), DeviceId(1)};
    CHECK(dvv::gpu::rankDevices(devices) == expected);
    CHECK(dvv::gpu::rankDevices({}).empty());
}

TEST_CASE("a Vulkan 1.4 device with every required feature misses nothing") {
    CHECK(missingRequirements(complete()).empty());
}

TEST_CASE("missing requirements are named in order") {
    DeviceSupport support = complete();
    support.apiVersion = VK_MAKE_API_VERSION(0, 1, 3, 280);
    support.universalQueueFamily.reset();
    support.features.get<vk::PhysicalDeviceVulkan12Features>().bufferDeviceAddress = VK_FALSE;
    support.features.get<vk::PhysicalDeviceVulkan14Features>().maintenance6 = VK_FALSE;
    const std::vector<std::string> expected{"Vulkan 1.4 (the device has 1.3.280)",
                                            "a queue family with graphics and compute",
                                            "bufferDeviceAddress", "maintenance6"};
    CHECK(missingRequirements(support) == expected);
}

TEST_CASE("the required features include images in shaders") {
    dvv::gpu::detail::FeatureChain chain{};
    std::vector<std::string> names;
    forEachRequiredFeature(
        chain, [&](std::string_view name, vk::Bool32& /*flag*/) { names.emplace_back(name); });
    const auto has = [&names](std::string_view name) {
        return std::ranges::find(names, name) != names.end();
    };
    CHECK(has("pushDescriptor"));
    CHECK(has("shaderStorageImageExtendedFormats"));
    CHECK(has("shaderStorageImageWriteWithoutFormat"));
}

TEST_CASE("shaderStorageImageReadWithoutFormat is required") {
    DeviceSupport support = complete();
    std::vector<std::string> names;
    forEachRequiredFeature(support.features, [&](std::string_view name, vk::Bool32& /*flag*/) {
        names.emplace_back(name);
    });
    const auto write = std::ranges::find(names, "shaderStorageImageWriteWithoutFormat");
    REQUIRE(write != names.end());
    REQUIRE(std::next(write) != names.end());
    CHECK(*std::next(write) == "shaderStorageImageReadWithoutFormat");

    support.features.get<vk::PhysicalDeviceFeatures2>()
        .features.shaderStorageImageReadWithoutFormat = VK_FALSE;
    CHECK(missingRequirements(support) ==
          std::vector<std::string>{"shaderStorageImageReadWithoutFormat"});
}

TEST_CASE("a device is created with every required feature and the optional ones it has") {
    const auto features = enabledFeatures({.shaderInt16 = true, .hostImageCopy = true});
    std::vector<std::string> disabled;
    forEachRequiredFeature(features, [&disabled](std::string_view name, vk::Bool32 flag) {
        if (flag != VK_TRUE) {
            disabled.emplace_back(name);
        }
    });
    CHECK(disabled.empty());
    CHECK(features.get<vk::PhysicalDeviceFeatures2>().features.shaderInt16 == VK_TRUE);
    CHECK(features.get<vk::PhysicalDeviceVulkan12Features>().shaderFloat16 == VK_FALSE);
    CHECK(features.get<vk::PhysicalDeviceVulkan14Features>().hostImageCopy == VK_TRUE);
    // The structures are chained for vkCreateDevice.
    CHECK(features.get<vk::PhysicalDeviceFeatures2>().pNext ==
          &features.get<vk::PhysicalDeviceVulkan11Features>());
}

TEST_CASE("device information formats as text and JSON") {
    CHECK(dvv::gpu::deviceTypeName(DeviceType::DiscreteGpu) == "discrete GPU");
    CHECK(dvv::gpu::deviceTypeName(DeviceType::Cpu) == "CPU");
    CHECK(dvv::gpu::formatUuid({0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x00, 0x11, 0x22,
                                0x33, 0x44, 0x55, 0x66, 0xff}) ==
          "01234567-89ab-cdef-0011-2233445566ff");

    GpuInfo info = device(2, DeviceType::Cpu, 1024, false);
    info.name = "llvmpipe";
    info.apiVersion = VK_MAKE_API_VERSION(0, 1, 4, 318);
    info.optional.memoryBudget = true;
    const dvv::json::Value json = dvv::gpu::toJson(info);
    CHECK(json.at("id") == 2);
    CHECK(json.at("name") == "llvmpipe");
    CHECK(json.at("type") == "CPU");
    CHECK(json.at("apiVersion") == "1.4.318");
    CHECK(json.at("deviceUuid") == "00000000-0000-0000-0000-000000000000");
    CHECK(json.at("deviceLocalMemoryBytes") == 1024);
    CHECK(json.at("usable") == false);
    CHECK(json.at("missingRequirements") == dvv::json::Value::array({"maintenance5"}));
    CHECK(json.at("optionalFeatures").at("memoryBudget") == true);
    CHECK(json.at("optionalFeatures").at("shaderInt16") == false);
    CHECK(json.at("optionalFeatures").at("swapchain") == false);
    info.optional.swapchain = true;
    CHECK(dvv::gpu::toJson(info).at("optionalFeatures").at("swapchain") == true);
}

TEST_CASE("presentation needs VK_KHR_swapchain") {
    GpuInfo info = device(1, DeviceType::DiscreteGpu, 8);
    info.name = "test GPU";
    const auto unsupported = deviceExtensions(info, true);
    REQUIRE_FALSE(unsupported.has_value());
    CHECK(unsupported.error().code == dvv::Errc::Unsupported);
    CHECK(unsupported.error().message.find("VK_KHR_swapchain") != std::string::npos);
    const auto headless = deviceExtensions(info, false);
    REQUIRE(headless.has_value());
    CHECK(headless->empty());

    info.optional.swapchain = true;
    info.optional.memoryBudget = true;
    const auto presenting = deviceExtensions(info, true);
    REQUIRE(presenting.has_value());
    const std::vector<std::string_view> presenting_names(presenting->begin(), presenting->end());
    CHECK(std::ranges::find(presenting_names, "VK_KHR_swapchain") != presenting_names.end());
    CHECK(std::ranges::find(presenting_names, "VK_EXT_memory_budget") != presenting_names.end());
    CHECK(presenting_names.size() == 2);
    const auto budget_only = deviceExtensions(info, false);
    REQUIRE(budget_only.has_value());
    CHECK(std::vector<std::string_view>(budget_only->begin(), budget_only->end()) ==
          std::vector<std::string_view>{"VK_EXT_memory_budget"});
}
