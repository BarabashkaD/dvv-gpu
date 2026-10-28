#include "device_core.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <format>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>

#include "devices.h"
#include "dvv/base/log.h"
#include "instance.h"
#include "vk_result.h"

namespace dvv::gpu::detail {

Result<std::shared_ptr<DeviceCore>> DeviceCore::create(std::shared_ptr<const GpuRegistry> registry,
                                                       const InstanceCore& instance,
                                                       const PhysicalDevice& physical,
                                                       const GpuInfo& info, bool presentation) {
    if (!info.usable() || !physical.universalQueueFamily) {
        return makeError(
            Errc::Unsupported,
            std::format("device {} ({}) does not meet the requirements", info.id, info.name));
    }
    DVV_TRY(const std::vector<const char*> extensions, deviceExtensions(info, presentation));
    // Every member is set step by step; the destructor releases whatever exists when a step fails.
    auto core = std::make_shared<DeviceCore>();
    core->registry_ = std::move(registry);
    core->id_ = info.id;
    core->info_ = info;
    core->presentation_ = presentation;
    core->debug_utils_ = instance.debugUtils();
    core->physical_ = physical.handle;
    // The device is Vulkan 1.4, so the push-descriptor properties may be chained.
    vk::StructureChain<vk::PhysicalDeviceProperties2, vk::PhysicalDevicePushDescriptorProperties>
        properties;
    core->physical_.getProperties2(&properties.get<vk::PhysicalDeviceProperties2>(),
                                   instance.dispatch());
    core->limits_ = properties.get<vk::PhysicalDeviceProperties2>().properties.limits;
    core->max_push_descriptors_ =
        properties.get<vk::PhysicalDevicePushDescriptorProperties>().maxPushDescriptors;
    core->queue_family_ = *physical.universalQueueFamily;

    const float priority = 1.0F;
    const vk::DeviceQueueCreateInfo queue_info{
        .queueFamilyIndex = core->queue_family_, .queueCount = 1, .pQueuePriorities = &priority};
    const FeatureChain features = enabledFeatures(info.optional);
    const vk::DeviceCreateInfo device_info{
        .pNext = &features.get<vk::PhysicalDeviceFeatures2>(),
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
        .enabledExtensionCount = static_cast<std::uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data()};
    const auto device = core->physical_.createDevice(device_info, nullptr, instance.dispatch());
    DVV_TRY_VOID(vkCheck(device.result, "vkCreateDevice"));
    core->device_ = device.value;
    core->dispatch_.init(
        static_cast<VkInstance>(instance.instance()), instance.dispatch().vkGetInstanceProcAddr,
        static_cast<VkDevice>(core->device_), instance.dispatch().vkGetDeviceProcAddr);
    core->queue_ = core->device_.getQueue(core->queue_family_, 0, core->dispatch_);

    VmaVulkanFunctions functions{};
    functions.vkGetInstanceProcAddr = core->dispatch_.vkGetInstanceProcAddr;
    functions.vkGetDeviceProcAddr = core->dispatch_.vkGetDeviceProcAddr;
    VmaAllocatorCreateInfo allocator_info{};
    allocator_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    if (info.optional.memoryBudget) {
        allocator_info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
    }
    allocator_info.physicalDevice = core->physical_;
    allocator_info.device = core->device_;
    allocator_info.instance = instance.instance();
    allocator_info.pVulkanFunctions = &functions;
    allocator_info.vulkanApiVersion = VK_API_VERSION_1_4;
    DVV_TRY_VOID(
        vkCheck(vmaCreateAllocator(&allocator_info, &core->allocator_), "vmaCreateAllocator"));

    vk::SemaphoreTypeCreateInfo timeline_type{.semaphoreType = vk::SemaphoreType::eTimeline,
                                              .initialValue = 0};
    const auto timeline =
        core->device_.createSemaphore({.pNext = &timeline_type}, nullptr, core->dispatch_);
    DVV_TRY_VOID(vkCheck(timeline.result, "vkCreateSemaphore"));
    core->timeline_ = timeline.value;

    const std::string label =
        std::format("dvv device {} generation {}", info.id, core->generation_);
    core->setDebugName(core->device_, label);
    core->setDebugName(core->queue_, label + " queue");
    core->setDebugName(core->timeline_, label + " timeline");
    DVV_LOG_INFO("Vulkan device created: {} ({}), Vulkan {}{}", info.name, info.id,
                 formatVulkanVersion(info.apiVersion), presentation ? ", presentation" : "");
    return core;
}

DeviceCore::~DeviceCore() {
    if (!device_) {
        return;
    }
    {
        const auto lock = lockQueue();
        const vk::Result idle = device_.waitIdle(dispatch_);
        if (idle == vk::Result::eErrorDeviceLost) {
            // Expected on a lost device: nothing runs on it any more.
            DVV_LOG_INFO("vkDeviceWaitIdle while destroying device {}: the device was lost", id_);
        } else if (idle != vk::Result::eSuccess) {
            DVV_LOG_ERROR("vkDeviceWaitIdle failed while destroying device {}: {}", id_,
                          vk::to_string(idle));
        }
    }
    freeRetired(UINT64_MAX);  // also everything a lost device left behind
    if (timeline_) {
        device_.destroySemaphore(timeline_, nullptr, dispatch_);
    }
    if (allocator_ != nullptr) {
        vmaDestroyAllocator(allocator_);
    }
    device_.destroy(nullptr, dispatch_);
}

std::uint64_t DeviceCore::reserveSubmitValue() noexcept {
    return last_reserved_.fetch_add(1, std::memory_order_acq_rel) + 1;
}

void DeviceCore::setSubmitted(std::uint64_t value,
                              const std::unique_lock<std::mutex>& queue_lock) noexcept {
    assert(queue_lock.owns_lock() && queue_lock.mutex() == &queue_mutex_);
    static_cast<void>(queue_lock);  // used only by the assertion in Release builds
    last_submitted_.store(value, std::memory_order_release);
}

Result<std::uint64_t> DeviceCore::completedValue() {
    const auto value = device_.getSemaphoreCounterValue(timeline_, dispatch_);
    if (value.result == vk::Result::eErrorDeviceLost) {
        markLost(LostCause::Driver, "vkGetSemaphoreCounterValue");
    }
    DVV_TRY_VOID(vkCheck(value.result, "vkGetSemaphoreCounterValue"));
    return value.value;
}

bool DeviceCore::markLost(LostCause cause, std::string_view where) noexcept {
    LostCause expected = LostCause::None;
    if (!lost_.compare_exchange_strong(expected, cause, std::memory_order_acq_rel)) {
        return false;
    }
    try {
        DVV_LOG_WARN("device {} lost ({})", id_,
                     cause == LostCause::Injected ? std::string_view("injected") : where);
    } catch (...) {
        // A failed log line must not escape a noexcept path; say so without allocating.
        std::fputs("dvv: a device-lost warning could not be logged\n", stderr);
    }
    return true;
}

Error DeviceCore::lostError() const {
    return makeError(Errc::DeviceLost, lost_.load(std::memory_order_acquire) == LostCause::Injected
                                           ? "the device was lost (injected)"
                                           : "the device was lost")
        .error();
}

void DeviceCore::retire(RetiredHandle handle, VmaAllocation allocation) noexcept {
    const std::scoped_lock lock(retire_mutex_);
    // Tagged with the latest reserved submission: any work that used the handle was submitted, or
    // is being recorded, no later.
    retired_.push_back(
        {.timelineValue = lastReservedValue(), .handle = handle, .allocation = allocation});
    ++retired_total_;
}

std::size_t DeviceCore::collect() noexcept {
    const auto value = device_.getSemaphoreCounterValue(timeline_, dispatch_);
    if (value.result == vk::Result::eErrorDeviceLost) {
        // Freed by drain() or ~DeviceCore, once nothing can use them any more.
        markLost(LostCause::Driver, "vkGetSemaphoreCounterValue");
        return 0;
    }
    if (value.result != vk::Result::eSuccess) {
        DVV_LOG_ERROR("vkGetSemaphoreCounterValue failed on device {}: {}", id_,
                      vk::to_string(value.result));
        return 0;
    }
    return freeRetired(value.value);
}

Result<void> DeviceCore::drain() {
    std::uint64_t submitted = 0;
    vk::Result idle = vk::Result::eSuccess;
    {
        const auto lock = lockQueue();
        submitted = lastSubmittedValue();  // no submission can follow while the lock is held
        idle = device_.waitIdle(dispatch_);
    }
    // No logger call under the queue lock.
    if (idle == vk::Result::eErrorDeviceLost) {
        markLost(LostCause::Driver, "vkDeviceWaitIdle");
        // Nothing submitted runs any more; what is reserved but not submitted stays (below).
        freeRetired(submitted);
        return std::unexpected(lostError());
    }
    DVV_TRY_VOID(vkCheck(idle, "vkDeviceWaitIdle"));
    // Objects tagged with a value that is reserved but not yet submitted may still be used by
    // the commands being recorded for it; they stay.
    freeRetired(submitted);
    return {};
}

RetireStats DeviceCore::retireStats() const noexcept {
    const std::scoped_lock lock(retire_mutex_);
    return {.retiring = retired_.size(), .retired = retired_total_, .freed = freed_total_};
}

std::size_t DeviceCore::freeRetired(std::uint64_t reached) noexcept {
    std::vector<RetiredObject> ready;
    {
        const std::scoped_lock lock(retire_mutex_);
        const auto done = std::ranges::stable_partition(
            retired_,
            [reached](const RetiredObject& object) { return object.timelineValue > reached; });
        ready.assign(std::make_move_iterator(done.begin()), std::make_move_iterator(done.end()));
        retired_.erase(done.begin(), done.end());
        freed_total_ += ready.size();
    }
    for (const RetiredObject& object : ready) {
        destroy(object);
    }
    return ready.size();
}

void DeviceCore::destroy(const RetiredObject& object) const noexcept {
    if (const auto* buffer = std::get_if<vk::Buffer>(&object.handle)) {
        vmaDestroyBuffer(allocator_, static_cast<VkBuffer>(*buffer), object.allocation);
    } else if (const auto* pipeline = std::get_if<vk::Pipeline>(&object.handle)) {
        device_.destroyPipeline(*pipeline, nullptr, dispatch_);
    } else if (const auto* layout = std::get_if<vk::PipelineLayout>(&object.handle)) {
        device_.destroyPipelineLayout(*layout, nullptr, dispatch_);
    } else if (const auto* image = std::get_if<vk::Image>(&object.handle)) {
        vmaDestroyImage(allocator_, static_cast<VkImage>(*image), object.allocation);
    } else if (const auto* view = std::get_if<vk::ImageView>(&object.handle)) {
        device_.destroyImageView(*view, nullptr, dispatch_);
    } else if (const auto* set_layout = std::get_if<vk::DescriptorSetLayout>(&object.handle)) {
        device_.destroyDescriptorSetLayout(*set_layout, nullptr, dispatch_);
    }
}

VkFormatFeatureFlags2 DeviceCore::formatFeatures(VkFormat format) const {
    {
        const std::scoped_lock lock(format_mutex_);
        const auto cached = std::ranges::find(format_features_, format,
                                              &std::pair<VkFormat, VkFormatFeatureFlags2>::first);
        if (cached != format_features_.end()) {
            return cached->second;
        }
    }
    // Queried outside the lock; a thread that raced with this one stores nothing new.
    const auto chain = physical_.getFormatProperties2<vk::FormatProperties2, vk::FormatProperties3>(
        static_cast<vk::Format>(format), dispatch_);
    const auto features = static_cast<VkFormatFeatureFlags2>(
        chain.get<vk::FormatProperties3>().optimalTilingFeatures);
    const std::scoped_lock lock(format_mutex_);
    if (std::ranges::find(format_features_, format,
                          &std::pair<VkFormat, VkFormatFeatureFlags2>::first) ==
        format_features_.end()) {
        format_features_.emplace_back(format, features);
    }
    return features;
}

void DeviceCore::setObjectName(vk::ObjectType type, std::uint64_t handle,
                               const std::string& name) const noexcept {
    if (!debug_utils_ || name.empty()) {
        return;
    }
    const vk::DebugUtilsObjectNameInfoEXT info{
        .objectType = type, .objectHandle = handle, .pObjectName = name.c_str()};
    // A missing debug name is not worth failing for.
    static_cast<void>(device_.setDebugUtilsObjectNameEXT(info, dispatch_));
}

}  // namespace dvv::gpu::detail
