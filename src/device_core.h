#pragma once

// The device behind an ExecutionContext (spec §5.2, §5.3): device, universal queue, VMA allocator,
// timeline semaphore and retire queue. Shared by the context and every resource it created, so the
// device stays alive until the last deferred destruction is done. An internal part of
// ExecutionContext, not registered separately: the managed objects that hold it are.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <vk_mem_alloc.h>

#include "dvv/base/error.h"
#include "dvv/gpu/device_id.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/gpu_info.h"
#include "dvv/gpu/gpu_registry.h"
#include "vk.h"

namespace dvv::gpu::detail {

class InstanceCore;
struct PhysicalDevice;

// What the retire queue destroys: resources with their memory, image views, pipelines and their
// layouts.
using RetiredHandle = std::variant<vk::Buffer, vk::Image, vk::ImageView, vk::Pipeline,
                                   vk::PipelineLayout, vk::DescriptorSetLayout>;

// A handle waiting until the GPU no longer uses it, with its memory (none for views, pipelines and
// layouts).
struct RetiredObject {
    std::uint64_t timelineValue = 0;  // freed once the timeline semaphore reaches this value
    RetiredHandle handle{};
    VmaAllocation allocation = nullptr;
};

// Why a device is lost (design M3 §4.1): not at all, a VK_ERROR_DEVICE_LOST from a Vulkan call, or
// testing::injectDeviceLost().
enum class LostCause : std::uint8_t { None, Driver, Injected };

class DeviceCore {
public:
    // Creates the device with the required features (and the supported optional ones), its
    // universal queue, the VMA allocator (buffer device addresses; memory budget when present) and
    // the timeline semaphore. Keeps the registry, and with it the instance, alive. `presentation`
    // also enables VK_KHR_swapchain (Errc::Unsupported when the device lacks it,
    // deviceExtensions()); the caller checks that the instance has a window system.
    [[nodiscard]] static Result<std::shared_ptr<DeviceCore>> create(
        std::shared_ptr<const GpuRegistry> registry, const InstanceCore& instance,
        const PhysicalDevice& physical, const GpuInfo& info, bool presentation = false);

    DeviceCore(const DeviceCore&) = delete;
    DeviceCore& operator=(const DeviceCore&) = delete;
    DeviceCore(DeviceCore&&) = delete;
    DeviceCore& operator=(DeviceCore&&) = delete;
    // Waits until the device is idle, frees everything retired, then destroys the timeline, the
    // allocator and the device.
    ~DeviceCore();

    [[nodiscard]] DeviceId id() const noexcept { return id_; }
    [[nodiscard]] DeviceGeneration generation() const noexcept { return generation_; }
    [[nodiscard]] const GpuInfo& info() const noexcept { return info_; }
    // VK_KHR_swapchain is enabled.
    [[nodiscard]] bool presentation() const noexcept { return presentation_; }
    [[nodiscard]] vk::PhysicalDevice physicalDevice() const noexcept { return physical_; }
    // The physical device's limits (push-constant size, compute group counts, ...).
    [[nodiscard]] const vk::PhysicalDeviceLimits& limits() const noexcept { return limits_; }
    // The most descriptors one push-descriptor set may have
    // (VkPhysicalDevicePushDescriptorProperties).
    [[nodiscard]] std::uint32_t maxPushDescriptors() const noexcept {
        return max_push_descriptors_;
    }
    [[nodiscard]] vk::Device device() const noexcept { return device_; }
    [[nodiscard]] const vk::detail::DispatchLoaderDynamic& dispatch() const noexcept {
        return dispatch_;
    }
    [[nodiscard]] VmaAllocator allocator() const noexcept { return allocator_; }
    [[nodiscard]] std::uint32_t queueFamily() const noexcept { return queue_family_; }
    [[nodiscard]] vk::Queue queue() const noexcept { return queue_; }
    [[nodiscard]] vk::Semaphore timeline() const noexcept { return timeline_; }

    // Queue access (spec §4): every vkQueue* call and vkDeviceWaitIdle happen under this lock.
    [[nodiscard]] std::unique_lock<std::mutex> lockQueue() {
        return std::unique_lock(queue_mutex_);
    }
    // Reserves the timeline value of the next submission before its commands are recorded, so that
    // everything retired from then on (also while recording) waits for that submission.
    // Contract: only the submit thread reserves values, and every reserved value is signalled,
    // in order, also when recording or submitting fails (then by a batch with the signal alone
    // or, as the last resort, from the host once every earlier signal is reached; see
    // submitter.h); otherwise the objects retired for it would wait until drain(). The submitter
    // reports each signal with setSubmitted().
    [[nodiscard]] std::uint64_t reserveSubmitValue() noexcept;
    // Records that `value` (the latest reserved one) is signalled: its signal is on the queue, or
    // the host has signalled it. The caller holds the queue lock (checked by an assertion).
    void setSubmitted(std::uint64_t value, const std::unique_lock<std::mutex>& queue_lock) noexcept;
    // The latest reserved value (0 before the first submission); retired objects wait for it.
    [[nodiscard]] std::uint64_t lastReservedValue() const noexcept {
        return last_reserved_.load(std::memory_order_acquire);
    }
    // The value of the latest submission on the queue (0 before the first one).
    [[nodiscard]] std::uint64_t lastSubmittedValue() const noexcept {
        return last_submitted_.load(std::memory_order_acquire);
    }
    // The value the timeline semaphore has reached. VK_ERROR_DEVICE_LOST marks the device lost.
    [[nodiscard]] Result<std::uint64_t> completedValue();

    // True once the device is lost; never reset.
    [[nodiscard]] bool lost() const noexcept {
        return lost_.load(std::memory_order_acquire) != LostCause::None;
    }
    // Enters the lost state; the first cause wins. Logs once (Warn): "device <id> lost (<where>)"
    // or "device <id> lost (injected)". Never throws; true for the call that set it. Any thread.
    bool markLost(LostCause cause, std::string_view where) noexcept;
    // Errc::DeviceLost "the device was lost" (Driver) or "the device was lost (injected)".
    [[nodiscard]] Error lostError() const;

    // Queues a handle and its memory for destruction once the work submitted or being recorded so
    // far is complete.
    void retire(RetiredHandle handle, VmaAllocation allocation) noexcept;
    // Frees the retired objects whose timeline value has been reached; returns how many. Under an
    // injected loss the real counter keeps working, so nothing is freed early.
    std::size_t collect() noexcept;
    // Waits until the device is idle, then frees everything retired for submissions made so far.
    // On a really lost device (VK_ERROR_DEVICE_LOST from the idle wait) it marks the device lost,
    // still frees what was submitted and returns lostError().
    [[nodiscard]] Result<void> drain();
    [[nodiscard]] RetireStats retireStats() const noexcept;

    // The optimal-tiling features of a format (VkFormatProperties3), queried once per format and
    // cached. Any thread.
    [[nodiscard]] VkFormatFeatureFlags2 formatFeatures(VkFormat format) const;

    // Names an object for debuggers and validation messages; nothing without VK_EXT_debug_utils.
    template <class Handle>
    void setDebugName(Handle handle, const std::string& name) const noexcept;

    // Use create().
    DeviceCore() = default;

private:
    std::size_t freeRetired(std::uint64_t reached) noexcept;
    void destroy(const RetiredObject& object) const noexcept;
    void setObjectName(vk::ObjectType type, std::uint64_t handle,
                       const std::string& name) const noexcept;

    std::shared_ptr<const GpuRegistry> registry_;
    DeviceId id_{0};
    DeviceGeneration generation_ = FIRST_GENERATION;
    GpuInfo info_;
    bool presentation_ = false;
    bool debug_utils_ = false;
    vk::PhysicalDevice physical_;
    vk::PhysicalDeviceLimits limits_{};
    std::uint32_t max_push_descriptors_ = 0;
    vk::Device device_;
    vk::detail::DispatchLoaderDynamic dispatch_;
    VmaAllocator allocator_ = nullptr;
    std::uint32_t queue_family_ = 0;
    vk::Queue queue_;
    vk::Semaphore timeline_;

    std::mutex queue_mutex_;
    std::atomic<std::uint64_t> last_reserved_{0};
    std::atomic<std::uint64_t> last_submitted_{0};
    std::atomic<LostCause> lost_{LostCause::None};

    mutable std::mutex retire_mutex_;
    std::vector<RetiredObject> retired_;  // not sorted: retire() may race with a submission
    std::uint64_t retired_total_ = 0;
    std::uint64_t freed_total_ = 0;

    mutable std::mutex format_mutex_;
    mutable std::vector<std::pair<VkFormat, VkFormatFeatureFlags2>> format_features_;
};

template <class Handle>
void DeviceCore::setDebugName(Handle handle, const std::string& name) const noexcept {
    using CHandle = typename Handle::CType;
    setObjectName(Handle::objectType, reinterpret_cast<std::uint64_t>(static_cast<CHandle>(handle)),
                  name);
}

}  // namespace dvv::gpu::detail
