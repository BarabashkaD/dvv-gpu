#pragma once

#include <filesystem>
#include <memory>
#include <source_location>
#include <string_view>

#include "dvv/base/error.h"
#include "dvv/base/lifetime.h"
#include "dvv/gpu/device_id.h"
#include "dvv/gpu/gpu_fwd.h"
#include "dvv/gpu/gpu_info.h"

namespace dvv::gpu {

namespace detail {
class DeviceCore;
class PipelineCache;
class Submitter;
struct Access;
}  // namespace detail

// Options for GpuInstance::createContext().
struct ContextOptions {
    // The on-disk pipeline cache, e.g. defaultPipelineCacheFile(); empty: in memory
    // only. A file written for another device or driver build, or one whose header or data is
    // damaged (dvv checks a checksum), is ignored with a log line. The cache is saved there by
    // savePipelineCache() and when the context closes (only when it changed). Each save replaces
    // the whole file, so contexts or processes sharing one file are last-writer-wins.
    std::filesystem::path pipelineCacheFile{};
    // Enables VK_KHR_swapchain. Errc::Internal when the GPU instance has no window system
    // (GpuInstanceOptions::windowSystem); Errc::Unsupported when the device lacks the extension
    // (GpuInfo::optional.swapchain).
    bool presentation = false;
};

// <user cache directory of app>/vulkan-pipelines-<device UUID>.bin (dvv::paths::userCacheDir):
// one file per device, since a cache only fits the device and driver build that wrote it.
[[nodiscard]] Result<std::filesystem::path> defaultPipelineCacheFile(std::string_view app,
                                                                     const GpuInfo& device);

// One Vulkan device and everything that works on it: the device with the required
// features, one universal queue (graphics, compute and transfer; access serialised by a mutex),
// the VMA allocator, the timeline semaphore, the retire queue, and the submit and waiter threads.
// Components reach the GPU only through a context. Created by GpuInstance::createContext(); keeps
// its GPU instance alive. Thread-safe: every member may be called from any thread.
//
// Blocking work runs on the thread that releases the last reference: the context's destructor
// writes the pipeline-cache file and waits until the device is idle, and the destructors of
// resources and pipelines free retired memory. A pending submission holds the context, so that
// thread is often the executor the submission resumes on; release the last context reference off
// latency-sensitive executors.
class ExecutionContext final : public Managed<ExecutionContext>,
                               public std::enable_shared_from_this<ExecutionContext> {
public:
    ExecutionContext(CreationKey<GpuInstance> key, std::shared_ptr<GpuInstance> gpu_instance,
                     std::shared_ptr<detail::DeviceCore> core,
                     std::unique_ptr<detail::PipelineCache> pipeline_cache,
                     std::unique_ptr<detail::Submitter> submitter) noexcept;
    // Every pending submission holds the context, so none is queued or in flight here. Stops the
    // (idle) submit and waiter threads, waits until the device is idle and frees what was retired
    // for submitted work, then saves the pipeline cache when it changed (a failure is logged).
    // Resources and pipelines that are still alive keep the device until they are destroyed.
    // On a lost context (lost(), also when the idle wait reports the loss) the pipeline cache is
    // not saved (logged at Info) and a DeviceLost from the idle wait is ignored; destruction never
    // hangs.
    ~ExecutionContext();

    [[nodiscard]] const std::shared_ptr<GpuInstance>& gpuInstance() const noexcept {
        return gpu_instance_;
    }
    [[nodiscard]] DeviceId deviceId() const noexcept;
    // Device generation; FIRST_GENERATION until device-lost recovery exists.
    [[nodiscard]] DeviceGeneration generation() const noexcept;
    [[nodiscard]] const GpuInfo& deviceInfo() const noexcept;
    // Whether VK_KHR_swapchain is enabled (ContextOptions::presentation).
    [[nodiscard]] bool presentation() const noexcept;
    // True once the device is lost: a Vulkan call of this context returned VK_ERROR_DEVICE_LOST
    // (submit, timeline wait or counter query, device-idle wait, or a presenter's acquire, fence,
    // present or queue-idle call) or testing::injectDeviceLost() was called. Never reset; a new
    // context is the only way on. Presenter losses reach the application through the presenter's
    // onError. On a lost context createBuffer, createImage and createComputePipeline
    // (context_vulkan.h) and createPresenter (presenter_vulkan.h) return Errc::DeviceLost ("the
    // device was lost" or "the device was lost (injected)"), and submit() completes with
    // DeviceLost without recording (the record function is released on resume_on as usual).
    [[nodiscard]] bool lost() const noexcept;
    // Writes the pipeline cache to ContextOptions::pipelineCacheFile; nothing without a file.
    // Errc::Io when the file cannot be written.
    [[nodiscard]] Result<void> savePipelineCache();

    // Waits until the device is idle, then frees everything retired for submissions already on
    // the queue. On Windows, do not call it on a window's thread while a presenter of this context
    // lives (see presenter.h); destroy the presenter first.
    [[nodiscard]] Result<void> drain();

private:
    friend struct detail::Access;

    // The key of an object this context creates. The creators are the free functions of
    // context_vulkan.h and presenter_vulkan.h; they reach this through detail::Access.
    template <class T>
    [[nodiscard]] CreationKey<ExecutionContext> issueKey(std::source_location where) const {
        return CreationKey<ExecutionContext>(
            LifetimeRegistry::instance().issue<T>(objectId(), where));
    }

    std::shared_ptr<GpuInstance> gpu_instance_;
    std::shared_ptr<detail::DeviceCore> core_;
    std::unique_ptr<detail::PipelineCache> pipeline_cache_;
    std::unique_ptr<detail::Submitter> submitter_;
};

}  // namespace dvv::gpu
