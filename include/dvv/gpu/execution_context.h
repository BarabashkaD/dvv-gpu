#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string_view>

#include "dvv/base/error.h"
#include "dvv/base/executor.h"
#include "dvv/base/lifetime.h"
#include "dvv/gpu/compute_pipeline.h"
#include "dvv/gpu/device_id.h"
#include "dvv/gpu/embedded_shader.h"
#include "dvv/gpu/gpu_info.h"
#include "dvv/gpu/presenter.h"
#include "dvv/gpu/resource_pool.h"
#include "dvv/gpu/resources.h"
#include "dvv/gpu/submission.h"

namespace dvv::gpu {

class GpuRegistry;

namespace detail {
class DeviceCore;
class PipelineCache;
class Submitter;
struct Access;
}  // namespace detail

// Settings for GpuRegistry::createContext().
struct ContextOptions {
    // The on-disk pipeline cache (spec §5.2), e.g. defaultPipelineCacheFile(); empty: in memory
    // only. A file written for another device or driver build, or one whose header or data is
    // damaged (dvv checks a checksum), is ignored with a log line. The cache is saved there by
    // savePipelineCache() and when the context closes (only when it changed). Each save replaces
    // the whole file, so contexts or processes sharing one file are last-writer-wins.
    std::filesystem::path pipelineCacheFile{};
    // Enables VK_KHR_swapchain. Errc::Internal when the registry has no window system
    // (GpuRegistryOptions::windowSystem); Errc::Unsupported when the device lacks the extension
    // (GpuInfo::optional.swapchain).
    bool presentation = false;
};

// <user cache directory of app>/vulkan-pipelines-<device UUID>.bin (dvv::paths::userCacheDir):
// one file per device, since a cache only fits the device and driver build that wrote it.
[[nodiscard]] Result<std::filesystem::path> defaultPipelineCacheFile(std::string_view app,
                                                                     const GpuInfo& device);

// Counters of the retire queue (spec §5.3, §9.4).
struct RetireStats {
    std::size_t retiring = 0;   // handles waiting until the GPU no longer uses them
    std::uint64_t retired = 0;  // handles ever queued
    std::uint64_t freed = 0;    // handles ever destroyed
};

// One Vulkan device and everything that works on it (spec §5.2): the device with the required
// features, one universal queue (graphics, compute and transfer; access serialised by a mutex),
// the VMA allocator, the timeline semaphore, the retire queue, and the submit and waiter threads.
// Components reach the GPU only through a context. Created by GpuRegistry::createContext(); keeps
// its registry alive. Thread-safe: every member may be called from any thread.
//
// Blocking work runs on the thread that releases the last reference: the context's destructor
// writes the pipeline-cache file and waits until the device is idle, and the destructors of
// resources and pipelines free retired memory. A pending submission holds the context, so that
// thread is often the executor the submission resumes on; release the last context reference off
// latency-sensitive executors.
class ExecutionContext final : public Managed<ExecutionContext>,
                               public std::enable_shared_from_this<ExecutionContext> {
public:
    ExecutionContext(CreationKey<GpuRegistry> key, std::shared_ptr<GpuRegistry> registry,
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

    [[nodiscard]] const std::shared_ptr<GpuRegistry>& registry() const noexcept {
        return registry_;
    }
    [[nodiscard]] DeviceId deviceId() const noexcept;
    // Device generation (spec §5.2); FIRST_GENERATION until device-lost recovery exists.
    [[nodiscard]] DeviceGeneration generation() const noexcept;
    [[nodiscard]] const GpuInfo& deviceInfo() const noexcept;
    // Whether VK_KHR_swapchain is enabled (ContextOptions::presentation).
    [[nodiscard]] bool presentation() const noexcept;
    // True once the device is lost: a Vulkan call of this context returned VK_ERROR_DEVICE_LOST
    // (submit, timeline wait or counter query, device-idle wait, or a presenter's acquire, fence,
    // present or queue-idle call) or testing::injectDeviceLost() was called. Never reset; a new
    // context is the only way on. Presenter losses reach the application through the presenter's
    // onError (rawreel: Session::reportDeviceLost).
    // On a lost context createBuffer, createImage, createResourcePool, createComputePipeline (both
    // overloads) and createPresenter return Errc::DeviceLost ("the device was lost" or "the device
    // was lost (injected)"), and submit() completes with DeviceLost without recording (the record
    // function is released on resume_on as usual).
    [[nodiscard]] bool lost() const noexcept;
    // A presenter for one native window (spec §4, §10.2; see presenter.h). Errc::Internal on a
    // context without ContextOptions::presentation, for an empty frame provider and for a window of
    // another window system (or none); Errc::Unsupported when the device's queue cannot present to
    // the window or no swapchain format and usage fit (TRANSFER_DST usage, a UNORM or SRGB format
    // with BLIT_DST);
    // surface errors as converted by dvv-gpu (a window system the loader lacks: Unsupported); a
    // presenter thread that cannot start: Internal (OutOfMemory for std::bad_alloc).
    [[nodiscard]] Result<std::unique_ptr<Presenter>> createPresenter(PresenterOptions options);

    // A buffer (spec §5.3). Errc::Internal for a zero size; Vulkan errors as converted by dvv-gpu
    // (e.g. OutOfMemory). Frees retired memory the GPU has finished with first.
    [[nodiscard]] Result<std::shared_ptr<GpuBuffer>> createBuffer(BufferDesc desc);
    // A 2D image (spec §5.3). Errc::Internal when width, height, format or usage is missing;
    // Errc::Unsupported when the device cannot create it (format, usage or size). STORAGE usage
    // needs a format with VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT (else Errc::Unsupported "image
    // '<name>': device <id> has no storage images of <format>"). STORAGE or SAMPLED usage also
    // creates the image's view (GpuImage::view(): 2D, the whole image, the image's format),
    // retired with the image.
    [[nodiscard]] Result<std::shared_ptr<GpuImage>> createImage(ImageDesc desc);
    // A pool that reuses the buffers it hands out (spec §5.3); it keeps this context alive.
    [[nodiscard]] Result<std::shared_ptr<ResourcePool>> createResourcePool();
    // A compute pipeline (spec §5.4) from an embedded shader (entry point main) with a
    // push-constant range of `push_constant_size` bytes (0: none), made through the context's
    // pipeline cache. The range must cover the shader's push-constant block, which is checked when
    // the shader has its reflection JSON. Errc::Internal when the shader is not SPIR-V, the size is
    // not a multiple of 4 or smaller than the block; Errc::Format when the reflection JSON cannot
    // be read; Errc::Unsupported when the size exceeds the device's limit; Vulkan errors as
    // converted. The same as createComputePipeline(shader, push_constant_size, ComputeBindings{}).
    [[nodiscard]] Result<std::shared_ptr<ComputePipeline>> createComputePipeline(
        const EmbeddedShader& shader, std::uint32_t push_constant_size);
    // As above, with the storage images the pipeline binds (set 0, bindings 0..storageImages-1,
    // pushed per dispatch). Errc::Internal when bindings.storageImages exceeds the device's
    // maxPushDescriptors, or when the shader's reflection declares descriptor bindings other than
    // set 0, bindings 0..storageImages-1 (checked when the shader has its reflection JSON).
    [[nodiscard]] Result<std::shared_ptr<ComputePipeline>> createComputePipeline(
        const EmbeddedShader& shader, std::uint32_t push_constant_size, ComputeBindings bindings);
    // The device's optimal-tiling features of a format (VkFormatProperties3), e.g.
    // VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT.
    [[nodiscard]] VkFormatFeatureFlags2 formatFeatures(VkFormat format) const;
    // Writes the pipeline cache to ContextOptions::pipelineCacheFile; nothing without a file.
    // Errc::Io when the file cannot be written.
    [[nodiscard]] Result<void> savePipelineCache();

    // One submission (spec §4): co_await ctx->submit(executor, record). The submit thread calls
    // record with a CommandRecorder, then submits the commands with a signal of the next timeline
    // value; the awaiting coroutine resumes on `resume_on` once the GPU has finished them, with
    // that value or the error that stopped the submission. A record function that throws fails the
    // submission (Errc::Internal); its timeline value is signalled all the same, so later
    // submissions are not held up. The record function, and everything it captures, is released on
    // `resume_on` before the coroutine resumes: capturing the resources a submission uses keeps
    // them alive until the GPU is done with them. An empty record function submits only the final
    // host-visibility barrier. The record function must not block or wait for another submission:
    // the submit thread records one submission at a time. `resume_on` must outlive the operation.
    [[nodiscard]] SubmitOperation submit(IExecutor& resume_on, RecordFunction record);

    // Waits until the device is idle, then frees everything retired for submissions already on
    // the queue. On Windows, do not call it on a window's thread while a presenter of this context
    // lives (see presenter.h); destroy the presenter first.
    [[nodiscard]] Result<void> drain();
    [[nodiscard]] RetireStats retireStats() const noexcept;

private:
    friend struct detail::Access;
    friend class SubmitOperation;

    std::shared_ptr<GpuRegistry> registry_;
    std::shared_ptr<detail::DeviceCore> core_;
    std::unique_ptr<detail::PipelineCache> pipeline_cache_;
    std::unique_ptr<detail::Submitter> submitter_;
};

}  // namespace dvv::gpu
