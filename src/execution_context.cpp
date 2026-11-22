#include "dvv/gpu/execution_context.h"

#include <exception>
#include <format>
#include <memory>
#include <string_view>
#include <utility>

#include "access.h"
#include "device_core.h"
#include "dvv/base/log.h"
#include "dvv/base/paths.h"
#include "dvv/gpu/gpu_instance.h"
#include "pipeline_cache.h"
#include "submitter.h"

namespace dvv::gpu {

static_assert(ManagedType<ExecutionContext>);

ExecutionContext::ExecutionContext(CreationKey<GpuInstance> key,
                                   std::shared_ptr<GpuInstance> gpu_instance,
                                   std::shared_ptr<detail::DeviceCore> core,
                                   std::unique_ptr<detail::PipelineCache> pipeline_cache,
                                   std::unique_ptr<detail::Submitter> submitter) noexcept
    : Managed(std::move(key)),
      gpu_instance_(std::move(gpu_instance)),
      core_(std::move(core)),
      pipeline_cache_(std::move(pipeline_cache)),
      submitter_(std::move(submitter)) {}

ExecutionContext::~ExecutionContext() {
    submitter_->stop();  // its waits are bounded, so this returns on a lost device too
    // The idle wait comes before the cache decision: it is the last call that can see a loss no
    // other call has seen (e.g. a driver reset while the context was idle).
    const auto drained = core_->drain();
    if (core_->lost()) {
        // The driver's cache data cannot be trusted after a loss.
        DVV_LOG_INFO("the pipeline cache of device {} is not saved: the device was lost",
                     core_->id());
    } else {
        try {
            if (const auto saved = pipeline_cache_->saveIfChanged(); !saved) {
                DVV_LOG_ERROR("device {}: {}", core_->id(), saved.error().describe());
            }
        } catch (const std::exception& error) {
            DVV_LOG_ERROR("saving the pipeline cache of device {} failed: {}", core_->id(),
                          error.what());
        }
    }
    if (!drained && drained.error().code != Errc::DeviceLost) {
        DVV_LOG_ERROR("closing the context of device {} failed: {}", core_->id(),
                      drained.error().describe());
    }
}

DeviceId ExecutionContext::deviceId() const noexcept {
    return core_->id();
}

DeviceGeneration ExecutionContext::generation() const noexcept {
    return core_->generation();
}

const GpuInfo& ExecutionContext::deviceInfo() const noexcept {
    return core_->info();
}

bool ExecutionContext::presentation() const noexcept {
    return core_->presentation();
}

bool ExecutionContext::lost() const noexcept {
    return core_->lost();
}

Result<void> ExecutionContext::drain() {
    return core_->drain();
}

Result<void> ExecutionContext::savePipelineCache() {
    return pipeline_cache_->save();
}

Result<std::filesystem::path> defaultPipelineCacheFile(std::string_view app,
                                                       const GpuInfo& device) {
    DVV_TRY(std::filesystem::path directory, paths::userCacheDir(app));
    return directory / std::format("vulkan-pipelines-{}.bin", formatUuid(device.deviceUuid));
}

SubmitOperation::SubmitOperation(std::shared_ptr<ExecutionContext> context, IExecutor& resume_on,
                                 RecordFunction record) noexcept
    : context_(std::move(context)), resume_on_(&resume_on), record_(std::move(record)) {}

void SubmitOperation::await_suspend(std::coroutine_handle<> continuation) {
    auto submission = std::make_unique<detail::Submission>();
    submission->record = std::move(record_);
    submission->resumeOn = resume_on_;
    submission->continuation = continuation;
    submission->result = &result_;
    // The last use of this object: the coroutine may resume (on another thread) at once.
    detail::Access::submitter(*context_).enqueue(std::move(submission));
}

}  // namespace dvv::gpu
