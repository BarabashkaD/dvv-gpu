#pragma once

// Access to the private parts of dvv-gpu's public classes, for its own sources and white-box tests.

#include <atomic>
#include <cstddef>
#include <memory>
#include <source_location>
#include <utility>

#include "device_core.h"
#include "dvv/base/executor.h"
#include "dvv/gpu/command_recorder.h"
#include "dvv/gpu/compute_pipeline.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/gpu_instance.h"
#include "dvv/gpu/image.h"
#include "dvv/gpu/resources.h"
#include "dvv/gpu/submission.h"
#include "image_state.h"
#include "instance.h"
#include "pipeline_cache.h"
#include "presenter_core.h"
#include "submitter.h"

namespace dvv::gpu::detail {

struct Access {
    [[nodiscard]] static InstanceCore& instance(const GpuInstance& gpu_instance) noexcept {
        return *gpu_instance.instance_;
    }
    [[nodiscard]] static DeviceCore& core(const ExecutionContext& context) noexcept {
        return *context.core_;
    }
    [[nodiscard]] static const PresenterCore& core(const Presenter& presenter) noexcept {
        return *presenter.core_;
    }
    [[nodiscard]] static DeviceCore& core(const GpuBuffer& buffer) noexcept {
        return *buffer.core_;
    }
    [[nodiscard]] static VmaAllocation_T* allocation(const GpuBuffer& buffer) noexcept {
        return buffer.allocation_;
    }
    [[nodiscard]] static DeviceCore& core(const ComputePipeline& pipeline) noexcept {
        return *pipeline.core_;
    }
    [[nodiscard]] static PipelineCache& pipelineCache(const ExecutionContext& context) noexcept {
        return *context.pipeline_cache_;
    }
    [[nodiscard]] static Submitter& submitter(const ExecutionContext& context) noexcept {
        return *context.submitter_;
    }
    // The submitter's last-resort signal; tests call it while the submit thread is idle.
    static void hostSignal(Submitter& submitter, Submission& submission) noexcept {
        submitter.hostSignal(submission);
    }
    // The command buffers the submitter has allocated (busy or free); tests call it while the
    // submit thread is idle, after the submissions they made have completed.
    [[nodiscard]] static std::size_t commandBufferCount(const Submitter& submitter) noexcept {
        return submitter.busy_buffers_.size() + submitter.free_buffers_.size();
    }
    // The presenter's blit into a swapchain image (CommandRecorder::presentBlit).
    static void presentBlit(CommandRecorder& recorder, GpuImage& source, VkImage swapchain_image,
                            VkExtent2D swapchain_extent) {
        recorder.presentBlit(source, swapchain_image, swapchain_extent);
    }
    // The context's device as an owner, for the objects the free creators make (they keep the
    // device alive).
    [[nodiscard]] static const std::shared_ptr<DeviceCore>& sharedCore(
        const ExecutionContext& context) noexcept {
        return context.core_;
    }
    // The creation key for an object of `context` (the free creators).
    template <class T>
    [[nodiscard]] static CreationKey<ExecutionContext> issueKey(
        const ExecutionContext& context,
        std::source_location where = std::source_location::current()) {
        return context.issueKey<T>(where);
    }
    // One submission of `context` (context_vulkan.h's submit()).
    [[nodiscard]] static SubmitOperation submit(ExecutionContext& context, IExecutor& resume_on,
                                                RecordFunction record) {
        return SubmitOperation(context.shared_from_this(), resume_on, std::move(record));
    }

    [[nodiscard]] static DeviceCore& core(const GpuImage& image) noexcept {
        return *image.state_->core;
    }
    static void setLayout(GpuImage& image, VkImageLayout layout) noexcept {
        image.state_->layout.store(layout, std::memory_order_release);
    }
    static void setSelf(GpuImage& image, const std::shared_ptr<GpuImage>& self) noexcept {
        image.state_->self = self;
    }
    // The command recorder keeps the images whose layout a submission changes. Never null: every
    // GpuImage comes from createImage (the only key issuer), which sets the weak self before the
    // image escapes, and the caller holds the image while it records.
    [[nodiscard]] static std::shared_ptr<GpuImage> retain(GpuImage& image) noexcept {
        return image.state_->self.lock();
    }
    [[nodiscard]] static const ImageDesc& imageDesc(const GpuImage& image) noexcept {
        return image.state_->desc;
    }
    [[nodiscard]] static VkImage imageHandle(const GpuImage& image) noexcept {
        return image.state_->image;
    }
    [[nodiscard]] static VkImageView imageView(const GpuImage& image) noexcept {
        return image.state_->view;
    }
    [[nodiscard]] static VkImageLayout imageLayout(const GpuImage& image) noexcept {
        return image.state_->layout.load(std::memory_order_acquire);
    }
};

}  // namespace dvv::gpu::detail
