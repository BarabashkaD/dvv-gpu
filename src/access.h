#pragma once

// Access to the private parts of dvv-gpu's public classes, for its own sources and white-box tests.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "device_core.h"
#include "dvv/base/error.h"
#include "dvv/base/executor.h"
#include "dvv/gpu/command_recorder.h"
#include "dvv/gpu/compute_pipeline.h"
#include "dvv/gpu/embedded_shader.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/gpu_instance.h"
#include "dvv/gpu/presenter.h"
#include "dvv/gpu/resources.h"
#include "dvv/gpu/submission.h"
#include "instance.h"
#include "pipeline_cache.h"
#include "submitter.h"

namespace dvv::gpu::detail {

struct Access {
    [[nodiscard]] static InstanceCore& instance(const GpuInstance& gpu_instance) noexcept {
        return *gpu_instance.instance_;
    }
    [[nodiscard]] static DeviceCore& core(const ExecutionContext& context) noexcept {
        return *context.core_;
    }
    [[nodiscard]] static DeviceCore& core(const GpuBuffer& buffer) noexcept {
        return *buffer.core_;
    }
    [[nodiscard]] static VmaAllocation_T* allocation(const GpuBuffer& buffer) noexcept {
        return buffer.allocation_;
    }
    [[nodiscard]] static DeviceCore& core(const GpuImage& image) noexcept { return *image.core_; }
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
    // The command recorder tracks the layout its commands leave an image in.
    static void setLayout(GpuImage& image, VkImageLayout layout) noexcept {
        image.layout_.store(layout, std::memory_order_release);
    }
    // The presenter's blit into a swapchain image (CommandRecorder::presentBlit).
    static void presentBlit(CommandRecorder& recorder, GpuImage& source, VkImage swapchain_image,
                            VkExtent2D swapchain_extent) {
        recorder.presentBlit(source, swapchain_image, swapchain_extent);
    }
    // The command recorder keeps the images whose layout a submission changes. 0.3.x: the public
    // enable_shared_from_this base; 1.0.0: a private weak self in the image.
    [[nodiscard]] static std::shared_ptr<GpuImage> retain(GpuImage& image) {
        return image.shared_from_this();
    }

    // context_vulkan.h's free functions reach the members they replace through these (private
    // members from 1.0.0 on), so their errors are the members'.
    [[nodiscard]] static Result<std::shared_ptr<GpuBuffer>> createBuffer(ExecutionContext& context,
                                                                         BufferDesc desc) {
        return context.createBuffer(std::move(desc));
    }
    [[nodiscard]] static Result<std::shared_ptr<GpuImage>> createImage(ExecutionContext& context,
                                                                       ImageDesc desc) {
        return context.createImage(std::move(desc));
    }
    [[nodiscard]] static Result<std::shared_ptr<ComputePipeline>> createComputePipeline(
        ExecutionContext& context, const EmbeddedShader& shader, std::uint32_t push_constant_size,
        ComputeBindings bindings) {
        return context.createComputePipeline(shader, push_constant_size, bindings);
    }
    [[nodiscard]] static VkFormatFeatureFlags2 formatFeatures(const ExecutionContext& context,
                                                              VkFormat format) {
        return context.formatFeatures(format);
    }
    [[nodiscard]] static SubmitOperation submit(ExecutionContext& context, IExecutor& resume_on,
                                                RecordFunction record) {
        return context.submit(resume_on, std::move(record));
    }
    // presenter_vulkan.h's createPresenter reaches the context's private creator.
    [[nodiscard]] static Result<std::unique_ptr<Presenter>> makePresenter(ExecutionContext& context,
                                                                          PresenterOptions options,
                                                                          FrameProvider frames) {
        return context.makePresenter(std::move(options), std::move(frames));
    }

    // The fields behind context_vulkan.h's image accessors.
    [[nodiscard]] static const ImageDesc& imageDesc(const GpuImage& image) noexcept {
        return image.desc_;
    }
    [[nodiscard]] static VkImage imageHandle(const GpuImage& image) noexcept {
        return image.image_;
    }
    [[nodiscard]] static VkImageView imageView(const GpuImage& image) noexcept {
        return image.view_;
    }
    [[nodiscard]] static VkImageLayout imageLayout(const GpuImage& image) noexcept {
        return image.layout_.load(std::memory_order_acquire);
    }
};

}  // namespace dvv::gpu::detail
