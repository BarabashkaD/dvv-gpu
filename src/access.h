#pragma once

// Access to the private parts of dvv-gpu's public classes, for its own sources and white-box tests.

#include <cstddef>

#include "device_core.h"
#include "dvv/gpu/command_recorder.h"
#include "dvv/gpu/compute_pipeline.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/gpu_registry.h"
#include "dvv/gpu/resources.h"
#include "instance.h"
#include "pipeline_cache.h"
#include "submitter.h"

namespace dvv::gpu::detail {

struct Access {
    [[nodiscard]] static InstanceCore& instance(const GpuRegistry& registry) noexcept {
        return *registry.instance_;
    }
    [[nodiscard]] static DeviceCore& core(const ExecutionContext& context) noexcept {
        return *context.core_;
    }
    [[nodiscard]] static DeviceCore& core(const GpuBuffer& buffer) noexcept {
        return *buffer.core_;
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
};

}  // namespace dvv::gpu::detail
