#pragma once

// include/dvv/gpu/context_vulkan.h — the Vulkan entry points of a context and an image.
// 0.3.x: includes execution_context.h, resources.h, compute_pipeline.h, embedded_shader.h and
// submission.h. 1.0.0: the same list, with image.h for GpuImage.

#include <cstdint>
#include <memory>

#include <vulkan/vulkan_core.h>

#include "dvv/base/error.h"
#include "dvv/base/executor.h"
#include "dvv/gpu/compute_pipeline.h"
#include "dvv/gpu/embedded_shader.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/resources.h"
#include "dvv/gpu/submission.h"

namespace dvv::gpu {

// The errors of each function are those of the member it replaces (execution_context.h).
[[nodiscard]] Result<std::shared_ptr<GpuBuffer>> createBuffer(ExecutionContext& context,
                                                              BufferDesc desc);
[[nodiscard]] Result<std::shared_ptr<GpuImage>> createImage(ExecutionContext& context,
                                                            ImageDesc desc);
[[nodiscard]] Result<std::shared_ptr<ComputePipeline>> createComputePipeline(
    ExecutionContext& context, const EmbeddedShader& shader, std::uint32_t push_constant_size,
    ComputeBindings bindings = {});
[[nodiscard]] VkFormatFeatureFlags2 formatFeatures(const ExecutionContext& context,
                                                   VkFormat format);
[[nodiscard]] SubmitOperation submit(ExecutionContext& context, IExecutor& resume_on,
                                     RecordFunction record);

// The image's Vulkan side.
[[nodiscard]] const ImageDesc& vulkanDesc(const GpuImage& image) noexcept;
[[nodiscard]] VkFormat vulkanFormat(const GpuImage& image) noexcept;
[[nodiscard]] VkImage vulkanHandle(const GpuImage& image) noexcept;
// VK_NULL_HANDLE without STORAGE or SAMPLED usage.
[[nodiscard]] VkImageView vulkanView(const GpuImage& image) noexcept;
// The layout the recorded commands leave the image in (acquire load, as today).
[[nodiscard]] VkImageLayout vulkanLayout(const GpuImage& image) noexcept;

}  // namespace dvv::gpu
