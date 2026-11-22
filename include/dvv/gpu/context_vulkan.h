#pragma once

// The Vulkan entry points of a context and an image. Includes execution_context.h, image.h,
// resources.h, compute_pipeline.h, embedded_shader.h and submission.h.

#include <cstdint>
#include <memory>

#include <vulkan/vulkan_core.h>

#include "dvv/base/error.h"
#include "dvv/base/executor.h"
#include "dvv/gpu/compute_pipeline.h"
#include "dvv/gpu/embedded_shader.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/image.h"
#include "dvv/gpu/resources.h"
#include "dvv/gpu/submission.h"

namespace dvv::gpu {

// A buffer. Errc::DeviceLost on a lost context; Errc::Internal for a zero size; Vulkan errors as
// converted by dvv-gpu (e.g. OutOfMemory). Frees retired memory the GPU has finished with first.
[[nodiscard]] Result<std::shared_ptr<GpuBuffer>> createBuffer(ExecutionContext& context,
                                                              BufferDesc desc);
// A 2D image. Errc::DeviceLost on a lost context; Errc::Internal when width, height, format or
// usage is missing; Errc::Unsupported when the device cannot create it (format, usage or size).
// STORAGE usage needs a format with VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT (else Errc::Unsupported
// "image '<name>': device <id> has no storage images of <format>"). STORAGE or SAMPLED usage also
// creates the image's view (vulkanView(): 2D, the whole image, the image's format), retired with
// the image.
[[nodiscard]] Result<std::shared_ptr<GpuImage>> createImage(ExecutionContext& context,
                                                            ImageDesc desc);
// A compute pipeline from an embedded shader (entry point main) with a push-constant range of
// `push_constant_size` bytes (0: none), made through the context's pipeline cache, and the storage
// images it binds (set 0, bindings 0..storageImages-1, pushed per dispatch). The range must cover
// the shader's push-constant block. Errc::DeviceLost on a lost context; Errc::Internal when the
// shader is not SPIR-V, the size is not a multiple of 4 or smaller than the block,
// bindings.storageImages exceeds the device's maxPushDescriptors, or the shader declares descriptor
// bindings other than set 0, bindings 0..storageImages-1 (the block and the bindings are checked
// when the shader has its reflection JSON); Errc::Format when the reflection JSON cannot be read;
// Errc::Unsupported when the size exceeds the device's limit; Vulkan errors as converted.
[[nodiscard]] Result<std::shared_ptr<ComputePipeline>> createComputePipeline(
    ExecutionContext& context, const EmbeddedShader& shader, std::uint32_t push_constant_size,
    ComputeBindings bindings = {});
// The device's optimal-tiling features of a format (VkFormatProperties3), e.g.
// VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT.
[[nodiscard]] VkFormatFeatureFlags2 formatFeatures(const ExecutionContext& context,
                                                   VkFormat format);
// One submission: co_await submit(context, executor, record). The submit thread calls `record`
// with a CommandRecorder, then submits the commands with a signal of the next timeline value; the
// awaiting coroutine resumes on `resume_on` once the GPU has finished them, with that value or the
// error that stopped the submission. A record function that throws fails the submission
// (Errc::Internal); its timeline value is signalled all the same, so later submissions are not
// held up. The record function, and everything it captures, is released on `resume_on` before the
// coroutine resumes: capturing the resources a submission uses keeps them alive until the GPU is
// done with them. An empty record function submits only the final host-visibility barrier. The
// record function must not block or wait for another submission: the submit thread records one
// submission at a time. `resume_on` must outlive the operation. On a lost context the submission
// completes with Errc::DeviceLost without recording (the record function is released on
// `resume_on` as usual).
[[nodiscard]] SubmitOperation submit(ExecutionContext& context, IExecutor& resume_on,
                                     RecordFunction record);

// The image's Vulkan side.
[[nodiscard]] const ImageDesc& vulkanDesc(const GpuImage& image) noexcept;
[[nodiscard]] VkFormat vulkanFormat(const GpuImage& image) noexcept;
[[nodiscard]] VkImage vulkanHandle(const GpuImage& image) noexcept;
// The view of the whole image (2D, the image's format); VK_NULL_HANDLE without STORAGE or SAMPLED
// usage.
[[nodiscard]] VkImageView vulkanView(const GpuImage& image) noexcept;
// The layout the recorded commands leave the image in (read with acquire order);
// VK_IMAGE_LAYOUT_UNDEFINED at first.
[[nodiscard]] VkImageLayout vulkanLayout(const GpuImage& image) noexcept;

}  // namespace dvv::gpu
