#include "dvv/gpu/context_vulkan.h"

#include <utility>

#include "access.h"

namespace dvv::gpu {

Result<std::shared_ptr<GpuBuffer>> createBuffer(ExecutionContext& context, BufferDesc desc) {
    return detail::Access::createBuffer(context, std::move(desc));
}

Result<std::shared_ptr<GpuImage>> createImage(ExecutionContext& context, ImageDesc desc) {
    return detail::Access::createImage(context, std::move(desc));
}

Result<std::shared_ptr<ComputePipeline>> createComputePipeline(ExecutionContext& context,
                                                               const EmbeddedShader& shader,
                                                               std::uint32_t push_constant_size,
                                                               ComputeBindings bindings) {
    return detail::Access::createComputePipeline(context, shader, push_constant_size, bindings);
}

VkFormatFeatureFlags2 formatFeatures(const ExecutionContext& context, VkFormat format) {
    return detail::Access::formatFeatures(context, format);
}

SubmitOperation submit(ExecutionContext& context, IExecutor& resume_on, RecordFunction record) {
    return detail::Access::submit(context, resume_on, std::move(record));
}

const ImageDesc& vulkanDesc(const GpuImage& image) noexcept {
    return detail::Access::imageDesc(image);
}

VkFormat vulkanFormat(const GpuImage& image) noexcept {
    return detail::Access::imageDesc(image).format;
}

VkImage vulkanHandle(const GpuImage& image) noexcept {
    return detail::Access::imageHandle(image);
}

VkImageView vulkanView(const GpuImage& image) noexcept {
    return detail::Access::imageView(image);
}

VkImageLayout vulkanLayout(const GpuImage& image) noexcept {
    return detail::Access::imageLayout(image);
}

}  // namespace dvv::gpu
