#include "dvv/gpu/resources.h"

#include <utility>

#include "device_core.h"
#include "vk_result.h"

namespace dvv::gpu {

static_assert(ManagedType<GpuBuffer>);
static_assert(ManagedType<GpuImage>);

GpuBuffer::GpuBuffer(CreationKey<ExecutionContext> key, std::shared_ptr<detail::DeviceCore> core,
                     BufferDesc desc, VkBuffer buffer, VmaAllocation_T* allocation,
                     VkDeviceAddress address, std::span<std::byte> mapped) noexcept
    : Managed(std::move(key)),
      core_(std::move(core)),
      desc_(std::move(desc)),
      buffer_(buffer),
      allocation_(allocation),
      address_(address),
      mapped_(mapped),
      generation_(core_->generation()) {}

GpuBuffer::~GpuBuffer() {
    core_->retire(vk::Buffer(buffer_), allocation_);
    core_->collect();
}

Result<void> GpuBuffer::flush(std::uint64_t offset, std::uint64_t size) {
    return detail::vkCheck(vmaFlushAllocation(core_->allocator(), allocation_, offset, size),
                           "vmaFlushAllocation");
}

Result<void> GpuBuffer::invalidate(std::uint64_t offset, std::uint64_t size) {
    return detail::vkCheck(vmaInvalidateAllocation(core_->allocator(), allocation_, offset, size),
                           "vmaInvalidateAllocation");
}

DeviceId GpuBuffer::deviceId() const noexcept {
    return core_->id();
}

GpuImage::GpuImage(CreationKey<ExecutionContext> key, std::shared_ptr<detail::DeviceCore> core,
                   ImageDesc desc, VkImage image, VmaAllocation_T* allocation,
                   VkImageView view) noexcept
    : Managed(std::move(key)),
      core_(std::move(core)),
      desc_(std::move(desc)),
      image_(image),
      allocation_(allocation),
      view_(view),
      generation_(core_->generation()) {}

GpuImage::~GpuImage() {
    // The retire queue frees in order, so the view is never destroyed after its image.
    if (view_ != VK_NULL_HANDLE) {
        core_->retire(vk::ImageView(view_), nullptr);
    }
    core_->retire(vk::Image(image_), allocation_);
    core_->collect();
}

DeviceId GpuImage::deviceId() const noexcept {
    return core_->id();
}

}  // namespace dvv::gpu
