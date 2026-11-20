#include "dvv/gpu/resources.h"

#include <utility>

#include "device_core.h"
#include "vk_result.h"

namespace dvv::gpu {

static_assert(ManagedType<GpuBuffer>);
static_assert(ManagedType<GpuImage>);

namespace {

// The body of GpuBuffer::invalidate(), shared with mapRead().
Result<void> invalidateAllocation(detail::DeviceCore& core, VmaAllocation_T* allocation,
                                  std::uint64_t offset, std::uint64_t size) {
    return detail::vkCheck(vmaInvalidateAllocation(core.allocator(), allocation, offset, size),
                           "vmaInvalidateAllocation");
}

}  // namespace

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

MappedWrite::MappedWrite(GpuBuffer& buffer, std::span<std::byte> mapped_bytes) noexcept
    : buffer_(&buffer), bytes_(mapped_bytes) {}

MappedWrite::MappedWrite(MappedWrite&& other) noexcept
    : buffer_(std::exchange(other.buffer_, nullptr)), bytes_(std::exchange(other.bytes_, {})) {}

MappedWrite& MappedWrite::operator=(MappedWrite&& other) noexcept {
    if (this != &other) {
        buffer_ = std::exchange(other.buffer_, nullptr);
        bytes_ = std::exchange(other.bytes_, {});
    }
    return *this;
}

std::span<std::byte> MappedWrite::bytes() const noexcept {
    return bytes_;
}

Result<void> MappedWrite::flush(std::uint64_t offset, std::uint64_t size) {
    if (buffer_ == nullptr || bytes_.empty()) {
        return {};  // moved from, or a DeviceOnly buffer
    }
    return buffer_->flush(offset, size);
}

MappedRead::MappedRead(std::span<const std::byte> mapped_bytes) noexcept : bytes_(mapped_bytes) {}

MappedRead::MappedRead(MappedRead&& other) noexcept : bytes_(std::exchange(other.bytes_, {})) {}

MappedRead& MappedRead::operator=(MappedRead&& other) noexcept {
    if (this != &other) {
        bytes_ = std::exchange(other.bytes_, {});
    }
    return *this;
}

std::span<const std::byte> MappedRead::bytes() const noexcept {
    return bytes_;
}

MappedWrite GpuBuffer::mapWrite() noexcept {
    return {*this, mapped_};
}

Result<MappedRead> GpuBuffer::mapRead() const {
    if (!mapped_.empty()) {
        DVV_TRY_VOID(invalidateAllocation(*core_, allocation_, 0, VK_WHOLE_SIZE));
    }
    return MappedRead(mapped_);
}

Result<void> GpuBuffer::flush(std::uint64_t offset, std::uint64_t size) {
    return detail::vkCheck(vmaFlushAllocation(core_->allocator(), allocation_, offset, size),
                           "vmaFlushAllocation");
}

Result<void> GpuBuffer::invalidate(std::uint64_t offset, std::uint64_t size) {
    return invalidateAllocation(*core_, allocation_, offset, size);
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
