#pragma once

// GPU buffers and images (spec §5.3): RAII over a VMA allocation, the Vulkan handle and a debug
// name, created by an ExecutionContext. The destructor hands the handle and its memory to the
// context's retire queue, which frees them once the GPU has finished every submission made so far.
// The host reaches a buffer's memory through the scopes MappedWrite and MappedRead (spec §5.6).

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

#include <vulkan/vulkan_core.h>

#include "dvv/base/error.h"
#include "dvv/base/lifetime.h"
#include "dvv/gpu/device_id.h"

struct VmaAllocation_T;

namespace dvv::gpu {

class ExecutionContext;

namespace detail {
class DeviceCore;
struct Access;
}  // namespace detail

// Where a buffer's memory lives and how the host reaches it.
enum class MemoryUsage : std::uint8_t {
    DeviceOnly,    // device-local; no host access
    HostUpload,    // host-visible and mapped; the host writes it sequentially
    HostReadback,  // host-visible and mapped; the host reads (and writes) it in any order
};

struct BufferDesc {
    std::uint64_t size = 0;
    // Vulkan usage flags; VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT is always added, because
    // shaders receive buffers by device address (spec §5.4).
    VkBufferUsageFlags usage = 0;
    MemoryUsage memory = MemoryUsage::DeviceOnly;
    std::string name{};  // debug name for tools and validation messages
};

// A 2D image with one mip level and one layer in device-local memory, optimal tiling.
struct ImageDesc {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageUsageFlags usage = 0;
    std::string name{};
};

class GpuBuffer;  // the friend declarations below do not make the name visible

// A write view of a host-visible buffer's mapped memory, from GpuBuffer::mapWrite(). Move-only;
// the buffer must outlive it (the caller holds the buffer). bytes() stays valid while the scope
// lives; a scope may be stored (e.g. in a frame). Call flush() after writing and before the GPU
// reads; the destructor does not flush. One write scope of a buffer at a time.
class MappedWrite {
public:
    MappedWrite(MappedWrite&& other) noexcept;
    MappedWrite& operator=(MappedWrite&& other) noexcept;
    MappedWrite(const MappedWrite&) = delete;
    MappedWrite& operator=(const MappedWrite&) = delete;
    ~MappedWrite() = default;
    // Empty for a DeviceOnly buffer.
    [[nodiscard]] std::span<std::byte> bytes() const noexcept;
    // Makes host writes in [offset, offset + size) visible to the device (offsets from the buffer's
    // start). Does nothing on host-coherent memory or an empty scope.
    [[nodiscard]] Result<void> flush(std::uint64_t offset = 0, std::uint64_t size = VK_WHOLE_SIZE);

private:
    friend class GpuBuffer;
    MappedWrite(GpuBuffer& buffer, std::span<std::byte> mapped_bytes) noexcept;
    GpuBuffer* buffer_ = nullptr;  // null when moved from
    std::span<std::byte> bytes_{};
};

// A read view, from GpuBuffer::mapRead(), which made device writes visible first. Move-only; the
// buffer must outlive it. bytes() stays valid while the scope lives; a scope may be stored (e.g.
// `std::optional<MappedRead>`, or `std::unique_ptr<MappedRead>` where only a declaration is
// visible). It sees later device writes only after a new mapRead().
class MappedRead {
public:
    MappedRead(MappedRead&& other) noexcept;
    MappedRead& operator=(MappedRead&& other) noexcept;
    MappedRead(const MappedRead&) = delete;
    MappedRead& operator=(const MappedRead&) = delete;
    ~MappedRead() = default;
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept;  // empty for DeviceOnly

private:
    friend class GpuBuffer;
    explicit MappedRead(std::span<const std::byte> mapped_bytes) noexcept;
    std::span<const std::byte> bytes_{};
};

class GpuBuffer final : public Managed<GpuBuffer> {
public:
    GpuBuffer(CreationKey<ExecutionContext> key, std::shared_ptr<detail::DeviceCore> core,
              BufferDesc desc, VkBuffer buffer, VmaAllocation_T* allocation,
              VkDeviceAddress address, std::span<std::byte> mapped) noexcept;
    // Retires the buffer and its memory (see above).
    ~GpuBuffer();

    [[nodiscard]] const BufferDesc& desc() const noexcept { return desc_; }
    [[nodiscard]] std::uint64_t size() const noexcept { return desc_.size; }
    [[nodiscard]] VkBuffer handle() const noexcept { return buffer_; }
    [[nodiscard]] VkDeviceAddress deviceAddress() const noexcept { return address_; }
    // Non-const: only a holder that may change the buffer can write it.
    [[nodiscard]] MappedWrite mapWrite() noexcept;
    // Invalidates the mapped range (call after the GPU work completed), then returns the view.
    // Vulkan errors as converted; an empty scope for DeviceOnly without a call.
    [[nodiscard]] Result<MappedRead> mapRead() const;
    // Deprecated (0.3.0): use mapWrite() or mapRead(); removed in 1.0.0.
    // The mapped memory of a host-visible buffer (HostUpload, HostReadback); empty otherwise.
    [[nodiscard]] std::span<std::byte> mapped() const noexcept { return mapped_; }
    // Deprecated (0.3.0): use MappedWrite::flush(); removed in 1.0.0.
    // Makes host writes visible to the device; call after writing mapped() and before the GPU
    // reads. Does nothing on host-coherent memory.
    [[nodiscard]] Result<void> flush(std::uint64_t offset = 0, std::uint64_t size = VK_WHOLE_SIZE);
    // Deprecated (0.3.0): use mapRead(); removed in 1.0.0.
    // Makes device writes visible to the host; call after the GPU work completed and before
    // reading mapped(). Does nothing on host-coherent memory.
    [[nodiscard]] Result<void> invalidate(std::uint64_t offset = 0,
                                          std::uint64_t size = VK_WHOLE_SIZE);
    [[nodiscard]] DeviceId deviceId() const noexcept;
    [[nodiscard]] DeviceGeneration generation() const noexcept { return generation_; }

private:
    friend struct detail::Access;  // the command recorder checks the device

    std::shared_ptr<detail::DeviceCore> core_;
    BufferDesc desc_;
    VkBuffer buffer_;
    VmaAllocation_T* allocation_;
    VkDeviceAddress address_;
    std::span<std::byte> mapped_;
    DeviceGeneration generation_;
};

// Shared from this: the command recorder keeps the images whose layout a submission changes.
class GpuImage final : public Managed<GpuImage>, public std::enable_shared_from_this<GpuImage> {
public:
    GpuImage(CreationKey<ExecutionContext> key, std::shared_ptr<detail::DeviceCore> core,
             ImageDesc desc, VkImage image, VmaAllocation_T* allocation, VkImageView view) noexcept;
    // Retires the view, then the image and its memory (see above).
    ~GpuImage();

    [[nodiscard]] const ImageDesc& desc() const noexcept { return desc_; }
    [[nodiscard]] std::uint32_t width() const noexcept { return desc_.width; }
    [[nodiscard]] std::uint32_t height() const noexcept { return desc_.height; }
    [[nodiscard]] VkFormat format() const noexcept { return desc_.format; }
    [[nodiscard]] VkImage handle() const noexcept { return image_; }
    // The view of the whole image (2D, the image's format); VK_NULL_HANDLE without STORAGE or
    // SAMPLED usage.
    [[nodiscard]] VkImageView view() const noexcept { return view_; }
    // The layout the recorded commands leave the image in; VK_IMAGE_LAYOUT_UNDEFINED at first.
    [[nodiscard]] VkImageLayout layout() const noexcept {
        return layout_.load(std::memory_order_acquire);
    }
    [[nodiscard]] DeviceId deviceId() const noexcept;
    [[nodiscard]] DeviceGeneration generation() const noexcept { return generation_; }

private:
    friend struct detail::Access;  // the command recorder checks the device, updates the layout

    std::shared_ptr<detail::DeviceCore> core_;
    ImageDesc desc_;
    VkImage image_;
    VmaAllocation_T* allocation_;
    VkImageView view_;
    std::atomic<VkImageLayout> layout_{VK_IMAGE_LAYOUT_UNDEFINED};
    DeviceGeneration generation_;
};

}  // namespace dvv::gpu
