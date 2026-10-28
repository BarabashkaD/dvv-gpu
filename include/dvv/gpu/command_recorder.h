#pragma once

// Records the commands of one submission (spec §4, §5.3). The submit thread passes a recorder to
// the function given to ExecutionContext::submit(); it is valid only during that call.
//
// Synchronisation is automatic and conservative: before every work command and before raw() a
// global memory barrier makes all earlier writes of compute shaders and transfers (also of earlier
// submissions) visible to it, and the end of every submission makes all writes visible to the
// host. Shaders reach buffers through device addresses, which synchronisation validation cannot
// see, so commands never rely on the caller for barriers.
//
// A command that cannot be recorded (a resource of another device, a range outside a buffer, a
// missing usage flag, ...) is a programming error: the recorder records nothing more, and the
// submission signals its timeline value without running any command and completes with that error
// (Errc::Internal).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <vulkan/vulkan_core.h>

#include "dvv/base/error.h"
#include "dvv/gpu/compute_pipeline.h"
#include "dvv/gpu/resources.h"

namespace dvv::gpu {

namespace detail {
class DeviceCore;
class Submitter;
struct Access;
}  // namespace detail

// The escape hatch for commands the recorder does not offer (CommandRecorder::raw()). Device
// functions are loaded with getDeviceProcAddr(device, "vkCmd...").
struct RawCommands {
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    PFN_vkGetDeviceProcAddr getDeviceProcAddr = nullptr;
};

class CommandRecorder {
public:
    CommandRecorder(const CommandRecorder&) = delete;
    CommandRecorder& operator=(const CommandRecorder&) = delete;
    CommandRecorder(CommandRecorder&&) = delete;
    CommandRecorder& operator=(CommandRecorder&&) = delete;
    ~CommandRecorder() = default;

    // Runs the pipeline's shader on group_count_x x y x z work groups with these push-constant
    // bytes, which must be exactly the pipeline's push-constant size (std::as_bytes of the
    // shader's push-constant struct). Group counts are 1 to the device's maxComputeWorkGroupCount.
    // The pipeline must bind no storage images.
    void dispatch(const ComputePipeline& pipeline, std::span<const std::byte> push_constants,
                  std::uint32_t group_count_x, std::uint32_t group_count_y = 1,
                  std::uint32_t group_count_z = 1);
    // As dispatch() above, and binds exactly pipeline.bindings().storageImages images, binding i =
    // storage_images[i]. Each image belongs to this device and generation, has STORAGE usage and
    // appears once. Images not yet in GENERAL move there in the automatic barrier (tracked like
    // every GpuImage layout); then the descriptors are pushed (vkCmdPushDescriptorSet, set 0) and
    // the dispatch is recorded. Otherwise the recorder fails (Errc::Internal). A bound image is
    // writable by the shader: do not bind an image others read (M4 adds read-only bindings).
    void dispatch(const ComputePipeline& pipeline, std::span<const std::byte> push_constants,
                  std::span<GpuImage* const> storage_images, std::uint32_t group_count_x,
                  std::uint32_t group_count_y = 1, std::uint32_t group_count_z = 1);
    // Fills `size` bytes from `offset` with a 32-bit value (VK_WHOLE_SIZE: to the end, rounded
    // down to a multiple of 4). The buffer needs VK_BUFFER_USAGE_TRANSFER_DST_BIT; offset and size
    // are multiples of 4.
    void fillBuffer(const GpuBuffer& buffer, std::uint32_t value, std::uint64_t offset = 0,
                    std::uint64_t size = VK_WHOLE_SIZE);
    // Copies the whole source buffer to the start of the destination.
    void copyBuffer(const GpuBuffer& source, const GpuBuffer& destination);
    // Copies `size` bytes. Source needs TRANSFER_SRC, destination TRANSFER_DST usage.
    void copyBuffer(const GpuBuffer& source, std::uint64_t source_offset,
                    const GpuBuffer& destination, std::uint64_t destination_offset,
                    std::uint64_t size);
    // Copies tightly packed texels from the buffer into the whole image. The image moves to
    // TRANSFER_DST_OPTIMAL first unless it is in that layout or GENERAL. Image copies and
    // transitions handle single-plane colour formats with 1x1 texel blocks only (no depth,
    // stencil, multi-planar or block-compressed formats).
    void copyBufferToImage(const GpuBuffer& source, GpuImage& destination,
                           std::uint64_t source_offset = 0);
    // Copies the whole image into the buffer, tightly packed. The image moves to
    // TRANSFER_SRC_OPTIMAL first unless it is in that layout or GENERAL.
    void copyImageToBuffer(GpuImage& source, const GpuBuffer& destination,
                           std::uint64_t destination_offset = 0);
    // Moves the image to a layout (nothing when it is already there): GENERAL,
    // TRANSFER_SRC_OPTIMAL or TRANSFER_DST_OPTIMAL (the image needs that transfer usage) or
    // SHADER_READ_ONLY_OPTIMAL (it needs SAMPLED usage); other layouts fail the recorder.
    // GpuImage::layout() reports the new layout once the submission is on the queue.
    void transition(GpuImage& image, VkImageLayout layout);

    // The command buffer for commands of one's own, after the automatic barrier. The caller
    // synchronises between its own raw commands, and must not change image layouts (the recorder
    // would not know). The automatic barriers cover compute and transfer stages only; raw commands
    // in other stages need their own barriers before and after them. Empty (null handles) once
    // the recorder has failed.
    [[nodiscard]] RawCommands raw();

    // Whether a command could not be recorded; the submission will complete with error().
    [[nodiscard]] bool failed() const noexcept { return error_.has_value(); }
    [[nodiscard]] const std::optional<Error>& error() const noexcept { return error_; }

private:
    friend class detail::Submitter;
    friend struct detail::Access;

    CommandRecorder(detail::DeviceCore& core, VkCommandBuffer command_buffer) noexcept;

    // Moves `source` to TRANSFER_SRC_OPTIMAL (tracked like every GpuImage), moves the swapchain
    // image UNDEFINED -> TRANSFER_DST_OPTIMAL, blits the whole source onto the whole destination
    // (presentBlitFilter: NEAREST when the sizes match; otherwise LINEAR when the source format has
    // SAMPLED_IMAGE_FILTER_LINEAR, else NEAREST; it never fails for this reason), then moves the
    // swapchain image to PRESENT_SRC_KHR (destination stage and access NONE). Fails the recorder
    // (Internal) for an image of another device, without TRANSFER_SRC usage or whose format
    // presentBlitSource refuses (no BLIT_SRC, integer, depth/stencil, multi-planar or
    // block-compressed), and for a null swapchain image or an empty extent. Reached through
    // detail::Access::presentBlit; the context must have presentation enabled. The caller
    // guarantees a swapchain image with TRANSFER_DST usage, a non-integer colour format with
    // BLIT_DST, `swapchain_extent` equal to the image's real extent, and presentation on this
    // context's queue.
    void presentBlit(GpuImage& source, VkImage swapchain_image, VkExtent2D swapchain_extent);

    // Records the barrier that makes every write visible to the host (end of the submission).
    void finish();
    // Stores the layouts recorded for each image in the images; after a successful submission.
    void applyLayouts() noexcept;

    // False once a command has failed: nothing more is recorded.
    [[nodiscard]] bool usable() const noexcept;
    // Fails the recorder with "<command>: <problem>" unless the condition holds.
    [[nodiscard]] bool check(bool condition, const char* command, const std::string& problem);
    [[nodiscard]] bool checkBuffer(const char* command, const GpuBuffer& buffer,
                                   VkBufferUsageFlags usage, std::uint64_t offset,
                                   std::uint64_t size);
    [[nodiscard]] bool checkImage(const char* command, const GpuImage& image,
                                  VkImageUsageFlags usage);
    // Single-plane colour formats with 1x1 texel blocks: what the copies and transitions handle.
    [[nodiscard]] bool checkFormat(const char* command, const GpuImage& image);
    void fail(std::string message);

    // The layout the commands recorded so far leave the image in.
    [[nodiscard]] VkImageLayout layoutOf(const GpuImage& image) const noexcept;
    // Global memory barrier, combined with the image's layout change when `image` is set and not
    // in `layout` already.
    void barrier(GpuImage* image = nullptr, VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED);
    // Global memory barrier, combined with a layout change for every image not yet in `layout`.
    void barrier(std::span<GpuImage* const> images, VkImageLayout layout);
    // Moves the image to `layout` unless it is in `layout` or GENERAL already; returns the layout
    // the command uses.
    VkImageLayout prepareTransfer(GpuImage& image, VkImageLayout layout);

    detail::DeviceCore* core_;
    VkCommandBuffer command_buffer_;
    // The images whose layout this submission changes, with their latest layout. Owning
    // references: an image dropped by the record function stays valid until applyLayouts().
    std::vector<std::pair<std::shared_ptr<GpuImage>, VkImageLayout>> layouts_;
    std::optional<Error> error_;
};

}  // namespace dvv::gpu
