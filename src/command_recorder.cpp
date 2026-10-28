#include "dvv/gpu/command_recorder.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "access.h"
#include "device_core.h"
#include "present_blit.h"
#include "vk.h"

namespace dvv::gpu {

namespace {

// Every command the recorder records runs in one of these stages.
constexpr vk::PipelineStageFlags2 WORK_STAGES =
    vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eAllTransfer;
constexpr vk::AccessFlags2 WORK_ACCESS =
    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
constexpr vk::ImageSubresourceRange COLOR_RANGE{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                .baseMipLevel = 0,
                                                .levelCount = 1,
                                                .baseArrayLayer = 0,
                                                .layerCount = 1};
constexpr vk::ImageSubresourceLayers COLOR_LAYERS{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                  .mipLevel = 0,
                                                  .baseArrayLayer = 0,
                                                  .layerCount = 1};

std::string describe(const GpuBuffer& buffer) {
    return std::format("buffer '{}' ({} bytes)", buffer.desc().name, buffer.size());
}

std::string describe(const GpuImage& image) {
    return std::format("image '{}' ({}x{} {})", image.desc().name, image.width(), image.height(),
                       vk::to_string(static_cast<vk::Format>(image.format())));
}

std::string layoutName(VkImageLayout layout) {
    return vk::to_string(static_cast<vk::ImageLayout>(layout));
}

// Bytes of the whole image, tightly packed.
std::uint64_t imageBytes(const GpuImage& image) {
    return std::uint64_t{image.width()} * image.height() *
           vk::blockSize(static_cast<vk::Format>(image.format()));
}

}  // namespace

CommandRecorder::CommandRecorder(detail::DeviceCore& core, VkCommandBuffer command_buffer) noexcept
    : core_(&core), command_buffer_(command_buffer) {}

void CommandRecorder::dispatch(const ComputePipeline& pipeline,
                               std::span<const std::byte> push_constants,
                               std::uint32_t group_count_x, std::uint32_t group_count_y,
                               std::uint32_t group_count_z) {
    dispatch(pipeline, push_constants, std::span<GpuImage* const>{}, group_count_x, group_count_y,
             group_count_z);
}

void CommandRecorder::dispatch(const ComputePipeline& pipeline,
                               std::span<const std::byte> push_constants,
                               std::span<GpuImage* const> storage_images,
                               std::uint32_t group_count_x, std::uint32_t group_count_y,
                               std::uint32_t group_count_z) {
    if (!usable()) {
        return;
    }
    // The messages are formatted only on failure: dispatch is the most frequent command.
    const auto& max = core_->limits().maxComputeWorkGroupCount;
    if (&detail::Access::core(pipeline) != core_ || pipeline.generation() != core_->generation()) {
        static_cast<void>(
            check(false, "dispatch",
                  std::format("pipeline '{}' belongs to another device", pipeline.name())));
        return;
    }
    if (push_constants.size() != pipeline.pushConstantSize()) {
        static_cast<void>(check(
            false, "dispatch",
            std::format("pipeline '{}' takes {} bytes of push constants, not {}", pipeline.name(),
                        pipeline.pushConstantSize(), push_constants.size())));
        return;
    }
    if (group_count_x < 1 || group_count_y < 1 || group_count_z < 1 || group_count_x > max[0] ||
        group_count_y > max[1] || group_count_z > max[2]) {
        static_cast<void>(check(
            false, "dispatch",
            std::format("the group counts {}x{}x{} must be at least 1 and at most {}x{}x{}",
                        group_count_x, group_count_y, group_count_z, max[0], max[1], max[2])));
        return;
    }
    if (storage_images.size() != pipeline.bindings().storageImages) {
        static_cast<void>(
            check(false, "dispatch",
                  std::format("pipeline '{}' binds {} storage images, not {}", pipeline.name(),
                              pipeline.bindings().storageImages, storage_images.size())));
        return;
    }
    for (std::size_t index = 0; index < storage_images.size(); ++index) {
        GpuImage* const image = storage_images[index];
        const auto earlier = storage_images.first(index);
        if (!check(image != nullptr, "dispatch", std::format("storage image {} is null", index)) ||
            !checkImage("dispatch", *image, VK_IMAGE_USAGE_STORAGE_BIT) ||
            !checkFormat("dispatch", *image) ||
            !check(std::ranges::find(earlier, image) == earlier.end(), "dispatch",
                   std::format("{} appears twice", describe(*image)))) {
            return;
        }
    }
    barrier(storage_images, VK_IMAGE_LAYOUT_GENERAL);
    const vk::CommandBuffer commands(command_buffer_);
    commands.bindPipeline(vk::PipelineBindPoint::eCompute, vk::Pipeline(pipeline.handle()),
                          core_->dispatch());
    if (!storage_images.empty()) {
        std::vector<vk::DescriptorImageInfo> infos(storage_images.size());
        std::vector<vk::WriteDescriptorSet> writes(storage_images.size());
        for (std::size_t index = 0; index < storage_images.size(); ++index) {
            infos[index] = {.imageView = vk::ImageView(storage_images[index]->view()),
                            .imageLayout = vk::ImageLayout::eGeneral};
            writes[index] = {.dstBinding = static_cast<std::uint32_t>(index),
                             .descriptorCount = 1,
                             .descriptorType = vk::DescriptorType::eStorageImage,
                             .pImageInfo = &infos[index]};
        }
        commands.pushDescriptorSet(vk::PipelineBindPoint::eCompute,
                                   vk::PipelineLayout(pipeline.layout()), 0, writes,
                                   core_->dispatch());
    }
    if (!push_constants.empty()) {
        const vk::PushConstantsInfo push{.layout = vk::PipelineLayout(pipeline.layout()),
                                         .stageFlags = vk::ShaderStageFlagBits::eCompute,
                                         .offset = 0,
                                         .size = static_cast<std::uint32_t>(push_constants.size()),
                                         .pValues = push_constants.data()};
        commands.pushConstants2(push, core_->dispatch());
    }
    commands.dispatch(group_count_x, group_count_y, group_count_z, core_->dispatch());
}

void CommandRecorder::fillBuffer(const GpuBuffer& buffer, std::uint32_t value, std::uint64_t offset,
                                 std::uint64_t size) {
    if (!usable()) {
        return;
    }
    // VK_WHOLE_SIZE: the rest of the buffer, rounded down to whole 32-bit values.
    const std::uint64_t length =
        size != VK_WHOLE_SIZE ? size
                              : (offset < buffer.size() ? (buffer.size() - offset) & ~3ULL : 0);
    if (!check(offset % 4 == 0 && length % 4 == 0 && length > 0, "fillBuffer",
               std::format("offset {} and size {} must be multiples of 4, the size not 0", offset,
                           length)) ||
        !checkBuffer("fillBuffer", buffer, VK_BUFFER_USAGE_TRANSFER_DST_BIT, offset, length)) {
        return;
    }
    barrier();
    vk::CommandBuffer(command_buffer_)
        .fillBuffer(vk::Buffer(buffer.handle()), offset, length, value, core_->dispatch());
}

void CommandRecorder::copyBuffer(const GpuBuffer& source, const GpuBuffer& destination) {
    copyBuffer(source, 0, destination, 0, source.size());
}

void CommandRecorder::copyBuffer(const GpuBuffer& source, std::uint64_t source_offset,
                                 const GpuBuffer& destination, std::uint64_t destination_offset,
                                 std::uint64_t size) {
    if (!usable()) {
        return;
    }
    if (!check(size > 0, "copyBuffer", "the size is 0") ||
        !checkBuffer("copyBuffer", source, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, source_offset, size) ||
        !checkBuffer("copyBuffer", destination, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     destination_offset, size) ||
        !check(&source != &destination || source_offset + size <= destination_offset ||
                   destination_offset + size <= source_offset,
               "copyBuffer", std::format("the ranges overlap in {}", describe(source)))) {
        return;
    }
    barrier();
    const vk::BufferCopy region{
        .srcOffset = source_offset, .dstOffset = destination_offset, .size = size};
    vk::CommandBuffer(command_buffer_)
        .copyBuffer(vk::Buffer(source.handle()), vk::Buffer(destination.handle()), region,
                    core_->dispatch());
}

void CommandRecorder::copyBufferToImage(const GpuBuffer& source, GpuImage& destination,
                                        std::uint64_t source_offset) {
    if (!usable()) {
        return;
    }
    const std::uint64_t texel = vk::blockSize(static_cast<vk::Format>(destination.format()));
    if (!checkImage("copyBufferToImage", destination, VK_IMAGE_USAGE_TRANSFER_DST_BIT) ||
        !checkFormat("copyBufferToImage", destination) ||
        !check(source_offset % texel == 0, "copyBufferToImage",
               std::format("the offset {} is not a multiple of the texel size {}", source_offset,
                           texel)) ||
        !checkBuffer("copyBufferToImage", source, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, source_offset,
                     imageBytes(destination))) {
        return;
    }
    const VkImageLayout layout = prepareTransfer(destination, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    const vk::BufferImageCopy region{
        .bufferOffset = source_offset,
        .imageSubresource = COLOR_LAYERS,
        .imageExtent = {.width = destination.width(), .height = destination.height(), .depth = 1}};
    vk::CommandBuffer(command_buffer_)
        .copyBufferToImage(vk::Buffer(source.handle()), vk::Image(destination.handle()),
                           static_cast<vk::ImageLayout>(layout), region, core_->dispatch());
}

void CommandRecorder::copyImageToBuffer(GpuImage& source, const GpuBuffer& destination,
                                        std::uint64_t destination_offset) {
    if (!usable()) {
        return;
    }
    const std::uint64_t texel = vk::blockSize(static_cast<vk::Format>(source.format()));
    if (!checkImage("copyImageToBuffer", source, VK_IMAGE_USAGE_TRANSFER_SRC_BIT) ||
        !checkFormat("copyImageToBuffer", source) ||
        !check(destination_offset % texel == 0, "copyImageToBuffer",
               std::format("the offset {} is not a multiple of the texel size {}",
                           destination_offset, texel)) ||
        !checkBuffer("copyImageToBuffer", destination, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     destination_offset, imageBytes(source))) {
        return;
    }
    const VkImageLayout layout = prepareTransfer(source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    const vk::BufferImageCopy region{
        .bufferOffset = destination_offset,
        .imageSubresource = COLOR_LAYERS,
        .imageExtent = {.width = source.width(), .height = source.height(), .depth = 1}};
    vk::CommandBuffer(command_buffer_)
        .copyImageToBuffer(vk::Image(source.handle()), static_cast<vk::ImageLayout>(layout),
                           vk::Buffer(destination.handle()), region, core_->dispatch());
}

void CommandRecorder::transition(GpuImage& image, VkImageLayout layout) {
    if (!usable()) {
        return;
    }
    // The layouts compute and transfer work uses, with the usage each one needs.
    bool supported = true;
    VkImageUsageFlags usage = 0;
    switch (layout) {
        case VK_IMAGE_LAYOUT_GENERAL: break;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL: usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT; break;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT; break;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: usage = VK_IMAGE_USAGE_SAMPLED_BIT; break;
        default: supported = false; break;
    }
    if (!check(supported, "transition",
               std::format("{} cannot move to {}", describe(image), layoutName(layout))) ||
        !checkImage("transition", image, usage) || !checkFormat("transition", image)) {
        return;
    }
    if (layoutOf(image) != layout) {
        barrier(&image, layout);
    }
}

void CommandRecorder::presentBlit(GpuImage& source, VkImage swapchain_image,
                                  VkExtent2D swapchain_extent) {
    if (!usable()) {
        return;
    }
    if (!checkImage("presentBlit", source, VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
        return;
    }
    const VkFormatFeatureFlags2 features = core_->formatFeatures(source.format());
    if (!check(detail::presentBlitSource(source.format(), features), "presentBlit",
               std::format("{} has a format that cannot be blitted to the swapchain",
                           describe(source))) ||
        !check(swapchain_image != VK_NULL_HANDLE && swapchain_extent.width > 0 &&
                   swapchain_extent.height > 0,
               "presentBlit", "the swapchain image is null or empty")) {
        return;
    }
    const VkImageLayout source_layout =
        prepareTransfer(source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    const vk::CommandBuffer commands(command_buffer_);
    const vk::Image target(swapchain_image);
    // The presentation engine was done with the image when the acquire fence signalled, and the
    // host waited for that fence before this submission, so there is nothing to wait for (stage
    // NONE). UNDEFINED discards the old contents.
    const vk::ImageMemoryBarrier2 to_transfer{.srcStageMask = vk::PipelineStageFlagBits2::eNone,
                                              .srcAccessMask = vk::AccessFlagBits2::eNone,
                                              .dstStageMask = vk::PipelineStageFlagBits2::eBlit,
                                              .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
                                              .oldLayout = vk::ImageLayout::eUndefined,
                                              .newLayout = vk::ImageLayout::eTransferDstOptimal,
                                              .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                              .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                              .image = target,
                                              .subresourceRange = COLOR_RANGE};
    commands.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_transfer},
                              core_->dispatch());
    const VkExtent2D source_extent{.width = source.width(), .height = source.height()};
    const vk::ImageBlit2 region{
        .srcSubresource = COLOR_LAYERS,
        .srcOffsets =
            std::array<vk::Offset3D, 2>{
                vk::Offset3D{.x = 0, .y = 0, .z = 0},
                vk::Offset3D{.x = static_cast<std::int32_t>(source_extent.width),
                             .y = static_cast<std::int32_t>(source_extent.height),
                             .z = 1}},
        .dstSubresource = COLOR_LAYERS,
        .dstOffsets = std::array<vk::Offset3D, 2>{
            vk::Offset3D{.x = 0, .y = 0, .z = 0},
            vk::Offset3D{.x = static_cast<std::int32_t>(swapchain_extent.width),
                         .y = static_cast<std::int32_t>(swapchain_extent.height),
                         .z = 1}}};
    commands.blitImage2({.srcImage = vk::Image(source.handle()),
                         .srcImageLayout = static_cast<vk::ImageLayout>(source_layout),
                         .dstImage = target,
                         .dstImageLayout = vk::ImageLayout::eTransferDstOptimal,
                         .regionCount = 1,
                         .pRegions = &region,
                         .filter = static_cast<vk::Filter>(
                             detail::presentBlitFilter(source_extent, swapchain_extent, features))},
                        core_->dispatch());
    // Presentation needs no destination stage or access: the queue's later present is ordered
    // after the submission completed (the host waited for it).
    const vk::ImageMemoryBarrier2 to_present{.srcStageMask = vk::PipelineStageFlagBits2::eBlit,
                                             .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                                             .dstStageMask = vk::PipelineStageFlagBits2::eNone,
                                             .dstAccessMask = vk::AccessFlagBits2::eNone,
                                             .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                                             .newLayout = vk::ImageLayout::ePresentSrcKHR,
                                             .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                             .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                             .image = target,
                                             .subresourceRange = COLOR_RANGE};
    commands.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_present},
                              core_->dispatch());
}

RawCommands CommandRecorder::raw() {
    if (!usable()) {
        return {};
    }
    barrier();
    return {.commandBuffer = command_buffer_,
            .device = static_cast<VkDevice>(core_->device()),
            .getDeviceProcAddr = core_->dispatch().vkGetDeviceProcAddr};
}

void CommandRecorder::finish() {
    const vk::MemoryBarrier2 to_host{.srcStageMask = WORK_STAGES,
                                     .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                                     .dstStageMask = vk::PipelineStageFlagBits2::eHost,
                                     .dstAccessMask = vk::AccessFlagBits2::eHostRead};
    vk::CommandBuffer(command_buffer_)
        .pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &to_host},
                          core_->dispatch());
}

void CommandRecorder::applyLayouts() noexcept {
    for (const auto& [image, layout] : layouts_) {
        detail::Access::setLayout(*image, layout);
    }
}

bool CommandRecorder::usable() const noexcept {
    return !error_.has_value();
}

bool CommandRecorder::check(bool condition, const char* command, const std::string& problem) {
    if (!condition) {
        fail(std::format("{}: {}", command, problem));
    }
    return condition;
}

bool CommandRecorder::checkBuffer(const char* command, const GpuBuffer& buffer,
                                  VkBufferUsageFlags usage, std::uint64_t offset,
                                  std::uint64_t size) {
    return check(
               &detail::Access::core(buffer) == core_ && buffer.generation() == core_->generation(),
               command, std::format("{} belongs to another device", describe(buffer))) &&
           check((buffer.desc().usage & usage) == usage, command,
                 std::format("{} lacks the usage {}", describe(buffer),
                             vk::to_string(vk::BufferUsageFlags(usage)))) &&
           check(offset <= buffer.size() && size <= buffer.size() - offset, command,
                 std::format("{} has no {} bytes at offset {}", describe(buffer), size, offset));
}

bool CommandRecorder::checkImage(const char* command, const GpuImage& image,
                                 VkImageUsageFlags usage) {
    return check(&detail::Access::core(image) == core_ && image.generation() == core_->generation(),
                 command, std::format("{} belongs to another device", describe(image))) &&
           check((image.desc().usage & usage) == usage, command,
                 std::format("{} lacks the usage {}", describe(image),
                             vk::to_string(vk::ImageUsageFlags(usage))));
}

bool CommandRecorder::checkFormat(const char* command, const GpuImage& image) {
    const auto format = static_cast<vk::Format>(image.format());
    const auto block = vk::blockExtent(format);
    const std::string_view component =
        vk::componentCount(format) > 0 ? vk::componentName(format, 0) : "";
    return check(vk::planeCount(format) == 1 && block[0] == 1 && block[1] == 1 && block[2] == 1 &&
                     component != "D" && component != "S",
                 command,
                 std::format("{} has a format the recorder does not handle", describe(image)));
}

void CommandRecorder::fail(std::string message) {
    if (!error_) {
        error_ = makeError(Errc::Internal, std::move(message)).error();
    }
}

VkImageLayout CommandRecorder::layoutOf(const GpuImage& image) const noexcept {
    const auto recorded = std::ranges::find_if(
        layouts_, [&image](const auto& entry) { return entry.first.get() == &image; });
    return recorded != layouts_.end() ? recorded->second : image.layout();
}

void CommandRecorder::barrier(GpuImage* image, VkImageLayout layout) {
    if (image == nullptr) {
        barrier(std::span<GpuImage* const>{}, layout);
    } else {
        barrier(std::span<GpuImage* const>(&image, 1), layout);
    }
}

void CommandRecorder::barrier(std::span<GpuImage* const> images, VkImageLayout layout) {
    const vk::MemoryBarrier2 memory{.srcStageMask = WORK_STAGES,
                                    .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                                    .dstStageMask = WORK_STAGES,
                                    .dstAccessMask = WORK_ACCESS};
    std::vector<vk::ImageMemoryBarrier2> changes;
    std::vector<GpuImage*> changed;
    changes.reserve(images.size());
    changed.reserve(images.size());
    for (GpuImage* const image : images) {
        const VkImageLayout current = layoutOf(*image);
        if (current == layout) {
            continue;
        }
        changes.push_back({.srcStageMask = WORK_STAGES,
                           .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                           .dstStageMask = WORK_STAGES,
                           .dstAccessMask = WORK_ACCESS,
                           .oldLayout = static_cast<vk::ImageLayout>(current),
                           .newLayout = static_cast<vk::ImageLayout>(layout),
                           .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                           .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                           .image = vk::Image(image->handle()),
                           .subresourceRange = COLOR_RANGE});
        changed.push_back(image);
    }
    // Remembered before recording; reserving first makes the updates below nothrow, so a throw
    // leaves both the tracked layouts and the command buffer unchanged.
    layouts_.reserve(layouts_.size() + changed.size());
    for (GpuImage* const image : changed) {
        const auto recorded = std::ranges::find_if(
            layouts_, [image](const auto& entry) { return entry.first.get() == image; });
        if (recorded != layouts_.end()) {
            recorded->second = layout;
        } else {
            layouts_.emplace_back(image->shared_from_this(), layout);
        }
    }
    const vk::DependencyInfo dependency{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &memory,
        .imageMemoryBarrierCount = static_cast<std::uint32_t>(changes.size()),
        .pImageMemoryBarriers = changes.data()};
    vk::CommandBuffer(command_buffer_).pipelineBarrier2(dependency, core_->dispatch());
}

VkImageLayout CommandRecorder::prepareTransfer(GpuImage& image, VkImageLayout layout) {
    const VkImageLayout current = layoutOf(image);
    if (current == layout || current == VK_IMAGE_LAYOUT_GENERAL) {
        barrier();
        return current;
    }
    barrier(&image, layout);
    return layout;
}

}  // namespace dvv::gpu
