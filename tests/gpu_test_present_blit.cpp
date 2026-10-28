#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <vk_mem_alloc.h>

#include "access.h"
#include "device_core.h"
#include "dvv/base/error.h"
#include "dvv/base/executors.h"
#include "dvv/gpu/command_recorder.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/gpu_registry.h"
#include "dvv/gpu/resources.h"
#include "dvv/platform/native_window.h"
#include "gpu_fixture.h"

using dvv::gpu::CommandRecorder;
using dvv::gpu::ExecutionContext;
using dvv::gpu::GpuImage;
using dvv::gpu::GpuRegistry;
using dvv::gpu::MemoryUsage;
using dvv::gpu::detail::Access;

namespace {

constexpr std::uint32_t RGBA = 4;

// A registry with the platform's window system and a presentation context on the best device;
// validation comes from the environment, as in GpuTest. At the end the context goes first, then
// no validation message may have been counted.
class PresentBlitTest {
public:
    PresentBlitTest() {
        auto registry = GpuRegistry::create({.windowSystem = dvv::platform::defaultWindowSystem()});
        if (!registry) {
            FAIL(registry.error().describe());
        }
        registry_ = std::move(*registry);
        context_ = presentationContext();
        executor_ = dvv::ExecutorFactory::makeThreadPool(1).value();
    }
    PresentBlitTest(const PresentBlitTest&) = delete;
    PresentBlitTest& operator=(const PresentBlitTest&) = delete;
    PresentBlitTest(PresentBlitTest&&) = delete;
    PresentBlitTest& operator=(PresentBlitTest&&) = delete;
    ~PresentBlitTest() {
        context_.reset();
        executor_.reset();
        CHECK(registry_->validationMessageCount() == 0);
    }

    // Another presentation context on the same device.
    [[nodiscard]] std::shared_ptr<ExecutionContext> presentationContext() {
        const auto best = registry_->bestDevice();
        if (!best) {
            FAIL("no usable Vulkan device");
        }
        auto context =
            registry_->createContext(best.value_or(dvv::gpu::DeviceId(0)), {.presentation = true});
        if (!context) {
            FAIL(context.error().describe());
        }
        return std::move(*context);
    }

    [[nodiscard]] ExecutionContext& context() const noexcept { return *context_; }
    [[nodiscard]] dvv::gpu::detail::DeviceCore& core() const noexcept {
        return Access::core(*context_);
    }
    [[nodiscard]] dvv::Result<std::uint64_t> submitAndWait(dvv::gpu::RecordFunction record) {
        return dvv::gpu::test::submitAndWait(*context_, *executor_, std::move(record));
    }

private:
    std::shared_ptr<GpuRegistry> registry_;
    std::shared_ptr<ExecutionContext> context_;
    std::shared_ptr<dvv::ThreadPool> executor_;
};

// A stand-in for a swapchain image: a B8G8R8A8_UNORM image made directly through VMA.
class RawImage {
public:
    RawImage(dvv::gpu::detail::DeviceCore& core, VkExtent2D extent) : core_(&core) {
        const vk::ImageCreateInfo info{
            .imageType = vk::ImageType::e2D,
            .format = vk::Format::eB8G8R8A8Unorm,
            .extent = {.width = extent.width, .height = extent.height, .depth = 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined};
        VmaAllocationCreateInfo allocation_info{};
        allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
        REQUIRE(vmaCreateImage(core.allocator(), &static_cast<const VkImageCreateInfo&>(info),
                               &allocation_info, &image_, &allocation_, nullptr) == VK_SUCCESS);
    }
    RawImage(const RawImage&) = delete;
    RawImage& operator=(const RawImage&) = delete;
    RawImage(RawImage&&) = delete;
    RawImage& operator=(RawImage&&) = delete;
    // The submissions that used the image have completed (submitAndWait).
    ~RawImage() { vmaDestroyImage(core_->allocator(), image_, allocation_); }

    [[nodiscard]] VkImage handle() const noexcept { return image_; }

private:
    dvv::gpu::detail::DeviceCore* core_;
    VkImage image_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = nullptr;
};

struct Blitted {
    std::vector<std::uint8_t> bgra;    // the target image read back, BGRA8
    std::shared_ptr<GpuImage> source;  // the client image after the blit
};

// Uploads `rgba` into an R8G8B8A8_UNORM image of the source extent, blits it with presentBlit into
// a B8G8R8A8_UNORM image of the target extent and reads that back.
Blitted blitAndRead(PresentBlitTest& test, std::span<const std::uint8_t> rgba, VkExtent2D source,
                    VkExtent2D target) {
    ExecutionContext& ctx = test.context();
    REQUIRE(rgba.size() == std::size_t{source.width} * source.height * RGBA);
    const auto upload = ctx.createBuffer({.size = rgba.size(),
                                          .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                          .memory = MemoryUsage::HostUpload,
                                          .name = "upload"})
                            .value();
    std::memcpy(upload->mapped().data(), rgba.data(), rgba.size());
    REQUIRE(upload->flush().has_value());
    auto image =
        ctx.createImage({.width = source.width,
                         .height = source.height,
                         .format = VK_FORMAT_R8G8B8A8_UNORM,
                         .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                         .name = "client frame"})
            .value();
    const RawImage swapchain_image(test.core(), target);
    const std::uint64_t target_bytes = std::uint64_t{target.width} * target.height * RGBA;
    const auto readback = ctx.createBuffer({.size = target_bytes,
                                            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                            .memory = MemoryUsage::HostReadback,
                                            .name = "readback"})
                              .value();

    const auto blit = [&](CommandRecorder& recorder) {
        recorder.copyBufferToImage(*upload, *image);
        Access::presentBlit(recorder, *image, swapchain_image.handle(), target);
    };
    REQUIRE(test.submitAndWait(blit).has_value());

    // The second submission starts after the first one completed; its barrier takes the image
    // out of PRESENT_SRC_KHR for the copy.
    const auto& dispatch = test.core().dispatch();
    const auto read = [&](CommandRecorder& recorder) {
        const dvv::gpu::RawCommands raw = recorder.raw();
        if (raw.commandBuffer == VK_NULL_HANDLE) {
            return;
        }
        const vk::CommandBuffer commands(raw.commandBuffer);
        const vk::ImageMemoryBarrier2 to_read{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
            .oldLayout = vk::ImageLayout::ePresentSrcKHR,
            .newLayout = vk::ImageLayout::eTransferSrcOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = vk::Image(swapchain_image.handle()),
            .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                                 .baseMipLevel = 0,
                                 .levelCount = 1,
                                 .baseArrayLayer = 0,
                                 .layerCount = 1}};
        commands.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_read},
                                  dispatch);
        const vk::BufferImageCopy region{
            .bufferOffset = 0,
            .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                                 .mipLevel = 0,
                                 .baseArrayLayer = 0,
                                 .layerCount = 1},
            .imageExtent = {.width = target.width, .height = target.height, .depth = 1}};
        commands.copyImageToBuffer(vk::Image(swapchain_image.handle()),
                                   vk::ImageLayout::eTransferSrcOptimal,
                                   vk::Buffer(readback->handle()), region, dispatch);
    };
    REQUIRE(test.submitAndWait(read).has_value());

    REQUIRE(readback->invalidate().has_value());
    const auto bytes = readback->mapped();
    Blitted result{.bgra = std::vector<std::uint8_t>(target_bytes), .source = std::move(image)};
    std::memcpy(result.bgra.data(), bytes.data(), result.bgra.size());
    return result;
}

// r = x, g = y, b = (x * 7 + y) & 0xFF, a = 255.
std::vector<std::uint8_t> gradient(VkExtent2D extent) {
    std::vector<std::uint8_t> rgba;
    rgba.reserve(std::size_t{extent.width} * extent.height * RGBA);
    for (std::uint32_t y = 0; y < extent.height; ++y) {
        for (std::uint32_t x = 0; x < extent.width; ++x) {
            rgba.push_back(static_cast<std::uint8_t>(x));
            rgba.push_back(static_cast<std::uint8_t>(y));
            rgba.push_back(static_cast<std::uint8_t>((x * 7 + y) & 0xFFU));
            rgba.push_back(255);
        }
    }
    return rgba;
}

}  // namespace

TEST_CASE_FIXTURE(PresentBlitTest, "presentBlit copies a same-size image exactly") {
    constexpr VkExtent2D EXTENT{.width = 33, .height = 17};
    const std::vector<std::uint8_t> rgba = gradient(EXTENT);
    const Blitted blitted = blitAndRead(*this, rgba, EXTENT, EXTENT);
    REQUIRE(blitted.bgra.size() == rgba.size());
    std::size_t mismatches = 0;
    for (std::size_t pixel = 0; pixel < rgba.size(); pixel += RGBA) {
        // The blit converts RGBA to BGRA.
        if (blitted.bgra[pixel + 0] != rgba[pixel + 2] ||
            blitted.bgra[pixel + 1] != rgba[pixel + 1] ||
            blitted.bgra[pixel + 2] != rgba[pixel + 0] || blitted.bgra[pixel + 3] != 255) {
            ++mismatches;
        }
    }
    CHECK(mismatches == 0);
}

TEST_CASE_FIXTURE(PresentBlitTest, "presentBlit scales with linear filtering") {
    const VkFormatFeatureFlags2 rgba_features = core().formatFeatures(VK_FORMAT_R8G8B8A8_UNORM);
    // Mandatory for this format.
    REQUIRE((rgba_features & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0);
    CHECK(core().formatFeatures(VK_FORMAT_R8G8B8A8_UNORM) == rgba_features);  // stable across calls
    CHECK((core().formatFeatures(VK_FORMAT_B8G8R8A8_UNORM) & VK_FORMAT_FEATURE_2_BLIT_DST_BIT) !=
          0);

    const std::vector<std::uint8_t> black_white{0, 0, 0, 255, 255, 255, 255, 255};
    const Blitted blitted =
        blitAndRead(*this, black_white, {.width = 2, .height = 1}, {.width = 4, .height = 1});
    REQUIRE(blitted.bgra.size() == 16);
    // The red channel (byte 2 of each BGRA pixel) at u = 0.25, 0.75, 1.25, 1.75 source texels.
    const std::uint8_t p0 = blitted.bgra[2];
    const std::uint8_t p1 = blitted.bgra[6];
    const std::uint8_t p2 = blitted.bgra[10];
    const std::uint8_t p3 = blitted.bgra[14];
    CHECK(p0 <= 4);
    CHECK(p1 >= 48);
    CHECK(p1 <= 80);
    CHECK(p2 >= 175);
    CHECK(p2 <= 207);
    CHECK(p3 >= 251);
}

TEST_CASE_FIXTURE(PresentBlitTest, "presentBlit leaves the source in TRANSFER_SRC_OPTIMAL") {
    constexpr VkExtent2D EXTENT{.width = 33, .height = 17};
    const Blitted blitted = blitAndRead(*this, gradient(EXTENT), EXTENT, EXTENT);
    CHECK(blitted.source->layout() == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
}

TEST_CASE_FIXTURE(PresentBlitTest, "presentBlit refuses an image without TRANSFER_SRC") {
    constexpr VkExtent2D EXTENT{.width = 8, .height = 8};
    const auto image =
        context()
            .createImage({.width = EXTENT.width,
                          .height = EXTENT.height,
                          .format = VK_FORMAT_R8G8B8A8_UNORM,
                          .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                          .name = "no source"})
            .value();
    const RawImage swapchain_image(core(), EXTENT);
    const auto result = submitAndWait([&](CommandRecorder& recorder) {
        Access::presentBlit(recorder, *image, swapchain_image.handle(), EXTENT);
    });
    REQUIRE_FALSE(result.has_value());
    CHECK(result.error().code == dvv::Errc::Internal);
    CHECK(result.error().message.find("presentBlit") != std::string::npos);
    CHECK(result.error().message.find("lacks the usage") != std::string::npos);
}

TEST_CASE_FIXTURE(PresentBlitTest, "presentBlit refuses an image of another device") {
    constexpr VkExtent2D EXTENT{.width = 8, .height = 8};
    const auto other = presentationContext();
    const auto image = other
                           ->createImage({.width = EXTENT.width,
                                          .height = EXTENT.height,
                                          .format = VK_FORMAT_R8G8B8A8_UNORM,
                                          .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                                   VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                          .name = "other device"})
                           .value();
    const RawImage swapchain_image(core(), EXTENT);
    const auto result = submitAndWait([&](CommandRecorder& recorder) {
        Access::presentBlit(recorder, *image, swapchain_image.handle(), EXTENT);
    });
    REQUIRE_FALSE(result.has_value());
    CHECK(result.error().code == dvv::Errc::Internal);
    CHECK(result.error().message.find("belongs to another device") != std::string::npos);
}

TEST_CASE_FIXTURE(PresentBlitTest,
                  "presentBlit refuses an integer format, a null swapchain image and an empty "
                  "extent") {
    constexpr VkExtent2D EXTENT{.width = 8, .height = 8};
    const auto make_source = [&](VkFormat format) {
        return context()
            .createImage({.width = EXTENT.width,
                          .height = EXTENT.height,
                          .format = format,
                          .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                          .name = "refused source"})
            .value();
    };
    const auto refused = [&](GpuImage& source, VkImage target, VkExtent2D extent,
                             const char* problem) {
        const auto result = submitAndWait([&](CommandRecorder& recorder) {
            Access::presentBlit(recorder, source, target, extent);
        });
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().code == dvv::Errc::Internal);
        CHECK(result.error().message.find("presentBlit") != std::string::npos);
        CHECK(result.error().message.find(problem) != std::string::npos);
    };
    SUBCASE("an integer format") {
        const auto image = make_source(VK_FORMAT_R8G8B8A8_UINT);
        const RawImage swapchain_image(core(), EXTENT);
        refused(*image, swapchain_image.handle(), EXTENT, "cannot be blitted");
    }
    SUBCASE("a null swapchain image") {
        const auto image = make_source(VK_FORMAT_R8G8B8A8_UNORM);
        refused(*image, VK_NULL_HANDLE, EXTENT, "null or empty");
    }
    SUBCASE("an empty extent") {
        const auto image = make_source(VK_FORMAT_R8G8B8A8_UNORM);
        const RawImage swapchain_image(core(), EXTENT);
        refused(*image, swapchain_image.handle(), {.width = 0, .height = 8}, "null or empty");
    }
}
