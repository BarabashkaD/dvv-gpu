// Images in shaders (M2 design D6): storage images bound through push descriptors.
#include <doctest/doctest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "access.h"
#include "device_core.h"
#include "dvv/base/error.h"
#include "dvv/gpu/command_recorder.h"
#include "dvv/gpu/compute_pipeline.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/resources.h"
#include "gpu_fixture.h"
#include "test_shaders/add_one.h"
#include "test_shaders/write_image.h"
#include "test_shaders/write_two_images.h"

using dvv::gpu::BufferDesc;
using dvv::gpu::CommandRecorder;
using dvv::gpu::ComputeBindings;
using dvv::gpu::GpuBuffer;
using dvv::gpu::GpuImage;
using dvv::gpu::ImageDesc;
using dvv::gpu::MemoryUsage;
using dvv::gpu::detail::Access;
using dvv::gpu::test::GpuTest;

namespace {

constexpr std::uint32_t GROUP_SIZE = 8;  // [numthreads(8, 8, 1)] in write_image.slang

// The push constants of write_image and write_two_images.
struct Push {
    std::uint32_t w = 0;
    std::uint32_t h = 0;
};
static_assert(sizeof(Push) == 8);

constexpr std::uint32_t ADD_ONE_PUSH_SIZE = 16;  // add_one: buffer address, count, padding

std::uint32_t groups(std::uint32_t size) {
    return (size + GROUP_SIZE - 1) / GROUP_SIZE;
}

ImageDesc storageImage(std::uint32_t width, std::uint32_t height, VkFormat format,
                       std::string name) {
    return {.width = width,
            .height = height,
            .format = format,
            .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            .name = std::move(name)};
}

BufferDesc readback(std::uint64_t size, std::string name) {
    return {.size = size,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .memory = MemoryUsage::HostReadback,
            .name = std::move(name)};
}

template <class T>
std::vector<T> read(GpuBuffer& buffer) {
    REQUIRE(buffer.invalidate().has_value());
    std::vector<T> values(buffer.size() / sizeof(T));
    std::memcpy(values.data(), buffer.mapped().data(), values.size() * sizeof(T));
    return values;
}

// The number of texels of a width x height image whose value is not wanted(x, y).
template <class T, class Wanted>
std::size_t mismatches(const std::vector<T>& values, std::uint32_t width, std::uint32_t height,
                       Wanted wanted) {
    REQUIRE(values.size() == std::size_t{width} * height);
    std::size_t count = 0;
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            count += values[(std::size_t{y} * width) + x] != wanted(x, y) ? 1 : 0;
        }
    }
    return count;
}

}  // namespace

TEST_CASE_FIXTURE(GpuTest, "a compute shader writes a storage image through a push descriptor") {
    constexpr std::uint32_t WIDTH = 37;
    constexpr std::uint32_t HEIGHT = 23;
    const auto pipeline = context()
                              .createComputePipeline(test_shaders::writeImage(), sizeof(Push),
                                                     ComputeBindings{.storageImages = 1})
                              .value();
    CHECK(pipeline->bindings() == ComputeBindings{.storageImages = 1});
    CHECK(pipeline->setLayout() != VK_NULL_HANDLE);
    const auto image =
        context().createImage(storageImage(WIDTH, HEIGHT, VK_FORMAT_R16_UINT, "written")).value();
    const auto buffer =
        context()
            .createBuffer(readback(std::uint64_t{WIDTH} * HEIGHT * 2, "written texels"))
            .value();
    const Push push{.w = WIDTH, .h = HEIGHT};

    GpuImage* raw = image.get();
    REQUIRE(submitAndWait([&](CommandRecorder& recorder) {
                recorder.dispatch(*pipeline, std::as_bytes(std::span(&push, 1)), {&raw, 1},
                                  groups(WIDTH), groups(HEIGHT));
                // A copy would read the image in GENERAL; the explicit move lets the next dispatch
                // bring it back to GENERAL.
                recorder.transition(*image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
                recorder.copyImageToBuffer(*image, *buffer);
            }).has_value());
    CHECK(mismatches(read<std::uint16_t>(*buffer), WIDTH, HEIGHT,
                     [](std::uint32_t x, std::uint32_t y) {
                         return static_cast<std::uint16_t>(x + (y * WIDTH));
                     }) == 0);
    CHECK(image->layout() == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    REQUIRE(submitAndWait([&](CommandRecorder& recorder) {
                recorder.dispatch(*pipeline, std::as_bytes(std::span(&push, 1)), {&raw, 1},
                                  groups(WIDTH), groups(HEIGHT));
            }).has_value());
    CHECK(image->layout() == VK_IMAGE_LAYOUT_GENERAL);
}

TEST_CASE_FIXTURE(GpuTest, "two storage images in one dispatch") {
    constexpr std::uint32_t SIZE = 16;
    const auto pipeline =
        context()
            .createComputePipeline(test_shaders::writeTwoImages(), sizeof(Push), ComputeBindings{2})
            .value();
    const auto first =
        context().createImage(storageImage(SIZE, SIZE, VK_FORMAT_R8_UINT, "first")).value();
    const auto second =
        context().createImage(storageImage(SIZE, SIZE, VK_FORMAT_R16_UINT, "second")).value();
    const auto first_texels =
        context().createBuffer(readback(std::uint64_t{SIZE} * SIZE, "first texels")).value();
    const auto second_texels =
        context().createBuffer(readback(std::uint64_t{SIZE} * SIZE * 2, "second texels")).value();
    const Push push{.w = SIZE, .h = SIZE};

    const std::array<GpuImage*, 2> images{first.get(), second.get()};
    REQUIRE(submitAndWait([&](CommandRecorder& recorder) {
                recorder.dispatch(*pipeline, std::as_bytes(std::span(&push, 1)), images,
                                  groups(SIZE), groups(SIZE));
                recorder.copyImageToBuffer(*first, *first_texels);
                recorder.copyImageToBuffer(*second, *second_texels);
            }).has_value());
    CHECK(mismatches(read<std::uint8_t>(*first_texels), SIZE, SIZE,
                     [](std::uint32_t x, std::uint32_t y) {
                         return static_cast<std::uint8_t>((x + (y * SIZE)) & 0xFFU);
                     }) == 0);
    CHECK(mismatches(read<std::uint16_t>(*second_texels), SIZE, SIZE,
                     [](std::uint32_t x, std::uint32_t y) {
                         return static_cast<std::uint16_t>(x + (y * SIZE) + 1);
                     }) == 0);
}

TEST_CASE_FIXTURE(GpuTest, "an image appears once per dispatch") {
    const auto pipeline =
        context()
            .createComputePipeline(test_shaders::writeTwoImages(), sizeof(Push), ComputeBindings{2})
            .value();
    const auto a = context().createImage(storageImage(8, 8, VK_FORMAT_R16_UINT, "a")).value();
    const Push push{.w = 8, .h = 8};

    const std::array<GpuImage*, 2> both{a.get(), a.get()};
    const auto twice = submitAndWait([&](CommandRecorder& recorder) {
        recorder.dispatch(*pipeline, std::as_bytes(std::span(&push, 1)), both, 1, 1);
    });
    REQUIRE_FALSE(twice.has_value());
    CHECK(twice.error().code == dvv::Errc::Internal);
    CHECK(twice.error().message.find("appears twice") != std::string::npos);
}

TEST_CASE_FIXTURE(GpuTest, "dispatch binds exactly the pipeline's storage images") {
    const auto write_image = context()
                                 .createComputePipeline(test_shaders::writeImage(), sizeof(Push),
                                                        ComputeBindings{.storageImages = 1})
                                 .value();
    const auto add_one =
        context().createComputePipeline(test_shaders::addOne(), ADD_ONE_PUSH_SIZE).value();
    CHECK(add_one->bindings() == ComputeBindings{});
    CHECK(add_one->setLayout() == VK_NULL_HANDLE);
    const auto image = context().createImage(storageImage(8, 8, VK_FORMAT_R16_UINT, "one")).value();
    const Push push{.w = 8, .h = 8};
    const std::array<std::byte, ADD_ONE_PUSH_SIZE> add_one_push{};

    const auto none = submitAndWait([&](CommandRecorder& recorder) {
        recorder.dispatch(*write_image, std::as_bytes(std::span(&push, 1)), 1);
    });
    REQUIRE_FALSE(none.has_value());
    CHECK(none.error().code == dvv::Errc::Internal);
    CHECK(none.error().message.find("binds 1 storage images, not 0") != std::string::npos);

    GpuImage* raw = image.get();
    const auto extra = submitAndWait([&](CommandRecorder& recorder) {
        recorder.dispatch(*add_one, add_one_push, {&raw, 1}, 1);
    });
    REQUIRE_FALSE(extra.has_value());
    CHECK(extra.error().code == dvv::Errc::Internal);
    CHECK(extra.error().message.find("binds 0 storage images, not 1") != std::string::npos);
}

TEST_CASE_FIXTURE(GpuTest, "a storage image needs STORAGE usage") {
    const auto pipeline = context()
                              .createComputePipeline(test_shaders::writeImage(), sizeof(Push),
                                                     ComputeBindings{.storageImages = 1})
                              .value();
    const auto image = context()
                           .createImage({.width = 8,
                                         .height = 8,
                                         .format = VK_FORMAT_R16_UINT,
                                         .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                         .name = "no storage"})
                           .value();
    const Push push{.w = 8, .h = 8};

    GpuImage* raw = image.get();
    const auto result = submitAndWait([&](CommandRecorder& recorder) {
        recorder.dispatch(*pipeline, std::as_bytes(std::span(&push, 1)), {&raw, 1}, 1, 1);
    });
    REQUIRE_FALSE(result.has_value());
    CHECK(result.error().code == dvv::Errc::Internal);
    CHECK(result.error().message.find("lacks the usage") != std::string::npos);
}

TEST_CASE_FIXTURE(GpuTest, "view() exists for STORAGE and SAMPLED images only") {
    const auto transfer = context()
                              .createImage({.width = 8,
                                            .height = 8,
                                            .format = VK_FORMAT_R8G8B8A8_UNORM,
                                            .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT})
                              .value();
    CHECK(transfer->view() == VK_NULL_HANDLE);
    const auto sampled = context()
                             .createImage({.width = 8,
                                           .height = 8,
                                           .format = VK_FORMAT_R8G8B8A8_UNORM,
                                           .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
                                           .name = "sampled"})
                             .value();
    CHECK(sampled->view() != VK_NULL_HANDLE);
    const auto storage = context()
                             .createImage({.width = 8,
                                           .height = 8,
                                           .format = VK_FORMAT_R16_UINT,
                                           .usage = VK_IMAGE_USAGE_STORAGE_BIT,
                                           .name = "storage"})
                             .value();
    CHECK(storage->view() != VK_NULL_HANDLE);
}

TEST_CASE_FIXTURE(GpuTest, "pipeline bindings must fit the device and the shader") {
    const std::uint32_t max = Access::core(context()).maxPushDescriptors();
    CHECK(max >= 32);

    const auto too_many = context().createComputePipeline(test_shaders::writeImage(), sizeof(Push),
                                                          ComputeBindings{max + 1});
    REQUIRE_FALSE(too_many.has_value());
    CHECK(too_many.error().code == dvv::Errc::Internal);
    CHECK(too_many.error().message.find("maxPushDescriptors") != std::string::npos);

    for (const std::uint32_t count : {0U, 2U}) {
        INFO("storage images: " << count);
        const auto wrong = context().createComputePipeline(test_shaders::writeImage(), sizeof(Push),
                                                           ComputeBindings{count});
        REQUIRE_FALSE(wrong.has_value());
        CHECK(wrong.error().code == dvv::Errc::Internal);
        CHECK(wrong.error().message.find("declares descriptor bindings") != std::string::npos);
    }
    const auto add_one = context().createComputePipeline(test_shaders::addOne(), ADD_ONE_PUSH_SIZE,
                                                         ComputeBindings{1});
    REQUIRE_FALSE(add_one.has_value());
    CHECK(add_one.error().code == dvv::Errc::Internal);
    CHECK(add_one.error().message.find("declares descriptor bindings") != std::string::npos);
}

TEST_CASE_FIXTURE(GpuTest, "views and set layouts are retired and freed") {
    const std::uint64_t freed_before = context().retireStats().freed;
    {
        const auto image =
            context().createImage(storageImage(8, 8, VK_FORMAT_R16_UINT, "retired")).value();
        const auto pipeline = context()
                                  .createComputePipeline(test_shaders::writeImage(), sizeof(Push),
                                                         ComputeBindings{.storageImages = 1})
                                  .value();
        CHECK(image->view() != VK_NULL_HANDLE);
        CHECK(pipeline->setLayout() != VK_NULL_HANDLE);
    }
    REQUIRE(context().drain().has_value());
    const auto stats = context().retireStats();
    // The image, its view, the pipeline, its layout and its set layout.
    CHECK(stats.freed == freed_before + 5);
    CHECK(stats.retiring == 0);
}

TEST_CASE_FIXTURE(GpuTest,
                  "formatFeatures reports storage writes without format for R8_UINT and R16_UINT") {
    for (const VkFormat format : {VK_FORMAT_R8_UINT, VK_FORMAT_R16_UINT}) {
        INFO("format: " << static_cast<int>(format));
        const VkFormatFeatureFlags2 features = context().formatFeatures(format);
        CHECK((features & VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT) != 0);
        CHECK((features & VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT) != 0);
    }
}

TEST_CASE_FIXTURE(GpuTest, "a storage image of a format without storage support is Unsupported") {
    VkFormat chosen = VK_FORMAT_UNDEFINED;
    for (const VkFormat format :
         {VK_FORMAT_R8G8B8_UNORM, VK_FORMAT_D32_SFLOAT, VK_FORMAT_D16_UNORM}) {
        if ((context().formatFeatures(format) & VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT) == 0) {
            chosen = format;
            break;
        }
    }
    if (chosen == VK_FORMAT_UNDEFINED) {
        MESSAGE("every candidate format supports storage images: nothing to refuse");
        return;
    }
    const auto image = context().createImage({.width = 8,
                                              .height = 8,
                                              .format = chosen,
                                              .usage = VK_IMAGE_USAGE_STORAGE_BIT,
                                              .name = "no storage format"});
    REQUIRE_FALSE(image.has_value());
    CHECK(image.error().code == dvv::Errc::Unsupported);
    CHECK(image.error().message.find("no storage images") != std::string::npos);
}

// M4 Task 11-pre (coverage review gap #3): a frame larger than the device's image limit (for
// example a 20000x8 source) is refused by the maxExtent check before any Vulkan object is created,
// so no validation message is written (the fixture checks the count). The limit is the one the
// product checks, the format's maxExtent for this usage (Task 11-pre review M4): Vulkan only
// requires maxExtent >= maxImageDimension2D, so maxImageDimension2D + 1 can be a valid image.
TEST_CASE_FIXTURE(GpuTest, "an image larger than the format's maxExtent is Unsupported") {
    const dvv::gpu::detail::DeviceCore& core = Access::core(context());
    const ImageDesc probe = storageImage(8, 8, VK_FORMAT_R16_UINT, "probe");
    const auto supported = core.physicalDevice().getImageFormatProperties2(
        {.format = static_cast<vk::Format>(probe.format),
         .type = vk::ImageType::e2D,
         .tiling = vk::ImageTiling::eOptimal,
         .usage = vk::ImageUsageFlags(probe.usage)},
        core.dispatch());
    REQUIRE(supported.result == vk::Result::eSuccess);
    const vk::Extent3D& max = supported.value.imageFormatProperties.maxExtent;
    CAPTURE(max.width);
    CAPTURE(max.height);
    REQUIRE(max.width > 0);
    REQUIRE(max.height > 0);
    REQUIRE(max.width < UINT32_MAX);
    REQUIRE(max.height < UINT32_MAX);
    const std::array<std::pair<std::uint32_t, std::uint32_t>, 2> sizes{{
        {max.width + 1, 8},
        {8, max.height + 1},
    }};
    for (const auto& [width, height] : sizes) {
        CAPTURE(width);
        CAPTURE(height);
        const auto image =
            context().createImage(storageImage(width, height, VK_FORMAT_R16_UINT, "too large"));
        REQUIRE_FALSE(image.has_value());
        CHECK(image.error().code == dvv::Errc::Unsupported);
        CAPTURE(image.error().message);
        CHECK(image.error().message.find("exceeds the largest") != std::string::npos);
    }
    // The context stays usable after the refusals.
    CHECK(context().createImage(storageImage(8, 8, VK_FORMAT_R16_UINT, "after")).has_value());
}
