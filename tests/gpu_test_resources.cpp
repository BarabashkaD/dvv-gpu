#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "access.h"
#include "device_core.h"
#include "dvv/base/error.h"
#include "dvv/base/lifetime.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/gpu_registry.h"
#include "dvv/gpu/resources.h"
#include "gpu_fixture.h"

using dvv::gpu::BufferDesc;
using dvv::gpu::GpuBuffer;
using dvv::gpu::GpuImage;
using dvv::gpu::ImageDesc;
using dvv::gpu::MemoryUsage;
using dvv::gpu::detail::Access;
using dvv::gpu::test::GpuTest;

namespace {

// As if a submission signalling the next timeline value were in flight; returns that value.
std::uint64_t pretendSubmission(dvv::gpu::detail::DeviceCore& core) {
    const std::uint64_t value = core.reserveSubmitValue();
    const auto lock = core.lockQueue();
    core.setSubmitted(value, lock);
    return value;
}

void signalTimeline(dvv::gpu::detail::DeviceCore& core, std::uint64_t value) {
    const vk::SemaphoreSignalInfo signal{.semaphore = core.timeline(), .value = value};
    REQUIRE(core.device().signalSemaphore(signal, core.dispatch()) == vk::Result::eSuccess);
}

}  // namespace

TEST_CASE_FIXTURE(GpuTest, "a host-visible buffer is mapped and has a device address") {
    const auto buffer = context()
                            .createBuffer({.size = 1000 * sizeof(std::uint32_t),
                                           .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                           .memory = MemoryUsage::HostReadback,
                                           .name = "test values"})
                            .value();
    CHECK(buffer->size() == 4000);
    CHECK(buffer->desc().name == "test values");
    CHECK(buffer->handle() != VK_NULL_HANDLE);
    CHECK(buffer->deviceAddress() != 0);
    CHECK(buffer->deviceId() == device());
    CHECK(buffer->generation() == dvv::gpu::FIRST_GENERATION);
    CHECK(buffer->ownerId() == context().objectId());
    CHECK(buffer->typeName() == "dvv::gpu::GpuBuffer");

    const std::span<std::byte> mapped = buffer->mapped();
    REQUIRE(mapped.size() == 4000);
    std::vector<std::uint32_t> values(1000);
    for (std::uint32_t i = 0; i < values.size(); ++i) {
        values[i] = i * 3;
    }
    std::memcpy(mapped.data(), values.data(), mapped.size());
    REQUIRE(buffer->flush().has_value());
    REQUIRE(buffer->invalidate().has_value());
    CHECK(std::memcmp(mapped.data(), values.data(), mapped.size()) == 0);
}

TEST_CASE_FIXTURE(GpuTest, "an upload buffer is mapped, a device-only buffer is not") {
    const auto upload = context()
                            .createBuffer({.size = 64,
                                           .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                           .memory = MemoryUsage::HostUpload})
                            .value();
    CHECK(upload->mapped().size() == 64);

    const auto device_only = context()
                                 .createBuffer({.size = 64,
                                                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                .memory = MemoryUsage::DeviceOnly})
                                 .value();
    CHECK(device_only->mapped().empty());
    CHECK(device_only->deviceAddress() != 0);
    CHECK(device_only->deviceAddress() != upload->deviceAddress());
}

TEST_CASE_FIXTURE(GpuTest, "a buffer without a size is a programming error") {
    const auto buffer = context().createBuffer({.size = 0, .name = "empty"});
    REQUIRE_FALSE(buffer.has_value());
    CHECK(buffer.error().code == dvv::Errc::Internal);
    CHECK(buffer.error().message == "buffer 'empty': the size is zero");
}

TEST_CASE_FIXTURE(GpuTest, "an image is created undefined and tracks its layout") {
    const auto image =
        context()
            .createImage({.width = 33,
                          .height = 17,
                          .format = VK_FORMAT_R32_UINT,
                          .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                          .name = "test image"})
            .value();
    CHECK(image->width() == 33);
    CHECK(image->height() == 17);
    CHECK(image->format() == VK_FORMAT_R32_UINT);
    CHECK(image->handle() != VK_NULL_HANDLE);
    CHECK(image->deviceId() == device());
    CHECK(image->generation() == dvv::gpu::FIRST_GENERATION);
    CHECK(image->ownerId() == context().objectId());
    CHECK(image->layout() == VK_IMAGE_LAYOUT_UNDEFINED);

    Access::setLayout(*image, VK_IMAGE_LAYOUT_GENERAL);
    CHECK(image->layout() == VK_IMAGE_LAYOUT_GENERAL);
}

TEST_CASE_FIXTURE(GpuTest, "images the device cannot create are reported") {
    const auto unset = context().createImage({.width = 4, .height = 4, .name = "unset"});
    REQUIRE_FALSE(unset.has_value());
    CHECK(unset.error().code == dvv::Errc::Internal);

    // Block-compressed formats are never storage images.
    const auto compressed = context().createImage({.width = 4,
                                                   .height = 4,
                                                   .format = VK_FORMAT_BC1_RGB_UNORM_BLOCK,
                                                   .usage = VK_IMAGE_USAGE_STORAGE_BIT});
    REQUIRE_FALSE(compressed.has_value());
    CHECK(compressed.error().code == dvv::Errc::Unsupported);

    const auto huge = context().createImage({.width = 1U << 20U,
                                             .height = 1,
                                             .format = VK_FORMAT_R8G8B8A8_UNORM,
                                             .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT});
    REQUIRE_FALSE(huge.has_value());
    CHECK(huge.error().code == dvv::Errc::Unsupported);
}

TEST_CASE_FIXTURE(GpuTest, "a resource the GPU never used is freed when it is destroyed") {
    auto buffer =
        context().createBuffer({.size = 256, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}).value();
    auto image = context()
                     .createImage({.width = 8,
                                   .height = 8,
                                   .format = VK_FORMAT_R8G8B8A8_UNORM,
                                   .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT})
                     .value();
    const auto live_before = dvv::LifetimeRegistry::instance().liveCount();
    buffer.reset();
    image.reset();
    CHECK(dvv::LifetimeRegistry::instance().liveCount() == live_before - 2);
    const auto stats = context().retireStats();
    CHECK(stats.retiring == 0);
    CHECK(stats.retired == 2);
    CHECK(stats.freed == 2);
}

TEST_CASE_FIXTURE(GpuTest, "a retired resource waits for the submission that may use it") {
    auto& core = Access::core(context());
    auto buffer =
        context().createBuffer({.size = 256, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}).value();
    auto image = context()
                     .createImage({.width = 8,
                                   .height = 8,
                                   .format = VK_FORMAT_R8G8B8A8_UNORM,
                                   .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT})
                     .value();
    const std::uint64_t value = pretendSubmission(core);

    buffer.reset();
    image.reset();
    CHECK(context().retireStats().retiring == 2);
    CHECK(core.collect() == 0);

    signalTimeline(core, value);
    CHECK(core.collect() == 2);
    CHECK(context().retireStats().retiring == 0);
}

TEST_CASE_FIXTURE(GpuTest, "a resource keeps the device after its context is gone") {
    auto own = dvv::gpu::GpuRegistry::create().value();
    const std::weak_ptr<dvv::gpu::GpuRegistry> watch = own;
    std::shared_ptr<dvv::gpu::ExecutionContext> ctx = own->createContext(device()).value();
    auto buffer = ctx->createBuffer({.size = 1024,
                                     .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                     .memory = MemoryUsage::HostUpload})
                      .value();
    own.reset();
    ctx.reset();
    // Still usable: the buffer keeps the device, and the device keeps the registry.
    CHECK_FALSE(watch.expired());
    std::ranges::fill(buffer->mapped(), std::byte{0x5a});
    CHECK(buffer->flush().has_value());
    auto kept = watch.lock();
    REQUIRE(kept);
    buffer.reset();  // frees the buffer, then the device; `kept` holds the registry
    // Counted after the device is gone, so what destroying it reported counts too.
    CHECK(kept->validationMessageCount() == 0);
    CHECK(kept.use_count() == 1);
    kept.reset();
    CHECK(watch.expired());
}

TEST_CASE_FIXTURE(GpuTest, "buffers are created and retired from several threads") {
    constexpr int THREADS = 4;
    constexpr int BUFFERS_PER_THREAD = 50;
    constexpr std::uint64_t BUFFERS = std::uint64_t{THREADS} * BUFFERS_PER_THREAD;
    dvv::gpu::ExecutionContext& ctx = context();
    auto& core = Access::core(ctx);
    const auto before = ctx.retireStats();
    std::atomic<int> running{THREADS};
    std::atomic<int> failures{0};
    {
        std::vector<std::jthread> threads;
        threads.reserve(THREADS);
        for (int thread = 0; thread < THREADS; ++thread) {
            threads.emplace_back([&] {
                for (int i = 0; i < BUFFERS_PER_THREAD; ++i) {
                    const auto buffer = ctx.createBuffer(
                        {.size = 256, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT});
                    if (!buffer) {
                        failures.fetch_add(1);
                    }
                }
                running.fetch_sub(1);
            });
        }
        while (running.load() > 0) {
            static_cast<void>(core.collect());
            if (!ctx.drain()) {
                failures.fetch_add(1);
            }
        }
    }  // joins the threads
    CHECK(failures.load() == 0);
    const auto after = ctx.retireStats();
    CHECK(after.retired - before.retired == BUFFERS);
    CHECK(after.freed - before.freed == BUFFERS);
    CHECK(after.retiring == 0);
}
