#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include "access.h"
#include "device_core.h"
#include "dvv/base/error.h"
#include "dvv/base/lifetime.h"
#include "dvv/gpu/command_recorder.h"
#include "dvv/gpu/context_vulkan.h"
#include "dvv/gpu/embedded_shader.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/gpu_instance.h"
#include "dvv/gpu/resources.h"
#include "gpu_fixture.h"

using dvv::gpu::BufferDesc;
using dvv::gpu::CommandRecorder;
using dvv::gpu::GpuBuffer;
using dvv::gpu::GpuImage;
using dvv::gpu::ImageDesc;
using dvv::gpu::MappedRead;
using dvv::gpu::MappedWrite;
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

constexpr std::uint64_t SCOPE_BYTES = 256;

const BufferDesc UPLOAD{.size = SCOPE_BYTES,
                        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        .memory = MemoryUsage::HostUpload,
                        .name = "scope upload"};
const BufferDesc READBACK{.size = SCOPE_BYTES,
                          .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          .memory = MemoryUsage::HostReadback,
                          .name = "scope readback"};

// The bytes 0, 1, ..., SCOPE_BYTES - 1.
std::vector<std::byte> sequence() {
    std::vector<std::byte> values(SCOPE_BYTES);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<std::byte>(i);
    }
    return values;
}

void writeBytes(std::span<std::byte> target, const std::vector<std::byte>& values) {
    REQUIRE(target.size() == values.size());
    std::ranges::copy(values, target.begin());
}

bool holds(std::span<const std::byte> bytes, const std::vector<std::byte>& values) {
    return std::ranges::equal(bytes, values);
}

// Short calls, so that a line using a moved-from scope fits its NOLINT comment.
template <class Scope>
bool isEmpty(const Scope& scope) {
    return scope.bytes().empty();
}

bool flushes(MappedWrite& scope) {
    return scope.flush().has_value();
}

}  // namespace

TEST_CASE_FIXTURE(GpuTest, "a host-visible buffer is mapped and has a device address") {
    const auto buffer = createBuffer(context(), {.size = 1000 * sizeof(std::uint32_t),
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

    MappedWrite write = buffer->mapWrite();
    REQUIRE(write.bytes().size() == 4000);
    std::vector<std::uint32_t> values(1000);
    for (std::uint32_t i = 0; i < values.size(); ++i) {
        values[i] = i * 3;
    }
    std::memcpy(write.bytes().data(), values.data(), write.bytes().size());
    REQUIRE(write.flush().has_value());
    const MappedRead read = buffer->mapRead().value();
    REQUIRE(read.bytes().size() == 4000);
    CHECK(std::memcmp(read.bytes().data(), values.data(), read.bytes().size()) == 0);
}

TEST_CASE_FIXTURE(GpuTest, "an upload buffer is mapped, a device-only buffer is not") {
    const auto upload = createBuffer(context(), {.size = 64,
                                                 .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                 .memory = MemoryUsage::HostUpload})
                            .value();
    CHECK(upload->mapWrite().bytes().size() == 64);

    const auto device_only = createBuffer(context(), {.size = 64,
                                                      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                      .memory = MemoryUsage::DeviceOnly})
                                 .value();
    CHECK(device_only->mapWrite().bytes().empty());
    CHECK(device_only->deviceAddress() != 0);
    CHECK(device_only->deviceAddress() != upload->deviceAddress());
}

TEST_CASE_FIXTURE(GpuTest, "a buffer without a size is a programming error") {
    const auto buffer = createBuffer(context(), {.size = 0, .name = "empty"});
    REQUIRE_FALSE(buffer.has_value());
    CHECK(buffer.error().code == dvv::Errc::Internal);
    CHECK(buffer.error().message == "buffer 'empty': the size is zero");
}

TEST_CASE_FIXTURE(GpuTest, "an image is created undefined and tracks its layout") {
    const auto image = createImage(context(), {.width = 33,
                                               .height = 17,
                                               .format = VK_FORMAT_R32_UINT,
                                               .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                                        VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                                        VK_IMAGE_USAGE_STORAGE_BIT,
                                               .name = "test image"})
                           .value();
    CHECK(image->name() == "test image");
    CHECK(vulkanDesc(*image).name == "test image");
    CHECK(image->width() == 33);
    CHECK(image->height() == 17);
    CHECK(vulkanFormat(*image) == VK_FORMAT_R32_UINT);
    CHECK(vulkanHandle(*image) != VK_NULL_HANDLE);
    CHECK(image->deviceId() == device());
    CHECK(image->generation() == dvv::gpu::FIRST_GENERATION);
    CHECK(image->ownerId() == context().objectId());
    CHECK(vulkanLayout(*image) == VK_IMAGE_LAYOUT_UNDEFINED);

    Access::setLayout(*image, VK_IMAGE_LAYOUT_GENERAL);
    CHECK(vulkanLayout(*image) == VK_IMAGE_LAYOUT_GENERAL);
}

TEST_CASE_FIXTURE(GpuTest, "images the device cannot create are reported") {
    const auto unset = createImage(context(), {.width = 4, .height = 4, .name = "unset"});
    REQUIRE_FALSE(unset.has_value());
    CHECK(unset.error().code == dvv::Errc::Internal);

    // Block-compressed formats are never storage images.
    const auto compressed = createImage(context(), {.width = 4,
                                                    .height = 4,
                                                    .format = VK_FORMAT_BC1_RGB_UNORM_BLOCK,
                                                    .usage = VK_IMAGE_USAGE_STORAGE_BIT});
    REQUIRE_FALSE(compressed.has_value());
    CHECK(compressed.error().code == dvv::Errc::Unsupported);

    const auto huge = createImage(context(), {.width = 1U << 20U,
                                              .height = 1,
                                              .format = VK_FORMAT_R8G8B8A8_UNORM,
                                              .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT});
    REQUIRE_FALSE(huge.has_value());
    CHECK(huge.error().code == dvv::Errc::Unsupported);
}

TEST_CASE_FIXTURE(GpuTest, "a resource the GPU never used is freed when it is destroyed") {
    auto buffer =
        createBuffer(context(), {.size = 256, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}).value();
    auto image = createImage(context(), {.width = 8,
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
        createBuffer(context(), {.size = 256, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}).value();
    auto image = createImage(context(), {.width = 8,
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
    auto own = dvv::gpu::GpuInstance::create().value();
    const std::weak_ptr<dvv::gpu::GpuInstance> watch = own;
    std::shared_ptr<dvv::gpu::ExecutionContext> ctx = own->createContext(device()).value();
    auto buffer = createBuffer(*ctx, {.size = 1024,
                                      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                      .memory = MemoryUsage::HostUpload})
                      .value();
    own.reset();
    ctx.reset();
    // Still usable: the buffer keeps the device, and the device keeps the GPU instance.
    CHECK_FALSE(watch.expired());
    MappedWrite write = buffer->mapWrite();
    std::ranges::fill(write.bytes(), std::byte{0x5a});
    CHECK(write.flush().has_value());
    auto kept = watch.lock();
    REQUIRE(kept);
    buffer.reset();  // frees the buffer, then the device; `kept` holds the GPU instance
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
                    const auto buffer = createBuffer(
                        ctx, {.size = 256, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT});
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

// §5.13 F-15: a creator reserves the retire slots its object's destructor uses, so retiring never
// allocates (the queue's capacity stays as it was while the objects are dropped).
TEST_CASE_FIXTURE(GpuTest, "retire queue: creation reserves the slots its destructor uses") {
    constexpr int BUFFERS = 64;
    constexpr int IMAGES = 32;
    auto& core = Access::core(context());
    const std::size_t slots_before = core.retireSlots();
    std::vector<std::shared_ptr<GpuBuffer>> buffers;
    std::vector<std::shared_ptr<GpuImage>> images;
    buffers.reserve(BUFFERS);
    images.reserve(IMAGES);
    for (int index = 0; index < BUFFERS; ++index) {
        buffers.push_back(createBuffer(context(), {.size = 256,
                                                   .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                   .memory = MemoryUsage::HostUpload})
                              .value());
    }
    for (int index = 0; index < IMAGES; ++index) {
        // A storage image has a view, so it retires two handles.
        images.push_back(createImage(context(), {.width = 8,
                                                 .height = 8,
                                                 .format = VK_FORMAT_R8G8B8A8_UNORM,
                                                 .usage = VK_IMAGE_USAGE_STORAGE_BIT})
                             .value());
    }
    const auto capacity = core.retireCapacity();
    buffers.clear();
    images.clear();
    CHECK(core.retireCapacity() == capacity);
    CHECK(context().drain().has_value());
    CHECK(core.retireSlots() == slots_before);
}

// §5.13 F-15 (erratum E3): a creation that fails after its reservation gives the slots back, so
// repeated failures neither keep slots nor grow the queue.
TEST_CASE_FIXTURE(GpuTest, "retire queue: a failed creation gives its slots back") {
    constexpr int ATTEMPTS = 100;
    auto& core = Access::core(context());
    const ImageDesc probe{.width = 8,
                          .height = 8,
                          .format = VK_FORMAT_R16_UINT,
                          .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                          .name = "too large"};
    const auto supported = core.physicalDevice().getImageFormatProperties2(
        {.format = static_cast<vk::Format>(probe.format),
         .type = vk::ImageType::e2D,
         .tiling = vk::ImageTiling::eOptimal,
         .usage = vk::ImageUsageFlags(probe.usage)},
        core.dispatch());
    REQUIRE(supported.result == vk::Result::eSuccess);
    const std::uint32_t max_width = supported.value.imageFormatProperties.maxExtent.width;
    REQUIRE(max_width < UINT32_MAX);
    ImageDesc too_large = probe;
    too_large.width = max_width + 1;
    const std::size_t slots = core.retireSlots();
    std::size_t capacity = 0;
    for (int attempt = 0; attempt < ATTEMPTS; ++attempt) {
        const auto image = createImage(context(), too_large);
        REQUIRE_FALSE(image.has_value());
        if (!image) {
            CHECK(image.error().code == dvv::Errc::Unsupported);
        }
        CHECK(core.retireSlots() == slots);
        if (attempt == 0) {
            // The first reservation grows the new context's empty queue; the others reuse it.
            capacity = core.retireCapacity();
        }
    }
    CHECK(core.retireSlots() == slots);
    CHECK(core.retireCapacity() == capacity);
}

// §5.13 F-15: the same for a buffer and a pipeline that fail after their reservation.
TEST_CASE_FIXTURE(GpuTest, "retire queue: a failed buffer or pipeline gives its slots back") {
    auto& core = Access::core(context());
    const std::size_t slots = core.retireSlots();
    const auto buffer = createBuffer(context(), {.size = 0, .name = "empty"});
    REQUIRE_FALSE(buffer.has_value());
    CHECK(core.retireSlots() == slots);
    const dvv::gpu::EmbeddedShader no_code{.name = "empty"};
    const auto pipeline = createComputePipeline(context(), no_code, 0);
    REQUIRE_FALSE(pipeline.has_value());
    CHECK(core.retireSlots() == slots);
}

// §5.6: the scoped mapping.
TEST_CASE_FIXTURE(GpuTest, "a write scope flushes and a read scope sees device writes") {
    const auto upload = createBuffer(context(), UPLOAD).value();
    const auto readback = createBuffer(context(), READBACK).value();
    MappedWrite write = upload->mapWrite();
    writeBytes(write.bytes(), sequence());
    REQUIRE(write.flush().has_value());
    REQUIRE(submitAndWait([&](CommandRecorder& recorder) {
                recorder.copyBuffer(*upload, *readback);
            }).has_value());
    const auto read = readback->mapRead();
    CHECK(read.has_value());
    if (read) {
        CHECK(holds(read->bytes(), sequence()));
    }
}

TEST_CASE_FIXTURE(GpuTest, "a device-only buffer maps to empty scopes") {
    const auto buffer = createBuffer(context(), {.size = 64,
                                                 .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                 .memory = MemoryUsage::DeviceOnly})
                            .value();
    CHECK(buffer->mapWrite().bytes().empty());
    CHECK(buffer->mapWrite().flush().has_value());
    const auto read = buffer->mapRead();
    CHECK(read.has_value());
    if (read) {
        CHECK(read->bytes().empty());
    }
}

TEST_CASE_FIXTURE(GpuTest, "a stored read scope keeps its bytes") {
    const auto upload = createBuffer(context(), UPLOAD).value();
    const auto readback = createBuffer(context(), READBACK).value();
    MappedWrite write = upload->mapWrite();
    writeBytes(write.bytes(), sequence());
    REQUIRE(write.flush().has_value());
    REQUIRE(submitAndWait([&](CommandRecorder& recorder) {
                recorder.copyBuffer(*upload, *readback);
            }).has_value());

    SUBCASE("in a std::optional") {
        std::optional<MappedRead> scope;
        scope = readback->mapRead().value();
        struct Frame {
            std::optional<dvv::gpu::MappedRead> scope;
        };
        const Frame frame{.scope = std::move(scope)};
        CHECK(frame.scope.has_value());
        if (frame.scope) {
            CHECK(holds(frame.scope->bytes(), sequence()));
        }
    }
    // A scope stored behind a std::unique_ptr.
    SUBCASE("in a std::unique_ptr") {
        auto read = readback->mapRead();
        CHECK(read.has_value());
        if (read) {
            const std::unique_ptr<dvv::gpu::MappedRead> scope =
                std::make_unique<MappedRead>(std::move(*read));
            CHECK(holds(scope->bytes(), sequence()));
        }
    }
}

TEST_CASE_FIXTURE(GpuTest, "the deprecated mapped() still works (0.3.x)") {
    const auto upload = createBuffer(context(), UPLOAD).value();
    const auto readback = createBuffer(context(), READBACK).value();
    writeBytes(upload->mapped(), sequence());
    REQUIRE(upload->flush().has_value());
    REQUIRE(submitAndWait([&](CommandRecorder& recorder) {
                recorder.copyBuffer(*upload, *readback);
            }).has_value());
    REQUIRE(readback->invalidate().has_value());
    CHECK(holds(readback->mapped(), sequence()));
}

TEST_CASE_FIXTURE(GpuTest, "a moved-from map scope is empty and flushes without a call") {
    const auto buffer = createBuffer(context(), UPLOAD).value();
    const auto buffer2 = createBuffer(context(), UPLOAD).value();
    const auto readback = createBuffer(context(), READBACK).value();

    MappedWrite a = buffer->mapWrite();
    MappedWrite b = std::move(a);
    CHECK(isEmpty(a));  // NOLINT(bugprone-use-after-move): the moved-from state is under test
    CHECK(flushes(a));  // a non-const reference: tidy counts it as a reinitialisation
    writeBytes(b.bytes(), sequence());
    CHECK(b.flush().has_value());

    // The move-assigned scope writes the first buffer, not the one it was made from.
    std::vector<std::byte> reversed = sequence();
    std::ranges::reverse(reversed);
    MappedWrite c = buffer2->mapWrite();
    c = std::move(b);
    CHECK(isEmpty(b));  // NOLINT(bugprone-use-after-move): the moved-from state is under test
    CHECK(flushes(b));  // a non-const reference: tidy counts it as a reinitialisation
    writeBytes(c.bytes(), reversed);
    CHECK(c.flush().has_value());
    REQUIRE(submitAndWait([&](CommandRecorder& recorder) {
                recorder.copyBuffer(*buffer, *readback);
            }).has_value());

    MappedRead first = readback->mapRead().value();
    MappedRead second = std::move(first);
    CHECK(isEmpty(first));  // NOLINT(bugprone-use-after-move): the moved-from state is under test
    CHECK(holds(second.bytes(), reversed));
    MappedRead third = buffer2->mapRead().value();
    third = std::move(second);
    CHECK(isEmpty(second));  // NOLINT(bugprone-use-after-move): the moved-from state is under test
    CHECK(holds(third.bytes(), reversed));
}
