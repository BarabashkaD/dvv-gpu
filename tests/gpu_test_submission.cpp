#include <doctest/doctest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <latch>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "access.h"
#include "device_core.h"
#include "dvv/base/error.h"
#include "dvv/base/executors.h"
#include "dvv/base/task.h"
#include "dvv/gpu/command_recorder.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/resources.h"
#include "gpu_fixture.h"
#include "submitter.h"

using dvv::gpu::BufferDesc;
using dvv::gpu::CommandRecorder;
using dvv::gpu::GpuBuffer;
using dvv::gpu::MemoryUsage;
using dvv::gpu::detail::Access;
using dvv::gpu::test::GpuTest;

namespace {

constexpr VkBufferUsageFlags TRANSFER =
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

std::vector<std::uint32_t> read(GpuBuffer& buffer) {
    REQUIRE(buffer.invalidate().has_value());
    std::vector<std::uint32_t> values(buffer.size() / sizeof(std::uint32_t));
    std::memcpy(values.data(), buffer.mapped().data(), values.size() * sizeof(std::uint32_t));
    return values;
}

void write(GpuBuffer& buffer, const std::vector<std::uint32_t>& values) {
    REQUIRE(buffer.size() >= values.size() * sizeof(std::uint32_t));
    std::memcpy(buffer.mapped().data(), values.data(), values.size() * sizeof(std::uint32_t));
    REQUIRE(buffer.flush().has_value());
}

// Suspends on the submission, then reports the thread the coroutine resumed on.
dvv::Task<std::thread::id> resumedThread(dvv::gpu::ExecutionContext& context,
                                         dvv::IExecutor& executor,
                                         dvv::Result<std::uint64_t>& result) {
    result = co_await context.submit(executor, [](CommandRecorder&) {});
    co_return std::this_thread::get_id();
}

}  // namespace

TEST_CASE_FIXTURE(GpuTest, "a buffer is filled and copied on the GPU") {
    auto device_only = context()
                           .createBuffer({.size = 4096,
                                          .usage = TRANSFER,
                                          .memory = MemoryUsage::DeviceOnly,
                                          .name = "filled"})
                           .value();
    auto readback = context()
                        .createBuffer({.size = 4096,
                                       .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                       .memory = MemoryUsage::HostReadback,
                                       .name = "readback"})
                        .value();
    const auto value = submitAndWait([&](CommandRecorder& recorder) {
        recorder.fillBuffer(*device_only, 0xa5a5a5a5U);
        recorder.fillBuffer(*device_only, 7, 1024, 1024);  // overwrites part of the first fill
        recorder.copyBuffer(*device_only, *readback);
    });
    REQUIRE(value.has_value());
    CHECK(*value > 0);

    const std::vector<std::uint32_t> values = read(*readback);
    REQUIRE(values.size() == 1024);
    for (std::size_t i = 0; i < values.size(); ++i) {
        const std::uint32_t expected = i >= 256 && i < 512 ? 7U : 0xa5a5a5a5U;
        if (values[i] != expected) {
            FAIL("value " << i << " is " << values[i] << ", expected " << expected);
        }
    }
}

TEST_CASE_FIXTURE(GpuTest, "a buffer goes through an R32_UINT image and back") {
    constexpr std::uint32_t WIDTH = 33;
    constexpr std::uint32_t HEIGHT = 17;
    constexpr std::uint64_t BYTES = std::uint64_t{WIDTH} * HEIGHT * sizeof(std::uint32_t);
    auto upload = context()
                      .createBuffer({.size = BYTES,
                                     .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                     .memory = MemoryUsage::HostUpload,
                                     .name = "texels"})
                      .value();
    auto readback = context()
                        .createBuffer({.size = BYTES,
                                       .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                       .memory = MemoryUsage::HostReadback,
                                       .name = "texels back"})
                        .value();
    auto image = context()
                     .createImage({.width = WIDTH,
                                   .height = HEIGHT,
                                   .format = VK_FORMAT_R32_UINT,
                                   .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                            VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                   .name = "texel image"})
                     .value();
    std::vector<std::uint32_t> texels(std::size_t{WIDTH} * HEIGHT);
    for (std::uint32_t i = 0; i < texels.size(); ++i) {
        texels[i] = i * 7 + 1;
    }
    write(*upload, texels);

    REQUIRE(submitAndWait([&](CommandRecorder& recorder) {
                recorder.copyBufferToImage(*upload, *image);    // moves it to TRANSFER_DST_OPTIMAL
                recorder.copyImageToBuffer(*image, *readback);  // and on to TRANSFER_SRC_OPTIMAL
            }).has_value());
    CHECK(read(*readback) == texels);
    CHECK(image->layout() == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    // In GENERAL, both copies use the image as it is.
    REQUIRE(submitAndWait([&](CommandRecorder& recorder) {
                recorder.transition(*image, VK_IMAGE_LAYOUT_GENERAL);
                recorder.fillBuffer(*readback, 0);
                recorder.copyImageToBuffer(*image, *readback);
            }).has_value());
    CHECK(image->layout() == VK_IMAGE_LAYOUT_GENERAL);
    CHECK(read(*readback) == texels);
}

TEST_CASE_FIXTURE(GpuTest, "the awaiting coroutine resumes on the given executor") {
    std::thread::id pool_thread;
    std::latch probed(1);
    executor().post([&] {
        pool_thread = std::this_thread::get_id();
        probed.count_down();
    });
    probed.wait();

    dvv::Result<std::uint64_t> first;
    dvv::Result<std::uint64_t> second;
    CHECK(dvv::syncWait(resumedThread(context(), executor(), first)) == pool_thread);
    CHECK(dvv::syncWait(resumedThread(context(), executor(), second)) == pool_thread);
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());
    CHECK(*second == *first + 1);
    CHECK(Access::core(context()).completedValue().value() >= *second);
}

TEST_CASE_FIXTURE(GpuTest, "resources captured by the record function are freed after the GPU") {
    auto buffer =
        context().createBuffer({.size = 256, .usage = TRANSFER, .name = "captured"}).value();
    const std::weak_ptr<GpuBuffer> watch = buffer;
    const auto before = context().retireStats();
    REQUIRE(submitAndWait([buffer = std::move(buffer)](CommandRecorder& recorder) {
                recorder.fillBuffer(*buffer, 3);
            }).has_value());
    // Released on the executor once the GPU had finished, so freed at once.
    CHECK(watch.expired());
    const auto after = context().retireStats();
    CHECK(after.retired == before.retired + 1);
    CHECK(after.freed == before.freed + 1);
    CHECK(after.retiring == 0);
}

TEST_CASE_FIXTURE(GpuTest, "a resource dropped while its submission is recorded waits for it") {
    auto& core = Access::core(context());
    auto buffer = context()
                      .createBuffer({.size = 256,
                                     .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                     .name = "dropped while recording"})
                      .value();
    GpuBuffer* const borrowed = buffer.get();
    std::latch recording(1);
    std::latch dropped(1);
    dvv::Result<std::uint64_t> result;
    std::jthread submitting([&] {
        result = submitAndWait([&](CommandRecorder& recorder) {
            recorder.fillBuffer(*borrowed, 1);
            recording.count_down();
            dropped.wait();  // the test drops the buffer meanwhile
        });
    });
    recording.wait();
    // The recorded fill uses the buffer: it is retired for the submission being recorded, not
    // freed at once.
    buffer.reset();
    CHECK(context().retireStats().retiring == 1);
    CHECK(core.collect() == 0);
    dropped.count_down();
    submitting.join();
    REQUIRE(result.has_value());
    // The waiter freed it once the submission had finished.
    CHECK(context().retireStats().retiring == 0);
}

TEST_CASE_FIXTURE(GpuTest, "a record function that throws fails only its own submission") {
    auto& core = Access::core(context());
    auto readback = context()
                        .createBuffer({.size = 64,
                                       .usage = TRANSFER,
                                       .memory = MemoryUsage::HostReadback,
                                       .name = "untouched"})
                        .value();
    write(*readback, std::vector<std::uint32_t>(16, 5));
    const std::uint64_t reserved_before = core.lastReservedValue();

    const auto failed = submitAndWait([&](CommandRecorder& recorder) {
        recorder.fillBuffer(*readback, 9);
        throw std::runtime_error("broken recording");
    });
    REQUIRE_FALSE(failed.has_value());
    CHECK(failed.error().code == dvv::Errc::Internal);
    CHECK(failed.error().message.find("broken recording") != std::string::npos);
    // Its value was signalled without running the fill.
    CHECK(core.completedValue().value() == reserved_before + 1);
    CHECK(read(*readback) == std::vector<std::uint32_t>(16, 5));

    const auto next =
        submitAndWait([&](CommandRecorder& recorder) { recorder.fillBuffer(*readback, 9); });
    REQUIRE(next.has_value());
    CHECK(*next == reserved_before + 2);
    CHECK(read(*readback) == std::vector<std::uint32_t>(16, 9));
}

TEST_CASE_FIXTURE(GpuTest, "a command that cannot be recorded fails the submission") {
    auto small = context()
                     .createBuffer({.size = 64,
                                    .usage = TRANSFER,
                                    .memory = MemoryUsage::HostReadback,
                                    .name = "small"})
                     .value();
    auto no_transfer =
        context()
            .createBuffer(
                {.size = 64, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .name = "storage only"})
            .value();
    write(*small, std::vector<std::uint32_t>(16, 1));

    bool failed = false;
    bool raw_empty = false;
    const auto out_of_range = submitAndWait([&](CommandRecorder& recorder) {
        recorder.fillBuffer(*small, 2);
        recorder.fillBuffer(*small, 3, 32, 64);  // ends 32 bytes past the buffer
        failed = recorder.failed();
        recorder.fillBuffer(*small, 4);  // ignored: the recorder has failed
        raw_empty = recorder.raw().commandBuffer == VK_NULL_HANDLE;
    });
    CHECK(failed);
    CHECK(raw_empty);
    REQUIRE_FALSE(out_of_range.has_value());
    CHECK(out_of_range.error().code == dvv::Errc::Internal);
    CHECK(out_of_range.error().message ==
          "fillBuffer: buffer 'small' (64 bytes) has no 64 bytes at offset 32");
    CHECK(read(*small) == std::vector<std::uint32_t>(16, 1));  // nothing ran

    const auto usage = submitAndWait(
        [&](CommandRecorder& recorder) { recorder.copyBuffer(*small, *no_transfer); });
    REQUIRE_FALSE(usage.has_value());
    CHECK(usage.error().message ==
          "copyBuffer: buffer 'storage only' (64 bytes) lacks the usage { TransferDst }");

    const auto alignment =
        submitAndWait([&](CommandRecorder& recorder) { recorder.fillBuffer(*small, 0, 2, 8); });
    REQUIRE_FALSE(alignment.has_value());
    CHECK(alignment.error().message ==
          "fillBuffer: offset 2 and size 8 must be multiples of 4, the size not 0");

    // A buffer of another context (another device as far as the recorder is concerned).
    auto other = registry().createContext(device()).value();
    auto foreign = other->createBuffer({.size = 64, .usage = TRANSFER, .name = "foreign"}).value();
    const auto device_check =
        submitAndWait([&](CommandRecorder& recorder) { recorder.copyBuffer(*foreign, *small); });
    REQUIRE_FALSE(device_check.has_value());
    CHECK(device_check.error().message ==
          "copyBuffer: buffer 'foreign' (64 bytes) belongs to another device");
}

TEST_CASE_FIXTURE(GpuTest, "an empty record function only waits for the earlier work") {
    const auto first = submitAndWait([](CommandRecorder&) {});
    const auto second = submitAndWait(nullptr);
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());
    CHECK(*second == *first + 1);
}

TEST_CASE_FIXTURE(GpuTest, "raw commands get the command buffer after the automatic barrier") {
    auto buffer = context()
                      .createBuffer({.size = 64,
                                     .usage = TRANSFER,
                                     .memory = MemoryUsage::HostReadback,
                                     .name = "raw fill"})
                      .value();
    bool recorded = false;
    REQUIRE(submitAndWait([&](CommandRecorder& recorder) {
                recorder.fillBuffer(*buffer, 1);
                const dvv::gpu::RawCommands raw = recorder.raw();
                if (raw.commandBuffer == VK_NULL_HANDLE) {
                    return;
                }
                const auto fill = reinterpret_cast<PFN_vkCmdFillBuffer>(
                    raw.getDeviceProcAddr(raw.device, "vkCmdFillBuffer"));
                if (fill != nullptr) {
                    fill(raw.commandBuffer, buffer->handle(), 0, VK_WHOLE_SIZE, 6);
                    recorded = true;
                }
            }).has_value());
    CHECK(recorded);
    CHECK(read(*buffer) == std::vector<std::uint32_t>(16, 6));
}

TEST_CASE_FIXTURE(GpuTest, "a context released during a submission closes on the executor") {
    auto ctx = registry().createContext(device()).value();
    const std::weak_ptr<dvv::gpu::ExecutionContext> watch = ctx;
    auto buffer = ctx->createBuffer({.size = 64,
                                     .usage = TRANSFER,
                                     .memory = MemoryUsage::HostReadback,
                                     .name = "last work"})
                      .value();
    dvv::IExecutor& pool = executor();
    std::latch recording(1);
    std::latch released(1);
    dvv::Result<std::uint64_t> result;
    std::jthread submitting([&, &context = *ctx] {
        result = dvv::gpu::test::submitAndWait(context, pool, [&](CommandRecorder& recorder) {
            recorder.fillBuffer(*buffer, 3);
            recording.count_down();
            released.wait();
        });
    });
    recording.wait();
    ctx.reset();  // the awaiting operation now holds the last reference
    CHECK_FALSE(watch.expired());
    released.count_down();
    submitting.join();  // the context was closed on the executor thread before the task ended
    REQUIRE(result.has_value());
    CHECK(watch.expired());
    CHECK(read(*buffer) == std::vector<std::uint32_t>(16, 3));
}

TEST_CASE_FIXTURE(GpuTest, "image commands the recorder cannot handle fail the submission") {
    auto texels = context()
                      .createBuffer({.size = 64,
                                     .usage = TRANSFER,
                                     .memory = MemoryUsage::HostUpload,
                                     .name = "texels"})
                      .value();
    auto depth = context()
                     .createImage({.width = 4,
                                   .height = 4,
                                   .format = VK_FORMAT_D32_SFLOAT,
                                   .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                   .name = "depth"})
                     .value();
    const auto depth_copy = submitAndWait(
        [&](CommandRecorder& recorder) { recorder.copyBufferToImage(*texels, *depth); });
    REQUIRE_FALSE(depth_copy.has_value());
    CHECK(depth_copy.error().message ==
          "copyBufferToImage: image 'depth' (4x4 D32Sfloat) has a format the recorder does not "
          "handle");

    auto color = context()
                     .createImage({.width = 4,
                                   .height = 4,
                                   .format = VK_FORMAT_R32_UINT,
                                   .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                   .name = "color"})
                     .value();
    const auto attachment = submitAndWait([&](CommandRecorder& recorder) {
        recorder.transition(*color, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    });
    REQUIRE_FALSE(attachment.has_value());
    CHECK(attachment.error().message ==
          "transition: image 'color' (4x4 R32Uint) cannot move to ColorAttachmentOptimal");

    const auto missing_usage = submitAndWait([&](CommandRecorder& recorder) {
        recorder.transition(*color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    });
    REQUIRE_FALSE(missing_usage.has_value());
    CHECK(missing_usage.error().message ==
          "transition: image 'color' (4x4 R32Uint) lacks the usage { TransferSrc }");
    CHECK(color->layout() == VK_IMAGE_LAYOUT_UNDEFINED);  // nothing ran
}

TEST_CASE_FIXTURE(GpuTest,
                  "the host signals a reserved value once the earlier signals are reached") {
    // White-box: the last resort of the submit thread when the queue refuses even the signal.
    auto& core = Access::core(context());
    const auto first = submitAndWait([](CommandRecorder&) {});
    REQUIRE(first.has_value());

    dvv::gpu::detail::Submission submission;
    submission.value = core.reserveSubmitValue();  // the submit thread is idle
    Access::hostSignal(Access::submitter(context()), submission);
    CHECK(submission.signalled);
    CHECK(core.completedValue().value() == submission.value);
    CHECK(core.lastSubmittedValue() == submission.value);

    const auto next = submitAndWait([](CommandRecorder&) {});
    REQUIRE(next.has_value());
    CHECK(*next == submission.value + 1);
}

TEST_CASE_FIXTURE(GpuTest, "many submissions in flight complete and free what they used") {
    // 16 threads submit 4 times each, so up to 16 submissions are queued or on the GPU at once:
    // the submit thread records while the waiter waits, command buffers are still running when the
    // next one is needed, and resources are retired for values that are still outstanding. Each
    // submission fills a scratch buffer of its own, which it captures and which is dropped once
    // the GPU has finished, and copies part of it into its slice of one readback buffer.
    constexpr std::size_t THREADS = 16;
    constexpr std::size_t PER_THREAD = 4;
    constexpr std::size_t SUBMISSIONS = THREADS * PER_THREAD;
    constexpr std::uint64_t SLICE = 256;          // bytes of the readback buffer per submission
    constexpr std::uint64_t SCRATCH = 1U << 20U;  // a fill that keeps the GPU busy for a while
    dvv::gpu::ExecutionContext& ctx = context();
    dvv::IExecutor& pool = executor();  // both created before the threads start
    const auto readback = ctx.createBuffer({.size = SUBMISSIONS * SLICE,
                                            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                            .memory = MemoryUsage::HostReadback,
                                            .name = "slices"})
                              .value();
    const auto before = ctx.retireStats();

    std::vector<dvv::Result<std::uint64_t>> results(SUBMISSIONS);
    std::latch start(THREADS);
    {
        std::vector<std::jthread> threads;
        threads.reserve(THREADS);
        for (std::size_t thread = 0; thread < THREADS; ++thread) {
            threads.emplace_back([&, thread] {
                start.arrive_and_wait();
                for (std::size_t i = 0; i < PER_THREAD; ++i) {
                    const std::size_t index = (thread * PER_THREAD) + i;
                    auto scratch = ctx.createBuffer({.size = SCRATCH,
                                                     .usage = TRANSFER,
                                                     .memory = MemoryUsage::DeviceOnly,
                                                     .name = "scratch"});
                    if (!scratch) {
                        results[index] = std::unexpected(std::move(scratch).error());
                        continue;
                    }
                    auto record = [&readback, index,
                                   buffer = std::move(*scratch)](CommandRecorder& recorder) {
                        recorder.fillBuffer(*buffer, static_cast<std::uint32_t>(index + 1));
                        recorder.copyBuffer(*buffer, 0, *readback, index * SLICE, SLICE);
                    };
                    results[index] = dvv::gpu::test::submitAndWait(ctx, pool, std::move(record));
                }
            });
        }
    }  // joins the threads

    std::vector<std::uint64_t> values;
    values.reserve(SUBMISSIONS);
    for (std::size_t index = 0; index < SUBMISSIONS; ++index) {
        if (!results[index]) {
            FAIL("submission " << index << " failed: " << results[index].error().describe());
        }
        values.push_back(*results[index]);
        // One thread's submissions are queued one after another, so their values grow.
        if (index % PER_THREAD != 0) {
            CHECK(values[index] > values[index - 1]);
        }
    }
    std::ranges::sort(values);
    for (std::size_t i = 1; i < values.size(); ++i) {
        if (values[i] != values[0] + i) {
            FAIL("the timeline values are not distinct and contiguous at " << i);
        }
    }

    const std::vector<std::uint32_t> slices = read(*readback);
    const std::size_t words = SLICE / sizeof(std::uint32_t);
    for (std::size_t word = 0; word < slices.size(); ++word) {
        const auto expected = static_cast<std::uint32_t>((word / words) + 1);
        if (slices[word] != expected) {
            FAIL("word " << word << " is " << slices[word] << ", expected " << expected);
        }
    }

    REQUIRE(ctx.drain().has_value());
    const auto after = ctx.retireStats();
    CHECK(after.retiring == 0);
    CHECK(after.retired - before.retired == SUBMISSIONS);
    CHECK(after.freed - before.freed == SUBMISSIONS);
    // A command buffer was still running when the next submission needed one.
    CHECK(Access::commandBufferCount(Access::submitter(ctx)) > 1);
}
