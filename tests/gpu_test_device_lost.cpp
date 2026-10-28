#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <latch>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "dvv/base/error.h"
#include "dvv/base/executor.h"
#include "dvv/base/log.h"
#include "dvv/base/task.h"
#include "dvv/base/when_all.h"
#include "dvv/gpu/command_recorder.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/resources.h"
#include "dvv/gpu/submission.h"
#include "dvv/gpu/testing/fault_injection.h"
#include "gpu_fixture.h"
#include "test_shaders/add_one.h"

using dvv::Errc;
using dvv::Result;
using dvv::gpu::CommandRecorder;
using dvv::gpu::ContextOptions;
using dvv::gpu::ExecutionContext;
using dvv::gpu::MemoryUsage;
using dvv::gpu::RecordFunction;
using dvv::gpu::test::CapturedLog;
using dvv::gpu::test::GpuTest;
using dvv::gpu::test::TemporaryDirectory;

namespace {

constexpr std::uint32_t ADD_ONE_PUSH_SIZE = 16;  // add_one: buffer address, count, padding

dvv::Task<Result<std::uint64_t>> submitTask(ExecutionContext& context, dvv::IExecutor& executor,
                                            RecordFunction record) {
    co_return co_await context.submit(executor, std::move(record));
}

// A DeviceLost error caused by an injection.
bool injectedLoss(const Result<std::uint64_t>& result) {
    return !result.has_value() && result.error().code == Errc::DeviceLost &&
           result.error().message.find("injected") != std::string::npos;
}

}  // namespace

TEST_CASE_FIXTURE(GpuTest,
                  "pending and queued submissions complete with DeviceLost after an injection") {
    ExecutionContext& ctx = context();
    std::latch entered(1);
    std::latch release(1);
    std::atomic<int> recorded{0};
    std::vector<dvv::Task<Result<std::uint64_t>>> tasks;
    // The first record function blocks the submit thread (test only: record functions must not
    // block), so the other two are queued behind it when the loss is injected.
    tasks.push_back(submitTask(ctx, executor(), [&](CommandRecorder&) {
        ++recorded;
        entered.count_down();
        release.wait();
    }));
    for (int i = 0; i < 2; ++i) {
        tasks.push_back(submitTask(ctx, executor(), [&](CommandRecorder&) { ++recorded; }));
    }
    std::vector<Result<std::uint64_t>> results;
    std::jthread awaiting([&] { results = dvv::syncWait(dvv::whenAll(std::move(tasks))); });
    entered.wait();
    dvv::gpu::testing::injectDeviceLost(ctx);
    release.count_down();
    awaiting.join();
    REQUIRE(results.size() == 3);
    for (const auto& result : results) {
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().code == Errc::DeviceLost);
        CHECK(result.error().message.find("injected") != std::string::npos);
    }
    CHECK(recorded == 1);
    CHECK(ctx.lost());
}

TEST_CASE_FIXTURE(GpuTest,
                  "a submission after the loss completes with DeviceLost without recording") {
    CHECK_FALSE(context().lost());
    dvv::gpu::testing::injectDeviceLost(context());
    CHECK(context().lost());
    bool ran = false;
    const auto result = submitAndWait([&](CommandRecorder&) { ran = true; });
    REQUIRE_FALSE(result.has_value());
    CHECK(result.error().code == Errc::DeviceLost);
    CHECK(result.error().message == "the device was lost (injected)");
    CHECK_FALSE(ran);
    // A second injection is harmless.
    dvv::gpu::testing::injectDeviceLost(context());
    CHECK(context().lost());
    CHECK(injectedLoss(submitAndWait({})));
}

TEST_CASE_FIXTURE(GpuTest, "creators refuse a lost context") {
    dvv::gpu::testing::injectDeviceLost(context());
    const auto buffer = context().createBuffer({.size = 64,
                                                .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                .memory = MemoryUsage::DeviceOnly,
                                                .name = "lost"});
    REQUIRE_FALSE(buffer.has_value());
    CHECK(buffer.error().code == Errc::DeviceLost);
    const auto image = context().createImage({.width = 4,
                                              .height = 4,
                                              .format = VK_FORMAT_R8G8B8A8_UNORM,
                                              .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                              .name = "lost"});
    REQUIRE_FALSE(image.has_value());
    CHECK(image.error().code == Errc::DeviceLost);
    const auto pool = context().createResourcePool();
    REQUIRE_FALSE(pool.has_value());
    CHECK(pool.error().code == Errc::DeviceLost);
    const auto pipeline =
        context().createComputePipeline(test_shaders::addOne(), ADD_ONE_PUSH_SIZE);
    REQUIRE_FALSE(pipeline.has_value());
    CHECK(pipeline.error().code == Errc::DeviceLost);
    const auto bound = context().createComputePipeline(test_shaders::addOne(), ADD_ONE_PUSH_SIZE,
                                                       dvv::gpu::ComputeBindings{});
    REQUIRE_FALSE(bound.has_value());
    CHECK(bound.error().code == Errc::DeviceLost);
    CHECK(bound.error().message == "the device was lost (injected)");
}

TEST_CASE_FIXTURE(GpuTest, "objects released after an injection are freed by the real timeline") {
    auto buffer = context()
                      .createBuffer({.size = std::uint64_t{64} * 1024,
                                     .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                     .memory = MemoryUsage::DeviceOnly,
                                     .name = "released after the loss"})
                      .value();
    REQUIRE(submitAndWait([&](CommandRecorder& recorder) {
                recorder.fillBuffer(*buffer, 0x5a5a5a5aU);
            }).has_value());
    dvv::gpu::testing::injectDeviceLost(context());
    buffer.reset();
    // The real device is healthy, so the idle wait succeeds and frees what was retired.
    REQUIRE(context().drain().has_value());
    CHECK(context().retireStats().retiring == 0);
}

TEST_CASE_FIXTURE(GpuTest,
                  "destroying a lost context does not hang and does not save the pipeline cache") {
    for (int round = 0; round < 20; ++round) {
        CAPTURE(round);
        const TemporaryDirectory dir;
        const std::filesystem::path file = dir.path() / "cache.bin";
        auto ctx =
            registry().createContext(device(), ContextOptions{.pipelineCacheFile = file}).value();
        // A new pipeline changes the cache, so a healthy context would save it.
        REQUIRE(ctx->createComputePipeline(test_shaders::addOne(), ADD_ONE_PUSH_SIZE).has_value());
        REQUIRE(dvv::syncWait(submitTask(*ctx, executor(), {})).has_value());
        dvv::gpu::testing::injectDeviceLost(*ctx);
        CHECK(injectedLoss(dvv::syncWait(submitTask(*ctx, executor(), {}))));
        const CapturedLog log(dvv::LogLevel::Info);
        const auto start = std::chrono::steady_clock::now();
        ctx.reset();
        const auto elapsed = std::chrono::steady_clock::now() - start;
        CHECK(elapsed < std::chrono::seconds(5));
        CHECK_FALSE(std::filesystem::exists(file));
        CHECK(log.contains("not saved"));
    }
}

TEST_CASE_FIXTURE(GpuTest, "injection is thread-safe") {
    ExecutionContext& ctx = context();
    std::vector<Result<std::uint64_t>> results;
    results.reserve(20);
    {
        std::latch start(5);
        std::vector<std::jthread> injectors;
        injectors.reserve(4);
        for (int i = 0; i < 4; ++i) {
            injectors.emplace_back([&] {
                start.arrive_and_wait();
                for (int n = 0; n < 100; ++n) {
                    dvv::gpu::testing::injectDeviceLost(ctx);
                }
            });
        }
        start.arrive_and_wait();
        for (int i = 0; i < 20; ++i) {
            results.push_back(submitAndWait({}));
        }
    }
    CHECK(ctx.lost());
    REQUIRE(results.size() == 20);
    for (const auto& result : results) {
        CHECK((result.has_value() || injectedLoss(result)));
    }
}
