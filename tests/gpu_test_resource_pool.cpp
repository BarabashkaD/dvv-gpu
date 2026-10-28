#include <doctest/doctest.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "access.h"
#include "device_core.h"
#include "dvv/base/lifetime.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/resource_pool.h"
#include "dvv/gpu/resources.h"
#include "gpu_fixture.h"

using dvv::gpu::BufferDesc;
using dvv::gpu::GpuBuffer;
using dvv::gpu::MemoryUsage;
using dvv::gpu::ResourcePool;
using dvv::gpu::detail::Access;
using dvv::gpu::test::GpuTest;

namespace {

const BufferDesc STAGING{.size = 4096,
                         .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         .memory = MemoryUsage::HostUpload,
                         .name = "staging"};

void signalTimeline(dvv::gpu::detail::DeviceCore& core, std::uint64_t value) {
    const vk::SemaphoreSignalInfo signal{.semaphore = core.timeline(), .value = value};
    REQUIRE(core.device().signalSemaphore(signal, core.dispatch()) == vk::Result::eSuccess);
}

}  // namespace

TEST_CASE_FIXTURE(GpuTest, "a pool is a managed object of its context") {
    const auto pool = context().createResourcePool().value();
    CHECK(pool->ownerId() == context().objectId());
    CHECK(pool->typeName() == "dvv::gpu::ResourcePool");
    const auto stats = pool->stats();
    CHECK(stats.free == 0);
    CHECK(stats.leased == 0);
    CHECK(stats.created == 0);
    CHECK(stats.reused == 0);
}

TEST_CASE_FIXTURE(GpuTest, "a returned buffer is handed out again for the same description") {
    const auto pool = context().createResourcePool().value();
    auto first = pool->acquireBuffer(STAGING).value();
    const VkBuffer handle = first->handle();
    CHECK(first->size() == 4096);
    CHECK(first->desc().name == "staging");
    CHECK(pool->stats().leased == 1);

    const auto live = dvv::LifetimeRegistry::instance().liveCount();
    first.reset();  // nothing was submitted: reusable at once
    CHECK(dvv::LifetimeRegistry::instance().liveCount() == live);  // still alive in the pool
    CHECK(pool->stats().free == 1);
    CHECK(pool->stats().leased == 0);

    BufferDesc renamed = STAGING;
    renamed.name = "other name";
    const auto again = pool->acquireBuffer(renamed).value();
    CHECK(again->handle() == handle);
    CHECK(again->desc().name == "staging");  // a reused buffer keeps its name
    const auto stats = pool->stats();
    CHECK(stats.created == 1);
    CHECK(stats.reused == 1);
    CHECK(stats.free == 0);

    // Another size, usage or memory gets a new buffer.
    BufferDesc larger = STAGING;
    larger.size = 8192;
    CHECK(pool->acquireBuffer(larger).value()->handle() != handle);
    CHECK(pool->stats().created == 2);
}

TEST_CASE_FIXTURE(GpuTest, "a returned buffer waits for the submission being recorded") {
    auto& core = Access::core(context());
    const auto pool = context().createResourcePool().value();
    auto first = pool->acquireBuffer(STAGING).value();
    const VkBuffer first_handle = first->handle();
    // As if the submit thread were recording a submission: reserved, not yet submitted.
    const std::uint64_t recording = core.reserveSubmitValue();
    first.reset();  // the commands being recorded may use it

    auto second = pool->acquireBuffer(STAGING).value();
    CHECK(second->handle() != first_handle);
    CHECK(pool->stats().created == 2);

    signalTimeline(core, recording);
    const auto third = pool->acquireBuffer(STAGING).value();
    CHECK(third->handle() == first_handle);
    CHECK(pool->stats().reused == 1);
}

TEST_CASE_FIXTURE(GpuTest, "trim frees the returned buffers") {
    const auto pool = context().createResourcePool().value();
    auto first = pool->acquireBuffer(STAGING).value();
    auto second = pool->acquireBuffer(STAGING).value();
    const auto live = dvv::LifetimeRegistry::instance().liveCount();
    first.reset();
    second.reset();
    CHECK(pool->stats().free == 2);
    CHECK(pool->trim() == 2);
    CHECK(pool->stats().free == 0);
    CHECK(dvv::LifetimeRegistry::instance().liveCount() == live - 2);
    CHECK(context().retireStats().retiring == 0);
}

TEST_CASE_FIXTURE(GpuTest, "destroying a pool frees the buffers returned to it") {
    auto pool = context().createResourcePool().value();
    auto buffer = pool->acquireBuffer(STAGING).value();
    const auto live = dvv::LifetimeRegistry::instance().liveCount();
    buffer.reset();
    CHECK(pool->stats().free == 1);
    pool.reset();
    CHECK(dvv::LifetimeRegistry::instance().liveCount() == live - 2);  // the pool and the buffer
    CHECK(context().retireStats().retiring == 0);
}

TEST_CASE_FIXTURE(GpuTest, "a buffer released after its pool is freed") {
    auto pool = context().createResourcePool().value();
    auto buffer = pool->acquireBuffer(STAGING).value();
    const auto live = dvv::LifetimeRegistry::instance().liveCount();
    pool.reset();
    CHECK(dvv::LifetimeRegistry::instance().liveCount() == live - 1);  // the pool
    buffer.reset();
    CHECK(dvv::LifetimeRegistry::instance().liveCount() == live - 2);  // and the buffer
}

TEST_CASE_FIXTURE(GpuTest, "buffers are acquired and returned from several threads") {
    constexpr int THREADS = 4;
    constexpr int ROUNDS = 100;
    const auto pool = context().createResourcePool().value();
    std::atomic<int> failures{0};
    {
        std::vector<std::jthread> threads;
        threads.reserve(THREADS);
        for (int thread = 0; thread < THREADS; ++thread) {
            threads.emplace_back([&pool, &failures] {
                for (int round = 0; round < ROUNDS; ++round) {
                    if (!pool->acquireBuffer(STAGING)) {
                        failures.fetch_add(1);
                    }
                }
            });
        }
    }  // joins the threads
    CHECK(failures.load() == 0);
    const auto stats = pool->stats();
    CHECK(stats.leased == 0);
    CHECK(stats.created + stats.reused == std::uint64_t{THREADS} * ROUNDS);
    CHECK(stats.free == stats.created);
}
