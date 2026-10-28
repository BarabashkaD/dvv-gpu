// The M1a demo (spec §15): a compute shader built by slangc changes a buffer on the GPU.
#include <doctest/doctest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <latch>
#include <memory>
#include <span>
#include <thread>
#include <vector>

#include "dvv/base/error.h"
#include "dvv/gpu/command_recorder.h"
#include "dvv/gpu/compute_pipeline.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/resource_pool.h"
#include "dvv/gpu/resources.h"
#include "gpu_fixture.h"
#include "test_shaders/add_one.h"

using dvv::gpu::BufferDesc;
using dvv::gpu::CommandRecorder;
using dvv::gpu::ComputePipeline;
using dvv::gpu::GpuBuffer;
using dvv::gpu::MemoryUsage;
using dvv::gpu::test::GpuTest;

namespace {

constexpr std::uint32_t GROUP_SIZE = 64;  // [numthreads(64, 1, 1)] in add_one.slang
constexpr std::uint32_t COUNT = 1000;     // the values add_one changes; not a multiple of 64
constexpr std::uint32_t GROUPS = (COUNT + GROUP_SIZE - 1) / GROUP_SIZE;
// The buffer has room for every invocation of the last group: add_one must leave the values
// past COUNT alone.
constexpr std::uint32_t CAPACITY = GROUPS * GROUP_SIZE;
constexpr std::uint32_t UNTOUCHED = 0xdeadbeefU;

// add_one's push constants: the buffer by device address and the number of values.
struct AddOne {
    std::uint64_t data = 0;
    std::uint32_t count = 0;
    std::uint32_t padding = 0;
};
// Pipeline creation checks only the block's size (16): the offsets pin the order of its members.
static_assert(sizeof(AddOne) == 16);
static_assert(offsetof(AddOne, data) == 0 && offsetof(AddOne, count) == 8);

const BufferDesc VALUES{.size = std::uint64_t{CAPACITY} * sizeof(std::uint32_t),
                        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        .memory = MemoryUsage::HostReadback,
                        .name = "add_one values"};

// The buffer's content after `added` add_one dispatches.
std::vector<std::uint32_t> expected(std::uint32_t added) {
    std::vector<std::uint32_t> values(CAPACITY, UNTOUCHED);
    for (std::uint32_t i = 0; i < COUNT; ++i) {
        values[i] = i * 3 + added;
    }
    return values;
}

void write(GpuBuffer& buffer, const std::vector<std::uint32_t>& values) {
    REQUIRE(buffer.size() == values.size() * sizeof(std::uint32_t));
    std::memcpy(buffer.mapped().data(), values.data(), buffer.size());
    REQUIRE(buffer.flush().has_value());
}

std::vector<std::uint32_t> read(GpuBuffer& buffer) {
    REQUIRE(buffer.invalidate().has_value());
    std::vector<std::uint32_t> values(buffer.size() / sizeof(std::uint32_t));
    std::memcpy(values.data(), buffer.mapped().data(), buffer.size());
    return values;
}

// The number of values that differ from `wanted`.
std::size_t mismatches(const std::vector<std::uint32_t>& values,
                       const std::vector<std::uint32_t>& wanted) {
    REQUIRE(values.size() == wanted.size());
    std::size_t count = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        count += values[i] != wanted[i] ? 1 : 0;
    }
    return count;
}

// Records add_one over the first COUNT values of the buffer.
void addOne(CommandRecorder& recorder, const ComputePipeline& pipeline, const GpuBuffer& buffer) {
    const AddOne push{.data = buffer.deviceAddress(), .count = COUNT};
    recorder.dispatch(pipeline, std::as_bytes(std::span(&push, 1)), GROUPS);
}

}  // namespace

TEST_CASE_FIXTURE(GpuTest, "add_one adds 1 to every value of a buffer") {
    const auto pipeline =
        context().createComputePipeline(test_shaders::addOne(), sizeof(AddOne)).value();
    const auto buffer = context().createBuffer(VALUES).value();
    write(*buffer, expected(0));

    REQUIRE(submitAndWait([&](CommandRecorder& recorder) {
                addOne(recorder, *pipeline, *buffer);
            }).has_value());
    CHECK(mismatches(read(*buffer), expected(1)) == 0);
}

TEST_CASE_FIXTURE(GpuTest, "a dispatch sees the writes of the dispatch before it") {
    // Both dispatches reach the buffer only through its device address, which neither the
    // recorder nor synchronization validation can follow: only the recorder's automatic barrier
    // between the two dispatches orders them.
    const auto pipeline =
        context().createComputePipeline(test_shaders::addOne(), sizeof(AddOne)).value();
    const auto buffer = context().createBuffer(VALUES).value();
    write(*buffer, expected(0));

    REQUIRE(submitAndWait([&](CommandRecorder& recorder) {
                addOne(recorder, *pipeline, *buffer);
                addOne(recorder, *pipeline, *buffer);
            }).has_value());
    const std::vector<std::uint32_t> values = read(*buffer);
    // Without the barrier, the second dispatch may read values the first has not written yet.
    const std::vector<std::uint32_t> once = expected(1);
    std::size_t incremented_once = 0;
    for (std::size_t i = 0; i < COUNT; ++i) {
        incremented_once += values[i] == once[i] ? 1 : 0;
    }
    INFO("values incremented only once: " << incremented_once);
    CHECK(mismatches(values, expected(2)) == 0);
}

TEST_CASE_FIXTURE(GpuTest, "a pooled buffer is reused once the GPU has finished with it") {
    const auto pipeline =
        context().createComputePipeline(test_shaders::addOne(), sizeof(AddOne)).value();
    const auto pool = context().createResourcePool().value();
    auto lease = pool->acquireBuffer(VALUES).value();
    const VkBuffer handle = lease->handle();
    write(*lease, expected(0));

    // The dispatch uses the buffer by device address only: the pool, not the recorder, keeps the
    // buffer from being handed out again while the GPU may still use it.
    GpuBuffer* const borrowed = lease.get();
    std::latch recording(1);
    std::latch returned(1);
    std::atomic<bool> recorded = false;
    dvv::Result<std::uint64_t> result;
    std::jthread submitting([&] {
        result = submitAndWait([&](CommandRecorder& recorder) {
            addOne(recorder, *pipeline, *borrowed);
            recorded = true;
            recording.count_down();
            returned.wait();  // the test returns the buffer meanwhile
        });
        if (!recorded) {
            recording.count_down();  // the submission failed before recording: do not hang
        }
    });
    recording.wait();
    // Nothing may throw before returned.count_down(): unwinding would join `submitting` while it
    // waits for `returned`. The results are checked after the join.
    lease.reset();  // back in the pool while its submission is being recorded
    const std::size_t free_while_recording = pool->stats().free;
    const auto other = pool->acquireBuffer(VALUES);
    returned.count_down();
    submitting.join();
    REQUIRE(result.has_value());
    CHECK(free_while_recording == 1);
    REQUIRE(other.has_value());
    CHECK((*other)->handle() != handle);

    // The submission has finished: the buffer is handed out again, with what the GPU left in it.
    const auto again = pool->acquireBuffer(VALUES).value();
    CHECK(again->handle() == handle);
    CHECK(mismatches(read(*again), expected(1)) == 0);
    const auto stats = pool->stats();
    CHECK(stats.created == 2);
    CHECK(stats.reused == 1);
    CHECK(stats.leased == 2);
    CHECK(stats.free == 0);

    REQUIRE(submitAndWait([&](CommandRecorder& recorder) {
                addOne(recorder, *pipeline, *again);
            }).has_value());
    CHECK(mismatches(read(*again), expected(2)) == 0);
}
