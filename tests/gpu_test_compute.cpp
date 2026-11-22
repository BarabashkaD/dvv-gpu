// The M1a demo (spec §15): a compute shader built by slangc changes a buffer on the GPU.
#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <vector>

#include "dvv/base/error.h"
#include "dvv/gpu/command_recorder.h"
#include "dvv/gpu/compute_pipeline.h"
#include "dvv/gpu/context_vulkan.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/resources.h"
#include "gpu_fixture.h"
#include "test_shaders/add_one.h"

using dvv::gpu::BufferDesc;
using dvv::gpu::CommandRecorder;
using dvv::gpu::ComputePipeline;
using dvv::gpu::GpuBuffer;
using dvv::gpu::GroupCount;
using dvv::gpu::MappedRead;
using dvv::gpu::MappedWrite;
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
    MappedWrite scope = buffer.mapWrite();
    std::memcpy(scope.bytes().data(), values.data(), buffer.size());
    REQUIRE(scope.flush().has_value());
}

std::vector<std::uint32_t> read(GpuBuffer& buffer) {
    const MappedRead scope = buffer.mapRead().value();
    std::vector<std::uint32_t> values(buffer.size() / sizeof(std::uint32_t));
    std::memcpy(values.data(), scope.bytes().data(), buffer.size());
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
    recorder.dispatch(pipeline, std::as_bytes(std::span(&push, 1)), GroupCount{GROUPS});
}

// Overload safety (compile time only): every braced GroupCount call form resolves to exactly one
// overload.
template <class Recorder>
concept DispatchCallForms =
    requires(Recorder& recorder, const dvv::gpu::ComputePipeline& pipeline,
             std::span<const std::byte> bytes, std::span<dvv::gpu::GpuImage* const> images) {
        recorder.dispatch(pipeline, bytes, {8, 8});
        recorder.dispatch(pipeline, bytes, {8});
        recorder.dispatch(pipeline, bytes, images, {8, 8});
        recorder.dispatch(pipeline, bytes, images, {8});
    };
static_assert(DispatchCallForms<dvv::gpu::CommandRecorder>);

// The trailing-count forms are gone: an integer does not convert to GroupCount.
template <class Recorder>
concept TrailingCountDispatch =
    requires(Recorder& recorder, const dvv::gpu::ComputePipeline& pipeline,
             std::span<const std::byte> bytes) { recorder.dispatch(pipeline, bytes, 8); };
static_assert(!TrailingCountDispatch<dvv::gpu::CommandRecorder>);

template <class Recorder>
concept TrailingCountImageDispatch = requires(
    Recorder& recorder, const dvv::gpu::ComputePipeline& pipeline, std::span<const std::byte> bytes,
    std::span<dvv::gpu::GpuImage* const> images) { recorder.dispatch(pipeline, bytes, images, 8); };
static_assert(!TrailingCountImageDispatch<dvv::gpu::CommandRecorder>);

// The five-value copyBuffer is gone: a copy names its range with BufferCopy.
template <class Recorder>
concept FiveValueCopy = requires(Recorder& recorder, const dvv::gpu::GpuBuffer& buffer) {
    recorder.copyBuffer(buffer, 0, buffer, 0, 4);
};
static_assert(!FiveValueCopy<dvv::gpu::CommandRecorder>);

}  // namespace

TEST_CASE_FIXTURE(GpuTest, "add_one adds 1 to every value of a buffer") {
    const auto pipeline =
        createComputePipeline(context(), test_shaders::addOne(), sizeof(AddOne)).value();
    const auto buffer = createBuffer(context(), VALUES).value();
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
        createComputePipeline(context(), test_shaders::addOne(), sizeof(AddOne)).value();
    const auto buffer = createBuffer(context(), VALUES).value();
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

// The shader indexes by the x group only, so the y count stays 1 (a larger y would repeat every
// write and race).
TEST_CASE_FIXTURE(GpuTest, "a named and a braced GroupCount dispatch write the same values") {
    constexpr std::uint32_t X_GROUPS = 2;
    constexpr std::uint32_t WRITTEN = X_GROUPS * GROUP_SIZE;
    const auto pipeline =
        createComputePipeline(context(), test_shaders::addOne(), sizeof(AddOne)).value();
    const auto named = createBuffer(context(), VALUES).value();
    const auto braced = createBuffer(context(), VALUES).value();
    write(*named, expected(0));
    write(*braced, expected(0));
    const AddOne named_push{.data = named->deviceAddress(), .count = WRITTEN};
    const AddOne braced_push{.data = braced->deviceAddress(), .count = WRITTEN};

    REQUIRE(submitAndWait([&](CommandRecorder& recorder) {
                recorder.dispatch(*pipeline, std::as_bytes(std::span(&named_push, 1)),
                                  GroupCount{X_GROUPS, 1});
                recorder.dispatch(*pipeline, std::as_bytes(std::span(&braced_push, 1)),
                                  {X_GROUPS, 1});
            }).has_value());
    const std::vector<std::uint32_t> from_named = read(*named);
    CHECK(from_named == read(*braced));
    CHECK(from_named[WRITTEN - 1] == ((WRITTEN - 1) * 3) + 1);
    CHECK(from_named[WRITTEN] == (WRITTEN * 3));
}
