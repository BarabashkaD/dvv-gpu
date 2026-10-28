#include <doctest/doctest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "access.h"
#include "device_core.h"
#include "dvv/base/error.h"
#include "dvv/base/log.h"
#include "dvv/gpu/command_recorder.h"
#include "dvv/gpu/compute_pipeline.h"
#include "dvv/gpu/execution_context.h"
#include "gpu_fixture.h"
#include "pipeline_cache.h"
#include "test_shaders/add_one.h"

using dvv::gpu::CommandRecorder;
using dvv::gpu::ComputePipeline;
using dvv::gpu::ContextOptions;
using dvv::gpu::detail::Access;
using dvv::gpu::detail::PIPELINE_CACHE_TRAILER_SIZE;
using dvv::gpu::test::CapturedLog;
using dvv::gpu::test::GpuTest;
using dvv::gpu::test::TemporaryDirectory;

namespace {

constexpr std::uint32_t PUSH_CONSTANT_SIZE = 16;  // add_one: buffer address, count, padding

std::vector<std::byte> readFile(const std::filesystem::path& file) {
    std::ifstream stream(file, std::ios::binary);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(stream)),
                                  std::istreambuf_iterator<char>());
    std::vector<std::byte> data(bytes.size());
    std::memcpy(data.data(), bytes.data(), bytes.size());
    return data;
}

void writeFile(const std::filesystem::path& file, std::span<const std::byte> data) {
    std::ofstream stream(file, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char*>(data.data()),
                 static_cast<std::streamsize>(data.size()));
    REQUIRE(stream.good());
}

std::string utf8(const std::filesystem::path& path) {
    const std::u8string text = path.u8string();
    return {reinterpret_cast<const char*>(text.data()), text.size()};
}

// The driver's data of a pipeline cache file (checked: it has a matching dvv trailer).
std::vector<std::byte> driverData(std::vector<std::byte> file) {
    REQUIRE(dvv::gpu::detail::pipelineCacheTrailerMismatch(file) == std::nullopt);
    file.resize(file.size() - PIPELINE_CACHE_TRAILER_SIZE);
    return file;
}

// Files a save left behind in the folder (its temporary files).
std::size_t temporaryFiles(const std::filesystem::path& folder) {
    return static_cast<std::size_t>(
        std::ranges::count_if(std::filesystem::recursive_directory_iterator(folder),
                              [](const std::filesystem::directory_entry& entry) {
                                  return entry.path().extension() == ".tmp";
                              }));
}

dvv::gpu::detail::PipelineCacheIdentity identityOf(const dvv::gpu::detail::DeviceCore& core) {
    const auto properties = core.physicalDevice().getProperties(core.dispatch());
    dvv::gpu::detail::PipelineCacheIdentity identity{.vendorId = properties.vendorID,
                                                     .deviceId = properties.deviceID};
    std::ranges::copy(properties.pipelineCacheUUID, identity.cacheUuid.begin());
    return identity;
}

}  // namespace

TEST_CASE_FIXTURE(GpuTest, "a compute pipeline is made from an embedded shader") {
    auto pipeline =
        context().createComputePipeline(test_shaders::addOne(), PUSH_CONSTANT_SIZE).value();
    CHECK(pipeline->name() == "add_one");
    CHECK(pipeline->pushConstantSize() == PUSH_CONSTANT_SIZE);
    CHECK(pipeline->handle() != VK_NULL_HANDLE);
    CHECK(pipeline->layout() != VK_NULL_HANDLE);
    CHECK(pipeline->deviceId() == device());
    CHECK(pipeline->generation() == dvv::gpu::FIRST_GENERATION);
    CHECK(pipeline->ownerId() == context().objectId());
    CHECK(pipeline->typeName() == "dvv::gpu::ComputePipeline");

    const auto before = context().retireStats();
    pipeline.reset();  // nothing submitted: the pipeline and its layout are freed at once
    const auto after = context().retireStats();
    CHECK(after.retired == before.retired + 2);
    CHECK(after.freed == before.freed + 2);
}

TEST_CASE_FIXTURE(GpuTest, "pipelines that cannot be made are reported") {
    const dvv::gpu::EmbeddedShader empty{.name = "empty"};
    const auto no_code = context().createComputePipeline(empty, 0);
    REQUIRE_FALSE(no_code.has_value());
    CHECK(no_code.error().code == dvv::Errc::Internal);
    CHECK(no_code.error().message == "shader 'empty' is not SPIR-V");

    // Only the magic number: shorter than the SPIR-V header.
    const dvv::gpu::EmbeddedShader magic_only{.name = "magic_only",
                                              .spirv = test_shaders::addOne().spirv.first(1)};
    const auto short_code = context().createComputePipeline(magic_only, 0);
    REQUIRE_FALSE(short_code.has_value());
    CHECK(short_code.error().code == dvv::Errc::Internal);
    CHECK(short_code.error().message == "shader 'magic_only' is not SPIR-V");

    const auto unaligned = context().createComputePipeline(test_shaders::addOne(), 6);
    REQUIRE_FALSE(unaligned.has_value());
    CHECK(unaligned.error().code == dvv::Errc::Internal);

    // The range must cover the shader's 16-byte push-constant block (from its reflection).
    for (const std::uint32_t size : {0U, 8U}) {
        const auto too_small = context().createComputePipeline(test_shaders::addOne(), size);
        REQUIRE_FALSE(too_small.has_value());
        CHECK(too_small.error().code == dvv::Errc::Internal);
        CHECK(too_small.error().message ==
              std::format("shader 'add_one' needs 16 bytes of push constants, not {}", size));
    }

    dvv::gpu::EmbeddedShader broken = test_shaders::addOne();
    broken.reflectionJson = "{\"parameters\": [";
    const auto unreadable = context().createComputePipeline(broken, PUSH_CONSTANT_SIZE);
    REQUIRE_FALSE(unreadable.has_value());
    CHECK(unreadable.error().code == dvv::Errc::Format);
    CHECK(unreadable.error().message.starts_with(
        "shader 'add_one': its reflection cannot be read: "));

    const auto too_large = context().createComputePipeline(test_shaders::addOne(), 1U << 16U);
    REQUIRE_FALSE(too_large.has_value());
    CHECK(too_large.error().code == dvv::Errc::Unsupported);
}

TEST_CASE_FIXTURE(GpuTest, "a dispatch checks its pipeline, push constants and group counts") {
    const auto pipeline =
        context().createComputePipeline(test_shaders::addOne(), PUSH_CONSTANT_SIZE).value();
    const std::byte push_constants[PUSH_CONSTANT_SIZE]{};

    const auto short_push = submitAndWait([&](CommandRecorder& recorder) {
        recorder.dispatch(*pipeline, std::span(push_constants).first(8), 1);
    });
    REQUIRE_FALSE(short_push.has_value());
    CHECK(short_push.error().message ==
          "dispatch: pipeline 'add_one' takes 16 bytes of push constants, not 8");

    const auto no_groups = submitAndWait(
        [&](CommandRecorder& recorder) { recorder.dispatch(*pipeline, push_constants, 0); });
    REQUIRE_FALSE(no_groups.has_value());
    CHECK(no_groups.error().message.starts_with(
        "dispatch: the group counts 0x1x1 must be at least 1 and at most "));

    // One more than the device's limit in a dimension whose limit is below the largest count (a
    // driver may allow UINT32_MAX groups; recording that many would hang the device).
    const auto& limit = Access::core(context()).limits().maxComputeWorkGroupCount;
    std::size_t axis = 0;
    while (axis < std::size(limit) && limit[axis] == std::numeric_limits<std::uint32_t>::max()) {
        ++axis;
    }
    if (axis < std::size(limit)) {
        std::uint32_t counts[3]{1, 1, 1};
        counts[axis] = limit[axis] + 1;
        const auto too_many_groups = submitAndWait([&](CommandRecorder& recorder) {
            recorder.dispatch(*pipeline, push_constants, counts[0], counts[1], counts[2]);
        });
        REQUIRE_FALSE(too_many_groups.has_value());
        CHECK(too_many_groups.error().message.starts_with(
            std::format("dispatch: the group counts {}x{}x{} must be at least 1 and at most ",
                        counts[0], counts[1], counts[2])));
    } else {
        MESSAGE("every group-count limit is the largest 32-bit value: nothing to refuse");
    }

    auto other = registry().createContext(device()).value();
    const auto foreign =
        other->createComputePipeline(test_shaders::addOne(), PUSH_CONSTANT_SIZE).value();
    const auto device_check = submitAndWait(
        [&](CommandRecorder& recorder) { recorder.dispatch(*foreign, push_constants, 1); });
    REQUIRE_FALSE(device_check.has_value());
    CHECK(device_check.error().message == "dispatch: pipeline 'add_one' belongs to another device");
}

TEST_CASE_FIXTURE(GpuTest, "the pipeline cache is saved and loaded again") {
    const TemporaryDirectory folder;
    // A name outside every single-byte code page: paths reach the log as UTF-8.
    const std::filesystem::path file = folder.path() / u8"кэш-キャッシュ" / "pipelines.bin";
    {
        auto ctx =
            registry().createContext(device(), ContextOptions{.pipelineCacheFile = file}).value();
        CHECK_FALSE(Access::pipelineCache(*ctx).loaded());  // no file yet
        const auto pipeline =
            ctx->createComputePipeline(test_shaders::addOne(), PUSH_CONSTANT_SIZE).value();
        REQUIRE(ctx->savePipelineCache().has_value());
        REQUIRE(std::filesystem::exists(file));
        const std::vector<std::byte> saved = driverData(readFile(file));
        CHECK(saved.size() >= 32);
        CHECK(dvv::gpu::detail::pipelineCacheMismatch(saved, identityOf(Access::core(*ctx))) ==
              std::nullopt);
        CHECK(temporaryFiles(folder.path()) == 0);
        std::filesystem::remove(file);
    }  // the context saves the cache again when it closes: the file is missing
    REQUIRE(std::filesystem::exists(file));

    {
        const CapturedLog log(dvv::LogLevel::Info);
        auto loaded =
            registry().createContext(device(), ContextOptions{.pipelineCacheFile = file}).value();
        CHECK(Access::pipelineCache(*loaded).loaded());
        CHECK(log.contains("pipeline cache loaded from " + utf8(file)));
        CHECK(
            loaded->createComputePipeline(test_shaders::addOne(), PUSH_CONSTANT_SIZE).has_value());
    }

    // Nothing compiled: closing does not write the file again.
    const std::vector<std::byte> marker(8, std::byte{0x2a});
    {
        auto unchanged =
            registry().createContext(device(), ContextOptions{.pipelineCacheFile = file}).value();
        CHECK(Access::pipelineCache(*unchanged).loaded());
        writeFile(file, marker);
    }
    CHECK(readFile(file) == marker);
}

TEST_CASE_FIXTURE(GpuTest, "a corrupt, damaged or foreign pipeline cache file is ignored") {
    const TemporaryDirectory folder;
    const std::filesystem::path file = folder.path() / "pipelines.bin";
    const std::string ignored = "pipeline cache " + utf8(file) + " ignored: ";
    const CapturedLog log(dvv::LogLevel::Warn);
    {
        std::ofstream stream(file, std::ios::binary);
        stream << "not a pipeline cache, but long enough to have a header";
    }
    std::vector<std::byte> saved;
    {
        auto ctx =
            registry().createContext(device(), ContextOptions{.pipelineCacheFile = file}).value();
        CHECK_FALSE(Access::pipelineCache(*ctx).loaded());
        CHECK(log.contains(ignored + "it has no dvv trailer"));
        CHECK(ctx->createComputePipeline(test_shaders::addOne(), PUSH_CONSTANT_SIZE).has_value());
        REQUIRE(ctx->savePipelineCache().has_value());  // replaces the corrupt file
        saved = driverData(readFile(file));
        CHECK(dvv::gpu::detail::pipelineCacheMismatch(saved, identityOf(Access::core(*ctx))) ==
              std::nullopt);
    }

    // A damaged payload behind an intact header: the checksum no longer matches.
    REQUIRE(saved.size() >= 32);
    std::vector<std::byte> damaged = saved;
    dvv::gpu::detail::appendPipelineCacheTrailer(damaged);
    damaged[saved.size() - 1] ^= std::byte{0x01};
    writeFile(file, damaged);
    {
        auto ctx =
            registry().createContext(device(), ContextOptions{.pipelineCacheFile = file}).value();
        CHECK_FALSE(Access::pipelineCache(*ctx).loaded());
        CHECK(log.contains(ignored + "the data does not match the checksum of its dvv trailer"));
        CHECK(ctx->createComputePipeline(test_shaders::addOne(), PUSH_CONSTANT_SIZE).has_value());
    }

    // A well-formed cache of another vendor: this device's cache with a changed vendorID.
    std::vector<std::byte> foreign = saved;
    foreign[8] ^= std::byte{0xff};
    dvv::gpu::detail::appendPipelineCacheTrailer(foreign);
    writeFile(file, foreign);
    auto ctx =
        registry().createContext(device(), ContextOptions{.pipelineCacheFile = file}).value();
    CHECK_FALSE(Access::pipelineCache(*ctx).loaded());
    CHECK(log.contains(ignored + "it was written for device"));
    CHECK(ctx->createComputePipeline(test_shaders::addOne(), PUSH_CONSTANT_SIZE).has_value());
}

TEST_CASE_FIXTURE(GpuTest, "concurrent saves to one pipeline cache file leave a complete file") {
    const TemporaryDirectory folder;
    const std::filesystem::path file = folder.path() / "pipelines.bin";
    std::shared_ptr<dvv::gpu::ExecutionContext> contexts[2];
    std::shared_ptr<ComputePipeline> pipelines[2];
    for (std::size_t i = 0; i < 2; ++i) {
        contexts[i] =
            registry().createContext(device(), ContextOptions{.pipelineCacheFile = file}).value();
        pipelines[i] =
            contexts[i]->createComputePipeline(test_shaders::addOne(), PUSH_CONSTANT_SIZE).value();
    }
    constexpr int SAVES = 25;
    int failed[2] = {0, 0};
    {
        std::jthread first([&] {
            for (int i = 0; i < SAVES; ++i) {
                failed[0] += contexts[0]->savePipelineCache().has_value() ? 0 : 1;
            }
        });
        std::jthread second([&] {
            for (int i = 0; i < SAVES; ++i) {
                failed[1] += contexts[1]->savePipelineCache().has_value() ? 0 : 1;
            }
        });
    }
    CHECK(failed[0] == 0);
    CHECK(failed[1] == 0);
    CHECK(temporaryFiles(folder.path()) == 0);
    const std::vector<std::byte> saved = driverData(readFile(file));
    CHECK(dvv::gpu::detail::pipelineCacheMismatch(saved, identityOf(Access::core(*contexts[0]))) ==
          std::nullopt);
}

TEST_CASE_FIXTURE(GpuTest, "without a file the pipeline cache stays in memory") {
    CHECK(Access::pipelineCache(context()).file().empty());
    CHECK_FALSE(Access::pipelineCache(context()).loaded());
    CHECK(context().savePipelineCache().has_value());
}
