#include <doctest/doctest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "dvv/base/paths.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/gpu_info.h"
#include "pipeline_cache.h"

using dvv::gpu::detail::appendPipelineCacheTrailer;
using dvv::gpu::detail::PIPELINE_CACHE_TRAILER_SIZE;
using dvv::gpu::detail::PipelineCacheIdentity;
using dvv::gpu::detail::pipelineCacheMismatch;
using dvv::gpu::detail::pipelineCacheTrailerMismatch;

namespace {

const PipelineCacheIdentity DEVICE{
    .vendorId = 0x10de,
    .deviceId = 0x1e84,
    .cacheUuid = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}};

void put(std::vector<std::byte>& data, std::size_t offset, std::uint32_t word) {
    std::memcpy(data.data() + offset, &word, sizeof(word));
}

// A VkPipelineCacheHeaderVersionOne for `identity`, followed by `payload` bytes of driver data.
std::vector<std::byte> cacheData(const PipelineCacheIdentity& identity, std::size_t payload = 64) {
    std::vector<std::byte> data(32 + payload, std::byte{0x5a});
    put(data, 0, 32);
    put(data, 4, VK_PIPELINE_CACHE_HEADER_VERSION_ONE);
    put(data, 8, identity.vendorId);
    put(data, 12, identity.deviceId);
    std::memcpy(data.data() + 16, identity.cacheUuid.data(), identity.cacheUuid.size());
    return data;
}

}  // namespace

TEST_CASE("pipeline cache data of the same device and driver is accepted") {
    CHECK(pipelineCacheMismatch(cacheData(DEVICE), DEVICE) == std::nullopt);
    CHECK(pipelineCacheMismatch(cacheData(DEVICE, 0), DEVICE) == std::nullopt);
}

TEST_CASE("pipeline cache data that is too short or broken is refused") {
    CHECK(pipelineCacheMismatch({}, DEVICE) == "0 bytes are too short for a pipeline cache header");
    std::vector<std::byte> data = cacheData(DEVICE);
    data.resize(31);
    CHECK(pipelineCacheMismatch(data, DEVICE) ==
          "31 bytes are too short for a pipeline cache header");

    data = cacheData(DEVICE);
    put(data, 0, 16);
    CHECK(pipelineCacheMismatch(data, DEVICE) == "the header size 16 is invalid");
    put(data, 0, 4096);
    CHECK(pipelineCacheMismatch(data, DEVICE) == "the header size 4096 is invalid");
    put(data, 0, 64);  // fits in the data, but a version-one header is exactly 32 bytes
    CHECK(pipelineCacheMismatch(data, DEVICE) == "the header size 64 is invalid");

    data = cacheData(DEVICE);
    put(data, 4, 7);
    CHECK(pipelineCacheMismatch(data, DEVICE) == "the header version 7 is unknown");
}

TEST_CASE("pipeline cache data of another device or driver build is refused") {
    PipelineCacheIdentity other_vendor = DEVICE;
    other_vendor.vendorId = 0x1002;
    CHECK(pipelineCacheMismatch(cacheData(other_vendor), DEVICE) ==
          "it was written for device 0x1e84 of vendor 0x1002");

    PipelineCacheIdentity other_device = DEVICE;
    other_device.deviceId = 0x2204;
    CHECK(pipelineCacheMismatch(cacheData(other_device), DEVICE) ==
          "it was written for device 0x2204 of vendor 0x10de");

    PipelineCacheIdentity other_driver = DEVICE;
    other_driver.cacheUuid[15] = 99;
    CHECK(pipelineCacheMismatch(cacheData(other_driver), DEVICE) ==
          "it was written by another driver build");
}

TEST_CASE("64-bit FNV-1a") {
    CHECK(dvv::gpu::detail::fnv1a64({}) == 0xcbf29ce484222325ULL);
    const std::byte a[] = {std::byte{'a'}};
    CHECK(dvv::gpu::detail::fnv1a64(a) == 0xaf63dc4c8601ec8cULL);
}

TEST_CASE("a pipeline cache file is the driver data with a dvv trailer") {
    const std::vector<std::byte> payload = cacheData(DEVICE);
    std::vector<std::byte> file = payload;
    appendPipelineCacheTrailer(file);
    REQUIRE(file.size() == payload.size() + PIPELINE_CACHE_TRAILER_SIZE);
    CHECK(pipelineCacheTrailerMismatch(file) == std::nullopt);
    CHECK(std::equal(payload.begin(), payload.end(), file.begin()));

    std::vector<std::byte> empty;
    appendPipelineCacheTrailer(empty);
    CHECK(pipelineCacheTrailerMismatch(empty) == std::nullopt);
}

TEST_CASE("a pipeline cache file with a missing or broken dvv trailer is refused") {
    CHECK(pipelineCacheTrailerMismatch({}) == "it has no dvv trailer");
    CHECK(pipelineCacheTrailerMismatch(cacheData(DEVICE)) == "it has no dvv trailer");

    std::vector<std::byte> file = cacheData(DEVICE);
    appendPipelineCacheTrailer(file);
    std::vector<std::byte> truncated(file.begin() + 1, file.end());  // lost its first byte
    CHECK(pipelineCacheTrailerMismatch(truncated) ==
          "the dvv trailer gives 96 bytes of data, the file has 95");

    std::vector<std::byte> damaged = file;
    damaged[40] ^= std::byte{0x10};
    CHECK(pipelineCacheTrailerMismatch(damaged) ==
          "the data does not match the checksum of its dvv trailer");
}

TEST_CASE("the default pipeline cache file is per device in the user cache directory") {
    dvv::gpu::GpuInfo device;
    device.deviceUuid = {0x6d, 0x65, 0x73, 0x61, 0x32, 0x35, 0x2e, 0x32,
                         0x2e, 0x38, 0x2d, 0x30, 0x75, 0x62, 0x75, 0x00};
    const auto file = dvv::gpu::defaultPipelineCacheFile("dvv-gpu-test", device);
    REQUIRE(file.has_value());
    CHECK(file->parent_path() == dvv::paths::userCacheDir("dvv-gpu-test").value());
    CHECK(file->filename() == "vulkan-pipelines-6d657361-3235-2e32-2e38-2d3075627500.bin");
}
