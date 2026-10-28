#pragma once

// The pipeline cache of an ExecutionContext (spec §5.2): seeded from ContextOptions::
// pipelineCacheFile when the file was written for this device and driver, saved back on request
// and when the context closes. An internal part of the context (exempt from spec §9.4
// registration).
//
// The file holds the driver's data (vkGetPipelineCacheData) followed by a dvv trailer: the size of
// the data, its 64-bit FNV-1a checksum and a magic value, in native byte order (a cache only fits
// the machine that wrote it). A file whose trailer, checksum or header does not match is ignored.
// Every save writes its own temporary file and renames it over the cache file, so readers see a
// complete file and concurrent saves to one file (other contexts or processes) are
// last-writer-wins.

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "dvv/base/error.h"
#include "vk.h"

namespace dvv::gpu::detail {

class DeviceCore;

// What a pipeline cache header must match (VkPipelineCacheHeaderVersionOne).
struct PipelineCacheIdentity {
    std::uint32_t vendorId = 0;
    std::uint32_t deviceId = 0;
    std::array<std::uint8_t, VK_UUID_SIZE> cacheUuid{};
};

// Why the data cannot seed this device's pipeline cache (too short, a broken or unknown header,
// another device or another driver build); nullopt when it can.
[[nodiscard]] std::optional<std::string> pipelineCacheMismatch(std::span<const std::byte> data,
                                                               const PipelineCacheIdentity& device);

// Bytes of the dvv trailer at the end of a pipeline cache file.
inline constexpr std::size_t PIPELINE_CACHE_TRAILER_SIZE = 24;
// Largest pipeline cache file that is read.
inline constexpr std::uintmax_t MAX_PIPELINE_CACHE_FILE_SIZE = std::uintmax_t{512} << 20U;

// 64-bit FNV-1a.
[[nodiscard]] std::uint64_t fnv1a64(std::span<const std::byte> data) noexcept;
// Appends the dvv trailer for the driver data in `data`.
void appendPipelineCacheTrailer(std::vector<std::byte>& data);
// Why `file` is not driver data followed by its dvv trailer (no trailer, a wrong size or
// checksum); nullopt when it is: the driver data is then the file without its last
// PIPELINE_CACHE_TRAILER_SIZE bytes.
[[nodiscard]] std::optional<std::string> pipelineCacheTrailerMismatch(
    std::span<const std::byte> file);

class PipelineCache {
public:
    // Reads the file when it is set and exists; a file that cannot be read, is damaged or does not
    // match the device is ignored with a log line, and the cache starts empty. Fails only when the
    // Vulkan pipeline cache cannot be created.
    [[nodiscard]] static Result<std::unique_ptr<PipelineCache>> create(
        const std::shared_ptr<DeviceCore>& core, std::filesystem::path file);

    PipelineCache(const PipelineCache&) = delete;
    PipelineCache& operator=(const PipelineCache&) = delete;
    PipelineCache(PipelineCache&&) = delete;
    PipelineCache& operator=(PipelineCache&&) = delete;
    // Destroys the Vulkan pipeline cache (without saving it).
    ~PipelineCache();

    [[nodiscard]] vk::PipelineCache handle() const noexcept { return cache_; }
    [[nodiscard]] const std::filesystem::path& file() const noexcept { return file_; }
    // Whether the cache started from the file's data.
    [[nodiscard]] bool loaded() const noexcept { return loaded_; }

    // Writes the cache to the file (through a temporary file of this save and a rename, creating
    // the directory); nothing without a file. Errc::Io when the file cannot be written.
    [[nodiscard]] Result<void> save();
    // Like save(), but nothing when the cache data equals what was loaded or last saved and the
    // file still exists.
    [[nodiscard]] Result<void> saveIfChanged();

    // Use create().
    PipelineCache(std::shared_ptr<DeviceCore> core, std::filesystem::path file,
                  vk::PipelineCache cache, bool loaded) noexcept;

private:
    // Size and checksum of the driver data.
    struct Fingerprint {
        std::size_t size = 0;
        std::uint64_t checksum = 0;
        bool operator==(const Fingerprint&) const = default;
    };

    [[nodiscard]] Result<std::vector<std::byte>> fetch() const;
    [[nodiscard]] Result<void> store(bool only_if_changed);

    std::shared_ptr<DeviceCore> core_;
    std::filesystem::path file_;
    vk::PipelineCache cache_;
    bool loaded_;
    std::mutex save_mutex_;
    std::optional<Fingerprint> stored_;  // the data in the file (loaded or saved), guarded
};

}  // namespace dvv::gpu::detail
