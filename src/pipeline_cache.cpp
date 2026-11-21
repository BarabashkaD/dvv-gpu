#include "pipeline_cache.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <new>
#include <system_error>
#include <utility>
#include <vector>

#include "device_core.h"
#include "dvv/base/log.h"
#include "dvv/base/paths.h"
#include "dvv/base/utf8.h"
#include "dvv/platform/file_writer.h"
#include "vk_result.h"

namespace dvv::gpu::detail {

namespace {

// VkPipelineCacheHeaderVersionOne: headerSize, headerVersion, vendorID, deviceID (4 bytes each),
// then pipelineCacheUUID.
constexpr std::size_t HEADER_SIZE = 16 + VK_UUID_SIZE;
// The dvv trailer: the size of the driver data, its FNV-1a checksum, then this magic value.
constexpr char TRAILER_MAGIC[8] = {'d', 'v', 'v', 'p', 'c', 'a', 'c', '1'};
// A replace over a file that another process has open briefly fails on Windows: 4 more attempts,
// so 5 in total as before.
constexpr std::uint32_t REPLACE_RETRIES = 4;
constexpr auto REPLACE_RETRY_DELAY = std::chrono::milliseconds(10);

std::uint32_t readWord(std::span<const std::byte> data, std::size_t offset) noexcept {
    std::uint32_t word = 0;
    std::memcpy(&word, data.data() + offset, sizeof(word));
    return word;
}

std::uint64_t readQuad(std::span<const std::byte> data, std::size_t offset) noexcept {
    std::uint64_t quad = 0;
    std::memcpy(&quad, data.data() + offset, sizeof(quad));
    return quad;
}

void appendTrailer(std::vector<std::byte>& data, std::uint64_t checksum) {
    const std::uint64_t size = data.size();
    const std::size_t at = data.size();
    data.resize(at + PIPELINE_CACHE_TRAILER_SIZE);
    std::memcpy(data.data() + at, &size, sizeof(size));
    std::memcpy(data.data() + at + 8, &checksum, sizeof(checksum));
    std::memcpy(data.data() + at + 16, TRAILER_MAGIC, sizeof(TRAILER_MAGIC));
}

// The whole file; nullopt (with the reason in `problem`) when it cannot be read or is too large.
std::optional<std::vector<std::byte>> readFile(const std::filesystem::path& file,
                                               std::string& problem) {
    std::error_code error;
    const std::uintmax_t size = std::filesystem::file_size(file, error);
    if (error) {
        problem = std::format("its size cannot be read ({})", error.message());
        return std::nullopt;
    }
    if (size > MAX_PIPELINE_CACHE_FILE_SIZE) {
        problem = std::format("{} bytes are too large for a pipeline cache", size);
        return std::nullopt;
    }
    std::ifstream stream(file, std::ios::binary);
    if (!stream) {
        problem = "it cannot be opened";
        return std::nullopt;
    }
    std::vector<std::byte> data;
    try {
        data.resize(static_cast<std::size_t>(size));
    } catch (const std::bad_alloc&) {
        problem = std::format("{} bytes cannot be allocated", size);
        return std::nullopt;
    }
    stream.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
    if (std::cmp_not_equal(stream.gcount(), size)) {
        problem = "it cannot be read";
        return std::nullopt;
    }
    return data;
}

// Writes `data` to `file` through dvv-platform's atomic replace (a temporary file in the same
// folder, then a replace). A torn file is detected by the trailer check, so no flush is needed.
Result<void> writeFile(const std::filesystem::path& file, std::span<const std::byte> data) {
    if (file.has_parent_path()) {
        DVV_TRY_VOID(paths::ensureDirectory(file.parent_path()));
    }
    return platform::replaceFile(file, data,
                                 {.durability = platform::FileDurability::Unflushed,
                                  .retries = REPLACE_RETRIES,
                                  .retryDelay = REPLACE_RETRY_DELAY,
                                  .lostTarget = platform::LostTarget::RemoveTemporary});
}

// `error` with store()'s context in front of its message.
std::unexpected<Error> saveError(Error error) {
    error.message = std::format("saving the pipeline cache failed: {}", error.message);
    return std::unexpected(std::move(error));
}

}  // namespace

std::optional<std::string> pipelineCacheMismatch(std::span<const std::byte> data,
                                                 const PipelineCacheIdentity& device) {
    if (data.size() < HEADER_SIZE) {
        return std::format("{} bytes are too short for a pipeline cache header", data.size());
    }
    const std::uint32_t header_size = readWord(data, 0);
    if (header_size != HEADER_SIZE) {  // VUID-VkPipelineCacheHeaderVersionOne-headerSize-04967
        return std::format("the header size {} is invalid", header_size);
    }
    const std::uint32_t header_version = readWord(data, 4);
    if (header_version != static_cast<std::uint32_t>(VK_PIPELINE_CACHE_HEADER_VERSION_ONE)) {
        return std::format("the header version {} is unknown", header_version);
    }
    const std::uint32_t vendor_id = readWord(data, 8);
    const std::uint32_t device_id = readWord(data, 12);
    if (vendor_id != device.vendorId || device_id != device.deviceId) {
        return std::format("it was written for device {:#06x} of vendor {:#06x}", device_id,
                           vendor_id);
    }
    if (!std::ranges::equal(data.subspan(16, VK_UUID_SIZE), device.cacheUuid,
                            [](std::byte stored, std::uint8_t expected) {
                                return std::to_integer<std::uint8_t>(stored) == expected;
                            })) {
        return std::string("it was written by another driver build");
    }
    return std::nullopt;
}

std::uint64_t fnv1a64(std::span<const std::byte> data) noexcept {
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    for (const std::byte byte : data) {
        hash ^= std::to_integer<std::uint64_t>(byte);
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

void appendPipelineCacheTrailer(std::vector<std::byte>& data) {
    appendTrailer(data, fnv1a64(data));
}

std::optional<std::string> pipelineCacheTrailerMismatch(std::span<const std::byte> file) {
    if (file.size() < PIPELINE_CACHE_TRAILER_SIZE ||
        std::memcmp(file.data() + file.size() - sizeof(TRAILER_MAGIC), TRAILER_MAGIC,
                    sizeof(TRAILER_MAGIC)) != 0) {
        return std::string("it has no dvv trailer");
    }
    const std::size_t at = file.size() - PIPELINE_CACHE_TRAILER_SIZE;
    const std::uint64_t size = readQuad(file, at);
    if (size != at) {
        return std::format("the dvv trailer gives {} bytes of data, the file has {}", size, at);
    }
    if (readQuad(file, at + 8) != fnv1a64(file.first(at))) {
        return std::string("the data does not match the checksum of its dvv trailer");
    }
    return std::nullopt;
}

Result<std::unique_ptr<PipelineCache>> PipelineCache::create(
    const std::shared_ptr<DeviceCore>& core, std::filesystem::path file) {
    std::vector<std::byte> initial;
    if (!file.empty()) {
        std::string problem;
        std::error_code error;
        const bool present = std::filesystem::exists(file, error);
        if (error) {
            problem = std::format("it cannot be checked ({})", error.message());
        } else if (present) {
            if (auto data = readFile(file, problem)) {
                const auto properties = core->physicalDevice().getProperties(core->dispatch());
                PipelineCacheIdentity identity{.vendorId = properties.vendorID,
                                               .deviceId = properties.deviceID};
                std::ranges::copy(properties.pipelineCacheUUID, identity.cacheUuid.begin());
                if (auto damaged = pipelineCacheTrailerMismatch(*data)) {
                    problem = std::move(*damaged);
                } else {
                    data->resize(data->size() - PIPELINE_CACHE_TRAILER_SIZE);
                    if (auto mismatch = pipelineCacheMismatch(*data, identity)) {
                        problem = std::move(*mismatch);
                    } else {
                        initial = std::move(*data);
                    }
                }
            }
        }
        if (!problem.empty()) {
            DVV_LOG_WARN("pipeline cache {} ignored: {}", dvv::pathDisplayText(file), problem);
        }
    }

    vk::PipelineCacheCreateInfo info{.initialDataSize = initial.size(),
                                     .pInitialData = initial.data()};
    auto cache = core->device().createPipelineCache(info, nullptr, core->dispatch());
    if (cache.result != vk::Result::eSuccess && !initial.empty() &&
        errcOf(cache.result) != Errc::OutOfMemory) {
        DVV_LOG_WARN("pipeline cache {} ignored: the driver rejected it ({})",
                     dvv::pathDisplayText(file), vk::to_string(cache.result));
        initial.clear();
        info.initialDataSize = 0;
        info.pInitialData = nullptr;
        cache = core->device().createPipelineCache(info, nullptr, core->dispatch());
    }
    DVV_TRY_VOID(vkCheck(cache.result, "vkCreatePipelineCache"));
    const bool loaded = !initial.empty();
    std::unique_ptr<PipelineCache> made;
    try {
        if (loaded) {
            DVV_LOG_INFO("pipeline cache loaded from {} ({} bytes)", dvv::pathDisplayText(file),
                         initial.size());
        }
        made = std::make_unique<PipelineCache>(core, std::move(file), cache.value, loaded);
    } catch (...) {
        core->device().destroyPipelineCache(cache.value, nullptr, core->dispatch());
        throw;
    }
    if (loaded) {
        // What the driver returns for the loaded data: saveIfChanged() compares with it.
        if (const auto data = made->fetch()) {
            made->stored_ = Fingerprint{.size = data->size(), .checksum = fnv1a64(*data)};
        }
    }
    return made;
}

PipelineCache::PipelineCache(std::shared_ptr<DeviceCore> core, std::filesystem::path file,
                             vk::PipelineCache cache, bool loaded) noexcept
    : core_(std::move(core)), file_(std::move(file)), cache_(cache), loaded_(loaded) {}

PipelineCache::~PipelineCache() {
    core_->device().destroyPipelineCache(cache_, nullptr, core_->dispatch());
}

Result<void> PipelineCache::save() {
    return store(false);
}

Result<void> PipelineCache::saveIfChanged() {
    return store(true);
}

Result<std::vector<std::byte>> PipelineCache::fetch() const {
    std::size_t size = 0;
    DVV_TRY_VOID(
        vkCheck(core_->device().getPipelineCacheData(cache_, &size, nullptr, core_->dispatch()),
                "vkGetPipelineCacheData"));
    // Room for the trailer, so that appending it does not copy the data again.
    std::vector<std::byte> data;
    data.reserve(size + PIPELINE_CACHE_TRAILER_SIZE);
    data.resize(size);
    // eIncomplete: the cache grew between the calls; the shorter data is still a valid cache.
    const vk::Result fetched =
        core_->device().getPipelineCacheData(cache_, &size, data.data(), core_->dispatch());
    if (fetched != vk::Result::eSuccess && fetched != vk::Result::eIncomplete) {
        return vkError(fetched, "vkGetPipelineCacheData");
    }
    data.resize(size);
    return data;
}

Result<void> PipelineCache::store(bool only_if_changed) {
    if (file_.empty()) {
        return {};
    }
    const std::scoped_lock lock(save_mutex_);
    Result<std::vector<std::byte>> fetched = fetch();
    if (!fetched) {
        return saveError(std::move(fetched).error());
    }
    std::vector<std::byte> data = std::move(*fetched);
    const Fingerprint current{.size = data.size(), .checksum = fnv1a64(data)};
    if (only_if_changed && stored_ == current) {
        std::error_code error;
        if (std::filesystem::exists(file_, error)) {
            return {};
        }
    }
    appendTrailer(data, current.checksum);
    if (Result<void> written = writeFile(file_, data); !written) {
        return saveError(std::move(written).error());
    }
    stored_ = current;
    return {};
}

}  // namespace dvv::gpu::detail
