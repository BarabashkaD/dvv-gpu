#pragma once

// The free list of a ResourcePool (spec §5.3): returned resources by key, each reusable once the
// timeline semaphore has reached the value it was returned at. Not synchronised; the pool locks.

#include <algorithm>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <utility>

#include <vulkan/vulkan_core.h>

#include "dvv/gpu/resources.h"

namespace dvv::gpu::detail {

// What makes two buffers interchangeable.
struct BufferKey {
    std::uint64_t size = 0;
    VkBufferUsageFlags usage = 0;  // as created: with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
    MemoryUsage memory = MemoryUsage::DeviceOnly;

    friend auto operator<=>(const BufferKey&, const BufferKey&) = default;
};

[[nodiscard]] inline BufferKey bufferKey(const BufferDesc& desc) noexcept {
    return {.size = desc.size,
            .usage = desc.usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            .memory = desc.memory};
}

template <class T>
class FreeList {
public:
    // Adds an item that may be reused once the timeline has reached `reusable_at`. If that cannot
    // allocate, the item is destroyed here and std::bad_alloc propagates.
    void add(const BufferKey& key, T item, std::uint64_t reusable_at) {
        entries_[key].push_back({.item = std::move(item), .reusableAt = reusable_at});
        ++size_;
    }

    // The oldest item for the key whose value the timeline has reached (`completed`).
    [[nodiscard]] std::optional<T> take(const BufferKey& key, std::uint64_t completed) {
        const auto found = entries_.find(key);
        if (found == entries_.end()) {
            return std::nullopt;
        }
        auto& list = found->second;
        const auto ready = std::ranges::find_if(
            list, [completed](const Entry& entry) { return entry.reusableAt <= completed; });
        if (ready == list.end()) {
            return std::nullopt;
        }
        std::optional<T> item(std::move(ready->item));
        list.erase(ready);
        if (list.empty()) {
            entries_.erase(found);
        }
        --size_;
        return item;
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    // Moves every item into the returned list, so the caller can destroy them outside its lock.
    // (Constructing the returned list may allocate, e.g. std::map's head node on MSVC.)
    [[nodiscard]] FreeList takeAll() {
        FreeList all;
        all.entries_.swap(entries_);
        all.size_ = std::exchange(size_, 0);
        return all;
    }

private:
    struct Entry {
        T item;
        std::uint64_t reusableAt = 0;
    };

    std::map<BufferKey, std::deque<Entry>> entries_;
    std::size_t size_ = 0;
};

}  // namespace dvv::gpu::detail
