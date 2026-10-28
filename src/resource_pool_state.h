#pragma once

// The shared part of a ResourcePool: its free list and counters. The leases handed out hold it
// weakly, so a buffer released after its pool is gone is simply freed.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

#include "device_core.h"
#include "dvv/gpu/resources.h"
#include "free_list.h"

namespace dvv::gpu::detail {

struct ResourcePoolState {
    explicit ResourcePoolState(std::shared_ptr<DeviceCore> device) noexcept
        : core(std::move(device)) {}

    // Takes a buffer back; reusable once the latest reserved submission has finished.
    void giveBack(std::shared_ptr<GpuBuffer>& buffer);

    std::shared_ptr<DeviceCore> core;
    mutable std::mutex mutex;
    FreeList<std::shared_ptr<GpuBuffer>> free;
    std::atomic<std::size_t> leased{0};
    std::uint64_t created = 0;
    std::uint64_t reused = 0;
};

}  // namespace dvv::gpu::detail
