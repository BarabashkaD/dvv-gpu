#pragma once

// Reuse of transient GPU buffers (spec §5.3).

#include <cstddef>
#include <cstdint>
#include <memory>

#include "dvv/base/error.h"
#include "dvv/base/lifetime.h"
#include "dvv/gpu/resources.h"

namespace dvv::gpu {

class ExecutionContext;

namespace detail {
struct ResourcePoolState;
}  // namespace detail

struct ResourcePoolStats {
    std::size_t free = 0;       // returned buffers waiting in the pool
    std::size_t leased = 0;     // buffers handed out and not yet returned
    std::uint64_t created = 0;  // buffers the pool created
    std::uint64_t reused = 0;   // acquisitions served from the pool
};

// Hands out buffers and takes them back (spec §5.3). A buffer acquired from the pool returns to it
// when its last shared_ptr is released, and is handed out again, for the same size, usage and
// memory, once the GPU has finished every submission made or being recorded at its return. Each
// buffer is created (and registered in the lifetime registry) once. Created by
// ExecutionContext::createResourcePool(); keeps its context alive. Thread-safe.
// Pooled images are deferred to M2 (FrameStore, staging): a leased GpuImage must not hand out
// shared_from_this() (the command recorder keeps images that way), so image leases need another
// design.
class ResourcePool final : public Managed<ResourcePool> {
public:
    ResourcePool(CreationKey<ExecutionContext> key, std::shared_ptr<ExecutionContext> context,
                 std::shared_ptr<detail::ResourcePoolState> state) noexcept;
    // Frees the returned buffers. Buffers still handed out are freed when they are released.
    ~ResourcePool();

    // A returned buffer that matches size, usage and memory and that the GPU has finished with,
    // otherwise a new one from the context (errors as ExecutionContext::createBuffer()). A reused
    // buffer keeps the debug name it was created with.
    [[nodiscard]] Result<std::shared_ptr<GpuBuffer>> acquireBuffer(BufferDesc desc);
    // Frees every returned buffer; returns how many.
    std::size_t trim();
    [[nodiscard]] ResourcePoolStats stats() const;

private:
    std::shared_ptr<ExecutionContext> context_;
    std::shared_ptr<detail::ResourcePoolState> state_;
};

}  // namespace dvv::gpu
