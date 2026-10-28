#include "dvv/gpu/resource_pool.h"

#include <optional>
#include <utility>

#include "dvv/gpu/execution_context.h"
#include "resource_pool_state.h"

namespace dvv::gpu {

static_assert(ManagedType<ResourcePool>);

namespace {

// The deleter of a lease: returns the buffer to its pool, or frees it when the pool is gone.
class Return {
public:
    Return(std::weak_ptr<detail::ResourcePoolState> pool,
           std::shared_ptr<GpuBuffer> buffer) noexcept
        : pool_(std::move(pool)), buffer_(std::move(buffer)) {}

    void operator()(GpuBuffer* /*lease*/) noexcept {
        const std::shared_ptr<detail::ResourcePoolState> pool = pool_.lock();
        if (!pool) {
            buffer_.reset();
            return;
        }
        pool->leased.fetch_sub(1, std::memory_order_relaxed);
        try {
            pool->giveBack(buffer_);
        } catch (...) {
            buffer_.reset();  // out of memory: freed here, unless the free list already freed it
        }
    }

private:
    std::weak_ptr<detail::ResourcePoolState> pool_;
    std::shared_ptr<GpuBuffer> buffer_;
};

}  // namespace

namespace detail {

void ResourcePoolState::giveBack(std::shared_ptr<GpuBuffer>& buffer) {
    // The key first: the order in which the arguments below are evaluated is unspecified.
    const BufferKey key = bufferKey(buffer->desc());
    const std::scoped_lock lock(mutex);
    // Any submission that used the buffer was reserved no later than now. A buffer the list cannot
    // take (out of memory) is freed by add(), under this lock; the lock order is pool, then retire
    // queue, never the reverse.
    free.add(key, std::move(buffer), core->lastReservedValue());
}

}  // namespace detail

ResourcePool::ResourcePool(CreationKey<ExecutionContext> key,
                           std::shared_ptr<ExecutionContext> context,
                           std::shared_ptr<detail::ResourcePoolState> state) noexcept
    : Managed(std::move(key)), context_(std::move(context)), state_(std::move(state)) {}

// Destroying state_ frees the returned buffers (leases hold it weakly); nothing is allocated here.
ResourcePool::~ResourcePool() = default;

Result<std::shared_ptr<GpuBuffer>> ResourcePool::acquireBuffer(BufferDesc desc) {
    const detail::BufferKey key = detail::bufferKey(desc);
    // Read before taking the lock: the value only grows, so an entry it shows ready is ready.
    DVV_TRY(const std::uint64_t completed, state_->core->completedValue());
    std::shared_ptr<GpuBuffer> buffer;
    {
        const std::scoped_lock lock(state_->mutex);
        if (std::optional<std::shared_ptr<GpuBuffer>> ready = state_->free.take(key, completed)) {
            buffer = std::move(*ready);
            ++state_->reused;
        }
    }
    if (!buffer) {
        DVV_TRY(buffer, context_->createBuffer(std::move(desc)));
        const std::scoped_lock lock(state_->mutex);
        ++state_->created;
    }
    GpuBuffer* const lease = buffer.get();
    state_->leased.fetch_add(1, std::memory_order_relaxed);
    // Should the control block not be allocated, the deleter runs and the buffer returns.
    return std::shared_ptr<GpuBuffer>(lease, Return(state_, std::move(buffer)));
}

std::size_t ResourcePool::trim() {
    detail::FreeList<std::shared_ptr<GpuBuffer>> released;
    {
        const std::scoped_lock lock(state_->mutex);
        released = state_->free.takeAll();
    }
    return released.size();  // the buffers are freed (retired) here, outside the lock
}

ResourcePoolStats ResourcePool::stats() const {
    const std::scoped_lock lock(state_->mutex);
    return {.free = state_->free.size(),
            .leased = state_->leased.load(std::memory_order_relaxed),
            .created = state_->created,
            .reused = state_->reused};
}

}  // namespace dvv::gpu
