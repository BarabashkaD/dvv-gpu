#pragma once

#include <thread>

namespace dvv::gpu::detail {

// Joins `thread`; does nothing when it is not joinable, so a second call is harmless. On Windows
// it first waits for the thread's end while dispatching only the messages other threads send to
// the calling thread (SendMessage, e.g. from DXGI-based WSI to the window's thread); posted
// messages stay queued; a failed wait is logged, and the join follows on every path. Elsewhere it
// joins.
void joinDispatchingSentMessages(std::jthread& thread) noexcept;

}  // namespace dvv::gpu::detail
