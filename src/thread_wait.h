#pragma once

// Waiting for a thread on Windows while serving cross-thread sent messages (design M1b §5.5, R15).

namespace dvv::gpu::detail {

#if defined(_WIN32)
// Returns once the thread whose HANDLE is `thread` has ended. Meanwhile it dispatches only the
// messages other threads send to the calling thread (SendMessage, e.g. from DXGI-based WSI to the
// window's thread); posted messages (input, wx events) stay queued. A failed wait is logged and
// returns; the caller's join() then waits without dispatching.
void waitForThreadDispatchingSentMessages(void* thread) noexcept;
#endif

}  // namespace dvv::gpu::detail
