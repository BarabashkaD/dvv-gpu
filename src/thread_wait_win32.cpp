#include <windows.h>

#include "dvv/base/log.h"
#include "thread_wait.h"

namespace dvv::gpu::detail {

namespace {

// Returns once the thread `handle` names has ended, dispatching only sent messages meanwhile, or
// after logging a failed wait.
void waitDispatchingSentMessages(HANDLE handle) noexcept {
    while (true) {
        const DWORD woken =
            MsgWaitForMultipleObjectsEx(1, &handle, INFINITE, QS_SENDMESSAGE, MWMO_INPUTAVAILABLE);
        if (woken == WAIT_OBJECT_0) {
            return;  // the thread has ended
        }
        if (woken == WAIT_OBJECT_0 + 1) {
            // PeekMessageW dispatches the pending sent messages; with PM_QS_SENDMESSAGE and
            // PM_NOREMOVE it neither removes nor dispatches posted ones.
            MSG message{};
            static_cast<void>(
                PeekMessageW(&message, nullptr, 0, 0, PM_NOREMOVE | PM_QS_SENDMESSAGE));
            continue;
        }
        DVV_LOG_ERROR(
            "waiting for a presenter thread failed (Windows error {}); joining without "
            "dispatching messages",
            GetLastError());
        return;
    }
}

}  // namespace

void joinDispatchingSentMessages(std::jthread& thread) noexcept {
    if (!thread.joinable()) {
        return;
    }
    waitDispatchingSentMessages(thread.native_handle());
    thread.join();
}

}  // namespace dvv::gpu::detail
