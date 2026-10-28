#include <windows.h>

#include "dvv/base/log.h"
#include "thread_wait.h"

namespace dvv::gpu::detail {

void waitForThreadDispatchingSentMessages(void* thread) noexcept {
    const HANDLE handle = thread;
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

}  // namespace dvv::gpu::detail
