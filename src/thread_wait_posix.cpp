#include "thread_wait.h"

namespace dvv::gpu::detail {

void joinDispatchingSentMessages(std::jthread& thread) noexcept {
    if (thread.joinable()) {
        thread.join();
    }
}

}  // namespace dvv::gpu::detail
