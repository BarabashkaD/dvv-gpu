#pragma once

// Fault injection for tests and for an application's self-test of device loss. Part of the main
// dvv_gpu library, so that a shipped binary can run such a self-test.

namespace dvv::gpu {
class ExecutionContext;
}

namespace dvv::gpu::testing {

// Puts `context` into the lost state (ExecutionContext::lost() == true) as if the driver had
// reported VK_ERROR_DEVICE_LOST: submissions that complete afterwards, and all later ones, complete
// with Errc::DeviceLost "the device was lost (injected)"; presenters of the context fail with
// DeviceLost at their next frame completion or frame-provider error (one onError); creators
// return DeviceLost. The real device stays healthy, so destruction waits for real work as usual.
// Thread-safe; idempotent.
void injectDeviceLost(ExecutionContext& context) noexcept;

}  // namespace dvv::gpu::testing
