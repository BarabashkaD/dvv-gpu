#pragma once

// Fault injection for tests and rawreel-gui's --selftest lost (design M3 D8). Part of the main
// dvv_gpu library, because the GUI selftest is the shipped binary.

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
