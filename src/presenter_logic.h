#pragma once

// The presenter's decisions (spec §4; M1b design D10, 5.2-5.7), free of Vulkan calls and threads so
// that every rule is unit-tested on every platform. The presenter thread copies the GUI's request
// block in (setRequests), asks next() what to do, executes it and reports the outcome. Not
// thread-safe: only the presenter thread uses it.

#include <cstdint>
#include <optional>

#include <vulkan/vulkan_core.h>

#include "dvv/base/error.h"
#include "dvv/platform/native_window.h"

namespace dvv::gpu::detail {

// What a WSI call returned, as far as the presenter distinguishes it.
enum class VkOutcome : std::uint8_t {
    Success,
    Suboptimal,
    OutOfDate,
    Timeout,
    SurfaceLost,
    DeviceLost,
    OtherError
};

// VK_SUCCESS -> Success, VK_SUBOPTIMAL_KHR -> Suboptimal, VK_ERROR_OUT_OF_DATE_KHR -> OutOfDate,
// VK_TIMEOUT and VK_NOT_READY -> Timeout, VK_ERROR_SURFACE_LOST_KHR -> SurfaceLost,
// VK_ERROR_DEVICE_LOST -> DeviceLost, anything else -> OtherError.
[[nodiscard]] VkOutcome outcomeOf(VkResult result) noexcept;

class PresenterLogic {
public:
    enum class Action : std::uint8_t {
        Wait,               // nothing to do until a GUI request, a completion or stop
        WaitAndRetry,       // the acquire timed out: wait 10 ms, then call retryElapsed()
        RecreateSwapchain,  // design 5.3
        RenderFrame,        // design 5.2 steps 1-5
        Present,            // design 5.2 step 6 (the frame in flight has completed)
        Exit,               // stop or failed: teardown (5.5)
    };

    // Out-of-date results in a row after which the surface counts as hidden until a GUI event.
    static constexpr int OUT_OF_DATE_LIMIT = 4;

    // The GUI's request block, copied under the presenter mutex at the top of every loop.
    // request_seq grows with every resize, setVisible and requestFrame; stop is sticky.
    void setRequests(platform::PixelSize extent, bool visible, std::uint64_t request_seq,
                     bool stop) noexcept;
    // A swapchain was created for the hint `extent` (the extentHint() of the decision): stores it,
    // clears the recreate mark and the surface-hidden state, and sets the internal re-request.
    void swapchainCreated(platform::PixelSize extent) noexcept;
    // The surface reported extent 0: hidden until request_seq changes; the request stays pending.
    void surfaceHidden() noexcept;
    // The result of vkAcquireNextImageKHR and, for Success and Suboptimal, of the fence wait that
    // follows it (a fence wait that ends in DEVICE_LOST is reported as DeviceLost, the 5 s cap as
    // OtherError).
    void acquired(VkOutcome outcome) noexcept;
    // The 10 ms after WaitAndRetry have passed.
    void retryElapsed() noexcept;
    // The frame's submission was handed to the submit thread: phase InFlight; the request of the
    // current snapshot is served (servedSeq = request_seq of the snapshot); the internal
    // re-request is cleared.
    void frameStarted() noexcept;
    // The provider failed or its image was unusable: the request is served, the phase stays Idle.
    void frameSkipped() noexcept;
    // The frame in flight completed; an error fails the presenter.
    void frameCompleted(std::optional<Errc> error) noexcept;
    // The result of vkQueuePresentKHR for the completed frame.
    void presented(VkOutcome outcome) noexcept;

    [[nodiscard]] Action next() const noexcept;
    [[nodiscard]] bool failed() const noexcept { return failed_; }
    // Not visible, a 0-sized hint, or a surface hidden until the next GUI event.
    [[nodiscard]] bool hidden() const noexcept;
    [[nodiscard]] bool inFlight() const noexcept { return phase_ == Phase::InFlight; }
    // For PresentFrameInfo: true only when the previous provider call's frame completed.
    [[nodiscard]] bool previousFrameCompleted() const noexcept { return previous_completed_; }
    // The GUI's latest size hint (used for swapchain creation when currentExtent is 0xFFFFFFFF).
    [[nodiscard]] platform::PixelSize extentHint() const noexcept { return extent_; }

private:
    enum class Phase : std::uint8_t { Idle, InFlight, PresentPending };

    [[nodiscard]] bool framePending() const noexcept;
    void outOfDate() noexcept;

    platform::PixelSize extent_{};
    bool visible_ = false;
    std::uint64_t request_seq_ = 0;
    bool stop_ = false;

    Phase phase_ = Phase::Idle;
    std::uint64_t served_seq_ = 0;
    bool internal_request_ = false;  // after a swapchain creation or a present OUT_OF_DATE
    bool swapchain_ = false;
    bool recreate_ = false;  // out-of-date or suboptimal seen
    platform::PixelSize created_extent_{};
    std::optional<std::uint64_t> hidden_seq_{};  // hidden while request_seq_ equals it
    int out_of_date_streak_ = 0;
    bool retry_ = false;
    bool previous_completed_ = false;
    bool failed_ = false;
};

}  // namespace dvv::gpu::detail
