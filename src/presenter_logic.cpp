#include "presenter_logic.h"

#include <cassert>

namespace dvv::gpu::detail {

VkOutcome outcomeOf(VkResult result) noexcept {
    switch (result) {
        case VK_SUCCESS: return VkOutcome::Success;
        case VK_SUBOPTIMAL_KHR: return VkOutcome::Suboptimal;
        case VK_ERROR_OUT_OF_DATE_KHR: return VkOutcome::OutOfDate;
        case VK_TIMEOUT:
        case VK_NOT_READY: return VkOutcome::Timeout;
        case VK_ERROR_SURFACE_LOST_KHR: return VkOutcome::SurfaceLost;
        case VK_ERROR_DEVICE_LOST: return VkOutcome::DeviceLost;
        default: return VkOutcome::OtherError;
    }
}

void PresenterLogic::setRequests(platform::PixelSize extent, bool visible,
                                 std::uint64_t request_seq, bool stop) noexcept {
    extent_ = extent;
    visible_ = visible;
    request_seq_ = request_seq;
    stop_ = stop_ || stop;
}

void PresenterLogic::swapchainCreated(platform::PixelSize extent) noexcept {
    swapchain_ = true;
    recreate_ = false;
    created_extent_ = extent;
    hidden_seq_.reset();
    internal_request_ = true;
}

void PresenterLogic::surfaceHidden() noexcept {
    hidden_seq_ = request_seq_;
}

void PresenterLogic::acquired(VkOutcome outcome) noexcept {
    switch (outcome) {
        case VkOutcome::Success: out_of_date_streak_ = 0; break;
        case VkOutcome::Suboptimal:
            recreate_ = true;
            out_of_date_streak_ = 0;
            break;
        case VkOutcome::OutOfDate:
            recreate_ = true;
            previous_completed_ = false;
            outOfDate();
            break;
        case VkOutcome::Timeout:
            retry_ = true;
            previous_completed_ = false;
            break;
        default: failed_ = true; break;
    }
}

void PresenterLogic::retryElapsed() noexcept {
    retry_ = false;
}

void PresenterLogic::frameStarted() noexcept {
    assert(phase_ == Phase::Idle);
    if (phase_ != Phase::Idle) {
        return;
    }
    phase_ = Phase::InFlight;
    served_seq_ = request_seq_;
    internal_request_ = false;
}

void PresenterLogic::frameSkipped() noexcept {
    served_seq_ = request_seq_;
    internal_request_ = false;
    previous_completed_ = false;
}

void PresenterLogic::frameCompleted(std::optional<Errc> error) noexcept {
    assert(phase_ == Phase::InFlight);
    if (phase_ != Phase::InFlight) {
        return;
    }
    if (error) {
        failed_ = true;
        phase_ = Phase::Idle;
        previous_completed_ = false;
        return;
    }
    phase_ = Phase::PresentPending;
    previous_completed_ = true;
}

void PresenterLogic::presented(VkOutcome outcome) noexcept {
    assert(phase_ == Phase::PresentPending);
    if (phase_ != Phase::PresentPending) {
        return;
    }
    phase_ = Phase::Idle;
    switch (outcome) {
        case VkOutcome::Success: out_of_date_streak_ = 0; break;
        case VkOutcome::Suboptimal:
            recreate_ = true;
            out_of_date_streak_ = 0;
            break;
        case VkOutcome::OutOfDate:
            recreate_ = true;
            internal_request_ = true;
            outOfDate();
            break;
        default: failed_ = true; break;
    }
}

PresenterLogic::Action PresenterLogic::next() const noexcept {
    // The thread never exits with a frame in flight.
    if (phase_ == Phase::InFlight) {
        return Action::Wait;
    }
    // A completed frame is not presented after stop.
    if (failed_ || stop_) {
        return Action::Exit;
    }
    if (phase_ == Phase::PresentPending) {
        return Action::Present;
    }
    if (retry_) {
        return Action::WaitAndRetry;
    }
    if (hidden() || !framePending()) {
        return Action::Wait;
    }
    if (!swapchain_ || recreate_ || extent_ != created_extent_) {
        return Action::RecreateSwapchain;
    }
    return Action::RenderFrame;
}

bool PresenterLogic::hidden() const noexcept {
    return !visible_ || extent_.empty() || (hidden_seq_ && *hidden_seq_ == request_seq_);
}

bool PresenterLogic::framePending() const noexcept {
    return request_seq_ > served_seq_ || internal_request_;
}

void PresenterLogic::outOfDate() noexcept {
    if (++out_of_date_streak_ >= OUT_OF_DATE_LIMIT) {
        hidden_seq_ = request_seq_;
        out_of_date_streak_ = 0;
    }
}

}  // namespace dvv::gpu::detail
