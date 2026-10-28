#pragma once

// Presentation to one native window (spec §4, §10.2): a Presenter runs its own thread that
// acquires swapchain images, hands each frame's commands to the context's submit thread and
// presents. One frame is in flight at a time.

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include <vulkan/vulkan_core.h>

#include "dvv/base/error.h"
#include "dvv/base/lifetime.h"
#include "dvv/gpu/resources.h"
#include "dvv/gpu/submission.h"
#include "dvv/platform/native_window.h"

namespace dvv::gpu {

class ExecutionContext;

namespace detail {
class PresenterCore;
struct Access;
}  // namespace detail

struct PresentFrameInfo {
    platform::PixelSize extent;  // swapchain extent: render at this size for a 1:1 blit
    VkFormat format;             // swapchain format (informational; the blit converts)
    std::uint64_t frameNumber;   // 0, 1, 2, ... per provider call
    // True only when the previous provider call's frame was submitted and completed (its record
    // function ran). False after a dropped frame, a provider error, and for the first call.
    bool previousFrameCompleted;
};

// `image` is blitted to the swapchain image (linear scaling when sizes differ and the format has
// SAMPLED_IMAGE_FILTER_LINEAR, else nearest). It must belong to this context, have TRANSFER_SRC
// usage and a format with BLIT_SRC (e.g. R8G8B8A8_UNORM). `record` (may be empty) runs first in
// the same submission. Both are released on the presenter thread once the submission has
// completed.
struct PresentFrame {
    RecordFunction record{};
    std::shared_ptr<GpuImage> image{};
};

// Called on the presenter thread, at most once per frame and never while a frame is in flight (so
// one image can be reused). Must not block long (a new image on resize is fine). An error skips
// the frame and goes to onError; Errc::DeviceLost, or any error on a lost context, fails the
// presenter instead (failure(), one onError). The provider is called before the acquire: the
// record function of a dropped frame is destroyed without running (see
// PresentFrameInfo::previousFrameCompleted).
using FrameProvider = std::move_only_function<Result<PresentFrame>(const PresentFrameInfo&)>;
// Called on the presenter thread; the GUI forwards it (e.g. wx CallAfter).
using PresenterErrorHandler = std::move_only_function<void(const Error&) noexcept>;

struct PresenterOptions {
    platform::NativeWindow window{};  // must outlive the presenter
    platform::PixelSize extent{};     // current client size in physical pixels (a hint)
    bool visible = true;
    FrameProvider frames{};
    PresenterErrorHandler onError{};
    std::string name = "presenter";  // debug names and log lines
};

struct PresenterStats {
    std::uint64_t framesPresented = 0;
    std::uint64_t frameErrors = 0;          // frames skipped (provider error or unusable image)
    std::uint64_t swapchainsCreated = 0;    // first creation included
    platform::PixelSize presentedExtent{};  // of the latest presented frame
    bool hidden = false;                    // not visible, extent 0, or surface extent 0
    bool failed = false;                    // stopped presenting; see failure()
};

// Presents to one native window on its own thread (spec §4): acquire, hand the frame's commands to
// the submit thread, present, recreate the swapchain on resize/out-of-date/suboptimal. Created by
// ExecutionContext::createPresenter(); keeps the context alive. resize, setVisible, requestFrame
// and stats may be called from any thread.
// On Windows: create and destroy the presenter on the window's thread. The frame provider and
// onError must never wait for the window's thread and must never destroy the presenter. While a
// presenter of this context lives, do not call ExecutionContext::drain() on the window's thread.
class Presenter final : public Managed<Presenter> {
public:
    Presenter(CreationKey<ExecutionContext> key, std::shared_ptr<ExecutionContext> context,
              std::unique_ptr<detail::PresenterCore> core) noexcept;
    // Waits for the frame in flight (about one frame) and queue idle, destroys swapchain and
    // surface. Destroy it before the window. On Windows the wait dispatches cross-thread sent
    // messages only.
    ~Presenter();

    void resize(platform::PixelSize extent) noexcept;  // hint; latest wins; requests a frame
    void setVisible(bool visible) noexcept;  // hidden: no frames, no acquires; requests a frame
    void requestFrame() noexcept;            // coalesced until the next frame starts
    [[nodiscard]] PresenterStats stats() const;
    [[nodiscard]] std::optional<Error> failure() const;  // why presenting stopped
    [[nodiscard]] const std::shared_ptr<ExecutionContext>& context() const noexcept {
        return context_;
    }

private:
    friend struct detail::Access;
    // `context_` is declared before `core_` on purpose: the core (thread, swapchain, surface) is
    // destroyed first, while the device and instance are alive.
    std::shared_ptr<ExecutionContext> context_;
    std::unique_ptr<detail::PresenterCore> core_;
};

}  // namespace dvv::gpu
