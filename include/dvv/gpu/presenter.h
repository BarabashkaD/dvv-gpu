#pragma once

// Presentation to one native window: a Presenter runs its own thread that acquires swapchain
// images, hands each frame's commands to the context's submit thread and presents. One frame is
// in flight at a time. Created by createPresenter (presenter_vulkan.h).

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "dvv/base/error.h"
#include "dvv/base/lifetime.h"
#include "dvv/base/move_only_function.h"
#include "dvv/platform/pixel_size.h"
#include "dvv/platform/window_handle.h"

namespace dvv::gpu {

class ExecutionContext;

namespace detail {
class PresenterCore;
struct Access;
}  // namespace detail

// What the present step does to the values of PresentFrame::image as it writes the window.
// Linear: copies them as they are (the image holds display-encoded values). Srgb: applies the sRGB
// transfer function (the image holds linear values). Fixed per swapchain. The values are those the
// blit reads through the image's format: a *_SRGB image is decoded to linear first, so with Linear
// the window gets linear values. Render into a UNORM (or float) image to control the encoding.
enum class PresentEncoding : std::uint8_t { Linear, Srgb };

// The per-call input of the frame provider.
struct PresentFrameParams {
    platform::PixelSize extent;  // swapchain extent: render at this size for a 1:1 blit
    std::uint64_t frameNumber;   // 0, 1, 2, ... per provider call
    // True only when the previous provider call's frame was submitted and completed (its record
    // function ran). False after a dropped frame, a provider error, and for the first call.
    bool previousFrameCompleted;
    PresentEncoding encoding = PresentEncoding::Linear;
};

// Called on the presenter thread; the GUI forwards it (e.g. wx CallAfter). Like the frame
// provider, it must never wait for the window's thread and must never destroy the presenter.
using PresenterErrorHandler = MoveOnlyFunction<void(const Error&) noexcept>;

// A MetalWindow's CAMetalLayer: its owner (the GUI thread) sets the layer's frame and contentsScale
// and never its drawableSize; the swapchain sets drawableSize to its extent, which is the layer's
// bounds times contentsScale (also for fractional point bounds), i.e. the physical size passed here
// and to resize().
struct PresenterOptions {
    // Built with windowHandleOf() of the window system's header (window_handle_win32.h,
    // window_handle_xlib.h, window_handle_metal.h). The window must outlive the presenter.
    platform::WindowHandle window{};
    platform::PixelSize extent{};  // current client size in physical pixels (a hint)
    bool visible = true;
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

// Presents to one native window on its own thread: acquire, hand the frame's commands to
// the submit thread, present, recreate the swapchain on resize/out-of-date/suboptimal. Created by
// createPresenter (presenter_vulkan.h); keeps the context alive. resize, show, hide, requestFrame
// and stats may be called from any thread.
// On Windows: create and destroy the presenter on the window's thread. While a presenter of this
// context lives, do not call ExecutionContext::drain() on the window's thread.
class Presenter final : public Managed<Presenter> {
public:
    Presenter(CreationKey<ExecutionContext> key, std::shared_ptr<ExecutionContext> context,
              std::unique_ptr<detail::PresenterCore> core) noexcept;
    // Waits for the frame in flight (about one frame) and queue idle, destroys swapchain and
    // surface. Destroy it before the window. On Windows the wait dispatches cross-thread sent
    // messages only.
    ~Presenter();

    void resize(platform::PixelSize extent) noexcept;  // hint; latest wins; requests a frame
    void show() noexcept;                              // frames again; requests a frame
    void hide() noexcept;                              // no frames, no acquires; requests a frame
    void requestFrame() noexcept;                      // coalesced until the next frame starts
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
