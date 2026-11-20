#pragma once

// Presentation to one native window: a Presenter runs its own thread that
// acquires swapchain images, hands each frame's commands to the context's submit thread and
// presents. One frame is in flight at a time.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <vulkan/vulkan_core.h>

#include "dvv/base/error.h"
#include "dvv/base/lifetime.h"
#include "dvv/base/move_only_function.h"
#include "dvv/gpu/resources.h"
#include "dvv/gpu/submission.h"
#include "dvv/platform/native_window.h"

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

// The per-call input of the frame provider. PresentFrameInfo: the 0.3.x name of
// PresentFrameParams; the name PresentFrameInfo is deprecated (0.3.0) and removed in 1.0.0, which
// also drops `format`.
struct PresentFrameInfo {
    platform::PixelSize extent;  // swapchain extent: render at this size for a 1:1 blit
    // Deprecated (0.3.0): use encoding; removed in 1.0.0.
    VkFormat format = VK_FORMAT_UNDEFINED;
    std::uint64_t frameNumber;  // 0, 1, 2, ... per provider call
    // True only when the previous provider call's frame was submitted and completed (its record
    // function ran). False after a dropped frame, a provider error, and for the first call.
    bool previousFrameCompleted;
    PresentEncoding encoding = PresentEncoding::Linear;  // appended in 0.3.0
};
using PresentFrameParams = PresentFrameInfo;

// `image` is blitted to the swapchain image (linear scaling when sizes differ and the format has
// SAMPLED_IMAGE_FILTER_LINEAR, else nearest). It must belong to this context, have TRANSFER_SRC
// usage and a format with BLIT_SRC (e.g. R8G8B8A8_UNORM). `record` (may be empty) runs first in
// the same submission. Both are released on the presenter thread once the submission has
// completed.
// Deprecated (0.3.0) through presenter.h: use presenter_vulkan.h; removed in 1.0.0.
struct PresentFrame {
    RecordFunction record{};
    std::shared_ptr<GpuImage> image{};
};

// Called on the presenter thread, at most once per frame and never while a frame is in flight (so
// one image can be reused). Must not block long (a new image on resize is fine). An error skips
// the frame and goes to onError; Errc::DeviceLost, or any error on a lost context, fails the
// presenter instead (failure(), one onError). The provider is called before the acquire: the
// record function of a dropped frame is destroyed without running (see
// PresentFrameParams::previousFrameCompleted). The provider must never wait for the window's
// thread and must never destroy the presenter.
// Deprecated (0.3.0) through presenter.h: use presenter_vulkan.h; removed in 1.0.0.
using FrameProvider = MoveOnlyFunction<Result<PresentFrame>(const PresentFrameParams&)>;
// Called on the presenter thread; the GUI forwards it (e.g. wx CallAfter). Like the frame
// provider, it must never wait for the window's thread and must never destroy the presenter.
using PresenterErrorHandler = MoveOnlyFunction<void(const Error&) noexcept>;

// A MetalWindow's CAMetalLayer: its owner (the GUI thread) sets the layer's frame and contentsScale
// and never its drawableSize; the swapchain sets drawableSize to its extent, which is the layer's
// bounds times contentsScale (also for fractional point bounds), i.e. the physical size passed here
// and to resize().
struct PresenterOptions {
    // Deprecated (0.3.0): this field's type becomes dvv::platform::WindowHandle in 1.0.0.
    platform::NativeWindow window{};  // must outlive the presenter
    platform::PixelSize extent{};     // current client size in physical pixels (a hint)
    bool visible = true;
    // Deprecated (0.3.0): use createPresenter's `frames` parameter (presenter_vulkan.h); removed
    // in 1.0.0. That createPresenter refuses a provider here; only the deprecated
    // ExecutionContext::createPresenter reads it.
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
    // Deprecated (0.3.0): use show() or hide(); removed in 1.0.0. Calls show() when `visible`,
    // hide() otherwise.
    void setVisible(bool visible) noexcept;
    void requestFrame() noexcept;  // coalesced until the next frame starts
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
