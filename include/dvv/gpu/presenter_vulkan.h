#pragma once

// The Vulkan side of a presenter: the frame provider and createPresenter.

#include <memory>

#include "dvv/base/error.h"
#include "dvv/base/move_only_function.h"
#include "dvv/gpu/image.h"
#include "dvv/gpu/presenter.h"
#include "dvv/gpu/submission.h"

namespace dvv::gpu {

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
// PresentFrameParams::previousFrameCompleted). The provider must never wait for the window's
// thread and must never destroy the presenter.
using FrameProvider = MoveOnlyFunction<Result<PresentFrame>(const PresentFrameParams&)>;

// A presenter for one window (see presenter.h). Errc::DeviceLost on a lost context;
// Errc::Internal on a context without ContextOptions::presentation, for an empty `frames` and for
// a window of another window system (or an empty handle); Errc::Unsupported when the device's
// queue cannot present to the window or no swapchain format and usage fit (TRANSFER_DST usage, a
// UNORM or SRGB format with BLIT_DST); surface errors as converted by dvv-gpu (a window system the
// loader lacks: Unsupported); a presenter thread that cannot start: Internal (OutOfMemory for
// std::bad_alloc). Call it on the window's thread.
[[nodiscard]] Result<std::unique_ptr<Presenter>> createPresenter(ExecutionContext& context,
                                                                 PresenterOptions options,
                                                                 FrameProvider frames);

}  // namespace dvv::gpu
