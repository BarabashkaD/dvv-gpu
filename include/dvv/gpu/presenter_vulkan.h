#pragma once

// include/dvv/gpu/presenter_vulkan.h — includes presenter.h. 0.3.x: PresentFrame and
// FrameProvider are still declared in presenter.h; 1.0.0 declares them here.

#include <memory>

#include "dvv/base/error.h"
#include "dvv/gpu/presenter.h"

namespace dvv::gpu {

// A presenter for one native window (see presenter.h). Errors: those of the member it replaces —
// Errc::DeviceLost on a lost context; Errc::Internal on a context without
// ContextOptions::presentation, for an empty `frames`, for a window of another window system (or
// none), and (0.3.x) when options.frames is also set ("two frame providers"); Errc::Unsupported
// when the queue cannot present to the window or no swapchain format and usage fit; surface errors
// as converted; Internal (OutOfMemory for std::bad_alloc) when the presenter thread cannot start.
// Call it on the window's thread.
[[nodiscard]] Result<std::unique_ptr<Presenter>> createPresenter(ExecutionContext& context,
                                                                 PresenterOptions options,
                                                                 FrameProvider frames);

}  // namespace dvv::gpu
