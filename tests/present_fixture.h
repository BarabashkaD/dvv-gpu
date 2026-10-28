#pragma once

// The fixture of the presentation tests (design M1b §8.2): a registry for the build's window
// system, a context with presentation, a native test window and a frame provider the tests steer.

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "dvv/base/error.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/gpu_registry.h"
#include "dvv/gpu/presenter.h"
#include "dvv/platform/testing/test_window.h"

namespace dvv::gpu::test {

// Shared by a test (test thread) and its frame provider (presenter thread).
struct ProviderControl {
    std::atomic<std::uint64_t> calls{0};
    std::atomic<bool> failNext{false};            // the next call returns Errc::Io
    std::atomic<bool> withoutTransferSrc{false};  // the image lacks TRANSFER_SRC usage
    std::atomic<bool> integerFormat{false};       // the image is R8G8B8A8_UINT (not blittable)
    std::atomic<bool> newImage{false};            // the next call makes its buffer and image anew
};

// The errors a presenter reported through onError; any thread.
class ErrorLog {
public:
    void add(const Error& error);
    // An error that could not be stored (out of memory); size() still counts it.
    void addLost() noexcept;
    [[nodiscard]] std::vector<Error> errors() const;
    [[nodiscard]] std::size_t size() const;

private:
    mutable std::mutex mutex_;
    std::vector<Error> errors_;
    std::atomic<std::size_t> lost_{0};
};

// A provider that fills an R8G8B8A8_UNORM image at the swapchain extent with a colour that changes
// every frame (a buffer fill, then a copy). The buffer and image are made anew when the extent
// changes or ProviderControl::newImage is set.
[[nodiscard]] FrameProvider testFrameProvider(std::shared_ptr<ExecutionContext> context,
                                              std::shared_ptr<ProviderControl> control);

// Pumps the window's messages until `done` returns true or `limit` has passed; returns done().
[[nodiscard]] bool pumpUntil(platform::testing::TestWindow& window,
                             const std::function<bool()>& done,
                             std::chrono::milliseconds limit = std::chrono::seconds(5));

// Base of the presentation tests (TEST_CASE_FIXTURE). Presenters are locals of the test body, so
// they are destroyed before the fixture: the context first, then the validation count is checked,
// then the window and the registry go.
class PresentTest {
public:
    PresentTest();
    PresentTest(const PresentTest&) = delete;
    PresentTest& operator=(const PresentTest&) = delete;
    PresentTest(PresentTest&&) = delete;
    PresentTest& operator=(PresentTest&&) = delete;
    ~PresentTest();

    [[nodiscard]] GpuRegistry& registry() const noexcept { return *registry_; }
    [[nodiscard]] ExecutionContext& context() const noexcept { return *context_; }
    [[nodiscard]] const std::shared_ptr<ExecutionContext>& contextPointer() const noexcept {
        return context_;
    }
    [[nodiscard]] platform::testing::TestWindow& window() noexcept { return *window_; }
    [[nodiscard]] ProviderControl& control() const noexcept { return *control_; }
    [[nodiscard]] const ErrorLog& errors() const noexcept { return *errors_; }

    // Options for the fixture's window, provider and error log.
    [[nodiscard]] PresenterOptions options();
    // createPresenter(options()), REQUIREd.
    [[nodiscard]] std::unique_ptr<Presenter> makePresenter();

private:
    std::shared_ptr<GpuRegistry> registry_;
    std::shared_ptr<ExecutionContext> context_;
    std::optional<platform::testing::TestWindow> window_;
    std::shared_ptr<ProviderControl> control_ = std::make_shared<ProviderControl>();
    std::shared_ptr<ErrorLog> errors_ = std::make_shared<ErrorLog>();
};

}  // namespace dvv::gpu::test
