#include <doctest/doctest.h>

#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "dvv/base/error.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/presenter.h"
#include "dvv/platform/native_window.h"
#include "dvv/platform/testing/test_window.h"
#include "present_fixture.h"

namespace dvv::gpu::test {

namespace {

using namespace std::chrono_literals;

// Requests frames until the presenter has presented at least one.
void presentFirstFrame(platform::testing::TestWindow& window, Presenter& presenter) {
    REQUIRE(pumpUntil(window, [&] {
        presenter.requestFrame();
        return presenter.stats().framesPresented >= 1;
    }));
}

// Pumps the window for `duration`, requesting a frame on every iteration.
void pumpRequesting(platform::testing::TestWindow& window, Presenter& presenter,
                    std::chrono::milliseconds duration) {
    static_cast<void>(pumpUntil(
        window,
        [&] {
            presenter.requestFrame();
            return false;
        },
        duration));
}

}  // namespace

// Design M1b §8.2 case 2: each new hint recreates the swapchain once, at the window's real extent
// (currentExtent stays authoritative, D8).
TEST_CASE_FIXTURE(PresentTest, "each resize presents at the new extent") {
    const std::unique_ptr<Presenter> p = makePresenter();
    presentFirstFrame(window(), *p);
    for (const platform::PixelSize size : {platform::PixelSize{.width = 640, .height = 480},
                                           platform::PixelSize{.width = 97, .height = 61},
                                           platform::PixelSize{.width = 640, .height = 480}}) {
        CAPTURE(size.width);
        CAPTURE(size.height);
        const std::uint64_t created = p->stats().swapchainsCreated;
        window().resize(size);
        p->resize(window().clientSize());
        REQUIRE(pumpUntil(window(), [&] {
            p->requestFrame();
            const PresenterStats s = p->stats();
            return s.presentedExtent == window().clientSize() && s.swapchainsCreated > created;
        }));
        CHECK(window().clientSize() == size);
    }
    CHECK_FALSE(p->failure());
    CHECK(errors().size() == 0);
}

// Design M1b §8.2 case 3: a minimised window hides the presenter without failing it; restoring
// resumes the frames.
TEST_CASE_FIXTURE(PresentTest, "minimise and restore") {
    const std::unique_ptr<Presenter> p = makePresenter();
    presentFirstFrame(window(), *p);
#if defined(_WIN32)
    // The 0-extent path (D8): the hint differs from the last creation's hint, which forces a
    // recreation; currentExtent (0x0 while minimised) decides, not the hint.
    window().minimise();
    p->resize({.width = 321, .height = 241});
    REQUIRE(pumpUntil(window(), [&] {
        p->requestFrame();
        return p->stats().hidden;
    }));
    CHECK_FALSE(p->stats().failed);
#endif
    // As the GUI does on both platforms.
    p->setVisible(false);
#if !defined(_WIN32)
    window().minimise();  // Xlib: unmap (not run in v1.0, U-WIN)
#endif
    static_cast<void>(pumpUntil(window(), [] { return false; }, 100ms));

    window().restore();
    p->resize(window().clientSize());
    p->setVisible(true);
    const std::uint64_t n = p->stats().framesPresented;
    REQUIRE(pumpUntil(window(), [&] {
        p->requestFrame();
        const PresenterStats s = p->stats();
        return s.framesPresented > n && !s.hidden;
    }));
    CHECK_FALSE(p->failure());
    CHECK(errors().size() == 0);
}

// Design M1b §8.2 case 4: one frame already in flight may still be presented.
TEST_CASE_FIXTURE(PresentTest, "setVisible(false) stops frames until setVisible(true)") {
    const std::unique_ptr<Presenter> p = makePresenter();
    presentFirstFrame(window(), *p);
    p->setVisible(false);
    const std::uint64_t n = p->stats().framesPresented;
    pumpRequesting(window(), *p, 300ms);
    CHECK(p->stats().framesPresented <= n + 1);
    CHECK(p->stats().hidden);
    p->setVisible(true);
    REQUIRE(pumpUntil(window(), [&] {
        p->requestFrame();
        return p->stats().framesPresented > n + 1;
    }));
    CHECK_FALSE(p->failure());
    CHECK(errors().size() == 0);
}

// Design M1b §8.2 case 5.
TEST_CASE_FIXTURE(PresentTest, "a provider error is reported once and the next frame presents") {
    const std::unique_ptr<Presenter> p = makePresenter();
    presentFirstFrame(window(), *p);
    control().failNext = true;
    p->requestFrame();
    REQUIRE(pumpUntil(window(), [&] { return errors().size() >= 1; }));
    CHECK(errors().size() == 1);
    const auto reported = errors().errors();
    REQUIRE(reported.size() == 1);
    CHECK(reported[0].code == Errc::Io);
    INFO(reported[0].message);
    CHECK(reported[0].message.find("injected provider failure") != std::string::npos);
    CHECK(p->stats().frameErrors == 1);
    const std::uint64_t n = p->stats().framesPresented;
    REQUIRE(pumpUntil(window(), [&] {
        p->requestFrame();
        return p->stats().framesPresented > n;
    }));
    CHECK_FALSE(p->failure());
    CHECK(errors().size() == 1);
}

// Design M1b §8.2 case 8: two presenter threads share the context's queue lock and submit thread.
// Windows has no TSan (U-WIN): this case runs under vs2026-asan.
TEST_CASE_FIXTURE(PresentTest, "two presenters on one context present concurrently") {
    auto second =
        platform::testing::TestWindow::create({.width = 200, .height = 150}, "dvv present test 2");
    REQUIRE(second.has_value());
    const auto second_control = std::make_shared<ProviderControl>();
    const auto second_errors = std::make_shared<ErrorLog>();
    const std::unique_ptr<Presenter> p = makePresenter();
    auto created =
        context().createPresenter({.window = second->native(),
                                   .extent = second->clientSize(),
                                   .visible = true,
                                   .frames = testFrameProvider(contextPointer(), second_control),
                                   .onError =
                                       [log = second_errors](const Error& error) noexcept {
                                           try {
                                               log->add(error);
                                           } catch (const std::exception&) {
                                               log->addLost();
                                           }
                                       },
                                   .name = "present test 2"});
    REQUIRE(created.has_value());
    std::unique_ptr<Presenter> q = std::move(*created);

    const auto until = std::chrono::steady_clock::now() + 1s;
    while (std::chrono::steady_clock::now() < until) {
        p->requestFrame();
        q->requestFrame();
        window().pump();
        second->pump();
        std::this_thread::sleep_for(5ms);
    }
    const auto both_presented = [&] {
        return p->stats().framesPresented >= 1 && q->stats().framesPresented >= 1;
    };
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!both_presented() && std::chrono::steady_clock::now() < deadline) {
        p->requestFrame();
        q->requestFrame();
        window().pump();
        second->pump();
        std::this_thread::sleep_for(5ms);
    }
    REQUIRE(both_presented());
    CHECK_FALSE(p->stats().failed);
    CHECK_FALSE(q->stats().failed);
    CHECK_FALSE(q->failure());
    CHECK(second_errors->size() == 0);
    q.reset();  // before the second window
    CHECK_FALSE(p->failure());
    CHECK(errors().size() == 0);
}

}  // namespace dvv::gpu::test
