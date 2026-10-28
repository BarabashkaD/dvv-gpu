#include <doctest/doctest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include "dvv/base/error.h"
#include "dvv/base/lifetime.h"
#include "dvv/gpu/device_id.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/gpu_registry.h"
#include "dvv/gpu/presenter.h"
#include "dvv/platform/native_window.h"
#include "dvv/platform/testing/test_window.h"
#include "present_fixture.h"

namespace dvv::gpu::test {

namespace {

// The live Presenter objects in the lifetime registry.
std::int64_t livePresenters() {
    std::int64_t live = 0;
    for (const auto& stats : LifetimeRegistry::instance().typeStats()) {
        if (stats.name.ends_with("Presenter")) {
            live += stats.live;
        }
    }
    return live;
}

}  // namespace

// Design M1b §8.2 case 1.
TEST_CASE_FIXTURE(PresentTest, "a presenter presents frames at the window's size") {
    const std::unique_ptr<Presenter> presenter = makePresenter();
    REQUIRE(pumpUntil(window(), [&] {
        presenter->requestFrame();
        return presenter->stats().framesPresented >= 3;
    }));
    const PresenterStats stats = presenter->stats();
    CHECK(stats.presentedExtent == window().clientSize());
    CHECK(stats.swapchainsCreated >= 1);
    CHECK_FALSE(stats.failed);
    CHECK_FALSE(presenter->failure());
    CHECK(errors().size() == 0);
    CHECK(control().calls.load() >= 3);
}

// Design M1b §8.2 case 6: an unusable image skips the frame without failing the presenter. The
// integer format checks that the presenter refuses what presentBlit refuses (presentBlitSource).
TEST_CASE_FIXTURE(PresentTest,
                  "an image without TRANSFER_SRC is reported, skipped and never blitted") {
    std::string_view expected;
    SUBCASE("without TRANSFER_SRC usage") {
        control().withoutTransferSrc = true;
        expected = "TRANSFER_SRC";
    }
    SUBCASE("with an integer format") {
        control().integerFormat = true;
        expected = "R8G8B8A8Uint";
    }
    const std::unique_ptr<Presenter> presenter = makePresenter();
    REQUIRE(pumpUntil(window(), [&] {
        presenter->requestFrame();
        return errors().size() >= 1;
    }));
    const auto reported = errors().errors();
    REQUIRE_FALSE(reported.empty());
    CHECK(reported.front().code == Errc::Internal);
    INFO(reported.front().message);
    CHECK(reported.front().message.find(expected) != std::string::npos);
    const PresenterStats stats = presenter->stats();
    CHECK(stats.frameErrors >= 1);
    CHECK(stats.framesPresented == 0);
    CHECK_FALSE(stats.failed);
    CHECK_FALSE(presenter->failure());
}

// Design M1b §8.2 case 7: every round stops the presenter while a frame is acquiring or in flight.
// On Windows it guards R15 (a join that does not serve the window's sent messages could hang
// here); not observed to hang without the dispatching wait on this machine.
TEST_CASE_FIXTURE(PresentTest,
                  "destroying a presenter right after a request neither hangs nor "
                  "leaks") {
    for (int round = 0; round < 20; ++round) {
        CAPTURE(round);
        std::unique_ptr<Presenter> presenter = makePresenter();
        REQUIRE(pumpUntil(window(), [&] {
            presenter->requestFrame();
            return presenter->stats().framesPresented >= 1;
        }));
        const std::uint64_t calls = control().calls;
        presenter->requestFrame();
        REQUIRE(pumpUntil(window(), [&] { return control().calls > calls; }));
        const auto start = std::chrono::steady_clock::now();
        presenter.reset();
        const auto elapsed = std::chrono::steady_clock::now() - start;
        CHECK(elapsed < std::chrono::seconds(6));
    }
    CHECK(errors().size() == 0);
}

// Design M1b §8.2 case 9.
TEST_CASE_FIXTURE(PresentTest, "wrong configurations are refused with Internal") {
    SUBCASE("a context without presentation") {
        const auto best = registry().bestDevice();
        REQUIRE(best.has_value());
        const auto headless_context = registry().createContext(best.value_or(DeviceId(0)), {});
        REQUIRE(headless_context.has_value());
        const auto result = (*headless_context)->createPresenter(options());
        REQUIRE_FALSE(result);
        CHECK(result.error().code == Errc::Internal);
    }
    SUBCASE("no window") {
        PresenterOptions no_window = options();
        no_window.window = std::monostate{};
        const auto result = context().createPresenter(std::move(no_window));
        REQUIRE_FALSE(result);
        CHECK(result.error().code == Errc::Internal);
    }
    SUBCASE("a window of the other platform") {
        PresenterOptions other = options();
#if defined(_WIN32)
        other.window = platform::XlibWindow{};
#else
        other.window = platform::Win32Window{};
#endif
        const auto result = context().createPresenter(std::move(other));
        REQUIRE_FALSE(result);
        CHECK(result.error().code == Errc::Internal);
    }
    SUBCASE("no frame provider") {
        PresenterOptions no_frames = options();
        no_frames.frames = {};
        const auto result = context().createPresenter(std::move(no_frames));
        REQUIRE_FALSE(result);
        CHECK(result.error().code == Errc::Internal);
    }
    SUBCASE("presentation on a headless registry") {
        const auto headless = GpuRegistry::create({});
        REQUIRE(headless.has_value());
        const auto best = (*headless)->bestDevice();
        REQUIRE(best.has_value());
        const auto result =
            (*headless)->createContext(best.value_or(DeviceId(0)), {.presentation = true});
        REQUIRE_FALSE(result);
        CHECK(result.error().code == Errc::Internal);
        CHECK((*headless)->validationMessageCount() == 0);
    }
}

// Design M1b §8.2 case 10.
TEST_CASE("a presenter's objects are gone after it is destroyed") {
    const std::size_t before = LifetimeRegistry::instance().liveCount();
    auto registry = GpuRegistry::create({.windowSystem = platform::defaultWindowSystem()});
    REQUIRE(registry.has_value());
    const auto best = (*registry)->bestDevice();
    REQUIRE(best.has_value());
    auto context = (*registry)->createContext(best.value_or(DeviceId(0)), {.presentation = true});
    REQUIRE(context.has_value());
    auto window =
        platform::testing::TestWindow::create({.width = 320, .height = 240}, "dvv present test");
    REQUIRE(window.has_value());
    auto presenter = (*context)->createPresenter(
        {.window = window->native(),
         .extent = window->clientSize(),
         .visible = true,
         .frames = testFrameProvider(*context, std::make_shared<ProviderControl>()),
         .onError = {},
         .name = "present test"});
    REQUIRE(presenter.has_value());
    REQUIRE(pumpUntil(*window, [&] {
        (*presenter)->requestFrame();
        return (*presenter)->stats().framesPresented >= 1;
    }));
    CHECK(livePresenters() == 1);
    presenter->reset();
    CHECK(livePresenters() == 0);
    context->reset();
    CHECK((*registry)->validationMessageCount() == 0);
    registry->reset();
    CHECK(LifetimeRegistry::instance().liveCount() == before);
}

}  // namespace dvv::gpu::test
