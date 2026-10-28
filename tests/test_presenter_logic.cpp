#include <doctest/doctest.h>

#include <cstdint>
#include <optional>

#include <vulkan/vulkan_core.h>

#include "dvv/base/error.h"
#include "dvv/platform/native_window.h"
#include "presenter_logic.h"

using dvv::gpu::detail::outcomeOf;
using dvv::gpu::detail::PresenterLogic;
using dvv::gpu::detail::VkOutcome;
using dvv::platform::PixelSize;
using Action = PresenterLogic::Action;

namespace {

constexpr PixelSize SIZE{.width = 320, .height = 240};

// Design 5.7 case 5 fixes the number.
static_assert(PresenterLogic::OUT_OF_DATE_LIMIT == 4);

// Renders, completes and presents one frame; the logic must be about to render.
void completeFrame(PresenterLogic& logic) {
    REQUIRE(logic.next() == Action::RenderFrame);
    logic.acquired(VkOutcome::Success);
    logic.frameStarted();
    REQUIRE(logic.next() == Action::Wait);
    logic.frameCompleted(std::nullopt);
    REQUIRE(logic.next() == Action::Present);
    logic.presented(VkOutcome::Success);
}

// A logic that has created its swapchain and presented the first frame.
PresenterLogic presenting() {
    PresenterLogic logic;
    logic.setRequests(SIZE, true, 1, false);
    REQUIRE(logic.next() == Action::RecreateSwapchain);
    logic.swapchainCreated(SIZE);
    completeFrame(logic);
    REQUIRE(logic.next() == Action::Wait);
    return logic;
}

}  // namespace

TEST_CASE("outcomeOf sorts Vulkan results") {
    CHECK(outcomeOf(VK_SUCCESS) == VkOutcome::Success);
    CHECK(outcomeOf(VK_SUBOPTIMAL_KHR) == VkOutcome::Suboptimal);
    CHECK(outcomeOf(VK_ERROR_OUT_OF_DATE_KHR) == VkOutcome::OutOfDate);
    CHECK(outcomeOf(VK_TIMEOUT) == VkOutcome::Timeout);
    CHECK(outcomeOf(VK_NOT_READY) == VkOutcome::Timeout);
    CHECK(outcomeOf(VK_ERROR_SURFACE_LOST_KHR) == VkOutcome::SurfaceLost);
    CHECK(outcomeOf(VK_ERROR_DEVICE_LOST) == VkOutcome::DeviceLost);
    CHECK(outcomeOf(VK_ERROR_OUT_OF_HOST_MEMORY) == VkOutcome::OtherError);
    CHECK(outcomeOf(VK_ERROR_NATIVE_WINDOW_IN_USE_KHR) == VkOutcome::OtherError);
}

TEST_CASE("a new logic is hidden and waits") {
    const PresenterLogic logic;
    CHECK(logic.next() == Action::Wait);
    CHECK(logic.hidden());
    CHECK_FALSE(logic.failed());
    CHECK_FALSE(logic.inFlight());
    CHECK_FALSE(logic.previousFrameCompleted());
}

TEST_CASE("the first request creates the swapchain, then renders and presents one frame") {
    const PresenterLogic logic = presenting();
    CHECK(logic.previousFrameCompleted());
    CHECK(logic.next() == Action::Wait);
    CHECK(logic.extentHint() == SIZE);
}

TEST_CASE("no frame while hidden or 0-sized; the request is served once visible") {
    PresenterLogic logic = presenting();
    logic.setRequests(SIZE, false, 2, false);
    CHECK(logic.hidden());
    CHECK(logic.next() == Action::Wait);
    logic.setRequests({.width = 0, .height = 0}, true, 3, false);
    CHECK(logic.hidden());
    CHECK(logic.next() == Action::Wait);
    logic.setRequests(SIZE, true, 4, false);
    CHECK_FALSE(logic.hidden());
    CHECK(logic.next() == Action::RenderFrame);
    completeFrame(logic);
    CHECK(logic.next() == Action::Wait);  // one frame only
}

TEST_CASE("several resizes give one recreation at the latest hint") {
    PresenterLogic logic = presenting();
    logic.setRequests({.width = 640, .height = 480}, true, 2, false);
    logic.setRequests({.width = 97, .height = 61}, true, 3, false);
    logic.setRequests({.width = 800, .height = 600}, true, 4, false);
    REQUIRE(logic.next() == Action::RecreateSwapchain);
    CHECK(logic.extentHint() == PixelSize{.width = 800, .height = 600});
    logic.swapchainCreated({.width = 800, .height = 600});
    CHECK(logic.next() == Action::RenderFrame);
    completeFrame(logic);
    CHECK(logic.next() == Action::Wait);
    // The same hint never re-creates, whatever the surface's real size.
    logic.setRequests({.width = 800, .height = 600}, true, 5, false);
    CHECK(logic.next() == Action::RenderFrame);
    completeFrame(logic);
    logic.setRequests({.width = 801, .height = 600}, true, 6, false);
    CHECK(logic.next() == Action::RecreateSwapchain);
}

TEST_CASE("requests during a frame in flight give exactly one more frame") {
    PresenterLogic logic = presenting();
    logic.setRequests(SIZE, true, 2, false);
    REQUIRE(logic.next() == Action::RenderFrame);
    logic.acquired(VkOutcome::Success);
    logic.frameStarted();
    for (const std::uint64_t seq : {3U, 4U, 5U}) {
        logic.setRequests(SIZE, true, seq, false);
        CHECK(logic.next() == Action::Wait);
        CHECK(logic.inFlight());
    }
    logic.frameCompleted(std::nullopt);
    CHECK(logic.next() == Action::Present);
    logic.presented(VkOutcome::Success);
    CHECK(logic.next() == Action::RenderFrame);
    completeFrame(logic);
    CHECK(logic.next() == Action::Wait);

    // A completed frame is presented even when the surface was hidden meanwhile.
    PresenterLogic hiding = presenting();
    hiding.setRequests(SIZE, true, 2, false);
    REQUIRE(hiding.next() == Action::RenderFrame);
    hiding.acquired(VkOutcome::Success);
    hiding.frameStarted();
    hiding.setRequests(SIZE, false, 3, false);
    hiding.frameCompleted(std::nullopt);
    CHECK(hiding.next() == Action::Present);
    hiding.presented(VkOutcome::Success);
    CHECK(hiding.hidden());
    CHECK(hiding.next() == Action::Wait);
}

TEST_CASE("out-of-date and suboptimal results recreate the swapchain before the next frame") {
    SUBCASE("acquire out-of-date") {
        PresenterLogic logic = presenting();
        logic.setRequests(SIZE, true, 2, false);
        REQUIRE(logic.next() == Action::RenderFrame);
        logic.acquired(VkOutcome::OutOfDate);
        CHECK(logic.next() == Action::RecreateSwapchain);  // the request is still pending
        logic.swapchainCreated(SIZE);
        CHECK(logic.next() == Action::RenderFrame);
    }
    SUBCASE("present out-of-date") {
        PresenterLogic logic = presenting();
        logic.setRequests(SIZE, true, 2, false);
        REQUIRE(logic.next() == Action::RenderFrame);
        logic.acquired(VkOutcome::Success);
        logic.frameStarted();
        logic.frameCompleted(std::nullopt);
        REQUIRE(logic.next() == Action::Present);
        logic.presented(VkOutcome::OutOfDate);
        CHECK(logic.next() == Action::RecreateSwapchain);  // no new GUI request
        logic.swapchainCreated(SIZE);
        CHECK(logic.next() == Action::RenderFrame);
    }
    SUBCASE("acquire suboptimal") {
        PresenterLogic logic = presenting();
        logic.setRequests(SIZE, true, 2, false);
        REQUIRE(logic.next() == Action::RenderFrame);
        logic.acquired(VkOutcome::Suboptimal);
        logic.frameStarted();
        logic.frameCompleted(std::nullopt);
        logic.presented(VkOutcome::Success);
        CHECK(logic.next() == Action::Wait);
        logic.setRequests(SIZE, true, 3, false);
        CHECK(logic.next() == Action::RecreateSwapchain);
    }
    SUBCASE("present suboptimal") {
        PresenterLogic logic = presenting();
        logic.setRequests(SIZE, true, 2, false);
        REQUIRE(logic.next() == Action::RenderFrame);
        logic.acquired(VkOutcome::Success);
        logic.frameStarted();
        logic.frameCompleted(std::nullopt);
        logic.presented(VkOutcome::Suboptimal);
        CHECK(logic.next() == Action::Wait);
        logic.setRequests(SIZE, true, 3, false);
        CHECK(logic.next() == Action::RecreateSwapchain);
    }
}

TEST_CASE("four out-of-date results in a row hide the surface until a GUI event") {
    PresenterLogic logic = presenting();
    logic.setRequests(SIZE, true, 2, false);
    for (int attempt = 1; attempt < PresenterLogic::OUT_OF_DATE_LIMIT; ++attempt) {
        REQUIRE(logic.next() == Action::RenderFrame);
        logic.acquired(VkOutcome::OutOfDate);
        REQUIRE(logic.next() == Action::RecreateSwapchain);
        logic.swapchainCreated(SIZE);
    }
    REQUIRE(logic.next() == Action::RenderFrame);
    logic.acquired(VkOutcome::OutOfDate);
    CHECK(logic.hidden());
    CHECK(logic.next() == Action::Wait);
    logic.setRequests(SIZE, true, 2, false);  // no GUI event
    CHECK(logic.hidden());
    CHECK(logic.next() == Action::Wait);
    logic.setRequests(SIZE, true, 3, false);
    CHECK_FALSE(logic.hidden());
    CHECK(logic.next() == Action::RecreateSwapchain);
}

TEST_CASE(
    "the out-of-date streak counts acquire and present results in a row and resets on success") {
    const auto check_streak = [](VkOutcome separator) {
        PresenterLogic logic = presenting();
        logic.setRequests(SIZE, true, 2, false);
        // (1) Three acquire out-of-date results.
        for (int attempt = 0; attempt < 3; ++attempt) {
            REQUIRE(logic.next() == Action::RenderFrame);
            logic.acquired(VkOutcome::OutOfDate);
            CHECK_FALSE(logic.hidden());
            REQUIRE(logic.next() == Action::RecreateSwapchain);
            logic.swapchainCreated(SIZE);
        }
        // (2) A good acquire resets the streak; the present's out-of-date result counts 1.
        REQUIRE(logic.next() == Action::RenderFrame);
        logic.acquired(separator);
        logic.frameStarted();
        logic.frameCompleted(std::nullopt);
        REQUIRE(logic.next() == Action::Present);
        logic.presented(VkOutcome::OutOfDate);
        CHECK_FALSE(logic.hidden());
        REQUIRE(logic.next() == Action::RecreateSwapchain);
        logic.swapchainCreated(SIZE);
        // (3) Two more acquire out-of-date results: 3 in a row.
        for (int attempt = 0; attempt < 2; ++attempt) {
            REQUIRE(logic.next() == Action::RenderFrame);
            logic.acquired(VkOutcome::OutOfDate);
            CHECK_FALSE(logic.hidden());
            REQUIRE(logic.next() == Action::RecreateSwapchain);
            logic.swapchainCreated(SIZE);
        }
        // (4) The 4th in a row, counting the present, hides the surface.
        REQUIRE(logic.next() == Action::RenderFrame);
        logic.acquired(VkOutcome::OutOfDate);
        CHECK(logic.hidden());
        CHECK(logic.next() == Action::Wait);
        // (5) Hiding reset the streak: one more out-of-date result does not hide.
        logic.setRequests(SIZE, true, 3, false);
        REQUIRE(logic.next() == Action::RecreateSwapchain);
        logic.swapchainCreated(SIZE);
        REQUIRE(logic.next() == Action::RenderFrame);
        logic.acquired(VkOutcome::OutOfDate);
        CHECK_FALSE(logic.hidden());
    };
    SUBCASE("acquired(Success)") {
        check_streak(VkOutcome::Success);
    }
    SUBCASE("acquired(Suboptimal)") {
        check_streak(VkOutcome::Suboptimal);
    }
}

TEST_CASE("a timeout drops the frame and retries after the deadline") {
    PresenterLogic logic = presenting();
    logic.setRequests(SIZE, true, 2, false);
    REQUIRE(logic.next() == Action::RenderFrame);
    logic.acquired(VkOutcome::Timeout);
    CHECK(logic.next() == Action::WaitAndRetry);
    logic.setRequests(SIZE, false, 2, false);  // the retry comes before the hidden state
    CHECK(logic.next() == Action::WaitAndRetry);
    logic.setRequests(SIZE, true, 2, false);
    CHECK(logic.next() == Action::WaitAndRetry);
    logic.retryElapsed();
    CHECK(logic.next() == Action::RenderFrame);
    CHECK_FALSE(logic.previousFrameCompleted());
}

TEST_CASE("a lost surface fails the presenter") {
    SUBCASE("on acquire") {
        PresenterLogic logic = presenting();
        logic.setRequests(SIZE, true, 2, false);
        REQUIRE(logic.next() == Action::RenderFrame);
        logic.acquired(VkOutcome::SurfaceLost);
        CHECK(logic.failed());
        CHECK(logic.next() == Action::Exit);
    }
    SUBCASE("on present") {
        for (const VkOutcome outcome : {VkOutcome::SurfaceLost, VkOutcome::DeviceLost,
                                        VkOutcome::OtherError, VkOutcome::Timeout}) {
            CAPTURE(static_cast<int>(outcome));
            PresenterLogic logic = presenting();
            logic.setRequests(SIZE, true, 2, false);
            REQUIRE(logic.next() == Action::RenderFrame);
            logic.acquired(VkOutcome::Success);
            logic.frameStarted();
            logic.frameCompleted(std::nullopt);
            REQUIRE(logic.next() == Action::Present);
            logic.presented(outcome);
            CHECK(logic.failed());
            CHECK(logic.next() == Action::Exit);
        }
    }
}

TEST_CASE("a lost device fails the presenter for good") {
    PresenterLogic logic = presenting();
    logic.setRequests(SIZE, true, 2, false);
    REQUIRE(logic.next() == Action::RenderFrame);
    logic.acquired(VkOutcome::DeviceLost);
    CHECK(logic.next() == Action::Exit);
    logic.setRequests(SIZE, true, 99, false);
    CHECK(logic.next() == Action::Exit);
    CHECK(logic.failed());
}

TEST_CASE("a failed frame fails the presenter without presenting") {
    for (const dvv::Errc error :
         {dvv::Errc::DeviceLost, dvv::Errc::OutOfMemory, dvv::Errc::Internal}) {
        CAPTURE(dvv::errcName(error));
        PresenterLogic logic = presenting();
        logic.setRequests(SIZE, true, 2, false);
        REQUIRE(logic.next() == Action::RenderFrame);
        logic.acquired(VkOutcome::Success);
        logic.frameStarted();
        logic.frameCompleted(error);
        CHECK(logic.failed());
        CHECK_FALSE(logic.inFlight());
        CHECK_FALSE(logic.previousFrameCompleted());
        CHECK(logic.next() == Action::Exit);
    }
}

TEST_CASE("stop during a frame in flight waits for it and exits without presenting") {
    PresenterLogic logic = presenting();
    logic.setRequests(SIZE, true, 2, false);
    REQUIRE(logic.next() == Action::RenderFrame);
    logic.acquired(VkOutcome::Success);  // the fence wait has ended
    logic.frameStarted();
    logic.setRequests(SIZE, true, 2, true);
    CHECK(logic.next() == Action::Wait);
    logic.frameCompleted(std::nullopt);
    CHECK(logic.next() == Action::Exit);
}

TEST_CASE("a failed acquire fence wait fails the presenter") {
    for (const VkOutcome outcome : {VkOutcome::OtherError, VkOutcome::DeviceLost}) {
        CAPTURE(static_cast<int>(outcome));
        PresenterLogic logic = presenting();
        logic.setRequests(SIZE, true, 2, false);
        REQUIRE(logic.next() == Action::RenderFrame);
        logic.acquired(outcome);  // OtherError: the 5 s cap
        CHECK(logic.failed());
        CHECK(logic.next() == Action::Exit);
    }
}

TEST_CASE("stop while idle exits at once") {
    PresenterLogic fresh;
    fresh.setRequests(SIZE, true, 1, true);
    CHECK(fresh.next() == Action::Exit);

    PresenterLogic logic = presenting();
    logic.setRequests(SIZE, false, 2, true);  // hidden, a request pending
    CHECK(logic.next() == Action::Exit);
    // Stop is sticky.
    logic.setRequests(SIZE, true, 3, false);
    CHECK(logic.next() == Action::Exit);
}

TEST_CASE("a request that arrives during the provider call or the acquire is not lost") {
    PresenterLogic logic = presenting();
    logic.setRequests(SIZE, true, 2, false);
    REQUIRE(logic.next() == Action::RenderFrame);
    // The GUI raises the counter to 3 now, unseen by the presenter until its next loop.
    logic.acquired(VkOutcome::Success);
    logic.frameStarted();
    logic.frameCompleted(std::nullopt);
    logic.presented(VkOutcome::Success);
    logic.setRequests(SIZE, true, 3, false);
    CHECK(logic.next() == Action::RenderFrame);
}

TEST_CASE("a surface of extent 0 hides until a new request") {
    PresenterLogic logic;
    logic.setRequests(SIZE, true, 1, false);
    REQUIRE(logic.next() == Action::RecreateSwapchain);
    logic.surfaceHidden();
    CHECK(logic.hidden());
    CHECK(logic.next() == Action::Wait);
    logic.setRequests(SIZE, true, 1, false);
    CHECK(logic.next() == Action::Wait);
    logic.setRequests(SIZE, true, 2, false);
    CHECK_FALSE(logic.hidden());
    CHECK(logic.next() == Action::RecreateSwapchain);
}

TEST_CASE("previousFrameCompleted reports whether the last provider call's frame ran") {
    CHECK_FALSE(PresenterLogic{}.previousFrameCompleted());
    PresenterLogic logic = presenting();
    CHECK(logic.previousFrameCompleted());

    logic.setRequests(SIZE, true, 2, false);
    REQUIRE(logic.next() == Action::RenderFrame);
    logic.acquired(VkOutcome::Timeout);
    CHECK_FALSE(logic.previousFrameCompleted());
    logic.retryElapsed();
    completeFrame(logic);
    CHECK(logic.previousFrameCompleted());

    logic.setRequests(SIZE, true, 3, false);
    REQUIRE(logic.next() == Action::RenderFrame);
    logic.frameSkipped();
    CHECK_FALSE(logic.previousFrameCompleted());

    logic.setRequests(SIZE, true, 4, false);
    completeFrame(logic);
    CHECK(logic.previousFrameCompleted());

    logic.setRequests(SIZE, true, 5, false);
    REQUIRE(logic.next() == Action::RenderFrame);
    logic.acquired(VkOutcome::OutOfDate);
    CHECK_FALSE(logic.previousFrameCompleted());
}

TEST_CASE("a skipped frame serves its request and stays idle") {
    PresenterLogic logic = presenting();
    logic.setRequests(SIZE, true, 2, false);
    REQUIRE(logic.next() == Action::RenderFrame);
    logic.frameSkipped();
    CHECK(logic.next() == Action::Wait);
    CHECK_FALSE(logic.inFlight());
    logic.setRequests(SIZE, true, 3, false);
    CHECK(logic.next() == Action::RenderFrame);
}
