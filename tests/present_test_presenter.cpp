#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <format>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <vulkan/vulkan_core.h>

#include "access.h"
#include "device_core.h"
#include "dvv/base/error.h"
#include "dvv/base/lifetime.h"
#include "dvv/gpu/command_recorder.h"
#include "dvv/gpu/context_vulkan.h"
#include "dvv/gpu/device_id.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/gpu_instance.h"
#include "dvv/gpu/presenter.h"
#include "dvv/gpu/presenter_vulkan.h"
#include "dvv/gpu/resources.h"
#include "dvv/platform/pixel_size.h"
#include "dvv/platform/testing/test_window.h"
#include "dvv/platform/window_handle.h"
#include "present_blit.h"
#include "present_fixture.h"
#include "presenter_core.h"
#include "surface_vulkan.h"

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

TEST_CASE_FIXTURE(PresentTest, "the per-call encoding matches the swapchain format") {
    struct Record {
        PresentEncoding encoding = PresentEncoding::Linear;
    };
    // The provider runs on the presenter thread, the checks on the test thread.
    std::mutex mutex;
    std::vector<Record> records;
    FrameProvider recording =
        [&mutex, &records,
         frames = provider()](const PresentFrameParams& params) mutable -> Result<PresentFrame> {
        {
            const std::lock_guard lock(mutex);
            records.push_back({.encoding = params.encoding});
        }
        return frames(params);
    };
    auto presenter = createPresenter(context(), options(), std::move(recording));
    REQUIRE(presenter.has_value());
    if (!presenter) {
        return;
    }
    REQUIRE(pumpUntil(window(), [&] {
        (*presenter)->requestFrame();
        return (*presenter)->stats().framesPresented >= 3;
    }));
    presenter->reset();  // no provider call after this
    // The format the swapchain takes: the choice over the surface's formats.
    const SurfaceInputs inputs = surfaceInputs(gpuInstance());
    const auto surface =
        detail::createSurface(inputs.instance, inputs.getInstanceProcAddr, window().handle());
    REQUIRE(surface.has_value());
    if (!surface) {
        return;
    }
    const detail::DeviceCore& device = detail::Access::core(context());
    const auto& api = device.dispatch();
    std::vector<vk::SurfaceFormatKHR> listed_formats;
    vk::Result listed = vk::Result::eIncomplete;
    while (listed == vk::Result::eIncomplete) {
        std::uint32_t count = 0;
        REQUIRE(device.physicalDevice().getSurfaceFormatsKHR(static_cast<vk::SurfaceKHR>(*surface),
                                                             &count, nullptr,
                                                             api) == vk::Result::eSuccess);
        listed_formats.resize(count);
        listed = device.physicalDevice().getSurfaceFormatsKHR(static_cast<vk::SurfaceKHR>(*surface),
                                                              &count, listed_formats.data(), api);
        listed_formats.resize(count);
    }
    REQUIRE(listed == vk::Result::eSuccess);
    const std::vector<VkSurfaceFormatKHR> formats(listed_formats.begin(), listed_formats.end());
    const std::optional<VkSurfaceFormatKHR> f = detail::chooseSurfaceFormat(formats);
    REQUIRE(f);
    // Before the window goes.
    api.vkDestroySurfaceKHR(inputs.instance, *surface, nullptr);
    const std::lock_guard lock(mutex);
    REQUIRE(records.size() >= 3);
    if (f) {
        for (const Record& record : records) {
            CHECK(record.encoding == detail::presentEncoding(f->format));
        }
    }
    CHECK(errors().size() == 0);
}

TEST_CASE("PresentFrameParams builds with and without encoding") {
    // clang 19 (the Linux CI image) and Apple clang of Xcode 27 do not warn about an omitted
    // designated field that has a default member initialiser.
    const PresentFrameParams with_encoding{.extent = {},
                                           .frameNumber = 0,
                                           .previousFrameCompleted = false,
                                           .encoding = PresentEncoding::Srgb};
    CHECK(with_encoding.encoding == PresentEncoding::Srgb);
    const auto [extent, frame_number, previous_completed, encoding] = with_encoding;
    CHECK(extent.width == 0);
    CHECK(frame_number == 0);
    CHECK_FALSE(previous_completed);
    CHECK(encoding == PresentEncoding::Srgb);
    const PresentFrameParams without_encoding{
        .extent = {}, .frameNumber = 0, .previousFrameCompleted = false};
    CHECK(without_encoding.encoding == PresentEncoding::Linear);
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
        const auto best = gpuInstance().bestDevice();
        REQUIRE(best.has_value());
        const auto headless_context = gpuInstance().createContext(best.value_or(DeviceId(0)), {});
        REQUIRE(headless_context.has_value());
        const auto result = createPresenter(**headless_context, options(), provider());
        REQUIRE_FALSE(result);
        CHECK(result.error().code == Errc::Internal);
    }
    SUBCASE("no window") {
        PresenterOptions no_window = options();
        no_window.window = platform::WindowHandle{};
        const std::string name = no_window.name;
        const auto result = createPresenter(context(), std::move(no_window), provider());
        REQUIRE_FALSE(result);
        CHECK(result.error().code == Errc::Internal);
        CHECK(
            result.error().message ==
            std::format("presenter '{}': the window is no window, the GPU instance presents to {}",
                        name, platform::windowSystemName(platform::defaultWindowSystem())));
    }
    SUBCASE("a window of the other platform") {
        PresenterOptions other = options();
        const platform::WindowSystem other_system =
            platform::defaultWindowSystem() == platform::WindowSystem::Win32
                ? platform::WindowSystem::Xlib
                : platform::WindowSystem::Win32;
        other.window = platform::WindowHandle::make(other_system, 1, 1);
        const std::string name = other.name;
        const auto result = createPresenter(context(), std::move(other), provider());
        REQUIRE_FALSE(result);
        CHECK(result.error().code == Errc::Internal);
        CHECK(result.error().message ==
              std::format("presenter '{}': the window is {}, the GPU instance presents to {}", name,
                          platform::windowSystemName(other_system),
                          platform::windowSystemName(platform::defaultWindowSystem())));
    }
    SUBCASE("no frame provider") {
        const auto result = createPresenter(context(), options(), FrameProvider{});
        REQUIRE_FALSE(result);
        CHECK(result.error().code == Errc::Internal);
    }
    SUBCASE("presentation on a headless GPU instance") {
        const auto headless = GpuInstance::create({});
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

// A handle of the right system made before its native window exists: the surface creator refuses
// it.
TEST_CASE_FIXTURE(PresentTest,
                  "a window handle without a native window is refused with Unsupported") {
    PresenterOptions null_window = options();
    null_window.window = platform::WindowHandle::make(platform::defaultWindowSystem(), 0, 0);
    const std::string name = null_window.name;
    const auto result = createPresenter(context(), std::move(null_window), provider());
    REQUIRE_FALSE(result);
    CHECK(result.error().code == Errc::Unsupported);
    CHECK(result.error().message.starts_with(
        std::format("creating the window surface of presenter '{}'", name)));
    CHECK(livePresenters() == 0);
}

// Design M1b §8.2 case 10.
TEST_CASE("a presenter's objects are gone after it is destroyed") {
    const std::size_t before = LifetimeRegistry::instance().liveCount();
    auto gpu_instance = GpuInstance::create({.windowSystem = platform::defaultWindowSystem()});
    REQUIRE(gpu_instance.has_value());
    const auto best = (*gpu_instance)->bestDevice();
    REQUIRE(best.has_value());
    auto context =
        (*gpu_instance)->createContext(best.value_or(DeviceId(0)), {.presentation = true});
    REQUIRE(context.has_value());
    auto window =
        platform::testing::TestWindow::create({.width = 320, .height = 240}, "dvv present test");
    REQUIRE(window.has_value());
    auto presenter =
        createPresenter(**context,
                        {.window = window->handle(),
                         .extent = window->clientSize(),
                         .visible = true,
                         .onError = {},
                         .name = "present test"},
                        testFrameProvider(*context, std::make_shared<ProviderControl>()));
    REQUIRE(presenter.has_value());
    REQUIRE(pumpUntil(*window, [&] {
        (*presenter)->requestFrame();
        return (*presenter)->stats().framesPresented >= 1;
    }));
    CHECK(livePresenters() == 1);
    presenter->reset();
    CHECK(livePresenters() == 0);
    context->reset();
    CHECK((*gpu_instance)->validationMessageCount() == 0);
    gpu_instance->reset();
    CHECK(LifetimeRegistry::instance().liveCount() == before);
}

namespace {

// A provider that adds a `completed` callback to the fixture's frames. The callback runs on the
// presenter thread, the checks on the test thread.
struct CompletedLog {
    std::mutex mutex;
    std::vector<std::thread::id> threads;
    std::vector<bool> hadValue;
    std::vector<std::uint64_t> frameNumbers;  // of the frames that called back
    std::atomic<std::uint64_t> handedOut{0};  // frames the provider returned
    std::atomic<bool> lost{false};
};

// `failing_buffer`, when set, makes the recorder fail after the fixture's commands: a copy of size
// 0 on it.
FrameProvider withCompleted(FrameProvider frames, CompletedLog& log,
                            std::shared_ptr<GpuBuffer> failing_buffer = {}) {
    return [frames = std::move(frames), &log,
            failing_buffer = std::move(failing_buffer)](const PresentFrameParams& params) mutable {
        Result<PresentFrame> frame = frames(params);
        if (frame) {
            log.handedOut.fetch_add(1);
            if (failing_buffer) {
                frame->record = [inner = std::move(frame->record),
                                 failing_buffer](CommandRecorder& recorder) mutable {
                    inner(recorder);
                    recorder.copyBuffer(*failing_buffer, *failing_buffer, BufferCopy{.size = 0});
                };
            }
            frame->completed =
                [&log, number = params.frameNumber](const Result<std::uint64_t>& result) noexcept {
                    try {
                        const std::lock_guard lock(log.mutex);
                        log.threads.push_back(std::this_thread::get_id());
                        log.hadValue.push_back(result.has_value());
                        log.frameNumbers.push_back(number);
                    } catch (...) {
                        log.lost = true;  // out of memory
                    }
                };
        }
        return frame;
    };
}

}  // namespace

TEST_CASE_FIXTURE(PresentTest, "a frame's completed callback runs once after its submission") {
    CompletedLog log;
    auto presenter = createPresenter(context(), options(), withCompleted(provider(), log));
    REQUIRE(presenter.has_value());
    if (!presenter) {
        return;
    }
    REQUIRE(pumpUntil(window(), [&] {
        (*presenter)->requestFrame();
        return (*presenter)->stats().framesPresented >= 5;
    }));
    const std::uint64_t presented = (*presenter)->stats().framesPresented;
    presenter->reset();
    const std::lock_guard lock(log.mutex);
    // Every presented frame called back before it was counted. Frames can call back without being
    // counted (the one in flight at the stop), so only a lower bound holds on `presented`.
    CHECK_FALSE(log.lost.load());
    CHECK(log.threads.size() >= presented);
    CHECK(log.threads.size() <= log.handedOut.load());
    std::vector<std::uint64_t> numbers = log.frameNumbers;
    std::ranges::sort(numbers);
    CHECK(std::ranges::adjacent_find(numbers) == numbers.end());  // at most once per frame
    for (const std::thread::id id : log.threads) {
        CHECK(id == log.threads.front());
        CHECK(id != std::this_thread::get_id());
    }
    for (const bool value : log.hadValue) {
        CHECK(value);
    }
}

TEST_CASE_FIXTURE(PresentTest, "a refused frame's completed callback never runs") {
    control().withoutTransferSrc = true;
    CompletedLog log;
    const std::unique_ptr<Presenter> presenter = [&] {
        auto made = createPresenter(context(), options(), withCompleted(provider(), log));
        REQUIRE(made.has_value());
        return made ? std::move(*made) : nullptr;
    }();
    REQUIRE(presenter);
    REQUIRE(pumpUntil(window(), [&] {
        presenter->requestFrame();
        return errors().size() >= 1;
    }));
    const std::lock_guard lock(log.mutex);
    // The presenter refuses the frame before its submission and destroys it without a call.
    CHECK(log.threads.empty());
    CHECK(presenter->stats().framesPresented == 0);
}

TEST_CASE_FIXTURE(PresentTest, "a frame's completed callback gets the submission error") {
    CompletedLog log;
    auto buffer = createBuffer(
        context(), {.size = 16,
                    .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    .memory = MemoryUsage::DeviceOnly,
                    .name = "present test failing copy"});
    REQUIRE(buffer.has_value());
    if (!buffer) {
        return;
    }
    const std::unique_ptr<Presenter> presenter = [&] {
        auto made = createPresenter(context(), options(), withCompleted(provider(), log, *buffer));
        REQUIRE(made.has_value());
        return made ? std::move(*made) : nullptr;
    }();
    REQUIRE(presenter);
    REQUIRE(pumpUntil(window(), [&] {
        presenter->requestFrame();
        return errors().size() >= 1;
    }));
    const std::lock_guard lock(log.mutex);
    // The callback ran once, with the error, before the presenter reported it.
    REQUIRE(log.threads.size() == 1);
    CHECK_FALSE(log.hadValue.front());
    CHECK(log.threads.front() != std::this_thread::get_id());
    CHECK(presenter->stats().framesPresented == 0);
    const std::optional<Error> failure = presenter->failure();
    REQUIRE(failure.has_value());
    if (failure) {
        CHECK(failure->code == Errc::Internal);
    }
}

// A swapchain recreate retires the old swapchain instead of waiting until the shared queue is idle,
// so one presenter's resizes never stall another presenter on the same context.
TEST_CASE_FIXTURE(PresentTest,
                  "two presenters on one context: a resize of one never waits for the queue") {
    platform::testing::TestWindow& window_b = secondWindow();
    const std::unique_ptr<Presenter> a = makePresenter();
    PresenterOptions options_b = options();
    options_b.window = window_b.handle();
    options_b.extent = window_b.clientSize();
    options_b.name = "present test B";
    auto made_b = createPresenter(context(), std::move(options_b), provider());
    REQUIRE(made_b.has_value());
    if (!made_b) {
        return;
    }
    const std::unique_ptr<Presenter> b = std::move(*made_b);
    REQUIRE(pumpUntil(window(), [&] {
        window_b.pump();
        a->requestFrame();
        b->requestFrame();
        return a->stats().framesPresented >= 1 && b->stats().framesPresented >= 1;
    }));
    const detail::DeviceCore& device = detail::Access::core(context());
    const std::uint64_t waits = device.queueIdleWaits();
    const std::uint64_t presented_b = b->stats().framesPresented;
    constexpr int RESIZES = 20;
    for (int i = 0; i < RESIZES; ++i) {
        CAPTURE(i);
        const platform::PixelSize size = i % 2 == 0
                                             ? platform::PixelSize{.width = 400, .height = 300}
                                             : platform::PixelSize{.width = 320, .height = 240};
        const std::uint64_t created = a->stats().swapchainsCreated;
        const std::uint64_t before_b = b->stats().framesPresented;
        window().resize(size);
        a->resize(window().clientSize());
        REQUIRE(pumpUntil(window(), [&] {
            window_b.pump();
            a->requestFrame();
            b->requestFrame();
            const PresenterStats s = a->stats();
            return s.swapchainsCreated > created && s.presentedExtent == window().clientSize() &&
                   b->stats().framesPresented > before_b;
        }));
        // A frame at the new extent has completed, so no swapchain is left retired.
        CHECK(detail::Access::core(*a).retirementStats().pending() == 0);
    }
    CHECK(device.queueIdleWaits() == waits);
    CHECK(b->stats().framesPresented >= presented_b + RESIZES);
    CHECK_FALSE(a->stats().failed);
    CHECK_FALSE(b->stats().failed);
    CHECK_FALSE(a->failure());
    CHECK_FALSE(b->failure());
    CHECK(errors().size() == 0);
}

// The retirement rule: a swapchain replaced by a recreate waits in retired_ until a frame has
// completed on its successor. A second recreate before any frame completed destroys the newer
// swapchain at once (nothing was presented from it) and keeps the older retired one.
TEST_CASE_FIXTURE(PresentTest,
                  "a recreate without a completed frame destroys the newer swapchain") {
    const std::unique_ptr<Presenter> presenter = makePresenter();
    REQUIRE(pumpUntil(window(), [&] {
        presenter->requestFrame();
        return presenter->stats().framesPresented >= 1;
    }));
    const detail::PresenterCore& core = detail::Access::core(*presenter);
    const detail::PresenterCore::RetirementStats before = core.retirementStats();
    const PresenterStats stats = presenter->stats();
    // The first frame at the new size fails in the provider, so no frame completes on that
    // swapchain before the second recreate.
    window().resize({.width = 400, .height = 300});
    control().failAtWidth = static_cast<std::uint32_t>(window().clientSize().width);
    presenter->resize(window().clientSize());
    // No requestFrame while waiting: a second frame at this size would complete.
    REQUIRE(pumpUntil(window(), [&] {
        const PresenterStats s = presenter->stats();
        return s.swapchainsCreated > stats.swapchainsCreated && s.frameErrors > stats.frameErrors;
    }));
    window().resize({.width = 320, .height = 240});
    presenter->resize(window().clientSize());
    REQUIRE(pumpUntil(window(), [&] {
        presenter->requestFrame();
        return presenter->stats().swapchainsCreated >= stats.swapchainsCreated + 2;
    }));
    // A frame before the second recreate may still have been presented; count from here on.
    const std::uint64_t presented = presenter->stats().framesPresented;
    REQUIRE(pumpUntil(window(), [&] {
        presenter->requestFrame();
        return presenter->stats().framesPresented > presented;
    }));
    const detail::PresenterCore::RetirementStats after = core.retirementStats();
    CHECK(after.destroyedAtOnce - before.destroyedAtOnce == 1);
    CHECK(after.parked - before.parked == 1);
    CHECK(after.destroyedAfterFrame - before.destroyedAfterFrame == 1);
    CHECK(after.pending() == 0);
    CHECK_FALSE(presenter->stats().failed);
    CHECK(errors().size() == 1);  // the injected provider failure only
}

}  // namespace dvv::gpu::test
