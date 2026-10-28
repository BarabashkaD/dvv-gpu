#include <doctest/doctest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include <vulkan/vulkan_core.h>

#include "access.h"
#include "device_core.h"
#include "dvv/base/error.h"
#include "dvv/gpu/execution_context.h"
#include "dvv/gpu/gpu_registry.h"
#include "dvv/gpu/presenter.h"
#include "dvv/gpu/resources.h"
#include "dvv/gpu/testing/fault_injection.h"
#include "dvv/platform/testing/test_window.h"
#include "present_fixture.h"

namespace dvv::gpu::test {

namespace {

PresenterOptions lostTestOptions(platform::testing::TestWindow& window,
                                 const std::shared_ptr<ExecutionContext>& context,
                                 const std::shared_ptr<ProviderControl>& control,
                                 const std::shared_ptr<ErrorLog>& log) {
    return {.window = window.native(),
            .extent = window.clientSize(),
            .visible = true,
            .frames = testFrameProvider(context, control),
            .onError =
                [log](const Error& error) noexcept {
                    try {
                        log->add(error);
                    } catch (...) {
                        log->addLost();  // still counted by size()
                    }
                },
            .name = "lost test"};
}

// Stand-ins for the driver: a WSI call that reports a real loss to the presenter only.
VKAPI_ATTR VkResult VKAPI_CALL lostPresent(VkQueue /*queue*/, const VkPresentInfoKHR* /*info*/) {
    return VK_ERROR_DEVICE_LOST;
}

VKAPI_ATTR VkResult VKAPI_CALL lostAcquire(VkDevice /*device*/, VkSwapchainKHR /*swapchain*/,
                                           std::uint64_t /*timeout*/, VkSemaphore /*semaphore*/,
                                           VkFence /*fence*/, std::uint32_t* /*index*/) {
    return VK_ERROR_DEVICE_LOST;
}

// Replaces one entry of the context's dispatch table while it lives. Set before the presenter
// starts its thread and restored after it ended, so no Vulkan call races with the write.
template <typename Pfn>
class DispatchStub {
public:
    DispatchStub(Pfn& entry, Pfn stub) noexcept
        : entry_(entry), real_(std::exchange(entry, stub)) {}
    DispatchStub(const DispatchStub&) = delete;
    DispatchStub& operator=(const DispatchStub&) = delete;
    DispatchStub(DispatchStub&&) = delete;
    DispatchStub& operator=(DispatchStub&&) = delete;
    ~DispatchStub() { entry_ = real_; }

private:
    Pfn& entry_;
    Pfn real_;
};

vk::detail::DispatchLoaderDynamic& mutableDispatch(const ExecutionContext& context) {
    return const_cast<vk::detail::DispatchLoaderDynamic&>(detail::Access::core(context).dispatch());
}

}  // namespace

// Design M3 §4.1, E3: the presenter fails once with DeviceLost; nothing hangs.
TEST_CASE_FIXTURE(PresentTest,
                  "an injected loss fails the presenter once and destruction does not "
                  "hang") {
    for (int round = 0; round < 10; ++round) {
        CAPTURE(round);
        const auto best = registry().bestDevice();
        REQUIRE(best.has_value());
        auto ctx = registry()
                       .createContext(best.value_or(dvv::gpu::DeviceId{0}), {.presentation = true})
                       .value();
        const auto control = std::make_shared<ProviderControl>();
        const auto log = std::make_shared<ErrorLog>();
        auto created = ctx->createPresenter(lostTestOptions(window(), ctx, control, log));
        REQUIRE(created.has_value());
        std::unique_ptr<Presenter> p = std::move(*created);
        REQUIRE(pumpUntil(window(), [&] {
            p->requestFrame();
            return p->stats().framesPresented >= 2;
        }));
        testing::injectDeviceLost(*ctx);
        // failure() is set just before onError runs (presenter thread), so wait for both.
        REQUIRE(pumpUntil(window(), [&] {
            p->requestFrame();
            return p->failure().has_value() && log->size() >= 1;
        }));
        CHECK(p->failure().value_or(dvv::Error{}).code == Errc::DeviceLost);
        CHECK(log->size() == 1);
        REQUIRE_FALSE(log->errors().empty());
        CHECK(log->errors()[0].code == Errc::DeviceLost);

        const auto second = ctx->createPresenter(
            lostTestOptions(window(), ctx, std::make_shared<ProviderControl>(), log));
        REQUIRE_FALSE(second.has_value());
        CHECK(second.error().code == Errc::DeviceLost);

        const auto start = std::chrono::steady_clock::now();
        p.reset();
        const auto elapsed = std::chrono::steady_clock::now() - start;
        CHECK(elapsed < std::chrono::seconds(6));
        CHECK(log->size() == 1);  // onError once, also counting until the presenter thread ended
        ctx.reset();
    }
}

// Review fix (M3 Task 1): after a loss, a frame provider that fails (its own createBuffer returns
// DeviceLost, or it fails for another reason on the lost context) fails the presenter once with
// DeviceLost; it is not a skipped frame that repeats onError on every request.
TEST_CASE_FIXTURE(PresentTest, "a provider error on a lost context fails the presenter once") {
    for (const bool provider_allocates : {true, false}) {
        CAPTURE(provider_allocates);
        const auto best = registry().bestDevice();
        REQUIRE(best.has_value());
        auto ctx = registry()
                       .createContext(best.value_or(dvv::gpu::DeviceId{0}), {.presentation = true})
                       .value();
        const auto control = std::make_shared<ProviderControl>();
        const auto log = std::make_shared<ErrorLog>();
        auto created = ctx->createPresenter(lostTestOptions(window(), ctx, control, log));
        REQUIRE(created.has_value());
        std::unique_ptr<Presenter> p = std::move(*created);
        // Only the initial request: once it is presented, the presenter waits with no frame in
        // flight, so the next frame after the loss starts with a provider call.
        REQUIRE(pumpUntil(window(), [&] { return p->stats().framesPresented >= 1; }));
        testing::injectDeviceLost(*ctx);
        if (provider_allocates) {
            control->newImage = true;  // createBuffer on the lost context: DeviceLost
        } else {
            control->failNext = true;  // Errc::Io
        }
        REQUIRE(pumpUntil(window(), [&] {
            p->requestFrame();
            return p->failure().has_value() && log->size() >= 1;
        }));
        CHECK(p->failure().value_or(dvv::Error{}).code == Errc::DeviceLost);
        CHECK(p->stats().frameErrors == 0);
        // Failed is terminal: further requests call neither the provider nor onError.
        const std::uint64_t calls = control->calls.load();
        CHECK_FALSE(pumpUntil(
            window(),
            [&] {
                p->requestFrame();
                return control->calls.load() != calls;
            },
            std::chrono::milliseconds(300)));
        p.reset();
        CHECK(log->size() == 1);
        REQUIRE_FALSE(log->errors().empty());
        CHECK(log->errors()[0].code == Errc::DeviceLost);
        ctx.reset();
    }
}

// Review fix (M3 Task 1): a VK_ERROR_DEVICE_LOST that only the presenter sees (acquire or present)
// puts the context into the lost state, so creators refuse and destruction skips the cache save.
// No real loss can be produced, so a dispatch-table stub reports it.
TEST_CASE_FIXTURE(PresentTest, "a device loss seen only by the presenter marks the context lost") {
    for (const bool in_present : {true, false}) {
        CAPTURE(in_present);
        const auto best = registry().bestDevice();
        REQUIRE(best.has_value());
        auto ctx = registry()
                       .createContext(best.value_or(dvv::gpu::DeviceId{0}), {.presentation = true})
                       .value();
        auto& api = mutableDispatch(*ctx);
        std::optional<DispatchStub<PFN_vkQueuePresentKHR>> present_stub;
        std::optional<DispatchStub<PFN_vkAcquireNextImageKHR>> acquire_stub;
        if (in_present) {
            present_stub.emplace(api.vkQueuePresentKHR, &lostPresent);
        } else {
            acquire_stub.emplace(api.vkAcquireNextImageKHR, &lostAcquire);
        }
        const auto control = std::make_shared<ProviderControl>();
        const auto log = std::make_shared<ErrorLog>();
        auto created = ctx->createPresenter(lostTestOptions(window(), ctx, control, log));
        REQUIRE(created.has_value());
        std::unique_ptr<Presenter> p = std::move(*created);
        REQUIRE(pumpUntil(window(), [&] {
            p->requestFrame();
            return p->failure().has_value() && log->size() >= 1;
        }));
        CHECK(p->failure().value_or(dvv::Error{}).code == Errc::DeviceLost);
        CHECK(ctx->lost());
        const auto buffer = ctx->createBuffer({.size = 256,
                                               .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                               .memory = MemoryUsage::DeviceOnly,
                                               .name = "after a presenter loss"});
        REQUIRE_FALSE(buffer.has_value());
        CHECK(buffer.error().code == Errc::DeviceLost);
        p.reset();
        CHECK(log->size() == 1);
        present_stub.reset();  // before the context (and its dispatch table) goes
        acquire_stub.reset();
        ctx.reset();
    }
}

}  // namespace dvv::gpu::test
