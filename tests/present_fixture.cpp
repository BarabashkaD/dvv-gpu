#include "present_fixture.h"

#include <doctest/doctest.h>

#include <chrono>
#include <exception>
#include <memory>
#include <thread>
#include <utility>

#include <vulkan/vulkan_core.h>

#include "dvv/gpu/command_recorder.h"
#include "dvv/gpu/device_id.h"
#include "dvv/gpu/resources.h"

namespace dvv::gpu::test {

void ErrorLog::add(const Error& error) {
    const std::lock_guard lock(mutex_);
    errors_.push_back(error);
}

std::vector<Error> ErrorLog::errors() const {
    const std::lock_guard lock(mutex_);
    return errors_;
}

void ErrorLog::addLost() noexcept {
    lost_.fetch_add(1);
}

std::size_t ErrorLog::size() const {
    const std::lock_guard lock(mutex_);
    return errors_.size() + lost_.load();
}

FrameProvider testFrameProvider(std::shared_ptr<ExecutionContext> context,
                                std::shared_ptr<ProviderControl> control) {
    return [context = std::move(context), control = std::move(control),
            buffer = std::shared_ptr<GpuBuffer>{}, image = std::shared_ptr<GpuImage>{},
            extent = platform::PixelSize{}](
               const PresentFrameInfo& info) mutable -> Result<PresentFrame> {
        control->calls.fetch_add(1);
        if (control->failNext.exchange(false)) {
            return makeError(Errc::Io, "injected provider failure");
        }
        if (control->newImage.exchange(false) || !image || info.extent != extent) {
            const std::uint64_t size =
                static_cast<std::uint64_t>(info.extent.width) * info.extent.height * 4;
            DVV_TRY(auto new_buffer,
                    context->createBuffer({.size = size,
                                           .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           .memory = MemoryUsage::DeviceOnly,
                                           .name = "present test pixels"}));
            const VkImageUsageFlags usage =
                control->withoutTransferSrc.load()
                    ? VkImageUsageFlags{VK_IMAGE_USAGE_TRANSFER_DST_BIT}
                    : VkImageUsageFlags{VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                        VK_IMAGE_USAGE_TRANSFER_DST_BIT};
            DVV_TRY(auto new_image, context->createImage({.width = info.extent.width,
                                                          .height = info.extent.height,
                                                          .format = control->integerFormat.load()
                                                                        ? VK_FORMAT_R8G8B8A8_UINT
                                                                        : VK_FORMAT_R8G8B8A8_UNORM,
                                                          .usage = usage,
                                                          .name = "present test image"}));
            buffer = std::move(new_buffer);
            image = std::move(new_image);
            extent = info.extent;
        }
        const std::uint32_t value =
            0xFF000000U |
            (static_cast<std::uint32_t>(info.frameNumber * 0x00102030U) & 0x00FFFFFFU);
        return PresentFrame{.record =
                                [buffer = buffer, image = image, value](CommandRecorder& recorder) {
                                    recorder.fillBuffer(*buffer, value);
                                    recorder.copyBufferToImage(*buffer, *image);
                                },
                            .image = image};
    };
}

bool pumpUntil(platform::testing::TestWindow& window, const std::function<bool()>& done,
               std::chrono::milliseconds limit) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (std::chrono::steady_clock::now() < deadline) {
        window.pump();
        if (done()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return done();
}

PresentTest::PresentTest() {
    auto registry = GpuRegistry::create({.windowSystem = platform::defaultWindowSystem()});
    if (!registry) {
        FAIL(registry.error().describe());
    }
    registry_ = std::move(*registry);
    const auto best = registry_->bestDevice();
    REQUIRE(best.has_value());
    auto context = registry_->createContext(best.value_or(DeviceId(0)), {.presentation = true});
    if (!context) {
        FAIL(context.error().describe());
    }
    context_ = std::move(*context);
    auto window =
        platform::testing::TestWindow::create({.width = 320, .height = 240}, "dvv present test");
    if (!window) {
        FAIL(window.error().describe());
    }
    window_.emplace(std::move(*window));
}

PresentTest::~PresentTest() {
    context_.reset();
    CHECK(registry_->validationMessageCount() == 0);
}

PresenterOptions PresentTest::options() {
    return {.window = window().native(),
            .extent = window().clientSize(),
            .visible = true,
            .frames = testFrameProvider(context_, control_),
            .onError =
                [log = errors_](const Error& error) noexcept {
                    try {
                        log->add(error);
                    } catch (const std::exception&) {
                        log->addLost();  // still counted by size()
                    }
                },
            .name = "present test"};
}

std::unique_ptr<Presenter> PresentTest::makePresenter() {
    auto presenter = context_->createPresenter(options());
    if (!presenter) {
        FAIL(presenter.error().describe());
    }
    return std::move(*presenter);
}

}  // namespace dvv::gpu::test
