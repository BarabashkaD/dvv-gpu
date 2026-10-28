// Presentation tests (CTest labels gpu and display). Without a usable Vulkan 1.4 device that can
// present, or without a display, the executable exits with 77, which CTest reports as skipped
// (SKIP_RETURN_CODE), unless DVV_REQUIRE_GPU=1 turns that into a failure.
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <cstdio>
#include <exception>
#include <expected>
#include <memory>
#include <optional>
#include <print>
#include <string>

#include "dvv/base/environment.h"
#include "dvv/base/error.h"
#include "dvv/base/leak_check.h"
#include "dvv/base/lifetime.h"
#include "dvv/base/log.h"
#include "dvv/gpu/gpu_info.h"
#include "dvv/gpu/gpu_registry.h"
#include "dvv/platform/native_window.h"
#include "dvv/platform/testing/test_window.h"

namespace {

constexpr int EXIT_SKIPPED = 77;

// Why the tests cannot run here: nullopt when a device can present to a window of this build's
// window system and a display exists; the reason when the loader, a usable device, its swapchain
// support or the display is missing (Errc::Unsupported); an error for any other failure.
// Validation is off for this probe; the tests themselves validate.
dvv::Result<std::optional<std::string>> missingPresentation() {
    const auto registry =
        dvv::gpu::GpuRegistry::create({.validation = dvv::gpu::ValidationMode::Off,
                                       .windowSystem = dvv::platform::defaultWindowSystem()});
    if (!registry) {
        if (registry.error().code != dvv::Errc::Unsupported) {
            return std::unexpected(registry.error());
        }
        return registry.error().describe();
    }
    const auto best = (*registry)->bestDevice();
    if (!best) {
        return "no Vulkan 1.4 device meets the requirements";
    }
    if (!(*registry)->devices()[best->index()].optional.swapchain) {
        return "the best Vulkan device cannot present (no VK_KHR_swapchain)";
    }
    const auto window = dvv::platform::testing::TestWindow::create({.width = 64, .height = 64});
    if (!window) {
        if (window.error().code != dvv::Errc::Unsupported) {
            return std::unexpected(window.error());
        }
        return "no display: " + window.error().describe();
    }
    return std::nullopt;
}

int run(int argc, char** argv) {
    dvv::LifetimeRegistry::instance().setDetailed(true);
    // Validation messages and other warnings appear in the test output.
    dvv::Logger::instance().addSink(std::make_shared<dvv::StderrSink>());
    dvv::Logger::instance().setLevel(dvv::LogLevel::Warn);

    const auto probed = missingPresentation();
    if (!probed) {
        std::println(stderr, "GPU presentation tests failed: {}", probed.error().describe());
        return 1;
    }
    if (const std::optional<std::string>& missing = *probed; missing.has_value()) {
        if (dvv::getEnvironment("DVV_REQUIRE_GPU") == "1") {
            std::println(stderr, "GPU presentation tests failed: DVV_REQUIRE_GPU=1, but {}",
                         *missing);
            return 1;
        }
        std::println("GPU presentation tests skipped: {}", *missing);
        return EXIT_SKIPPED;
    }

    doctest::Context context(argc, argv);
    const int result = context.run();
    if (context.shouldExit()) {
        return result;
    }
    return dvv::finishWithLeakCheck(result);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "GPU presentation tests failed: %s\n", error.what());
        return 1;
    }
}
