// GPU tests (CTest label gpu). Without a usable Vulkan 1.4 device the executable exits with 77,
// which CTest reports as skipped (SKIP_RETURN_CODE), unless DVV_REQUIRE_GPU=1 turns that into a
// failure (CI on lavapipe).
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
#include "dvv/gpu/gpu_registry.h"

namespace {

constexpr int EXIT_SKIPPED = 77;

// Why the tests cannot run here: nullopt when a usable device exists, the reason when there is no
// loader or no usable Vulkan 1.4 device (Errc::Unsupported), an error for any other failure.
// Validation is off for this probe; the tests themselves validate.
dvv::Result<std::optional<std::string>> missingGpu() {
    const auto registry =
        dvv::gpu::GpuRegistry::create({.validation = dvv::gpu::ValidationMode::Off});
    if (!registry) {
        if (registry.error().code != dvv::Errc::Unsupported) {
            return std::unexpected(registry.error());
        }
        return registry.error().describe();
    }
    if (!(*registry)->bestDevice()) {
        return "no Vulkan 1.4 device meets the requirements";
    }
    return std::nullopt;
}

int run(int argc, char** argv) {
    dvv::LifetimeRegistry::instance().setDetailed(true);
    // Validation messages and other warnings appear in the test output.
    dvv::Logger::instance().addSink(std::make_shared<dvv::StderrSink>());
    dvv::Logger::instance().setLevel(dvv::LogLevel::Warn);

    const auto probed = missingGpu();
    if (!probed) {
        std::println(stderr, "GPU tests failed: {}", probed.error().describe());
        return 1;
    }
    if (const std::optional<std::string>& missing = *probed; missing.has_value()) {
        if (dvv::getEnvironment("DVV_REQUIRE_GPU") == "1") {
            std::println(stderr, "GPU tests failed: DVV_REQUIRE_GPU=1, but {}", *missing);
            return 1;
        }
        std::println("GPU tests skipped: {}", *missing);
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
        std::fprintf(stderr, "GPU tests failed: %s\n", error.what());
        return 1;
    }
}
