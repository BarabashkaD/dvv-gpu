// dvv-gpu-info: prints what the GpuRegistry finds (spec §5.1).
//   dvv-gpu-info [--json]
// Exit codes (spec §10.3): 0 a usable device exists, 1 bad arguments, 4 GPU error, e.g. no Vulkan
// loader or no usable device.

#include <cstdint>
#include <cstdio>
#include <exception>
#include <format>
#include <memory>
#include <print>
#include <string>
#include <string_view>
#include <utility>

#include "dvv/base/error.h"
#include "dvv/base/json.h"
#include "dvv/base/leak_check.h"
#include "dvv/base/log.h"
#include "dvv/gpu/gpu_info.h"
#include "dvv/gpu/gpu_info_json.h"
#include "dvv/gpu/gpu_registry.h"

namespace {

constexpr int EXIT_OK = 0;
constexpr int EXIT_BAD_ARGUMENTS = 1;
constexpr int EXIT_GPU_ERROR = 4;

constexpr std::string_view USAGE = "usage: dvv-gpu-info [--json]";

std::string yesNo(bool value) {
    return value ? "yes" : "no";
}

std::string formatBytes(std::uint64_t bytes) {
    constexpr double GIB = 1024.0 * 1024.0 * 1024.0;
    return std::format("{:.1f} GiB", static_cast<double>(bytes) / GIB);
}

void printText(const dvv::gpu::GpuRegistry& registry) {
    const auto best = registry.bestDevice();
    std::println("Vulkan loader {}, validation {}, {} device(s)",
                 dvv::gpu::formatVulkanVersion(registry.loaderVersion()),
                 dvv::gpu::validationModeName(registry.validation()), registry.devices().size());
    for (const dvv::gpu::DeviceId id : registry.rankedDevices()) {
        const dvv::gpu::GpuInfo& info = registry.devices()[id.index()];
        std::println("");
        std::println("Device {}: {} ({}){}", id, info.name, dvv::gpu::deviceTypeName(info.type),
                     best == id ? ", selected" : "");
        std::println("  Vulkan {}, vendor {:#06x}, device {:#06x}",
                     dvv::gpu::formatVulkanVersion(info.apiVersion), info.vendorId, info.productId);
        std::println("  driver {} {} (version {:#x})", info.driverName, info.driverInfo,
                     info.driverVersion);
        std::println("  device UUID {}", dvv::gpu::formatUuid(info.deviceUuid));
        std::println("  device-local memory {}", formatBytes(info.deviceLocalMemoryBytes));
        std::println(
            "  optional: shaderInt16 {}, shaderFloat16 {}, hostImageCopy {}, "
            "VK_EXT_memory_budget {}, VK_KHR_swapchain {}",
            yesNo(info.optional.shaderInt16), yesNo(info.optional.shaderFloat16),
            yesNo(info.optional.hostImageCopy), yesNo(info.optional.memoryBudget),
            yesNo(info.optional.swapchain));
        if (info.usable()) {
            std::println("  usable");
        } else {
            std::string missing;
            for (const std::string& requirement : info.missingRequirements) {
                missing += missing.empty() ? requirement : ", " + requirement;
            }
            std::println("  not usable, missing: {}", missing);
        }
    }
}

void printJson(const dvv::gpu::GpuRegistry& registry) {
    dvv::json::Value devices = dvv::json::Value::array();
    for (const dvv::gpu::DeviceId id : registry.rankedDevices()) {
        devices.push_back(dvv::gpu::toJson(registry.devices()[id.index()]));
    }
    const auto best = registry.bestDevice();
    const dvv::json::Value output{
        {"loaderVersion", dvv::gpu::formatVulkanVersion(registry.loaderVersion())},
        {"validation", dvv::gpu::validationModeName(registry.validation())},
        {"selectedDevice", best ? dvv::json::Value(best->index()) : dvv::json::Value(nullptr)},
        {"devices", std::move(devices)},
    };
    std::println("{}", dvv::json::dump(output, 2));
}

void printError(const dvv::Error& error, bool json) {
    if (json) {
        const dvv::json::Value output{
            {"error", {{"code", dvv::errcName(error.code)}, {"message", error.describe()}}}};
        std::println("{}", dvv::json::dump(output, 2));
    } else {
        std::println(stderr, "dvv-gpu-info: {}", error.describe());
    }
}

int run(int argc, char** argv) {
    bool json = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        if (argument == "--json") {
            json = true;
        } else if (argument == "--help" || argument == "-h") {
            std::println(
                "{}\nexit codes: 0 a usable device exists, 1 bad arguments, 4 GPU error "
                "(e.g. no Vulkan loader or no usable device)",
                USAGE);
            return EXIT_OK;
        } else {
            std::println(stderr, "dvv-gpu-info: unknown argument '{}'\n{}", argument, USAGE);
            return EXIT_BAD_ARGUMENTS;
        }
    }

    const auto registry = dvv::gpu::GpuRegistry::create();
    if (!registry) {
        printError(registry.error(), json);
        return EXIT_GPU_ERROR;
    }
    if (json) {
        printJson(**registry);
    } else {
        printText(**registry);
    }
    return (*registry)->bestDevice() ? EXIT_OK : EXIT_GPU_ERROR;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        dvv::Logger::instance().addSink(std::make_shared<dvv::StderrSink>());
        dvv::Logger::instance().setLevel(dvv::LogLevel::Warn);
        const int exit_code = run(argc, argv);
        return dvv::finishWithLeakCheck(exit_code);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "dvv-gpu-info: %s\n", error.what());
        return EXIT_GPU_ERROR;
    }
}
